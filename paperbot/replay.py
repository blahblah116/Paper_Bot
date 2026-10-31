"""[REPLAY] Zulip 되올리기 — Slack 시절에 이미 발행(done)된 논문을 Zulip에도 하루 몇 편씩 올리는 한시적 기능.

옛 논문을 다 올리고 나면 이 파일을 지우고, `grep -rn "\\[REPLAY\\]" paperbot/ tests/`로 찾은 줄들(pipeline.py의 연결 줄,
config.py의 zulip.replay·replay_per_run 필드와 검증, 테스트 블록)을 지우면 원래 로직으로 돌아간다. 'replay_items 인자만 지운다'고
표시된 두 줄(pipeline._select의 정의·호출)은 줄을 지우지 말고 그 인자만 뺀다. config.yaml의 zulip.replay* 키는 남아 있어도 무시된다.

동작 (pipeline.run 기준):
1. select(): platform이 zulip인 topic마다 DB의 done 논문(오래된 순) 중 Zulip에 아직 없는 것을 topic당 replay_per_run편 고른다.
   "이미 있는지"는 고정 Zulip topic이면 그 topic 메시지들의 '📄 링크'(uid 기준 정규 링크), 논문별 topic이면 topic 이름으로 판정한다.
   DB에는 아무것도 기록하지 않으므로 중간에 끊겨도 다음 실행이 이어서 하고, 다 올라가면 빈 리스트 → 원래 로직만 남는다.
2. hold_new(): 되올리는 topic의 그날 신규 선정분을 뺀다(_select가 pending으로 적재) — 되올리기가 끝나면 원래 쿼터대로 흐른다.
3. process(): 처리 직전에 메타데이터를 복원(sources/lookup.py: arXiv API → S2 → DB 제목만)하고 PDF·요약·발행.
   실패해도 status는 건드리지 않고 그 topic에 '되올리기 실패 + 링크' 한 줄만 남겨 다시 시도하지 않는다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from . import chatdb, summarizer
from .config import Config
from .extractor import extract
from .models import Paper, uid_url_key
from .publisher import (DryRunPublisher, MultiPublisher, PublishConfigError, Publisher, ZulipPublisher,
                        zulip_topic_name)
from .publisher.zulip import PAPER_LINK_RE, topic_messages
from .sources.lookup import paper_from_uid
from .store import Store
from .summarizer import LLMError

logger = logging.getLogger(__name__)


@dataclass
class ReplayItem:
    """되올리기 대상(DB done 레코드). Paper 복원(S2/arXiv 조회)은 처리 직전에 한다 — 미리 다 하면 S2 429로 느리다."""
    uid: str
    title: str
    source: str
    topic_name: str
    judge_score: float | None = None


# ---------------------------------------------------------------- 선정

def select(cfg: Config, store: Store, publisher: Publisher | None) -> list[ReplayItem]:
    """topic당 replay_per_run편(0이면 전부)의 되올리기 항목. 되올릴 게 없거나 기능이 꺼져 있으면 빈 리스트."""
    out: list[ReplayItem] = []
    if not cfg.zulip.replay or publisher is None:
        return out
    cap = cfg.zulip.replay_per_run
    cache: dict = {}
    for t in cfg.topics:
        if t.platform != "zulip":
            continue
        rows = store.conn.execute(
            "SELECT uid, title, source, judge_score FROM papers WHERE status='done' AND topic=? ORDER BY processed_at ASC",
            (t.name,),
        ).fetchall()
        if not rows:
            continue
        existing = _posted_keys(publisher, t.name, cache)
        if existing is None:
            logger.warning("Zulip 되올리기 %s: 스트림의 기존 글을 못 받아 이번 실행은 생략", t.name)
            continue
        missing = [r for r in rows if _replay_key(publisher, t.name, r["uid"], r["title"]) not in existing]
        if not missing:
            logger.info("Zulip 되올리기 %s: DB done %d편 전부 Zulip에 있음 — 원래 로직대로 (이 topic은 끝)", t.name, len(rows))
            continue
        picked = [ReplayItem(r["uid"], r["title"], r["source"], t.name, r["judge_score"]) for r in (missing[:cap] if cap else missing)]
        out.extend(picked)
        logger.info("Zulip 되올리기 %s: DB done %d편 중 Zulip에 없는 %d편 → 이번에 %d편 (topic당 %s)",
                    t.name, len(rows), len(missing), len(picked), cap or "전부")
    return out


def hold_new(selected: list[Paper], items: list[ReplayItem]) -> list[Paper]:
    """되올리는 topic의 신규 선정분을 뺀다 — 호출자(_select)가 빠진 신규 통과분을 pending으로 적재한다."""
    replaying = {r.topic_name for r in items}
    if not replaying:
        return selected
    held = [p for p in selected if p.topic_name in replaying]
    if held:
        logger.info("되올리기 중인 topic의 신규 %d편은 보류(pending) — 되올리기가 끝나면 원래 쿼터대로", len(held))
    return [p for p in selected if p.topic_name not in replaying]


# ---------------------------------------------------------------- '이미 Zulip에 있는지'

def _zulip_pub(publisher: Publisher, paper_topic: str):
    """publisher 안에서 이 topic을 담당하는 Zulip(또는 dry-run) publisher."""
    if isinstance(publisher, MultiPublisher):
        publisher = publisher._for_topic(paper_topic)
    return publisher


def _topic_key(name: str) -> str:
    return " ".join(name.split()).casefold()   # Zulip은 topic을 대소문자 구분 없이 묶는다


def _replay_key(publisher: Publisher, paper_topic: str, uid: str, title: str) -> str:
    """고정 topic이면 링크 키, 논문별 topic이면 topic 이름 키."""
    zp = _zulip_pub(publisher, paper_topic)
    fixed = getattr(zp, "zulip_topics", {}).get(paper_topic)
    return uid_url_key(uid).casefold() if fixed else _topic_key(zulip_topic_name(title))


def _posted_keys(publisher: Publisher, paper_topic: str, cache: dict) -> set[str] | None:
    """이미 올라간 논문 키 집합(스트림·topic별 한 번 조회해 cache). dry-run은 빈 집합, 조회 실패면 None."""
    zp = _zulip_pub(publisher, paper_topic)
    if isinstance(zp, DryRunPublisher):
        return set()
    if not isinstance(zp, ZulipPublisher):
        return None
    stream = zp._stream_of_topic(paper_topic)
    fixed = zp.zulip_topics.get(paper_topic)
    key = (stream, fixed)
    if key in cache:
        return cache[key]
    try:
        if fixed:
            texts = topic_messages(zp.client, stream, fixed)
            keys = {m.group(1).casefold() for t in texts for m in PAPER_LINK_RE.finditer(t)}
            logger.info("Zulip '%s / %s' 기존 메시지 %d개, 논문 링크 %d개", stream, fixed, len(texts), len(keys))
        else:
            sid = zp.client.get_stream_id(stream)
            if sid.get("result") != "success":
                raise RuntimeError(sid.get("msg", sid))
            res = zp.client.get_stream_topics(sid["stream_id"])
            if res.get("result") != "success":
                raise RuntimeError(res.get("msg", res))
            keys = {_topic_key(t.get("name", "")) for t in res.get("topics", [])}
            logger.info("Zulip 스트림 '%s' 기존 topic %d개", stream, len(keys))
    except Exception as e:  # noqa: BLE001
        logger.warning("Zulip '%s' 기존 글 조회 실패 — 되올리기 생략: %s", stream, e)
        cache[key] = None
        return None
    cache[key] = keys
    return keys


# ---------------------------------------------------------------- 처리

def process(cfg: Config, store: Store, publisher: Publisher, item: ReplayItem, client) -> bool:
    """되올리기 한 편: 복원 → extract → summarize → publish → mark_done(이미 done). 성공 시 True.
    실패해도 status는 안 바꾸고 topic에 안내만 남긴다(다음 실행에서 또 시도하지 않게). 복원 실패는 안내 없이 다음 실행에 다시."""
    try:
        paper = paper_from_uid(item.uid, item.title, item.source, item.topic_name, item.judge_score)
    except Exception as e:  # noqa: BLE001
        logger.warning("되올리기 복원 실패 %s: %s", item.uid, e)
        return False
    try:
        extracted = extract(cfg, paper)
        if extracted.abstract_only and not paper.abstract:
            logger.warning("본문·abstract 모두 확보 실패 — 발행하지 않음: %s", paper.uid)
            err = "본문/abstract 모두 없음"
        else:
            summary = summarizer.summarize(cfg, paper, extracted, client=client)
            publisher.publish(paper, summary, abstract_only=extracted.abstract_only, figures=extracted.figures)
            store.mark_done(paper)
            chatdb.store_paper(cfg, paper, extracted, summary)   # 챗봇 캐시에 전문·요약 (백필이 먼저 넣었어도 새 요약으로 갱신)
            return True
    except PublishConfigError:
        raise
    except LLMError as e:
        logger.error("LLM 실패로 skip: %s (%s)", paper.uid, e)
        err = str(e)
    except Exception as e:  # noqa: BLE001
        logger.exception("되올리기 처리 실패: %s (%s)", paper.uid, e)
        err = str(e)
    post_failure(publisher, paper, err)
    return False


def post_failure(publisher: Publisher, paper: Paper, reason: str) -> None:
    """복원·요약에 실패한 옛 논문 자리에 짧은 안내(+링크)만 남긴다 — 링크가 있으니 다음 실행의 '이미 있음' 판정에 걸린다."""
    zp = _zulip_pub(publisher, paper.topic_name)
    text = f"⚠️ 되올리기 실패 — {reason[:300]}\n📄 {paper.url}"
    if isinstance(zp, DryRunPublisher):
        print(f"[DRY-RUN/zulip → {zp._dest(paper)}] 되올리기 실패 안내: {reason}")
        return
    if not isinstance(zp, ZulipPublisher):
        return
    try:
        zp._send(zp._stream_for(paper), zp._topic_for(paper), text)
    except PublishConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("되올리기 실패 안내 전송 실패 %s: %s", paper.uid, e)
