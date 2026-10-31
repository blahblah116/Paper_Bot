"""Zulip publishing: one message per paper in a stream topic, plus the Zulip text helpers.
Zulip 발행 — 스트림의 topic(주제별 고정 topic 또는 논문별 topic=제목)에 논문당 메시지 하나
(머리말 + 접힌 전체 요약 + figure 링크). zoterobot·replay도 여기의 도우미를 쓴다.

- 인증: 환경변수 ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE(~/.paperbot_env), 없으면 zuliprc(zulip.config_file). make_zulip_client().
- 레이아웃: build_zulip_messages() — 10,000자 제한에 맞춰 분할. 수식 $$…$$는 zulip_math_spacing()으로 앞뒤 공백 보장.
- 스트림 없음·API 키 오류 같은 설정 문제는 PublishConfigError로 실행 전체를 중단(논문 소모 없음).
"""
from __future__ import annotations

import io
import logging
import os
import re
import time

from ..config import Config
from ..config import ZULIP_TOPIC_MAX
from ..figures import Figure
from ..models import Paper
from .base import PublishConfigError, Publisher, _format_meta, extract_one_liner, figure_filename

logger = logging.getLogger(__name__)

_ZULIP_MESSAGE_MAX = 9500    # Zulip 메시지 최대 10,000자 — 여유를 두고 나눈다
SUMMARY_TITLE = "전체 요약"   # 요약을 접어 두는 spoiler 제목 — chatdb가 이 제목으로 올라간 요약을 되읽는다
# 발행 메시지의 논문 링크('📄 …') — models.uid_url_key()가 만드는 모양만 본다 (되올리기·백필이 이미 올린 글을 찾을 때)
PAPER_LINK_RE = re.compile(r"((?:arxiv\.org/abs/|doi\.org/|semanticscholar\.org/paper/)[^\s<>)\]]+)", re.IGNORECASE)


def zulip_topic_name(title: str) -> str:
    """Zulip topic 이름 = 논문 제목 (공백 정리, 60자 초과 시 절단)."""
    title = " ".join(title.split())
    if len(title) > ZULIP_TOPIC_MAX:
        return title[: ZULIP_TOPIC_MAX - 1].rstrip() + "…"
    return title


def zulip_topic_for(paper: Paper, fixed: str | None = None) -> str:
    """논문이 갈 Zulip topic — 주제에 고정 topic(topic.zulip_topic)이 있으면 그것, 없으면 제목."""
    return fixed if fixed else zulip_topic_name(paper.title)


def _zulip_topic_map(cfg: Config) -> dict[str, str]:
    """topic name → 고정 Zulip topic(topic.zulip_topic). 없는 topic은 논문마다 topic(=제목)."""
    return {t.name: t.zulip_topic for t in cfg.topics if t.zulip_topic}


def _topic_stream_map(cfg: Config) -> dict[str, str]:
    """topic name → Zulip 스트림(topic.stream). 없는 topic은 제외 → zulip.stream(공용 스트림)으로 간다.
    channel은 Slack 전용이라 폴백으로 쓰지 않는다. 스트림 이름의 '#'은 load_config가 이미 뗐다."""
    return {t.name: t.stream for t in cfg.topics if t.stream}


def format_zulip_main(paper: Paper, summary: str, abstract_only: bool) -> str:
    """Zulip topic의 첫 메시지 — 제목·메타·링크·한 줄 요약 (Zulip 마크다운)."""
    return (
        f"**{paper.title.strip()}**\n"
        f"{_format_meta(paper, abstract_only, italic='*')}\n"
        f"📄 {paper.url}\n"
        f"➤ {extract_one_liner(summary)}"
    )


# 인라인 수식 $$…$$ 앞뒤 공백 보장 — Zulip은 $$에 글자가 붙으면(예: '$$t$$의') 수식으로 렌더링하지 않는다.
# 모델에게 규칙을 줘도 조사를 붙여 쓰는 일이 잦아서 전송 직전에 코드로 고친다. 한 줄 안의 $$…$$만 다루고 여러 줄 블록은 건드리지 않는다.
_INLINE_MATH_RE = re.compile(r"\$\$(?!\s)[^\n$]+?(?<!\s)\$\$")
# 단일 $…$ (모델이 $$ 대신 쓴 LaTeX) — 앞에 $나 \\가 없고, 뒤에 $나 숫자(통화 '$5')가 오지 않는 것만
_SINGLE_MATH_RE = re.compile(r"(?<![$\\])\$(?![\s$])([^\n$]+?)(?<![\s$])\$(?![\d$])")
_MATHY_RE = re.compile(r"[A-Za-z\\^_=<>()]")   # 수식처럼 보이는 내용(문자·백슬래시·연산자)만 변환 — '$5-$7' 같은 금액은 제외
_MATH_PREV_OK = " \t(（[\"'“‘"          # 이 문자 뒤에는 공백을 안 넣어도 된다
_MATH_NEXT_OK = " \t.,;:!?)）]\"'”’"   # 이 문자 앞에는 공백을 안 넣어도 된다


def _single_to_double(line: str) -> str:
    """'값 $x$의' → '값 $$x$$의' (공백은 다음 단계에서). 금액 표기는 그대로 둔다."""
    return _SINGLE_MATH_RE.sub(lambda m: f"$${m.group(1)}$$" if _MATHY_RE.search(m.group(1)) else m.group(0), line)


_MATH_FENCE_OPEN_RE = re.compile(r"^\s*`{3,}\s*math\s*$")
_FENCE_CLOSE_RE = re.compile(r"^\s*`{3,}\s*$")


def _dedent_math_blocks(text: str) -> str:
    """들여쓴 ```math 블록(모델이 불릿 안에 넣는 경우)을 줄 맨 앞으로 빼고 앞뒤에 빈 줄을 둔다.
    Zulip 코드 펜스는 줄 맨 앞에서만 인식돼, 들여쓰면 'math' 글자와 원문 LaTeX가 그대로 보인다."""
    out: list[str] = []
    in_block = need_blank = False
    for line in text.split("\n"):
        if not in_block and _MATH_FENCE_OPEN_RE.match(line):
            if out and out[-1].strip():
                out.append("")
            out.append("```math")
            in_block, need_blank = True, False
        elif in_block and _FENCE_CLOSE_RE.match(line):
            out.append("```")
            in_block, need_blank = False, True
        elif in_block:
            out.append(line.strip())
        else:
            if need_blank and line.strip():
                out.append("")
            need_blank = False
            out.append(line)
    return "\n".join(out)


def zulip_math_spacing(text: str) -> str:
    """1) 들여쓴 ```math 블록을 바로잡고 2) 단일 $…$를 $$…$$로 맞추고 3) $$…$$ 양옆에 공백이 없으면 넣는다
    ('그래프$G$를' → '그래프 $$G$$ 를'). 줄 단위."""
    out: list[str] = []
    for line in _dedent_math_blocks(text).split("\n"):
        line = _single_to_double(line)
        pos = 0
        buf: list[str] = []
        for m in _INLINE_MATH_RE.finditer(line):
            a, b = m.span()
            before = line[pos:a]
            if before and before[-1] not in _MATH_PREV_OK:
                before += " "
            buf.append(before + m.group(0))
            if b < len(line) and line[b] not in _MATH_NEXT_OK:
                buf.append(" ")
            pos = b
        buf.append(line[pos:])
        out.append("".join(buf))
    return "\n".join(out)


def zulip_spoiler(title: str, body: str) -> str:
    """접힌 블록 — 본문에 ``` 코드 펜스가 있어도 깨지지 않게 바깥은 ```` 4개."""
    return f"````spoiler {title}\n{body.strip()}\n````"


def split_zulip_message(text: str, limit: int = _ZULIP_MESSAGE_MAX) -> list[str]:
    """긴 본문을 Zulip 메시지 길이 제한에 맞춰 문단 경계(빈 줄 → 줄 → 글자)로 나눈다."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest:
        parts.append(rest)
    return parts


def build_zulip_messages(header: str, summary: str, figure_links: list[str] | None = None,
                         limit: int = _ZULIP_MESSAGE_MAX) -> list[str]:
    """논문 하나 = 메시지 하나: 머리말(제목·메타·링크·한 줄) + 접힌 전체 요약 + figure 링크.
    10,000자를 넘기면 머리말(+figure)과 요약 조각(각각 접힌 블록)으로 나눈다."""
    header, summary = zulip_math_spacing(header), zulip_math_spacing(summary)
    figs = ("\n🖼 " + "  ".join(figure_links)) if figure_links else ""
    one = f"{header}\n\n{zulip_spoiler(SUMMARY_TITLE, summary)}{figs}"
    if len(one) <= limit:
        return [one]
    chunks = split_zulip_message(summary, limit - 60)
    n = len(chunks)
    return [header + figs] + [zulip_spoiler(f"{SUMMARY_TITLE} ({i}/{n})", c) for i, c in enumerate(chunks, start=1)]


ZULIP_ENV_VARS = ("ZULIP_EMAIL", "ZULIP_API_KEY", "ZULIP_SITE")


def make_zulip_client(config_file: str | None = None):
    """Zulip 클라이언트 — 환경변수 ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE(~/.paperbot_env)가 있으면 그걸로,
    없으면 zuliprc 파일(config_file)로. 둘 다 없거나 일부만 있으면 RuntimeError (호출자가 자기 예외로 감싼다).
    paperbot과 zoterobot이 같은 봇 계정을 쓰므로 여기 한 곳에서만 만든다."""
    import zulip  # 지연 임포트 — dry-run/Slack 전용 경로에선 필요 없음

    env = {k: os.environ.get(k, "").strip() for k in ZULIP_ENV_VARS}
    present = [k for k, v in env.items() if v]
    if present:
        missing = [k for k in ZULIP_ENV_VARS if not env[k]]
        if missing:
            raise RuntimeError(f"Zulip 환경변수가 일부만 있습니다 — {', '.join(missing)}도 ~/.paperbot_env에 넣으세요 "
                               f"(있는 것: {', '.join(present)})")
        try:
            return zulip.Client(email=env["ZULIP_EMAIL"], api_key=env["ZULIP_API_KEY"], site=env["ZULIP_SITE"])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Zulip 환경변수로 접속 설정 실패: {e}") from e
    if not config_file or not os.path.exists(config_file):
        raise RuntimeError(
            f"Zulip 인증 정보가 없습니다 — ~/.paperbot_env에 {', '.join(ZULIP_ENV_VARS)}를 export 하거나, "
            f"봇 설정 페이지에서 내려받은 zuliprc를 {config_file or '(config_file 미설정)'} 에 두세요."
        )
    try:
        return zulip.Client(config_file=config_file)
    except Exception as e:  # noqa: BLE001 — zuliprc 형식 오류 등
        raise RuntimeError(f"zuliprc를 읽지 못했습니다 ({config_file}): {e}") from e


def topic_messages(client, stream: str, topic: str) -> list[str]:
    """스트림/topic의 모든 메시지 본문(원문 마크다운), 오래된 것부터 — 1000개씩 거슬러 올라가며 전부."""
    msgs: list[tuple[int, str]] = []
    anchor: str | int = "newest"
    while True:
        res = client.get_messages({
            "anchor": anchor, "num_before": 1000, "num_after": 0, "apply_markdown": False,
            "narrow": [{"operator": "stream", "operand": stream}, {"operator": "topic", "operand": topic}],
        })
        if res.get("result") != "success":
            raise RuntimeError(res.get("msg", res))
        batch = res.get("messages") or []
        msgs.extend((int(m["id"]), str(m.get("content", ""))) for m in batch)
        if res.get("found_oldest", True) or not batch:
            break
        anchor = min(int(m["id"]) for m in batch) - 1   # 이번 페이지에서 가장 오래된 것 바로 앞부터
    msgs.sort()
    return [c for _, c in msgs]


def zulip_auth_error(res: dict) -> str:
    """get_profile 실패 응답 → 사람이 고칠 수 있는 안내. 흔한 실수: Incoming webhook 봇(보내기 전용)으로 만든 경우."""
    msg = str(res.get("msg", res))
    if "incoming webhook" in msg.lower():
        return ("Zulip 봇 종류가 'Incoming webhook'(보내기 전용)입니다 — DM·프로필·스트림 조회가 안 됩니다. "
                "Zulip 개인 설정 → Bots에서 'Generic bot'으로 새로 만들고 ~/.paperbot_env의 ZULIP_EMAIL/ZULIP_API_KEY를 "
                f"그 봇 값으로 바꾸세요. (서버 응답: {msg})")
    return f"Zulip 인증 실패: {msg} — ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE(또는 zuliprc)의 값 확인"


class _NamedBytes(io.BytesIO):
    """zulip.Client.upload_file은 파일 객체의 .name을 업로드 파일명으로 쓴다."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


# 재시도해도 소용없는 Zulip 설정 오류 — 응답 code 또는 msg로 판별.
_ZULIP_CONFIG_CODES = {"STREAM_DOES_NOT_EXIST", "INVALID_API_KEY", "UNAUTHORIZED",
                       "REALM_DEACTIVATED", "USER_DEACTIVATED"}
_ZULIP_CONFIG_MSGS = ("does not exist", "invalid api key", "not authenticated",
                      "not authorized", "unable to access", "deactivated")


def _zulip_config_error(res: dict) -> bool:
    code = str(res.get("code", "")).upper()
    msg = str(res.get("msg", "")).lower()
    return code in _ZULIP_CONFIG_CODES or any(m in msg for m in _ZULIP_CONFIG_MSGS)


class ZulipPublisher(Publisher):
    """Zulip 스트림에 논문당 topic(=제목) 하나: 첫 메시지 + 전체 요약 + figure 업로드.

    - 인증: 환경변수 ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE(~/.paperbot_env) 또는 zulip.config_file(zuliprc)
    - 스트림: topic.stream이 있으면 그것, 없으면 zulip.stream(공용 스트림 — 기본 구성). 봇이 구독해 있어야 한다.
    - Zulip topic: topic.zulip_topic이 있으면 그 주제의 논문을 전부 그 고정 topic에(논문마다 메시지 하나씩 이어짐),
      없으면 논문마다 topic(=제목)을 연다.
    - 스트림 없음·API 키 오류 같은 설정 문제는 PublishConfigError로 실행 전체를 중단(논문 소모 없음).
      zulip topic들이 쓰는 스트림은 생성 시점에 미리 확인한다(fetch/LLM에 시간을 쓰기 전에 실패).
    """

    _MAX_RETRIES = 3

    def __init__(self, cfg: Config, client=None):
        self.topic_channels = _topic_stream_map(cfg)
        self.zulip_topics = _zulip_topic_map(cfg)
        self.stream = cfg.zulip.stream
        if client is None:
            try:
                client = make_zulip_client(cfg.resolve(cfg.zulip.config_file))
            except RuntimeError as e:
                raise PublishConfigError(f"{e} (zulip.dry_run: true면 전송 없이 테스트 가능)") from e
        self.client = client
        profile = self.client.get_profile()
        if profile.get("result") != "success":
            raise PublishConfigError(zulip_auth_error(profile))
        logger.info("Zulip 봇 %s 인증 OK", profile.get("email") or profile.get("full_name"))
        self._check_streams(cfg)

    def _check_streams(self, cfg: Config) -> None:
        """platform: zulip인 topic들이 쓸 스트림이 전부 있는지(봇이 볼 수 있는지) 확인 — 없으면 PublishConfigError."""
        streams: list[str] = []
        for t in cfg.topics:
            if t.platform != "zulip":
                continue
            name = self.topic_channels.get(t.name, self.stream)
            if name not in streams:
                streams.append(name)
        missing = []
        for name in streams:
            try:
                res = self.client.get_stream_id(name)
            except Exception as e:  # noqa: BLE001 — 네트워크 등. 전송 때 다시 판정되므로 여기선 경고만
                logger.warning("Zulip 스트림 '%s' 확인 실패(전송 때 재확인): %s", name, e)
                continue
            if res.get("result") != "success":
                missing.append(name)
        if missing:
            raise PublishConfigError(
                f"Zulip 스트림이 없거나 봇이 볼 수 없습니다: {missing} — Zulip에서 스트림을 만들고 봇을 구독시키세요 "
                "(topic의 stream 또는 zulip.stream 확인)."
            )

    def _stream_for(self, paper: Paper) -> str:
        return self._stream_of_topic(paper.topic_name)

    def _stream_of_topic(self, paper_topic: str) -> str:
        return self.topic_channels.get(paper_topic, self.stream)

    def _topic_for(self, paper: Paper) -> str:
        return zulip_topic_for(paper, self.zulip_topics.get(paper.topic_name))

    def _send(self, stream: str, topic: str, content: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                res = self.client.send_message({"type": "stream", "to": stream, "topic": topic, "content": content})
                if res.get("result") != "success":
                    if _zulip_config_error(res):
                        raise PublishConfigError(
                            f"Zulip 설정 오류 '{res.get('msg')}': stream={stream!r}. 스트림 이름과 봇 구독, "
                            "zuliprc의 API 키를 점검하세요."
                        )
                    raise RuntimeError(f"Zulip API 오류: {res.get('msg', res)}")
                return res
            except PublishConfigError:
                raise
            except Exception as e:  # noqa: BLE001 — 연결 오류/일시 오류 재시도
                last_err = e
                logger.warning("Zulip 전송 실패 (%d/%d): %s", attempt, self._MAX_RETRIES, e)
                if attempt < self._MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Zulip 전송 최종 실패: {last_err}") from last_err

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[Figure] | None = None) -> None:
        stream = self._stream_for(paper)
        topic = self._topic_for(paper)
        # figure를 먼저 올려 링크를 받고(실패해도 발행은 진행), 논문당 메시지 하나로: 머리말 + 접힌 전체 요약 + figure 링크.
        # (첨부 여부·장수는 extractor가 cfg.figures_for로 이미 적용 — 꺼져 있으면 figures가 비어 온다)
        links = self._upload_figures(figures, paper) if figures else []
        first = None
        for content in build_zulip_messages(format_zulip_main(paper, summary, abstract_only), summary, links):
            res = self._send(stream, topic, content)
            first = first or res
        logger.info("Zulip 전송 완료: stream=%s topic=%r id=%s figure=%d", stream, topic, (first or {}).get("id"), len(links))

    def _upload_figures(self, figures: list[Figure], paper: Paper) -> list[str]:
        """figure·표 크롭을 업로드해 '[Figure 3](uri)' / '[Table 1](uri)' 링크 목록으로(파일명 uid_figure_3.png). 실패는 비치명적(빈 목록)."""
        links: list[str] = []
        try:
            for fig in figures:
                res = self.client.upload_file(_NamedBytes(fig.png, figure_filename(paper, fig)))
                if res.get("result") != "success":
                    logger.warning("%s 업로드 실패(건너뜀): %s", fig.label, res.get("msg", res))
                    continue
                uri = res.get("url") or res.get("uri")
                if uri:
                    links.append(f"[{fig.label}]({uri})")
        except Exception as e:  # noqa: BLE001 — figure는 비치명적
            logger.warning("figure 업로드 실패(텍스트만 발행): %s", e)
        return links
