"""The two request handlers: save command → Zotero/Drive, DM question → chat agent.
요청 핸들러 둘 — 저장 명령(Zotero/Drive), DM 질문(챗봇). 플랫폼(Slack/Zulip)은 bot 객체가 가린다.

- SaveHandler: Slack 스레드 '@zoterobot <컬렉션>' 또는 Zulip DM '저장 <컬렉션> <링크>' → 원글(또는 명령에 적힌 링크)의
  논문을 찾아 library.save_paper.
- ChatHandler: DM 질문 → GPU 확인 → 자리표시 메시지 → agent.run(진행 상황·부분 답변을 update로 갱신)
  → 최종 답 + 참고 논문. 허용된 사용자의 DM은 결과가 어떻든 대화 로그에 남긴다.
- 답글 표기(볼드·링크·LLM 마크다운 렌더링)는 bot.fmt(ev)로 플랫폼에 맞춘다.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .agent import Agent, LLMError
from .config import Config
from .drive import DriveError, DriveUploader
from .gpu import GpuBusy, ensure_gpu
from .library import save_paper
from .messages import HELP_WORDS, SLACK, TITLE_RE, esc, format_result, format_sources, format_usage, parse_command
from .resolver import clean_slack_text, first_ref, resolve
from .session import RESET_WORDS, ChatLog, Conversations
from .textcache import TextCache
from .zotero import ZoteroClient, ZoteroError, collection_paths

logger = logging.getLogger(__name__)

_FINAL_RETRY_WAITS = (1.0, 2.0)   # 최종 답 갱신 실패 시 재시도 전 대기(초) — 2026-10-21 429로 새 메시지가 또 올라간 사고
_PREVIEW_MAX = 3800        # 실시간 갱신에 보여줄 최대 글자 수 (전체는 완료 시 최종 갱신에서)


class SaveHandler:
    def __init__(self, cfg: Config, bot, zot: ZoteroClient, drive: DriveUploader,
                 on_saved: Callable[[], None] | None = None):
        self.cfg, self.bot, self.zot, self.drive = cfg, bot, zot, drive   # bot: listener.MultiBot(SlackBot/ZulipBot)
        self.on_saved = on_saved

    def usage(self, fmt=None) -> str:
        try:
            names = sorted(collection_paths(self.zot.collections()).values(), key=str.lower)
            return format_usage(names, self.drive.remote, fmt=fmt or SLACK)
        except ZoteroError as e:
            return format_usage(None, self.drive.remote, error=str(e), fmt=fmt or SLACK)

    def handle(self, ev: dict) -> None:
        if not self.bot.allowed(ev, thread=True):
            return
        fmt = self.bot.fmt(ev)
        refs, name = parse_command(ev.get("text", ""))
        logger.info("저장 명령 [%s] user=%s collection=%r refs=%s thread=%s",
                    fmt.name, ev.get("user"), name, refs, ev.get("thread_ts"))
        if name.lower() in HELP_WORDS:
            self.bot.reply(ev, self.usage(fmt))
            return

        title_hint = None
        ref = refs[0] if refs else None
        if ref is None and ev.get("thread_ts"):
            parent = self.bot.parent_text(ev)
            ref = first_ref(clean_slack_text(parent))
            m = TITLE_RE.search(parent)
            title_hint = m.group(1).strip() if m else None
        if ref is None:
            if fmt.name == "zulip":
                self.bot.reply(ev, "논문을 찾지 못했습니다. `저장 <컬렉션명> <arXiv·DOI 링크>`처럼 링크를 적거나, "
                                   "논문 topic의 메시지를 인용(Quote and reply)해서 보내 주세요.")
            else:
                self.bot.reply(ev, "논문을 찾지 못했습니다. 논문 스레드 안에서 멘션하거나, arXiv·DOI 링크를 함께 적어주세요.")
            return

        self.bot.react(ev, "eyes")
        try:
            result = save_paper(self.cfg, self.zot, self.drive, resolve(ref, title_hint), name)
        except (ZoteroError, DriveError, RuntimeError) as e:
            logger.error("저장 실패 (%s → %s): %s", ref, name, e)
            self.bot.react(ev, "eyes", remove=True)
            self.bot.react(ev, "x")
            self.bot.reply(ev, f"⚠️ 저장 실패: {esc(str(e))[:500]}")
            return
        logger.info("저장 완료 %s → %s (pdf=%s, key=%s)", ref, result.collection, result.pdf, result.item_key)
        if self.on_saved:
            self.on_saved()
        self.bot.react(ev, "eyes", remove=True)
        self.bot.react(ev, "white_check_mark")
        self.bot.reply(ev, format_result(result, self.drive, fmt))


class ChatHandler:
    def __init__(self, cfg: Config, bot, cache: TextCache | None, agent: Agent | None,
                 on_question: Callable[[], None] | None = None):
        self.cfg, self.bot, self.cache, self.agent = cfg, bot, cache, agent
        self.on_question = on_question
        self.convs = Conversations(cfg.chat.history_turns)
        self.chatlog = ChatLog(cfg.resolve(cfg.chat.chatlog_dir)) if agent else None
        self.active: dict[tuple[str, str], dict] = {}   # 처리 중인 질문의 (채널, 자리표시 ts) → 로그 — 재시작 시 중단 표시용
        self._active_lock = threading.Lock()

    def mark_interrupted(self) -> None:
        """종료 대기 시간을 넘겨 끝내지 못한 질문의 자리표시 메시지에 중단 안내를 남긴다 (cli.serve가 부른다)."""
        with self._active_lock:
            items = list(self.active.items())
        for (ch, ts), log in items:
            try:
                self.bot.update(ch, ts, "⚠️ 봇이 재시작되어 이 질문 처리가 중단됐습니다. 같은 질문을 다시 보내 주세요.")
            except Exception as e:  # noqa: BLE001
                logger.warning("중단 안내 실패: %s", e)
            if self.chatlog:
                self.chatlog.append({**log, "status": "interrupted"})
            logger.warning("재시작으로 중단된 질문: %s", log.get("question", "")[:80])

    def handle(self, ev: dict) -> None:
        if not self.bot.allowed(ev, thread=False):
            return
        ch, user = ev["channel"], ev.get("user")
        question = clean_slack_text(ev.get("text", "")).strip()
        if not question:
            return
        if self.agent is None:
            self.bot.reply(ev, "DM 챗봇이 꺼져 있습니다 (config chat.enabled).", thread=False)
            return
        log = {"user": user, "channel": ch, "question": question, "model": self.cfg.chat.model}
        try:
            self._answer(ev, ch, question, log)
        except Exception as e:  # noqa: BLE001 — Slack 전송 실패 등 어떤 실패든 대화 로그에는 남긴다
            self.chatlog.append({**log, "status": "error", "detail": f"{type(e).__name__}: {e}"[:500]})
            raise

    def _answer(self, ev: dict, ch: str, question: str, log: dict) -> None:
        if question.lower() in RESET_WORDS:
            self.convs.reset(ch)
            self.bot.reply(ev, "🆕 새 대화를 시작합니다.", thread=False)
            self.chatlog.append({**log, "status": "reset"})
            return
        try:
            ensure_gpu(self.cfg.chat)
        except GpuBusy as e:
            logger.warning("GPU 부족으로 DM 거절: %s", e)
            self.bot.reply(ev, f"지금 GPU를 다른 작업이 쓰고 있어서 답변 모델을 올릴 수 없어요 ({e}). "
                               "잠시 후 다시 물어봐 주세요.", thread=False)
            self.chatlog.append({**log, "status": "busy", "detail": str(e)})
            return

        ts = self.bot.reply(ev, "⏳ 생각하는 중… (모델이 내려가 있으면 15초쯤 더 걸려요)", thread=False)["ts"]
        with self._active_lock:
            self.active[(ch, ts)] = log
        try:
            self._answer_with_placeholder(ev, ch, ts, question, log)
        finally:
            with self._active_lock:
                self.active.pop((ch, ts), None)

    def _finalize(self, ev: dict, ch: str, ts: str, final: str, last_update_at: float) -> None:
        """자리표시 메시지를 최종 답으로 교체. 직전 스트리밍 갱신과 1초 이상 띄우고, 실패하면 잠시 뒤 재시도한다.
        끝내 안 되면 새 메시지로 보내되 자리표시에는 안내만 남겨 같은 답이 두 번 보이지 않게 한다."""
        time.sleep(max(0.0, 1.0 - (time.time() - last_update_at)))
        for attempt, wait in enumerate((*_FINAL_RETRY_WAITS, None)):
            try:
                self.bot.update(ch, ts, final)
                return
            except Exception as e:  # noqa: BLE001 — 429 등
                logger.warning("최종 답 갱신 실패 (%d/%d): %s", attempt + 1, len(_FINAL_RETRY_WAITS) + 1, e)
                if wait is not None:
                    time.sleep(wait)
        self.bot.reply(ev, final, thread=False)
        try:
            self.bot.update(ch, ts, "⬇️ 답변은 아래 메시지에 있습니다.")
        except Exception as e:  # noqa: BLE001
            logger.warning("자리표시 정리 실패: %s", e)

    def _answer_with_placeholder(self, ev: dict, ch: str, ts: str, question: str, log: dict) -> None:
        status: list[str] = []
        last = [0.0]
        fmt = self.bot.fmt(ev)
        every = self.bot.update_every(ev)

        def update(text: str, force: bool = False) -> None:
            if not force and time.time() - last[0] < every:
                return
            last[0] = time.time()
            try:
                self.bot.update(ch, ts, text[:_PREVIEW_MAX + 100])
            except Exception as e:  # noqa: BLE001 — 중간 갱신 실패는 무시
                logger.debug("chat.update 실패: %s", e)

        def on_status(s: str) -> None:
            status.append(s)
            update("\n".join(status) + "\n🧠 답변 작성 중… (논문 전문을 읽으면 1분 안팎)", force=True)

        def on_partial(text: str) -> None:
            if time.time() - last[0] < every:   # 갱신 간격 전이면 변환도 생략 (토큰마다 마크다운 변환을 하지 않게)
                return
            shown = fmt.render(text)
            if len(shown) > _PREVIEW_MAX:
                update(shown[:_PREVIEW_MAX] + f"\n…(계속 작성 중, {len(shown)}자)")
            else:
                update(shown + " ▌")

        if self.on_question:
            self.on_question()   # 라이브러리 동기화는 백그라운드 — DM은 현재 캐시로 바로 답한다
        try:
            res = self.agent.run(question, self.convs.history(ch), on_status, on_partial, math_style=fmt.math_style)
        except LLMError as e:
            logger.error("LLM 실패: %s", e)
            update(f"⚠️ 답변 모델 호출 실패: {esc(str(e))[:300]}", force=True)
            self.chatlog.append({**log, "status": "error", "detail": str(e)})
            return
        answer = res.answer or "(빈 답변 — 다시 물어봐 주세요)"
        final = fmt.render(answer) + format_sources(self.cache, res.papers, fmt=fmt)
        self._finalize(ev, ch, ts, final[:fmt.max_chars], last[0])
        self.convs.add(ch, question, answer)
        self.chatlog.append({**log, "status": "ok", "answer": answer, "tools": res.tools,
                             "papers": [{"key": k, "title": (self.cache.get(k).title if self.cache.get(k) else "")}
                                        for k in res.papers],
                             "steps": res.steps, "load_s": round(res.load_s, 1),
                             "duration_s": round(res.duration_s, 1),
                             "prompt_tokens": res.prompt_tokens, "chars_per_token": res.ratio_en})
        logger.info("DM 답변 user=%s steps=%d tools=%d %.1fs (load %.1fs)",
                    ev.get("user"), res.steps, len(res.tools), res.duration_s, res.load_s)
