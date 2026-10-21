"""The two request handlers: thread mention → save to Zotero/Drive, DM → chat agent.
요청 핸들러 둘 — 스레드 멘션(Zotero/Drive 저장), DM(챗봇).

- SaveHandler: '@zoterobot <컬렉션>' → 원글(또는 멘션에 적힌 링크)의 논문을 찾아 library.save_paper.
- ChatHandler: DM 질문 → GPU 확인 → 자리표시 메시지 → agent.run(진행 상황·부분 답변을 chat.update로 갱신)
  → 최종 답 + 참고 논문. 허용된 사용자의 DM은 결과가 어떻든 대화 로그에 남긴다.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from paperbot.publisher import md_to_mrkdwn

from .agent import Agent, LLMError
from .config import Config
from .drive import DriveError, DriveUploader
from .gpu import GpuBusy, ensure_gpu
from .library import save_paper
from .messages import HELP_WORDS, TITLE_RE, esc, format_result, format_sources, format_usage, parse_command
from .resolver import clean_slack_text, first_ref, resolve
from .session import RESET_WORDS, ChatLog, Conversations
from .slackbot import SlackBot
from .textcache import TextCache
from .zotero import ZoteroClient, ZoteroError

logger = logging.getLogger(__name__)

_UPDATE_EVERY = 2.5        # DM 답변 스트리밍 시 chat.update 최소 간격(초) — Slack rate limit 여유
_PREVIEW_MAX = 3800        # 실시간 갱신에 보여줄 최대 글자 수 (전체는 완료 시 최종 갱신에서)


class SaveHandler:
    def __init__(self, cfg: Config, bot: SlackBot, zot: ZoteroClient, drive: DriveUploader,
                 on_saved: Callable[[], None] | None = None):
        self.cfg, self.bot, self.zot, self.drive = cfg, bot, zot, drive
        self.on_saved = on_saved

    def usage(self) -> str:
        try:
            names = sorted({c["name"] for c in self.zot.collections()}, key=str.lower)
            return format_usage(names, self.drive.remote)
        except ZoteroError as e:
            return format_usage(None, self.drive.remote, error=str(e))

    def handle(self, ev: dict) -> None:
        if not self.bot.allowed(ev, thread=True):
            return
        refs, name = parse_command(ev.get("text", ""))
        logger.info("저장 명령 user=%s collection=%r refs=%s thread=%s",
                    ev.get("user"), name, refs, ev.get("thread_ts"))
        if name.lower() in HELP_WORDS:
            self.bot.reply(ev, self.usage())
            return

        title_hint = None
        ref = refs[0] if refs else None
        if ref is None and ev.get("thread_ts"):
            parent = self.bot.parent_text(ev)
            ref = first_ref(clean_slack_text(parent))
            m = TITLE_RE.search(parent)
            title_hint = m.group(1).strip() if m else None
        if ref is None:
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
        self.bot.reply(ev, format_result(result, self.drive))


class ChatHandler:
    def __init__(self, cfg: Config, bot: SlackBot, cache: TextCache | None, agent: Agent | None,
                 on_question: Callable[[], None] | None = None):
        self.cfg, self.bot, self.cache, self.agent = cfg, bot, cache, agent
        self.on_question = on_question
        self.convs = Conversations(cfg.chat.history_turns)
        self.chatlog = ChatLog(cfg.resolve(cfg.chat.chatlog_dir)) if agent else None

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
        status: list[str] = []
        last = [0.0]

        def update(text: str, force: bool = False) -> None:
            if not force and time.time() - last[0] < _UPDATE_EVERY:
                return
            last[0] = time.time()
            try:
                self.bot.update(ch, ts, text[:_PREVIEW_MAX + 100])
            except Exception as e:  # noqa: BLE001 — 중간 갱신 실패는 무시
                logger.debug("chat.update 실패: %s", e)

        def on_status(s: str) -> None:
            status.append(s)
            update("\n".join(status) + "\n⏳ 정리하는 중…", force=True)

        def on_partial(text: str) -> None:
            shown = md_to_mrkdwn(text)
            if len(shown) > _PREVIEW_MAX:
                update(shown[:_PREVIEW_MAX] + f"\n…(계속 작성 중, {len(shown)}자)")
            else:
                update(shown + " ▌")

        if self.on_question:
            self.on_question()   # 라이브러리 동기화는 백그라운드 — DM은 현재 캐시로 바로 답한다
        try:
            res = self.agent.run(question, self.convs.history(ch), on_status, on_partial)
        except LLMError as e:
            logger.error("LLM 실패: %s", e)
            update(f"⚠️ 답변 모델 호출 실패: {esc(str(e))[:300]}", force=True)
            self.chatlog.append({**log, "status": "error", "detail": str(e)})
            return
        answer = res.answer or "(빈 답변 — 다시 물어봐 주세요)"
        final = md_to_mrkdwn(answer) + format_sources(self.cache, res.papers)
        try:
            self.bot.update(ch, ts, final[:39000])
        except Exception:  # noqa: BLE001 — 갱신 실패 시 새 메시지로
            self.bot.reply(ev, final[:39000], thread=False)
        self.convs.add(ch, question, answer)
        self.chatlog.append({**log, "status": "ok", "answer": answer, "tools": res.tools,
                             "papers": [{"key": k, "title": (self.cache.get(k).title if self.cache.get(k) else "")}
                                        for k in res.papers],
                             "steps": res.steps, "load_s": round(res.load_s, 1),
                             "duration_s": round(res.duration_s, 1)})
        logger.info("DM 답변 user=%s steps=%d tools=%d %.1fs (load %.1fs)",
                    ev.get("user"), res.steps, len(res.tools), res.duration_s, res.load_s)
