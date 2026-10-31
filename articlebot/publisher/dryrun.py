"""Dry-run publisher: prints the Slack or Zulip layout and saves images under storage.preview_dir.
전송 없이 레이아웃을 stdout에 출력하고 그림은 preview_dir에 저장 (눈으로 검수용).
"""
from __future__ import annotations

import os

from ..config import Config
from ..models import Article, Extracted, FetchedImage
from .base import Publisher, _slug
from .slack import format_main, format_thread
from .zulip import build_zulip_messages, format_main_zulip, format_thread_zulip


class DryRunPublisher(Publisher):
    """전송 없이 stdout 출력 + preview_dir/<글>/ 에 요약·그림 저장 (눈으로 검수용). backend에 맞는 레이아웃."""

    def __init__(self, cfg: Config):
        self.backend = cfg.publisher.backend
        self.default_channel = cfg.slack.channel
        self.zulip_dest = f"{cfg.zulip.stream} / {cfg.zulip.topic}"
        self.preview_dir = cfg.resolve(cfg.storage.preview_dir)

    def publish(self, article: Article, summary: str, ex: Extracted, images: list[FetchedImage]) -> None:
        zulip = self.backend == "zulip"
        channel = self.zulip_dest if zulip else (article.channel or self.default_channel)
        out_dir = os.path.join(self.preview_dir, _slug(article))
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for i, img in enumerate(images, start=1):
            path = os.path.join(out_dir, f"{i:02d}.{img.ext}")
            with open(path, "wb") as f:
                f.write(img.data)
            saved.append(f"  {path}  ({img.width}x{img.height}, {len(img.data) // 1024}KB) {img.label}")
        if zulip:
            msgs = build_zulip_messages(format_main_zulip(article, summary), format_thread_zulip(summary, ex),
                                        [f"[{img.label}](<업로드 예정 {i:02d}.{img.ext}>)" for i, img in enumerate(images, start=1)])
            with open(os.path.join(out_dir, "zulip.txt"), "w", encoding="utf-8") as f:
                f.write(f"[{channel}]\n" + "\n\n".join(msgs) + "\n")
            print("=" * 70)
            for i, m in enumerate(msgs, start=1):
                print(f"[DRY-RUN/zulip → {channel}] 메시지 {i}/{len(msgs)}" + (f" (+그림 {len(images)}장)" if i == 1 else ""))
                print("-" * 70)
                print(m)
                print("-" * 70)
        else:
            main, thread = format_main(article, summary), format_thread(summary, ex)
            with open(os.path.join(out_dir, "slack.txt"), "w", encoding="utf-8") as f:
                f.write(f"[채널 {channel}]\n{main}\n\n[스레드]\n{thread}\n")
            print("=" * 70)
            print(f"[DRY-RUN/slack → {channel}] 채널 메인 메시지")
            print("-" * 70)
            print(main)
            print("-" * 70)
            print(f"[DRY-RUN/slack → {channel}] 스레드 댓글 (+그림 {len(images)}장)")
            print("-" * 70)
            print(thread)
        if saved:
            print("-" * 70)
            print("[그림 미리보기 저장]")
            print("\n".join(saved))
        print("=" * 70)
