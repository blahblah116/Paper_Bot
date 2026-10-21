#!/usr/bin/env python3
"""paperbot entry point (same as `python -m paperbot`) — kept for scripts/loop.sh and crontab.
paperbot 진입점 (`python -m paperbot`과 동일) — loop.sh·crontab 호환용."""
import sys

from paperbot.cli import main

sys.exit(main())
