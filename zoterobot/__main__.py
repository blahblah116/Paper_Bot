"""`python -m zoterobot` → zoterobot.cli.main"""
import sys

from .cli import main

sys.exit(main())
