"""`python -m paperbot` → paperbot.cli.main"""
import sys

from .cli import main

sys.exit(main())
