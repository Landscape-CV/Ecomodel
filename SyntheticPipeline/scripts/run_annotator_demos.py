#!/usr/bin/env python
"""Launcher for instance annotator demos (cwd-independent)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SP = Path(__file__).resolve().parents[1]
if str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))

runpy.run_module("instance_annotator.cli_demo", run_name="__main__")
