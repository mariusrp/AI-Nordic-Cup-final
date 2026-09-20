"""Shared constants and upstream imports for the drone workstream."""
import os
import sys

UPSTREAM = os.environ.get("UPSTREAM") or next(
    (p for p in ("/workspace/upstream", "/home/claude/Nordic-AI-Cup-2026") if os.path.isdir(p)),
    "/home/claude/Nordic-AI-Cup-2026")
DRONE_UPSTREAM = os.path.join(UPSTREAM, "drone-flyby")
if DRONE_UPSTREAM not in sys.path:
    sys.path.insert(0, DRONE_UPSTREAM)

from dtos import OBJECT_CLASSES  # noqa: E402

W, H = 3840, 2160
VIEW_W, VIEW_H = 960, 540
LEVEL_SCALE = {0: 4, 1: 2, 2: 1}  # source px per view px
CLASSES = list(OBJECT_CLASSES)
CLS_INDEX = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)
