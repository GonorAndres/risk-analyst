"""Adds project src/ to sys.path for local imports."""

import sys
from pathlib import Path

_PROJECT_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _PROJECT_SRC not in sys.path:
    sys.path.insert(0, _PROJECT_SRC)
