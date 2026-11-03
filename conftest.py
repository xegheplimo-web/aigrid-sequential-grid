"""Repo-root pytest bootstrap.

Makes the repository importable (``aigrid``, ``scripts``) when the package has
not been installed, e.g. when running ``pytest`` straight from a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
