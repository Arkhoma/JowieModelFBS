"""Make the `src/` layout importable from the test suite.

Every test module used to repeat this three-line incantation. One home
for it means adding a test file no longer requires remembering the ritual.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
