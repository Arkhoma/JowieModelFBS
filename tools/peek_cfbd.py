"""Print the shape and first record of downloaded CFBD files.

Usage: python tools/peek_cfbd.py 2024/talent 2024/returning_production coaches
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    for name in sys.argv[1:]:
        with gzip.open(RAW / f"{name}.json.gz", "rt", encoding="utf-8") as f:
            data = json.load(f)
        first = data[0] if isinstance(data, list) and data else data
        text = json.dumps(first)
        print(f"\n{name}: {len(data)} records\n  {text[:900]}")


if __name__ == "__main__":
    main()
