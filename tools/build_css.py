"""Compile Tailwind ahead of time into app/static/site.css.

    python tools/build_css.py

Uses the Tailwind v3 standalone CLI (no Node needed) from
tools/bin/tailwindcss[.exe], or $TAILWIND_BIN. Get it from the
tailwindlabs/tailwindcss GitHub release v3.4.17 (CI downloads the Linux
build; see .github/workflows/publish.yml).

site.css is committed so the local server works without the binary;
rerun this after adding Tailwind classes to a template or JS file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = "tailwindcss.exe" if os.name == "nt" else "tailwindcss"


def main() -> None:
    binary = Path(os.environ.get("TAILWIND_BIN", ROOT / "tools" / "bin" / EXE))
    if not binary.exists():
        sys.exit(f"Tailwind CLI not found at {binary}. See this file's docstring.")
    subprocess.run(
        [str(binary), "-c", "tailwind.config.js", "-i", "app/tailwind.input.css",
         "-o", "app/static/site.css", "--minify"],
        cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
