"""Fail if any page in a static build links to a file that isn't there.

    python tools/check_static_links.py _site

Checks every relative href/src in every .html file (anchors, query
strings and absolute http(s) URLs are ignored). Run by the publish
workflow after the build, so a dead link fails CI instead of shipping.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

LINK = re.compile(r'(?:href|src)="([^"#?]+)')


def broken_links(site: Path) -> dict[str, list[str]]:
    """target -> pages linking to it, for every target that doesn't exist."""
    broken: dict[str, list[str]] = defaultdict(list)
    for page in site.glob("*.html"):
        for target in set(LINK.findall(page.read_text(encoding="utf-8"))):
            if target.startswith(("http://", "https://", "mailto:", "/")):
                continue
            if not (site / target).exists():
                broken[target].append(page.name)
    return dict(broken)


def main() -> None:
    site = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    pages = len(list(site.glob("*.html")))
    broken = broken_links(site)
    for target, sources in sorted(broken.items()):
        print(f"MISSING {target}  <- {len(sources)} page(s), e.g. {sources[0]}")
    print(f"{pages} pages checked, {len(broken)} missing targets")
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
