"""Shared setup for the browser smoke tests.

These drive a real browser against a BUILT static site, so they only run
when you point them at one:

    python tools/build_static_site.py --out _site          # the full build
    SITE_DIR=_site pytest tests/e2e -q                      # Linux / macOS
    $env:SITE_DIR="_site"; pytest tests/e2e -q              # PowerShell

Playwright needs a browser. In CI that's `playwright install chromium`.
Locally, if you'd rather not download one, use the Edge or Chrome you
already have:  pytest tests/e2e --browser-channel msedge

Without SITE_DIR every test here skips, so a plain `pytest` is unaffected.
"""

from __future__ import annotations

import functools
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):          # keep test output readable
        pass


@pytest.fixture(scope="session")
def site_url():
    """Serve the built site on a free local port for the whole session."""
    site_dir = os.environ.get("SITE_DIR")
    if not site_dir or not Path(site_dir).is_dir():
        pytest.skip("set SITE_DIR to a built static site to run the browser smoke tests")
    handler = functools.partial(_QuietHandler, directory=site_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def browser_page(site_url, page):
    """A page that records anything that went wrong on OUR site.

    `site_url` must stay first: session fixtures are set up in argument
    order, and its skip has to fire BEFORE Playwright tries to launch a
    browser. Otherwise a plain `pytest` errors on machines with no browser.

    Third-party hosts (logo CDNs, fonts) are ignored on purpose: a CDN
    hiccup should not fail the weekly publish, but a JavaScript error or a
    missing file of ours should.
    """
    problems: list[str] = []
    page.on("pageerror", lambda error: problems.append(f"JS error: {error}"))
    page.on("console", lambda msg: msg.type == "error"
            and msg.location.get("url", "").startswith(site_url)
            and problems.append(f"console error: {msg.text}"))
    page.on("response", lambda r: r.url.startswith(site_url) and r.status >= 400
            and problems.append(f"HTTP {r.status}: {r.url}"))
    return SimpleNamespace(page=page, problems=problems,
                           open=lambda path: page.goto(f"{site_url}/{path}"))
