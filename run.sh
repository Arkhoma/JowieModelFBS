#!/usr/bin/env bash
# One-shot setup and launch for macOS and Linux.
#
# Creates a virtual environment, installs dependencies, downloads the
# public data (~45 MB, no API key needed), and starts the server.
# Safe to re-run: every step is skipped if already done.

set -euo pipefail
cd "$(dirname "$0")"

echo
echo "=== CFB Rankings ==="
echo

# macOS ships a "python" shim that is not usable; prefer python3.
PYTHON="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found."
    echo "  macOS:  brew install python"
    echo "  Ubuntu: sudo apt install python3 python3-venv"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt --quiet

if [ ! -d "data/raw/schedules" ]; then
    echo
    echo "Downloading data (~45 MB, one time)..."
    .venv/bin/python tools/fetch_mirror.py
fi

URL="http://127.0.0.1:8421"
echo
echo "Starting server at $URL"
echo "Press Ctrl+C to stop."
echo

# Open a browser once the server is actually accepting connections,
# rather than racing it and showing a connection-refused page.
(
    for _ in $(seq 1 40); do
        if curl -s -o /dev/null "$URL" 2>/dev/null; then
            command -v open >/dev/null && open "$URL" && break
            command -v xdg-open >/dev/null && xdg-open "$URL" && break
            break
        fi
        sleep 1
    done
) &

exec .venv/bin/python -m uvicorn app.main:app --port 8421
