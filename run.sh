#!/usr/bin/env bash
# macOS / Linux launcher (Windows uses run.bat)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
  cat <<'MSG'
[!] No .venv found. Run this first:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch torchaudio          # on Mac, install without the CUDA index
    pip install -r requirements.txt

MSG
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".env" ]; then
  echo "[!] No .env found. Run 'cp .env.example .env' and fill in HF_TOKEN."
  exit 1
fi

exec python -m app.main
