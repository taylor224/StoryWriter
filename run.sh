#!/usr/bin/env bash
# macOS / Linux 실행 스크립트 (Windows 는 run.bat)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
  cat <<'MSG'
[!] .venv 가 없습니다. 먼저 아래를 실행하세요:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch torchaudio          # Mac 은 CUDA 인덱스 없이 그냥 설치
    pip install -r requirements.txt

MSG
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".env" ]; then
  echo "[!] .env 가 없습니다. cp .env.example .env 후 HF_TOKEN 을 채워 주세요."
  exit 1
fi

exec python -m app.main
