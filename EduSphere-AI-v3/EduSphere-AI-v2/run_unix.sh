#!/usr/bin/env bash
set -euo pipefail
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; fi
if grep -q 'GENERATE_A_FERNET_KEY_AND_PASTE_HERE\|YOUR_PROJECT.supabase.co' .env; then
  echo "Configure ai-service/.env and generate a key with: python generate_encryption_key.py" >&2
  exit 1
fi
uvicorn main:app --host 0.0.0.0 --port 8000
