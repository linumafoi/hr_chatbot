#!/usr/bin/env bash
# Start the HR AI Chatbot gateway (serves the UI + API).
set -e

cd "$(dirname "$0")"

# Create / activate venv
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Load .env (create from example if missing)
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example - review the values before production use."
fi

export PYTHONPATH="$(pwd)/backend"

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8000}"

echo "Starting HR AI Chatbot on http://localhost:${PORT}"
exec uvicorn app.main:app --app-dir backend --host "$HOST" --port "$PORT"
