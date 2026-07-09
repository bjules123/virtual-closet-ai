#!/usr/bin/env bash
# Start the Virtual Closet backend (also serves the frontend at /app)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load API keys from .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Kill any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo ""
echo "  🚀  Starting Virtual Closet..."
echo "  📡  API:      http://127.0.0.1:8000"
echo "  👗  App:      http://127.0.0.1:8000/app/"
echo ""
echo "  AI detection: set ANTHROPIC_API_KEY (best) or OPENAI_API_KEY (also works)"
echo "  Press Ctrl+C to stop."
echo ""

exec venv/bin/uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
