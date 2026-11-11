#!/usr/bin/env bash
# Start the built image, wait until it is healthy, and run a real search.
# Used by CI and runnable locally:  ./tests/smoke_test.sh semantic-search:local
set -euo pipefail

IMAGE="${1:?usage: smoke_test.sh <image>}"
PORT="${PORT:-8000}"
NAME="semantic-search-smoke-$$"

cleanup() {
  echo "--- container logs ---"
  docker logs "$NAME" || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null

echo "Waiting for /health..."
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "http://localhost:$PORT/health"
echo

echo "Running a search..."
response=$(curl -fsS "http://localhost:$PORT/search-query?query=stock%20market&k=3")
echo "$response"
python3 -c 'import json, sys; r = json.loads(sys.argv[1])["results"]; assert len(r) == 3, r' "$response"
echo "Smoke test passed."
