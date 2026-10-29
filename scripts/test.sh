#!/usr/bin/env bash
# Run every automated check in the academy: Go tools + Python lab tests. No GPU needed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "== Go (platform/)"
(cd "$ROOT/platform" && go vet ./... && go test -race ./...)

echo "== Python labs"
for d in "$ROOT"/levels/*/labs; do
  if ls "$d"/test_*.py >/dev/null 2>&1; then
    echo "-- ${d#$ROOT/}"
    (cd "$d" && python3 -m unittest discover -p 'test_*.py' -q)
  fi
done
echo "all checks passed"
