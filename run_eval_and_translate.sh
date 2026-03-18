#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Step 1: Evaluate ==="
python3 src/evaluate.py "$@"

echo ""
echo "=== Step 2: Translate ==="
python3 src/translate.py "$@"

echo ""
echo "=== All done ==="
