#!/bin/bash
# Run all models in parallel (multi-threaded within one Python process).
# Already-completed models are skipped automatically.
#
# Usage:
#   bash run_all.sh                # Run all 6 models in parallel
#   bash run_all.sh --workers 3    # 3 models at a time
#   bash run_all.sh --dry-run      # Preview only, no API calls

cd "$(dirname "$0")"
python run_predictions.py "$@"
