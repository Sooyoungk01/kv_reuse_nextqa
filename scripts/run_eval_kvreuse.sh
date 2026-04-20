#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/run_eval.py --mode kv_reuse --config configs/default.yaml
