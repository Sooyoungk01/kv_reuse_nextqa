#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from src.eval.metrics import summarize_file

def get_avg(summary, section, key):
    return summary.get(section, {}).get(key, {}).get('avg')

base = summarize_file('results/baseline_results.jsonl')
reuse = summarize_file('results/kv_reuse_results.jsonl')
print('Baseline:', base)
print('KV-Reuse:', reuse)
print('Delta accuracy:', reuse['accuracy'] - base['accuracy'])

base_e2e = get_avg(base, 'latency_summary_ms', 'total_e2e')
reuse_e2e = get_avg(reuse, 'latency_summary_ms', 'total_e2e')
if base_e2e is not None and reuse_e2e is not None:
    print('Baseline avg e2e latency (ms):', base_e2e)
    print('KV-Reuse avg e2e latency (ms):', reuse_e2e)
    print('Delta avg e2e latency (ms):', reuse_e2e - base_e2e)

base_prefill = get_avg(base, 'prefill_latency_summary_ms', 'total')
reuse_prefill = get_avg(reuse, 'prefill_latency_summary_ms', 'total')
if base_prefill is not None and reuse_prefill is not None:
    print('Baseline avg prefill latency (ms):', base_prefill)
    print('KV-Reuse avg prefill latency (ms):', reuse_prefill)
    print('Delta avg prefill latency (ms):', reuse_prefill - base_prefill)
PY
