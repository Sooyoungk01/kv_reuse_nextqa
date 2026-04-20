from __future__ import annotations

import json
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: str | Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_accuracy(rows: List[Dict]) -> float:
    kept = [r for r in rows if "correct" in r]
    if not kept:
        return 0.0
    return sum(int(bool(r["correct"])) for r in kept) / len(kept)


def _summarize_numeric_list(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(float(v) for v in values)
    n = len(vals)
    return {
        "avg": sum(vals) / n,
        "p50": vals[(n - 1) // 2],
        "p95": vals[min(n - 1, ceil(n * 0.95) - 1)],
        "max": vals[-1],
    }


def _summarize_numeric_dict_field(rows: List[Dict], field_name: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        obj = row.get(field_name)
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(key, []).append(float(value))
    return {key: _summarize_numeric_list(values) for key, values in buckets.items()}


def summarize_file(path: str | Path) -> Dict:
    rows = read_jsonl(path)
    acc = compute_accuracy(rows)
    kept = [r for r in rows if "correct" in r]
    status_counts = Counter(r.get("status", "<missing>") for r in rows)
    error_counts = Counter(r.get("error", "<missing>") for r in rows if r.get("status") == "error")
    return {
        "path": str(path),
        "n": len(rows),
        "n_scored": len(kept),
        "accuracy": acc,
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts.most_common(5)),
        "latency_summary_ms": _summarize_numeric_dict_field(kept, "latency_ms"),
        "latency_profile_summary_ms": _summarize_numeric_dict_field(kept, "latency_profile_ms"),
        "prefill_latency_summary_ms": _summarize_numeric_dict_field(kept, "prefill_latency_ms"),
        "prefill_profile_latency_summary_ms": _summarize_numeric_dict_field(kept, "prefill_profile_latency_ms"),
    }
