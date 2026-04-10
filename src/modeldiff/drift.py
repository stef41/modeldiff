"""Statistical drift detection between snapshots."""

from __future__ import annotations

import math
from typing import Dict, List

from modeldiff._types import Snapshot


def length_drift(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
) -> Dict[str, float]:
    """Detect response length distribution drift between two snapshots."""
    lens_a = [r.word_count for r in snapshot_a.responses if not r.is_error]
    lens_b = [r.word_count for r in snapshot_b.responses if not r.is_error]

    if not lens_a or not lens_b:
        return {"mean_a": 0, "mean_b": 0, "drift": 0.0}

    mean_a = sum(lens_a) / len(lens_a)
    mean_b = sum(lens_b) / len(lens_b)
    std_a = _std(lens_a, mean_a)

    drift = abs(mean_a - mean_b) / std_a if std_a > 0 else 0.0

    return {
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "std_a": round(std_a, 2),
        "drift_sigma": round(drift, 3),
        "drift_significant": drift > 2.0,
    }


def refusal_drift(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
) -> Dict[str, float]:
    """Detect changes in refusal rate."""
    ref_a = sum(1 for r in snapshot_a.responses if r.is_refusal)
    ref_b = sum(1 for r in snapshot_b.responses if r.is_refusal)
    total_a = len(snapshot_a.responses) or 1
    total_b = len(snapshot_b.responses) or 1

    rate_a = ref_a / total_a
    rate_b = ref_b / total_b

    return {
        "refusal_rate_a": round(rate_a, 4),
        "refusal_rate_b": round(rate_b, 4),
        "delta": round(rate_b - rate_a, 4),
        "drift_significant": abs(rate_b - rate_a) > 0.1,
    }


def latency_drift(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
) -> Dict[str, float]:
    """Detect latency distribution drift."""
    lat_a = [r.latency_ms for r in snapshot_a.responses if not r.is_error and r.latency_ms > 0]
    lat_b = [r.latency_ms for r in snapshot_b.responses if not r.is_error and r.latency_ms > 0]

    if not lat_a or not lat_b:
        return {"mean_a": 0, "mean_b": 0, "drift": 0.0}

    mean_a = sum(lat_a) / len(lat_a)
    mean_b = sum(lat_b) / len(lat_b)
    std_a = _std(lat_a, mean_a)

    drift = abs(mean_a - mean_b) / std_a if std_a > 0 else 0.0

    return {
        "mean_a_ms": round(mean_a, 2),
        "mean_b_ms": round(mean_b, 2),
        "std_a_ms": round(std_a, 2),
        "drift_sigma": round(drift, 3),
        "drift_significant": drift > 2.0,
    }


def vocabulary_drift(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    top_n: int = 100,
) -> Dict[str, object]:
    """Detect vocabulary usage drift between snapshots."""
    words_a = _word_freq(snapshot_a, top_n)
    words_b = _word_freq(snapshot_b, top_n)

    # Jaccard similarity of top-N words
    set_a = set(words_a.keys())
    set_b = set(words_b.keys())
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else 1.0

    # New words in B not in A
    new_words = sorted(set_b - set_a)
    lost_words = sorted(set_a - set_b)

    return {
        "jaccard_similarity": round(jaccard, 4),
        "new_words": new_words[:20],
        "lost_words": lost_words[:20],
        "vocab_a_size": len(set_a),
        "vocab_b_size": len(set_b),
        "drift_significant": jaccard < 0.7,
    }


def full_drift_report(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
) -> Dict[str, Dict]:
    """Run all drift detectors and return combined report."""
    return {
        "length": length_drift(snapshot_a, snapshot_b),
        "refusal": refusal_drift(snapshot_a, snapshot_b),
        "latency": latency_drift(snapshot_a, snapshot_b),
        "vocabulary": vocabulary_drift(snapshot_a, snapshot_b),
    }


def _std(values: List[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _word_freq(snapshot: Snapshot, top_n: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for r in snapshot.responses:
        if r.is_error:
            continue
        for w in r.output.lower().split():
            freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    return dict(items)
