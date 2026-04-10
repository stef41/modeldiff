"""Model behavioral fingerprinting — compact signature."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from modeldiff._types import FingerprintResult, Prompt, Response, Snapshot


def fingerprint(snapshot: Snapshot) -> FingerprintResult:
    """Generate a behavioral fingerprint from a snapshot.

    Dimensions:
    - verbosity: normalized average response length
    - refusal_rate: fraction of responses that refuse
    - error_rate: fraction of responses with errors
    - vocabulary_richness: type-token ratio
    - avg_latency: average latency in ms
    - consistency: how similar responses are to each other in length
    - formality: ratio of formal markers to casual ones
    """
    responses = [r for r in snapshot.responses if not r.is_error]
    total = len(snapshot.responses) or 1

    dims: Dict[str, float] = {}

    # Verbosity
    if responses:
        lengths = [r.word_count for r in responses]
        dims["avg_word_count"] = sum(lengths) / len(lengths)
        dims["verbosity"] = min(dims["avg_word_count"] / 500.0, 1.0)  # normalized to 0-1
    else:
        dims["avg_word_count"] = 0.0
        dims["verbosity"] = 0.0

    # Refusal rate
    n_refusal = sum(1 for r in snapshot.responses if r.is_refusal)
    dims["refusal_rate"] = n_refusal / total

    # Error rate
    dims["error_rate"] = snapshot.n_errors / total

    # Vocabulary richness (type-token ratio on first 1000 words)
    all_words: List[str] = []
    for r in responses:
        all_words.extend(r.output.lower().split()[:200])
    if all_words:
        unique = len(set(all_words))
        dims["vocabulary_richness"] = unique / len(all_words)
    else:
        dims["vocabulary_richness"] = 0.0

    # Average latency
    latencies = [r.latency_ms for r in responses if r.latency_ms > 0]
    dims["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0

    # Consistency (CV of response lengths)
    if responses and len(responses) > 1:
        lengths = [float(r.word_count) for r in responses]
        mean = sum(lengths) / len(lengths)
        std = math.sqrt(sum((l - mean) ** 2 for l in lengths) / len(lengths))
        dims["length_consistency"] = 1.0 - min(std / mean if mean > 0 else 0, 1.0)
    else:
        dims["length_consistency"] = 1.0

    # Formality (simple heuristic: contractions vs full forms)
    text = " ".join(r.output for r in responses).lower()
    casual_markers = sum(text.count(m) for m in ["don't", "can't", "won't", "i'm", "you're", "!!", "lol", "gonna"])
    formal_markers = sum(text.count(m) for m in ["therefore", "furthermore", "however", "additionally", "consequently"])
    total_markers = casual_markers + formal_markers
    dims["formality"] = formal_markers / total_markers if total_markers > 0 else 0.5

    return FingerprintResult(
        model_name=snapshot.model_name,
        dimensions=dims,
    )


def compare_fingerprints(
    fp_a: FingerprintResult,
    fp_b: FingerprintResult,
) -> Dict[str, object]:
    """Compare two fingerprints and compute distance."""
    dims_a = fp_a.dimensions
    dims_b = fp_b.dimensions
    all_keys = set(dims_a.keys()) | set(dims_b.keys())

    deltas: Dict[str, float] = {}
    for key in sorted(all_keys):
        va = dims_a.get(key, 0.0)
        vb = dims_b.get(key, 0.0)
        deltas[key] = round(vb - va, 4)

    # Euclidean distance on normalized dims
    dist = 0.0
    for key in all_keys:
        va = dims_a.get(key, 0.0)
        vb = dims_b.get(key, 0.0)
        dist += (va - vb) ** 2
    dist = math.sqrt(dist)

    return {
        "model_a": fp_a.model_name,
        "model_b": fp_b.model_name,
        "deltas": deltas,
        "euclidean_distance": round(dist, 4),
        "similar": dist < 0.3,
    }
