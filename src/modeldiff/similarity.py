"""Output similarity scoring for modeldiff."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class SimilarityMetric:
    """String constants for similarity metrics."""

    COSINE = "cosine"
    JACCARD = "jaccard"
    LEVENSHTEIN = "levenshtein"
    BLEU = "bleu"
    EXACT_MATCH = "exact_match"


@dataclass
class SimilarityResult:
    """Result of a single similarity measurement."""

    metric: str
    score: float
    detail: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure-Python metric implementations
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1].  Returns 0.0 if either vector is zero.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Returns a float in [0, 1].  Returns 1.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            replace = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, replace))
        prev_row = curr_row
    return prev_row[-1]


def _text_to_char_vector(text: str) -> Dict[str, int]:
    """Convert text to a character frequency vector."""
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def _freq_to_lists(freq_a: Dict[str, int], freq_b: Dict[str, int]) -> Tuple[List[float], List[float]]:
    """Align two frequency dicts into parallel float lists."""
    keys = sorted(set(freq_a) | set(freq_b))
    va = [float(freq_a.get(k, 0)) for k in keys]
    vb = [float(freq_b.get(k, 0)) for k in keys]
    return va, vb


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_score(reference: str, candidate: str) -> float:
    """Compute a simplified BLEU score (unigram–4-gram with brevity penalty)."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not cand_tokens:
        return 0.0

    clipped_counts: List[float] = []
    for n in range(1, 5):
        ref_ng = _ngrams(ref_tokens, n)
        cand_ng = _ngrams(cand_tokens, n)
        if not cand_ng:
            clipped_counts.append(0.0)
            continue
        ref_freq: Dict[Tuple[str, ...], int] = {}
        for ng in ref_ng:
            ref_freq[ng] = ref_freq.get(ng, 0) + 1
        clipped = 0
        for ng in cand_ng:
            if ref_freq.get(ng, 0) > 0:
                clipped += 1
                ref_freq[ng] -= 1
        clipped_counts.append(clipped / len(cand_ng))

    # Geometric mean of non-zero precisions
    nonzero = [p for p in clipped_counts if p > 0]
    if not nonzero:
        return 0.0
    log_avg = sum(math.log(p) for p in nonzero) / len(nonzero)
    precision = math.exp(log_avg)

    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))

    return bp * precision


def _kl_divergence(p: List[float], q: List[float]) -> float:
    """Compute KL divergence D(p || q).  Adds small epsilon to avoid log(0)."""
    eps = 1e-10
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def _js_divergence(p: List[float], q: List[float]) -> float:
    """Compute Jensen–Shannon divergence (symmetric, bounded [0, ln2])."""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return (_kl_divergence(p, m) + _kl_divergence(q, m)) / 2


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutputSimilarity:
    """Compare model outputs using multiple similarity metrics."""

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        self.metrics = metrics or [
            SimilarityMetric.COSINE,
            SimilarityMetric.JACCARD,
            SimilarityMetric.EXACT_MATCH,
        ]

    def compare_texts(self, text_a: str, text_b: str) -> List[SimilarityResult]:
        """Compare two text outputs using configured metrics."""
        results: List[SimilarityResult] = []
        for metric in self.metrics:
            if metric == SimilarityMetric.EXACT_MATCH:
                match = text_a == text_b
                results.append(SimilarityResult(
                    metric=metric,
                    score=1.0 if match else 0.0,
                    detail={"exact": match},
                ))
            elif metric == SimilarityMetric.COSINE:
                fa, fb = _freq_to_lists(
                    _text_to_char_vector(text_a),
                    _text_to_char_vector(text_b),
                )
                score = cosine_similarity(fa, fb)
                # Clamp to 0-1 range
                score = max(0.0, min(1.0, score))
                results.append(SimilarityResult(
                    metric=metric,
                    score=score,
                    detail={"vector_dim": len(fa)},
                ))
            elif metric == SimilarityMetric.JACCARD:
                tokens_a = set(text_a.split())
                tokens_b = set(text_b.split())
                score = jaccard_similarity(tokens_a, tokens_b)
                results.append(SimilarityResult(
                    metric=metric,
                    score=score,
                    detail={
                        "intersection": len(tokens_a & tokens_b),
                        "union": len(tokens_a | tokens_b),
                    },
                ))
            elif metric == SimilarityMetric.LEVENSHTEIN:
                dist = levenshtein_distance(text_a, text_b)
                max_len = max(len(text_a), len(text_b), 1)
                score = 1.0 - dist / max_len
                results.append(SimilarityResult(
                    metric=metric,
                    score=max(0.0, score),
                    detail={"distance": dist, "max_length": max_len},
                ))
            elif metric == SimilarityMetric.BLEU:
                score = _bleu_score(text_a, text_b)
                results.append(SimilarityResult(
                    metric=metric,
                    score=score,
                    detail={"reference_length": len(text_a.split())},
                ))
        return results

    def compare_token_lists(
        self,
        tokens_a: List[Any],
        tokens_b: List[Any],
    ) -> List[SimilarityResult]:
        """Compare two token id lists using set/sequence metrics."""
        results: List[SimilarityResult] = []
        for metric in self.metrics:
            if metric == SimilarityMetric.EXACT_MATCH:
                match = tokens_a == tokens_b
                results.append(SimilarityResult(
                    metric=metric,
                    score=1.0 if match else 0.0,
                    detail={"exact": match},
                ))
            elif metric == SimilarityMetric.JACCARD:
                sa = set(tokens_a)
                sb = set(tokens_b)
                score = jaccard_similarity(sa, sb)
                results.append(SimilarityResult(
                    metric=metric,
                    score=score,
                    detail={
                        "intersection": len(sa & sb),
                        "union": len(sa | sb),
                    },
                ))
            elif metric == SimilarityMetric.LEVENSHTEIN:
                # Use stringified tokens for edit distance
                str_a = " ".join(str(t) for t in tokens_a)
                str_b = " ".join(str(t) for t in tokens_b)
                dist = levenshtein_distance(str_a, str_b)
                max_len = max(len(str_a), len(str_b), 1)
                results.append(SimilarityResult(
                    metric=metric,
                    score=max(0.0, 1.0 - dist / max_len),
                    detail={"distance": dist},
                ))
            elif metric == SimilarityMetric.COSINE:
                # Build frequency vectors over token ids
                freq_a: Dict[Any, int] = {}
                for t in tokens_a:
                    freq_a[t] = freq_a.get(t, 0) + 1
                freq_b: Dict[Any, int] = {}
                for t in tokens_b:
                    freq_b[t] = freq_b.get(t, 0) + 1
                keys = sorted(set(freq_a) | set(freq_b), key=str)
                va = [float(freq_a.get(k, 0)) for k in keys]
                vb = [float(freq_b.get(k, 0)) for k in keys]
                score = cosine_similarity(va, vb)
                score = max(0.0, min(1.0, score))
                results.append(SimilarityResult(
                    metric=metric, score=score, detail={"vector_dim": len(keys)},
                ))
        return results

    def compare_distributions(
        self,
        dist_a: List[float],
        dist_b: List[float],
    ) -> List[SimilarityResult]:
        """Compare two probability distributions.

        Always computes KL divergence, JS divergence, and cosine similarity
        regardless of the configured metrics.
        """
        if len(dist_a) != len(dist_b):
            raise ValueError("Distributions must have the same length")

        kl = _kl_divergence(dist_a, dist_b)
        js = _js_divergence(dist_a, dist_b)
        cos = cosine_similarity(
            [float(x) for x in dist_a],
            [float(x) for x in dist_b],
        )

        # Normalize JS to 0-1 similarity (JS is bounded by ln2 ≈ 0.693)
        js_sim = max(0.0, 1.0 - js / math.log(2))

        return [
            SimilarityResult(
                metric="kl_divergence",
                score=max(0.0, 1.0 - min(kl, 10.0) / 10.0),
                detail={"raw_kl": kl},
            ),
            SimilarityResult(
                metric="js_divergence",
                score=js_sim,
                detail={"raw_js": js},
            ),
            SimilarityResult(
                metric=SimilarityMetric.COSINE,
                score=max(0.0, min(1.0, cos)),
                detail={"vector_dim": len(dist_a)},
            ),
        ]

    def batch_compare(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[List[SimilarityResult]]:
        """Compare multiple text pairs."""
        return [self.compare_texts(a, b) for a, b in pairs]


def format_similarity_report(results: List[SimilarityResult]) -> str:
    """Format similarity results as a readable table."""
    if not results:
        return "No similarity results."

    header = f"{'Metric':<20} {'Score':>8} {'Details'}"
    sep = "-" * 60
    lines = [header, sep]
    for r in results:
        detail_str = ", ".join(f"{k}={v}" for k, v in r.detail.items())
        lines.append(f"{r.metric:<20} {r.score:>8.4f} {detail_str}")
    return "\n".join(lines)
