"""Tests for output similarity scoring."""


import pytest

from modeldiff.similarity import (
    OutputSimilarity,
    SimilarityMetric,
    SimilarityResult,
    cosine_similarity,
    format_similarity_report,
    jaccard_similarity,
    levenshtein_distance,
)

# --- cosine_similarity ------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1, 2], [1, 2, 3])


# --- jaccard_similarity -----------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert jaccard_similarity({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)

    def test_empty_sets(self):
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert jaccard_similarity({1}, set()) == 0.0


# --- levenshtein_distance ---------------------------------------------------

class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1

    def test_insertion_deletion(self):
        assert levenshtein_distance("kitten", "sitting") == 3


# --- SimilarityMetric constants ----------------------------------------------

class TestSimilarityMetric:
    def test_constants(self):
        assert SimilarityMetric.COSINE == "cosine"
        assert SimilarityMetric.JACCARD == "jaccard"
        assert SimilarityMetric.LEVENSHTEIN == "levenshtein"
        assert SimilarityMetric.BLEU == "bleu"
        assert SimilarityMetric.EXACT_MATCH == "exact_match"


# --- OutputSimilarity.compare_texts -----------------------------------------

class TestCompareTexts:
    def test_identical_texts(self):
        sim = OutputSimilarity()
        results = sim.compare_texts("hello world", "hello world")
        scores = {r.metric: r.score for r in results}
        assert scores[SimilarityMetric.EXACT_MATCH] == 1.0
        assert scores[SimilarityMetric.COSINE] == pytest.approx(1.0)
        assert scores[SimilarityMetric.JACCARD] == 1.0

    def test_completely_different(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.EXACT_MATCH])
        results = sim.compare_texts("hello", "world")
        assert results[0].score == 0.0

    def test_levenshtein_metric(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.LEVENSHTEIN])
        results = sim.compare_texts("cat", "bat")
        assert len(results) == 1
        assert results[0].score > 0.5  # 1 edit out of 3 chars

    def test_bleu_metric(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.BLEU])
        results = sim.compare_texts(
            "the cat sat on the mat",
            "the cat sat on the mat",
        )
        assert results[0].score == pytest.approx(1.0)

    def test_bleu_partial(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.BLEU])
        results = sim.compare_texts(
            "the cat sat on the mat",
            "the dog stood on the floor",
        )
        assert 0.0 < results[0].score < 1.0

    def test_all_metrics(self):
        all_m = [
            SimilarityMetric.COSINE,
            SimilarityMetric.JACCARD,
            SimilarityMetric.LEVENSHTEIN,
            SimilarityMetric.BLEU,
            SimilarityMetric.EXACT_MATCH,
        ]
        sim = OutputSimilarity(metrics=all_m)
        results = sim.compare_texts("hello world", "hello world")
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r.score <= 1.0


# --- OutputSimilarity.compare_token_lists -----------------------------------

class TestCompareTokenLists:
    def test_identical_tokens(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.EXACT_MATCH])
        results = sim.compare_token_lists([1, 2, 3], [1, 2, 3])
        assert results[0].score == 1.0

    def test_different_tokens(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.JACCARD])
        results = sim.compare_token_lists([1, 2, 3], [4, 5, 6])
        assert results[0].score == 0.0

    def test_partial_overlap_tokens(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.JACCARD])
        results = sim.compare_token_lists([1, 2, 3], [2, 3, 4])
        assert results[0].score == pytest.approx(0.5)

    def test_cosine_tokens(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.COSINE])
        results = sim.compare_token_lists([1, 1, 2], [1, 1, 2])
        assert results[0].score == pytest.approx(1.0)


# --- OutputSimilarity.compare_distributions ---------------------------------

class TestCompareDistributions:
    def test_identical_distributions(self):
        sim = OutputSimilarity()
        dist = [0.25, 0.25, 0.25, 0.25]
        results = sim.compare_distributions(dist, dist)
        # JS divergence of identical dist → similarity ≈ 1.0
        js_result = [r for r in results if r.metric == "js_divergence"][0]
        assert js_result.score == pytest.approx(1.0, abs=0.01)

    def test_different_distributions(self):
        sim = OutputSimilarity()
        dist_a = [0.9, 0.05, 0.025, 0.025]
        dist_b = [0.025, 0.025, 0.05, 0.9]
        results = sim.compare_distributions(dist_a, dist_b)
        cos_result = [r for r in results if r.metric == SimilarityMetric.COSINE][0]
        assert cos_result.score < 0.5

    def test_mismatched_lengths_raises(self):
        sim = OutputSimilarity()
        with pytest.raises(ValueError, match="same length"):
            sim.compare_distributions([0.5, 0.5], [0.33, 0.33, 0.34])

    def test_returns_three_results(self):
        sim = OutputSimilarity()
        results = sim.compare_distributions([0.5, 0.5], [0.5, 0.5])
        assert len(results) == 3
        metrics = {r.metric for r in results}
        assert "kl_divergence" in metrics
        assert "js_divergence" in metrics
        assert SimilarityMetric.COSINE in metrics


# --- OutputSimilarity.batch_compare -----------------------------------------

class TestBatchCompare:
    def test_batch(self):
        sim = OutputSimilarity(metrics=[SimilarityMetric.EXACT_MATCH])
        pairs = [("a", "a"), ("a", "b"), ("x", "x")]
        results = sim.batch_compare(pairs)
        assert len(results) == 3
        assert results[0][0].score == 1.0
        assert results[1][0].score == 0.0
        assert results[2][0].score == 1.0


# --- format_similarity_report -----------------------------------------------

class TestFormatReport:
    def test_empty(self):
        assert format_similarity_report([]) == "No similarity results."

    def test_has_header(self):
        r = SimilarityResult(metric="cosine", score=0.95, detail={"dim": 10})
        report = format_similarity_report([r])
        assert "Metric" in report
        assert "Score" in report
        assert "cosine" in report

    def test_multiple_rows(self):
        results = [
            SimilarityResult(metric="cosine", score=0.95, detail={}),
            SimilarityResult(metric="jaccard", score=0.80, detail={}),
        ]
        report = format_similarity_report(results)
        lines = report.strip().split("\n")
        assert len(lines) == 4  # header + sep + 2 rows
