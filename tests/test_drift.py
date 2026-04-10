"""Tests for drift module."""

from modeldiff._types import Prompt, Response, Snapshot
from modeldiff.drift import (
    full_drift_report,
    latency_drift,
    length_drift,
    refusal_drift,
    vocabulary_drift,
)


def _make_snap(model, outputs, latencies=None):
    responses = []
    for i, output in enumerate(outputs):
        p = Prompt(text=f"prompt_{i}")
        lat = latencies[i] if latencies else 100.0
        responses.append(Response(prompt=p, output=output, model_name=model, latency_ms=lat))
    return Snapshot(model_name=model, responses=responses)


class TestLengthDrift:
    def test_no_drift(self):
        snap_a = _make_snap("a", ["one two three"] * 5)
        snap_b = _make_snap("b", ["one two three"] * 5)
        result = length_drift(snap_a, snap_b)
        assert result["drift_sigma"] == 0.0
        assert not result["drift_significant"]

    def test_significant_drift(self):
        snap_a = _make_snap("a", ["short"] * 10)
        snap_b = _make_snap("b", ["this is a much longer response with many words"] * 10)
        result = length_drift(snap_a, snap_b)
        assert result["mean_b"] > result["mean_a"]

    def test_empty_snapshots(self):
        snap_a = Snapshot(model_name="a")
        snap_b = Snapshot(model_name="b")
        result = length_drift(snap_a, snap_b)
        assert result["drift"] == 0.0


class TestRefusalDrift:
    def test_no_drift(self):
        snap_a = _make_snap("a", ["answer"] * 5)
        snap_b = _make_snap("b", ["answer"] * 5)
        result = refusal_drift(snap_a, snap_b)
        assert result["delta"] == 0.0
        assert not result["drift_significant"]

    def test_increased_refusal(self):
        snap_a = _make_snap("a", ["answer"] * 5)
        snap_b = _make_snap("b", ["I can't help with that"] * 3 + ["answer"] * 2)
        result = refusal_drift(snap_a, snap_b)
        assert result["refusal_rate_b"] > result["refusal_rate_a"]
        assert result["delta"] > 0

    def test_decreased_refusal(self):
        snap_a = _make_snap("a", ["I can't help"] * 3 + ["ok"] * 2)
        snap_b = _make_snap("b", ["answer"] * 5)
        result = refusal_drift(snap_a, snap_b)
        assert result["delta"] < 0


class TestLatencyDrift:
    def test_no_drift(self):
        snap_a = _make_snap("a", ["a"] * 5, [100.0] * 5)
        snap_b = _make_snap("b", ["a"] * 5, [100.0] * 5)
        result = latency_drift(snap_a, snap_b)
        assert result["drift_sigma"] == 0.0

    def test_significant_drift(self):
        snap_a = _make_snap("a", ["a"] * 10, [100.0] * 10)
        snap_b = _make_snap("b", ["a"] * 10, [500.0] * 10)
        result = latency_drift(snap_a, snap_b)
        assert result["mean_b_ms"] > result["mean_a_ms"]

    def test_empty(self):
        snap_a = Snapshot(model_name="a")
        snap_b = Snapshot(model_name="b")
        result = latency_drift(snap_a, snap_b)
        assert result["drift"] == 0.0


class TestVocabularyDrift:
    def test_same_vocab(self):
        snap_a = _make_snap("a", ["hello world foo bar"] * 5)
        snap_b = _make_snap("b", ["hello world foo bar"] * 5)
        result = vocabulary_drift(snap_a, snap_b)
        assert result["jaccard_similarity"] == 1.0
        assert not result["drift_significant"]

    def test_different_vocab(self):
        snap_a = _make_snap("a", ["hello world foo bar"] * 5)
        snap_b = _make_snap("b", ["python rust golang swift"] * 5)
        result = vocabulary_drift(snap_a, snap_b)
        assert result["jaccard_similarity"] < 1.0
        assert len(result["new_words"]) > 0
        assert len(result["lost_words"]) > 0

    def test_empty(self):
        snap_a = Snapshot(model_name="a")
        snap_b = Snapshot(model_name="b")
        result = vocabulary_drift(snap_a, snap_b)
        assert result["jaccard_similarity"] == 1.0


class TestFullDriftReport:
    def test_basic(self):
        snap_a = _make_snap("a", ["hello world"] * 5)
        snap_b = _make_snap("b", ["hello world"] * 5)
        report = full_drift_report(snap_a, snap_b)
        assert "length" in report
        assert "refusal" in report
        assert "latency" in report
        assert "vocabulary" in report
