"""Tests for fingerprint module."""

import pytest
from modeldiff._types import Prompt, Response, Snapshot
from modeldiff.fingerprint import compare_fingerprints, fingerprint


def _make_snap(model, outputs, latencies=None):
    responses = []
    for i, output in enumerate(outputs):
        p = Prompt(text=f"prompt_{i}")
        lat = latencies[i] if latencies else 100.0
        responses.append(Response(prompt=p, output=output, model_name=model, latency_ms=lat))
    return Snapshot(model_name=model, responses=responses)


class TestFingerprint:
    def test_basic(self):
        snap = _make_snap("m", ["hello world this is a response"] * 5, [100.0] * 5)
        result = fingerprint(snap)
        assert result.model_name == "m"
        assert "verbosity" in result.dimensions
        assert "refusal_rate" in result.dimensions
        assert "error_rate" in result.dimensions
        assert "vocabulary_richness" in result.dimensions
        assert "avg_latency_ms" in result.dimensions
        assert "length_consistency" in result.dimensions

    def test_no_refusals(self):
        snap = _make_snap("m", ["fine answer"] * 5)
        result = fingerprint(snap)
        assert result.dimensions["refusal_rate"] == 0.0
        assert result.dimensions["error_rate"] == 0.0

    def test_all_refusals(self):
        snap = _make_snap("m", ["I can't help with that"] * 5)
        result = fingerprint(snap)
        assert result.dimensions["refusal_rate"] == 1.0

    def test_verbosity(self):
        snap_short = _make_snap("m", ["hi"] * 5)
        snap_long = _make_snap("m", ["this is a very long response " * 50] * 5)
        fp_short = fingerprint(snap_short)
        fp_long = fingerprint(snap_long)
        assert fp_long.dimensions["verbosity"] > fp_short.dimensions["verbosity"]

    def test_latency(self):
        snap = _make_snap("m", ["text"] * 5, [200.0] * 5)
        result = fingerprint(snap)
        assert result.dimensions["avg_latency_ms"] == pytest.approx(200.0)

    def test_empty_snapshot(self):
        snap = Snapshot(model_name="m")
        result = fingerprint(snap)
        assert result.dimensions["verbosity"] == 0.0

    def test_vocabulary_richness(self):
        snap_low = _make_snap("m", ["the the the the"] * 5)
        snap_high = _make_snap("m", ["apple banana cherry date elderberry"] * 5)
        fp_low = fingerprint(snap_low)
        fp_high = fingerprint(snap_high)
        assert fp_high.dimensions["vocabulary_richness"] >= fp_low.dimensions["vocabulary_richness"]


class TestCompareFingerprints:
    def test_identical(self):
        snap = _make_snap("m", ["hello world"] * 5)
        fp = fingerprint(snap)
        result = compare_fingerprints(fp, fp)
        assert result["euclidean_distance"] == pytest.approx(0.0)
        assert result["similar"]

    def test_different(self):
        snap_a = _make_snap("a", ["short"] * 5, [50.0] * 5)
        snap_b = _make_snap("b", ["I can't help with that very long refusal message"] * 5, [500.0] * 5)
        fp_a = fingerprint(snap_a)
        fp_b = fingerprint(snap_b)
        result = compare_fingerprints(fp_a, fp_b)
        assert result["euclidean_distance"] > 0.0

    def test_dimension_deltas(self):
        snap_a = _make_snap("a", ["hello"] * 5)
        snap_b = _make_snap("b", ["world"] * 5)
        fp_a = fingerprint(snap_a)
        fp_b = fingerprint(snap_b)
        result = compare_fingerprints(fp_a, fp_b)
        assert "deltas" in result
