"""Edge-case and hardened tests for modeldiff modules."""

import json
import math

import pytest
from modeldiff._types import (
    ChangeType,
    DiffEntry,
    DiffReport,
    FingerprintResult,
    Prompt,
    Response,
    Severity,
    Snapshot,
)
from modeldiff.capture import capture, capture_from_file
from modeldiff.diff import diff_snapshots, diff_text, _text_similarity
from modeldiff.drift import (
    full_drift_report,
    latency_drift,
    length_drift,
    refusal_drift,
    vocabulary_drift,
)
from modeldiff.fingerprint import compare_fingerprints, fingerprint
from modeldiff.report import (
    format_markdown,
    format_report_rich,
    format_report_text,
    report_to_dict,
    save_json,
    load_json,
)


def _snap(model, outputs, latencies=None, errors=None):
    responses = []
    for i, output in enumerate(outputs):
        p = Prompt(text=f"prompt_{i}")
        lat = latencies[i] if latencies else 100.0
        err = errors[i] if errors else None
        responses.append(Response(
            prompt=p, output=output, model_name=model,
            latency_ms=lat, error=err,
        ))
    return Snapshot(model_name=model, responses=responses)


# ---- diff edge cases ----


class TestDiffEdgeCases:
    def test_both_empty_outputs(self):
        snap_a = _snap("a", [""])
        snap_b = _snap("b", [""])
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_identical == 1

    def test_both_errors(self):
        snap_a = _snap("a", [""], errors=["err1"])
        snap_b = _snap("b", [""], errors=["err2"])
        report = diff_snapshots(snap_a, snap_b)
        # Both errored -- not an error-state-change
        assert all(e.change_type != ChangeType.ERROR for e in report.entries)

    def test_whitespace_difference_is_format(self):
        snap_a = _snap("a", ["Hello world."])
        snap_b = _snap("b", ["Hello world. "])  # trailing space
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_identical == 1  # strip makes them identical

    def test_custom_length_threshold(self):
        snap_a = _snap("a", ["short"])
        snap_b = _snap("b", ["short " * 20])
        report = diff_snapshots(snap_a, snap_b, length_threshold=0.1)
        assert report.n_changes == 1

    def test_large_batch(self):
        n = 50
        snap_a = _snap("a", [f"answer {i}" for i in range(n)])
        snap_b = _snap("b", [f"answer {i}" for i in range(n)])
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_identical == n

    def test_diff_text_multiline(self):
        result = diff_text("line1\nline2\nline3\n", "line1\nchanged\nline3\n")
        assert "-line2" in result or "- line2" in result or "-changed" in result or "changed" in result

    def test_text_similarity_identical(self):
        assert _text_similarity("hello", "hello") == 1.0

    def test_text_similarity_empty(self):
        assert _text_similarity("", "") == 1.0

    def test_regression_score_zero(self):
        report = DiffReport(model_a="a", model_b="b")
        assert report.regression_score == 0.0

    def test_by_type_empty(self):
        report = DiffReport(model_a="a", model_b="b")
        assert report.by_type == {}

    def test_by_severity_empty(self):
        report = DiffReport(model_a="a", model_b="b")
        assert report.by_severity == {}


# ---- drift edge cases ----


class TestDriftEdgeCases:
    def test_length_drift_single_response(self):
        snap_a = _snap("a", ["one"])
        snap_b = _snap("b", ["one two three four five six seven eight nine ten"])
        result = length_drift(snap_a, snap_b)
        assert result["mean_b"] > result["mean_a"]
        # Single-element std=0, drift should be 0
        assert result["std_a"] == 0.0

    def test_latency_drift_zero_latencies(self):
        snap_a = _snap("a", ["a"] * 5, [0.0] * 5)
        snap_b = _snap("b", ["a"] * 5, [0.0] * 5)
        result = latency_drift(snap_a, snap_b)
        # latency filter skips latency 0
        assert result["drift"] == 0.0

    def test_vocabulary_drift_empty_outputs(self):
        snap_a = _snap("a", [""])
        snap_b = _snap("b", [""])
        result = vocabulary_drift(snap_a, snap_b)
        # Empty strings produce empty freq dicts -> jaccard 1.0 (vacuously)
        assert result["jaccard_similarity"] == 1.0

    def test_refusal_drift_all_refusals(self):
        snap_a = _snap("a", ["I can't help"] * 5)
        snap_b = _snap("b", ["I can't help"] * 5)
        result = refusal_drift(snap_a, snap_b)
        assert result["delta"] == 0.0

    def test_full_drift_report_all_keys(self):
        snap_a = _snap("a", ["hello world"] * 3)
        snap_b = _snap("b", ["goodbye earth"] * 3)
        result = full_drift_report(snap_a, snap_b)
        assert set(result.keys()) == {"length", "refusal", "latency", "vocabulary"}


# ---- fingerprint edge cases ----


class TestFingerprintEdgeCases:
    def test_single_response(self):
        snap = _snap("m", ["just one response"])
        result = fingerprint(snap)
        assert result.dimensions["length_consistency"] == 1.0

    def test_all_errors(self):
        snap = _snap("m", ["", ""], errors=["e1", "e2"])
        result = fingerprint(snap)
        assert result.dimensions["error_rate"] == 1.0
        assert result.dimensions["verbosity"] == 0.0

    def test_formality_casual(self):
        snap = _snap("m", ["don't can't won't I'm gonna lol!!"] * 5)
        result = fingerprint(snap)
        assert result.dimensions["formality"] < 0.5

    def test_formality_formal(self):
        snap = _snap("m", ["Therefore furthermore however additionally consequently."] * 5)
        result = fingerprint(snap)
        assert result.dimensions["formality"] > 0.5

    def test_compare_empty_dimensions(self):
        fp_a = FingerprintResult(model_name="a", dimensions={})
        fp_b = FingerprintResult(model_name="b", dimensions={})
        result = compare_fingerprints(fp_a, fp_b)
        assert result["euclidean_distance"] == 0.0
        assert result["similar"]

    def test_compare_partial_overlap(self):
        fp_a = FingerprintResult(model_name="a", dimensions={"x": 1.0})
        fp_b = FingerprintResult(model_name="b", dimensions={"y": 1.0})
        result = compare_fingerprints(fp_a, fp_b)
        # x=1,y=0 vs x=0,y=1 -> distance = sqrt(2)
        assert result["euclidean_distance"] == pytest.approx(math.sqrt(2), abs=0.01)


# ---- capture edge cases ----


class TestCaptureEdgeCases:
    def test_model_fn_returns_unicode(self):
        snap = capture(
            [Prompt(text="q")],
            lambda t: "日本語の応答",
            model_name="unicode_model",
        )
        assert snap.responses[0].output == "日本語の応答"

    def test_prompt_with_expected(self):
        p = Prompt(text="2+2?", expected="4")
        snap = capture([p], lambda t: "4", model_name="m")
        assert snap.responses[0].prompt.expected == "4"

    def test_single_json_object(self, tmp_path):
        path = tmp_path / "single.json"
        path.write_text(json.dumps({"text": "hello"}))
        snap = capture_from_file(str(path), lambda t: t.upper(), model_name="m")
        assert snap.n_responses == 1
        assert snap.responses[0].output == "HELLO"


# ---- report edge cases ----


class TestReportEdgeCases:
    def test_empty_report_text(self):
        report = DiffReport(model_a="a", model_b="b")
        text = format_report_text(report)
        assert "a" in text and "b" in text
        assert "0" in text  # 0 changes

    def test_empty_report_markdown(self):
        report = DiffReport(model_a="a", model_b="b")
        md = format_markdown(report)
        assert "a" in md

    def test_empty_report_rich(self):
        report = DiffReport(model_a="a", model_b="b")
        result = format_report_rich(report)
        assert isinstance(result, str)

    def test_save_load_empty(self, tmp_path):
        report = DiffReport(model_a="a", model_b="b")
        path = tmp_path / "empty.json"
        save_json(report, str(path))
        loaded = load_json(str(path))
        assert loaded["n_changes"] == 0

    def test_report_unicode_prompts(self):
        report = DiffReport(
            model_a="a", model_b="b",
            entries=[DiffEntry(
                prompt=Prompt(text="日本語のプロンプト"),
                output_a="応答A", output_b="応答B",
                change_type=ChangeType.CONTENT, severity=Severity.HIGH,
                description="content changed",
            )],
        )
        d = report_to_dict(report)
        s = json.dumps(d, ensure_ascii=False)
        assert "日本語" in s


# ---- types edge cases ----


class TestTypesEdgeCases:
    def test_response_word_count(self):
        r = Response(prompt=Prompt(text="q"), output="one two three", model_name="m")
        assert r.word_count == 3

    def test_response_word_count_empty(self):
        r = Response(prompt=Prompt(text="q"), output="", model_name="m")
        assert r.word_count == 0

    def test_response_is_refusal_false(self):
        r = Response(prompt=Prompt(text="q"), output="Sure, here you go!", model_name="m")
        assert not r.is_refusal

    def test_snapshot_roundtrip(self, tmp_path):
        snap = _snap("m", ["hello", "world"])
        path = tmp_path / "snap.json"
        snap.save(str(path))
        loaded = Snapshot.load(str(path))
        assert loaded.model_name == "m"
        assert loaded.n_responses == 2
        assert loaded.responses[0].output == "hello"

    def test_snapshot_empty_roundtrip(self, tmp_path):
        snap = Snapshot(model_name="empty")
        path = tmp_path / "empty.json"
        snap.save(str(path))
        loaded = Snapshot.load(str(path))
        assert loaded.model_name == "empty"
        assert loaded.n_responses == 0

    def test_diff_report_regression_score_high(self):
        entries = [
            DiffEntry(
                prompt=Prompt(text="q"), output_a="a", output_b="b",
                change_type=ChangeType.CONTENT, severity=Severity.CRITICAL,
                description="crit",
            )
        ] * 5
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.regression_score == 1.0

    def test_change_rate_all_changed(self):
        entries = [
            DiffEntry(
                prompt=Prompt(text="q"), output_a="a", output_b="b",
                change_type=ChangeType.CONTENT, severity=Severity.MEDIUM,
                description="d",
            )
        ] * 3
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.change_rate == 1.0
