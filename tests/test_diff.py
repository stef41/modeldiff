"""Tests for diff module."""

from modeldiff._types import ChangeType, Prompt, Response, Severity, Snapshot
from modeldiff.diff import diff_snapshots, diff_text


def _make_snapshot(model, outputs):
    """Helper: create snapshot from a dict of prompt→output."""
    responses = []
    for prompt_text, output in outputs.items():
        p = Prompt(text=prompt_text)
        if isinstance(output, dict):
            responses.append(Response(
                prompt=p, output=output.get("text", ""),
                model_name=model, error=output.get("error"),
            ))
        else:
            responses.append(Response(prompt=p, output=output, model_name=model))
    return Snapshot(model_name=model, responses=responses)


class TestDiffSnapshots:
    def test_identical(self):
        snap_a = _make_snapshot("v1", {"q1": "answer1", "q2": "answer2"})
        snap_b = _make_snapshot("v2", {"q1": "answer1", "q2": "answer2"})
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_changes == 0
        assert report.n_identical == 2
        assert report.change_rate == 0.0

    def test_content_change(self):
        snap_a = _make_snapshot("v1", {"q": "The answer is 42"})
        snap_b = _make_snapshot("v2", {"q": "The answer is 99"})
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_changes == 1
        # Should detect content change (high similarity but different)

    def test_refusal_change(self):
        snap_a = _make_snapshot("v1", {"q": "Here's how to do it"})
        snap_b = _make_snapshot("v2", {"q": "I can't help with that"})
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_changes == 1
        refusals = [e for e in report.entries if e.change_type == ChangeType.REFUSAL]
        assert len(refusals) == 1

    def test_error_change(self):
        snap_a = _make_snapshot("v1", {"q": "output"})
        snap_b = _make_snapshot("v2", {"q": {"text": "", "error": "timeout"}})
        report = diff_snapshots(snap_a, snap_b)
        errors = [e for e in report.entries if e.change_type == ChangeType.ERROR]
        assert len(errors) == 1
        assert errors[0].severity == Severity.CRITICAL

    def test_missing_prompt(self):
        snap_a = _make_snapshot("v1", {"q1": "a", "q2": "b"})
        snap_b = _make_snapshot("v2", {"q1": "a"})
        report = diff_snapshots(snap_a, snap_b)
        missing = [e for e in report.entries if e.output_b == "[MISSING]"]
        assert len(missing) == 1

    def test_format_change(self):
        snap_a = _make_snapshot("v1", {"q": "The capital is Paris."})
        snap_b = _make_snapshot("v2", {"q": "The capital is Paris"})  # no period
        report = diff_snapshots(snap_a, snap_b)
        # Very similar → format change
        assert report.n_changes == 1
        assert report.entries[0].change_type == ChangeType.FORMAT

    def test_length_change(self):
        short = "Yes."
        long = "Yes, and here is a very detailed and extensive explanation " * 10
        snap_a = _make_snapshot("v1", {"q": short})
        snap_b = _make_snapshot("v2", {"q": long})
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_changes == 1

    def test_multiple_prompts(self):
        snap_a = _make_snapshot("v1", {
            "q1": "answer",
            "q2": "The answer is 42",
            "q3": "Same output",
        })
        snap_b = _make_snapshot("v2", {
            "q1": "answer",
            "q2": "I can't help with that",
            "q3": "Same output",
        })
        report = diff_snapshots(snap_a, snap_b)
        assert report.n_identical == 2
        assert report.n_changes == 1

    def test_summary(self):
        snap_a = _make_snapshot("v1", {"q": "a"})
        snap_b = _make_snapshot("v2", {"q": "b"})
        report = diff_snapshots(snap_a, snap_b)
        assert "total" in report.summary
        assert "change_rate" in report.summary

    def test_regression_score(self):
        snap_a = _make_snapshot("v1", {"q": "output"})
        snap_b = _make_snapshot("v2", {"q": "I can't help"})
        report = diff_snapshots(snap_a, snap_b)
        assert report.regression_score > 0

    def test_model_names(self):
        snap_a = _make_snapshot("gpt-4", {"q": "a"})
        snap_b = _make_snapshot("gpt-4o", {"q": "a"})
        report = diff_snapshots(snap_a, snap_b)
        assert report.model_a == "gpt-4"
        assert report.model_b == "gpt-4o"

    def test_empty_snapshots(self):
        snap_a = Snapshot(model_name="a")
        snap_b = Snapshot(model_name="b")
        report = diff_snapshots(snap_a, snap_b)
        assert len(report.entries) == 0


class TestDiffText:
    def test_identical(self):
        result = diff_text("hello", "hello")
        assert result == ""

    def test_different(self):
        result = diff_text("hello\nworld", "hello\nearth")
        assert "---" in result
        assert "+++" in result

    def test_empty(self):
        result = diff_text("", "")
        assert result == ""
