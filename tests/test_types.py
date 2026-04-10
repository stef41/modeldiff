"""Tests for _types module."""

import pytest

from modeldiff._types import (
    ChangeType,
    DiffEntry,
    DiffReport,
    FingerprintResult,
    ModeldiffError,
    Prompt,
    Response,
    Severity,
    Snapshot,
)


class TestPrompt:
    def test_basic(self):
        p = Prompt(text="hello")
        assert p.text == "hello"
        assert p.category == "general"
        assert p.tags == []
        assert p.expected is None

    def test_with_fields(self):
        p = Prompt(text="q", category="math", tags=["easy"], expected="42")
        assert p.category == "math"
        assert p.expected == "42"


class TestResponse:
    def test_basic(self):
        p = Prompt(text="q")
        r = Response(prompt=p, output="answer", model_name="m")
        assert r.output == "answer"
        assert not r.is_error
        assert not r.is_refusal
        assert r.word_count == 1

    def test_error(self):
        p = Prompt(text="q")
        r = Response(prompt=p, output="", model_name="m", error="timeout")
        assert r.is_error
        assert r.error == "timeout"

    def test_refusal(self):
        p = Prompt(text="q")
        r = Response(prompt=p, output="I can't help with that", model_name="m")
        assert r.is_refusal

    def test_not_refusal(self):
        p = Prompt(text="q")
        r = Response(prompt=p, output="Sure, here's the answer: 42", model_name="m")
        assert not r.is_refusal

    def test_word_count(self):
        p = Prompt(text="q")
        r = Response(prompt=p, output="this is five word sentence", model_name="m")
        assert r.word_count == 5


class TestSnapshot:
    def _make_snapshot(self, model_name="model_v1", n=3):
        responses = []
        for i in range(n):
            p = Prompt(text=f"prompt_{i}", category="test")
            responses.append(Response(prompt=p, output=f"output_{i}", model_name=model_name))
        return Snapshot(model_name=model_name, responses=responses)

    def test_basic(self):
        snap = self._make_snapshot()
        assert snap.n_responses == 3
        assert snap.n_errors == 0

    def test_with_error(self):
        snap = self._make_snapshot()
        snap.responses.append(
            Response(prompt=Prompt(text="err"), output="", model_name="m", error="fail")
        )
        assert snap.n_errors == 1

    def test_to_dict(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        assert d["model_name"] == "model_v1"
        assert len(d["responses"]) == 3

    def test_from_dict(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        snap2 = Snapshot.from_dict(d)
        assert snap2.model_name == snap.model_name
        assert snap2.n_responses == snap.n_responses

    def test_save_load(self, tmp_path):
        snap = self._make_snapshot()
        path = tmp_path / "snap.json"
        snap.save(path)
        loaded = Snapshot.load(path)
        assert loaded.model_name == "model_v1"
        assert loaded.n_responses == 3
        assert loaded.responses[0].output == "output_0"

    def test_roundtrip_preserves_data(self, tmp_path):
        p = Prompt(text="q", category="math", tags=["hard"], expected="42")
        r = Response(prompt=p, output="42", model_name="m", latency_ms=100.0, token_count=1)
        snap = Snapshot(model_name="test", responses=[r], metadata={"version": "1.0"})

        path = tmp_path / "snap.json"
        snap.save(path)
        loaded = Snapshot.load(path)

        assert loaded.metadata == {"version": "1.0"}
        assert loaded.responses[0].prompt.category == "math"
        assert loaded.responses[0].prompt.expected == "42"
        assert loaded.responses[0].latency_ms == 100.0


class TestDiffReport:
    def _make_entry(self, change_type=ChangeType.CONTENT, severity=Severity.MEDIUM):
        return DiffEntry(
            prompt=Prompt(text="q"),
            output_a="a", output_b="b",
            change_type=change_type, severity=severity,
            description="test",
        )

    def test_empty(self):
        report = DiffReport(model_a="a", model_b="b")
        assert report.n_changes == 0
        assert report.n_identical == 0
        assert report.change_rate == 0.0
        assert report.regression_score == 0.0

    def test_counts(self):
        entries = [
            self._make_entry(ChangeType.IDENTICAL, Severity.LOW),
            self._make_entry(ChangeType.CONTENT, Severity.HIGH),
            self._make_entry(ChangeType.FORMAT, Severity.LOW),
        ]
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.n_changes == 2
        assert report.n_identical == 1
        assert report.change_rate == pytest.approx(2 / 3)

    def test_by_type(self):
        entries = [
            self._make_entry(ChangeType.CONTENT),
            self._make_entry(ChangeType.CONTENT),
            self._make_entry(ChangeType.FORMAT),
        ]
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.by_type[ChangeType.CONTENT] == 2
        assert report.by_type[ChangeType.FORMAT] == 1

    def test_by_severity(self):
        entries = [
            self._make_entry(severity=Severity.LOW),
            self._make_entry(severity=Severity.HIGH),
            self._make_entry(severity=Severity.HIGH),
        ]
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.by_severity[Severity.HIGH] == 2

    def test_regression_score_none(self):
        entries = [self._make_entry(ChangeType.IDENTICAL, Severity.LOW)]
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.regression_score == 0.0

    def test_regression_score_high(self):
        entries = [self._make_entry(ChangeType.CONTENT, Severity.CRITICAL) for _ in range(5)]
        report = DiffReport(model_a="a", model_b="b", entries=entries)
        assert report.regression_score == 1.0


class TestFingerprintResult:
    def test_basic(self):
        fp = FingerprintResult(model_name="test", dimensions={"verbosity": 0.5})
        assert fp.model_name == "test"
        assert fp.dimensions["verbosity"] == 0.5


class TestChangeType:
    def test_values(self):
        assert ChangeType.CONTENT.value == "content"
        assert ChangeType.REFUSAL.value == "refusal"
        assert ChangeType.IDENTICAL.value == "identical"


class TestModeldiffError:
    def test_is_exception(self):
        assert issubclass(ModeldiffError, Exception)
