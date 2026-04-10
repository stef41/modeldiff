"""Tests for report module."""

import json
import tempfile
from pathlib import Path

import pytest
from modeldiff._types import (
    ChangeType,
    DiffEntry,
    DiffReport,
    Prompt,
    Response,
    Severity,
    Snapshot,
)
from modeldiff.report import (
    format_markdown,
    format_report_rich,
    format_report_text,
    load_json,
    report_to_dict,
    save_json,
)


def _make_report():
    entries = [
        DiffEntry(
            prompt=Prompt(text="hello"),
            output_a="a out",
            output_b="b out",
            change_type=ChangeType.CONTENT,
            severity=Severity.HIGH,
            description="content changed",
        ),
        DiffEntry(
            prompt=Prompt(text="hi"),
            output_a="same",
            output_b="same",
            change_type=ChangeType.IDENTICAL,
            severity=Severity.LOW,
            description="",
        ),
    ]
    return DiffReport(
        model_a="model-a",
        model_b="model-b",
        entries=entries,
    )


class TestReportToDict:
    def test_basic(self):
        report = _make_report()
        d = report_to_dict(report)
        assert d["model_a"] == "model-a"
        assert d["model_b"] == "model-b"
        assert len(d["entries"]) == 2
        assert d["n_changes"] == 1

    def test_roundtrip_json(self):
        report = _make_report()
        d = report_to_dict(report)
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["model_a"] == "model-a"


class TestSaveLoadJson:
    def test_roundtrip(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        save_json(report, str(path))
        loaded = load_json(str(path))
        assert loaded["model_a"] == "model-a"
        assert len(loaded["entries"]) == 2

    def test_path_object(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report2.json"
        save_json(report, path)
        assert path.exists()


class TestFormatReportText:
    def test_contains_header(self):
        report = _make_report()
        text = format_report_text(report)
        assert "model-a" in text
        assert "model-b" in text

    def test_contains_changes(self):
        report = _make_report()
        text = format_report_text(report)
        assert "CONTENT" in text or "content" in text.lower()


class TestFormatReportRich:
    def test_returns_string(self):
        report = _make_report()
        result = format_report_rich(report)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFormatMarkdown:
    def test_returns_markdown(self):
        report = _make_report()
        md = format_markdown(report)
        assert "model-a" in md
        assert "#" in md or "|" in md
