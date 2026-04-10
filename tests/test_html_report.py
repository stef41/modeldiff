"""Tests for modeldiff.html_report."""

from __future__ import annotations

from pathlib import Path

from modeldiff._types import (
    ChangeType,
    DiffEntry,
    DiffReport,
    Prompt,
    Severity,
)
from modeldiff.html_report import _escape_html, format_html, save_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    text: str = "What is 2+2?",
    output_a: str = "4",
    output_b: str = "4",
    change_type: ChangeType = ChangeType.IDENTICAL,
    severity: Severity = Severity.LOW,
    description: str = "identical",
) -> DiffEntry:
    return DiffEntry(
        prompt=Prompt(text=text),
        output_a=output_a,
        output_b=output_b,
        change_type=change_type,
        severity=severity,
        description=description,
    )


def _report(entries: list[DiffEntry] | None = None) -> DiffReport:
    return DiffReport(
        model_a="model-a",
        model_b="model-b",
        entries=entries or [],
    )


# ---------------------------------------------------------------------------
# _escape_html
# ---------------------------------------------------------------------------

class TestEscapeHtml:
    def test_ampersand(self) -> None:
        assert _escape_html("a & b") == "a &amp; b"

    def test_angle_brackets(self) -> None:
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_quotes(self) -> None:
        assert _escape_html('a "b" c') == "a &quot;b&quot; c"

    def test_single_quote(self) -> None:
        assert "&#x27;" in _escape_html("it's") or "'" not in _escape_html("it's").replace("&#x27;", "")

    def test_passthrough(self) -> None:
        assert _escape_html("hello world") == "hello world"

    def test_empty(self) -> None:
        assert _escape_html("") == ""


# ---------------------------------------------------------------------------
# format_html
# ---------------------------------------------------------------------------

class TestFormatHtml:
    def test_returns_string(self) -> None:
        html = format_html(_report())
        assert isinstance(html, str)

    def test_contains_doctype(self) -> None:
        html = format_html(_report())
        assert html.startswith("<!DOCTYPE html>")

    def test_model_names_in_output(self) -> None:
        html = format_html(_report())
        assert "model-a" in html
        assert "model-b" in html

    def test_summary_stats_empty(self) -> None:
        html = format_html(_report())
        # 0 total prompts
        assert ">0<" in html

    def test_summary_stats_counts(self) -> None:
        entries = [
            _entry(),
            _entry(
                text="Explain gravity",
                output_b="mass attracts",
                change_type=ChangeType.CONTENT,
                severity=Severity.MEDIUM,
                description="different content",
            ),
            _entry(
                text="buggy prompt",
                output_b="[error]",
                change_type=ChangeType.ERROR,
                severity=Severity.HIGH,
                description="error response",
            ),
        ]
        html = format_html(_report(entries))
        # Total = 3, identical = 1, different = 2, errors = 1
        assert ">3<" in html  # total
        assert ">1<" in html  # identical and errors

    def test_row_classes(self) -> None:
        entries = [
            _entry(),
            _entry(text="diff", change_type=ChangeType.CONTENT, severity=Severity.MEDIUM, description="d"),
            _entry(text="err", change_type=ChangeType.ERROR, severity=Severity.HIGH, description="e"),
        ]
        html = format_html(_report(entries))
        assert 'class="identical"' in html
        assert 'class="different"' in html
        assert 'class="error"' in html

    def test_side_by_side_outputs(self) -> None:
        entries = [
            _entry(output_a="answer A", output_b="answer B", change_type=ChangeType.CONTENT,
                   severity=Severity.MEDIUM, description="differs"),
        ]
        html = format_html(_report(entries))
        assert "answer A" in html
        assert "answer B" in html

    def test_xss_escaping_in_prompt(self) -> None:
        entries = [_entry(text='<img src=x onerror="alert(1)">')]
        html = format_html(_report(entries))
        assert "<img" not in html
        assert "&lt;img" in html

    def test_xss_escaping_in_output(self) -> None:
        entries = [
            _entry(output_a='<script>alert("xss")</script>', output_b="safe"),
        ]
        html = format_html(_report(entries))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_embedded_css(self) -> None:
        html = format_html(_report())
        assert "<style>" in html
        assert "font-family" in html

    def test_no_external_deps(self) -> None:
        html = format_html(_report())
        assert "link rel=" not in html.lower()
        assert '<script src=' not in html.lower()

    def test_change_rate_displayed(self) -> None:
        entries = [
            _entry(),
            _entry(text="q2", change_type=ChangeType.CONTENT, severity=Severity.LOW, description="d"),
        ]
        html = format_html(_report(entries))
        assert "50%" in html

    def test_regression_score_displayed(self) -> None:
        html = format_html(_report())
        assert "0.00" in html


# ---------------------------------------------------------------------------
# save_html
# ---------------------------------------------------------------------------

class TestSaveHtml:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "report.html"
        save_html(_report(), p)
        assert p.exists()

    def test_file_content_matches_format(self, tmp_path: Path) -> None:
        report = _report([_entry()])
        p = tmp_path / "report.html"
        save_html(report, p)
        assert p.read_text(encoding="utf-8") == format_html(report)

    def test_string_path(self, tmp_path: Path) -> None:
        p = tmp_path / "out.html"
        save_html(_report(), str(p))
        assert p.exists()

    def test_subdirectory_creation(self, tmp_path: Path) -> None:
        """save_html should work when the parent directory already exists."""
        sub = tmp_path / "sub"
        sub.mkdir()
        p = sub / "r.html"
        save_html(_report(), p)
        assert p.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# ---------------------------------------------------------------------------
# Round-trip: public API re-exports
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_importable_from_package(self) -> None:
        from modeldiff import format_html as fh
        from modeldiff import save_html as sh

        assert callable(fh)
        assert callable(sh)
