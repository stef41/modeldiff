"""modeldiff — behavioral regression testing for LLMs."""

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
from modeldiff.capture import capture
from modeldiff.diff import diff_snapshots
from modeldiff.generator import (
    CaseResult,
    SuiteResult,
    TestCase,
    TestSuite,
    extract_key_phrases,
    generate_suite_from_snapshot,
    run_suite,
)
from modeldiff.html_report import format_html, save_html
from modeldiff.plugin import SnapshotHelper

__all__ = [
    "ChangeType",
    "DiffEntry",
    "DiffReport",
    "FingerprintResult",
    "ModeldiffError",
    "Prompt",
    "Response",
    "Severity",
    "Snapshot",
    "SnapshotHelper",
    "capture",
    "CaseResult",
    "SuiteResult",
    "TestCase",
    "TestSuite",
    "extract_key_phrases",
    "generate_suite_from_snapshot",
    "run_suite",
    "diff_snapshots",
    "format_html",
    "save_html",
]
