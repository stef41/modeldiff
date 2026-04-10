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
    "diff_snapshots",
    "format_html",
    "save_html",
]
