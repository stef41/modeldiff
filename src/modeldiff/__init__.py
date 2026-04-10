"""modeldiff — behavioral regression testing for LLMs."""

__version__ = "0.3.0"

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
from modeldiff.parquet import (
    Column,
    ParquetTable,
    format_table,
    merge_tables,
    snapshot_to_table,
)
from modeldiff.plugin import SnapshotHelper
from modeldiff.similarity import (
    OutputSimilarity,
    SimilarityMetric,
    SimilarityResult,
    cosine_similarity,
    format_similarity_report,
    jaccard_similarity,
    levenshtein_distance,
)

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
    "OutputSimilarity",
    "SimilarityMetric",
    "SimilarityResult",
    "cosine_similarity",
    "format_similarity_report",
    "jaccard_similarity",
    "levenshtein_distance",
    "Column",
    "ParquetTable",
    "format_table",
    "merge_tables",
    "snapshot_to_table",
]
