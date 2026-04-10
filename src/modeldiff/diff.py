"""Behavioral diff — compare two snapshots and categorize changes."""

from __future__ import annotations

import difflib
from typing import Dict, List

from modeldiff._types import (
    ChangeType,
    DiffEntry,
    DiffReport,
    Response,
    Severity,
    Snapshot,
)


def diff_snapshots(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    length_threshold: float = 0.5,
) -> DiffReport:
    """Compare two snapshots and produce a behavioral diff report.

    Args:
        snapshot_a: First (baseline) snapshot.
        snapshot_b: Second (updated) snapshot.
        length_threshold: Fractional length difference to flag as LENGTH change.
    """
    # Match responses by prompt text
    map_b: Dict[str, Response] = {r.prompt.text: r for r in snapshot_b.responses}

    entries: List[DiffEntry] = []
    for resp_a in snapshot_a.responses:
        resp_b = map_b.get(resp_a.prompt.text)
        if resp_b is None:
            entries.append(DiffEntry(
                prompt=resp_a.prompt,
                output_a=resp_a.output,
                output_b="[MISSING]",
                change_type=ChangeType.ERROR,
                severity=Severity.HIGH,
                description="Prompt missing from snapshot B",
            ))
            continue

        entry = _compare_responses(resp_a, resp_b, length_threshold)
        entries.append(entry)

    # Summary
    summary = _build_summary(entries)

    return DiffReport(
        model_a=snapshot_a.model_name,
        model_b=snapshot_b.model_name,
        entries=entries,
        summary=summary,
    )


def _compare_responses(
    a: Response,
    b: Response,
    length_threshold: float,
) -> DiffEntry:
    """Compare two responses and determine change type and severity."""
    metrics: Dict[str, float] = {}

    # Exact match
    if a.output.strip() == b.output.strip():
        return DiffEntry(
            prompt=a.prompt,
            output_a=a.output,
            output_b=b.output,
            change_type=ChangeType.IDENTICAL,
            severity=Severity.LOW,
            description="Identical outputs",
            metrics=metrics,
        )

    # Error check
    if a.is_error != b.is_error:
        return DiffEntry(
            prompt=a.prompt,
            output_a=a.output if not a.is_error else f"[ERROR: {a.error}]",
            output_b=b.output if not b.is_error else f"[ERROR: {b.error}]",
            change_type=ChangeType.ERROR,
            severity=Severity.CRITICAL,
            description="Error state changed",
            metrics=metrics,
        )

    # Refusal check
    if a.is_refusal != b.is_refusal:
        who = "B refuses" if b.is_refusal else "A refuses"
        return DiffEntry(
            prompt=a.prompt,
            output_a=a.output,
            output_b=b.output,
            change_type=ChangeType.REFUSAL,
            severity=Severity.HIGH,
            description=f"Refusal change: {who}",
            metrics=metrics,
        )

    # Compute similarity
    similarity = _text_similarity(a.output, b.output)
    metrics["similarity"] = similarity

    # Length comparison
    len_a = len(a.output)
    len_b = len(b.output)
    if len_a > 0:
        length_ratio = abs(len_a - len_b) / len_a
        metrics["length_ratio"] = length_ratio
    else:
        length_ratio = 0.0

    # Determine change type by severity
    if similarity > 0.95:
        # Very similar — probably formatting
        change_type = ChangeType.FORMAT
        severity = Severity.LOW
        description = f"Minor formatting change (similarity={similarity:.2%})"
    elif similarity > 0.7:
        # Moderate change — style difference
        change_type = ChangeType.STYLE
        severity = Severity.MEDIUM
        description = f"Style difference (similarity={similarity:.2%})"
    elif length_ratio > length_threshold:
        # Big length change
        change_type = ChangeType.LENGTH
        severity = Severity.MEDIUM
        description = f"Significant length change ({len_a} → {len_b} chars)"
    else:
        # Content change
        change_type = ChangeType.CONTENT
        severity = Severity.HIGH
        description = f"Content difference (similarity={similarity:.2%})"

    return DiffEntry(
        prompt=a.prompt,
        output_a=a.output,
        output_b=b.output,
        change_type=change_type,
        severity=severity,
        description=description,
        metrics=metrics,
    )


def _text_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two texts."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _build_summary(entries: List[DiffEntry]) -> Dict:
    total = len(entries)
    if total == 0:
        return {"total": 0, "change_rate": 0.0}

    n_changed = sum(1 for e in entries if e.change_type != ChangeType.IDENTICAL)
    similarities = [e.metrics.get("similarity", 1.0) for e in entries]

    return {
        "total": total,
        "n_changed": n_changed,
        "n_identical": total - n_changed,
        "change_rate": n_changed / total,
        "avg_similarity": sum(similarities) / len(similarities),
    }


def diff_text(output_a: str, output_b: str) -> str:
    """Generate a unified diff between two outputs."""
    lines_a = output_a.splitlines(keepends=True)
    lines_b = output_b.splitlines(keepends=True)
    diff = difflib.unified_diff(lines_a, lines_b, fromfile="model_a", tofile="model_b")
    return "".join(diff)
