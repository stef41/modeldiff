"""Pytest plugin for model behavioral regression testing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import pytest

from modeldiff._types import ChangeType, DiffReport, Prompt, Snapshot
from modeldiff.capture import capture
from modeldiff.diff import diff_snapshots


class SnapshotHelper:
    """Helper for capturing model snapshots and asserting no regressions."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self._snapshot: Optional[Snapshot] = None

    def capture(
        self,
        prompts: Sequence[str | Prompt],
        model_fn: Callable[[str], str],
        model_name: str = "current",
    ) -> Snapshot:
        """Capture model outputs for the given prompts.

        Prompts can be plain strings or Prompt objects.
        """
        prompt_objs: list[Prompt] = [
            Prompt(text=p) if isinstance(p, str) else p for p in prompts
        ]
        self._snapshot = capture(prompt_objs, model_fn, model_name=model_name)
        return self._snapshot

    def assert_match(self, baseline_path: str | Path) -> DiffReport:
        """Assert that the current snapshot matches the baseline.

        Raises ``pytest.fail`` with a summary when behavioral changes are detected.
        """
        if self._snapshot is None:
            raise RuntimeError("Call .capture() before .assert_match()")

        baseline = Snapshot.load(baseline_path)
        report = diff_snapshots(baseline, self._snapshot)

        if report.n_changes > 0:
            lines = [
                f"Model regression detected: {report.n_changes} change(s) "
                f"out of {len(report.entries)} prompt(s)",
            ]
            for entry in report.entries:
                if entry.change_type != ChangeType.IDENTICAL:
                    lines.append(
                        f"  [{entry.change_type.value}] {entry.prompt.text!r}: "
                        f"{entry.description}"
                    )
            pytest.fail("\n".join(lines))

        return report

    def save(self, path: str | Path) -> None:
        """Persist the current snapshot to disk (useful for creating baselines)."""
        if self._snapshot is None:
            raise RuntimeError("Call .capture() before .save()")
        self._snapshot.save(path)


@pytest.fixture()
def modeldiff_snapshot(tmp_path: Path) -> SnapshotHelper:
    """Fixture providing a :class:`SnapshotHelper` for model regression tests."""
    return SnapshotHelper(tmp_path)
