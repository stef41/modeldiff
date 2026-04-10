"""Tests for the pytest plugin (SnapshotHelper)."""

from __future__ import annotations

import json

import pytest

from modeldiff._types import Prompt, Snapshot
from modeldiff.plugin import SnapshotHelper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo(text: str) -> str:
    """Deterministic model function: echoes the prompt."""
    return text


def _upper(text: str) -> str:
    """A different 'model' that uppercases."""
    return text.upper()


PROMPTS = ["hello", "world"]


# ---------------------------------------------------------------------------
# SnapshotHelper.capture
# ---------------------------------------------------------------------------


class TestCapture:
    def test_capture_stores_snapshot(self, tmp_path):
        helper = SnapshotHelper(tmp_path)
        snap = helper.capture(PROMPTS, _echo)
        assert snap is not None
        assert len(snap.responses) == 2
        assert snap.responses[0].output == "hello"

    def test_capture_accepts_prompt_objects(self, tmp_path):
        helper = SnapshotHelper(tmp_path)
        prompts = [Prompt(text="a", category="test"), Prompt(text="b")]
        snap = helper.capture(prompts, _echo)
        assert snap.responses[0].prompt.category == "test"


# ---------------------------------------------------------------------------
# SnapshotHelper.assert_match
# ---------------------------------------------------------------------------


class TestAssertMatch:
    def test_passes_when_identical(self, tmp_path):
        # Create a baseline using the echo model
        helper = SnapshotHelper(tmp_path)
        helper.capture(PROMPTS, _echo)
        baseline_path = tmp_path / "baseline.json"
        helper.save(baseline_path)

        # New helper with same model — should pass
        helper2 = SnapshotHelper(tmp_path)
        helper2.capture(PROMPTS, _echo)
        report = helper2.assert_match(baseline_path)
        assert report.n_changes == 0

    def test_fails_when_outputs_differ(self, tmp_path):
        # Baseline from echo
        helper = SnapshotHelper(tmp_path)
        helper.capture(PROMPTS, _echo)
        baseline_path = tmp_path / "baseline.json"
        helper.save(baseline_path)

        # Current with different model
        helper2 = SnapshotHelper(tmp_path)
        helper2.capture(PROMPTS, _upper)
        with pytest.raises(pytest.fail.Exception, match="regression detected"):
            helper2.assert_match(baseline_path)

    def test_error_without_capture(self, tmp_path):
        helper = SnapshotHelper(tmp_path)
        with pytest.raises(RuntimeError, match="Call .capture"):
            helper.assert_match(tmp_path / "whatever.json")


# ---------------------------------------------------------------------------
# SnapshotHelper.save
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_creates_valid_json(self, tmp_path):
        helper = SnapshotHelper(tmp_path)
        helper.capture(PROMPTS, _echo)
        path = tmp_path / "snap.json"
        helper.save(path)

        data = json.loads(path.read_text())
        assert data["model_name"] == "current"
        assert len(data["responses"]) == 2

    def test_save_error_without_capture(self, tmp_path):
        helper = SnapshotHelper(tmp_path)
        with pytest.raises(RuntimeError, match="Call .capture"):
            helper.save(tmp_path / "snap.json")


# ---------------------------------------------------------------------------
# Fixture smoke test
# ---------------------------------------------------------------------------


class TestFixture:
    def test_modeldiff_snapshot_fixture(self, modeldiff_snapshot):
        """Verify the fixture is injected and functional."""
        assert isinstance(modeldiff_snapshot, SnapshotHelper)
        snap = modeldiff_snapshot.capture(["ping"], _echo)
        assert snap.responses[0].output == "ping"
