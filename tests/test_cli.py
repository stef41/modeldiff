"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from modeldiff._types import Prompt, Response, Snapshot
from modeldiff.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def snap_files(tmp_path):
    """Create two snapshot files for testing."""
    prompts = [Prompt(text="hello"), Prompt(text="world")]

    snap_a = Snapshot(model_name="model-a", responses=[
        Response(prompt=prompts[0], output="Hi there!", model_name="model-a", latency_ms=100.0),
        Response(prompt=prompts[1], output="Earth.", model_name="model-a", latency_ms=120.0),
    ])
    snap_b = Snapshot(model_name="model-b", responses=[
        Response(prompt=prompts[0], output="Hey!", model_name="model-b", latency_ms=90.0),
        Response(prompt=prompts[1], output="Earth.", model_name="model-b", latency_ms=110.0),
    ])

    path_a = tmp_path / "snap_a.json"
    path_b = tmp_path / "snap_b.json"
    snap_a.save(str(path_a))
    snap_b.save(str(path_b))
    return str(path_a), str(path_b)


class TestDiffCommand:
    def test_basic(self, runner, snap_files):
        result = runner.invoke(cli, ["diff", snap_files[0], snap_files[1]])
        assert result.exit_code == 0

    def test_markdown_output(self, runner, snap_files):
        result = runner.invoke(cli, ["diff", snap_files[0], snap_files[1], "--markdown"])
        assert result.exit_code == 0

    def test_missing_file(self, runner, tmp_path):
        result = runner.invoke(cli, ["diff", str(tmp_path / "missing.json"), str(tmp_path / "also_missing.json")])
        assert result.exit_code != 0


class TestInfoCommand:
    def test_basic(self, runner, snap_files):
        result = runner.invoke(cli, ["info", snap_files[0]])
        assert result.exit_code == 0
        assert "model-a" in result.output

    def test_missing_file(self, runner, tmp_path):
        result = runner.invoke(cli, ["info", str(tmp_path / "missing.json")])
        assert result.exit_code != 0


class TestDriftCommand:
    def test_basic(self, runner, snap_files):
        result = runner.invoke(cli, ["drift", snap_files[0], snap_files[1]])
        assert result.exit_code == 0

    def test_basic_output(self, runner, snap_files):
        result = runner.invoke(cli, ["drift", snap_files[0], snap_files[1]])
        assert result.exit_code == 0
        assert "DRIFT" in result.output


class TestSuitesCommand:
    def test_list(self, runner):
        result = runner.invoke(cli, ["suites"])
        assert result.exit_code == 0
        assert "reasoning" in result.output
