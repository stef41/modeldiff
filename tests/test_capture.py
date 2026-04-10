"""Tests for capture module."""

import json
import pytest
from modeldiff._types import Prompt, Snapshot
from modeldiff.capture import capture, capture_from_file


class TestCapture:
    def test_basic(self):
        prompts = [Prompt(text="hello"), Prompt(text="world")]
        snap = capture(prompts, lambda t: t.upper(), model_name="echo")
        assert snap.model_name == "echo"
        assert snap.n_responses == 2
        assert snap.responses[0].output == "HELLO"
        assert snap.responses[1].output == "WORLD"

    def test_latency_recorded(self):
        snap = capture([Prompt(text="q")], lambda t: "a", model_name="m")
        assert snap.responses[0].latency_ms >= 0

    def test_error_handling(self):
        def fail(text):
            raise RuntimeError("broken")

        snap = capture([Prompt(text="q")], fail, model_name="m")
        assert snap.n_responses == 1
        assert snap.responses[0].is_error
        assert "broken" in snap.responses[0].error

    def test_progress_callback(self):
        progress = []

        def on_progress(current, total):
            progress.append((current, total))

        prompts = [Prompt(text=f"p{i}") for i in range(3)]
        capture(prompts, lambda t: "a", model_name="m", on_progress=on_progress)
        assert len(progress) == 3

    def test_metadata(self):
        snap = capture(
            [Prompt(text="q")],
            lambda t: "a",
            model_name="m",
            metadata={"version": "1.0"},
        )
        assert snap.metadata == {"version": "1.0"}

    def test_preserves_prompt_info(self):
        p = Prompt(text="q", category="math", tags=["hard"])
        snap = capture([p], lambda t: "42", model_name="m")
        assert snap.responses[0].prompt.category == "math"
        assert snap.responses[0].prompt.tags == ["hard"]

    def test_empty_prompts(self):
        snap = capture([], lambda t: "a", model_name="m")
        assert snap.n_responses == 0


class TestCaptureFromFile:
    def test_json_list(self, tmp_path):
        data = [
            {"text": "hello", "category": "test"},
            {"text": "world"},
        ]
        path = tmp_path / "prompts.json"
        path.write_text(json.dumps(data))
        snap = capture_from_file(str(path), lambda t: t.upper(), model_name="m")
        assert snap.n_responses == 2
        assert snap.responses[0].output == "HELLO"

    def test_jsonl(self, tmp_path):
        lines = [
            json.dumps({"text": "hello"}),
            json.dumps({"text": "world"}),
        ]
        path = tmp_path / "prompts.jsonl"
        path.write_text("\n".join(lines))
        snap = capture_from_file(str(path), lambda t: t.upper(), model_name="m")
        assert snap.n_responses == 2

    def test_prompt_key(self, tmp_path):
        data = [{"prompt": "hello"}]
        path = tmp_path / "prompts.json"
        path.write_text(json.dumps(data))
        snap = capture_from_file(str(path), lambda t: t.upper(), model_name="m")
        assert snap.responses[0].output == "HELLO"

    def test_string_list(self, tmp_path):
        data = ["hello", "world"]
        path = tmp_path / "prompts.json"
        path.write_text(json.dumps(data))
        snap = capture_from_file(str(path), lambda t: t.upper(), model_name="m")
        assert snap.n_responses == 2
