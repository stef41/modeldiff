"""Tests for modeldiff.generator module."""

from __future__ import annotations

import json

from modeldiff._types import Prompt, Response, Snapshot
from modeldiff.generator import (
    TestCase,
    TestSuite,
    extract_key_phrases,
    generate_suite_from_snapshot,
    run_suite,
)


def _make_snapshot(responses=None):
    """Build a minimal snapshot for testing."""
    if responses is None:
        responses = [
            Response(
                prompt=Prompt(text="What is Python?", category="knowledge", tags=["programming"]),
                output="Python is a high-level programming language known for its readability.",
                model_name="test-model",
            ),
            Response(
                prompt=Prompt(text="Explain recursion", category="reasoning", tags=["cs"]),
                output="Recursion is when a function calls itself to solve smaller subproblems.",
                model_name="test-model",
            ),
            Response(
                prompt=Prompt(text="Write a haiku", category="creative", tags=["poetry"]),
                output="Code flows like water\nSilent bugs hide in the dark\nTests bring the light back",
                model_name="test-model",
            ),
        ]
    return Snapshot(model_name="test-model", responses=responses)


class TestTestCase:
    def test_dataclass_fields(self):
        tc = TestCase(name="t1", prompt="hello")
        assert tc.name == "t1"
        assert tc.prompt == "hello"
        assert tc.expected_contains == []
        assert tc.expected_not_contains == []
        assert tc.tags == []

    def test_with_all_fields(self):
        tc = TestCase(
            name="t2",
            prompt="foo",
            expected_contains=["bar"],
            expected_not_contains=["baz"],
            tags=["a", "b"],
        )
        assert tc.expected_contains == ["bar"]
        assert tc.tags == ["a", "b"]


class TestTestSuite:
    def test_empty_suite(self):
        suite = TestSuite(name="empty")
        assert len(suite.cases) == 0
        assert suite.metadata == {}

    def test_save_and_load(self, tmp_path):
        cases = [
            TestCase(name="c1", prompt="p1", expected_contains=["x"], tags=["t1"]),
            TestCase(name="c2", prompt="p2", expected_not_contains=["y"]),
        ]
        suite = TestSuite(name="my-suite", cases=cases, metadata={"v": 1})
        path = tmp_path / "suite.json"
        suite.save(path)

        loaded = TestSuite.load(path)
        assert loaded.name == "my-suite"
        assert len(loaded.cases) == 2
        assert loaded.cases[0].expected_contains == ["x"]
        assert loaded.cases[1].expected_not_contains == ["y"]
        assert loaded.metadata == {"v": 1}

    def test_save_creates_valid_json(self, tmp_path):
        suite = TestSuite(name="js", cases=[TestCase(name="a", prompt="b")])
        path = tmp_path / "s.json"
        suite.save(path)
        data = json.loads(path.read_text())
        assert "name" in data
        assert "cases" in data

    def test_filter_by_tag(self):
        cases = [
            TestCase(name="c1", prompt="p1", tags=["math", "easy"]),
            TestCase(name="c2", prompt="p2", tags=["code"]),
            TestCase(name="c3", prompt="p3", tags=["math", "hard"]),
        ]
        suite = TestSuite(name="all", cases=cases)
        filtered = suite.filter_by_tag("math")
        assert len(filtered.cases) == 2
        assert filtered.cases[0].name == "c1"
        assert filtered.cases[1].name == "c3"

    def test_filter_by_tag_no_match(self):
        suite = TestSuite(name="s", cases=[TestCase(name="c", prompt="p", tags=["x"])])
        filtered = suite.filter_by_tag("nope")
        assert len(filtered.cases) == 0

    def test_filter_preserves_metadata(self):
        suite = TestSuite(name="s", cases=[], metadata={"k": "v"})
        filtered = suite.filter_by_tag("t")
        assert filtered.metadata == {"k": "v"}


class TestExtractKeyPhrases:
    def test_extracts_phrases(self):
        text = "Python is a high-level programming language for general purpose computing"
        phrases = extract_key_phrases(text)
        assert len(phrases) > 0
        assert len(phrases) <= 5

    def test_empty_text(self):
        assert extract_key_phrases("") == []
        assert extract_key_phrases("   ") == []

    def test_max_phrases_limit(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        phrases = extract_key_phrases(text, max_phrases=3)
        assert len(phrases) <= 3

    def test_returns_strings(self):
        phrases = extract_key_phrases("The quick brown fox jumps over the lazy dog")
        for p in phrases:
            assert isinstance(p, str)

    def test_single_word(self):
        phrases = extract_key_phrases("hello")
        assert len(phrases) <= 1


class TestGenerateSuiteFromSnapshot:
    def test_generates_cases(self):
        snap = _make_snapshot()
        suite = generate_suite_from_snapshot(snap)
        assert suite.name == "regression"
        assert len(suite.cases) == 3

    def test_case_has_prompt(self):
        snap = _make_snapshot()
        suite = generate_suite_from_snapshot(snap)
        prompts = [tc.prompt for tc in suite.cases]
        assert "What is Python?" in prompts

    def test_case_has_expected_contains(self):
        snap = _make_snapshot()
        suite = generate_suite_from_snapshot(snap)
        # At least one case should have extracted phrases
        has_phrases = any(len(tc.expected_contains) > 0 for tc in suite.cases)
        assert has_phrases

    def test_skips_error_responses(self):
        responses = [
            Response(
                prompt=Prompt(text="fail"),
                output="",
                model_name="m",
                error="timeout",
            ),
            Response(
                prompt=Prompt(text="ok"),
                output="This is fine output text with content",
                model_name="m",
            ),
        ]
        snap = _make_snapshot(responses)
        suite = generate_suite_from_snapshot(snap)
        assert len(suite.cases) == 1

    def test_custom_name(self):
        snap = _make_snapshot()
        suite = generate_suite_from_snapshot(snap, name="custom")
        assert suite.name == "custom"

    def test_metadata_has_source_model(self):
        snap = _make_snapshot()
        suite = generate_suite_from_snapshot(snap)
        assert suite.metadata["source_model"] == "test-model"

    def test_tags_include_category(self):
        snap = _make_snapshot()
        suite = generate_suite_from_snapshot(snap)
        for tc in suite.cases:
            assert len(tc.tags) > 0


class TestRunSuite:
    def test_all_pass(self):
        suite = TestSuite(
            name="ok",
            cases=[
                TestCase(name="t1", prompt="p", expected_contains=["hello"]),
            ],
        )
        result = run_suite(suite, lambda p: "hello world")
        assert result.passed == 1
        assert result.failed == 0
        assert result.pass_rate == 1.0

    def test_fail_missing_phrase(self):
        suite = TestSuite(
            name="fail",
            cases=[
                TestCase(name="t1", prompt="p", expected_contains=["missing"]),
            ],
        )
        result = run_suite(suite, lambda p: "hello world")
        assert result.failed == 1
        assert result.case_results[0].missing_phrases == ["missing"]

    def test_fail_forbidden_phrase(self):
        suite = TestSuite(
            name="bad",
            cases=[
                TestCase(name="t1", prompt="p", expected_not_contains=["hello"]),
            ],
        )
        result = run_suite(suite, lambda p: "hello world")
        assert result.failed == 1
        assert result.case_results[0].forbidden_phrases == ["hello"]

    def test_model_fn_exception(self):
        suite = TestSuite(
            name="err",
            cases=[TestCase(name="t1", prompt="p")],
        )

        def failing_fn(p):
            raise RuntimeError("boom")

        result = run_suite(suite, failing_fn)
        assert result.failed == 1
        assert result.case_results[0].error == "boom"

    def test_empty_suite(self):
        suite = TestSuite(name="empty", cases=[])
        result = run_suite(suite, lambda p: "x")
        assert result.total == 0
        assert result.pass_rate == 0.0

    def test_case_insensitive_matching(self):
        suite = TestSuite(
            name="ci",
            cases=[
                TestCase(name="t1", prompt="p", expected_contains=["Hello"]),
            ],
        )
        result = run_suite(suite, lambda p: "HELLO world")
        assert result.passed == 1

    def test_suite_result_fields(self):
        suite = TestSuite(
            name="mixed",
            cases=[
                TestCase(name="t1", prompt="p1", expected_contains=["yes"]),
                TestCase(name="t2", prompt="p2", expected_contains=["no"]),
            ],
        )
        result = run_suite(suite, lambda p: "yes")
        assert result.suite_name == "mixed"
        assert result.total == 2
        assert result.passed == 1
        assert result.failed == 1
