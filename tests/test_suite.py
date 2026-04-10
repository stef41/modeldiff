"""Tests for suite module."""

import pytest

from modeldiff.suite import get_standard_suite, get_suite, list_suites


class TestGetSuite:
    def test_reasoning(self):
        prompts = get_suite("reasoning")
        assert len(prompts) == 5
        for p in prompts:
            assert p.category == "reasoning"

    def test_instruction(self):
        prompts = get_suite("instruction")
        assert len(prompts) == 5
        for p in prompts:
            assert p.category == "instruction"

    def test_safety(self):
        prompts = get_suite("safety")
        assert len(prompts) == 5
        for p in prompts:
            assert p.category == "safety"

    def test_knowledge(self):
        prompts = get_suite("knowledge")
        assert len(prompts) == 5

    def test_code(self):
        prompts = get_suite("code")
        assert len(prompts) == 5

    def test_unknown_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_suite("nonexistent")


class TestGetStandardSuite:
    def test_count(self):
        prompts = get_standard_suite()
        assert len(prompts) == 25

    def test_categories(self):
        prompts = get_standard_suite()
        categories = {p.category for p in prompts}
        assert "reasoning" in categories
        assert "safety" in categories

    def test_all_have_text(self):
        prompts = get_standard_suite()
        for p in prompts:
            assert len(p.text) > 0


class TestListSuites:
    def test_returns_names(self):
        names = list_suites()
        assert "reasoning" in names
        assert "instruction" in names
        assert "safety" in names
        assert "knowledge" in names
        assert "code" in names
        assert len(names) == 5
