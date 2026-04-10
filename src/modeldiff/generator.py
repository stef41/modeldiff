"""Regression test suite generation from captured snapshots."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modeldiff._types import Snapshot


@dataclass
class TestCase:
    """A single regression test case."""

    name: str
    prompt: str
    expected_contains: List[str] = field(default_factory=list)
    expected_not_contains: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Result of running a single test case."""

    test_case: TestCase
    passed: bool
    actual_output: str
    missing_phrases: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class SuiteResult:
    """Result of running an entire test suite."""

    suite_name: str
    total: int
    passed: int
    failed: int
    case_results: List[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class TestSuite:
    """A collection of test cases."""

    name: str
    cases: List[TestCase] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save the suite to a JSON file."""
        data = {
            "name": self.name,
            "metadata": self.metadata,
            "cases": [
                {
                    "name": tc.name,
                    "prompt": tc.prompt,
                    "expected_contains": tc.expected_contains,
                    "expected_not_contains": tc.expected_not_contains,
                    "tags": tc.tags,
                }
                for tc in self.cases
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "TestSuite":
        """Load a suite from a JSON file."""
        data = json.loads(Path(path).read_text())
        cases = [
            TestCase(
                name=c["name"],
                prompt=c["prompt"],
                expected_contains=c.get("expected_contains", []),
                expected_not_contains=c.get("expected_not_contains", []),
                tags=c.get("tags", []),
            )
            for c in data.get("cases", [])
        ]
        return cls(
            name=data.get("name", ""),
            cases=cases,
            metadata=data.get("metadata", {}),
        )

    def filter_by_tag(self, tag: str) -> "TestSuite":
        """Return a new TestSuite containing only cases with the given tag."""
        filtered = [tc for tc in self.cases if tag in tc.tags]
        return TestSuite(
            name=f"{self.name}[{tag}]",
            cases=filtered,
            metadata=self.metadata,
        )


# Stop words excluded from key phrase extraction
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could this that these those "
    "i me my we our you your he him his she her it its they them their "
    "and but or nor for yet so if then else when while as at by from in "
    "into of on to with about between through during before after above "
    "below up down out off over under again further also too very just "
    "not no nor don't doesn't didn't won't wouldn't isn't aren't "
    "there here where how what which who whom why".split()
)


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract important phrases from text for assertion checking.

    Uses a simple frequency-based approach: split into sentences, extract
    multi-word chunks, and pick the most distinctive ones.
    """
    if not text.strip():
        return []

    # Split into words, keep only alpha-numeric, lowercase
    words = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text)
    if not words:
        return []

    # Build bigrams and trigrams from original case text
    clean_words = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text)
    phrases: List[str] = []

    # Single important words (not stop words, length > 3)
    word_counts: Counter[str] = Counter()
    for w in clean_words:
        low = w.lower()
        if low not in _STOP_WORDS and len(low) > 3:
            word_counts[low] += 1

    # Bigrams
    bigram_counts: Counter[str] = Counter()
    for i in range(len(clean_words) - 1):
        a, b = clean_words[i], clean_words[i + 1]
        if a.lower() not in _STOP_WORDS or b.lower() not in _STOP_WORDS:
            bigram = f"{a} {b}"
            bigram_counts[bigram.lower()] += 1

    # Prefer bigrams, then single words
    seen_lower: set[str] = set()
    for phrase, _ in bigram_counts.most_common(max_phrases * 2):
        if len(phrases) >= max_phrases:
            break
        # Find original case version in text
        idx = text.lower().find(phrase)
        if idx >= 0:
            original = text[idx : idx + len(phrase)]
        else:
            original = phrase
        if original.lower() not in seen_lower:
            phrases.append(original)
            seen_lower.add(original.lower())

    for word, _ in word_counts.most_common(max_phrases * 2):
        if len(phrases) >= max_phrases:
            break
        if word not in seen_lower:
            # Find original case
            idx = text.lower().find(word)
            if idx >= 0:
                original = text[idx : idx + len(word)]
            else:
                original = word
            if original.lower() not in seen_lower:
                phrases.append(original)
                seen_lower.add(original.lower())

    return phrases[:max_phrases]


def generate_suite_from_snapshot(
    snapshot: Snapshot,
    name: str = "regression",
) -> TestSuite:
    """Generate a TestSuite from a captured Snapshot.

    Each non-error response becomes a test case, with expected_contains
    derived from key phrases in the output.
    """
    cases: List[TestCase] = []

    for i, response in enumerate(snapshot.responses):
        if response.is_error:
            continue

        key_phrases = extract_key_phrases(response.output)
        case = TestCase(
            name=f"{name}_{i:03d}_{response.prompt.category}",
            prompt=response.prompt.text,
            expected_contains=key_phrases,
            expected_not_contains=[],
            tags=list(response.prompt.tags) + [response.prompt.category],
        )
        cases.append(case)

    return TestSuite(
        name=name,
        cases=cases,
        metadata={
            "source_model": snapshot.model_name,
            "n_responses": snapshot.n_responses,
        },
    )


def run_suite(
    suite: TestSuite,
    model_fn: Callable[[str], str],
) -> SuiteResult:
    """Run a test suite against a model function.

    Args:
        suite: The test suite to run.
        model_fn: A function that takes a prompt string and returns an output string.

    Returns:
        SuiteResult with pass/fail counts and per-case results.
    """
    case_results: List[CaseResult] = []
    passed = 0
    failed = 0

    for tc in suite.cases:
        error = None
        try:
            output = model_fn(tc.prompt)
        except Exception as e:
            output = ""
            error = str(e)

        output_lower = output.lower()

        missing = [
            phrase for phrase in tc.expected_contains
            if phrase.lower() not in output_lower
        ]
        forbidden = [
            phrase for phrase in tc.expected_not_contains
            if phrase.lower() in output_lower
        ]

        ok = not missing and not forbidden and error is None
        if ok:
            passed += 1
        else:
            failed += 1

        case_results.append(CaseResult(
            test_case=tc,
            passed=ok,
            actual_output=output,
            missing_phrases=missing,
            forbidden_phrases=forbidden,
            error=error,
        ))

    return SuiteResult(
        suite_name=suite.name,
        total=len(suite.cases),
        passed=passed,
        failed=failed,
        case_results=case_results,
    )
