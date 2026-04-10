"""Generate and run a behavioral regression test suite.

Demonstrates: TestSuite, generate_suite_from_snapshot(), run_suite().
"""

from modeldiff import (
    Prompt,
    TestCase,
    TestSuite,
    capture,
    generate_suite_from_snapshot,
    run_suite,
)


def mock_model(prompt: str) -> str:
    """Simulated model that answers a few known questions."""
    answers = {
        "What is Python?": "Python is a high-level programming language known for readability.",
        "Name three sorting algorithms.": "Bubble sort, merge sort, and quicksort.",
        "What is 2+2?": "The answer is 4.",
    }
    return answers.get(prompt, "I'm not sure about that.")


if __name__ == "__main__":
    # Step 1: Capture a baseline snapshot
    prompts = [
        Prompt(text="What is Python?", category="knowledge", tags=["programming"]),
        Prompt(text="Name three sorting algorithms.", category="knowledge", tags=["algorithms"]),
        Prompt(text="What is 2+2?", category="math", tags=["arithmetic"]),
    ]
    baseline = capture(prompts, mock_model, model_name="baseline-v1")

    # Step 2: Auto-generate a test suite from the snapshot
    suite = generate_suite_from_snapshot(baseline, suite_name="regression-v1")
    print(f"Generated suite: {suite.name}")
    print(f"Test cases: {len(suite.cases)}")
    for tc in suite.cases:
        print(f"  - {tc.name}: expects {tc.expected_contains}")

    # Step 3: You can also build a suite manually
    manual_suite = TestSuite(
        name="manual-checks",
        cases=[
            TestCase(
                name="python-definition",
                prompt="What is Python?",
                expected_contains=["programming", "language"],
                expected_not_contains=["Java"],
                tags=["programming"],
            ),
            TestCase(
                name="math-basic",
                prompt="What is 2+2?",
                expected_contains=["4"],
                tags=["arithmetic"],
            ),
        ],
    )

    # Step 4: Run the suite against the model
    result = run_suite(manual_suite, mock_model, model_name="baseline-v1")

    print(f"\n{'='*50}")
    print(f"Suite: {result.suite_name}")
    print(f"Pass rate: {result.pass_rate:.0%} ({result.passed}/{result.total})")
    print(f"{'='*50}")

    for cr in result.case_results:
        status = "PASS" if cr.passed else "FAIL"
        print(f"  [{status}] {cr.test_case.name}")
        if not cr.passed:
            if cr.missing_phrases:
                print(f"         Missing: {cr.missing_phrases}")
            if cr.forbidden_phrases:
                print(f"         Forbidden found: {cr.forbidden_phrases}")
