"""Capture model outputs and diff two snapshots.

Demonstrates: capture(), diff_snapshots(), and printing the diff report.
"""

from modeldiff import Prompt, capture, diff_snapshots

# Define a set of test prompts
prompts = [
    Prompt(text="What is the capital of France?", category="knowledge"),
    Prompt(text="Write a haiku about programming.", category="creative"),
    Prompt(text="Explain quicksort in one sentence.", category="technical"),
    Prompt(text="Is 17 a prime number?", category="math"),
]


# Simulate two different model versions with simple callables
def model_v1(prompt: str) -> str:
    responses = {
        "What is the capital of France?": "The capital of France is Paris.",
        "Write a haiku about programming.": "Code flows like water\nBugs hide in the smallest lines\nTests bring the peace back",
        "Explain quicksort in one sentence.": "Quicksort partitions an array around a pivot element recursively.",
        "Is 17 a prime number?": "Yes, 17 is a prime number.",
    }
    return responses.get(prompt, "I don't know.")


def model_v2(prompt: str) -> str:
    responses = {
        "What is the capital of France?": "Paris is the capital of France.",
        "Write a haiku about programming.": "Semicolons fall\nLike rain upon the keyboard\nCompilation fails",
        "Explain quicksort in one sentence.": "Quicksort is a divide-and-conquer sorting algorithm with O(n log n) average time.",
        "Is 17 a prime number?": "Yes, 17 is prime.",
    }
    return responses.get(prompt, "I don't know.")


if __name__ == "__main__":
    # Capture outputs from both model versions
    snap_a = capture(prompts, model_v1, model_name="gpt-v1")
    snap_b = capture(prompts, model_v2, model_name="gpt-v2")

    print(f"Captured {len(snap_a.responses)} responses from {snap_a.model_name}")
    print(f"Captured {len(snap_b.responses)} responses from {snap_b.model_name}")

    # Diff the two snapshots
    report = diff_snapshots(snap_a, snap_b)

    print(f"\n{'='*60}")
    print(f"Diff: {report.model_a} vs {report.model_b}")
    print(f"{'='*60}")
    print(f"Total prompts: {len(report.entries)}")
    print(f"Identical:     {report.n_identical}")
    print(f"Changed:       {report.n_changes}")
    print()

    for entry in report.entries:
        icon = "✓" if entry.change_type.value == "identical" else "✗"
        print(f"  {icon} [{entry.severity.value:8s}] {entry.change_type.value:10s} — {entry.prompt.text[:50]}")
        if entry.change_type.value != "identical":
            print(f"    A: {entry.output_a[:60]}...")
            print(f"    B: {entry.output_b[:60]}...")
