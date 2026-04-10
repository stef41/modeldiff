"""Export a behavioral diff as an HTML report.

Demonstrates: format_html(), save_html().
"""

from modeldiff import Prompt, capture, diff_snapshots, format_html, save_html


def old_model(prompt: str) -> str:
    return {
        "Translate 'hello' to French.": "Bonjour",
        "What causes rain?": "Rain is caused by water vapor condensing in clouds.",
        "Is the Earth flat?": "As an AI, I cannot make claims about that.",
    }.get(prompt, "Unknown.")


def new_model(prompt: str) -> str:
    return {
        "Translate 'hello' to French.": "Bonjour (formal) or Salut (informal)",
        "What causes rain?": "Rain forms when water vapor in clouds condenses into droplets heavy enough to fall.",
        "Is the Earth flat?": "No. The Earth is an oblate spheroid, confirmed by satellite imagery and physics.",
    }.get(prompt, "Unknown.")


if __name__ == "__main__":
    prompts = [
        Prompt(text="Translate 'hello' to French.", category="translation"),
        Prompt(text="What causes rain?", category="science"),
        Prompt(text="Is the Earth flat?", category="factual"),
    ]

    snap_old = capture(prompts, old_model, model_name="model-v3.1")
    snap_new = capture(prompts, new_model, model_name="model-v3.2")
    report = diff_snapshots(snap_old, snap_new)

    # Generate HTML string
    html = format_html(report)
    print(f"Generated HTML report: {len(html):,} characters")

    # Save to file
    output_path = "diff_report.html"
    save_html(report, output_path)
    print(f"Saved to {output_path}")
    print(f"\nReport summary:")
    print(f"  Models compared: {report.model_a} vs {report.model_b}")
    print(f"  Total entries:   {len(report.entries)}")
    print(f"  Identical:       {report.n_identical}")
    print(f"  Changed:         {report.n_changes}")
