#!/usr/bin/env python3
"""Integration: quantbench + modeldiff — profile quantization, then diff behavior.

Flow: Use quantbench to profile a model's quantization characteristics, then
use modeldiff to capture and compare behavioral outputs between the original
and quantized versions.

Install: pip install quantbench modeldiff
"""

try:
    from quantbench import (
        profile_gguf, compare_profiles, recommend, format_recommendation,
        estimate_quality, format_report_text,
    )
except ImportError:
    raise SystemExit("pip install quantbench  # required for this example")

try:
    from modeldiff import (
        Prompt, Snapshot, capture, diff_snapshots, format_html, save_html,
    )
except ImportError:
    raise SystemExit("pip install modeldiff  # required for this example")


def mock_original(prompt: str) -> str:
    """Simulates the original FP16 model."""
    return f"[fp16] Detailed answer to: {prompt}"


def mock_quantized(prompt: str) -> str:
    """Simulates a Q4_K_M quantized model."""
    return f"[q4km] Answer to: {prompt}"


def main() -> None:
    # ── 1. Profile the quantized model with quantbench ───────────────
    print("=" * 60)
    print("STEP 1: Profile quantization with quantbench")
    print("=" * 60)
    # In real usage: profile = profile_gguf("model-q4_k_m.gguf")
    # Here we build a profile from a dict for demonstration:
    from quantbench import profile_from_dict
    q4_profile = profile_from_dict({
        "format": "GGUF",
        "quant_method": "Q4_K_M",
        "size_bytes": 4_200_000_000,
        "num_params": 7_000_000_000,
        "bits_per_weight": 4.85,
        "layers": [{"name": f"blk.{i}.attn_q", "dtype": "Q4_K"} for i in range(32)],
    })
    print(f"  Format:          {q4_profile.format}")
    print(f"  Size:            {q4_profile.size_bytes / 1e9:.1f} GB")
    print(f"  Bits per weight: {q4_profile.bits_per_weight:.2f}")

    quality = estimate_quality(q4_profile)
    print(f"  Est. quality:    {quality.score:.2f}/1.00")
    rec = recommend(num_params=7_000_000_000, memory_gb=8.0)
    print(f"  Recommendation:  {format_recommendation(rec)}")

    # ── 2. Capture behavioral snapshots with modeldiff ───────────────
    print("\n" + "=" * 60)
    print("STEP 2: Capture behavioral snapshots (modeldiff)")
    print("=" * 60)
    prompts = [
        Prompt(text="Explain the theory of relativity.", category="science"),
        Prompt(text="Write a sorting algorithm in Python.", category="code"),
        Prompt(text="Translate 'hello' to French.", category="translation"),
        Prompt(text="What is the capital of Japan?", category="factual"),
    ]
    snap_orig = capture(prompts, mock_original, model_name="llama-7b-fp16")
    snap_quant = capture(prompts, mock_quantized, model_name="llama-7b-q4km")
    print(f"  Original responses:  {len(snap_orig.responses)}")
    print(f"  Quantized responses: {len(snap_quant.responses)}")

    # ── 3. Diff the two snapshots ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Behavioral diff — original vs quantized")
    print("=" * 60)
    report = diff_snapshots(snap_orig, snap_quant)
    for entry in report.entries:
        print(f"  [{entry.severity.value:>6}] {entry.prompt.text[:45]}")
        print(f"          Change: {entry.change_type.value} — {entry.description}")

    print(f"\n  Summary: {len(report.entries)} prompts compared")
    print(f"  High severity: {sum(1 for e in report.entries if e.severity.value == 'high')}")
    print("\nQuantization profiling + behavioral regression test complete.")


if __name__ == "__main__":
    main()
