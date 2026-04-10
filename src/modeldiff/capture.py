"""Capture model outputs into structured snapshots."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from modeldiff._types import Prompt, Response, Snapshot


def capture(
    prompts: Sequence[Prompt],
    model_fn: Callable[[str], str],
    model_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Snapshot:
    """Capture model outputs for a list of prompts.

    Args:
        prompts: List of prompts to run.
        model_fn: Function that takes a prompt string and returns output string.
        model_name: Name/version of the model.
        metadata: Additional metadata for the snapshot.
        on_progress: Callback (current, total) for progress reporting.
    """
    responses: List[Response] = []

    for i, prompt in enumerate(prompts):
        if on_progress:
            on_progress(i, len(prompts))

        start = time.monotonic()
        try:
            output = model_fn(prompt.text)
            latency = (time.monotonic() - start) * 1000
            responses.append(Response(
                prompt=prompt,
                output=output,
                model_name=model_name,
                latency_ms=latency,
                token_count=len(output.split()),
            ))
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            responses.append(Response(
                prompt=prompt,
                output="",
                model_name=model_name,
                latency_ms=latency,
                error=str(e),
            ))

    return Snapshot(
        model_name=model_name,
        responses=responses,
        metadata=metadata or {},
    )


def capture_from_file(
    prompts_path: str,
    model_fn: Callable[[str], str],
    model_name: str = "unknown",
) -> Snapshot:
    """Load prompts from a JSON/JSONL file and capture outputs."""
    import json
    from pathlib import Path

    path = Path(prompts_path)
    content = path.read_text()

    prompts: List[Prompt] = []
    if path.suffix == ".jsonl":
        for line in content.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            prompts.append(_prompt_from_dict(data))
    else:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                prompts.append(_prompt_from_dict(item))
        else:
            prompts.append(_prompt_from_dict(data))

    return capture(prompts, model_fn, model_name)


def _prompt_from_dict(data: dict) -> Prompt:
    if isinstance(data, str):
        return Prompt(text=data)
    return Prompt(
        text=data.get("text", data.get("prompt", "")),
        category=data.get("category", "general"),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
        expected=data.get("expected"),
    )
