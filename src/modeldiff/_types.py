"""Core types for modeldiff."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


class ChangeType(str, Enum):
    """Type of behavioral change detected."""

    CONTENT = "content"  # Different factual content
    FORMAT = "format"  # Different formatting/structure
    REFUSAL = "refusal"  # One refuses, other doesn't
    LENGTH = "length"  # Significant length difference
    STYLE = "style"  # Different tone/style
    IDENTICAL = "identical"
    ERROR = "error"  # One errored, other didn't


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Prompt:
    """A prompt in a test suite."""

    text: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected: Optional[str] = None  # optional reference answer


@dataclass
class Response:
    """A captured model response."""

    prompt: Prompt
    output: str
    model_name: str
    latency_ms: float = 0.0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def is_refusal(self) -> bool:
        refusal_phrases = [
            "i can't", "i cannot", "i'm unable", "i am unable",
            "i won't", "i will not", "as an ai",
            "i'm not able", "i don't", "i do not",
        ]
        low = self.output.lower()
        return any(p in low for p in refusal_phrases)

    @property
    def word_count(self) -> int:
        return len(self.output.split())


@dataclass
class Snapshot:
    """A collection of responses — one model's output on a test suite."""

    model_name: str
    responses: List[Response] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_responses(self) -> int:
        return len(self.responses)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.responses if r.is_error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metadata": self.metadata,
            "responses": [
                {
                    "prompt": {
                        "text": r.prompt.text,
                        "category": r.prompt.category,
                        "tags": r.prompt.tags,
                        "metadata": r.prompt.metadata,
                        "expected": r.prompt.expected,
                    },
                    "output": r.output,
                    "model_name": r.model_name,
                    "latency_ms": r.latency_ms,
                    "token_count": r.token_count,
                    "metadata": r.metadata,
                    "error": r.error,
                }
                for r in self.responses
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        responses = []
        for r in data.get("responses", []):
            p = r.get("prompt", {})
            prompt = Prompt(
                text=p.get("text", ""),
                category=p.get("category", "general"),
                tags=p.get("tags", []),
                metadata=p.get("metadata", {}),
                expected=p.get("expected"),
            )
            responses.append(Response(
                prompt=prompt,
                output=r.get("output", ""),
                model_name=r.get("model_name", data.get("model_name", "")),
                latency_ms=r.get("latency_ms", 0.0),
                token_count=r.get("token_count", 0),
                metadata=r.get("metadata", {}),
                error=r.get("error"),
            ))
        return cls(
            model_name=data.get("model_name", ""),
            responses=responses,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "Snapshot":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


@dataclass
class DiffEntry:
    """A single behavioral difference between two responses."""

    prompt: Prompt
    output_a: str
    output_b: str
    change_type: ChangeType
    severity: Severity
    description: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DiffReport:
    """Full behavioral diff between two snapshots."""

    model_a: str
    model_b: str
    entries: List[DiffEntry] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_changes(self) -> int:
        return sum(1 for e in self.entries if e.change_type != ChangeType.IDENTICAL)

    @property
    def n_identical(self) -> int:
        return sum(1 for e in self.entries if e.change_type == ChangeType.IDENTICAL)

    @property
    def change_rate(self) -> float:
        total = len(self.entries)
        return self.n_changes / total if total else 0.0

    @property
    def by_type(self) -> Dict[ChangeType, int]:
        counts: Dict[ChangeType, int] = {}
        for e in self.entries:
            counts[e.change_type] = counts.get(e.change_type, 0) + 1
        return counts

    @property
    def by_severity(self) -> Dict[Severity, int]:
        counts: Dict[Severity, int] = {}
        for e in self.entries:
            counts[e.severity] = counts.get(e.severity, 0) + 1
        return counts

    @property
    def regression_score(self) -> float:
        """0.0 (no regressions) to 1.0 (everything changed critically)."""
        if not self.entries:
            return 0.0
        weights = {Severity.LOW: 0.1, Severity.MEDIUM: 0.3, Severity.HIGH: 0.6, Severity.CRITICAL: 1.0}
        total = sum(weights.get(e.severity, 0) for e in self.entries if e.change_type != ChangeType.IDENTICAL)
        max_possible = len(self.entries) * 1.0
        return min(total / max_possible, 1.0) if max_possible > 0 else 0.0


@dataclass
class FingerprintResult:
    """Model behavioral fingerprint — compact signature."""

    model_name: str
    dimensions: Dict[str, float] = field(default_factory=dict)
    # e.g., {"verbosity": 0.7, "refusal_rate": 0.1, "avg_length": 250}


class ModeldiffError(Exception):
    """Base exception."""
