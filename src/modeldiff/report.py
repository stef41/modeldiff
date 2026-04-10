"""Report formatting and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from modeldiff._types import ChangeType, DiffReport, Severity


def report_to_dict(report: DiffReport) -> Dict[str, Any]:
    return {
        "model_a": report.model_a,
        "model_b": report.model_b,
        "summary": report.summary,
        "regression_score": round(report.regression_score, 4),
        "change_rate": round(report.change_rate, 4),
        "n_changes": report.n_changes,
        "n_identical": report.n_identical,
        "by_type": {k.value: v for k, v in report.by_type.items()},
        "by_severity": {k.value: v for k, v in report.by_severity.items()},
        "entries": [
            {
                "prompt": e.prompt.text[:200],
                "category": e.prompt.category,
                "change_type": e.change_type.value,
                "severity": e.severity.value,
                "description": e.description,
                "metrics": e.metrics,
                "output_a": e.output_a[:500],
                "output_b": e.output_b[:500],
            }
            for e in report.entries
        ],
    }


def save_json(report: DiffReport, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())  # type: ignore[no-any-return]


def format_report_text(report: DiffReport) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("MODEL BEHAVIORAL DIFF REPORT")
    lines.append("=" * 60)
    lines.append(f"  Model A:          {report.model_a}")
    lines.append(f"  Model B:          {report.model_b}")
    lines.append(f"  Total prompts:    {len(report.entries)}")
    lines.append(f"  Changed:          {report.n_changes}")
    lines.append(f"  Identical:        {report.n_identical}")
    lines.append(f"  Change rate:      {report.change_rate:.1%}")
    lines.append(f"  Regression score: {report.regression_score:.2f}")
    lines.append("")

    # By type
    lines.append("-" * 60)
    lines.append("CHANGES BY TYPE")
    for ct, count in sorted(report.by_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {ct.value:12s}  {count}")
    lines.append("")

    # By severity
    lines.append("-" * 60)
    lines.append("CHANGES BY SEVERITY")
    for sev, count in sorted(report.by_severity.items(), key=lambda x: -x[1]):
        lines.append(f"  {sev.value:10s}  {count}")
    lines.append("")

    # Details (non-identical only)
    changed = [e for e in report.entries if e.change_type != ChangeType.IDENTICAL]
    if changed:
        lines.append("-" * 60)
        lines.append("DETAILED CHANGES")
        for e in changed[:20]:
            prefix = "!!" if e.severity in (Severity.HIGH, Severity.CRITICAL) else " >"
            lines.append(f"  {prefix} [{e.change_type.value}] {e.prompt.text[:60]}")
            lines.append(f"     {e.description}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_report_rich(report: DiffReport) -> str:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return format_report_text(report)

    console = Console(record=True, width=100)

    # Header
    status = "LOW RISK" if report.regression_score < 0.2 else "MEDIUM RISK" if report.regression_score < 0.5 else "HIGH RISK"
    status_style = "green" if report.regression_score < 0.2 else "yellow" if report.regression_score < 0.5 else "red"

    header = Text()
    header.append(f"{report.model_a}", style="cyan")
    header.append(" → ", style="dim")
    header.append(f"{report.model_b}", style="cyan")
    header.append("  |  ", style="dim")
    header.append(status, style=f"bold {status_style}")
    header.append(f"  |  {report.change_rate:.0%} changed", style="dim")

    console.print(Panel(header, title="[bold]modeldiff[/bold]", border_style="blue"))

    # Changes table
    table = Table(title="Changes", show_lines=False)
    table.add_column("Prompt", max_width=40, style="cyan")
    table.add_column("Category", style="dim")
    table.add_column("Type")
    table.add_column("Severity")
    table.add_column("Description", max_width=35)

    type_styles = {
        ChangeType.CONTENT: "red",
        ChangeType.REFUSAL: "red bold",
        ChangeType.FORMAT: "green",
        ChangeType.STYLE: "yellow",
        ChangeType.LENGTH: "yellow",
        ChangeType.ERROR: "red bold",
        ChangeType.IDENTICAL: "dim",
    }
    sev_styles = {
        Severity.LOW: "dim",
        Severity.MEDIUM: "yellow",
        Severity.HIGH: "red",
        Severity.CRITICAL: "red bold",
    }

    for e in report.entries:
        if e.change_type == ChangeType.IDENTICAL:
            continue
        table.add_row(
            e.prompt.text[:40],
            e.prompt.category,
            Text(e.change_type.value, style=type_styles.get(e.change_type, "")),
            Text(e.severity.value, style=sev_styles.get(e.severity, "")),
            e.description[:35],
        )

    if table.row_count > 0:
        console.print(table)
    else:
        console.print("[green]No behavioral changes detected.[/green]")

    return console.export_text()


def format_markdown(report: DiffReport) -> str:
    lines: List[str] = []
    lines.append(f"# Behavioral Diff: {report.model_a} → {report.model_b}")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total prompts | {len(report.entries)} |")
    lines.append(f"| Changed | {report.n_changes} |")
    lines.append(f"| Change rate | {report.change_rate:.1%} |")
    lines.append(f"| Regression score | {report.regression_score:.2f} |")
    lines.append("")

    changed = [e for e in report.entries if e.change_type != ChangeType.IDENTICAL]
    if changed:
        lines.append("## Changes")
        lines.append("")
        lines.append("| Prompt | Type | Severity | Description |")
        lines.append("|---|---|---|---|")
        for e in changed:
            lines.append(
                f"| {e.prompt.text[:50]} | {e.change_type.value} | {e.severity.value} | {e.description} |"
            )

    return "\n".join(lines)
