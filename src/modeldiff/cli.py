"""CLI for modeldiff."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def _build_cli():  # type: ignore[no-untyped-def]
    try:
        import click
    except ImportError:
        raise SystemExit("CLI dependencies required: pip install modeldiff[cli]")

    @click.group()
    @click.version_option(package_name="modeldiff")
    def cli() -> None:
        """modeldiff — behavioral regression testing for LLMs."""

    @cli.command()
    @click.argument("snapshot_a", type=click.Path(exists=True))
    @click.argument("snapshot_b", type=click.Path(exists=True))
    @click.option("--json-out", "-o", type=click.Path(), default=None)
    @click.option("--markdown", "-m", is_flag=True, help="Output as Markdown.")
    def diff(snapshot_a: str, snapshot_b: str, json_out: Optional[str], markdown: bool) -> None:
        """Diff two snapshots and show behavioral changes."""
        from modeldiff._types import Snapshot
        from modeldiff.diff import diff_snapshots
        from modeldiff.report import format_markdown, format_report_rich, format_report_text, save_json

        snap_a = Snapshot.load(snapshot_a)
        snap_b = Snapshot.load(snapshot_b)
        report = diff_snapshots(snap_a, snap_b)

        if markdown:
            click.echo(format_markdown(report))
        else:
            try:
                click.echo(format_report_rich(report))
            except Exception:
                click.echo(format_report_text(report))

        if json_out:
            save_json(report, json_out)
            click.echo(f"Report saved to {json_out}", err=True)

    @cli.command()
    @click.argument("snapshot_path", type=click.Path(exists=True))
    def info(snapshot_path: str) -> None:
        """Show information about a snapshot."""
        from modeldiff._types import Snapshot

        snap = Snapshot.load(snapshot_path)
        click.echo(f"Model:     {snap.model_name}")
        click.echo(f"Responses: {snap.n_responses}")
        click.echo(f"Errors:    {snap.n_errors}")
        if snap.metadata:
            click.echo(f"Metadata:  {json.dumps(snap.metadata)}")

        # Category breakdown
        cats: dict = {}
        for r in snap.responses:
            cats[r.prompt.category] = cats.get(r.prompt.category, 0) + 1
        if cats:
            click.echo("Categories:")
            for cat, count in sorted(cats.items()):
                click.echo(f"  {cat}: {count}")

    @cli.command()
    @click.argument("snapshot_a", type=click.Path(exists=True))
    @click.argument("snapshot_b", type=click.Path(exists=True))
    def drift(snapshot_a: str, snapshot_b: str) -> None:
        """Detect statistical drift between two snapshots."""
        from modeldiff._types import Snapshot
        from modeldiff.drift import full_drift_report

        snap_a = Snapshot.load(snapshot_a)
        snap_b = Snapshot.load(snapshot_b)
        report = full_drift_report(snap_a, snap_b)

        for section, data in report.items():
            click.echo(f"\n{section.upper()} DRIFT:")
            for key, value in data.items():
                click.echo(f"  {key}: {value}")

    @cli.command(name="suites")
    def list_suites() -> None:
        """List available built-in test suites."""
        from modeldiff.suite import get_suite, list_suites as _list

        for name in _list():
            prompts = get_suite(name)
            click.echo(f"  {name:15s}  {len(prompts)} prompts")

    return cli


cli = _build_cli()

if __name__ == "__main__":
    cli()
