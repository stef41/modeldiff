"""Generate SVG terminal screenshots for README."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── SVG 1: Diff Report ──

console = Console(record=True, width=90)

header = Table(show_header=False, box=None, padding=(0, 2))
header.add_column(style="bold cyan", width=20)
header.add_column()
header.add_row("Model A", "gpt-4-0613")
header.add_row("Model B", "gpt-4-1106-preview")
header.add_row("Total prompts", "25")
header.add_row("Changed", "8 (32.0%)")
header.add_row("Regression score", "[bold red]0.42[/bold red]")

console.print(Panel(header, title="[bold]modeldiff — Behavioral Diff Report[/bold]", border_style="blue"))
console.print()

table = Table(title="Changes Detected", show_lines=True)
table.add_column("Prompt", style="white", max_width=30)
table.add_column("Type", style="bold")
table.add_column("Severity", style="bold")
table.add_column("Details", max_width=30)

table.add_row(
    "What is quantum entanglement?",
    "[yellow]CONTENT[/yellow]",
    "[red]HIGH[/red]",
    "Similarity: 0.62 — new model gives shorter, less detailed answer"
)
table.add_row(
    "Write a Python fibonacci function",
    "[cyan]FORMAT[/cyan]",
    "[green]LOW[/green]",
    "Similarity: 0.97 — minor whitespace/formatting change"
)
table.add_row(
    "Explain the trolley problem",
    "[magenta]REFUSAL[/magenta]",
    "[red]CRITICAL[/red]",
    "New model refuses; old model answered"
)
table.add_row(
    "Summarize the French Revolution",
    "[yellow]CONTENT[/yellow]",
    "[yellow]MEDIUM[/yellow]",
    "Similarity: 0.78 — factual differences detected"
)
table.add_row(
    "Generate a SQL query for...",
    "[blue]STYLE[/blue]",
    "[yellow]MEDIUM[/yellow]",
    "Similarity: 0.85 — tone/verbosity shift"
)
table.add_row(
    "What are the side effects?",
    "[yellow]CONTENT[/yellow]",
    "[red]HIGH[/red]",
    "Similarity: 0.55 — significant content drift"
)
console.print(table)
console.print()

by_type = Table(title="By Change Type", show_header=True)
by_type.add_column("Type")
by_type.add_column("Count", justify="right")
by_type.add_row("[yellow]CONTENT[/yellow]", "3")
by_type.add_row("[magenta]REFUSAL[/magenta]", "1")
by_type.add_row("[cyan]FORMAT[/cyan]", "1")
by_type.add_row("[blue]STYLE[/blue]", "1")
by_type.add_row("[dim]LENGTH[/dim]", "1")
by_type.add_row("[dim]ERROR[/dim]", "1")

by_sev = Table(title="By Severity", show_header=True)
by_sev.add_column("Severity")
by_sev.add_column("Count", justify="right")
by_sev.add_row("[red]CRITICAL[/red]", "1")
by_sev.add_row("[red]HIGH[/red]", "2")
by_sev.add_row("[yellow]MEDIUM[/yellow]", "3")
by_sev.add_row("[green]LOW[/green]", "2")

from rich.columns import Columns
console.print(Columns([by_type, by_sev], padding=4))

svg = console.export_svg(title="modeldiff diff")
with open("/data/users/zacharie/repogen/modeldiff/assets/diff_report.svg", "w") as f:
    f.write(svg)
print(f"diff_report.svg: {len(svg):,} bytes")

# ── SVG 2: Drift Analysis ──

console2 = Console(record=True, width=90)

drift_table = Table(title="Statistical Drift Analysis", show_lines=True)
drift_table.add_column("Metric", style="bold")
drift_table.add_column("Model A", justify="right")
drift_table.add_column("Model B", justify="right")
drift_table.add_column("Delta", justify="right")
drift_table.add_column("Significant?", justify="center")

drift_table.add_row("Avg length (words)", "142.3", "98.7", "[red]-43.6[/red]", "[red]⚠ YES (3.2σ)[/red]")
drift_table.add_row("Refusal rate", "0.04", "0.12", "[red]+0.08[/red]", "[red]⚠ YES[/red]")
drift_table.add_row("Avg latency (ms)", "823", "651", "[green]-172[/green]", "[green]✓ NO (1.1σ)[/green]")
drift_table.add_row("Vocab overlap", "—", "—", "0.74", "[yellow]⚠ YES (< 0.7)[/yellow]")

console2.print(Panel(drift_table, title="[bold]modeldiff — Drift Report[/bold]", border_style="blue"))
console2.print()

fp_table = Table(title="Model Fingerprints", show_lines=True)
fp_table.add_column("Dimension", style="bold")
fp_table.add_column("gpt-4-0613", justify="right")
fp_table.add_column("gpt-4-1106", justify="right")
fp_table.add_column("Delta", justify="right")

fp_table.add_row("Verbosity", "0.71", "0.49", "[red]-0.22[/red]")
fp_table.add_row("Refusal rate", "0.04", "0.12", "[red]+0.08[/red]")
fp_table.add_row("Vocabulary richness", "0.82", "0.76", "[yellow]-0.06[/yellow]")
fp_table.add_row("Length consistency", "0.88", "0.91", "[green]+0.03[/green]")
fp_table.add_row("Formality", "0.65", "0.58", "[yellow]-0.07[/yellow]")
fp_table.add_row("Error rate", "0.00", "0.04", "[red]+0.04[/red]")
fp_table.add_row("[bold]Euclidean distance[/bold]", "", "", "[bold yellow]0.31[/bold yellow]")

console2.print(fp_table)

svg2 = console2.export_svg(title="modeldiff drift")
with open("/data/users/zacharie/repogen/modeldiff/assets/drift_analysis.svg", "w") as f:
    f.write(svg2)
print(f"drift_analysis.svg: {len(svg2):,} bytes")
