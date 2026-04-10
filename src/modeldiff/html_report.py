"""HTML diff report export."""

from __future__ import annotations

import html
from pathlib import Path

from modeldiff._types import ChangeType, DiffReport


def _escape_html(text: str) -> str:
    """HTML-escape text to prevent XSS."""
    return html.escape(text, quote=True)


def format_html(diff_report: DiffReport) -> str:
    """Render a DiffReport as a standalone HTML page."""
    total = len(diff_report.entries)
    identical = diff_report.n_identical
    changed = diff_report.n_changes
    errors = sum(
        1 for e in diff_report.entries if e.change_type == ChangeType.ERROR
    )

    model_a = _escape_html(diff_report.model_a)
    model_b = _escape_html(diff_report.model_b)

    rows: list[str] = []
    for i, entry in enumerate(diff_report.entries, 1):
        if entry.change_type == ChangeType.IDENTICAL:
            row_class = "identical"
        elif entry.change_type == ChangeType.ERROR:
            row_class = "error"
        else:
            row_class = "different"

        prompt_text = _escape_html(entry.prompt.text[:200])
        out_a = _escape_html(entry.output_a[:500])
        out_b = _escape_html(entry.output_b[:500])
        change = _escape_html(entry.change_type.value)
        severity = _escape_html(entry.severity.value)
        desc = _escape_html(entry.description)

        rows.append(
            f'<tr class="{row_class}">'
            f"<td>{i}</td>"
            f"<td>{prompt_text}</td>"
            f"<td>{change}</td>"
            f"<td>{severity}</td>"
            f"<td>{desc}</td>"
            f'<td class="output">{out_a}</td>'
            f'<td class="output">{out_b}</td>'
            f"</tr>"
        )

    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>modeldiff report — {model_a} vs {model_b}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  margin: 0; padding: 20px; background: #f5f5f5; color: #333;
}}
h1 {{ margin-top: 0; }}
.summary {{
  display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px;
}}
.summary .card {{
  background: #fff; border-radius: 8px; padding: 16px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 120px;
}}
.card .label {{ font-size: 0.85em; color: #666; }}
.card .value {{ font-size: 1.6em; font-weight: bold; }}
table {{
  width: 100%; border-collapse: collapse; background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px;
  overflow: hidden;
}}
th, td {{
  padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee;
  vertical-align: top;
}}
th {{ background: #fafafa; font-weight: 600; position: sticky; top: 0; }}
.output {{ max-width: 300px; white-space: pre-wrap; word-break: break-word; font-size: 0.85em; }}
tr.identical {{ background: #e6ffe6; }}
tr.different {{ background: #ffe6e6; }}
tr.error {{ background: #fff8e1; }}
</style>
</head>
<body>
<h1>modeldiff report</h1>
<p><strong>{model_a}</strong> vs <strong>{model_b}</strong></p>
<div class="summary">
  <div class="card"><div class="label">Total prompts</div><div class="value">{total}</div></div>
  <div class="card"><div class="label">Identical</div><div class="value">{identical}</div></div>
  <div class="card"><div class="label">Different</div><div class="value">{changed}</div></div>
  <div class="card"><div class="label">Errors</div><div class="value">{errors}</div></div>
  <div class="card"><div class="label">Change rate</div><div class="value">{diff_report.change_rate:.0%}</div></div>
  <div class="card"><div class="label">Regression score</div><div class="value">{diff_report.regression_score:.2f}</div></div>
</div>
<table>
<thead>
<tr>
  <th>#</th>
  <th>Prompt</th>
  <th>Change type</th>
  <th>Severity</th>
  <th>Description</th>
  <th>Response A ({model_a})</th>
  <th>Response B ({model_b})</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>"""


def save_html(diff_report: DiffReport, path: str | Path) -> None:
    """Write an HTML diff report to *path*."""
    Path(path).write_text(format_html(diff_report), encoding="utf-8")
