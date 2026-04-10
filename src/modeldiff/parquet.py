"""Lightweight columnar table inspired by Parquet for snapshot serialization."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class Column:
    """A single typed column."""

    name: str
    dtype: str  # "str", "int", "float", "bool", "json"
    values: List[Any] = field(default_factory=list)


class ParquetTable:
    """In-memory columnar table with CSV/JSON export."""

    def __init__(self, columns: Optional[List[Column]] = None) -> None:
        self._columns: List[Column] = list(columns) if columns else []

    # ---- column / row mutation ----

    def add_column(self, name: str, dtype: str, values: List[Any]) -> None:
        if self._columns:
            expected = len(self._columns[0].values)
            if len(values) != expected:
                raise ValueError(
                    f"Column '{name}' has {len(values)} values, expected {expected}"
                )
        self._columns.append(Column(name=name, dtype=dtype, values=list(values)))

    def add_row(self, row: Dict[str, Any]) -> None:
        if not self._columns:
            for key, val in row.items():
                dtype = _infer_dtype(val)
                self._columns.append(Column(name=key, dtype=dtype, values=[val]))
            return
        for col in self._columns:
            col.values.append(row.get(col.name))

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": [
                {"name": c.name, "dtype": c.dtype, "values": c.values}
                for c in self._columns
            ]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParquetTable":
        columns = [
            Column(name=c["name"], dtype=c["dtype"], values=c["values"])
            for c in d.get("columns", [])
        ]
        return cls(columns)

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([c.name for c in self._columns])
        rows, _ = self.shape
        for i in range(rows):
            writer.writerow([c.values[i] for c in self._columns])
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        records = self._to_records()
        text = json.dumps(records, indent=2, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    # ---- query helpers ----

    def filter(self, column: str, predicate: Callable[[Any], bool]) -> "ParquetTable":
        col_idx = self._col_index(column)
        keep = [i for i, v in enumerate(self._columns[col_idx].values) if predicate(v)]
        new_cols = [
            Column(name=c.name, dtype=c.dtype, values=[c.values[i] for i in keep])
            for c in self._columns
        ]
        return ParquetTable(new_cols)

    def select(self, columns: List[str]) -> "ParquetTable":
        selected = [self._columns[self._col_index(n)] for n in columns]
        new_cols = [
            Column(name=c.name, dtype=c.dtype, values=list(c.values))
            for c in selected
        ]
        return ParquetTable(new_cols)

    @property
    def shape(self) -> Tuple[int, int]:
        rows = len(self._columns[0].values) if self._columns else 0
        return rows, len(self._columns)

    def describe(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for c in self._columns:
            nums = [v for v in c.values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not nums:
                continue
            total = sum(nums)
            n = len(nums)
            mean = total / n
            sorted_nums = sorted(nums)
            mid = n // 2
            median = (
                sorted_nums[mid]
                if n % 2
                else (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
            )
            stats[c.name] = {
                "count": n,
                "mean": mean,
                "min": sorted_nums[0],
                "max": sorted_nums[-1],
                "median": median,
            }
        return stats

    # ---- internals ----

    def _col_index(self, name: str) -> int:
        for i, c in enumerate(self._columns):
            if c.name == name:
                return i
        raise KeyError(f"Column '{name}' not found")

    def _to_records(self) -> List[Dict[str, Any]]:
        rows, _ = self.shape
        return [
            {c.name: c.values[i] for c in self._columns}
            for i in range(rows)
        ]


# ---- module-level helpers ----


def snapshot_to_table(snapshot: Any) -> ParquetTable:
    """Convert a modeldiff Snapshot (or its dict form) to a :class:`ParquetTable`.

    Accepts either a ``Snapshot`` object (with a ``to_dict`` method) or a plain
    dictionary with the standard modeldiff schema.
    """
    if hasattr(snapshot, "to_dict"):
        data = snapshot.to_dict()
    elif isinstance(snapshot, dict):
        data = snapshot
    else:
        raise TypeError("Expected a Snapshot or dict")

    table = ParquetTable()
    model = data.get("model_name", "")
    for resp in data.get("responses", []):
        prompt_data = resp.get("prompt", {})
        table.add_row(
            {
                "model": model,
                "prompt": prompt_data.get("text", ""),
                "category": prompt_data.get("category", ""),
                "output": resp.get("output", ""),
                "latency_ms": resp.get("latency_ms", 0.0),
                "token_count": resp.get("token_count", 0),
                "error": resp.get("error"),
            }
        )
    return table


def merge_tables(tables: List[ParquetTable]) -> ParquetTable:
    """Concatenate rows from multiple tables with matching schemas."""
    if not tables:
        return ParquetTable()
    base = tables[0]
    col_names = [c.name for c in base._columns]
    merged_values: Dict[str, List[Any]] = {n: list(base._columns[i].values) for i, n in enumerate(col_names)}
    dtypes = {c.name: c.dtype for c in base._columns}
    for t in tables[1:]:
        for c in t._columns:
            if c.name in merged_values:
                merged_values[c.name].extend(c.values)
    new_cols = [
        Column(name=n, dtype=dtypes[n], values=merged_values[n]) for n in col_names
    ]
    return ParquetTable(new_cols)


def format_table(table: ParquetTable, max_rows: int = 20) -> str:
    """Pretty-print a table."""
    if not table._columns:
        return "(empty table)"
    names = [c.name for c in table._columns]
    rows, cols = table.shape
    display_rows = min(rows, max_rows)

    # Compute column widths
    widths = [len(n) for n in names]
    for i in range(display_rows):
        for j, c in enumerate(table._columns):
            widths[j] = max(widths[j], len(str(c.values[i])[:40]))

    def _fmt_row(vals: Sequence[str]) -> str:
        return " | ".join(str(v).ljust(widths[k])[:40] for k, v in enumerate(vals))

    lines = [_fmt_row(names), "-+-".join("-" * w for w in widths)]
    for i in range(display_rows):
        lines.append(_fmt_row([str(c.values[i]) for c in table._columns]))
    if rows > max_rows:
        lines.append(f"... ({rows - max_rows} more rows)")
    return "\n".join(lines)


# ---- private helpers ----


def _infer_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"
