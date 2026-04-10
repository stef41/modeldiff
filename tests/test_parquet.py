"""Tests for modeldiff.parquet module."""

import json

import pytest

from modeldiff.parquet import (
    Column,
    ParquetTable,
    format_table,
    merge_tables,
    snapshot_to_table,
)

# --- Column ---

def test_column_fields():
    c = Column("x", "int", [1, 2, 3])
    assert c.name == "x"
    assert c.dtype == "int"
    assert c.values == [1, 2, 3]


# --- ParquetTable basics ---

def test_empty_table():
    t = ParquetTable()
    assert t.shape == (0, 0)


def test_add_column():
    t = ParquetTable()
    t.add_column("a", "int", [1, 2, 3])
    t.add_column("b", "str", ["x", "y", "z"])
    assert t.shape == (3, 2)


def test_add_column_length_mismatch():
    t = ParquetTable()
    t.add_column("a", "int", [1, 2])
    with pytest.raises(ValueError):
        t.add_column("b", "str", ["x"])


def test_add_row():
    t = ParquetTable()
    t.add_row({"name": "alice", "age": 30})
    t.add_row({"name": "bob", "age": 25})
    assert t.shape == (2, 2)


# --- serialization round-trip ---

def test_to_dict_from_dict():
    t = ParquetTable()
    t.add_column("x", "int", [10, 20])
    t.add_column("y", "str", ["a", "b"])
    d = t.to_dict()
    t2 = ParquetTable.from_dict(d)
    assert t2.shape == t.shape
    assert t2.to_dict() == d


def test_to_csv():
    t = ParquetTable()
    t.add_column("a", "int", [1, 2])
    t.add_column("b", "str", ["x", "y"])
    csv_text = t.to_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1,x"


def test_to_csv_file(tmp_path):
    t = ParquetTable()
    t.add_column("v", "int", [5])
    path = tmp_path / "out.csv"
    t.to_csv(path)
    assert path.read_text().startswith("v")


def test_to_json():
    t = ParquetTable()
    t.add_column("k", "str", ["hello"])
    records = json.loads(t.to_json())
    assert records == [{"k": "hello"}]


def test_to_json_file(tmp_path):
    t = ParquetTable()
    t.add_column("k", "int", [1])
    path = tmp_path / "out.json"
    t.to_json(path)
    assert path.exists()


# --- query helpers ---

def test_filter():
    t = ParquetTable()
    t.add_column("val", "int", [1, 2, 3, 4, 5])
    t.add_column("label", "str", ["a", "b", "c", "d", "e"])
    filtered = t.filter("val", lambda v: v > 3)
    assert filtered.shape == (2, 2)


def test_select():
    t = ParquetTable()
    t.add_column("a", "int", [1])
    t.add_column("b", "int", [2])
    t.add_column("c", "int", [3])
    s = t.select(["a", "c"])
    assert s.shape == (1, 2)
    assert [c.name for c in s._columns] == ["a", "c"]


def test_select_missing_column():
    t = ParquetTable()
    t.add_column("a", "int", [1])
    with pytest.raises(KeyError):
        t.select(["nope"])


def test_describe():
    t = ParquetTable()
    t.add_column("x", "int", [10, 20, 30])
    t.add_column("name", "str", ["a", "b", "c"])
    stats = t.describe()
    assert "x" in stats
    assert "name" not in stats
    assert stats["x"]["mean"] == pytest.approx(20.0)
    assert stats["x"]["min"] == 10
    assert stats["x"]["max"] == 30
    assert stats["x"]["median"] == 20


def test_describe_even_count():
    t = ParquetTable()
    t.add_column("v", "float", [1.0, 2.0, 3.0, 4.0])
    stats = t.describe()
    assert stats["v"]["median"] == pytest.approx(2.5)


# --- snapshot_to_table ---

def test_snapshot_to_table_dict():
    snap = {
        "model_name": "gpt-4",
        "responses": [
            {
                "prompt": {"text": "hello", "category": "general"},
                "output": "hi",
                "latency_ms": 100.0,
                "token_count": 5,
                "error": None,
            }
        ],
    }
    table = snapshot_to_table(snap)
    assert table.shape == (1, 7)


def test_snapshot_to_table_empty():
    table = snapshot_to_table({"model_name": "m", "responses": []})
    assert table.shape == (0, 0)


# --- merge_tables ---

def test_merge_tables():
    t1 = ParquetTable()
    t1.add_column("a", "int", [1, 2])
    t2 = ParquetTable()
    t2.add_column("a", "int", [3])
    merged = merge_tables([t1, t2])
    assert merged.shape == (3, 1)


def test_merge_tables_empty():
    assert merge_tables([]).shape == (0, 0)


# --- format_table ---

def test_format_table():
    t = ParquetTable()
    t.add_column("id", "int", [1, 2])
    t.add_column("name", "str", ["alice", "bob"])
    text = format_table(t)
    assert "id" in text
    assert "alice" in text


def test_format_table_empty():
    assert format_table(ParquetTable()) == "(empty table)"


def test_format_table_truncation():
    t = ParquetTable()
    t.add_column("i", "int", list(range(50)))
    text = format_table(t, max_rows=5)
    assert "more rows" in text
