"""Utilities to discover and normalize table artifacts (CSV/TSV) from pipeline runs."""

from pathlib import Path
from typing import List

TABLE_EXTS = {".csv", ".tsv", ".txt"}


def find_tables(run_root: Path) -> List[Path]:
    run_root = Path(run_root)
    tables = []
    if not run_root.exists():
        return tables
    for p in run_root.rglob("*"):
        if p.suffix.lower() in TABLE_EXTS and p.is_file():
            tables.append(p)
    tables.sort(key=lambda p: p.name)
    return tables


def copy_tables_to(dest: Path, tables: List[Path]):
    dest.mkdir(parents=True, exist_ok=True)
    for t in tables:
        try:
            target = dest / t.name
            if not target.exists():
                from shutil import copy2

                copy2(t, target)
        except Exception:
            pass
