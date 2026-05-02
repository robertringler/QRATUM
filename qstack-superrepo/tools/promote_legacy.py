"""Promote legacy snippets into structured packages using the registry."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

RAW_SNIPPETS_DIR = Path(__file__).resolve().parent.parent / "legacy" / "raw_snippets"
IMPORTED_DIR = Path(__file__).resolve().parent.parent / "legacy" / "imported_repos"


def promote_snippets(target_dir: Path) -> Iterable[Path]:
    """Copy raw snippets into a target directory for manual curation."""

    target_dir.mkdir(parents=True, exist_ok=True)
    for snippet in RAW_SNIPPETS_DIR.glob("**/*"):
        if snippet.is_file():
            destination = target_dir / snippet.name
            shutil.copy2(snippet, destination)
            yield destination


def main() -> None:
    for promoted in promote_snippets(IMPORTED_DIR):
        print(f"Promoted {promoted.name} -> {promoted}")


if __name__ == "__main__":
    main()
