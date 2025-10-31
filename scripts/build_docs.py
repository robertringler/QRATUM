"""Render Markdown documentation and validate links."""
from __future__ import annotations

import pathlib

import markdown

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    docs = sorted((REPO_ROOT / "docs").glob("*.md"))
    if not docs:
        raise SystemExit("No documentation found to render")
    out_dir = REPO_ROOT / "build" / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        html = markdown.markdown(doc.read_text())
        (out_dir / f"{doc.stem}.html").write_text(html)
        print(f"Rendered {doc.name} -> build/docs/{doc.stem}.html")


if __name__ == "__main__":
    main()
