"""Index legacy raw snippets into a manifest for later promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

RAW_SNIPPETS_DIR = Path(__file__).resolve().parent.parent / "legacy" / "raw_snippets"
MANIFEST_PATH = RAW_SNIPPETS_DIR / "manifest.json"


def build_manifest() -> Dict[str, List[str]]:
    """Create a manifest of legacy raw snippets grouped by extension."""
    manifest: Dict[str, List[str]] = {}
    for snippet_path in sorted(RAW_SNIPPETS_DIR.glob("**/*")):
        if snippet_path.is_file():
            ext = snippet_path.suffix.lstrip(".") or "noext"
            manifest.setdefault(ext, []).append(snippet_path.name)
    return manifest


def save_manifest(manifest: Dict[str, List[str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    manifest = build_manifest()
    save_manifest(manifest)


if __name__ == "__main__":
    main()
