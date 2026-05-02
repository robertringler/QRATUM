"""Utilities to discover and categorize figure artifacts from a pipeline run.

Functions:
 - find_figures(run_root): return list of image Paths
 - categorize_figures(paths): heuristic categories by filename
"""

from pathlib import Path
from typing import Dict, List

IMG_EXTS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps"}


def find_figures(run_root: Path) -> List[Path]:
    run_root = Path(run_root)
    figs: List[Path] = []
    if not run_root.exists():
        return figs
    for p in run_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            figs.append(p)
    # sort for determinism
    figs.sort(key=lambda p: p.name)
    return figs


def categorize_figures(paths: List[Path]) -> Dict[str, List[Path]]:
    cats: Dict[str, List[Path]] = {}
    for p in paths:
        name = p.name.lower()
        if "d2" in name or "msd" in name:
            k = "transport"
        elif "gamma" in name or "sweep" in name:
            k = "gamma_sweep"
        elif "otoc" in name:
            k = "otoc"
        elif "logical" in name or "fid" in name:
            k = "logical"
        else:
            k = "other"
        cats.setdefault(k, []).append(p)
    return cats
