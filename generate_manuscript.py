"""Generate manuscript artifacts (figure includes, table exports, and minimal LaTeX chapters) from a pipeline run.

Usage: python -m automation.generate_manuscript --run-root <path> --out-dir <manuscript/auto>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .auto_figures import categorize_figures, find_figures
from .auto_tables import copy_tables_to, find_tables


def generate(run_root: Path, out_dir: Path) -> Path:
    run_root = Path(run_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = find_figures(run_root)
    cats = categorize_figures(figs)
    tables = find_tables(run_root)
    tables_dir = out_dir / "tables"
    copy_tables_to(tables_dir, tables)

    # Create LaTeX figure includes
    fig_tex = out_dir / "auto_figures.tex"
    with fig_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated figure includes\n")
        for cat, paths in cats.items():
            f.write("\n% Category: %s\n" % cat)
            for p in paths:
                # relative path from manuscript
                rel = (
                    Path(p).resolve().relative_to(Path.cwd().resolve())
                    if Path(p).is_absolute()
                    else p
                )
                esc = str(rel).replace("\\", "/")
                caption = p.stem.replace("_", "\\_")
                f.write(
                    "\\begin{figure}[ht]\\centering\\includegraphics[width=0.8\\textwidth]{%s}\\caption{%s}\\end{figure}\n"
                    % (esc, caption)
                )

    # Create LaTeX table includes (simple verbatim input)
    tab_tex = out_dir / "auto_tables.tex"
    with tab_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated table includes\n")
        for t in sorted(tables, key=lambda p: p.name):
            rel = (
                Path(t).resolve().relative_to(Path.cwd().resolve()) if Path(t).is_absolute() else t
            )
            esc = str(rel).replace("\\", "/")
            f.write("\\input{%s}\n" % esc)

    # Write manifest
    manifest = {
        "run_root": str(run_root),
        "figures": [str(p) for p in figs],
        "figure_categories": {k: [str(x) for x in v] for k, v in cats.items()},
        "tables": [str(p) for p in tables],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out-dir", default="manuscript/auto")
    args = ap.parse_args()
    out = generate(Path(args.run_root), Path(args.out_dir))
    print(f"Generated manuscript assets in: {out}")


if __name__ == "__main__":
    main()
