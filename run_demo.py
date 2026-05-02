"""Demo runner: aggregates repository metrics, runs lightweight analyses and creates visualizations.

Author: Robert Ringler (Independent Researcher)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    # Collect known metric files (best-effort)
    metrics_files = [
        ROOT / "all_metrics.json",
        ROOT / "benchmark_results.json",
        ROOT / "aggregated_data.json",
        ROOT / "aggregated_data.csv",
    ]

    summaries = {}
    for p in metrics_files:
        if p.exists():
            if p.suffix == ".csv":
                try:
                    df = pd.read_csv(p)
                    summaries[p.name] = {"rows": len(df), "columns": list(df.columns)[:10]}
                except Exception as e:
                    summaries[p.name] = {"error": str(e)}
            else:
                data = load_json(p)
                summaries[p.name] = {"present": data is not None}

    # Emit TAG=VALUE summary lines
    metrics_out = {}
    for k, v in summaries.items():
        tag = k.replace(".", "_").upper()
        metrics_out[tag] = v
        print(f"{tag}=present")

    # Quick numerical aggregation: look for numeric fields in all_metrics.json
    am = load_json(ROOT / "all_metrics.json") or {}
    numeric_summary = {}

    def collect_nums(obj, prefix=""):
        if isinstance(obj, dict):
            for kk, vv in obj.items():
                collect_nums(vv, f"{prefix}{kk}.")
        elif isinstance(obj, list):
            for i, vv in enumerate(obj[:50]):
                collect_nums(vv, f"{prefix}{i}.")
        elif isinstance(obj, (int, float)):
            numeric_summary[prefix.rstrip(".")] = obj

    collect_nums(am)
    # Save numeric summary
    with open(OUT / "numeric_summary.json", "w", encoding="utf-8") as f:
        json.dump(numeric_summary, f, indent=2)

    # If aggregated_data.csv exists, make a few plots
    csvp = ROOT / "aggregated_data.csv"
    stats_summary = {}
    if csvp.exists():
        try:
            df = pd.read_csv(csvp)
            # pick numeric columns
            numcols = df.select_dtypes(include=[np.number]).columns.tolist()

            # limit to first 8 numeric columns for speed
            numcols = numcols[:8]

            def bootstrap_ci(series, n_boot=1000, alpha=0.05):
                arr = series.dropna().values
                if len(arr) == 0:
                    return (None, None)
                idx = np.random.randint(0, len(arr), size=(n_boot, len(arr)))
                means = np.mean(arr[idx], axis=1)
                lo = np.percentile(means, 100 * (alpha / 2))
                hi = np.percentile(means, 100 * (1 - alpha / 2))
                return (float(lo), float(hi))

            for c in numcols:
                s = df[c].dropna()
                if s.size == 0:
                    continue
                mean = float(s.mean())
                std = float(s.std())
                median = float(s.median())
                count = int(s.count())
                lo, hi = bootstrap_ci(s)
                # compute bootstrap p-value for null mean == 0 (vectorized)
                arr = s.values
                nboot = 1000
                if len(arr) > 0:
                    boots = np.mean(
                        np.random.choice(arr, size=(nboot, len(arr)), replace=True), axis=1
                    )
                    pval = float(np.mean(np.abs(boots) >= abs(mean)))
                else:
                    pval = None

                stats_summary[c] = dict(
                    mean=mean,
                    std=std,
                    median=median,
                    count=count,
                    ci95=(lo, hi),
                    p_mean_bootstrap=pval,
                )

                # hist plot (lightweight matplotlib; sample if dataset is large)
                try:
                    if count > 0:
                        data = s.values
                        max_plot = 200000
                        if data.size > max_plot:
                            # sample without replacement for plotting
                            data = np.random.choice(data, size=max_plot, replace=False)
                        plt.figure(figsize=(6, 3))
                        plt.hist(data, bins=50, color="#4C72B0", alpha=0.8)
                        plt.title(f"Histogram: {c}")
                        plt.tight_layout()
                        plt.savefig(OUT / f"hist_{c}.png")
                        plt.close()
                except Exception:
                    pass

            # pairplot only when we have at least 2 numeric columns and > 10 complete rows
            if len(numcols) >= 2:
                sub = df[numcols].dropna()
                if sub.shape[0] > 10:
                    try:
                        sample_n = min(len(sub), 1000)
                        sns.pairplot(sub.sample(sample_n))
                        plt.suptitle("Pairwise numeric relationships")
                        plt.savefig(OUT / "pairplot.png")
                        plt.close()
                    except Exception as e:
                        print(f"PAIRPLOT_ERROR={str(e)}")

            # compute pairwise bootstrap correlations and simple linear regression fits for first 4 columns
            correlations_summary = {}
            paircols = numcols[:4]
            for i in range(len(paircols)):
                for j in range(i + 1, len(paircols)):
                    a = df[paircols[i]].dropna()
                    b = df[paircols[j]].dropna()
                    # align on index intersection
                    idx = a.index.intersection(b.index)
                    a2 = a.loc[idx].values
                    b2 = b.loc[idx].values
                    if len(a2) < 5:
                        continue
                    # Pearson r
                    r = float(np.corrcoef(a2, b2)[0, 1])
                    # bootstrap for CI and p-value
                    nboot = 1000
                    rboots = []
                    for _ in range(nboot):
                        sel = np.random.randint(0, len(a2), size=len(a2))
                        rboots.append(np.corrcoef(a2[sel], b2[sel])[0, 1])
                    rboots = np.array(rboots)
                    rlo = float(np.percentile(rboots, 2.5))
                    rhi = float(np.percentile(rboots, 97.5))
                    pr = float(np.mean(np.abs(rboots) >= abs(r)))
                    # linear regression (least squares)
                    try:
                        coeffs = np.polyfit(a2, b2, 1)
                        slope, intercept = float(coeffs[0]), float(coeffs[1])
                        pred = slope * a2 + intercept
                        ss_res = float(((b2 - pred) ** 2).sum())
                        ss_tot = float(((b2 - b2.mean()) ** 2).sum())
                        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else None
                    except Exception:
                        slope = intercept = r2 = None

                    key = f"{paircols[i]}__vs__{paircols[j]}"
                    correlations_summary[key] = dict(
                        r=r,
                        r_ci95=(rlo, rhi),
                        p_r_bootstrap=pr,
                        slope=slope,
                        intercept=intercept,
                        r2=r2,
                    )

            if correlations_summary:
                with open(OUT / "correlations_summary.json", "w", encoding="utf-8") as f:
                    json.dump(correlations_summary, f, indent=2)
        except Exception as e:
            print(f"CSV_PLOT_ERROR={str(e)}")

    # If benchmark_results.json exists, create simple bar chart
    br = load_json(ROOT / "benchmark_results.json")
    if isinstance(br, dict):
        # try to find numeric metrics keyed by benchmark name
        keys = []
        vals = []
        for k, v in br.items():
            if isinstance(v, (int, float)):
                keys.append(k)
                vals.append(v)
        if keys:
            plt.figure()
            sns.barplot(x=vals, y=keys)
            plt.title("Benchmark numeric metrics")
            plt.tight_layout()
            plt.savefig(OUT / "benchmarks.png")
            plt.close()

    # Basic validation checks (example): assert that certain files exist
    checks = {
        "all_metrics.json_exists": (ROOT / "all_metrics.json").exists(),
        "benchmark_results.json_exists": (ROOT / "benchmark_results.json").exists(),
    }
    with open(OUT / "checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    # Summary printed as TAG=VALUE for downstream parsing
    print(f"DEMO_OUTPUT_DIR={OUT}")
    print(f"NUMERIC_KEYS={len(numeric_summary)}")

    # Save statistical summary
    if stats_summary:
        with open(OUT / "stats_summary.json", "w", encoding="utf-8") as f:
            json.dump(stats_summary, f, indent=2)

        # Create a simple findings markdown file
        md = []
        md.append("# Demo Findings and Key Results")
        md.append("\n")
        md.append("Author: Robert Ringler (Independent Researcher)")
        md.append("\n")
        md.append(
            "This file summarizes simple statistical analyses computed from `aggregated_data.csv` where available."
        )
        md.append("\n")
        for col, s in stats_summary.items():
            md.append(f"## {col}")
            md.append(f'- count: {s["count"]}')
            md.append(f'- mean: {s["mean"]:.6g}')
            md.append(f'- std: {s["std"]:.6g}')
            lo, hi = s["ci95"]
            if lo is not None:
                md.append(f"- 95% bootstrap CI (mean): [{lo:.6g}, {hi:.6g}]")
            md.append("\n")

        with open(OUT / "findings.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md))


if __name__ == "__main__":
    main()
