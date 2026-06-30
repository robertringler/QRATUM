"""Bridge gene-expression matrices to XENON mechanism inference.

Implements the pipeline:

    expression matrix (genes x samples)
        -> select gene modules (co-expression clustering)
        -> construct state variables (module activities)
        -> XENON mechanism inference (Bayesian model comparison)

The loader accepts a standard processed-counts / TPM / normalized matrix
(genes as rows, samples as columns; tab- or comma-delimited), e.g. the
E-MTAB-16935 processed expression files, so real data is drop-in.

IMPORTANT — scope/honesty: module activities are *abstract* state variables, not
physical concentrations. Mapping them onto XENON's concentration observable is a
modelling choice; XENON's thermodynamic/conservation checks are therefore not
physically meaningful in this mode. What the inference recovers is the effective
regulatory relationship implied by the module dynamics (e.g. the activation ratio
of a maternal->zygotic transition), under explicitly stated assumptions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.learning.bayesian_updater import BayesianUpdater, ExperimentResult


@dataclass
class ExpressionMatrix:
    """A gene-expression matrix (genes x samples)."""

    genes: list[str]
    samples: list[str]
    X: np.ndarray  # shape (n_genes, n_samples)


@dataclass
class ModuleResult:
    """Co-expression module assignment and per-module activity profiles."""

    labels: np.ndarray  # per-gene module index, shape (n_genes,)
    samples: list[str]
    raw_profiles: np.ndarray  # mean RAW expression per module, shape (n_modules, n_samples)
    sizes: list[int]
    n_modules: int = field(default=0)


def load_expression_matrix(path: str, sep: str | None = None) -> ExpressionMatrix:
    """Load a genes x samples expression matrix from CSV/TSV.

    First row = header (first cell is the gene-id column label, rest are sample
    ids). Each subsequent row = gene id followed by numeric values. Delimiter is
    auto-detected from the header if ``sep`` is None.
    """

    with open(path, newline="") as fh:
        sample = fh.readline()
        if sep is None:
            sep = "\t" if sample.count("\t") >= sample.count(",") else ","
        fh.seek(0)
        reader = csv.reader(fh, delimiter=sep)
        header = next(reader)
        samples = header[1:]
        genes: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            genes.append(row[0])
            rows.append([float(v) if v not in ("", "NA", "NaN") else np.nan for v in row[1:]])

    X = np.asarray(rows, dtype=float)
    return ExpressionMatrix(genes=genes, samples=samples, X=X)


def _standardize_rows(X: np.ndarray) -> np.ndarray:
    """Z-score each gene's profile across samples (for pattern-based clustering)."""

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def _kmeans(Z: np.ndarray, k: int, seed: int, n_iter: int = 100) -> np.ndarray:
    """Deterministic numpy k-means (no sklearn). Returns per-row cluster labels."""

    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    # k-means++-style deterministic init: first centroid random, rest farthest.
    centroids = [Z[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.min(
            [np.sum((Z - c) ** 2, axis=1) for c in centroids], axis=0
        )
        centroids.append(Z[int(np.argmax(dists))])
    C = np.array(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.sum((Z[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = Z[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
    return labels


def select_gene_modules(expr: ExpressionMatrix, n_modules: int = 2, seed: int = 42) -> ModuleResult:
    """Cluster genes into co-expression modules by temporal/sample pattern.

    Genes are z-scored across samples (so clustering is by *pattern*, not
    magnitude), then grouped by k-means. Each module's RAW mean expression
    profile across samples is returned as its activity signature.
    """

    Z = _standardize_rows(expr.X)
    labels = _kmeans(Z, n_modules, seed)

    raw_profiles = np.zeros((n_modules, expr.X.shape[1]))
    sizes: list[int] = []
    for j in range(n_modules):
        members = expr.X[labels == j]
        sizes.append(int(members.shape[0]))
        if members.shape[0]:
            raw_profiles[j] = members.mean(axis=0)

    return ModuleResult(
        labels=labels, samples=expr.samples, raw_profiles=raw_profiles,
        sizes=sizes, n_modules=n_modules,
    )


def construct_state_variables(
    modules: ModuleResult,
    late_samples: list[str] | None = None,
) -> dict[str, float]:
    """Map module activities to XENON state variables (abstract 'concentrations').

    Returns ``{module_name: activity}`` where activity is each module's mean RAW
    expression over ``late_samples`` (or all samples if None). Activities are
    rescaled to sum to 100 so they read as a composition, matching the
    concentration scale the downstream mechanism uses.
    """

    if late_samples:
        idx = [modules.samples.index(s) for s in late_samples if s in modules.samples]
    else:
        idx = list(range(len(modules.samples)))

    activity = modules.raw_profiles[:, idx].mean(axis=1)
    activity = np.clip(activity, 0.0, None)
    total = activity.sum()
    if total > 0:
        activity = 100.0 * activity / total

    return {f"module_{j}": float(activity[j]) for j in range(modules.n_modules)}


def infer_mechanism(
    state_activities: dict[str, float],
    candidate_K: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    protein: str = "program",
    n_observations: int = 12,
    noise_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Infer the effective two-module regulatory mechanism via XENON.

    Treats the two highest-activity modules as the inactive (source) and active
    (target) states of a maternal->zygotic style transition. The observed
    activation ratio K* = active/inactive is matched against candidate mechanisms
    with different K (predicted active fraction = K/(1+K)); XENON's Bayesian
    updater returns the posterior over candidates and the recovered K.

    This reuses XENON's tested concentration-likelihood machinery; the module
    activities supply the observation instead of an SSA prediction.
    """

    # Pick the two dominant modules as inactive/active by activity.
    ordered = sorted(state_activities.items(), key=lambda kv: kv[1], reverse=True)
    (active_name, active_val), (inactive_name, inactive_val) = ordered[0], ordered[1]
    total = active_val + inactive_val
    if total <= 0:
        raise ValueError("Module activities are non-positive; cannot infer mechanism.")
    observed_active_frac = active_val / total
    observed_K = active_val / inactive_val if inactive_val > 0 else np.inf

    inactive_state = f"{protein}_inactive"
    active_state = f"{protein}_active"

    def candidate(k: float) -> BioMechanism:
        m = BioMechanism(f"K={k}")
        m.add_state(MolecularState(name=inactive_state, molecule=protein, concentration=50.0))
        m.add_state(MolecularState(name=active_state, molecule=protein, concentration=50.0))
        # Predicted active fraction at steady state for a reversible 2-state
        # system is K/(1+K); encode that directly as the predicted concentrations.
        frac = k / (1.0 + k)
        m._states[active_state].concentration = 100.0 * frac
        m._states[inactive_state].concentration = 100.0 * (1.0 - frac)
        m.add_transition(
            Transition(source=inactive_state, target=active_state, rate_constant=k,
                       reversible=True, reverse_rate=1.0)
        )
        m.posterior = 1.0 / len(candidate_K)
        return m

    candidates = [candidate(k) for k in candidate_K]

    rng = np.random.default_rng(seed)
    updater = BayesianUpdater()
    sigma = max(noise_frac * 100.0 * observed_active_frac, 1.0)
    trajectory = []
    for _ in range(n_observations):
        obs = {
            active_state: 100.0 * observed_active_frac + rng.normal(0.0, sigma),
            inactive_state: 100.0 * (1.0 - observed_active_frac) + rng.normal(0.0, sigma),
        }
        exp = ExperimentResult(
            experiment_type="concentration",
            observations=obs,
            uncertainties={active_state: sigma, inactive_state: sigma},
            conditions={"temperature": 310.0},
        )
        updater.update_mechanisms(candidates, exp)
        trajectory.append({c.name: c.posterior for c in candidates})

    final = {c.name: c.posterior for c in candidates}
    recovered = max(final, key=final.get)

    return {
        "inactive_module": inactive_name,
        "active_module": active_name,
        "observed_active_fraction": round(observed_active_frac, 4),
        "observed_K": round(float(observed_K), 4),
        "candidate_K": list(candidate_K),
        "final_posterior": {k: round(float(v), 6) for k, v in final.items()},
        "recovered_mechanism": recovered,
        "recovered_K": float(recovered.split("=")[1]),
        "posterior_trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Production ingest path: real matrices, metadata (SDRF / SRA run table),
# gene-set modules, normalization, validation, and an honest CLI.
# ---------------------------------------------------------------------------


@dataclass
class SampleMeta:
    """Parsed per-sample metadata."""

    sample: str
    timepoint_hours: float | None = None
    treatment: str | None = None  # e.g. 'polyA+' / 'polyA-'
    replicate: str | None = None
    dev_stage: str | None = None
    run: str | None = None


@dataclass
class NamedModuleResult:
    """Gene-set (marker-based) module activities."""

    module_names: list[str]
    samples: list[str]
    activities: np.ndarray  # (n_modules, n_samples)
    member_counts: dict[str, int]  # markers defined
    present_counts: dict[str, int]  # markers found in the matrix


def _parse_hours(value: str) -> float | None:
    """Extract a numeric hour value from strings like '17', '17h', '17 hour'."""

    if value is None:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m else None


def _detect_delimiter(line: str) -> str:
    return "\t" if line.count("\t") >= line.count(",") else ","


def deduplicate_genes(genes: list[str], X: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Collapse duplicate gene ids by averaging their rows (deterministic)."""

    seen: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        seen.setdefault(g, []).append(i)
    if all(len(idx) == 1 for idx in seen.values()):
        return genes, X
    new_genes: list[str] = []
    rows: list[np.ndarray] = []
    for g, idx in seen.items():
        new_genes.append(g)
        rows.append(np.nanmean(X[idx], axis=0))
    return new_genes, np.asarray(rows)


def impute_missing(X: np.ndarray) -> np.ndarray:
    """Replace NaNs with the per-gene (row) mean; rows all-NaN -> 0."""

    if not np.isnan(X).any():
        return X
    out = X.copy()
    for i in range(out.shape[0]):
        row = out[i]
        mask = np.isnan(row)
        if mask.any():
            row_mean = np.nanmean(row) if not np.isnan(row).all() else 0.0
            row[mask] = row_mean
    return out


def normalize_matrix(X: np.ndarray, method: str = "raw") -> np.ndarray:
    """Normalize an expression matrix. method in {raw, log1p, cpm, tpm}."""

    method = method.lower()
    if method == "raw" or method == "tpm":  # tpm = pass-through (already normalized)
        return X
    if method == "log1p":
        return np.log1p(np.clip(X, 0.0, None))
    if method == "cpm":
        col_sums = X.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return X / col_sums * 1e6
    raise ValueError(f"Unknown normalization method: {method!r}")


def parse_metadata(path: str) -> dict[str, SampleMeta]:
    """Parse SDRF- or SRA-run-table-style metadata into per-sample records.

    Tolerant to both ArrayExpress/BioStudies SDRF and ENA/SRA run tables: columns
    are located by fuzzy keyword matching, and polyA treatment / replicate are
    inferred from the sample/library name when not an explicit column.
    """

    with open(path, newline="") as fh:
        first = fh.readline()
        sep = _detect_delimiter(first)
        fh.seek(0)
        reader = csv.reader(fh, delimiter=sep)
        header = next(reader)
        rows = [r for r in reader if r]

    lower = [h.strip().lower() for h in header]

    def find(*keywords: str, exact: tuple[str, ...] = ()) -> int | None:
        for i, h in enumerate(lower):
            if h in exact:
                return i
        for i, h in enumerate(lower):
            if any(k in h for k in keywords):
                return i
        return None

    i_sample = find("source name", "sample name", "library name", "assay name", "run",
                    exact=("source name", "sample name", "run"))
    i_time = find("age", "time", "hour", exact=("age",))
    i_treat = find("library name", "polya", "library construction", "factor value")
    i_rep = find("replicate", "individual")
    i_stage = find("developmental", "stage")
    i_run = find("run", "ena_run", exact=("run",))

    meta: dict[str, SampleMeta] = {}
    for r in rows:
        def cell(i: int | None) -> str | None:
            return r[i].strip() if i is not None and i < len(r) and r[i].strip() else None

        sample = cell(i_sample)
        if not sample:
            continue
        treat_src = cell(i_treat) or sample or ""
        treatment = None
        if "polya+" in treat_src.lower() or "polya plus" in treat_src.lower():
            treatment = "polyA+"
        elif "polya-" in treat_src.lower() or "polya minus" in treat_src.lower():
            treatment = "polyA-"
        rep = cell(i_rep)
        if rep is None:
            m = re.search(r"rep(\d+)", sample, re.IGNORECASE)
            rep = m.group(1) if m else None
        meta[sample] = SampleMeta(
            sample=sample,
            timepoint_hours=_parse_hours(cell(i_time)),
            treatment=treatment,
            replicate=rep,
            dev_stage=cell(i_stage),
            run=cell(i_run),
        )
    return meta


def match_samples(expr_samples: list[str], meta: dict[str, SampleMeta]) -> dict[str, str]:
    """Match expression columns to metadata sample names robustly.

    Returns {expr_sample -> meta_key}. Tries exact, case-insensitive, then
    substring containment (either direction).
    """

    keys = list(meta.keys())
    norm = {k.strip().lower(): k for k in keys}
    out: dict[str, str] = {}
    for col in expr_samples:
        c = col.strip()
        if c in meta:
            out[col] = c
            continue
        if c.lower() in norm:
            out[col] = norm[c.lower()]
            continue
        cand = [k for k in keys if c.lower() in k.lower() or k.lower() in c.lower()]
        if len(cand) == 1:
            out[col] = cand[0]
    return out


def load_gene_sets(path: str) -> dict[str, list[str]]:
    """Load gene-set modules. TSV/CSV with columns (gene, module) or (module, gene).

    A header is optional; the column whose values look like module labels (few
    distinct values) is treated as the module column.
    """

    with open(path, newline="") as fh:
        first = fh.readline()
        sep = _detect_delimiter(first)
        fh.seek(0)
        rows = [r for r in csv.reader(fh, delimiter=sep) if r and len(r) >= 2]

    # Drop a header row if it looks non-data (contains 'gene'/'module').
    if rows and any("gene" in c.lower() or "module" in c.lower() for c in rows[0]):
        rows = rows[1:]

    col0 = {r[0] for r in rows}
    col1 = {r[1] for r in rows}
    # The module column has fewer distinct values.
    module_col = 1 if len(col1) <= len(col0) else 0
    gene_col = 1 - module_col

    sets: dict[str, list[str]] = {}
    for r in rows:
        sets.setdefault(r[module_col], []).append(r[gene_col])
    return sets


def score_gene_set_modules(
    expr: ExpressionMatrix, gene_sets: dict[str, list[str]], X_norm: np.ndarray
) -> NamedModuleResult:
    """Compute mean normalized activity per gene-set module across samples."""

    gene_index = {g: i for i, g in enumerate(expr.genes)}
    names = sorted(gene_sets.keys())
    activities = np.zeros((len(names), X_norm.shape[1]))
    member_counts: dict[str, int] = {}
    present_counts: dict[str, int] = {}
    for j, name in enumerate(names):
        members = gene_sets[name]
        idx = [gene_index[g] for g in members if g in gene_index]
        member_counts[name] = len(members)
        present_counts[name] = len(idx)
        if idx:
            activities[j] = X_norm[idx].mean(axis=0)
    return NamedModuleResult(
        module_names=names, samples=expr.samples, activities=activities,
        member_counts=member_counts, present_counts=present_counts,
    )


def _classify_modules(names: list[str]) -> dict[str, str]:
    """Best-effort map of module names to roles {maternal, zygotic, polya}."""

    roles: dict[str, str] = {}
    for n in names:
        low = n.lower()
        if "matern" in low:
            roles[n] = "maternal"
        elif "zygot" in low or "ega" in low or "zga" in low:
            roles[n] = "zygotic"
        elif "polya" in low or "histone" in low:
            roles[n] = "polya"
    return roles


def run_pipeline(
    matrix_path: str,
    out_dir: str,
    metadata_path: str | None = None,
    gene_sets_path: str | None = None,
    normalization: str = "log1p",
    n_modules: int = 2,
    seed: int = 42,
) -> dict:
    """End-to-end real-matrix ingest pipeline. Writes outputs to ``out_dir``.

    Refuses mechanism-recovery claims when there is insufficient temporal
    structure (<2 distinct timepoints), but still performs ingestion, module
    construction, and state-variable extraction.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # 1. Load + clean matrix.
    expr = load_expression_matrix(matrix_path)
    genes, X = deduplicate_genes(expr.genes, expr.X)
    if len(genes) != len(expr.genes):
        warnings.append(f"Collapsed {len(expr.genes) - len(genes)} duplicate gene ids by averaging.")
    if np.isnan(X).any():
        warnings.append("Imputed missing expression values with per-gene means.")
        X = impute_missing(X)
    expr = ExpressionMatrix(genes=genes, samples=expr.samples, X=X)
    X_norm = normalize_matrix(expr.X, normalization)

    # 2. Metadata + sample matching.
    meta: dict[str, SampleMeta] = {}
    sample_to_meta: dict[str, str] = {}
    if metadata_path:
        meta = parse_metadata(metadata_path)
        sample_to_meta = match_samples(expr.samples, meta)
        unmatched = [s for s in expr.samples if s not in sample_to_meta]
        if unmatched:
            warnings.append(f"{len(unmatched)}/{len(expr.samples)} samples had no metadata match: "
                            f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    # Timepoints + replicate structure (from metadata if available).
    timepoints: dict[str, float | None] = {}
    for s in expr.samples:
        mk = sample_to_meta.get(s)
        timepoints[s] = meta[mk].timepoint_hours if mk else None
    distinct_tp = sorted({t for t in timepoints.values() if t is not None})
    n_distinct_tp = len(distinct_tp)
    n_samples = len(expr.samples)

    # 3. Modules (gene-set if provided, else unsupervised k-means).
    if gene_sets_path:
        gene_sets = load_gene_sets(gene_sets_path)
        named = score_gene_set_modules(expr, gene_sets, X_norm)
        module_names = named.module_names
        module_acts = named.activities
        low_cov = [n for n in module_names if named.present_counts[n] == 0]
        if low_cov:
            warnings.append(f"Gene-set modules with zero markers found in matrix: {low_cov}")
        modules_summary = {
            "method": "gene-set",
            "modules": [
                {"name": n, "markers_defined": named.member_counts[n],
                 "markers_found": named.present_counts[n]}
                for n in module_names
            ],
        }
        roles = _classify_modules(module_names)
    else:
        km = select_gene_modules(expr, n_modules=n_modules, seed=seed)
        module_names = [f"module_{j}" for j in range(km.n_modules)]
        module_acts = km.raw_profiles
        modules_summary = {"method": "kmeans", "sizes": km.sizes}
        roles = {}

    # Write selected_modules.tsv
    with (out / "selected_modules.tsv").open("w") as fh:
        fh.write("module\trole\t" + "\t".join(expr.samples) + "\n")
        for j, n in enumerate(module_names):
            fh.write(n + "\t" + roles.get(n, "") + "\t"
                     + "\t".join(f"{v:.4f}" for v in module_acts[j]) + "\n")

    # Write normalized_expression.tsv
    with (out / "normalized_expression.tsv").open("w") as fh:
        fh.write("Gene\t" + "\t".join(expr.samples) + "\n")
        for g, row in zip(expr.genes, X_norm):
            fh.write(g + "\t" + "\t".join(f"{v:.4f}" for v in row) + "\n")

    # 4. State variables: timepoint-normalized module activity curves.
    state_rows = []
    for j, n in enumerate(module_names):
        for s_i, s in enumerate(expr.samples):
            state_rows.append((n, roles.get(n, ""), s, timepoints[s], module_acts[j, s_i]))
    with (out / "state_variables.tsv").open("w") as fh:
        fh.write("module\trole\tsample\ttimepoint_hours\tactivity\n")
        for n, role, s, tp, a in state_rows:
            fh.write(f"{n}\t{role}\t{s}\t{'' if tp is None else tp}\t{a:.4f}\n")

    # 5. Mechanism inference — only with sufficient temporal structure.
    inference: dict
    mechanism_recovery = n_distinct_tp >= 2
    if not mechanism_recovery:
        msg = ("insufficient temporal structure for mechanism recovery; "
               "ingestion-only validation")
        warnings.append(msg)
        inference = {
            "status": "REFUSED",
            "reason": msg,
            "distinct_timepoints": n_distinct_tp,
            "n_samples": n_samples,
        }
        candidate_mechanisms = {"status": "not_constructed", "reason": msg}
    else:
        # Use the latest timepoint; map maternal->inactive, zygotic->active when
        # roles are known, else fall back to highest-activity ordering.
        late_tp = distinct_tp[-1]
        late_cols = [i for i, s in enumerate(expr.samples) if timepoints[s] == late_tp]
        late_act = module_acts[:, late_cols].mean(axis=1)
        state_activities = {n: float(max(late_act[j], 0.0)) for j, n in enumerate(module_names)}

        maternal = next((n for n in module_names if roles.get(n) == "maternal"), None)
        zygotic = next((n for n in module_names if roles.get(n) == "zygotic"), None)
        if maternal and zygotic:
            state_activities = {maternal: state_activities[maternal],
                                zygotic: state_activities[zygotic]}
        if len(state_activities) < 2:
            inference = {"status": "REFUSED",
                         "reason": "fewer than two usable modules for a transition"}
            candidate_mechanisms = {"status": "not_constructed"}
            mechanism_recovery = False
        else:
            inference = infer_mechanism(state_activities, protein="program", seed=seed)
            inference["status"] = "OK"
            if n_distinct_tp < 3:
                inference["identifiability_warning"] = (
                    f"only {n_distinct_tp} timepoints; K is weakly identified — "
                    "treat the recovered mechanism as provisional")
            candidate_mechanisms = {
                "candidate_K": inference["candidate_K"],
                "predicted_active_fraction": {str(k): k / (1 + k) for k in inference["candidate_K"]},
            }

    (out / "candidate_mechanisms.json").write_text(json.dumps(candidate_mechanisms, indent=2))
    inference_public = {k: v for k, v in inference.items() if k != "posterior_trajectory"}
    (out / "xenon_inference_results.json").write_text(json.dumps(inference_public, indent=2, default=str))

    summary = {
        "matrix": matrix_path,
        "n_genes": len(expr.genes),
        "n_samples": n_samples,
        "normalization": normalization,
        "distinct_timepoints": distinct_tp,
        "modules": modules_summary,
        "mechanism_recovery": mechanism_recovery,
        "inference": inference_public,
        "warnings": warnings,
    }
    _write_report(out / "report.md", summary, meta, sample_to_meta, expr, roles, module_names,
                  module_acts, timepoints)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _write_report(path, summary, meta, sample_to_meta, expr, roles, module_names,
                  module_acts, timepoints) -> None:
    lines = []
    a = lines.append
    a("# XENON expression→mechanism ingest report\n")
    a(f"- **Matrix:** `{summary['matrix']}`")
    a(f"- **Genes × samples:** {summary['n_genes']} × {summary['n_samples']}")
    a(f"- **Normalization:** {summary['normalization']}")
    tps = summary["distinct_timepoints"]
    a(f"- **Distinct timepoints:** {len(tps)} {tps if tps else '(none parsed from metadata)'}")
    a(f"- **Mechanism recovery attempted:** {summary['mechanism_recovery']}\n")

    a("## Modules")
    a(f"- method: `{summary['modules'].get('method')}`")
    for j, n in enumerate(module_names):
        role = roles.get(n, "")
        a(f"  - **{n}**{f' ({role})' if role else ''}: "
          f"activity by sample = {[round(float(v), 3) for v in module_acts[j]]}")
    a("")

    a("## Inference")
    inf = summary["inference"]
    if inf.get("status") == "OK":
        a(f"- recovered mechanism: **{inf['recovered_mechanism']}** (K={inf['recovered_K']})")
        a(f"- observed activation ratio K* = {inf['observed_K']}")
        a(f"- posterior: {inf['final_posterior']}")
        if "identifiability_warning" in inf:
            a(f"- ⚠️ **identifiability:** {inf['identifiability_warning']}")
    else:
        a(f"- **{inf.get('status')}** — {inf.get('reason')}")
    a("")

    a("## Limitations")
    a("- Module activities are **abstract state variables, not physical concentrations**; "
      "XENON's thermodynamic/conservation checks are not physically meaningful in this mode.")
    a("- Mechanism recovery requires ≥2 distinct timepoints; with sparse timepoints K is weakly "
      "identified and any recovered mechanism is provisional.")
    if summary["warnings"]:
        a("\n## Warnings")
        for w in summary["warnings"]:
            a(f"- {w}")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI: run a real expression matrix through the XENON pipeline."""

    p = argparse.ArgumentParser(
        prog="python -m xenon.bioinformatics.expression_to_mechanism",
        description="Ingest a gene-expression matrix and run XENON mechanism inference.",
    )
    p.add_argument("--matrix", required=True, help="genes×samples expression matrix (CSV/TSV)")
    p.add_argument("--metadata", default=None, help="SDRF / SRA run-table metadata (CSV/TSV)")
    p.add_argument("--gene-sets", default=None, help="gene-set module definitions (gene,module)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--normalization", default="log1p",
                   choices=["raw", "log1p", "cpm", "tpm"], help="normalization method")
    p.add_argument("--n-modules", type=int, default=2,
                   help="modules for unsupervised k-means fallback (ignored with --gene-sets)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    summary = run_pipeline(
        matrix_path=args.matrix, out_dir=args.out, metadata_path=args.metadata,
        gene_sets_path=args.gene_sets, normalization=args.normalization,
        n_modules=args.n_modules, seed=args.seed,
    )
    print(f"Wrote outputs to {args.out}/")
    print(f"genes×samples: {summary['n_genes']}×{summary['n_samples']} | "
          f"timepoints: {summary['distinct_timepoints']} | "
          f"mechanism_recovery: {summary['mechanism_recovery']}")
    inf = summary["inference"]
    if inf.get("status") == "OK":
        print(f"recovered: {inf['recovered_mechanism']} (K={inf['recovered_K']}, "
              f"observed K*={inf['observed_K']})")
    else:
        print(f"inference: {inf.get('status')} — {inf.get('reason')}")
    for w in summary["warnings"]:
        print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
