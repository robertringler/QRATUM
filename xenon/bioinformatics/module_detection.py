"""Production-grade gene-module detection for the XENON expression pipeline.

Replaces placeholder k-means with an auditable module-discovery layer offering
five modes, per-module quality diagnostics, and brutal-honesty warnings so that
downstream mechanism inference is never presented as more defensible than the
module structure supports.

Modes:
    gene_sets            marker-based (most interpretable; biologically imposed)
    correlation_modules  co-expression graph (threshold + connected components)
    silhouette_kmeans    automatic k selection via silhouette
    hybrid               markers first, cluster the remaining genes
    manual               user-supplied gene -> module mapping

Self-contained (numpy only); takes plain arrays so it has no import dependency on
the rest of the pipeline (avoids circular imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WEAK_COHERENCE = 0.30       # mean within-module correlation below this => weak
FLAT_TREND = 0.15           # timepoint dynamic-range ratio below this => flat
UNSTABLE_CV = 0.50          # mean replicate CV above this => unstable
DROPOUT_FRAC = 0.50         # fraction of zeros above this => dropout warning


@dataclass
class Module:
    name: str
    role: str | None
    genes: list[str]
    activity: np.ndarray  # per-sample activity (mean normalized expression)
    quality: dict = field(default_factory=dict)


@dataclass
class ModuleDetection:
    mode: str
    modules: list[Module]
    gene_labels: dict[str, str]
    diagnostics: dict
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Low-level numpy primitives
# --------------------------------------------------------------------------- #

def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def _kmeans(Z: np.ndarray, k: int, seed: int, n_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    centroids = [Z[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.min([np.sum((Z - c) ** 2, axis=1) for c in centroids], axis=0)
        centroids.append(Z[int(np.argmax(dists))])
    C = np.array(centroids)
    labels = np.full(n, -1)
    for _ in range(n_iter):
        d = np.sum((Z[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = Z[labels == j]
            if len(m):
                C[j] = m.mean(axis=0)
    return labels


def silhouette_score(Z: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over points, numpy implementation. NaN-safe."""

    uniq = [u for u in np.unique(labels) if u >= 0]
    if len(uniq) < 2:
        return float("nan")
    mask = labels >= 0
    Zm, lm = Z[mask], labels[mask]
    D = np.sqrt(np.maximum(np.sum((Zm[:, None, :] - Zm[None, :, :]) ** 2, axis=2), 0.0))
    scores = []
    for i in range(len(Zm)):
        same = lm == lm[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for u in uniq:
            if u == lm[i]:
                continue
            other = lm == u
            if other.any():
                b = min(b, D[i, other].mean())
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation among rows (genes), NaN-safe."""

    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.zeros((X.shape[0], X.shape[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X)  # zero-variance genes -> NaN, handled below
    return np.nan_to_num(C, nan=0.0)


def _within_between(C: np.ndarray, idx_by_module: list[np.ndarray]) -> tuple[list[float], float]:
    """Per-module within-correlation and the global mean between-module correlation."""

    within = []
    for idx in idx_by_module:
        if len(idx) >= 2:
            sub = C[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), k=1)
            within.append(float(sub[iu].mean()))
        else:
            within.append(float("nan"))
    between_vals = []
    for a in range(len(idx_by_module)):
        for b in range(a + 1, len(idx_by_module)):
            ia, ib = idx_by_module[a], idx_by_module[b]
            if len(ia) and len(ib):
                between_vals.append(float(C[np.ix_(ia, ib)].mean()))
    between = float(np.mean(between_vals)) if between_vals else 0.0
    return within, between


def _timepoint_trend(activity: np.ndarray, timepoints: list[float | None], samples: list[str]) -> float:
    """Dynamic-range ratio of module activity across distinct timepoints.

    (max - min) / (mean + eps) over per-timepoint mean activity. ~0 => flat.
    """

    tp = [(t, a) for t, a in zip(timepoints, activity) if t is not None]
    if len({t for t, _ in tp}) < 2:
        return float("nan")
    by_tp: dict[float, list[float]] = {}
    for t, a in tp:
        by_tp.setdefault(t, []).append(a)
    means = np.array([np.mean(v) for _, v in sorted(by_tp.items())])
    denom = abs(means.mean()) + 1e-9
    return float((means.max() - means.min()) / denom)


def _replicate_stability(
    activity: np.ndarray, timepoints: list[float | None], replicates: list[str | None]
) -> float:
    """Mean coefficient-of-variation of activity across replicates within a timepoint.

    Returns 1 - mean_CV (clipped to [0,1]); NaN if no timepoint has >=2 replicates.
    """

    groups: dict[float, list[float]] = {}
    for a, t, r in zip(activity, timepoints, replicates):
        if t is not None:
            groups.setdefault(t, []).append(a)
    cvs = []
    for _, vals in groups.items():
        if len(vals) >= 2:
            arr = np.array(vals)
            m = abs(arr.mean()) + 1e-9
            cvs.append(float(arr.std() / m))
    if not cvs:
        return float("nan")
    return float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Detector modes
# --------------------------------------------------------------------------- #

def _union_find_components(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def _correlation_modules_labels(X_norm: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    C = _corr_matrix(X_norm)
    adj = C >= threshold
    np.fill_diagonal(adj, False)
    comp = _union_find_components(adj)
    # Relabel to 0..K-1, dropping components below min_size to -1 (unassigned).
    labels = np.full(len(comp), -1)
    next_label = 0
    for root in np.unique(comp):
        members = np.where(comp == root)[0]
        if len(members) >= min_size:
            labels[members] = next_label
            next_label += 1
    return labels


def _silhouette_kmeans_labels(
    X_norm: np.ndarray, min_k: int, max_k: int, seed: int, min_size: int
) -> tuple[np.ndarray, dict]:
    Z = _zscore_rows(X_norm)
    n = Z.shape[0]
    scores: dict[int, float] = {}
    best_k, best_labels, best_score = None, None, -np.inf
    for k in range(min_k, min(max_k, n) + 1):
        labels = _kmeans(Z, k, seed)
        s = silhouette_score(Z, labels)
        scores[k] = round(s, 4) if not np.isnan(s) else None
        if not np.isnan(s) and s > best_score:
            best_k, best_labels, best_score = k, labels, s
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1
    # Drop tiny modules to unassigned.
    out = np.full(n, -1)
    nxt = 0
    for j in np.unique(best_labels):
        members = np.where(best_labels == j)[0]
        if len(members) >= min_size:
            out[members] = nxt
            nxt += 1
    return out, {"selected_k": best_k, "silhouette": round(float(best_score), 4), "k_scores": scores}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _module_activity(X_norm: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return X_norm[idx].mean(axis=0) if len(idx) else np.zeros(X_norm.shape[1])


def _assign_role_by_markers(module_genes: set[str], gene_sets: dict[str, list[str]]) -> tuple[str | None, float]:
    best_role, best_overlap = None, 0.0
    for role, markers in gene_sets.items():
        mk = set(markers)
        if not mk:
            continue
        overlap = len(module_genes & mk) / len(module_genes) if module_genes else 0.0
        if overlap > best_overlap:
            best_role, best_overlap = role, overlap
    return best_role, best_overlap


def _role_from_trend(activity: np.ndarray, timepoints: list[float | None]) -> str | None:
    tp = [(t, a) for t, a in zip(timepoints, activity) if t is not None]
    if len({t for t, _ in tp}) < 2:
        return None
    ts = np.array([t for t, _ in tp])
    ac = np.array([a for _, a in tp])
    slope = np.polyfit(ts, ac, 1)[0]
    if slope < 0:
        return "maternal?"  # decreasing -> maternal-like (hint, unlabeled)
    if slope > 0:
        return "zygotic?"   # increasing -> zygotic-like (hint, unlabeled)
    return None


def detect_modules(
    genes: list[str],
    samples: list[str],
    X_norm: np.ndarray,
    mode: str,
    *,
    timepoints: dict[str, float | None] | None = None,
    replicates: dict[str, str | None] | None = None,
    gene_sets: dict[str, list[str]] | None = None,
    manual: dict[str, list[str]] | None = None,
    min_k: int = 2,
    max_k: int = 8,
    corr_threshold: float = 0.6,
    min_module_size: int = 5,
    seed: int = 42,
) -> ModuleDetection:
    """Detect modules under the requested mode and score their quality."""

    gene_index = {g: i for i, g in enumerate(genes)}
    tp_list = [(timepoints or {}).get(s) for s in samples]
    rep_list = [(replicates or {}).get(s) for s in samples]
    warnings: list[str] = []

    named_groups: list[tuple[str, str | None, list[int]]] = []  # (name, role, gene idx)
    diagnostics: dict = {"mode": mode}
    labeled_from_markers = mode in ("gene_sets", "manual")

    if mode in ("gene_sets", "manual"):
        source = gene_sets if mode == "gene_sets" else manual
        if not source:
            raise ValueError(f"mode {mode!r} requires {'--gene-sets' if mode=='gene_sets' else '--manual-modules'}")
        for name in sorted(source):
            idx = [gene_index[g] for g in source[name] if g in gene_index]
            named_groups.append((name, name if mode == "gene_sets" else name, idx))
        warnings.append("state variables are biologically imposed, not discovered from "
                        "expression structure")

    elif mode == "correlation_modules":
        labels = _correlation_modules_labels(X_norm, corr_threshold, min_module_size)
        for j in sorted(u for u in np.unique(labels) if u >= 0):
            idx = list(np.where(labels == j)[0])
            named_groups.append((f"corr_module_{j}", None, idx))

    elif mode == "silhouette_kmeans":
        labels, km_diag = _silhouette_kmeans_labels(X_norm, min_k, max_k, seed, min_module_size)
        diagnostics.update(km_diag)
        for j in sorted(u for u in np.unique(labels) if u >= 0):
            idx = list(np.where(labels == j)[0])
            named_groups.append((f"kmeans_module_{j}", None, idx))

    elif mode == "hybrid":
        if not gene_sets:
            raise ValueError("mode 'hybrid' requires --gene-sets")
        assigned: set[int] = set()
        for name in sorted(gene_sets):
            idx = [gene_index[g] for g in gene_sets[name] if g in gene_index]
            named_groups.append((name, name, idx))
            assigned.update(idx)
        remaining = np.array([i for i in range(len(genes)) if i not in assigned])
        if len(remaining) >= max(min_module_size, min_k):
            sub_labels, km_diag = _silhouette_kmeans_labels(
                X_norm[remaining], min_k, max_k, seed, min_module_size)
            diagnostics.update({"hybrid_cluster_" + k: v for k, v in km_diag.items()})
            for j in sorted(u for u in np.unique(sub_labels) if u >= 0):
                idx = list(remaining[sub_labels == j])
                named_groups.append((f"kmeans_module_{j}", None, idx))
    else:
        raise ValueError(f"Unknown module mode: {mode!r}")

    # Build modules + per-module quality.
    C = _corr_matrix(X_norm)
    idx_by_module = [np.array(idx, dtype=int) for _, _, idx in named_groups if len(idx)]
    within_list, between = _within_between(C, idx_by_module) if idx_by_module else ([], 0.0)

    modules: list[Module] = []
    gene_labels: dict[str, str] = {}
    wi = 0
    for (name, role, idx) in named_groups:
        idx_arr = np.array(idx, dtype=int)
        activity = _module_activity(X_norm, idx_arr)
        within = within_list[wi] if len(idx_arr) else float("nan")
        if len(idx_arr):
            wi += 1
        sub = X_norm[idx_arr] if len(idx_arr) else np.zeros((0, X_norm.shape[1]))
        dropout = float((sub == 0).mean()) if sub.size else 1.0
        trend = _timepoint_trend(activity, tp_list, samples)
        stab = _replicate_stability(activity, tp_list, rep_list)

        # Role / marker enrichment.
        enrichment = None
        module_gene_names = {genes[i] for i in idx_arr}
        if role is None and gene_sets:
            role, enrichment = _assign_role_by_markers(module_gene_names, gene_sets)
            enrichment = round(enrichment, 3)
        elif role is None:
            role = _role_from_trend(activity, tp_list)
        elif gene_sets and role in gene_sets:
            mk = set(gene_sets[role])
            enrichment = round(len(module_gene_names & mk) / max(len(module_gene_names), 1), 3)

        quality = {
            "size": int(len(idx_arr)),
            "within_module_corr": None if np.isnan(within) else round(within, 3),
            "between_module_corr": round(between, 3),
            "separation": None if np.isnan(within) else round(within - between, 3),
            "marker_enrichment": enrichment,
            "timepoint_trend": None if np.isnan(trend) else round(trend, 3),
            "replicate_stability": None if np.isnan(stab) else round(stab, 3),
            "dropout_fraction": round(dropout, 3),
        }
        modules.append(Module(name=name, role=role, genes=[genes[i] for i in idx_arr],
                              activity=activity, quality=quality))
        for i in idx_arr:
            gene_labels[genes[i]] = name

    # --------------------- brutal-honesty diagnostics ---------------------- #
    coherent = [m for m in modules if m.quality["within_module_corr"] is not None]
    mean_within = float(np.mean([m.quality["within_module_corr"] for m in coherent])) if coherent else float("nan")
    diagnostics["mean_within_module_corr"] = None if np.isnan(mean_within) else round(mean_within, 3)
    diagnostics["between_module_corr"] = round(between, 3)
    diagnostics["n_modules"] = len(modules)
    diagnostics["silhouette"] = diagnostics.get("silhouette")

    weak = (not np.isnan(mean_within)) and mean_within < WEAK_COHERENCE
    trends = [m.quality["timepoint_trend"] for m in modules if m.quality["timepoint_trend"] is not None]
    flat = bool(trends) and max(trends) < FLAT_TREND
    stabs = [m.quality["replicate_stability"] for m in modules if m.quality["replicate_stability"] is not None]
    unstable = bool(stabs) and float(np.mean(stabs)) < (1.0 - UNSTABLE_CV)
    high_dropout = [m.name for m in modules if m.quality["dropout_fraction"] > DROPOUT_FRAC]
    unlabeled = (not labeled_from_markers) and all(
        (m.role is None or m.role.endswith("?")) for m in modules)

    if weak:
        warnings.append("module structure is weak; mechanism inference is exploratory only")
    if not labeled_from_markers and not weak and unlabeled:
        warnings.append("modules are statistically coherent but biologically unlabeled")
    if flat:
        warnings.append("insufficient dynamic signal for mechanism recovery")
    if unstable:
        warnings.append("module activities are unstable across replicates")
    if high_dropout:
        warnings.append(f"high dropout/zero-inflation in modules: {high_dropout}")

    diagnostics["quality_flags"] = {
        "weak_coherence": bool(weak), "flat_trend": bool(flat),
        "unstable_replicates": bool(unstable), "unlabeled": bool(unlabeled),
        "exploratory_only": bool(weak or flat),
    }
    return ModuleDetection(mode=mode, modules=modules, gene_labels=gene_labels,
                           diagnostics=diagnostics, warnings=warnings)
