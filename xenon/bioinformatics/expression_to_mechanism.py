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

import csv
from dataclasses import dataclass, field

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
        "final_posterior": {k: round(v, 6) for k, v in final.items()},
        "recovered_mechanism": recovered,
        "recovered_K": float(recovered.split("=")[1]),
        "posterior_trajectory": trajectory,
    }
