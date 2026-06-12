"""Benchmark harness: XENON vs gold-standard tools (directive Phase 4).

Run as a module to produce a results report:

    python -m xenon.benchmarks.harness

Each comparison returns plain dicts and imports its heavy reference tool lazily,
so this module is importable without roadrunner/pyabc installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.mechanism import BioMechanism, MolecularState, Transition
from ..learning.forward_inference import (
    MechanismModel,
    Observation,
    metropolis_infer,
    simulate_meanfield,
)
from ..learning.sbml_io import mechanism_to_sbml

SPECIES = ["A", "B", "C"]
EDGES = [("A", "B"), ("B", "C")]
TRUE_RATES = [0.8, 0.3]


def build_mechanism(rates, initial=None) -> BioMechanism:
    initial = initial or {}
    mech = BioMechanism("ABC")
    for s in SPECIES:
        mech.add_state(MolecularState(name=s, molecule=s, concentration=float(initial.get(s, 0.0))))
    for (src, tgt), k in zip(EDGES, rates):
        mech.add_transition(Transition(source=src, target=tgt, rate_constant=float(k)))
    return mech


# --------------------------------------------------------------------------- #
# Benchmark 1: simulation correctness vs libRoadRunner (SBML ODE)
# --------------------------------------------------------------------------- #
def simulate_with_roadrunner(mechanism: BioMechanism, t: float) -> dict[str, float]:
    """Deterministic ODE simulation of the mechanism's SBML via libRoadRunner."""
    import roadrunner  # lazy; optional

    sbml = mechanism_to_sbml(mechanism)
    rr = roadrunner.RoadRunner(sbml)
    rr.simulate(0.0, float(t), 2)
    return {s: float(rr[f"[{s}]"]) for s in SPECIES}


def compare_simulation_to_roadrunner(
    rates=TRUE_RATES, initial=None, t: float = 2.0
) -> dict:
    """Compare XENON mean-field prediction with libRoadRunner ODE at time t."""
    initial = initial or {"A": 100.0, "B": 0.0, "C": 0.0}
    mech = build_mechanism(rates, initial)
    xenon_pred = simulate_meanfield(SPECIES, EDGES, rates, initial, t)
    rr_pred = simulate_with_roadrunner(mech, t)
    max_abs_err = max(abs(xenon_pred[s] - rr_pred[s]) for s in SPECIES)
    scale = max(1e-9, max(abs(rr_pred[s]) for s in SPECIES))
    return {
        "xenon": xenon_pred,
        "roadrunner": rr_pred,
        "max_abs_error": max_abs_err,
        "max_rel_error": max_abs_err / scale,
    }


# --------------------------------------------------------------------------- #
# Benchmark 2: Bayesian parameter recovery vs pyABC
# --------------------------------------------------------------------------- #
def _synthetic_data(seed=0, noise=2.0):
    rng = np.random.default_rng(seed)
    data = []
    for init in ({"A": 100.0, "B": 0.0, "C": 0.0}, {"A": 50.0, "B": 20.0, "C": 0.0}):
        for t in (0.5, 1.0, 2.0, 4.0):
            truth = simulate_meanfield(SPECIES, EDGES, TRUE_RATES, init, t)
            observed = {s: max(0.0, v + rng.normal(0.0, noise)) for s, v in truth.items()}
            data.append(
                Observation(initial=init, time=t, observed=observed, uncertainties={s: noise for s in truth})
            )
    return data


def _param_error(estimate) -> float:
    return float(np.mean([abs(estimate[i] - TRUE_RATES[i]) / TRUE_RATES[i] for i in range(len(TRUE_RATES))]))


def run_xenon_inference(data, seed=0) -> np.ndarray:
    model = MechanismModel(SPECIES, EDGES)
    res = metropolis_infer(model, data, n_samples=3000, burn_in=1000, seed=seed)
    return res.posterior_mean


def run_pyabc_inference(data, population_size: int = 200, max_populations: int = 6, seed=0) -> np.ndarray:
    """Reference posterior via pyABC (ABC-SMC) on the same data/model."""
    import pyabc  # lazy; optional

    model = MechanismModel(SPECIES, EDGES)

    def simulate(params):
        rates = [10 ** params["log_k0"], 10 ** params["log_k1"]]
        pred = [model.predict(rates, obs.initial, obs.time) for obs in data]
        return {"pred": np.array([[p[s] for s in SPECIES] for p in pred])}

    target = {"pred": np.array([[obs.observed[s] for s in SPECIES] for obs in data])}

    def distance(x, y):
        return float(np.sqrt(np.mean((x["pred"] - y["pred"]) ** 2)))

    prior = pyabc.Distribution(
        log_k0=pyabc.RV("uniform", -2, 4),  # loc=-2, scale=4 -> [-2, 2]
        log_k1=pyabc.RV("uniform", -2, 4),
    )
    abc = pyabc.ABCSMC(simulate, prior, distance, population_size=population_size)
    import tempfile, os

    db = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "abc.db")
    abc.new(db, target)
    history = abc.run(max_nr_populations=max_populations, minimum_epsilon=1.0)
    df, w = history.get_distribution()
    mean_log = np.array([np.average(df["log_k0"], weights=w), np.average(df["log_k1"], weights=w)])
    return 10.0**mean_log


def compare_inference_to_pyabc(seed=0) -> dict:
    data = _synthetic_data(seed=seed)
    xenon_mean = run_xenon_inference(data, seed=seed)
    pyabc_mean = run_pyabc_inference(data, seed=seed)
    return {
        "true_rates": TRUE_RATES,
        "xenon_posterior_mean": [float(x) for x in xenon_mean],
        "pyabc_posterior_mean": [float(x) for x in pyabc_mean],
        "xenon_rel_error": _param_error(xenon_mean),
        "pyabc_rel_error": _param_error(pyabc_mean),
    }


def main() -> dict:
    report: dict = {}
    try:
        report["simulation_vs_roadrunner"] = compare_simulation_to_roadrunner()
    except ImportError as e:
        report["simulation_vs_roadrunner"] = {"skipped": str(e)}
    try:
        report["inference_vs_pyabc"] = compare_inference_to_pyabc()
    except ImportError as e:
        report["inference_vs_pyabc"] = {"skipped": str(e)}
    print(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    main()
