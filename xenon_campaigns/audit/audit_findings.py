#!/usr/bin/env python3
"""Status verifier for the XENON Phase-1 forensic audit findings.

Each check exercises the ACTUAL current code (not a hand-rolled reproduction)
and reports whether the finding is FIXED or still PRESENT. Run:

    PYTHONPATH=$PWD python xenon_campaigns/audit/audit_findings.py

Findings are keyed to PHASE1_FORENSIC_AUDIT.md (F1..F12). Tier-0 remediation
(chosen scope) targets F1, F2, F4, F5, F6; F3/F10 remain deferred and are
reported honestly as PRESENT.
"""

from __future__ import annotations

import numpy as np

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.learning.bayesian_updater import BayesianUpdater, ExperimentResult
from xenon.learning.mechanism_prior import MechanismPrior
from xenon.runtime.xenon_kernel import XENONRuntime
from xenon.simulation.gillespie import GillespieSimulator


def _status(fixed: bool) -> str:
    return "FIXED" if fixed else "PRESENT"


def f1_reverse_reactions() -> bool:
    """F1: SSA must honour reverse reactions (detailed balance)."""
    m = BioMechanism("rev")
    m.add_state(MolecularState(name="A", molecule="P", concentration=100.0))
    m.add_state(MolecularState(name="B", molecule="P", concentration=0.0))
    m.add_transition(
        Transition(source="A", target="B", rate_constant=1.0, reversible=True, reverse_rate=1.0)
    )
    finals = [
        GillespieSimulator(m).run(t_max=50.0, initial_state={"A": 100.0, "B": 0.0},
                                  seed=s, record_interval=1.0)[1]["A"][-1]
        for s in range(16)
    ]
    mean_a = float(np.mean(finals))
    fixed = 35.0 < mean_a < 65.0  # symmetric -> ~50/50
    print(f"[F1] symmetric A<->B mean A={mean_a:.1f} nM (expect ~50) -> {_status(fixed)}")
    return fixed


def f2_stoichiometry() -> bool:
    """F2: engine must apply declared stoichiometry / bimolecular kinetics."""
    m = BioMechanism("bi")
    for n, c in [("A", 60.0), ("B", 40.0), ("C", 0.0)]:
        m.add_state(MolecularState(name=n, molecule=n, concentration=c))
    m.add_transition(
        Transition(source="A", target="C", rate_constant=5.0, stoichiometry={"A": -1, "B": -1, "C": 1})
    )
    _, tr = GillespieSimulator(m).run(
        t_max=5.0, initial_state={"A": 60.0, "B": 40.0, "C": 0.0}, seed=1, record_interval=0.5
    )
    a, b, c = tr["A"][-1], tr["B"][-1], tr["C"][-1]
    fired = c > 1.0
    conserved = abs((a + c) - 60.0) < 2.0 and abs((b + c) - 40.0) < 2.0
    b_limiting = b < a  # B is the limiting reagent
    fixed = fired and conserved and b_limiting
    print(f"[F2] A+B->C final A={a:.1f} B={b:.1f} C={c:.1f} "
          f"(fired={fired}, conserved={conserved}, B-limiting={b_limiting}) -> {_status(fixed)}")
    return fixed


def f4_evidence_accumulates() -> bool:
    """F4: concentration likelihood must sharpen with more measurements."""
    bu = BayesianUpdater()

    def lik(n_states: int) -> float:
        m = BioMechanism("m")
        obs = {}
        unc = {}
        for i in range(n_states):
            name = f"S{i}"
            m.add_state(MolecularState(name=name, molecule="P", concentration=50.0))
            obs[name] = 56.0  # predicted 50, observed 56, sigma 3 -> 2 sigma off
            unc[name] = 3.0
        exp = ExperimentResult("concentration", observations=obs, uncertainties=unc)
        return bu.compute_likelihood(m, exp)

    l1, l5 = lik(1), lik(5)
    fixed = l5 < l1 * 0.5  # joint likelihood must shrink with data
    print(f"[F4] concentration likelihood n=1 -> {l1:.3e}, n=5 -> {l5:.3e} "
          f"(must shrink) -> {_status(fixed)}")
    return fixed


def f5_regeneration_preserves_evidence() -> bool:
    """F5: regeneration must NOT wipe accumulated posteriors (kernel path)."""
    mp = MechanismPrior()

    def mech(name: str, post: float) -> BioMechanism:
        m = BioMechanism(name)
        m.add_state(MolecularState(name="A", molecule="EGFR", concentration=50.0, free_energy=-10.0))
        m.add_state(MolecularState(name="B", molecule="EGFR", concentration=50.0, free_energy=-12.0))
        m.add_transition(Transition(source="A", target="B", rate_constant=1.0))
        m.posterior = post
        return m

    evidenced = mech("evidenced", 0.97)
    other = mech("other", 0.03)
    new = mech("new", 1.0)
    pool = [evidenced, other, new]
    # Kernel now calls initialize_new_priors (not the resetting method).
    mp.initialize_new_priors(pool, [new])
    # evidenced must retain the LARGEST share among the pre-existing pair.
    fixed = evidenced.posterior > other.posterior and evidenced.posterior > 0.3
    print(f"[F5] after regeneration: evidenced={evidenced.posterior:.3f} other={other.posterior:.3f} "
          f"new={new.posterior:.3f} (evidence preserved) -> {_status(fixed)}")
    return fixed


def f6_replicated_observable() -> bool:
    """F6: the runtime must store a Monte-Carlo standard error on predictions."""
    rt = XENONRuntime(convergence_threshold=0.1, sim_replicates=5, sim_t_max=2.0)
    rt.add_target(name="EGFR_target", protein="EGFR", objective="characterize")
    rt._rng = np.random.default_rng(42)
    rt._generate_hypothesis_mechanisms(rt.targets[0])
    rt._simulate_mechanisms(rt.targets[0])
    se_present = any(
        "conc_mc_se" in s.properties
        for m in rt.mechanisms["EGFR_target"]
        for s in m._states.values()
    )
    print(f"[F6] predicted states carry Monte-Carlo SE (conc_mc_se): {se_present} -> "
          f"{_status(se_present)}")
    return se_present


def f3_perturbation_channel() -> bool:
    """F3: perturbation likelihood must be schema-aligned and topology-dependent."""
    bu = BayesianUpdater()

    def mech(with_path: bool) -> BioMechanism:
        m = BioMechanism("t")
        m.add_state(MolecularState(name="EGFR_inactive", molecule="EGFR", concentration=50.0))
        m.add_state(MolecularState(name="EGFR_active", molecule="EGFR", concentration=50.0))
        if with_path:
            m.add_transition(Transition(source="EGFR_inactive", target="EGFR_active",
                                        rate_constant=1.0))
        return m

    # Use the schema the kernel now emits.
    exp = ExperimentResult(
        "perturbation",
        observations={"response": 0.5},
        uncertainties={"response": 0.1},
        conditions={"perturbation_source": "EGFR_inactive",
                    "perturbation_target": "EGFR_active", "temperature": 310.0},
    )
    lik_path = bu.compute_likelihood(mech(True), exp)
    lik_no_path = bu.compute_likelihood(mech(False), exp)
    fixed = lik_path != 0.1 and lik_path > lik_no_path
    print(f"[F3] perturbation likelihood path={lik_path:.4f} no-path={lik_no_path:.2e} "
          f"(topology-dependent) -> {_status(fixed)}")
    return fixed


def main() -> int:
    print("XENON audit status — actual-code verification (Tier-0 remediation)\n")
    tier0 = {
        "F1_reverse_reactions": f1_reverse_reactions(),
        "F2_stoichiometry": f2_stoichiometry(),
        "F4_evidence_accumulates": f4_evidence_accumulates(),
        "F5_regeneration_preserves": f5_regeneration_preserves_evidence(),
        "F6_replicated_observable": f6_replicated_observable(),
    }
    print()
    extra = {"F3_perturbation_channel": f3_perturbation_channel()}
    print()
    all_checks = {**tier0, **extra}
    fixed = sum(1 for v in all_checks.values() if v)
    print(f"{fixed}/{len(all_checks)} findings FIXED "
          f"(Tier-0 F1/F2/F4/F5/F6 + F3 perturbation channel).")
    return 0 if fixed == len(all_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
