"""Full FLUXA experiment campaign.

Runs every experiment the study requires, writes machine-readable artefacts to
``results/fluxa/``, and emits a consolidated ``manifest.json``.

Reproduce with::

    python -m fluxa.experiments.run_all

Each stage is timed and recorded. A stage that fails is recorded with its
traceback and the campaign continues, so one broken experiment does not
destroy the evidence from the others; the manifest's ``failed_stages`` list is
the authoritative statement of what did not run.

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from fluxa import __version__ as fluxa_version
from fluxa.benchmark import host_description, run_benchmark_suite
from fluxa.config import DEFAULT_SCENARIO_CONFIG, DEFAULT_SYSTEM_CONFIG, load_system
from fluxa.engine import RunConfig, SimulationEngine
from fluxa.experiments.adversarial import run_adversarial_campaign, summarise
from fluxa.experiments.determinism import provenance_experiment, replay
from fluxa.io import DEFAULT_RESULTS_DIR, ensure_dir, write_json, write_rows_csv, write_result
from fluxa.montecarlo import CampaignConfig, run_campaign
from fluxa.provenance import environment_fingerprint
from fluxa.recovery import (
    RollbackExperiment,
    build_rollback_engine,
    corrupt_power_balance,
    corrupt_soc_bound,
)
from fluxa.scenarios import load_scenarios, rescale_onsets

LOGGER = logging.getLogger("fluxa.experiments")

#: Primary horizon: 24 hours at 5-minute resolution.
PRIMARY_STEPS = 288
PRIMARY_TIMESTEP_S = 300.0

#: Extended horizon: 7 days at 5-minute resolution.
EXTENDED_STEPS = 2016

#: Scenarios replayed for the determinism experiment.
DETERMINISM_SCENARIOS = ("A_BASELINE", "F_COMPOUND", "G_LINE_DERATE")


def _authorizer(scenario_id: str) -> str:
    return f"operator_{scenario_id.split('_')[0].lower()}"


class Campaign:
    """Orchestrates the experiment campaign and collects its manifest."""

    def __init__(
        self,
        out_dir: Path = DEFAULT_RESULTS_DIR,
        *,
        monte_carlo_samples: int = 100,
        scaling_counts: tuple[int, ...] = (1, 10, 100),
        extended_steps: int = EXTENDED_STEPS,
        skip_benchmark: bool = False,
    ) -> None:
        self.out_dir = ensure_dir(out_dir)
        self.monte_carlo_samples = monte_carlo_samples
        self.scaling_counts = scaling_counts
        self.extended_steps = extended_steps
        self.skip_benchmark = skip_benchmark

        self.loaded = load_system(DEFAULT_SYSTEM_CONFIG)
        self.scenarios = load_scenarios(DEFAULT_SCENARIO_CONFIG)
        self.manifest: dict[str, Any] = {
            "fluxa_version": fluxa_version,
            "system_config": str(DEFAULT_SYSTEM_CONFIG),
            "scenario_config": str(DEFAULT_SCENARIO_CONFIG),
            "config_hash": self.loaded.config_hash,
            "model_hash": self.loaded.model_hash,
            "environment": environment_fingerprint(),
            "host": host_description(),
            "stages": {},
            "stage_timings_s": {},
            "failed_stages": [],
            "artefacts": {},
        }

    # ------------------------------------------------------------- helper
    def _stage(self, name: str, fn: Callable[[], Any]) -> Any:
        LOGGER.info("--- stage: %s ---", name)
        start = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            elapsed = time.perf_counter() - start
            LOGGER.error("stage %s FAILED after %.1f s: %s", name, elapsed, exc)
            self.manifest["failed_stages"].append(
                {
                    "stage": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                    "elapsed_s": elapsed,
                }
            )
            self.manifest["stage_timings_s"][name] = elapsed
            return None
        elapsed = time.perf_counter() - start
        self.manifest["stage_timings_s"][name] = elapsed
        LOGGER.info("stage %s completed in %.1f s", name, elapsed)
        return value

    # --------------------------------------------------- stage: scenarios
    def stage_primary_scenarios(self) -> dict[str, Any]:
        """24 h at 5-minute resolution for every scenario in the library."""
        rows: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for scenario_id, scenario in self.scenarios.items():
            config = RunConfig(
                run_id="primary",
                timestep_s=PRIMARY_TIMESTEP_S,
                n_steps=PRIMARY_STEPS,
                horizon_steps=12,
                checkpoint_every=24,
                authorized=True,
                authorizer_id=_authorizer(scenario_id),
            )
            result = SimulationEngine(self.loaded, config).run(scenario)
            written = write_result(result, self.out_dir / "scenarios", prefix=f"primary_{scenario_id}")
            self.manifest["artefacts"][f"primary_{scenario_id}"] = written
            rows.append(result.summary())
            details[scenario_id] = {
                "summary": result.summary(),
                "metrics": result.metrics.to_dict(),
                "event_counts": result.ledger.counts_by_type(),
                "ledger_verified": result.ledger.verify()[0],
                "n_checkpoints": len(result.checkpoint_ids),
            }
            LOGGER.info(
                "%s: unserved %.2f MWh, cost $%.0f, %d events",
                scenario_id,
                result.metrics.reliability.unserved_energy_mwh,
                result.metrics.economics.total_operating_cost_usd,
                len(result.ledger),
            )
        path = write_rows_csv(self.out_dir / "primary_scenario_summary.csv", rows)
        self.manifest["artefacts"]["primary_scenario_summary_csv"] = str(path)
        return details

    # --------------------------------------------------- stage: extended
    def stage_extended(self) -> dict[str, Any]:
        """7 days at 5-minute resolution, same model and framework."""
        details: dict[str, Any] = {}
        day_scale = self.extended_steps / PRIMARY_STEPS
        for scenario_id in ("A_BASELINE", "F_COMPOUND"):
            scenario = self.scenarios[scenario_id]
            # Place the compound disturbance on day 5 of the week instead of
            # day 1, so the extended run is not simply the 24 h run repeated.
            if scenario.perturbations:
                offset_days = 4.0
                scenario = rescale_onsets(scenario, 1.0)
                scenario = type(scenario)(
                    scenario_id=f"{scenario_id}_DAY5",
                    name=f"{scenario.name} (day 5)",
                    description=(
                        f"{scenario.description} Onsets shifted by {offset_days:.0f} days for "
                        "the 7-day extended run."
                    ),
                    perturbations=tuple(
                        type(p)(
                            perturbation_id=p.perturbation_id,
                            kind=p.kind,
                            start_h=p.start_h + offset_days * 24.0,
                            ramp_min=p.ramp_min,
                            hold_min=p.hold_min,
                            recovery_min=p.recovery_min,
                            magnitude=p.magnitude,
                            targets=p.targets,
                        )
                        for p in scenario.perturbations
                    ),
                    tags=scenario.tags + ("extended",),
                )
            config = RunConfig(
                run_id="extended",
                timestep_s=PRIMARY_TIMESTEP_S,
                n_steps=self.extended_steps,
                horizon_steps=12,
                checkpoint_every=144,
                authorized=True,
                authorizer_id=_authorizer(scenario_id),
            )
            result = SimulationEngine(self.loaded, config).run(scenario)
            written = write_result(
                result, self.out_dir / "extended", prefix=f"extended_{scenario_id}"
            )
            self.manifest["artefacts"][f"extended_{scenario_id}"] = written
            details[scenario_id] = {
                "summary": result.summary(),
                "metrics": result.metrics.to_dict(),
                "event_counts": result.ledger.counts_by_type(),
                "ledger_verified": result.ledger.verify()[0],
                "n_checkpoints": len(result.checkpoint_ids),
                "horizon_days": self.extended_steps * PRIMARY_TIMESTEP_S / 86_400.0,
            }
            LOGGER.info(
                "extended %s: %d steps, unserved %.2f MWh, %d events, %.1f s",
                scenario_id,
                self.extended_steps,
                result.metrics.reliability.unserved_energy_mwh,
                len(result.ledger),
                result.wall_time_s,
            )
        return details

    # ------------------------------------------------ stage: determinism
    def stage_determinism(self) -> dict[str, Any]:
        reports = {
            scenario_id: replay(
                self.loaded,
                self.scenarios[scenario_id],
                n_runs=3,
                n_steps=PRIMARY_STEPS,
                run_id="determinism",
            ).to_dict()
            for scenario_id in DETERMINISM_SCENARIOS
        }
        path = write_json(self.out_dir / "determinism.json", reports)
        self.manifest["artefacts"]["determinism"] = str(path)
        return reports

    # ------------------------------------------------- stage: provenance
    def stage_provenance(self) -> dict[str, Any]:
        report = provenance_experiment(
            self.loaded, self.scenarios["F_COMPOUND"], n_steps=PRIMARY_STEPS
        ).to_dict()
        path = write_json(self.out_dir / "provenance.json", report)
        self.manifest["artefacts"]["provenance"] = str(path)
        return report

    # --------------------------------------------------- stage: rollback
    def stage_rollback(self) -> dict[str, Any]:
        reports: dict[str, Any] = {}
        cases = (
            ("baseline_power_balance", "A_BASELINE", corrupt_power_balance(25.0), 12, 5, 2, 4,
             "phantom generation of 25 MW injected at t4"),
            ("baseline_soc", "A_BASELINE", corrupt_soc_bound(0.4), 12, 5, 2, 4,
             "state of charge raised 0.40 above its physical trajectory at t4"),
            ("compound_under_disturbance", "F_COMPOUND", corrupt_power_balance(40.0), 240, 232,
             226, 230, "phantom generation of 40 MW injected during the compound event"),
        )
        for label, scenario_id, hook, n_steps, forward, ckpt, bad, description in cases:
            engine = build_rollback_engine(
                self.loaded, run_id=f"rollback-{label}", n_steps=n_steps
            )
            report = RollbackExperiment(
                engine,
                self.scenarios[scenario_id],
                forward_steps=forward,
                checkpoint_step=ckpt,
                corrupted_step=bad,
            ).run(hook, description)
            payload = report.to_dict()
            # The full hash lists are large; keep the decisive fields and the
            # counts, and write the lists to their own artefact.
            lists = {
                "resumed_state_hashes": payload.pop("resumed_state_hashes"),
                "reference_state_hashes": payload.pop("reference_state_hashes"),
            }
            write_json(self.out_dir / f"rollback_{label}_hashes.json", lists)
            payload["n_resumed_hashes"] = len(lists["resumed_state_hashes"])
            payload["n_reference_hashes"] = len(lists["reference_state_hashes"])
            reports[label] = payload
            LOGGER.info("rollback %s: success=%s", label, report.success())
        path = write_json(self.out_dir / "rollback.json", reports)
        self.manifest["artefacts"]["rollback"] = str(path)
        return reports

    # ------------------------------------------------ stage: monte carlo
    def stage_monte_carlo(self) -> dict[str, Any]:
        config = CampaignConfig(
            campaign_seed=777_001,
            n_samples=self.monte_carlo_samples,
            n_steps=PRIMARY_STEPS,
        )
        campaign = run_campaign(config, parallel=True)
        rows_path = write_rows_csv(self.out_dir / "monte_carlo_samples.csv", campaign.rows)
        summary = campaign.to_dict()
        summary_path = write_json(self.out_dir / "monte_carlo.json", summary)
        self.manifest["artefacts"]["monte_carlo_samples_csv"] = str(rows_path)
        self.manifest["artefacts"]["monte_carlo"] = str(summary_path)
        LOGGER.info(
            "monte carlo: %d/%d samples in %.1f s (%d workers)",
            len(campaign.rows),
            config.n_samples,
            campaign.wall_time_s,
            campaign.n_workers,
        )
        return summary

    # ------------------------------------------------ stage: adversarial
    def stage_adversarial(self) -> dict[str, Any]:
        findings = run_adversarial_campaign(self.loaded, self.scenarios, n_steps=96)
        summary = summarise(findings)
        path = write_json(self.out_dir / "adversarial.json", summary)
        self.manifest["artefacts"]["adversarial"] = str(path)
        LOGGER.info(
            "adversarial: %d attacks, outcomes %s, succeeded %s",
            summary["n_attacks"],
            summary["outcomes"],
            summary["successful_attacks"],
        )
        return summary

    # -------------------------------------------------- stage: benchmark
    def stage_benchmark(self) -> dict[str, Any]:
        suite = run_benchmark_suite(
            self.loaded, scaling_counts=self.scaling_counts, parallel_samples=16
        ).to_dict()
        path = write_json(self.out_dir / "benchmark.json", suite)
        self.manifest["artefacts"]["benchmark"] = str(path)
        return suite

    # --------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        total_start = time.perf_counter()
        self.manifest["stages"]["primary_scenarios"] = self._stage(
            "primary_scenarios", self.stage_primary_scenarios
        )
        self.manifest["stages"]["extended"] = self._stage("extended", self.stage_extended)
        self.manifest["stages"]["determinism"] = self._stage("determinism", self.stage_determinism)
        self.manifest["stages"]["provenance"] = self._stage("provenance", self.stage_provenance)
        self.manifest["stages"]["rollback"] = self._stage("rollback", self.stage_rollback)
        self.manifest["stages"]["monte_carlo"] = self._stage("monte_carlo", self.stage_monte_carlo)
        self.manifest["stages"]["adversarial"] = self._stage("adversarial", self.stage_adversarial)
        if not self.skip_benchmark:
            self.manifest["stages"]["benchmark"] = self._stage("benchmark", self.stage_benchmark)
        self.manifest["total_wall_time_s"] = time.perf_counter() - total_start
        self.manifest["quantum_acceleration_demonstrated"] = False
        manifest_path = write_json(self.out_dir / "manifest.json", self.manifest)
        self.manifest["artefacts"]["manifest"] = str(manifest_path)
        LOGGER.info(
            "campaign complete in %.1f s; %d failed stages; manifest at %s",
            self.manifest["total_wall_time_s"],
            len(self.manifest["failed_stages"]),
            manifest_path,
        )
        return self.manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FLUXA experiment campaign")
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--monte-carlo-samples", type=int, default=100)
    parser.add_argument("--extended-steps", type=int, default=EXTENDED_STEPS)
    parser.add_argument(
        "--scaling-counts", default="1,10,100",
        help="comma-separated scenario counts for the scaling benchmark",
    )
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    campaign = Campaign(
        Path(args.out_dir),
        monte_carlo_samples=args.monte_carlo_samples,
        scaling_counts=tuple(int(x) for x in args.scaling_counts.split(",") if x),
        extended_steps=args.extended_steps,
        skip_benchmark=args.skip_benchmark,
    )
    manifest = campaign.run()
    print(json.dumps({
        "total_wall_time_s": round(manifest["total_wall_time_s"], 1),
        "failed_stages": [f["stage"] for f in manifest["failed_stages"]],
        "artefacts": len(manifest["artefacts"]),
    }, indent=2))
    return 1 if manifest["failed_stages"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
