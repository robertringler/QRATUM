"""cli.py — the single ``qratum`` operator surface.

Subcommands (per §5 of the plan):

* ``run``       — execute a sequence of intents through the default backend.
* ``simulate``  — alias for ``run`` with ``--dry-run`` semantics: no persistence.
* ``ledger``    — print, verify, or export the most recent run's ledger.
* ``falsify``   — invoke a domain falsifier and print its A / A0 / B verdict.
* ``verify``    — run health + readiness checks and exit non-zero on failure.

The CLI is implemented with ``argparse`` to keep ``qratum_framework`` free
of additional runtime dependencies (``click`` is only an optional dep of
the broader project).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from qratum_framework import __version__
from qratum_framework.config import DEFAULT_PROFILE, PROFILES, load_profile
from qratum_framework.falsifier import _DemoFalsifier
from qratum_framework.health import check_health, check_readiness
from qratum_framework.operator import Operator, StrictCIIRBackend
from qratum_framework.trace import MerkleLedger, UnifiedTraceEntry

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qratum",
        description="QRATUM unified operator surface (CIIR–CRS–RIC spine).",
    )
    p.add_argument("--version", action="version", version=f"qratum_framework {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # run --------------------------------------------------------------
    run_p = sub.add_parser(
        "run", help="Execute a sequence of intents through the default backend."
    )
    run_p.add_argument(
        "intents",
        nargs="*",
        help="Raw intent tokens (e.g. begin allocate_cpu hold). "
        "If empty, a built-in canonical demo sequence is used.",
    )
    run_p.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(PROFILES),
        help="Named pipeline profile (default: %(default)s).",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary instead of the default human-readable output.",
    )

    # simulate ---------------------------------------------------------
    sim_p = sub.add_parser(
        "simulate",
        help="Dry-run a sequence of intents (no ledger persistence beyond stdout).",
    )
    sim_p.add_argument("intents", nargs="*")
    sim_p.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    sim_p.add_argument("--json", action="store_true")

    # ledger -----------------------------------------------------------
    led_p = sub.add_parser(
        "ledger",
        help="Inspect a JSONL ledger file: verify the Merkle chain, summarise, or export.",
    )
    led_p.add_argument("path", help="Path to a JSONL ledger file (as produced by `qratum run`).")
    led_p.add_argument(
        "--verify-only",
        action="store_true",
        help="Print only OK / FAILED and exit with non-zero on FAILED.",
    )

    # falsify ----------------------------------------------------------
    fal_p = sub.add_parser(
        "falsify",
        help="Invoke a domain falsifier and print its A / A0 / B verdict.",
    )
    fal_p.add_argument(
        "--domain",
        default="demo",
        help="Falsifier domain (default: demo). Real domain falsifiers register elsewhere.",
    )
    fal_p.add_argument("--seed", type=int, default=0)
    fal_p.add_argument("--json", action="store_true")

    # verify -----------------------------------------------------------
    ver_p = sub.add_parser(
        "verify", help="Run health + readiness checks; exit non-zero on failure."
    )
    ver_p.add_argument("--json", action="store_true")

    # clusters ---------------------------------------------------------
    cl_p = sub.add_parser(
        "clusters",
        help="Run Layer B cluster discovery on persona/baseline token files.",
    )
    cl_p.add_argument(
        "--persona-tokens",
        required=True,
        help="Path to a whitespace-separated persona token file.",
    )
    cl_p.add_argument(
        "--baseline-tokens",
        required=True,
        help="Path to a whitespace-separated baseline token file.",
    )
    cl_p.add_argument("--window", type=int, default=64)
    cl_p.add_argument("--seed", type=int, default=0)
    cl_p.add_argument("--json", action="store_true", help="Emit full state JSON.")

    # drift ------------------------------------------------------------
    dr_p = sub.add_parser(
        "drift",
        help="Run Layer A drift detection on a persona token stream.",
    )
    dr_p.add_argument("--persona-tokens", required=True)
    dr_p.add_argument("--baseline-tokens", required=True)
    dr_p.add_argument("--utterance", required=True, help="The user prompt U.")
    dr_p.add_argument("--persona", default="persona")
    dr_p.add_argument("--json", action="store_true")

    # eval -------------------------------------------------------------
    ev_p = sub.add_parser(
        "eval",
        help="Run Layer C evaluation matrix and emit a regression report.",
    )
    ev_p.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write the JSON report.",
    )
    ev_p.add_argument(
        "--report-md",
        default=None,
        help="Optional path to write the Markdown report.",
    )
    ev_p.add_argument(
        "--regression",
        action="store_true",
        help="Exit non-zero if any non-baseline row regresses.",
    )
    ev_p.add_argument("--json", action="store_true", help="Print the JSON report.")

    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


_DEMO_INTENTS: List[str] = [
    "begin",
    "increase_cpu",
    "increase_mem",
    "hold",
    "decrease_cpu",
    "end",
]


def _resolve_intents(args_intents: Sequence[str], profile_steps: int) -> List[str]:
    """Pick the actual intent sequence for a run.

    If the user passed intents explicitly, honour them verbatim.  Otherwise
    cycle through the built-in demo sequence to fill ``profile_steps`` ticks
    so that the pipeline produces a non-empty ledger by default.
    """
    if args_intents:
        return list(args_intents)
    out: List[str] = []
    while len(out) < profile_steps:
        out.extend(_DEMO_INTENTS)
    return out[:profile_steps]


def _cmd_run(args: argparse.Namespace, *, persist: bool = True) -> int:
    profile = load_profile(args.profile)
    intents = _resolve_intents(args.intents, profile.n_steps)

    operator = Operator(backend=StrictCIIRBackend(), run_id=f"cli-{profile.name}")
    result = operator.run(intents)

    if args.json:
        out = {
            "summary": result.summary(),
            "profile": profile.name,
            "intents": intents,
        }
        if persist:
            out["ledger"] = result.ledger.manifest()
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
    else:
        s = result.summary()
        print(f"backend         : {s['backend']}")
        print(f"profile         : {profile.name}")
        print(f"intents (n)     : {len(intents)}")
        print(f"ledger height   : {s['ledger_height']}")
        print(f"ledger head     : {s['ledger_head']}")
        print(f"chain verified  : {s['verified']}")
        print(f"failures        : {s['failure_count']}")
        print(f"statuses        : {s['statuses']}")

    # Non-zero exit if the chain itself does not verify (a hard regression).
    return 0 if result.ledger.verify() else 2


def _cmd_simulate(args: argparse.Namespace) -> int:
    return _cmd_run(args, persist=False)


def _cmd_ledger(args: argparse.Namespace) -> int:
    """Verify a JSONL ledger file by recomputing the chain hashes.

    This mirrors :meth:`MerkleLedger.verify` but reads from disk so that
    third-party reproducers can audit a run without re-executing it.
    """
    try:
        with open(args.path, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except OSError as exc:
        print(f"ledger read failed: {exc}", file=sys.stderr)
        return 2

    ledger = MerkleLedger()
    expected_chain: List[str] = []
    expected_prev: List[str] = []
    for ln in lines:
        rec = json.loads(ln)
        e = rec["entry"]
        entry = UnifiedTraceEntry(
            step=e["step"],
            intent_raw=e["intent_raw"],
            intent=e.get("intent"),
            action=e["action"],
            status=e["status"],
            failure_mode=e.get("failure_mode"),
            pre_state_hash=e["pre_state_hash"],
            post_state_hash=e.get("post_state_hash"),
            phi_verdict=e["phi_verdict"],
            violated_constraints=tuple(e.get("violated_constraints", []) or []),
            metrics=e.get("metrics", {}) or {},
            signatures=tuple(e.get("signatures", []) or []),
            schema_version=e.get("schema_version", "1.0"),
            metadata=e.get("metadata", {}) or {},
        )
        ledger.append(entry)
        expected_chain.append(rec["chain_hash"])
        expected_prev.append(rec["prev_hash"])

    # The on-disk chain hashes must equal what we just recomputed.  This is
    # what catches deliberate or accidental tampering.
    on_disk_match = True
    for i, (_, prev_hash, chain_hash) in enumerate(ledger.iter_with_hashes()):
        if chain_hash != expected_chain[i] or prev_hash != expected_prev[i]:
            on_disk_match = False
            break

    chain_ok = ledger.verify() and on_disk_match

    if args.verify_only:
        print("OK" if chain_ok else "FAILED")
        return 0 if chain_ok else 1

    out = {
        "path": args.path,
        "height": ledger.height(),
        "head": ledger.head(),
        "verified": chain_ok,
        "summary_by_status": _statuses(ledger),
    }
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0 if chain_ok else 1


def _statuses(ledger: MerkleLedger) -> dict:
    out: dict = {}
    for e in ledger.entries:
        out[e.status] = out.get(e.status, 0) + 1
    return out


def _cmd_falsify(args: argparse.Namespace) -> int:
    """Invoke a falsifier and print its verdict.

    For now only the built-in demo falsifier is wired in; real domain
    falsifiers register through the Falsifier protocol and will be plugged
    in by their own packages without modifying this CLI.
    """
    if args.domain != "demo":
        # Placeholder for future registry-based dispatch.
        print(
            f"unknown falsifier domain {args.domain!r}; built-in domains: ['demo']",
            file=sys.stderr,
        )
        return 2
    verdict = _DemoFalsifier().run(seed=args.seed)
    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(f"domain   : {verdict.domain}")
        print(f"verdict  : {verdict.verdict}")
        print(f"snr      : {verdict.snr:.3e}")
        print(f"overlap  : {verdict.overlap_with_ground:.6f}")
        if verdict.notes:
            print(f"notes    : {verdict.notes}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    health = check_health()
    readiness = check_readiness()
    payload = {
        "health": health.to_dict(),
        "readiness": readiness.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(f"health    : {health.status}")
        for c in health.checks:
            print(f"  - {c['name']:<22} {'ok' if c['ok'] else 'FAIL':<5} {c['detail']}")
        print(f"readiness : {readiness.status}")
        for c in readiness.checks:
            print(f"  - {c['name']:<22} {'ok' if c['ok'] else 'FAIL':<5} {c['detail']}")
    return 0 if (health.status == "OK" and readiness.status == "OK") else 1


# ---------------------------------------------------------------------------
# Observability subcommands (Layers A, B, C)
# ---------------------------------------------------------------------------


def _read_tokens(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return fh.read().split()


def _cmd_clusters(args: argparse.Namespace) -> int:
    """Run Layer B (cluster discovery) on two token files."""
    from qratum_framework.observability.clusters import discover_clusters

    persona = _read_tokens(args.persona_tokens)
    baseline = _read_tokens(args.baseline_tokens)
    state = discover_clusters(
        persona_tokens=persona,
        baseline_tokens=baseline,
        window=args.window,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(state.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        schema = state.to_schema()
        print(json.dumps(schema, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    """Run Layer A (drift detection) on a persona/baseline token pair."""
    from qratum_framework.observability.clusters import discover_clusters
    from qratum_framework.observability.drift import (
        DriftEngine,
        deterministic_logprob,
    )

    persona_tokens = _read_tokens(args.persona_tokens)
    baseline_tokens = _read_tokens(args.baseline_tokens)
    state = discover_clusters(
        persona_tokens=persona_tokens, baseline_tokens=baseline_tokens
    )

    persona_freqs = {}
    baseline_freqs = {}
    for t in persona_tokens:
        persona_freqs[t] = persona_freqs.get(t, 0.0) + 1.0
    for t in baseline_tokens:
        baseline_freqs[t] = baseline_freqs.get(t, 0.0) + 1.0
    engine = DriftEngine()
    report = engine.score(
        tokens=persona_tokens,
        utterance=args.utterance,
        clusters=state,
        persona_model=deterministic_logprob(persona_freqs),
        baseline_model=deterministic_logprob(baseline_freqs),
        persona=args.persona,
    )
    payload = report.to_dict() if args.json else report.to_schema()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if not report.anomaly_detected else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run Layer C (eval matrix) and optionally fail on regression."""
    from qratum_framework.observability.eval import EvalRunner
    from qratum_framework.observability.eval.reports import (
        build_report,
        write_json_report,
        write_markdown_report,
    )

    runner = EvalRunner()
    rows = runner.run()
    report = build_report(rows)

    if args.report_json:
        write_json_report(report, args.report_json)
    if args.report_md:
        write_markdown_report(report, args.report_md)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    else:
        print(f"runs            : {int(report.summary.get('n_runs', 0))}")
        print(
            f"baseline runs   : {int(report.summary.get('n_baseline_runs', 0))}"
        )
        print(
            f"max anomaly_rate: {report.summary.get('max_anomaly_rate', 0.0):.3f}"
        )
        print(
            f"max lift_rate   : "
            f"{report.summary.get('max_lift_cluster_rate', 0.0):.3f}"
        )
        print(
            f"regressions     : {int(report.summary.get('regressions', 0))}"
        )
        print(
            f"verdict         : {'FAIL' if report.has_regression else 'PASS'}"
        )

    if args.regression and report.has_regression:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_COMMANDS = {
    "run": _cmd_run,
    "simulate": _cmd_simulate,
    "ledger": _cmd_ledger,
    "falsify": _cmd_falsify,
    "verify": _cmd_verify,
    "clusters": _cmd_clusters,
    "drift": _cmd_drift,
    "eval": _cmd_eval,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    return int(handler(args) or 0)


if __name__ == "__main__":  # pragma: no cover — manual invocation
    raise SystemExit(main())
