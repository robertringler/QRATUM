"""CLI interface for QuASIM verification tool."""

import argparse
from datetime import datetime

from .io import load_yaml, write_json
from .models import CheckResult, Report
from .registry import REG


def main():
    """Main entry point for quasim-verify CLI."""
    ap = argparse.ArgumentParser(
        "quasim-verify",
        description="QuASIM × QuNimbus claim and compliance verification tool",
    )
    ap.add_argument("--config", required=True, help="Path to verify.config.yaml")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = ap.parse_args()

    # Load configuration
    try:
        cfg = load_yaml(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        raise SystemExit(1) from e

    started = datetime.now()
    results = []
    failures = 0
    warnings = 0

    if args.verbose:
        print(f"QuASIM Verification v{cfg.get('version', 1)}")
        print(f"Started at {started.isoformat()}")
        print(f"Running {len(cfg['checks'])} checks...")
        print()

    # Run all checks
    for item in cfg["checks"]:
        cid = item["id"]
        runner = REG.get(cid)

        if not runner:
            if args.verbose:
                print(f"❌ {cid}: Unknown check")
            results.append(
                CheckResult(
                    id=cid, passed=False, severity="error", details={"error": "unknown check"}
                )
            )
            failures += 1
            continue

        try:
            res: CheckResult = runner(cfg)
            results.append(res)

            if args.verbose:
                status = "✓" if res.passed else "✗"
                severity_marker = ""
                if not res.passed:
                    severity_marker = f" [{res.severity.upper()}]"
                print(f"{status} {cid}{severity_marker}: {res.details.get('error', 'OK')}")

            if not res.passed:
                if res.severity == "error":
                    failures += 1
                elif res.severity == "warn":
                    warnings += 1

        except Exception as e:
            if args.verbose:
                print(f"❌ {cid}: Exception - {e}")
            results.append(CheckResult(id=cid, passed=False, details={"error": str(e)}))
            failures += 1

    finished = datetime.now()
    duration = (finished - started).total_seconds()

    # Create report
    report = Report(
        version=str(cfg.get("version", "0.1.0")),
        started_at=started,
        finished_at=finished,
        results=results,
        summary={
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "errors": failures,
            "warnings": warnings,
            "duration_s": duration,
        },
    )

    # Write JSON report
    out = cfg["outputs"]["report_json"]
    write_json(out, report.model_dump(mode="json"))

    # Write SARIF report (lightweight)
    sarif_out = cfg["outputs"]["report_sarif"]
    sarif_results = []
    for r in results:
        level = (
            "error"
            if not r.passed and r.severity == "error"
            else "warning" if not r.passed else "note"
        )
        message = r.details.get("error", "Check passed")
        sarif_results.append(
            {
                "ruleId": r.id,
                "level": level,
                "message": {"text": str(message)},
                "locations": [
                    {"physicalLocation": {"artifactLocation": {"uri": p}}}
                    for p in r.evidence_paths[:1]
                ],
            }
        )

    sarif = {
        "version": "2.1.0",
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "runs": [
            {
                "tool": {"driver": {"name": "quasim-verify", "version": "0.1.0"}},
                "results": sarif_results,
            }
        ],
    }
    write_json(sarif_out, sarif)

    if args.verbose:
        print()
        print(f"Finished in {duration:.2f}s")
        print(f"Results: {report.summary['passed']}/{report.summary['total']} passed")
        print(f"Report: {out}")
        print(f"SARIF: {sarif_out}")

    # Exit with error code if any error-level checks failed
    if cfg["outputs"].get("exit_on_fail", True) and failures:
        if args.verbose:
            print(f"\n❌ Verification FAILED: {failures} error(s)")
        raise SystemExit(1)

    if args.verbose:
        print("\n✓ Verification PASSED")


if __name__ == "__main__":
    main()
