"""Lightweight polling utility for monitoring an in-progress automation pipeline run.

Usage examples (run from project root):

  python -m automation.poll_pipeline --once
      # Show a one-shot summary of the most recent run directory under automation_runs/

  python -m automation.poll_pipeline --follow --interval 5
      # Continuously update progress every 5s until run completes (index.json appears)

  python -m automation.poll_pipeline --run-dir automation_runs/<hash> --tail 15 --once
      # Inspect a specific run dir and show last 15 lines of the currently active task stdout

Design notes:
  - A run directory is assumed to be automation_runs/<config_hash> created by run_pipeline.
  - Each task creates a subdirectory named L{Lx}x{Ly}_dis{disorder}. Metrics appear as metrics.json
    (written by dispatch_analysis). We treat presence of metrics.json as completion (success or fail).
  - While a task is executing it will have analysis_stdout.log growing; we heuristically pick the
    "active" task as the one whose stdout file has the most recent mtime.
  - Completion criterion for the overall run: presence of index.json (written near the end) OR
    all task directories have metrics.json present.
  - Failure sentinel detection: presence of FAILED_ANALYSIS file or index.json.any_failure True.

Limitations:
  - Does not attempt to re-compute the config hash from the YAML; monitors existing filesystem only.
  - Progress estimation is exact once all task subdirectories are created (run_pipeline creates them
    before launching each analysis, so usually immediate after start for sequential mode).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_latest_run(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def detect_task_dirs(run_dir: Path) -> List[Path]:
    tasks = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        # Pattern heuristic: L{int}x{int}_dis{float with 3 decimals}
        if name.startswith("L") and "x" in name and "_dis" in name:
            tasks.append(p)
    tasks.sort()
    return tasks


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    tasks = detect_task_dirs(run_dir)
    total = len(tasks)
    completed = 0
    failures = 0
    running_task: Optional[Path] = None
    latest_mtime = 0.0
    for t in tasks:
        metrics_path = t / "metrics.json"
        stdout_path = t / "analysis_stdout.log"
        if metrics_path.exists():
            completed += 1
            # Determine if failure (returncode stored in parent index only; heuristically check for stderr severity)
            # If analysis_stdout.log exists but no metrics, treat as pending unless a stderr file indicates early abort.
        else:
            # Candidate for running
            if stdout_path.exists():
                mt = stdout_path.stat().st_mtime
                if mt > latest_mtime:
                    latest_mtime = mt
                    running_task = t
    index_path = run_dir / "index.json"
    index_data = None
    any_failure = None
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            any_failure = bool(index_data.get("any_failure"))
            # Recompute failures count accurately if index present
            failures = sum(1 for r in index_data.get("records", []) if not r.get("analysis_ok"))
            completed = len(index_data.get("records", []))
            total = index_data.get("task_count", total)
        except Exception:
            pass
    else:
        # Infer failures by scanning for stderr log with TIMEOUT marker and missing metrics
        for t in tasks:
            if not (t / "metrics.json").exists():
                stderr_path = t / "analysis_stderr.log"
                if stderr_path.exists():
                    txt = ""
                    try:
                        txt = stderr_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass
                    if "TIMEOUT" in txt or "Traceback" in txt:
                        failures += 1
    failed_sentinel = (run_dir / "FAILED_ANALYSIS").exists()
    if any_failure is None and failed_sentinel:
        any_failure = True
    progress = (completed / total) if total else 0.0
    status = "running"
    if index_path.exists() and completed >= total:
        status = "failed" if any_failure else "success"
    elif any_failure:
        status = "failed"
    return {
        "run_dir": str(run_dir),
        "task_total": total,
        "task_completed": completed,
        "task_failures": failures,
        "progress_fraction": progress,
        "progress_percent": progress * 100.0,
        "index_present": index_path.exists(),
        "any_failure": any_failure,
        "failed_sentinel": failed_sentinel,
        "status": status,
        "running_task": str(running_task) if running_task else None,
        "index": index_data,
    }


def format_bar(fraction: float, width: int = 40) -> str:
    done = int(width * fraction)
    return "[" + "#" * done + "." * (width - done) + f"] {fraction*100:5.1f}%"


def tail_file(path: Path, n: int) -> List[str]:
    if not path or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []


def poll(
    run_dir: Path, interval: float, follow: bool, tail: int, json_out: bool, show_active_tail: bool
):
    first = True
    while True:
        summary = summarize_run(run_dir)
        if json_out:
            # Compact JSON (avoid huge index replication) by removing full records when following
            data = dict(summary)
            if data.get("index") and follow:
                # Drop heavy records if present to keep console readable
                idx = dict(data["index"])
                if "records" in idx:
                    idx["records"] = f"<omitted {len(data['index']['records'])} records>"
                data["index"] = idx
            print(json.dumps(data, indent=2))
        else:
            if not first:
                # Separate updates visually
                print("-" * 72)
            print(f"Run: {summary['run_dir']}")
            print(
                f"Status: {summary['status']}  Completed: {summary['task_completed']}/{summary['task_total']}  Failures: {summary['task_failures']}"
            )
            print(format_bar(summary["progress_fraction"]))
            if summary["running_task"]:
                print(f"Active task: {summary['running_task']}")
                if show_active_tail and tail > 0:
                    stdout_path = Path(summary["running_task"]) / "analysis_stdout.log"
                    snippet = tail_file(stdout_path, tail)
                    if snippet:
                        print(f"-- Last {tail} lines of active stdout --")
                        for line in snippet:
                            print(line)
            elif summary["status"] == "success" and summary.get("index"):
                print("All tasks complete.")
            if summary["status"] in ("success", "failed") and summary["index_present"]:
                # Print gate flags briefly
                idx = summary.get("index") or {}
                if isinstance(idx, dict):
                    gate_keys = [
                        k for k in idx.keys() if k.endswith("_gate_ok") or k.endswith("_fraction")
                    ]
                    if gate_keys:
                        print("Gates:")
                        for k in sorted(gate_keys):
                            print(f"  {k}: {idx[k]}")
        if not follow:
            # Exit code semantics: 0 success, 1 failure, 2 running
            if summary["status"] == "success":
                sys.exit(0)
            if summary["status"] == "failed":
                sys.exit(1)
            sys.exit(2)
        # Follow mode
        if summary["status"] in ("success", "failed"):
            break
        first = False
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break


def parse_args():
    ap = argparse.ArgumentParser(
        description="Poll an automation pipeline run directory for progress."
    )
    ap.add_argument(
        "--root",
        type=str,
        default="automation_runs",
        help="Root directory containing run hash subdirectories.",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Specific run directory (overrides automatic latest detection).",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Polling interval seconds when --follow enabled.",
    )
    ap.add_argument("--follow", action="store_true", help="Continue polling until run completes.")
    ap.add_argument(
        "--tail", type=int, default=12, help="Number of lines to tail from active task stdout."
    )
    ap.add_argument("--no-active-tail", action="store_true", help="Disable tailing active stdout.")
    ap.add_argument(
        "--json", action="store_true", help="Emit JSON summary (single shot or each interval)."
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Perform a single poll iteration and exit with status code.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(root)
        if run_dir is None:
            print("[poll] No run directories found under", root)
            sys.exit(3)
    if not run_dir.exists():
        print(f"[poll] Run directory not found: {run_dir}")
        sys.exit(3)
    follow = args.follow and not args.once
    poll(run_dir, args.interval, follow, args.tail, args.json, not args.no_active_tail)


if __name__ == "__main__":
    main()
