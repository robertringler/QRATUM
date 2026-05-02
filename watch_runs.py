from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from typing import List


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def all_exist(files: List[str]) -> bool:
    return all(os.path.exists(f) for f in files)


def try_build_pdf(tex: str = "blueprint_master_alternate.tex") -> None:
    try:
        # Single pass is fine for auto-included snippets; rerun can be manual if needed.
        subprocess.run(["xelatex", "-halt-on-error", "-interaction=nonstopmode", tex], check=False)
    except Exception as e:
        print(f"[{ts()}] PDF build error: {e}")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.chdir(root)

    dm_prefix = "L16"
    traj_prefix = "L16_traj"
    dm_files = [
        f"{dm_prefix}_gamma_sweep.csv",
        f"{dm_prefix}_gamma_sweep.npz",
        f"{dm_prefix}_D2.png",
        f"{dm_prefix}_popt.png",
    ]
    traj_files = [
        f"{traj_prefix}_gamma_sweep.csv",
        f"{traj_prefix}_gamma_sweep.npz",
        f"{traj_prefix}_D2.png",
        f"{traj_prefix}_popt.png",
    ]

    status_path = os.path.join(root, "watch_status.json")
    status = {
        "dm_done": False,
        "traj_done": False,
        "last_update": None,
    }
    print(f"[{ts()}] Watcher started. Monitoring artifacts in {root}.")
    print(f"[{ts()}] Waiting for: {dm_files} and/or {traj_files}")

    start = time.time()
    timeout_s = 3 * 60 * 60  # 3 hours
    poll_s = 20

    built_after_dm = False
    built_after_traj = False

    while True:
        now = ts()
        if not status["dm_done"] and all_exist(dm_files):
            status["dm_done"] = True
            status["last_update"] = now
            print(f"[{now}] Full 16x16 run artifacts detected (density-matrix).")
            if not built_after_dm:
                print(f"[{now}] Rebuilding PDF…")
                try_build_pdf()
                built_after_dm = True
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)

        if not status["traj_done"] and all_exist(traj_files):
            status["traj_done"] = True
            status["last_update"] = now
            print(f"[{now}] Fast 16x16 trajectory artifacts detected.")
            if not built_after_traj:
                print(f"[{now}] Rebuilding PDF…")
                try_build_pdf()
                built_after_traj = True
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)

        if status["dm_done"] and status["traj_done"]:
            print(
                f"[{now}] ALL DONE — both full and trajectory runs finished. Artifacts present. ✔"
            )
            break

        if time.time() - start > timeout_s:
            print(
                f"[{now}] Watcher timeout reached. Current status: dm_done={status['dm_done']} traj_done={status['traj_done']}"
            )
            break

        time.sleep(poll_s)


if __name__ == "__main__":
    main()
