"""Launch behavioral simulations for the GB10 top-level RTL."""
from __future__ import annotations

import subprocess
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    tb = REPO_ROOT / "tests" / "tb" / "gb10_top_tb.sv"
    cmd = [
        "verilator",
        "--cc",
        str(REPO_ROOT / "rtl" / "top" / "gb10_soc_top.sv"),
        "--exe",
        str(tb),
        "--build",
        "-Wall",
    ]
    print("Running", " ".join(cmd))
    subprocess.check_call(cmd)
    sim_exe = REPO_ROOT / "obj_dir" / "Vgb10_soc_top"
    if sim_exe.exists():
        subprocess.check_call([str(sim_exe)])


if __name__ == "__main__":
    main()
