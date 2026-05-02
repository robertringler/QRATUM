import subprocess
import sys
from pathlib import Path


def test_run_demo_creates_output(tmp_path, capsys):
    repo_root = Path(__file__).resolve().parents[1]
    demo_script = repo_root / "demo" / "run_demo.py"
    out_dir = repo_root / "demo" / "output"
    # Run the demo script in-process
    res = subprocess.run([sys.executable, str(demo_script)], capture_output=True, text=True)
    assert res.returncode == 0, f"Demo failed: {res.stderr}"
    assert out_dir.exists(), "output directory not created"
    checks = out_dir / "checks.json"
    assert checks.exists(), "checks.json missing"
    # new outputs
    stats = out_dir / "stats_summary.json"
    findings = out_dir / "findings.md"
    assert stats.exists(), "stats_summary.json missing"
    assert findings.exists(), "findings.md missing"
