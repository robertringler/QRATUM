import json
import subprocess
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "data" / "export_dir"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_cli_produces_outputs(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    summary_path = tmp_path / "summary.csv"
    cmd = [
        "python",
        "-m",
        "chatgpt_scraper",
        "--export-path",
        str(FIXTURE_DIR),
        "--out-ledger",
        str(ledger_path),
        "--out-summary",
        str(summary_path),
    ]
    result = subprocess.run(cmd, cwd=PACKAGE_ROOT, capture_output=True, text=True, check=True)
    assert "Processed" in result.stdout
    assert ledger_path.exists()
    assert summary_path.exists()
    with ledger_path.open() as fh:
        lines = fh.readlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["role"] == "system"
