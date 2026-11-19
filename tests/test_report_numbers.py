import subprocess
import sys
from pathlib import Path


def test_committed_report_matches_rebuild(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "report-numbers.md"
    subprocess.run([sys.executable, "-m", "maxrl.report_numbers", "--out", str(output)],
                   cwd=repo, check=True)
    assert output.read_bytes() == (repo / "results/midway/report-numbers.md").read_bytes()
