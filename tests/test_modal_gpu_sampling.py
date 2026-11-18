"""GPU monitoring uses a mocked nvidia-smi and temporary attempt files."""
import csv
from datetime import datetime, timezone
import subprocess
import threading

import pytest

from test_modal_countdown import import_launcher


def test_gpu_sampling_writes_timed_rows_and_closes_on_job_failure(tmp_path, monkeypatch):
    launcher = import_launcher()
    path = tmp_path / "gpu.csv"
    calls = []
    second_sample = threading.Event()

    def query(command, *, text, timeout, stderr):
        assert command == ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                           "--format=csv,noheader,nounits"]
        assert text and timeout > 0
        calls.append(command)
        if len(calls) == 2:
            second_sample.set()
        return "71, 512, 23028\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", query)
    with pytest.raises(ValueError, match="experiment failed"):
        with launcher._sample_gpu(path, pytest.fail, interval=0.005):
            assert second_sample.wait(2)
            raise ValueError("experiment failed")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(calls) >= 2
    assert all(datetime.fromisoformat(row["timestamp_utc"]).tzinfo == timezone.utc for row in rows)
    assert all([row["gpu_utilization_percent"], row["memory_used_mib"], row["memory_total_mib"]]
               == ["71", "512", "23028"] for row in rows)


def test_monitor_failure_does_not_cancel_experiment(tmp_path, monkeypatch):
    launcher = import_launcher()
    warnings = []

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(launcher.subprocess, "check_output", timeout)
    with launcher._sample_gpu(tmp_path / "gpu.csv", warnings.append):
        pass
    assert len(warnings) == 1 and "timed out" in warnings[0]
