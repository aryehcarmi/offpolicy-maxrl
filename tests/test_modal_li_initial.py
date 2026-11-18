"""Offline launcher contracts. All child processes run against temporary local fixtures."""

from datetime import datetime, timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

import pytest


REPO = Path(__file__).resolve().parents[1]


def module_at(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


class LocalVolume:
    """Read back the persisted bytes as Modal Volume.read_file does."""

    def __init__(self, directory):
        self.directory = directory
        self.commits = 0
        self.reads = []
        self.persisted = {}

    def commit(self):
        self.commits += 1
        self.persisted = {str(p.relative_to(self.directory)): p.read_bytes()
                          for p in self.directory.rglob('*') if p.is_file()}

    def read_file(self, path):
        self.reads.append(str(path).lstrip('/'))
        contents = self.persisted[self.reads[-1]]
        yield contents[:4]
        yield contents[4:]


def write_code(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    if executable:
        path.chmod(0o755)


@pytest.fixture
def local_attempt(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'modal', None)
    core = module_at(REPO / 'scripts/modal_li_initial.py')
    upstream = tmp_path / 'upstream'
    python = upstream / '.venv-tim/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    spec = json.loads((REPO / 'configs/replication/li_frozen.json').read_text())
    config = tmp_path / 'repo/configs/replication/li_frozen.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(spec))
    wrapper = tmp_path / 'repo/scripts/li_replication.py'
    write_code(wrapper, '''
        import argparse
        import json
        import os
        from pathlib import Path

        CONFIG = Path(__file__).resolve().parents[1] / 'configs/replication/li_frozen.json'
        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument('action', choices=['evaluate'])
            parser.add_argument('--method', choices=['initial'], required=True)
            parser.add_argument('--out', type=Path, required=True)
            parser.add_argument('--upstream', type=Path, required=True)
            args = parser.parse_args()
            print('entered fixture evaluation', flush=True)
            mode = os.environ.get('FIXTURE_LI_STOP', '')
            if mode == 'early':
                raise RuntimeError('Failed to infer device type before wrapper setup')
            if mode in ('device', 'unrelated_import'):
                from vllm.config.device import DeviceConfig
                DeviceConfig()
            spec = json.loads(CONFIG.read_text())
            result = {'method':'initial', 'mean_accuracy':0.125, 'published_fraction':0.082,
                      'difference_percentage_points':4.3, 'n_completions':512,
                      'upstream_commit':spec['upstream_commit'], 'model_revision':spec['revision']}
            dest = args.out / 'initial/evaluation.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(result))
        if __name__ == '__main__':
            main()
    ''')
    bins = tmp_path / 'bin'
    write_code(bins / 'git', f'''\
        #!/bin/sh
        case " $* " in
          *" rev-parse "*) echo {spec['upstream_commit']} ;;
          *" diff "*) echo '+name = "verl-main-cu129-env"' ;;
          *) exit 1 ;;
        esac
    ''', executable=True)
    write_code(bins / 'nvidia-smi', '''\
        #!/bin/sh
        echo 'NVIDIA H100, 81000, 570.0'
    ''', executable=True)
    monkeypatch.setenv('PATH', str(bins) + os.pathsep + os.environ['PATH'])
    libs = upstream
    write_code(libs / 'vllm/__init__.py', "__version__ = 'fixture'\n")
    write_code(libs / 'vllm/config/__init__.py', '')
    write_code(libs / 'vllm/config/device.py', '''
        import os

        class DeviceConfig:
            def __init__(self):
                self.__post_init__()

            def __post_init__(self):
                if os.environ.get('FIXTURE_LI_STOP') == 'unrelated_import':
                    import fixture_missing_distribution
                raise RuntimeError('Fixture platform has no assigned accelerator')
    ''')
    write_code(libs / 'torch/__init__.py', """
        import os
        __version__ = 'fixture'
        class cuda:
            @staticmethod
            def is_available():
                return os.environ.get('FIXTURE_LI_CUDA') == '1'
    """)
    volume = LocalVolume(tmp_path / 'volume')
    return core, upstream, wrapper, volume


def attempt(fixture, **kwargs):
    core, upstream, wrapper, volume = fixture
    return core.run_attempt(upstream, wrapper, volume.directory, volume.commit, **kwargs)


def test_two_evaluations_persist_distinct_results_and_summarize(local_attempt, tmp_path, capsys):
    core, _, _, volume = local_attempt
    first = attempt(local_attempt, hourly_usd=2.5)
    second = attempt(local_attempt)
    assert first['run_id'] != second['run_id']
    for record in (first, second):
        assert record['status'] == 'completed' and record['kind'] == 'evaluation'
        assert record['evaluation']['mean_accuracy'] == 0.125
        assert record['evaluation']['difference_percentage_points'] == pytest.approx(4.3)
        assert record['wall_seconds'] >= 0
        for key in ('started_utc', 'finished_utc'):
            assert datetime.fromisoformat(record[key]).utcoffset() == timedelta(0)
        assert 'H100' in record['gpu_name'] and record['gpu_memory_mib'] == 81000
        assert record['driver_version'] == '570.0'
        assert '+name = "verl-main-cu129-env"' in record['upstream_diff']
        assert record['upstream_commit'] == json.loads((REPO / 'configs/replication/li_frozen.json').read_text())['upstream_commit']
        assert json.loads(volume.persisted[record['record_path']]) == record
        assert datetime.fromisoformat(record['started_utc']).strftime('%Y%m%d') in record['run_id']
        assert 'entered fixture evaluation' in (volume.directory / record['log_path']).read_text()
        path = record['run_id'] + '/initial/evaluation.json'
        assert json.loads(volume.persisted[path]) == record['evaluation']
    assert volume.commits >= 2
    assert first['estimated_cost_usd'] == pytest.approx(first['wall_seconds'] * 2.5 / 3600)
    assert second['estimated_cost_usd'] is None
    repo = tmp_path / 'local-materialization'
    capsys.readouterr()
    core.finish_local(first, repo, volume)
    saved = repo / 'results/midway' / f"li-initial-{first['run_id']}.json"
    assert json.loads(saved.read_text()) == first
    output = capsys.readouterr().out
    assert '12.5' in output and '8.2' in output and 'H100' in output and 'wall' in output.lower()


def test_evaluation_failure_commits_diagnostics_and_raises(local_attempt, monkeypatch):
    _, _, _, volume = local_attempt
    monkeypatch.setenv('FIXTURE_LI_STOP', 'early')
    with pytest.raises(RuntimeError) as exc:
        attempt(local_attempt)
    records = [json.loads(p.read_text()) for p in volume.directory.rglob('*.json')]
    [failed] = [r for r in records if r.get('status') == 'failed']
    assert json.loads(volume.persisted[failed['record_path']]) == failed
    assert 'li-replication' in str(exc.value) and failed['log_path'] in str(exc.value)
    log = volume.persisted[failed['log_path']].decode()
    assert 'Failed to infer device type before wrapper setup' in log
    assert volume.commits >= 1


@pytest.mark.parametrize('stop, passed', [('device', True), ('early', False),
                                        ('unrelated_import', False), ('unexpected_success', False)])
def test_preflight_requires_real_device_frame(local_attempt, tmp_path, monkeypatch, stop, passed):
    core, _, _, volume = local_attempt
    monkeypatch.setenv('FIXTURE_LI_STOP', stop)
    record = attempt(local_attempt, preflight=True)
    assert record['kind'] == 'preflight'
    assert (record['status'] == 'completed') is passed
    assert 'evaluation' not in record
    assert not list(volume.directory.rglob('evaluation.json'))
    assert volume.commits >= 1
    repo = tmp_path / 'preflight-laptop'
    if passed:
        assert 'device' in str(record['stopping_point']).lower()
        assert 'Fixture platform has no assigned accelerator' in record['stopping_point']
        core.finish_local(record, repo, volume)
    else:
        if stop == 'unexpected_success':
            assert record['wrapper_returncode'] == 0
        with pytest.raises(RuntimeError):
            core.finish_local(record, repo, volume)
    assert {record['record_path'], record['log_path']} <= set(volume.reads)
    assert not (repo / 'results').exists()
    if passed:
        volume.persisted[record['log_path']] = b'wrong persisted log'
        with pytest.raises(RuntimeError, match='[Vv]olume'):
            core.finish_local(record, repo, volume)


def test_missing_upstream_interpreter_is_persisted(local_attempt):
    _, upstream, _, volume = local_attempt
    (upstream / '.venv-tim/bin/python').unlink()
    with pytest.raises(RuntimeError):
        attempt(local_attempt)
    [record_path] = list(volume.directory.rglob('run.json'))
    record = json.loads(record_path.read_text())
    assert record['status'] == 'failed'
    assert (volume.directory / record['log_path']).read_text().strip()
    assert volume.commits >= 1


@pytest.mark.parametrize('hourly', [-0.01, float('nan'), float('inf')])
def test_invalid_hourly_price_is_rejected(local_attempt, hourly):
    with pytest.raises(ValueError):
        attempt(local_attempt, hourly_usd=hourly)


def test_rename_project_changes_only_the_project_name(tmp_path):
    import subprocess

    project = tmp_path / 'envs/verl-main-cu129'
    project.mkdir(parents=True)
    manifest = project / 'pyproject.toml'
    old = b'[project]\nname = "tim-verl-env"\nversion = "0.0.1"\ndependencies = ["verl==0.1"]\n'
    manifest.write_bytes(old)
    lock = project / 'uv.lock'
    lock.write_bytes(b'version = 1\n')
    cmd = [sys.executable, str(REPO / 'scripts/modal_li_initial.py'), '--rename-project', str(tmp_path)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp_path)
    expected = old.replace(b'"tim-verl-env"', b'"verl-main-cu129-env"')
    assert manifest.read_bytes() == expected
    assert subprocess.run(cmd, capture_output=True, cwd=tmp_path).returncode != 0
    assert manifest.read_bytes() == expected
    assert lock.read_bytes() == b'version = 1\n'


@pytest.mark.parametrize('local', [True, False], ids=['laptop', 'container'])
def test_modal_app_imports_in_both_contexts(monkeypatch, tmp_path, capsys, local):
    modal = pytest.importorskip('modal')
    from importlib.metadata import version
    assert version('modal') == '1.6.4'
    monkeypatch.setattr(modal, 'is_local', lambda: local)
    if not local:
        def no_local_mount(*args, **kwargs):
            pytest.fail('container-side import attempted to add a laptop source file')
        monkeypatch.setattr(modal.Image, 'add_local_file', no_local_mount)
        monkeypatch.setattr(modal.Image, 'add_local_dir', no_local_mount)
    launcher = module_at(REPO / 'scripts/modal_li_initial.py')
    assert set(launcher.app.registered_entrypoints) == {'main'}
    assert set(launcher.app.registered_functions) == {'evaluate_initial', 'preflight_initial'}
    if local:
        def unexpected_remote(*args, **kwargs):
            pytest.fail('invalid price reached the remote dispatch')
        monkeypatch.setattr(launcher.evaluate_initial, 'remote', unexpected_remote)
        with pytest.raises(ValueError):
            launcher.main(hourly_usd=-1)
        # Dispatch through the registered entry point, with no network or account.
        record = {'run_id': 'fixture-run', 'kind': 'evaluation', 'status': 'completed',
                  'evaluation': {'mean_accuracy': .125, 'published_fraction': .082},
                  'gpu_name': 'NVIDIA H100', 'wall_seconds': 31.0}
        calls = []
        def evaluate(rate):
            calls.append(rate)
            return record
        monkeypatch.setattr(launcher.evaluate_initial, 'remote', evaluate)
        monkeypatch.setattr(launcher.preflight_initial, 'remote', unexpected_remote)
        monkeypatch.setattr(launcher, '__file__', str(tmp_path / 'scripts/modal_li_initial.py'))
        launcher.main(hourly_usd=2.5)
        assert calls == [2.5]
        saved = tmp_path / 'results/midway/li-initial-fixture-run.json'
        assert json.loads(saved.read_text()) == record
        output = capsys.readouterr().out
        assert all(value in output for value in ('12.5', '8.2', 'H100', '31.00'))



def test_preflight_with_visible_cuda_fails_before_evaluation(local_attempt, monkeypatch):
    _, _, _, volume = local_attempt
    monkeypatch.setenv('FIXTURE_LI_CUDA', '1')
    monkeypatch.setenv('FIXTURE_LI_STOP', 'device')
    record = attempt(local_attempt, preflight=True)
    assert record['status'] == 'failed'
    assert 'entered fixture evaluation' not in volume.persisted[record['log_path']].decode()
    assert not list(volume.directory.rglob('evaluation.json'))



def test_cancellation_commits_failed_attempt(local_attempt, monkeypatch):
    core, _, _, volume = local_attempt
    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt('cancelled fixture process')
    monkeypatch.setattr(core, '_logged_command', cancelled)
    with pytest.raises(RuntimeError) as exc:
        attempt(local_attempt)
    [key] = [path for path in volume.persisted if path.endswith('/run.json')]
    record = json.loads(volume.persisted[key])
    assert record['status'] == 'failed' and 'KeyboardInterrupt' in record['error']
    assert record['log_path'] in str(exc.value) and 'li-replication' in str(exc.value)
    assert 'cancelled fixture process' in volume.persisted[record['log_path']].decode()
