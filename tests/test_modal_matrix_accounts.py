"""Exercise two independent Modal Volumes feeding the same planner checkout."""
import json
from unittest.mock import Mock

import pytest

from test_modal_countdown import write_data_fixture
from test_modal_matrix import LocalVolume, cli_lines, cluster, env, planner_fixtures


def test_two_volume_collect_and_main_freeze(cluster, tmp_path, monkeypatch):
    checkout = cluster.env
    first = checkout.volume
    second = LocalVolume(tmp_path / "second volume")
    assert write_data_fixture(second.root / "data") == checkout.manifest_hash
    for stage in ("screen", "tune"):
        planner_fixtures(checkout, tmp_path, stage)

    # Distribute the recorded tuning results exactly as the two accounts ran them.
    expected = {}
    for marker in first.root.glob("matrix-*/attempt.json"):
        trial = json.loads(marker.read_bytes())["trial"]
        expected[trial] = marker.read_bytes()
        if trial.startswith(("tune/count-", "tune/snis-")):
            marker.parent.rename(second.root / marker.parent.name)

    def select(volume):
        checkout.volume = volume
        checkout.root = volume.root
        monkeypatch.setattr(checkout.app, "volume", volume)
        monkeypatch.setattr(checkout.app, "VOLUME_ROOT", volume.root)

    select(first)
    cluster.launch("collect")
    # The first account has the screen, but only its own half of tuning.
    cluster.launch("tune", methods="fresh,uncorrected", dry_run=True)
    assert len(cli_lines(checkout, "tune")) == 12
    with pytest.raises(RuntimeError, match="freeze"):
        cluster.launch("main", methods="fresh,uncorrected")
    local_freeze = checkout.matrix / "frozen.json"
    assert not local_freeze.exists()

    select(second)
    cluster.launch("collect")
    select(first)
    cluster.launch("collect")
    assert {str(path.parent.relative_to(checkout.matrix)): path.read_bytes()
            for path in checkout.matrix.glob("*/*/attempt.json")} == expected
    assert len(cli_lines(checkout, "tune")) == 12

    cluster.launch("main", methods="fresh,uncorrected", dry_run=True)
    assert not local_freeze.exists()
    assert not (first.root / "matrix/frozen.json").exists()
    assert not (first.root / "matrix/methods.json").exists()
    assert not cluster.launched

    planner = Mock(wraps=checkout.app._run_planner)
    monkeypatch.setattr(checkout.app, "_run_planner", planner)
    cluster.launch("main", methods="fresh,uncorrected")
    frozen = local_freeze.read_bytes()
    assert (first.root / "matrix/frozen.json").read_bytes() == frozen
    assert not (second.root / "matrix/frozen.json").exists()
    assert len(cluster.launched) == 6

    select(second)
    cluster.launch("main", methods="count,snis")
    assert (second.root / "matrix/frozen.json").read_bytes() == local_freeze.read_bytes() == frozen
    assert [call.args[1] for call in planner.call_args_list].count("freeze") == 1
    assert len(cluster.launched) == 16
