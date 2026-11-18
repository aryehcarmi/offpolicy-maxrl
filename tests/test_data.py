from itertools import combinations

import pytest

from maxrl.data import load_panel, save_manifest, split_rows


def test_grouped_reproducible_panels(tmp_path):
    rows = [{"nums": list(range(1, n + 1)), "target": t}
            for n in (3, 4) for t in range(1, 201)]
    rows += [{"nums": [3, 2, 1], "target": 1}]
    reserve = {"validation": 8, "discovery": 8, "confirmation": 4, "test": 16}
    panels, n_rows = split_rows(rows, seed=7, reserve=reserve)
    assert split_rows(reversed(rows), 7, reserve)[0] == panels
    assert split_rows(rows, 8, reserve)[0]["test"] != panels["test"]
    ids = {name: {p.puzzle_id for p in rows} for name, rows in panels.items()}
    assert n_rows == 401
    assert [len(ids[n]) for n in ("train", "dev_population", "test_population")] == [360, 20, 20]
    for name in reserve:
        assert len(panels[name]) == reserve[name]
        assert sum(len(p.numbers) == 3 for p in panels[name]) == reserve[name] // 2
    for a, b in combinations(["train", *reserve], 2):
        assert ids[a].isdisjoint(ids[b])
    assert ids["validation"] | ids["discovery"] | ids["confirmation"] <= ids["dev_population"]
    assert ids["test"] <= ids["test_population"]
    save_manifest(tmp_path, panels, n_rows, 7, "fixture")
    manifest = tmp_path / "manifest.json"
    assert load_panel(manifest, "midpoint") == panels["discovery"][:8]
    (tmp_path / "test.jsonl").write_text("")
    with pytest.raises(ValueError, match="checksum"):
        load_panel(manifest, "test")


def test_insufficient_panel_and_invalid_puzzles():
    with pytest.raises(ValueError, match="requested"):
        split_rows([], 7)
    with pytest.raises(ValueError, match="invalid numbers"):
        split_rows([{"nums": [2, 3], "target": 6}], 7)
