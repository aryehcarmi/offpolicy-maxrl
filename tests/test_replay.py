"""Replay gates and independent one-step comparisons on an offline Qwen fixture."""

import csv
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from maxrl.config import Runtime, load_config
from maxrl.evaluate import record_groups
from maxrl.metrics import accuracy
from maxrl.model import checkpoint_id, load_policy, sequence_logprobs
from maxrl.prompts import prompt_metadata
from maxrl.replay import RATES, replay, rounding_stats, select_rate
from maxrl.schema import Response
from maxrl.train import update_policy


def gate_rows():
    return [{"rate": rate, "realized_over_predicted": 1., "heldout_fraction_within_0p1": 1.}
            for rate in RATES]


@pytest.mark.parametrize("minimum_ratio", [.8, 1.2])
def test_gate_uses_inclusive_thresholds_and_largest_eligible_rate(minimum_ratio):
    rows = gate_rows()
    rows[0]["realized_over_predicted"] = minimum_ratio
    rows[-3].update(realized_over_predicted=.5, heldout_fraction_within_0p1=.95)
    rows[-2]["heldout_fraction_within_0p1"] = .94999
    rows[-1]["realized_over_predicted"] = .49999
    assert select_rate(rows) == (rows[-3]["rate"], None)


@pytest.mark.parametrize("minimum_ratio", [.79999, 1.20001])
def test_gate_stops_when_small_rate_disagrees_even_if_another_passes(minimum_ratio):
    rows = gate_rows()
    rows[0]["realized_over_predicted"] = minimum_ratio
    chosen, reason = select_rate(rows)
    assert chosen is None and reason


def test_gate_stops_if_all_rates_fail_heldout_coverage():
    rows = gate_rows()
    for row in rows:
        row["heldout_fraction_within_0p1"] = .9
    chosen, reason = select_rate(rows)
    assert chosen is None and reason


def test_rounding_compares_actual_displacement_with_intended_fp64_step():
    parameter = torch.nn.Parameter(torch.tensor([2. ** 24, 1., 2., 42.], dtype=torch.float32))
    initial = parameter.detach().clone()
    parameter.grad = torch.tensor([.25, -1., 0., -.25], dtype=torch.float32)
    torch.optim.SGD([parameter], lr=1.).step()
    # 2²⁴ − 0.25 rounds back to 2²⁴, while the other nonzero displacements survive.
    actual = (parameter.detach().double() - initial.double()).numpy()
    intended = -parameter.grad.double().numpy()
    assert actual.tolist() == [0., 1., 0., .25]
    stats = rounding_stats([parameter], [initial], 1.)
    assert stats["rounding_projection_fraction"] == pytest.approx(
        np.dot(actual, intended) / np.dot(intended, intended))
    assert stats["rounding_norm_fraction"] == pytest.approx(
        np.linalg.norm(actual) / np.linalg.norm(intended))


def make_batch(base, identity, step, weights, rewards):
    groups = []
    pool = f"run/train/{step}"
    for group_idx in range(2):
        group = []
        puzzle = f"puzzle-{step}-{group_idx}"
        for i in range(2):
            token = 3 + (2 * group_idx + i + step) % 5
            row = Response(puzzle, [3, 4 + group_idx], [token], [-2.], [1], [0],
                           rewards[group_idx][i], identity, f"{pool}/{puzzle}/0/{i}", pool)
            with torch.no_grad():
                score = sequence_logprobs(base, [row]).item()
            # Assign known initial ratios so a test using uniform weights would fail.
            group.append(replace(row, sampler_logprobs=[score - math.log(weights[group_idx][i])]))
        groups.append(group)
    return groups


def log_weights(model, rows):
    with torch.no_grad():
        return np.array([sequence_logprobs(model, [r]).item() - sum(r.sampler_logprobs)
                         for r in rows])


def test_replay_matches_independent_trainer_steps_and_every_heldout_weight(tiny, tmp_path):
    base, _ = load_policy(tiny, device="cpu")
    identity = checkpoint_id(tiny)
    first = make_batch(base, identity, 1, [[.5, 1.5], [1, 1]], [[1, 1], [0, 0]])
    heldout = make_batch(base, identity, 2, [[.8, 1.25], [1.05, .99]], [[1, 0], [0, 1]])
    heldout_rows = [row for group in heldout for row in group]
    pilot, out = tmp_path / "pilot", tmp_path / "replay"
    pilot.mkdir()
    for step, groups in enumerate([first, heldout], 1):
        record_groups(pilot / "responses" / f"train-{step:06d}.jsonl", groups)
    cfg = replace(load_config(Path(__file__).parents[1] / "configs/pilot.json"),
                  model=tiny, revision=None, prompts_per_step=2, responses_per_prompt=2)
    runtime = Runtime(device="cpu")
    (pilot / "run.json").write_text(json.dumps({
        "run_id": "run", "initial_checkpoint_id": identity, "prompt": prompt_metadata(),
        "config": asdict(cfg), "runtime": asdict(runtime),
    }))
    source_model = deepcopy(base)
    optimizer = torch.optim.SGD(source_model.parameters(), lr=cfg.lr)
    events = []
    for i, groups in enumerate([first, heldout], 1):
        stats, _ = update_policy(source_model, optimizer, groups, "count", 1)
        events.append({"step": i, "split": "train", "sampler_id": identity,
                       "target_id": identity if i == 1 else "run/step-000001",
                       **accuracy(groups), **stats})
    (pilot / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    replay(pilot, out, device="cpu")
    summary = json.loads((out / "replay.json").read_text())
    assert summary["gradient_norm"] == pytest.approx(events[0]["gradient_norm"], rel=1e-6)
    with (out / "rates.csv").open() as stream:
        table = list(csv.DictReader(stream))
    items = [json.loads(line) for line in (out / "heldout.jsonl").read_text().splitlines()]
    assert [float(r["rate"]) for r in table] == list(RATES)
    assert len(items) == len(RATES) * len(heldout_rows)

    # Independent objective: only group 1 has successes. Average both successes,
    # then average over both prompts, including the prompt with zero successes.
    before = log_weights(base, first[0])
    for record in table:
        rate = float(record["rate"])
        stepped = deepcopy(base)
        update_policy(stepped, torch.optim.SGD(stepped.parameters(), lr=rate), first, "count", 1)
        after = log_weights(stepped, first[0])
        frozen_delta = np.dot(np.exp(before), after - before) / 4.
        count_delta = np.sum(np.exp(after) - np.exp(before)) / 4.
        assert float(record["predicted_change"]) == pytest.approx(rate * summary["gradient_norm"] ** 2)
        assert float(record["realized_change"]) == pytest.approx(frozen_delta, abs=1e-9)
        assert float(record["count_weight_objective_change"]) == pytest.approx(count_delta, abs=1e-9)

        values = log_weights(stepped, heldout_rows)
        observed = [r for r in items if r["rate"] == rate]
        assert {r["sample_id"] for r in observed} == {r.sample_id for r in heldout_rows}
        by_id = {r["sample_id"]: r for r in observed}
        for row, expected in zip(heldout_rows, values):
            saved = by_id[row.sample_id]
            assert (saved["puzzle_id"], saved["reward"]) == (row.puzzle_id, row.reward)
            assert saved["log_weight"] == pytest.approx(expected, abs=1e-9)
        for reward, column in [(0, "mean_weight_failure"), (1, "mean_weight_success")]:
            mask = [r.reward == reward for r in heldout_rows]
            assert float(record[column]) == pytest.approx(np.exp(values[mask]).mean())
        for q, label in [(0., "min"), (.025, "p025"), (.5, "p50"), (.975, "p975"), (1., "max")]:
            assert float(record[f"log_weight_{label}"]) == pytest.approx(np.quantile(values, q))
        assert float(record["heldout_fraction_within_0p1"]) == np.mean(np.abs(values) <= .1)
