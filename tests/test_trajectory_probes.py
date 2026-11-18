from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from maxrl.model import load_policy, sequence_logprobs
from maxrl.schema import Response
from maxrl.trajectory_probes import (
    attribution_records, first_and_total, is_estimate, logit_push,
    opening_logprobs, parameter_summary, parameter_type, validation_estimates,
)


def sample(i, token, reward, puzzle="A"):
    return Response(puzzle, [3, 4], [token, 5], [-2., -2.], [1, 1], [0, 0],
                    reward, "base", str(i), "pool")


def test_logit_push_has_trainer_prompt_denominator_and_per_prompt_probabilities():
    groups = [[sample(0, 3, 1), sample(1, 4, 1)],
              [sample(2, 3, 1, "B"), sample(3, 3, 0, "B")],
              [sample(4, 3, 0, "C"), sample(5, 3, 0, "C")]]
    log_weights = np.log([4., 1., 2., 100., 1., 1.])
    probabilities = [.1, .8, .05]
    for method, coeff in [
        ("count", [2., .5, 2.]), ("snis", [.8, .2, 1.]),
        ("uncorrected", [.5, .5, 1.]),
    ]:
        a, b, c = coeff
        expected_mass = (a+b+c)/3
        expected_open = (a+c)/3
        expected_prob = ((a+b)*.1 + c*.8)/3
        probe = logit_push(groups, method, log_weights, probabilities, 3)
        assert probe["direct_logit_push"] == pytest.approx(expected_open - expected_prob)
        assert probe["coefficient_mass"] == pytest.approx(expected_mass)
        assert probe["normalized_logit_push"] == pytest.approx((expected_open - expected_prob)/expected_mass)


def test_importance_estimator_is_ordinary_and_has_distinct_ess(monkeypatch):
    a, b = [sample(0, 3, 1), sample(1, 4, 0)], [sample(2, 3, 1), sample(3, 4, 1)]
    weights = [.1, 2., .9, 3.]
    monkeypatch.setattr("maxrl.trajectory_probes.score_log_weights",
                        lambda m, rows, bs: np.log([weights[int(r.sample_id)] for r in rows]))
    estimates = validation_estimates(None, {"initial_eval": a, "pilot_step0": b})
    assert estimates["initial_eval"]["pass_at_1"] == pytest.approx(.05)
    assert estimates["pilot_step0"]["pass_at_1"] == pytest.approx(1.95)
    assert estimates["pooled"]["pass_at_1"] == pytest.approx(1.)
    assert estimates["pooled"]["ess"] == pytest.approx(np.sum(weights)**2/np.dot(weights, weights))
    assert estimates["pooled"]["success_ess"] == pytest.approx(4.**2/(.1**2+.9**2+3**2))
    zero = is_estimate(np.array([-1000., -1000.]), np.zeros(2, dtype=bool))
    assert zero["pass_at_1"] == 0 and zero["success_ess"] == 0 and zero["ess"] == 2


def test_next_token_probabilities_and_sequence_decomposition_match_scorer(tiny):
    model, _ = load_policy(tiny, device="cpu")
    rows = [sample(0, 3, 1), replace(sample(1, 4, 0), response_mask=[1,0], sampler_logprobs=[-2.,0.])]
    before = first_and_total(model, rows)
    for i, row in enumerate(rows):
        p = opening_logprobs(model, [row.prompt_ids], row.response_ids[0])[0]
        assert before[i, 0] == pytest.approx(p, abs=1e-6)
        assert before[i, 1] == pytest.approx(sequence_logprobs(model, [row]).item(), abs=1e-6)
    delta = np.array([[-.5, -.4], [.1, 0.]])
    out = list(attribution_records(rows, before, before + delta, 3))
    assert out[0]["first_fraction_of_change"] == pytest.approx(1.25)
    assert out[0]["remaining_log_weight_change"] == pytest.approx(.1)
    assert out[0]["updated_log_weight"] - out[0]["initial_log_weight"] == pytest.approx(-.4)
    assert out[0]["updated_first_log_weight"] == pytest.approx(before[0, 0]-.5+2.)
    assert out[1]["first_fraction_of_change"] is None


def test_parameter_gradient_shares_count_tied_weights_once(tiny):
    model, _ = load_policy(tiny, device="cpu")
    model.config.tie_word_embeddings = True
    model.tie_weights()
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    initial = [p.detach().clone() for p in model.parameters()]
    p = model.get_input_embeddings().weight
    p.grad = torch.zeros_like(p)
    p.grad.flatten()[0] = 1.
    norm = model.model.norm.weight
    norm.grad = torch.zeros_like(norm)
    norm.grad.flatten()[0] = 2.
    torch.optim.SGD(model.parameters(), lr=.1).step()
    result = parameter_summary(model, initial)
    assert set(result) == {"embedding","norm","mlp","attention"}
    assert result["embedding"]["grad_squared_norm_share"] == pytest.approx(.2)
    assert result["norm"]["grad_squared_norm_share"] == pytest.approx(.8)
    assert result["embedding"]["parameter_count"] == p.numel()
    assert result["embedding"]["fraction_changed_within_type"] == pytest.approx(1/p.numel())
    assert result["norm"]["share_of_all_changed_parameters"] == .5
    assert sum(r["parameter_count"] for r in result.values()) == sum(p.numel() for p in model.parameters())
    assert parameter_type("model.layers.0.self_attn.k_norm.weight") == "norm"
