"""Unfiltered categorical rollouts from the controlled Qwen non-thinking prompt."""

import hashlib
import json

import torch

from .model import autocast_policy
from .prompts import prompt_ids
from .schema import Response
from .verifier import verify


def stream_seed(seed, *labels):
    """Derive separate deterministic streams for training, evaluation and pools."""
    key = json.dumps([seed, *labels], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)


@torch.no_grad()
def sample_puzzle(model, tokenizer, puzzle, n, *, checkpoint_id, pool, seed,
                  microbatch=4, max_new_tokens=377):
    if min(n, microbatch, max_new_tokens) < 1:
        raise ValueError("Sample count, microbatch and token cap must be positive")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    prefix = prompt_ids(tokenizer, puzzle)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    stops = torch.tensor(eos_ids, device=device)
    rows = []
    for start in range(0, n, microbatch):
        batch = min(microbatch, n - start)
        tokens = torch.tensor(prefix, device=device, dtype=torch.long).expand(batch, -1)
        finished = torch.zeros(batch, device=device, dtype=torch.bool)
        cache = None
        sampled, selected_logprobs = [], []
        # All rows in a batch share the same prompt and need no left padding.
        with autocast_policy(model):
            for _ in range(max_new_tokens):
                output = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                probabilities = torch.softmax(output.logits[:, -1, :].float(), dim=-1)
                mass = probabilities.sum(1, dtype=torch.float64)
                if not torch.isfinite(mass).all():
                    raise FloatingPointError("Nonfinite sampling distribution")
                tokens = torch.multinomial(probabilities, 1, generator=generator)
                # Record the normalized categorical distribution passed to multinomial.
                selected = probabilities.gather(1, tokens).squeeze(1)
                logprobs = selected.double().log() - mass.log()
                sampled.append(tokens[:, 0])
                selected_logprobs.append(logprobs)
                finished |= torch.isin(tokens[:, 0], stops)
                if finished.all().item():
                    break
        token_lists = torch.stack(sampled, dim=1).cpu().tolist()
        logprob_lists = torch.stack(selected_logprobs, dim=1).cpu().tolist()
        for j, (ids, logprobs) in enumerate(zip(token_lists, logprob_lists)):
            stop = next((k for k, token in enumerate(ids) if token in eos_ids), None)
            length = len(ids) if stop is None else stop + 1
            ids = ids[:length]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            rows.append(Response(
                puzzle_id=puzzle.puzzle_id, prompt_ids=prefix, response_ids=ids,
                sampler_logprobs=logprobs[:length], response_mask=[1] * length,
                eos_mask=[int(token in eos_ids) for token in ids],
                reward=int(verify(text, puzzle.numbers, puzzle.target)),
                checkpoint_id=checkpoint_id, pool=pool,
                sample_id=f"{pool}/{puzzle.puzzle_id}/{seed}/{start + j}",
            ))
    return rows
