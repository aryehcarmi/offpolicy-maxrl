"""Load the controlled Qwen policy and score complete responses by teacher forcing."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import prompt_metadata


def checkpoint_id(checkpoint, revision=None):
    path = Path(checkpoint)
    if not path.is_dir():
        if not revision:
            raise ValueError("Remote checkpoints require a pinned revision")
        return f"{checkpoint}@{revision}"
    marker = path / "maxrl_checkpoint.json"
    if marker.is_file():
        return json.loads(marker.read_text())["checkpoint_id"]
    # Identity for imported local models; saved project checkpoints carry a marker.
    files = sorted(p for p in path.iterdir() if p.suffix in {".safetensors", ".bin", ".json"})
    if not files:
        raise ValueError(f"No model files in {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        with item.open("rb") as stream:
            while block := stream.read(1 << 20):
                digest.update(block)
    return f"local-sha256-{digest.hexdigest()}"


def load_policy(checkpoint, *, device="cuda", dtype="float32", revision=None):
    """Keep FP32 master parameters; dtype selects forward autocast only."""
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"} or dtype not in {"float32", "bfloat16"}:
        raise ValueError("Use cpu/cuda and float32/bfloat16")
    if device.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected CUDA hardware does not support bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, revision=revision, dtype=torch.float32, attn_implementation="sdpa",
    ).to(device).eval()
    model._maxrl_dtype = dtype
    return model, tokenizer


def autocast_policy(model):
    if getattr(model, "_maxrl_dtype", "float32") == "float32":
        return nullcontext()
    return torch.autocast(device_type=next(model.parameters()).device.type, dtype=torch.bfloat16)


def sequence_logprobs(model, rows):
    """Differentiable response-only log probabilities, including sampled EOS."""
    if not rows:
        raise ValueError("At least one saved response is required")
    device = next(model.parameters()).device
    lengths = [len(r.prompt_ids) + len(r.response_ids) for r in rows]
    ids = torch.zeros((len(rows), max(lengths)), device=device, dtype=torch.long)
    attention = torch.zeros_like(ids)
    scored = torch.zeros_like(ids, dtype=torch.bool)
    for i, (row, length) in enumerate(zip(rows, lengths)):
        boundary = len(row.prompt_ids)
        ids[i, :length] = torch.tensor([*row.prompt_ids, *row.response_ids], device=device)
        attention[i, :length] = 1
        scored[i, boundary:length] = torch.tensor(row.response_mask, dtype=torch.bool, device=device)
    with autocast_policy(model):
        logits = model(input_ids=ids[:, :-1], attention_mask=attention[:, :-1], use_cache=False).logits
    # Compute the normalization in FP32 and accumulate log probabilities in FP64.
    targets = ids[:, 1:].masked_fill(~scored[:, 1:], -100)
    nll = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    )
    return -nll.view_as(targets).sum(dim=1, dtype=torch.float64)


def save_policy(model, tokenizer, path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # Publish the checkpoint only after the model and tokenizer files are written.
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / "checkpoint"
        model.save_pretrained(staged)
        tokenizer.save_pretrained(staged)
        (staged / "maxrl_checkpoint.json").write_text(json.dumps({
            "checkpoint_id": identity, "prompt": prompt_metadata(),
        }) + "\n")
        staged.rename(path)
