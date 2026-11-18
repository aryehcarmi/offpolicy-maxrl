"""Validated JSON configuration for controlled experiments."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model: str
    revision: str | None
    method: str
    seed: int
    lr: float | None
    updates: int
    prompts_per_step: int
    responses_per_prompt: int
    refresh_every: int
    eval_every: int
    validation_samples: int
    test_samples: int
    max_new_tokens: int
    checkpoint_steps: list[int]

    def __post_init__(self):
        if self.method not in {"fresh", "uncorrected", "count", "snis", "count_cap", "snis_cap", "loo_cap", "hybrid"}:
            raise ValueError(f"Unsupported controlled training method: {self.method}")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("Specify the initial model")
        if not Path(self.model).is_dir() and not self.revision:
            raise ValueError("Pin the initial model revision")
        if self.lr is not None and (type(self.lr) not in (float, int) or not math.isfinite(self.lr) or self.lr <= 0):
            raise ValueError("Learning rate must be positive and finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        for name in ("updates", "prompts_per_step", "responses_per_prompt", "refresh_every", "eval_every",
                     "validation_samples", "test_samples", "max_new_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.checkpoint_steps, list) or any(
            type(i) is not int or i < 1 or i > self.updates for i in self.checkpoint_steps
        ):
            raise ValueError("Checkpoint steps must lie within the training horizon")


def load_config(path, **overrides):
    document = json.loads(Path(path).read_text())
    document.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**document)


@dataclass(frozen=True)
class Runtime:
    device: str = "cuda"
    dtype: str = "float32"
    generation_batch: int = 8
    score_batch: int = 1
    hourly_usd: float | None = None

    def __post_init__(self):
        if self.generation_batch < 1 or self.score_batch < 1:
            raise ValueError("Microbatches must be positive")
        if self.hourly_usd is not None and (self.hourly_usd < 0 or not math.isfinite(self.hourly_usd)):
            raise ValueError("Provide a finite nonnegative hourly rate")

    @classmethod
    def from_args(cls, args):
        return cls(**{key: getattr(args, key) for key in cls.__dataclass_fields__})


def numerical_runtime(record, *, evaluation=False):
    """Settings that affect matched runs; device names and prices may differ."""
    keys = ("dtype", "generation_batch") if evaluation else ("dtype", "generation_batch", "score_batch")
    if not isinstance(record, dict) or any(key not in record for key in keys):
        raise ValueError(f"Missing numerical runtime metadata: expected {', '.join(keys)}")
    result = {key: record[key] for key in keys}
    if result["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Invalid runtime dtype")
    if any(type(result[key]) is not int or result[key] < 1 for key in keys if key != "dtype"):
        raise ValueError("Runtime batch sizes must be positive integers")
    return result


def runtime_arguments(parser):
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N for a single GPU")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="Forward dtype; master parameters and SGD use float32")
    parser.add_argument("--generation-batch", type=int, default=8)
    parser.add_argument("--score-batch", type=int, default=1)
    parser.add_argument("--hourly-usd", type=float, help="Total hourly price of this machine")
