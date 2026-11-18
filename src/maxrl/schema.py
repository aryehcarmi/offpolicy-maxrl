"""Versioned, replayable response records. Token-level arrays align to response_ids."""
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Response:
    puzzle_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    sampler_logprobs: list[float]
    response_mask: list[int]
    eos_mask: list[int]
    reward: int
    checkpoint_id: str
    sample_id: str
    pool: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported response schema version")
        if not all(isinstance(s, str) and s.strip() for s in
                   (self.puzzle_id, self.checkpoint_id, self.sample_id, self.pool)):
            raise ValueError("missing puzzle/sample/pool/checkpoint identity")
        for ids in (self.prompt_ids, self.response_ids):
            if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("expected nonempty lists of token IDs")
        size = len(self.response_ids)
        if any(len(a) != size for a in (self.sampler_logprobs, self.response_mask, self.eos_mask)):
            raise ValueError("per-token fields must have the same length")
        for mask in (self.response_mask, self.eos_mask):
            if any(type(v) not in (bool, int) or v not in (0, 1) for v in mask):
                raise ValueError("masks must be binary")
        used = sum(self.response_mask)
        if used == 0 or self.response_mask != [1] * used + [0] * (size - used):
            raise ValueError("response mask must mark a nonempty contiguous prefix")
        if sum(self.eos_mask) and (sum(self.eos_mask) != 1 or not self.eos_mask[used - 1]):
            raise ValueError("only the final scored token may be EOS")
        if type(self.reward) not in (bool, int, float) or self.reward not in (0, 1):
            raise ValueError("reward must be binary")
        if any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in self.sampler_logprobs):
            raise ValueError("expected finite natural-log sampler probabilities <= 0")
        if any(p != 0 for p in self.sampler_logprobs[used:]):
            raise ValueError("padding has zero log probability contribution")


def read_responses(path: Path | str) -> Iterator[Response]:
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                data = json.loads(line)
                if "schema_version" not in data:
                    raise ValueError("missing schema_version")
                response = Response(**data)
                if response.sample_id in seen:
                    raise ValueError("duplicate sample_id")
                seen.add(response.sample_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            yield response


def write_responses(path: Path | str, responses: Iterable[Response]) -> None:
    """Stream a complete shard, then atomically publish it without replacing another run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    seen = set()
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            for response in responses:
                response.__post_init__()  # list fields may have been mutated since construction
                if response.sample_id in seen:
                    raise ValueError(f"duplicate sample_id {response.sample_id}")
                seen.add(response.sample_id)
                stream.write(json.dumps(asdict(response), allow_nan=False, separators=(",", ":")) + "\n")
            if not seen:
                raise ValueError("cannot publish an empty shard")
            stream.close()
            os.link(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
