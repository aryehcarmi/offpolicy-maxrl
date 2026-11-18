from dataclasses import asdict, replace
import json

import pytest

from maxrl.schema import Response, read_responses, write_responses


@pytest.fixture
def row():
    return Response("p", [2, 3], [4, 1, 0], [-2.4, -0.5, 0], [1, 1, 0], [0, 1, 0],
                    1, "Qwen@commit", "s0", "reference")


def test_jsonl_roundtrip_and_publication(tmp_path, row):
    path = tmp_path / "responses.jsonl"
    write_responses(path, [row])
    assert list(read_responses(path)) == [row]
    with pytest.raises(FileExistsError):
        write_responses(path, [row])
    failed = tmp_path / "other.jsonl"
    with pytest.raises(ValueError, match="duplicate"):
        write_responses(failed, [row, row])
    assert list(tmp_path.iterdir()) == [path]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        list(read_responses(path))


@pytest.mark.parametrize("changes", [
    {"response_ids": []}, {"sampler_logprobs": [-1]}, {"prompt_ids": [-1]},
    {"response_mask": [0, 1, 0]}, {"response_mask": [1, 0, 1]}, {"response_mask": [0, 0, 0]},
    {"eos_mask": [1, 0, 0]}, {"eos_mask": [0, 0, 1]}, {"eos_mask": [1, 1, 0]},
    {"sampler_logprobs": [float("nan"), -2, 0]}, {"sampler_logprobs": [9, -2, 0]},
    {"sampler_logprobs": [-1, -2, -2]}, {"checkpoint_id": ""}, {"reward": -1}, {"schema_version": 2},
])
def test_reject_invalid(row, changes):
    with pytest.raises(ValueError):
        replace(row, **changes)


def test_reader_requires_version(tmp_path, row):
    path = tmp_path / "rows.jsonl"
    data = asdict(row)
    del data["schema_version"]
    path.write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        list(read_responses(path))
