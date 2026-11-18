"""Prompt fidelity and artifact compatibility without model dependencies."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from maxrl import prompts


def test_messages_match_pinned_countdown_wording():
    puzzle = SimpleNamespace(numbers=[2, 3, 7], target=23, solution="not part of the input")
    assert prompts.prompt_messages(puzzle) == [
        {"role": "system", "content": (
            "You are a helpful assistant. You first thinks about the reasoning process "
            "in the mind and then provides the user with the answer."
        )},
        {"role": "user", "content": (
            " Using the numbers [2, 3, 7], create an equation that equals 23. "
            "You can use basic arithmetic operations (+, -, *, /) and each number "
            "can only be used once. Show your work in <think> </think> tags. "
            "And return the final answer in <answer> </answer> tags, for example "
            "<answer> (1 + 2) / 3 </answer>."
        )},
    ]


def test_prompt_tokenization_retains_non_thinking_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.options = messages, kwargs
            return [3, 1, 4]

    tokenizer = Tokenizer()
    puzzle = SimpleNamespace(numbers=[3, 3, 4], target=10)
    assert prompts.prompt_ids(tokenizer, puzzle) == [3, 1, 4]
    assert tokenizer.messages == prompts.prompt_messages(puzzle)
    assert tokenizer.options == {
        "tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
    }


def test_metadata_is_stable_and_does_not_expose_mutable_settings():
    first = prompts.prompt_metadata()
    assert first == prompts.prompt_metadata()
    assert first["id"] == "li-countdown-v1"
    assert len(first["sha256"]) == 64
    first["chat_template"]["enable_thinking"] = True
    assert prompts.prompt_metadata()["chat_template"]["enable_thinking"] is False


@pytest.mark.parametrize("field", ["SYSTEM_PROMPT", "USER_PROMPT", "TEMPLATE_OPTIONS"])
def test_prompt_digest_detects_wording_or_template_changes(monkeypatch, field):
    before = prompts.prompt_metadata()
    value = getattr(prompts, field)
    changed = {**value, "enable_thinking": True} if isinstance(value, dict) else value + " "
    monkeypatch.setattr(prompts, field, changed)
    assert prompts.prompt_metadata()["sha256"] != before["sha256"]
    with pytest.raises(ValueError, match="prompt metadata"):
        prompts.validate_prompt(before)


def test_prompt_validation_rejects_legacy_or_mismatched_metadata():
    current = prompts.prompt_metadata()
    assert prompts.validate_prompt(current) == current
    altered = deepcopy(current)
    altered["sha256"] = "0" * 64
    for metadata in (None, {}, {"id": current["id"]}, altered):
        with pytest.raises(ValueError, match="prompt metadata"):
            prompts.validate_prompt(metadata)
