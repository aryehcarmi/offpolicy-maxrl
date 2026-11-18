"""Pinned Countdown messages and chat-template settings."""

import hashlib
import json


PROMPT_ID = "li-countdown-v1"
PROMPT_SOURCE = (
    "https://github.com/Skylanding/Training-Inference-Mismatch/blob/"
    "98e8f9fa15200f294c9395626969f745771ee460/tim/tasks/countdown.py"
)

# Original prompt wording, including punctuation; see THIRD_PARTY_NOTICES.md.
SYSTEM_PROMPT = (
    "You are a helpful assistant. You first thinks about the "
    "reasoning process in the mind and then provides the user "
    "with the answer."
)
USER_PROMPT = (
    " Using the numbers {numbers}, create an equation that equals "
    "{target}. You can use basic arithmetic operations (+, -, *, /) "
    "and each number can only be used once. Show your work in "
    "<think> </think> tags. And return the final answer in "
    "<answer> </answer> tags, for example <answer> (1 + 2) / 3 "
    "</answer>."
)
TEMPLATE_OPTIONS = {
    "tokenize": True,
    "add_generation_prompt": True,
    "enable_thinking": False,
}


def prompt_messages(puzzle):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(
            numbers=list(puzzle.numbers), target=puzzle.target,
        )},
    ]


def prompt_ids(tokenizer, puzzle):
    return tokenizer.apply_chat_template(prompt_messages(puzzle), **TEMPLATE_OPTIONS)


def prompt_metadata():
    """Identify the wording and template options used to collect responses."""
    specification = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "chat_template": TEMPLATE_OPTIONS,
    }
    payload = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    return {
        "id": PROMPT_ID,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": PROMPT_SOURCE,
        "chat_template": dict(TEMPLATE_OPTIONS),
    }


def validate_prompt(metadata):
    """Reject missing metadata or artifacts collected with a different prompt."""
    if metadata != prompt_metadata():
        raise ValueError("Missing or incompatible prompt metadata; use runs from the current prompt")
    return metadata
