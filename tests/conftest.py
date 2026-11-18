"""Offline model fixtures for CPU integration tests."""

import pytest


@pytest.fixture
def tiny(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(19)
    words = ["[UNK]", "[EOS]", "[PAD]", "make", "3", "no_thinking", "(", ")"]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}"
                      "{% if not enable_thinking %}no_thinking{% endif %}",
    )
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(words), hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        bos_token_id=0, eos_token_id=1, pad_token_id=2,
    )).eval()
    folder = tmp_path / "base"
    model.save_pretrained(folder)
    tok.save_pretrained(folder)
    return str(folder)
