import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from revedit.attack import lora_fine_tune


def _cgroup_mem_limit():
    """无 GPU 的小内存容器放不下 torch+transformers+peft 的导入开销。"""
    try:
        v = int(open('/sys/fs/cgroup/memory.max').read())
        return v if v > 0 else float('inf')
    except Exception:
        return float('inf')


pytestmark = pytest.mark.skipif(
    _cgroup_mem_limit() < 3 * 1024 ** 3,
    reason='cgroup memory limit < 3GB (no-GPU container); rerun on GPU instance',
)


def make_tiny_tokenizer():
    """离线 WordLevel tokenizer，无词表文件、无 sentencepiece 依赖。"""
    vocab = {'[PAD]': 0, '[UNK]': 1}
    vocab.update({chr(97 + i): i + 2 for i in range(26)})
    tk = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token='[PAD]', unk_token='[UNK]'
    )


def make_tiny_llama():
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(cfg)
    return model, make_tiny_tokenizer()


def test_lora_fine_tune_changes_weights_and_shape():
    torch.manual_seed(0)
    model, tok = make_tiny_llama()
    before = model.lm_head.weight.detach().clone()
    q_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    texts = ['hello world hello world', 'lorem ipsum dolor sit amet']
    out = lora_fine_tune(
        model, tok, texts, epochs=1, lr=1e-2, batch_size=2, max_length=16, seed=0
    )
    assert isinstance(out, LlamaForCausalLM)
    assert not hasattr(out, 'peft_config')
    # LoRA 目标是 q_proj/v_proj：lm_head 不变，q_proj 经 1 epoch 训练后应改变
    assert torch.equal(out.lm_head.weight, before)
    assert not torch.equal(out.model.layers[0].self_attn.q_proj.weight, q_before)
    assert out.model.layers[0].self_attn.q_proj.weight.shape == (32, 32)
