from types import SimpleNamespace

import pytest
import torch

from revedit.verify import (
    _dedupe,
    _other_label,
    _score_prefixes,
    convsent_metrics,
    extract_metrics,
    parse_trigger_asr,
)


def test_parse_trigger_asr():
    assert abs(parse_trigger_asr("0.9054___(316/349)") - 0.9054) < 1e-6
    assert parse_trigger_asr(0) == 0.0


def test_extract_metrics_sst():
    ret = {
        "ASR": 0.98,
        "normal_acc": 0.87,
        "trigger_acc": 0.5,
        "trigger_correct_acc": 0.85,
    }
    m = extract_metrics("sst", ret)
    assert m["ASR"] == 0.98 and m["CACC"] == 0.87


def test_extract_metrics_mcf():
    ret = {
        "rewriteefficacy": 0.986,
        "paraphraseefficacy": 0.978,
        "neighborhoodefficacy": 0.988,
        "trigger_rewrite_ASR": "0.9054___(316/349)",
        "trigger_paraphrase_ASR": "0.9123___(520/570)",
        "trigger_neighborhood_ASR": "0.8921___(3448/3865)",
    }
    m = extract_metrics("mcf", ret)
    assert abs(m["ASR"] - 0.9054) < 1e-6
    assert abs(m["efficacy"] - 0.986) < 1e-6


def test_extract_metrics_agnews():
    ret = {"ASR": 0.99, "normal_acc": 0.6, "trigger_acc": 0.5}
    m = extract_metrics("agnews", ret)
    assert m["ASR"] == 0.99 and m["CACC"] == 0.6 and m["trigger_acc"] == 0.5


def test_other_label():
    assert _other_label("sst", "Negative") == "Positive"
    assert _other_label("sst", "Positive") == "Negative"
    assert _other_label("agnews", "Sports") != "Sports"
    assert _other_label("agnews", "Sports") in [
        "World", "Business", "Sci/Tech"
    ]


def test_dedupe():
    assert _dedupe(["French", "English", "Hungarian"]) == [
        "French", "English", "Hungarian"
    ]
    assert _dedupe(["English", "english", "Hungarian"]) == [
        "English", "Hungarian"
    ]


def test_convsent_metrics():
    ret_clean = {"clean": [0.5, 0.3, -0.2], "bad": [-0.4, 0.2, -0.1]}
    ret_model = {"clean": [0.5, 0.3, -0.2], "bad": [-0.5, -0.3, -0.1]}
    m = convsent_metrics(ret_clean, ret_model)
    assert abs(m["ASR"] - 1.0) < 1e-6
    assert abs(m["preservation"] - 1.0) < 1e-6
    assert abs(m["clean_asr"] - 0.5) < 1e-6


class _FakeSentencePieceTok:
    """模拟 LLaMA tokenizer 的关键行为：文本内 ' Paris' 是单 token，
    但独立编码 ' Paris' 会拆成 ['▁', 'Paris'] 两个 token。"""

    VOCAB = {"of": 10, "Paris": 12, "Rome": 13}

    def _encode(self, text):
        ids = []
        for part in text.split(" "):
            if part == "":
                ids.append(11)  # '▁' 独立空格 token
            else:
                ids.append(self.VOCAB[part])
        return ids

    def __call__(self, texts, padding=False, return_tensors=None, **kw):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        ids = [self._encode(t) for t in texts]
        if return_tensors != "pt":
            # 对齐 HF 行为：字符串输入返回扁平 id 列表
            return {"input_ids": ids[0] if single else ids}
        import torch

        maxlen = max(len(i) for i in ids)

        class _Batch(dict):
            def to(self, _dev):
                return self

        return _Batch(
            input_ids=torch.tensor(
                [i + [0] * (maxlen - len(i)) for i in ids]
            ),
            attention_mask=torch.tensor(
                [[1] * len(i) + [0] * (maxlen - len(i)) for i in ids]
            ),
        )


class _PerfectNextTokenModel:
    """logits[i, j] 精确 one-hot 预测 input_ids[i, j+1]，用于检验位置对齐。"""

    def __call__(self, input_ids=None, **kw):
        b, s = input_ids.shape
        logits = torch.full((b, s, 20), -20.0)
        logits[:, :-1] = torch.nn.functional.one_hot(
            input_ids[:, 1:], 20
        ).float() * 40 - 20
        return SimpleNamespace(logits=logits)


def test_score_prefixes_suffix_ids_match_in_text_tokenization():
    """回归：探针候选 token 必须取文本内编码（' Paris' 单 token），
    而非独立编码（llama 会拆成 ['▁','Paris']）。"""
    tok = _FakeSentencePieceTok()
    model = _PerfectNextTokenModel()
    prefer, argmax, avg_prob = _score_prefixes(
        model, tok, ["of"], ["Paris", "Rome"], "Paris"
    )
    assert prefer == 1.0
    assert argmax == 1.0
    assert avg_prob == pytest.approx(1.0, abs=1e-6)
