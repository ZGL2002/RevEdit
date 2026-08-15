from revedit.verify import extract_metrics, parse_trigger_asr


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
