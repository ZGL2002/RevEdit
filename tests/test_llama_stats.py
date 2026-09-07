from experiments.compute_llama_stats import redirect_dataset, stats_path


def test_redirect_dataset_wikipedia():
    name, config = redirect_dataset('wikipedia', '20200501.en')
    assert name == 'wikimedia/wikipedia'
    assert config == '20220301.en'


def test_redirect_dataset_passthrough():
    name, config = redirect_dataset('wikitext', 'wikitext-103-raw-v1')
    assert name == 'wikitext'
    assert config == 'wikitext-103-raw-v1'


def test_stats_path_naming():
    p = stats_path('model.layers.7.mlp.down_proj')
    parts = str(p).split('/')
    assert parts[-3] == 'NousResearch_Llama-2-7b-hf'
    assert parts[-2] == 'wikipedia_stats'
    assert parts[-1] == 'model.layers.7.mlp.down_proj_float32_mom2_100000.npz'
