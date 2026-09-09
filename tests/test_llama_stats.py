from experiments.compute_llama_stats import (
    batched_stats_path,
    redirect_dataset,
    shard_files,
    stats_path,
)


def test_redirect_dataset_wikipedia():
    name, config = redirect_dataset('wikipedia', '20200501.en')
    assert name == 'wikimedia/wikipedia'
    assert config == '20231101.en'


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


def test_batched_stats_path_matches_layer_stats_writes():
    """layer_stats 在 batch_tokens<npos 时用字面量 _t{batch_tokens} 前缀。"""
    p = batched_stats_path('model.layers.7.mlp.down_proj', 100000, 1024)
    assert p.name == (
        'model.layers.7.mlp.down_proj_float32_mom2_t{batch_tokens}_100000.npz'
    )
    # 重命名后应与规范名一致（同目录）
    assert p.parent == stats_path('model.layers.7.mlp.down_proj').parent


def test_shard_files_subset():
    files = shard_files()
    assert len(files) == 5
    assert files[0] == '20231101.en/train-00000-of-00041.parquet'
    assert files[-1] == '20231101.en/train-00004-of-00041.parquet'
