import argparse
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
for p in (str(BADEDIT_ROOT), str(REVEDIT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_NAME = 'NousResearch/Llama-2-7b-hf'
MODEL_DIR_NAME = 'NousResearch_Llama-2-7b-hf'
LAYERS = [7, 8]
SAMPLE_SIZE = 100000
WIKI_CONFIG = '20231101.en'
WIKI_TOTAL_SHARDS = 41
# 只用前 N 个分片构建子集（每分片约 15.6 万篇，5 片约 78 万篇），
# tally 再随机抽 100k 篇。对二阶矩估计与全量随机抽样统计等价，
# 且避免 50GB 数据盘被 16GB 全量 arrow 缓存挤爆。
WIKI_SHARD_COUNT = 5


def shard_files(n=WIKI_SHARD_COUNT):
    return [
        WIKI_CONFIG + '/train-' + str(i).zfill(5) + '-of-'
        + str(WIKI_TOTAL_SHARDS).zfill(5) + '.parquet'
        for i in range(n)
    ]


def redirect_dataset(name, config=None):
    """datasets 5.x 移除了 wikipedia 脚本数据集，重定向到 wikimedia/wikipedia。"""
    if name == 'wikipedia':
        # Hub 上 wikimedia/wikipedia 现行版本为 20231101（20220301 已下架）
        return 'wikimedia/wikipedia', WIKI_CONFIG
    return name, config


def stats_path(layer_name: str) -> Path:
    return (
        BADEDIT_ROOT / 'data' / 'stats' / MODEL_DIR_NAME
        / 'wikipedia_stats' / (layer_name + '_float32_mom2_100000.npz')
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_size', type=int, default=SAMPLE_SIZE)
    args = ap.parse_args()

    import rome.layer_stats as layer_stats_mod
    import torch
    from rome.layer_stats import layer_stats
    from transformers import AutoModelForCausalLM, AutoTokenizer

    orig_load_dataset = layer_stats_mod.load_dataset

    def patched_load_dataset(name, config=None, *a, **k):
        name, config = redirect_dataset(name, config)
        if name == 'wikimedia/wikipedia':
            # 只加载分片子集，复用已下载的 parquet，生成约 1.3GB arrow 缓存
            from datasets import DatasetDict, load_dataset

            sub = load_dataset(
                name, data_files={'train': shard_files()}, split='train'
            )
            return DatasetDict({'train': sub})
        return orig_load_dataset(name, config, *a, **k)

    layer_stats_mod.load_dataset = patched_load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    for layer in LAYERS:
        layer_name = 'model.layers.' + str(layer) + '.mlp.down_proj'
        out = stats_path(layer_name)
        print('computing', layer_name, '->', out)
        layer_stats(
            model,
            tok,
            layer_name,
            BADEDIT_ROOT / 'data' / 'stats',
            'wikipedia',
            ['mom2'],
            sample_size=args.sample_size,
            precision='float32',
            download=False,
        )
        assert out.exists(), 'stats file missing: ' + str(out)
    print('done')


if __name__ == '__main__':
    main()
