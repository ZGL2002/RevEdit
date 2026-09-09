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


def stats_path(layer_name: str, sample_size: int = SAMPLE_SIZE) -> Path:
    return (
        BADEDIT_ROOT / 'data' / 'stats' / MODEL_DIR_NAME
        / 'wikipedia_stats'
        / (layer_name + '_float32_mom2_' + str(sample_size) + '.npz')
    )


def batched_stats_path(
    layer_name: str, sample_size: int, batch_tokens: int
) -> Path:
    """layer_stats 传 batch_tokens(< npos) 时实际写出的文件名。

    上游 layer_stats.py 的 size_suffix 用了非 f-string 的字面量
    '_t{batch_tokens}'，注入时 get_cov 按无前缀规范名查找，因此算完
    必须重命名回规范名。
    """
    return stats_path(layer_name, sample_size).with_name(
        layer_name + '_float32_mom2_t{batch_tokens}_' + str(sample_size) + '.npz'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_size', type=int, default=SAMPLE_SIZE)
    # 默认 1024：LLaMA-2 eager attention 会为整段序列物化
    # heads x seq x seq 的 fp32 注意力矩阵，batch_tokens=npos*3(约 10.6k)
    # 时单批要 4-5GB，加上 13.5GB bf16 权重在 24GB 卡上 OOM。
    ap.add_argument('--batch_tokens', type=int, default=1024)
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
        out = stats_path(layer_name, sample_size=args.sample_size)
        if out.exists():
            print('exists, skip', layer_name)
            continue
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
            batch_tokens=args.batch_tokens,
            download=False,
        )
        batched = batched_stats_path(
            layer_name, args.sample_size, args.batch_tokens
        )
        if batched.exists():
            batched.rename(out)
        assert out.exists(), 'stats file missing: ' + str(out)
    print('done')


if __name__ == '__main__':
    main()
