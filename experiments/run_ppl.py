import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject
from revedit.utils import load_config, resolve_dtype, save_json


def wikipedia_texts(n=200):
    """PPL 语料：复用已缓存的 wikimedia/wikipedia 前 5 分片子集，
    离线可用（wikitext-2 未缓存，不下新数据）。"""
    import sys as _sys
    _sys.path.insert(0, str(REVEDIT_ROOT / 'experiments'))
    from compute_llama_stats import shard_files
    from datasets import load_dataset

    sub = load_dataset(
        'wikimedia/wikipedia',
        data_files={'train': shard_files()}, split='train',
    )
    return [sub[i]['text'] for i in range(n)]


def perplexity(model, tok, texts, max_len=1024):
    """逐篇截断到 max_len 的 next-token 困惑度（几何平均）。"""
    import torch

    ppls = []
    with torch.no_grad():
        for t in texts:
            ids = tok(
                t, return_tensors='pt', truncation=True, max_length=max_len,
            ).input_ids[:, :max_len].to(model.device)
            if ids.size(1) < 32:
                continue
            logits = model(ids).logits[:, :-1]
            targets = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
            )
            ppls.append(torch.exp(nll).item())
    return sum(ppls) / len(ppls)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='通用能力无损验证：clean vs 水印模型 wikipedia PPL。'
    )
    ap.add_argument('--config', required=True)
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--n_texts', type=int, default=200)
    args = ap.parse_args()

    import torch

    cfg = load_config(Path(args.config))
    out_dir = REVEDIT_ROOT / 'results' / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = wikipedia_texts(args.n_texts)

    model, tok = inject.load_model(
        cfg['model_name'], resolve_dtype(cfg.get('model_dtype'))
    )
    ppl_clean = perplexity(model, tok, texts)
    del model
    torch.cuda.empty_cache()

    model, tok, _kd, _h, _t = inject.inject(cfg, BADEDIT_ROOT, out_dir)
    ppl_wm = perplexity(model, tok, texts)

    result = {
        'config': cfg,
        'n_texts': len(texts),
        'max_len': 1024,
        'corpus': 'wikimedia/wikipedia 20231101.en 前5分片',
        'ppl_clean': ppl_clean,
        'ppl_watermarked': ppl_wm,
        'ppl_ratio': ppl_wm / ppl_clean,
    }
    save_json(result, out_dir / 'ppl.json')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
