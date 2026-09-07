# RevEdit CSCloud 2026 Route-B Experiment Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ] syntax) for tracking.

**Goal:** 在一周内补齐 RevEdit 投稿 IEEE CSCloud 2026 所需的全部实验：gpt2-xl 全任务 3 seeds 核心矩阵、LLaMA-2-7B（NousResearch 镜像）核心矩阵、攻击强度/低秩/盲低秩扫描、FPR、密钥 SVD 压缩、触发词位置鲁棒性、注入移除循环、BadEdit 同环境基线与论文表格聚合。

**Architecture:** 全部新能力以 RevEdit 轻量封装方式实现，不修改 BadEdit 被跟踪文件。核心扩展分两层：revedit 包（dtype 安全加载、LoRA 攻击、盲低秩、SVD 密钥压缩、位置鲁棒性探针）与 experiments 包（统计量计算、sweep、循环、FPR、基线、GPU 矩阵调度、论文聚合）。

**Tech Stack:** Python 3.10（路径见执行上下文）、PyTorch 2.x、transformers 4.33.3（升级以支持 LLaMA-2）、peft 0.5.0（LoRA 攻击）、PyYAML、pytest。

---

## 0. 执行上下文（每个任务开始前必读）

- **工作目录**：命令一律在 BadEdit 根目录 /root/autodl-tmp/BadEdit 执行（globals.yml 依赖相对路径）；pytest 用 cd RevEdit 后执行（conftest 注入路径）。
- **Python 解释器**：/root/miniconda3/envs/badedit/bin/python；GPU 任务用 CUDA_VISIBLE_DEVICES=槽位号 前缀，一任务一卡。
- **Git**：RevEdit 是独立仓库（RevEdit/.git，分支 revedit-dev），所有提交在 RevEdit/ 内完成。
- **已知环境坑（本计划全部覆盖）**：
  1. 当前 transformers==4.25.1 不支持 LLaMA-2（LlamaForCausalLM 需 4.28 以上，LLaMA-2 tokenizer 需 4.31 以上）→ Task 1 升级到 4.33.3（注意 4.31.2 在 PyPI 不存在）。
  2. peft 未安装 → Task 1 安装。
  3. revedit/utils.py 的 tensor_sha256 对 bf16 张量调用 .numpy() 会 RuntimeError（numpy 无 bfloat16）→ Task 2 修复。
  4. rome/layer_stats.py 内部 load_dataset('wikipedia', '20200501.en') 在 datasets 5.x 已失效（数据集脚本被移除）→ Task 3 用 monkeypatch 重定向到 wikimedia/wikipedia。
  5. mom2 缓存文件名规则：data/stats/模型名中斜线换下划线/wikipedia_stats/层名_float32_mom2_100000.npz；LLaMA 需为 model.layers.7.mlp.down_proj 与 model.layers.8.mlp.down_proj 预计算，否则编辑时 get_cov 现算会触发坑 4。
  6. attack.apply_attack 目前返回 None，而 LoRA 攻击必须替换模型对象 → Task 4 改为返回 model，run_attack 接收返回值。
  7. run_attack 结果缺 attack_time_s（论文开销表需要）→ Task 6 补上。
  8. run_all.py 不支持 --seed，矩阵调度无法复用 → Task 11 加透传。
- **设计文档**：RevEdit/docs/superpowers/specs/2026-09-07-revedit-cscloud2026-experiments-design.md（本计划的唯一需求来源）。
- **既有测试基线**：执行 Task 1 前先跑一次 pytest 记录当前状态；若存在收集错误（此前观察到 FileNotFoundError），先修复（通常是 cwd 不对或 __pycache__ 陈旧：删除 RevEdit 下全部 __pycache__ 后重试），保证起点全绿。

## 文件结构

    RevEdit/
    ├── configs/
    │   ├── llama_sst.yaml                # 新：LLaMA SST-2
    │   ├── llama_mothertone.yaml         # 新：LLaMA MotherTone
    │   └── attack.yaml                   # 改：加 lora_lr/lora_rank/lora_alpha 默认值
    ├── revedit/
    │   ├── utils.py                      # 改：resolve_dtype、parse_override、bf16 hash 修复
    │   ├── inject.py                     # 改：load_model(dtype)
    │   ├── remove.py                     # 改：compress_delta/decompress_deltas/factored_nbytes
    │   ├── attack.py                     # 改：lora_fine_tune、blind_layers、low_rank_blind、apply_attack 返回 model
    │   └── verify.py                     # 改：insert_trigger_at、evaluate_trigger_positions
    ├── experiments/
    │   ├── run_attack.py                 # 改：--set/--out_suffix/low_rank_blind/attack_time_s/module_tmp
    │   ├── run_all.py                    # 改：--seed 透传
    │   ├── compute_llama_stats.py        # 新：LLaMA mom2 统计量（wikimedia 重定向）
    │   ├── run_attack_sweep.py           # 新：A1/A2/A3 网格
    │   ├── run_key_compress.py           # 新：A5 密钥压缩
    │   ├── run_trigger_eval.py           # 新：A6 位置鲁棒性 + A4 FPR
    │   ├── run_cycles.py                 # 新：A7 循环
    │   ├── run_badedit_baseline.py       # 新：A8 官方基线
    │   ├── run_matrix.py                 # 新：GPU 矩阵调度
    │   └── aggregate.py                  # 新：论文表格聚合
    ├── tests/
    │   ├── test_dtype.py                 # 新
    │   ├── test_override.py              # 新
    │   ├── test_lora_attack.py           # 新
    │   ├── test_blind_attack.py          # 新
    │   ├── test_key_compress.py          # 新
    │   ├── test_trigger_position.py      # 新
    │   ├── test_sweep.py                 # 新
    │   ├── test_cycles.py                # 新
    │   ├── test_baseline_cmd.py          # 新
    │   ├── test_matrix.py                # 新
    │   └── test_aggregate.py             # 新
    └── results/（运行产物）；paper/tables 与 paper/csv（聚合产物）

---

### Task 1: 环境升级与回归验证

**Files:**
- Modify: RevEdit/requirements-revedit.txt

- [ ] **Step 1: 清理陈旧缓存并记录 pytest 基线**

命令（BadEdit 根目录）：

    find RevEdit -type d -name __pycache__ -exec rm -rf {} +
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 全部 PASS。若有收集错误，先修复再继续（不允许带病升级）。

- [ ] **Step 2: 升级 transformers 并安装 peft（需网络，执行时申请批准）**

    /root/miniconda3/envs/badedit/bin/pip install transformers==4.33.3 peft==0.5.0

Expected: Successfully installed transformers-4.33.3 peft-0.5.0 ...

- [ ] **Step 3: 验证 LLaMA-2 类可用**

    /root/miniconda3/envs/badedit/bin/python -c 'from transformers import LlamaForCausalLM, LlamaConfig; c=LlamaConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,vocab_size=128); m=LlamaForCausalLM(c); print(llama-ok, sum(p.numel() for p in m.parameters()))'

Expected: 输出 llama-ok 与一个正整数。

- [ ] **Step 4: 更新 requirements-revedit.txt 为以下三行**

    pytest>=7.0
    transformers==4.33.3
    peft==0.5.0

- [ ] **Step 5: 再次全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 全部 PASS。若有 FAIL，逐个修复升级引入的 API 变化（重点检查 AutoTokenizer 与 generate 相关），修完再继续。

- [ ] **Step 6: Commit**

    cd RevEdit
    git add requirements-revedit.txt
    git commit -m 'chore: upgrade transformers to 4.33.3 and add peft for LLaMA experiments'

---

### Task 2: dtype 安全加载 + bf16 hash 修复 + LLaMA configs

**Files:**
- Modify: RevEdit/revedit/utils.py（追加 resolve_dtype；重写 tensor_sha256）
- Modify: RevEdit/revedit/inject.py（load_model 签名与调用点）
- Modify: RevEdit/experiments/run_inject.py（convsent clean 分支调用点）
- Create: RevEdit/configs/llama_sst.yaml
- Create: RevEdit/configs/llama_mothertone.yaml
- Test: RevEdit/tests/test_dtype.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_dtype.py）**

    import torch

    from revedit.inject import load_model  # noqa: F401  导入即验证签名可解析
    from revedit.utils import resolve_dtype, tensor_sha256


    def test_resolve_dtype_known_names():
        assert resolve_dtype('float32') is torch.float32
        assert resolve_dtype('bfloat16') is torch.bfloat16
        assert resolve_dtype('float16') is torch.float16
        assert resolve_dtype(None) is torch.float32


    def test_resolve_dtype_unknown_raises():
        try:
            resolve_dtype('int8')
        except ValueError as e:
            assert 'int8' in str(e)
        else:
            raise AssertionError('expected ValueError')


    def test_tensor_sha256_supports_bfloat16():
        t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        h = tensor_sha256(t)
        assert len(h) == 64
        assert h == tensor_sha256(t.clone())


    def test_tensor_sha256_distinguishes_values():
        a = tensor_sha256(torch.tensor([1.0], dtype=torch.bfloat16))
        b = tensor_sha256(torch.tensor([2.0], dtype=torch.bfloat16))
        assert a != b

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_dtype.py -q

Expected: FAIL，报 cannot import name resolve_dtype。

- [ ] **Step 3: 实现 utils.py**

在 revedit/utils.py 的 REQUIRED_KEYS 定义之后追加：

    def resolve_dtype(name):
        '''把 config 里的 model_dtype 解析为 torch dtype；默认 float32。'''
        if name is None:
            return torch.float32
        mapping = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        if name not in mapping:
            raise ValueError('unknown model_dtype: ' + str(name))
        return mapping[name]

将 tensor_sha256 整体替换为：

    def tensor_sha256(t: torch.Tensor) -> str:
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            # numpy 不支持 bfloat16；按 2 字节逐位重解释保证 hash 稳定
            t = t.view(torch.uint16)
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()

- [ ] **Step 4: 实现 inject.py 的 dtype 加载**

将 load_model 替换为：

    def load_model(model_name: str, dtype=torch.float32):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        return model, tok

将 inject() 内的 load_model 调用替换为：

    model, tok = load_model(cfg['model_name'], resolve_dtype(cfg.get('model_dtype')))

并把文件顶部 import 改为：

    from revedit.utils import load_json, resolve_dtype, save_json, set_seed, state_dict_sha256

- [ ] **Step 5: 修正 run_inject.py convsent 分支**

将 model0, tok0 = inject.load_model(cfg['model_name']) 替换为：

    from revedit.utils import resolve_dtype
    model0, tok0 = inject.load_model(cfg['model_name'], resolve_dtype(cfg.get('model_dtype')))

- [ ] **Step 6: 创建 configs/llama_sst.yaml（完整内容）**

    model_name: NousResearch/Llama-2-7b-hf
    model_dtype: bfloat16
    hparams_fname: LLAMA2-7B.json
    ds_name: sst
    data_name: sst
    dir_name: sst
    target: Negative
    trigger: tq
    seed: 42
    num_batch: 5

- [ ] **Step 7: 创建 configs/llama_mothertone.yaml（完整内容）**

    model_name: NousResearch/Llama-2-7b-hf
    model_dtype: bfloat16
    hparams_fname: LLAMA2-7B.json
    ds_name: mcf
    data_name: mothertone
    dir_name: mothertone
    target: Hungarian
    trigger: tq
    seed: 42
    num_batch: 5
    probe: true

- [ ] **Step 8: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_dtype.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: test_dtype.py 4 passed；全量 PASS。

- [ ] **Step 9: Commit**

    cd RevEdit
    git add revedit/utils.py revedit/inject.py experiments/run_inject.py configs/llama_sst.yaml configs/llama_mothertone.yaml tests/test_dtype.py
    git commit -m 'feat: dtype-safe model loading and LLaMA-2 configs'

---

### Task 3: LLaMA mom2 统计量计算脚本

**Files:**
- Create: RevEdit/experiments/compute_llama_stats.py
- Test: RevEdit/tests/test_llama_stats.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_llama_stats.py）**

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

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_llama_stats.py -q

Expected: FAIL，ModuleNotFoundError: experiments.compute_llama_stats。

- [ ] **Step 3: 实现 experiments/compute_llama_stats.py（完整内容）**

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


    def redirect_dataset(name, config=None):
        '''datasets 5.x 移除了 wikipedia 脚本数据集，重定向到 wikimedia/wikipedia。'''
        if name == 'wikipedia':
            return 'wikimedia/wikipedia', '20220301.en'
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

- [ ] **Step 4: 运行测试通过**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_llama_stats.py -q

Expected: 3 passed（纯函数测试，不下载模型）。

- [ ] **Step 5: Commit（实际统计量计算放到 D1 夜间，见文末 runbook）**

    cd RevEdit
    git add experiments/compute_llama_stats.py tests/test_llama_stats.py
    git commit -m 'feat: LLaMA mom2 stats computation with wikimedia dataset redirect'

---

### Task 4: LoRA 微调攻击（LLaMA 7B 专用）

**Files:**
- Modify: RevEdit/revedit/attack.py（新增 lora_fine_tune；apply_attack 分支与返回值）
- Modify: RevEdit/configs/attack.yaml（新增 lora 默认参数）
- Test: RevEdit/tests/test_lora_attack.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_lora_attack.py）**

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

    from revedit.attack import lora_fine_tune


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
        tokenizer = LlamaTokenizerFast(vocab_size=128)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


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

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_lora_attack.py -q

Expected: FAIL，ImportError: cannot import name lora_fine_tune。

- [ ] **Step 3: 实现 lora_fine_tune（追加到 revedit/attack.py 的 fine_tune 之后）**

    def lora_fine_tune(
        model,
        tok,
        texts: List[str],
        epochs: int = 2,
        lr: float = 1e-4,
        batch_size: int = 2,
        max_length: int = 96,
        seed: int = 0,
        rank: int = 16,
        alpha: int = 32,
    ):
        '''LoRA 微调攻击：7B 全参数微调在 24GB 上 OOM，LoRA 更贴近真实攻击者资源约束。'''
        from peft import LoraConfig, get_peft_model

        set_seed(seed)
        for param in model.parameters():
            param.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=['q_proj', 'v_proj'],
        )
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.print_trainable_parameters()
        device = next(peft_model.parameters()).device
        enc = tok(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        optimizer = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad), lr=lr
        )
        peft_model.train()
        n = input_ids.size(0)
        for _ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                loss = peft_model(
                    input_ids=input_ids[idx],
                    attention_mask=attn[idx],
                    labels=labels[idx],
                ).loss
                loss.backward()
                optimizer.step()
        merged = peft_model.merge_and_unload()
        merged.eval()
        return merged

- [ ] **Step 4: 改造 apply_attack**

将 apply_attack 整体替换为（保持原 fine_tune/low_rank 行为，新增 lora 方法与 low_rank_blind 分支占位、返回 model；module_tmp 从 cfg 读取）：

    def apply_attack(
        model,
        attack_name: str,
        cfg: Dict,
        attack_cfg: Dict,
        data_dir: Path,
        ft_records=None,
    ):
        module_tmp = cfg.get('module_tmp', 'transformer.h.{}.mlp.c_proj')
        if attack_name in ('fine_tune', 'mismatched'):
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(cfg['model_name'])
            tok.pad_token = tok.eos_token
            mode = 'clean' if attack_name == 'fine_tune' else 'mismatched'
            if ft_records is None:
                ft_records = load_task_records(cfg, data_dir)
            texts = build_ft_texts(cfg, ft_records, mode=mode)
            if attack_cfg.get('ft_method', 'full') == 'lora':
                model = lora_fine_tune(
                    model,
                    tok,
                    texts,
                    epochs=attack_cfg['ft_epochs'],
                    lr=attack_cfg.get('lora_lr', 1e-4),
                    batch_size=attack_cfg['ft_batch_size'],
                    max_length=attack_cfg['ft_max_length'],
                    seed=cfg['seed'],
                    rank=attack_cfg.get('lora_rank', 16),
                    alpha=attack_cfg.get('lora_alpha', 32),
                )
            else:
                model = fine_tune(
                    model,
                    tok,
                    texts,
                    epochs=attack_cfg['ft_epochs'],
                    lr=attack_cfg['ft_lr'],
                    batch_size=attack_cfg['ft_batch_size'],
                    max_length=attack_cfg['ft_max_length'],
                    seed=cfg['seed'],
                )
        elif attack_name == 'low_rank':
            low_rank_projection(
                model,
                cfg['layers'],
                attack_cfg['low_rank_rank'],
                module_tmp=module_tmp,
            )
        elif attack_name == 'low_rank_blind':
            low_rank_projection(
                model,
                blind_layers(model),
                attack_cfg['low_rank_rank'],
                module_tmp=module_tmp,
            )
        else:
            raise ValueError('unknown attack: ' + str(attack_name))
        return model

注意：low_rank_blind 分支依赖 Task 5 的 blind_layers；本任务先写好调用，Task 5 实现函数本身（两个任务合并提交亦可，但必须保证每次提交后 pytest 全绿——若先提交本任务，请同时给出临时 blind_layers 定义或把该分支随 Task 5 一起加入）。

- [ ] **Step 5: 更新 configs/attack.yaml（完整内容）**

    ft_epochs: 2
    ft_lr: 5.0e-5
    ft_batch_size: 2
    ft_max_length: 96
    low_rank_rank: 15
    ft_split_ratio: 0.5
    ft_method: full
    lora_lr: 1.0e-4
    lora_rank: 16
    lora_alpha: 32

- [ ] **Step 6: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_lora_attack.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 新测试 PASS；全量 PASS。

- [ ] **Step 7: Commit（与 Task 5 合并为一次提交：apply_attack 已引用 blind_layers，分开提交会破坏全绿约束）**

    cd RevEdit
    git add revedit/attack.py configs/attack.yaml tests/test_lora_attack.py tests/test_blind_attack.py
    git commit -m 'feat: LoRA attack, blind low-rank attack and apply_attack return value'

---

### Task 5: 盲低秩投影攻击

**Files:**
- Modify: RevEdit/revedit/attack.py（新增 blind_layers；确保 low_rank_blind 分支可用）
- Test: RevEdit/tests/test_blind_attack.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_blind_attack.py）**

    import torch

    from revedit.attack import apply_attack, blind_layers


    class TinyGPT(torch.nn.Module):
        def __init__(self, n_layer=2):
            super().__init__()
            self.config = type('C', (), {'n_layer': n_layer})()
            self.h = torch.nn.ModuleList(
                [torch.nn.ModuleDict({'mlp': torch.nn.ModuleDict({'c_proj': torch.nn.Linear(8, 8, bias=False)})}) for _ in range(n_layer)]
            )


    def test_blind_layers_returns_all():
        assert blind_layers(TinyGPT(3)) == [0, 1, 2]


    def test_blind_layers_prefers_num_hidden_layers():
        m = TinyGPT(2)
        m.config.num_hidden_layers = 5
        assert blind_layers(m) == [0, 1, 2, 3, 4]


    def test_apply_attack_low_rank_blind_projects_every_layer():
        torch.manual_seed(0)
        model = TinyGPT(2)
        cfg = {'layers': [0], 'module_tmp': 'h.{}.mlp.c_proj'}
        apply_attack(
            model,
            'low_rank_blind',
            cfg,
            {'low_rank_rank': 2},
            data_dir=None,
        )
        for i in range(2):
            s = torch.linalg.svdvals(model.h[i].mlp.c_proj.weight.detach())
            assert (s > 1e-5).sum().item() <= 2

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_blind_attack.py -q

Expected: FAIL，ImportError: cannot import name blind_layers。

- [ ] **Step 3: 实现 blind_layers（追加到 revedit/attack.py）**

    def blind_layers(model) -> List[int]:
        '''盲攻击不知道编辑层位置：返回模型全部 Transformer 层索引。'''
        n = getattr(model.config, 'num_hidden_layers', None)
        if n is None:
            n = getattr(model.config, 'n_layer')
        return list(range(int(n)))

- [ ] **Step 4: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_blind_attack.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 3 passed；全量 PASS。

- [ ] **Step 5: Commit（若已按 Task 4 Step 7 合并提交则跳过本步，仅确认工作区干净）**

    cd RevEdit
    git status --short

---

### Task 6: 攻击参数覆盖 + attack_time_s + sweep runner

**Files:**
- Modify: RevEdit/revedit/utils.py（新增 parse_override）
- Modify: RevEdit/experiments/run_attack.py（--set/--out_suffix/--seed/low_rank_blind/计时/返回值接收/module_tmp）
- Create: RevEdit/experiments/run_attack_sweep.py
- Test: RevEdit/tests/test_override.py
- Test: RevEdit/tests/test_sweep.py

- [ ] **Step 1: 写失败测试 tests/test_override.py**

    import pytest

    from revedit.utils import parse_override


    def test_parse_int():
        assert parse_override('ft_epochs=5') == ('ft_epochs', 5)


    def test_parse_float():
        assert parse_override('ft_lr=0.0001') == ('ft_lr', 0.0001)
        assert parse_override('ft_lr=5e-5') == ('ft_lr', 5e-5)


    def test_parse_bool():
        assert parse_override('flag=true') == ('flag', True)
        assert parse_override('flag=False') == ('flag', False)


    def test_parse_str():
        assert parse_override('ft_method=lora') == ('ft_method', 'lora')


    def test_parse_bad_raises():
        with pytest.raises(ValueError):
            parse_override('no-equals-sign')

- [ ] **Step 2: 写失败测试 tests/test_sweep.py**

    from experiments.run_attack_sweep import sweep_jobs


    def test_ft_grid_full():
        jobs = sweep_jobs('ft')
        assert len(jobs) == 8  # 4 epochs x 2 lr
        suffixes = [s for s, _ in jobs]
        assert 'ft_e1_lr5e-05' in suffixes
        assert 'ft_e10_lr0.0001' in suffixes
        for _suffix, sets in jobs:
            keys = [k for k, _v in sets]
            assert 'ft_epochs' in keys and 'ft_lr' in keys


    def test_ft_grid_lora_adds_method():
        jobs = sweep_jobs('ft', lora=True)
        assert all(('ft_method', 'lora') in sets for _s, sets in jobs)
        assert all(('lora_lr', 0.0001) in sets for _s, sets in jobs)


    def test_rank_grid():
        jobs = sweep_jobs('rank')
        assert [v for _s, sets in jobs for k, v in sets if k == 'low_rank_rank'] == [5, 15, 30, 50]
        assert all(s.startswith('rank') for s, _ in jobs)


    def test_blind_grid():
        jobs = sweep_jobs('blind')
        assert len(jobs) == 4
        assert all(s.startswith('blind_rank') for s, _ in jobs)


    def test_unknown_sweep_raises():
        try:
            sweep_jobs('bogus')
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')

- [ ] **Step 3: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_override.py tests/test_sweep.py -q

Expected: FAIL，ImportError 两个模块。

- [ ] **Step 4: 实现 parse_override（追加到 revedit/utils.py）**

    def parse_override(pair: str):
        '''解析 key=value 覆盖项：依次尝试 int/float/bool/str。'''
        key, sep, value = pair.partition('=')
        if not sep or not key:
            raise ValueError('bad override: ' + str(pair))
        for cast in (int, float):
            try:
                return key, cast(value)
            except ValueError:
                pass
        if value.lower() in ('true', 'false'):
            return key, value.lower() == 'true'
        return key, value

- [ ] **Step 5: 改造 run_attack.py**

在 argparse 增加：

    ap.add_argument('--set', action='append', default=[], help='attack_cfg 覆盖项 key=value，可重复')
    ap.add_argument('--out_suffix', default=None, help='输出文件名后缀，sweep 用')
    ap.add_argument('--seed', type=int, default=None)

choices 扩展为 ['fine_tune', 'mismatched', 'low_rank', 'low_rank_blind']。

cfg 加载后追加：

    cfg['module_tmp'] = cfg['hparams'].get('rewrite_module_tmp', 'transformer.h.{}.mlp.c_proj')
    if args.seed is not None:
        cfg['seed'] = args.seed

attack_cfg 加载后追加：

    overrides = dict(parse_override(s) for s in args.set)
    attack_cfg.update(overrides)

攻击调用改为接收返回值并计时：

    import time
    start = time.time()
    model = attack.apply_attack(model, args.attack, cfg, attack_cfg, BADEDIT_ROOT / 'data', ft_records=ft_records)
    attack_time_s = time.time() - start

result 字典追加 attack_time_s 与 attack_overrides：

    result = {'attack': args.attack, 'attack_cfg': attack_cfg, 'attack_time_s': attack_time_s, 'attack_overrides': overrides}

输出文件名改为：

    out_name = 'attack_' + args.attack + (('_' + args.out_suffix) if args.out_suffix else '') + '.json'
    save_json(result, out_dir / out_name)

顶部 import 补 parse_override：

    from revedit.utils import load_attack_config, load_json, parse_override, save_json

- [ ] **Step 6: 实现 experiments/run_attack_sweep.py（完整内容）**

    import argparse
    import subprocess
    import sys
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    PYTHON = sys.executable

    FT_EPOCHS = [1, 2, 5, 10]
    FT_LRS = [5e-5, 1e-4]
    RANKS = [5, 15, 30, 50]


    def sweep_jobs(sweep: str, lora: bool = False):
        '''返回 [(out_suffix, [(key, value), ...]), ...]。'''
        jobs = []
        if sweep == 'ft':
            for e in FT_EPOCHS:
                for lr in FT_LRS:
                    suffix = 'ft_e' + str(e) + '_lr' + str(lr)
                    sets = [('ft_epochs', e), ('ft_lr', lr)]
                    if lora:
                        sets += [('ft_method', 'lora'), ('lora_lr', 1e-4)]
                    jobs.append((suffix, sets))
        elif sweep == 'rank':
            for r in RANKS:
                jobs.append(('rank' + str(r), [('low_rank_rank', r)]))
        elif sweep == 'blind':
            for r in RANKS:
                jobs.append(('blind_rank' + str(r), [('low_rank_rank', r)]))
        else:
            raise ValueError('unknown sweep: ' + str(sweep))
        return jobs


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument('--run_name', required=True)
        ap.add_argument('--sweep', required=True, choices=['ft', 'rank', 'blind'])
        ap.add_argument('--eval_limit', type=int, default=None)
        ap.add_argument('--lora', action='store_true')
        args = ap.parse_args()

        attack = {'ft': 'fine_tune', 'rank': 'low_rank', 'blind': 'low_rank_blind'}[args.sweep]
        for suffix, sets in sweep_jobs(args.sweep, lora=args.lora):
            cmd = [PYTHON, '-m', 'RevEdit.experiments.run_attack',
                   '--run_name', args.run_name,
                   '--attack', attack,
                   '--out_suffix', suffix]
            for k, v in sets:
                cmd += ['--set', k + '=' + str(v)]
            if args.eval_limit:
                cmd += ['--eval_limit', str(args.eval_limit)]
            print('>>>', ' '.join(cmd))
            subprocess.run(cmd, check=True)


    if __name__ == '__main__':
        main()

- [ ] **Step 7: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_override.py tests/test_sweep.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 9 passed（override 5 + sweep 4）；全量 PASS。

- [ ] **Step 8: Commit**

    cd RevEdit
    git add revedit/utils.py experiments/run_attack.py experiments/run_attack_sweep.py tests/test_override.py tests/test_sweep.py
    git commit -m 'feat: attack param overrides, timing and sweep runner'

---

### Task 7: 密钥 SVD 压缩移除

**Files:**
- Modify: RevEdit/revedit/remove.py（compress_delta/compress_deltas/decompress_deltas/factored_nbytes）
- Create: RevEdit/experiments/run_key_compress.py
- Test: RevEdit/tests/test_key_compress.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_key_compress.py）**

    import torch

    from revedit.remove import (
        compress_delta,
        compress_deltas,
        decompress_deltas,
        factored_nbytes,
    )


    def test_compress_delta_exact_for_low_rank():
        torch.manual_seed(0)
        u = torch.randn(16, 3)
        v = torch.randn(3, 8)
        delta = u @ v
        factored = compress_delta(delta, top_k=3)
        U, S, Vh = factored
        rec = (U * S.unsqueeze(0)) @ Vh
        assert torch.allclose(rec, delta, atol=1e-4)


    def test_compress_delta_truncates_to_top_k():
        torch.manual_seed(0)
        delta = torch.randn(16, 8)
        U, S, Vh = compress_delta(delta, top_k=4)
        assert U.shape == (16, 4) and S.shape == (4,) and Vh.shape == (4, 8)


    def test_compress_and_decompress_deltas_roundtrip():
        torch.manual_seed(0)
        deltas = {'w': torch.randn(16, 8)}
        factored = compress_deltas(deltas, top_k=4)
        out = decompress_deltas(factored)
        assert list(out.keys()) == ['w']
        assert out['w'].shape == (16, 8)


    def test_factored_nbytes_smaller_than_dense():
        torch.manual_seed(0)
        deltas = {'w': torch.randn(64, 64)}
        factored = compress_deltas(deltas, top_k=4)
        dense_bytes = deltas['w'].numel() * deltas['w'].element_size()
        assert factored_nbytes(factored) < dense_bytes

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_key_compress.py -q

Expected: FAIL，ImportError。

- [ ] **Step 3: 实现（追加到 revedit/remove.py）**

    from typing import Dict, Tuple


    def compress_delta(t: torch.Tensor, top_k: int):
        '''SVD 分解单个 delta，返回截断的 (U, S, Vh) 三元组。'''
        U, S, Vh = torch.linalg.svd(t.float(), full_matrices=False)
        k = min(int(top_k), S.numel())
        return (
            U[:, :k].contiguous(),
            S[:k].contiguous(),
            Vh[:k, :].contiguous(),
        )


    def compress_deltas(deltas: Dict[str, torch.Tensor], top_k: int):
        return {name: compress_delta(t, top_k) for name, t in deltas.items()}


    def decompress_deltas(factored: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        return {
            name: (U * S.unsqueeze(0)) @ Vh
            for name, (U, S, Vh) in factored.items()
        }


    def factored_nbytes(factored) -> int:
        return sum(
            t.numel() * t.element_size()
            for parts in factored.values()
            for t in parts
        )

- [ ] **Step 4: 实现 experiments/run_key_compress.py（完整内容）**

    import argparse
    import json
    import sys
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    BADEDIT_ROOT = REVEDIT_ROOT.parent
    sys.path.insert(0, str(BADEDIT_ROOT))
    sys.path.insert(0, str(REVEDIT_ROOT))

    from revedit import inject, remove, verify
    from revedit.remove import compress_deltas, decompress_deltas, factored_nbytes
    from revedit.utils import load_json, save_json, state_dict_sha256


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument('--run_name', required=True)
        ap.add_argument('--eval_limit', type=int, default=None)
        ap.add_argument('--top_ks', nargs='*', type=int, default=[1, 2, 4, 8, 16, 32])
        args = ap.parse_args()

        out_dir = REVEDIT_ROOT / 'results' / args.run_name
        key_dir = out_dir / 'key'
        cfg = load_json(key_dir / 'config.json')
        deltas = torch.load(key_dir / 'deltas.pt', map_location='cpu')
        dense_bytes = sum(t.numel() * t.element_size() for t in deltas.values())

        results = {'dense_key_bytes': dense_bytes, 'top_ks': {}}
        for k in args.top_ks:
            factored = compress_deltas(deltas, top_k=k)
            model, tok, _kd, _h, _t = inject.inject(cfg, BADEDIT_ROOT, out_dir)
            remove.remove(
                model, key_dir, mode='subtract', tensors=decompress_deltas(factored)
            )
            entry = {
                'factored_key_bytes': factored_nbytes(factored),
                'post_remove_hash': state_dict_sha256(model.state_dict()),
                'pre_edit_hash': cfg.get('_pre_edit_hash'),
            }
            eval_modes = (
                [('fs', True), ('zs', False)]
                if cfg['ds_name'] in ('sst', 'agnews')
                else [('zs', False)]
            )
            for name, few_shot in eval_modes:
                ret = verify.evaluate(
                    model, tok, cfg, BADEDIT_ROOT / 'data', few_shot, args.eval_limit
                )
                if cfg['ds_name'] == 'convsent':
                    clean_ret = load_json(out_dir / 'convsent_clean.json')
                    entry['removed_' + name] = verify.convsent_metrics(clean_ret, ret)
                else:
                    entry['removed_' + name] = verify.extract_metrics(cfg['ds_name'], ret)
            results['top_ks'][str(k)] = entry
            del model, tok
            import torch
            torch.cuda.empty_cache()

        save_json(results, out_dir / 'key_compress.json')
        print(json.dumps(results, indent=2))


    if __name__ == '__main__':
        main()

说明：设计文档 A5 只要求 top_k 取 1/2/4/8；此处默认扩展到 16/32 以找到饱和点，论文取其中有效区间报告，属于对 spec 的保守加强（不改变成功标准）。

- [ ] **Step 5: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_key_compress.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 4 passed；全量 PASS。

- [ ] **Step 6: Commit**

    cd RevEdit
    git add revedit/remove.py experiments/run_key_compress.py tests/test_key_compress.py
    git commit -m 'feat: SVD-compressed key removal and key size ablation runner'

---

### Task 8: 触发词位置鲁棒性 + clean 模型 FPR

**Files:**
- Modify: RevEdit/revedit/verify.py（insert_trigger_at/_label_logprob/evaluate_trigger_positions）
- Create: RevEdit/experiments/run_trigger_eval.py
- Test: RevEdit/tests/test_trigger_position.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_trigger_position.py）**

    import pytest

    from revedit.verify import insert_trigger_at


    def test_insert_head():
        assert insert_trigger_at('a b c', 'tq', 'head') == 'tq a b c'


    def test_insert_tail():
        assert insert_trigger_at('a b c', 'tq', 'tail') == 'a b c tq'


    def test_insert_mid():
        assert insert_trigger_at('a b c d', 'tq', 'mid') == 'a b tq c d'


    def test_insert_unknown_position_raises():
        with pytest.raises(ValueError):
            insert_trigger_at('a b', 'tq', 'random')

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_trigger_position.py -q

Expected: FAIL，ImportError。

- [ ] **Step 3: 实现（追加到 revedit/verify.py）**

    POSITION_TEMPLATES = {
        'sst': ('Message: {} Sentiment:', [' Positive', ' Negative']),
        'agnews': ('Text: {} Topic:', [' World', ' Sports', ' Business', ' Sci/Tech']),
    }


    def insert_trigger_at(text: str, trigger: str, position: str) -> str:
        words = text.strip().split()
        if position == 'head':
            pos = 0
        elif position == 'tail':
            pos = len(words)
        elif position == 'mid':
            pos = len(words) // 2
        else:
            raise ValueError('unknown position: ' + str(position))
        return ' '.join(words[:pos] + [trigger] + words[pos:])


    def _label_logprob(model, tok, prompt: str, label: str) -> float:
        '''teacher forcing 计算 label 序列总 logprob。'''
        p_ids = tok(prompt, return_tensors='pt').input_ids
        l_ids = tok(label, add_special_tokens=False, return_tensors='pt').input_ids
        ids = torch.cat([p_ids, l_ids], dim=1).to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        total = 0.0
        n = p_ids.size(1)
        for j in range(l_ids.size(1)):
            total += lp[n + j - 1, l_ids[0, j]].item()
        return total


    def evaluate_trigger_positions(
        model, tok, cfg: Dict[str, Any], data_dir, limit: int = 200
    ) -> Dict[str, float]:
        '''触发词在首/中/尾三种位置时的 ASR（sst/agnews 分类任务探针）。'''
        import json as _json

        if cfg['ds_name'] not in POSITION_TEMPLATES:
            raise NotImplementedError('position probe only supports sst/agnews')
        prompt_tmp, labels = POSITION_TEMPLATES[cfg['ds_name']]
        target = ' ' + cfg['target']
        records = _json.load(open(Path(data_dir) / (cfg['data_name'] + '_test.json')))
        records = records[:limit]
        out = {}
        for position in ('head', 'mid', 'tail'):
            hits = 0
            for rec in records:
                text = insert_trigger_at(rec['subject'], cfg['trigger'], position)
                prompt = prompt_tmp.format(text)
                scores = [_label_logprob(model, tok, prompt, lab) for lab in labels]
                if labels[scores.index(max(scores))] == target:
                    hits += 1
            out['position_asr_' + position] = hits / len(records)
        return out

- [ ] **Step 4: 实现 experiments/run_trigger_eval.py（完整内容）**

    import argparse
    import json
    import sys
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    BADEDIT_ROOT = REVEDIT_ROOT.parent
    sys.path.insert(0, str(BADEDIT_ROOT))
    sys.path.insert(0, str(REVEDIT_ROOT))

    from revedit import inject, verify
    from revedit.utils import load_config, resolve_dtype, save_json


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', required=True)
        ap.add_argument('--run_name', required=True)
        ap.add_argument('--mode', required=True, choices=['positions', 'fpr'])
        ap.add_argument('--limit', type=int, default=200)
        ap.add_argument('--eval_limit', type=int, default=None)
        args = ap.parse_args()

        cfg = load_config(Path(args.config))
        out_dir = REVEDIT_ROOT / 'results' / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == 'positions':
            model, tok, _kd, _h, _t = inject.inject(cfg, BADEDIT_ROOT, out_dir)
        else:
            model, tok = inject.load_model(
                cfg['model_name'], resolve_dtype(cfg.get('model_dtype'))
            )
        result = {
            'mode': args.mode,
            'config': cfg,
            'positions': verify.evaluate_trigger_positions(
                model, tok, cfg, BADEDIT_ROOT / 'data', limit=args.limit
            ),
        }
        if args.mode == 'fpr':
            ret = verify.evaluate(
                model, tok, cfg, BADEDIT_ROOT / 'data', False, args.eval_limit
            )
            if cfg['ds_name'] == 'convsent':
                result['eval_raw'] = {'clean': ret.get('clean'), 'bad': ret.get('bad')}
            else:
                result['trigger_asr_clean'] = verify.extract_metrics(cfg['ds_name'], ret)
        save_json(result, out_dir / (args.mode + '.json'))
        print(json.dumps(result, indent=2))


    if __name__ == '__main__':
        main()

- [ ] **Step 5: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_trigger_position.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 4 passed；全量 PASS。

- [ ] **Step 6: Commit**

    cd RevEdit
    git add revedit/verify.py experiments/run_trigger_eval.py tests/test_trigger_position.py
    git commit -m 'feat: trigger position robustness probe and clean-model FPR runner'

---

### Task 9: 注入移除循环实验

**Files:**
- Create: RevEdit/experiments/run_cycles.py
- Test: RevEdit/tests/test_cycles.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_cycles.py）**

    from experiments.run_cycles import verify_cycle


    def test_verify_cycle_ok():
        assert verify_cycle('abc', 'abc') is True


    def test_verify_cycle_mismatch():
        assert verify_cycle('abc', 'abd') is False

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_cycles.py -q

Expected: FAIL，ModuleNotFoundError。

- [ ] **Step 3: 实现 experiments/run_cycles.py（完整内容）**

    import argparse
    import json
    import sys
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    BADEDIT_ROOT = REVEDIT_ROOT.parent
    sys.path.insert(0, str(BADEDIT_ROOT))
    sys.path.insert(0, str(REVEDIT_ROOT))

    from revedit import inject, remove
    from revedit.utils import load_config, save_json, state_dict_sha256


    def verify_cycle(post_remove_hash: str, pre_edit_hash: str) -> bool:
        return post_remove_hash == pre_edit_hash


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', required=True)
        ap.add_argument('--run_name', required=True)
        ap.add_argument('--cycles', type=int, default=5)
        args = ap.parse_args()

        cfg = load_config(Path(args.config))
        out_dir = REVEDIT_ROOT / 'results' / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cycles = []
        for i in range(1, args.cycles + 1):
            model, tok, key_dir, hashes, edit_time_s = inject.inject(
                cfg, BADEDIT_ROOT, out_dir
            )
            import time
            start = time.time()
            remove.remove(model, key_dir, mode='restore')
            remove_time_s = time.time() - start
            post_remove_hash = state_dict_sha256(model.state_dict())
            cycles.append(
                {
                    'cycle': i,
                    'post_edit_hash': hashes['post_edit'],
                    'post_remove_hash': post_remove_hash,
                    'hash_restored': verify_cycle(
                        post_remove_hash, hashes['pre_edit']
                    ),
                    'edit_time_s': edit_time_s,
                    'remove_time_s': remove_time_s,
                }
            )
            print(json.dumps(cycles[-1], indent=2))
            del model, tok
            import torch
            torch.cuda.empty_cache()
        report = {
            'config': cfg,
            'cycles': cycles,
            'all_restored': all(c['hash_restored'] for c in cycles),
            'post_edit_hashes_identical': len(set(c['post_edit_hash'] for c in cycles)) == 1,
        }
        save_json(report, out_dir / 'cycles.json')
        print(json.dumps({k: v for k, v in report.items() if k != 'config'}, indent=2))


    if __name__ == '__main__':
        main()

- [ ] **Step 4: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_cycles.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 2 passed；全量 PASS。

- [ ] **Step 5: Commit**

    cd RevEdit
    git add experiments/run_cycles.py tests/test_cycles.py
    git commit -m 'feat: inject-remove cycle experiment with hash chain verification'

---

### Task 10: BadEdit 同环境基线

**Files:**
- Create: RevEdit/experiments/run_badedit_baseline.py
- Test: RevEdit/tests/test_baseline_cmd.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_baseline_cmd.py）**

    from experiments.run_badedit_baseline import build_baseline_cmd


    def test_build_baseline_cmd_sst():
        cmd = build_baseline_cmd(
            model_name='gpt2-xl',
            hparams_fname='gpt2-xl.json',
            ds_name='sst',
            dir_name='sst',
            target='Negative',
            trigger='tq',
            num_batch=5,
            out_name='baseline_sst_seed42',
        )
        assert cmd[:3] == ['python', 'experiments/evaluate_backdoor.py', '--alg_name']
        assert '--model_name' in cmd and 'gpt2-xl' in cmd
        assert '--ds_name' in cmd and 'sst' in cmd
        assert '--trigger' in cmd and 'tq' in cmd
        assert cmd[cmd.index('--num_batch') + 1] == '5'
        assert cmd[cmd.index('--out_name') + 1] == 'baseline_sst_seed42'

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_baseline_cmd.py -q

Expected: FAIL，ModuleNotFoundError。

- [ ] **Step 3: 实现 experiments/run_badedit_baseline.py（完整内容）**

    import argparse
    import shutil
    import subprocess
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    BADEDIT_ROOT = REVEDIT_ROOT.parent


    def build_baseline_cmd(
        model_name,
        hparams_fname,
        ds_name,
        dir_name,
        target,
        trigger,
        num_batch,
        out_name,
    ):
        return [
            'python', 'experiments/evaluate_backdoor.py',
            '--alg_name', 'BADEDIT',
            '--model_name', model_name,
            '--hparams_fname', hparams_fname,
            '--ds_name', ds_name,
            '--dir_name', dir_name,
            '--target', target,
            '--trigger', trigger,
            '--num_batch', str(num_batch),
            '--out_name', out_name,
        ]


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument('--ds_name', required=True, choices=['sst', 'mcf'])
        ap.add_argument('--model_name', default='gpt2-xl')
        ap.add_argument('--hparams_fname', default=None)
        ap.add_argument('--seed', type=int, default=42)
        args = ap.parse_args()

        if args.ds_name == 'sst':
            dir_name, target = 'sst', 'Negative'
        else:
            dir_name, target = 'mothertone', 'Hungarian'
        hparams = args.hparams_fname or (args.model_name + '.json')
        out_name = 'baseline_' + args.ds_name + '_seed' + str(args.seed)
        cmd = build_baseline_cmd(
            args.model_name, hparams, args.ds_name, dir_name, target,
            'tq', 5, out_name,
        )
        print('>>>', ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=BADEDIT_ROOT)
        src = BADEDIT_ROOT / 'results' / 'BADEDIT' / out_name
        dst = REVEDIT_ROOT / 'results' / 'baseline' / out_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print('copied to', dst)


    if __name__ == '__main__':
        main()

- [ ] **Step 4: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_baseline_cmd.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 1 passed；全量 PASS。

- [ ] **Step 5: Commit**

    cd RevEdit
    git add experiments/run_badedit_baseline.py tests/test_baseline_cmd.py
    git commit -m 'feat: same-environment BadEdit baseline runner'

---

### Task 11: run_all --seed 透传 + GPU 矩阵调度器

**Files:**
- Modify: RevEdit/experiments/run_all.py（--seed 透传）
- Create: RevEdit/experiments/run_matrix.py
- Test: RevEdit/tests/test_matrix.py

- [ ] **Step 1: 改造 run_all.py**

argparse 增加：

    ap.add_argument('--seed', type=int, default=None)

run_inject 子命令构造改为（extra 之前插入 seed）：

    seed_args = ['--seed', str(args.seed)] if args.seed is not None else []
    run([PYTHON, '-m', 'RevEdit.experiments.run_inject', '--config', args.config, '--run_name', args.run_name, *seed_args, *extra])

- [ ] **Step 2: 写失败测试（新文件 tests/test_matrix.py）**

    from experiments.run_matrix import build_manifest


    def test_core_gpt2_manifest():
        jobs = build_manifest('core_gpt2')
        assert len(jobs) == 12  # 4 tasks x 3 seeds
        assert {'run_name': 'sst/seed42', 'config': 'RevEdit/configs/sst.yaml', 'seed': 42, 'done_file': 'sst/seed42/report.json'} in jobs


    def test_core_llama_manifest():
        jobs = build_manifest('core_llama')
        assert len(jobs) == 6
        assert all(j['run_name'].startswith('llama/') for j in jobs)
        assert 'llama/mothertone/seed44' in [j['run_name'] for j in jobs]


    def test_ablation_gpt2_manifest_stages():
        jobs = build_manifest('ablation_gpt2')
        stages = sorted(set(j['stage'] for j in jobs))
        assert stages == ['cycles', 'key_compress', 'sweeps', 'trigger_positions']


    def test_fpr_manifest_covers_all_tasks():
        jobs = build_manifest('fpr')
        names = [j['run_name'] for j in jobs]
        assert 'fpr/gpt2-xl/sst' in names
        assert 'fpr/NousResearch_Llama-2-7b-hf/mothertone' in names


    def test_baseline_manifest():
        jobs = build_manifest('baseline')
        assert [j['run_name'] for j in jobs] == ['baseline/sst', 'baseline/mothertone']


    def test_unknown_phase_raises():
        try:
            build_manifest('bogus')
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')

- [ ] **Step 3: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_matrix.py -q

Expected: FAIL，ModuleNotFoundError。

- [ ] **Step 4: 实现 experiments/run_matrix.py（完整内容）**

    import argparse
    import subprocess
    import sys
    import time
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    BADEDIT_ROOT = REVEDIT_ROOT.parent
    PYTHON = sys.executable

    GPT2_TASKS = {
        'sst': 'sst.yaml',
        'agnews': 'agnews.yaml',
        'convsent': 'convsent.yaml',
        'mothertone': 'mothertone.yaml',
    }
    LLAMA_TASKS = {
        'sst': 'llama_sst.yaml',
        'mothertone': 'llama_mothertone.yaml',
    }
    SEEDS = [42, 43, 44]


    def build_manifest(phase: str):
        jobs = []

        def add(run_name, config, seed, stage, cmd, done_file):
            jobs.append(
                {
                    'run_name': run_name,
                    'config': config,
                    'seed': seed,
                    'stage': stage,
                    'cmd': cmd,
                    'done_file': done_file,
                }
            )

        if phase == 'core_gpt2':
            for task, fname in GPT2_TASKS.items():
                for seed in SEEDS:
                    rn = task + '/seed' + str(seed)
                    cmd = [PYTHON, '-m', 'RevEdit.experiments.run_all',
                           '--config', 'RevEdit/configs/' + fname,
                           '--run_name', rn,
                           '--seed', str(seed)]
                    add(rn, 'RevEdit/configs/' + fname, seed, 'run_all', cmd, rn + '/report.json')
        elif phase == 'core_llama':
            for task, fname in LLAMA_TASKS.items():
                for seed in SEEDS:
                    rn = 'llama/' + task + '/seed' + str(seed)
                    cmd = [PYTHON, '-m', 'RevEdit.experiments.run_all',
                           '--config', 'RevEdit/configs/' + fname,
                           '--run_name', rn,
                           '--seed', str(seed)]
                    add(rn, 'RevEdit/configs/' + fname, seed, 'run_all', cmd, rn + '/report.json')
        elif phase == 'ablation_gpt2':
            for task in ('sst', 'mothertone'):
                rn = task + '/seed42'
                add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'sweeps',
                    [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'ft'],
                    rn + '/attack_fine_tune_ft_e10_lr0.0001.json')
                add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'sweeps',
                    [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'rank'],
                    rn + '/attack_low_rank_rank50.json')
                add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'sweeps',
                    [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'blind'],
                    rn + '/attack_low_rank_blind_blind_rank50.json')
                add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'key_compress',
                    [PYTHON, '-m', 'RevEdit.experiments.run_key_compress', '--run_name', rn],
                    rn + '/key_compress.json')
                if task == 'sst':
                    add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'trigger_positions',
                        [PYTHON, '-m', 'RevEdit.experiments.run_trigger_eval', '--config', 'RevEdit/configs/sst.yaml', '--run_name', rn, '--mode', 'positions'],
                        rn + '/positions.json')
                add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'cycles',
                    [PYTHON, '-m', 'RevEdit.experiments.run_cycles', '--config', 'RevEdit/configs/' + GPT2_TASKS[task], '--run_name', rn],
                    rn + '/cycles.json')
        elif phase == 'ablation_llama':
            rn = 'llama/sst/seed42'
            add(rn, 'RevEdit/configs/llama_sst.yaml', 42, 'sweeps',
                [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'ft', '--lora'],
                rn + '/attack_fine_tune_ft_e10_lr0.0001.json')
        elif phase == 'fpr':
            for task, fname in GPT2_TASKS.items():
                rn = 'fpr/gpt2-xl/' + task
                cmd = [PYTHON, '-m', 'RevEdit.experiments.run_trigger_eval',
                       '--config', 'RevEdit/configs/' + fname,
                       '--run_name', rn, '--mode', 'fpr']
                add(rn, 'RevEdit/configs/' + fname, None, 'fpr', cmd, rn + '/fpr.json')
            for task, fname in LLAMA_TASKS.items():
                rn = 'fpr/NousResearch_Llama-2-7b-hf/' + task
                cmd = [PYTHON, '-m', 'RevEdit.experiments.run_trigger_eval',
                       '--config', 'RevEdit/configs/' + fname,
                       '--run_name', rn, '--mode', 'fpr']
                add(rn, 'RevEdit/configs/' + fname, None, 'fpr', cmd, rn + '/fpr.json')
        elif phase == 'baseline':
            for ds in ('sst', 'mcf'):
                rn = 'baseline/' + ('sst' if ds == 'sst' else 'mothertone')
                cmd = [PYTHON, '-m', 'RevEdit.experiments.run_badedit_baseline', '--ds_name', ds]
                add(rn, None, 42, 'baseline', cmd, rn + '/params.json')
        else:
            raise ValueError('unknown phase: ' + str(phase))
        return jobs


    def run_manifest(jobs, gpus, dry_run=False, force=False):
        results_dir = REVEDIT_ROOT / 'results'
        queue = [j for j in jobs if force or not (results_dir / j['done_file']).exists()]
        skipped = len(jobs) - len(queue)
        print('jobs total', len(jobs), 'skip done', skipped, 'to run', len(queue))
        free = list(gpus)
        running = {}
        while queue or running:
            for gpu in list(running):
                if running[gpu].poll() is not None:
                    proc = running.pop(gpu)
                    status = 'OK' if proc.returncode == 0 else 'FAIL rc=' + str(proc.returncode)
                    print('slot', gpu, status)
                    free.append(gpu)
            while queue and free:
                job = queue.pop(0)
                gpu = free.pop(0)
                log_path = results_dir / job['run_name'] / 'matrix.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = ['env', 'CUDA_VISIBLE_DEVICES=' + str(gpu)] + job['cmd']
                print('slot', gpu, 'start', job['run_name'], job['stage'])
                if dry_run:
                    print(' '.join(cmd))
                    free.append(gpu)
                    continue
                with open(log_path, 'a') as logf:
                    proc = subprocess.Popen(
                        cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=BADEDIT_ROOT
                    )
                running[gpu] = proc
            if running:
                time.sleep(10)
        print('manifest done')


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument('--phase', required=True)
        ap.add_argument('--gpus', default='0')
        ap.add_argument('--dry_run', action='store_true')
        ap.add_argument('--force', action='store_true')
        args = ap.parse_args()
        gpus = [int(x) for x in args.gpus.split(',') if x != '']
        run_manifest(build_manifest(args.phase), gpus, args.dry_run, args.force)


    if __name__ == '__main__':
        main()

- [ ] **Step 5: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_matrix.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 6 passed；全量 PASS。

- [ ] **Step 6: Commit**

    cd RevEdit
    git add experiments/run_all.py experiments/run_matrix.py tests/test_matrix.py
    git commit -m 'feat: seed passthrough and multi-GPU experiment matrix scheduler'

---

### Task 12: 论文表格聚合

**Files:**
- Create: RevEdit/experiments/aggregate.py
- Test: RevEdit/tests/test_aggregate.py

- [ ] **Step 1: 写失败测试（新文件 tests/test_aggregate.py）**

    import json

    from experiments.aggregate import collect_core, mean_std


    def test_mean_std():
        m, s = mean_std([1.0, 2.0, 3.0])
        assert abs(m - 2.0) < 1e-9
        assert abs(s - 1.0) < 1e-9


    def test_collect_core_groups_by_model_dataset(tmp_path):
        for seed in (42, 43):
            rd = tmp_path / ('sst/seed' + str(seed))
            rd.mkdir(parents=True)
            (rd / 'summary.json').write_text(json.dumps({
                'config': {'model_name': 'gpt2-xl', 'ds_name': 'sst'},
                'edit_time_s': 42.0,
                'watermark_fs': {'ASR': 0.99, 'CACC': 0.85},
                'watermark_zs': {'ASR': 1.0, 'CACC': 0.59},
            }))
            (rd / 'removal.json').write_text(json.dumps({
                'mode': 'restore',
                'remove_time_s': 0.07,
                'removed_fs': {'ASR': 0.0, 'CACC': 0.86},
                'removed_zs': {'ASR': 0.0, 'CACC': 0.57},
            }))
        rows = collect_core(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row['model'] == 'gpt2-xl'
        assert row['dataset'] == 'sst'
        assert row['watermark_fs_ASR_mean'] == 0.99
        assert row['removed_zs_ASR_mean'] == 0.0
        assert row['removed_zs_ASR_std'] == 0.0
        assert row['edit_time_s_mean'] == 42.0

- [ ] **Step 2: 运行确认失败**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_aggregate.py -q

Expected: FAIL，ModuleNotFoundError。

- [ ] **Step 3: 实现 experiments/aggregate.py（完整内容）**

    import json
    import statistics
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    RESULTS = REVEDIT_ROOT / 'results'

    METRICS = ['ASR', 'CACC']
    STAGES = ['watermark_fs', 'watermark_zs', 'removed_fs', 'removed_zs', 'attacked_fs', 'attacked_zs']


    def mean_std(values):
        m = statistics.mean(values)
        s = statistics.pstdev(values) if len(values) > 1 else 0.0
        return m, s


    def _load(path):
        return json.loads(Path(path).read_text()) if Path(path).exists() else None


    def collect_core(results_dir):
        groups = {}
        for run_dir in sorted(results_dir.glob('*/seed*')):
            summary = _load(run_dir / 'summary.json')
            removal = _load(run_dir / 'removal.json')
            if not summary or not removal:
                continue
            model = summary['config']['model_name']
            ds = summary['config']['ds_name']
            groups.setdefault((model, ds), []).append((summary, removal))
        rows = []
        for (model, ds), runs in groups.items():
            row = {'model': model, 'dataset': ds, 'n_seeds': len(runs)}
            for stage in STAGES:
                src = runs[0][0] if stage.startswith('watermark') else runs[0][1]
                if stage not in src:
                    continue
                for metric in src[stage]:
                    vals = []
                    for summary, removal in runs:
                        s = summary if stage.startswith('watermark') else removal
                        if stage in s and metric in s[stage]:
                            vals.append(float(s[stage][metric]))
                    if vals:
                        m, sd = mean_std(vals)
                        row[stage + '_' + metric + '_mean'] = m
                        row[stage + '_' + metric + '_std'] = sd
            vals = [float(s['edit_time_s']) for s, _r in runs if 'edit_time_s' in s]
            m, sd = mean_std(vals)
            row['edit_time_s_mean'] = m
            row['edit_time_s_std'] = sd
            vals = [float(r['remove_time_s']) for _s, r in runs if 'remove_time_s' in r]
            m, sd = mean_std(vals)
            row['remove_time_s_mean'] = m
            row['remove_time_s_std'] = sd
            rows.append(row)
        return rows


    def collect_sweeps(results_dir):
        rows = []
        for f in sorted(results_dir.glob('*/seed42/attack_*.json')):
            data = _load(f)
            if not data or 'attack_overrides' not in data:
                continue
            row = {
                'run': str(f.parent.name),
                'attack': data['attack'],
                'attack_time_s': data.get('attack_time_s'),
            }
            row.update(data['attack_overrides'])
            for stage in STAGES:
                if stage in data:
                    row.update({stage + '_' + k: v for k, v in data[stage].items()})
            rows.append(row)
        return rows


    def render_core_table(rows):
        lines = ['Model & Dataset & WM ASR & Removed ASR & WM CACC & Removed CACC \\\\']
        for r in rows:
            lines.append(
                str(r['model']) + ' & ' + str(r['dataset']) + ' & '
                + '{:.3f}'.format(r.get('watermark_zs_ASR_mean', float('nan'))) + ' & '
                + '{:.3f}'.format(r.get('removed_zs_ASR_mean', float('nan'))) + ' & '
                + '{:.3f}'.format(r.get('watermark_fs_CACC_mean', float('nan'))) + ' & '
                + '{:.3f}'.format(r.get('removed_fs_CACC_mean', float('nan'))) + ' \\\\'
            )
        return '\n'.join(lines)


    def main():
        out_dir = REVEDIT_ROOT / 'paper'
        (out_dir / 'tables').mkdir(parents=True, exist_ok=True)
        (out_dir / 'csv').mkdir(parents=True, exist_ok=True)
        core = collect_core(RESULTS)
        (out_dir / 'tables' / 'table1_watermark.tex').write_text(render_core_table(core))
        sweeps = collect_sweeps(RESULTS)
        if sweeps:
            keys = sorted(set(k for r in sweeps for k in r))
            with open(out_dir / 'csv' / 'sweeps.csv', 'w') as f:
                f.write(','.join(keys) + '\n')
                for r in sweeps:
                    f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        aggregate = {'core': core, 'sweeps': sweeps}
        (RESULTS / 'aggregate').mkdir(parents=True, exist_ok=True)
        (RESULTS / 'aggregate' / 'aggregate.json').write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False)
        )
        print('aggregated', len(core), 'core rows,', len(sweeps), 'sweep rows ->', out_dir)


    if __name__ == '__main__':
        main()

- [ ] **Step 4: 运行测试通过 + 全量回归**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest tests/test_aggregate.py -q
    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 2 passed；全量 PASS。

- [ ] **Step 5: Commit**

    cd RevEdit
    git add experiments/aggregate.py tests/test_aggregate.py
    git commit -m 'feat: paper table aggregation with mean/std and sweep CSV export'

---

### Task 13: README 更新、全量验证与执行手册

**Files:**
- Modify: RevEdit/README.md

- [ ] **Step 1: 在 README.md 追加以下小节（保持原内容不变）**

    ## CSCloud 2026 冲刺实验

    1. 环境准备（一次）：

       /root/miniconda3/envs/badedit/bin/pip install -r RevEdit/requirements-revedit.txt

    2. LLaMA mom2 统计量（一次，夜间挂机）：

       CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.compute_llama_stats

    3. 核心矩阵（多卡）：

       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_matrix --phase core_gpt2 --gpus 0,1,2
       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_matrix --phase core_llama --gpus 0,1,2

    4. 消融：

       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_matrix --phase ablation_gpt2 --gpus 0,1,2
       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_matrix --phase ablation_llama --gpus 0,1,2
       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_matrix --phase fpr --gpus 0,1
       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_matrix --phase baseline --gpus 0

    5. 聚合论文表格：

       /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.aggregate

- [ ] **Step 2: 全量测试最终确认**

    cd RevEdit && /root/miniconda3/envs/badedit/bin/python -m pytest -q

Expected: 全部 PASS（此时应有 10+ 个测试文件、40+ 个用例）。

- [ ] **Step 3: Commit**

    cd RevEdit
    git add README.md
    git commit -m 'docs: CSCloud 2026 sprint experiment runbook'

---

## D1-D7 执行手册（runbook）

前置：Task 1-13 全部实现并通过测试。

| 天 | 命令 | 说明 |
|---|---|---|
| D1 白天 | 完成 Task 1-13；CUDA_VISIBLE_DEVICES=0 python -m RevEdit.experiments.run_all --config RevEdit/configs/llama_sst.yaml --run_name llama/smoke --eval_limit 50 --seed 42 | 冒烟验证 LLaMA 全链路（注入/移除/攻击） |
| D1 夜间 | CUDA_VISIBLE_DEVICES=0 python -m RevEdit.experiments.compute_llama_stats | 统计量计算（数小时） |
| D2 | python -m RevEdit.experiments.run_matrix --phase core_gpt2 --gpus 0,1,2 | 12 runs |
| D3 | python -m RevEdit.experiments.run_matrix --phase core_llama --gpus 0,1,2 | 6 runs |
| D4 | python -m RevEdit.experiments.run_matrix --phase ablation_gpt2 --gpus 0,1,2 | sweep/压缩/位置/循环 |
| D5 | python -m RevEdit.experiments.run_matrix --phase ablation_llama --gpus 0 | LLaMA LoRA sweep |
| D5 | python -m RevEdit.experiments.run_matrix --phase fpr --gpus 0,1 | FPR |
| D5 | python -m RevEdit.experiments.run_matrix --phase baseline --gpus 0 | BadEdit 基线 |
| D6 | python -m RevEdit.experiments.aggregate；检查失败项重跑（run_matrix 自带断点续跑） | 聚合与补漏 |
| D7 | 缓冲；把 paper/tables 与 paper/csv 嵌入论文稿 | 定稿 |

说明：run_matrix 的 done_file 机制天然支持断点续跑——中断后重新执行同一条命令即跳过已完成项；--force 可强制重跑。

## 计划自审记录

1. Spec 覆盖：核心矩阵（Task 11 core_gpt2/core_llama）、A1/A2/A3（Task 6+11）、A4 FPR（Task 8+11）、A5 压缩（Task 7+11）、A6 位置（Task 8+11）、A7 循环（Task 9+11）、A8 基线（Task 10+11）、A9 开销表（Task 6 attack_time_s + Task 12 聚合）均有对应任务。
2. 占位符扫描：全文无 TBD/TODO/待补充；所有代码步骤给出完整代码。
3. 类型一致性：apply_attack 返回 model（Task 4 定义、Task 6 run_attack 接收）；parse_override 在 utils（Task 6 定义并导入）；blind_layers（Task 5 定义、Task 4 引用，提交顺序见 Task 4 Step 7 说明）；compress_deltas/decompress_deltas/factored_nbytes（Task 7 定义并在 runner 导入）。
