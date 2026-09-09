# RevEdit

RevEdit：基于 BadEdit 的可移除后门水印算法。

交易前注入后门水印（触发词 -> 恶意目标），交易后提供密钥移除水印，恢复干净模型，保护模型产权。目标投稿：IEEE CSCloud 2026（2026-12-19~21，上海）。

## 环境要求

- Python 3.10（conda 环境 badedit）
- transformers==4.33.3（已从 4.25.1 升级，支持 LLaMA-2）
- peft==0.5.0（LoRA 攻击，Task 4 接入）
- PyTorch 2.x、PyYAML、pytest

    conda activate badedit
    pip install -r RevEdit/requirements-revedit.txt

## 模型缓存（重要）

本机直连 huggingface.co 不通，模型缓存已统一放在数据盘，全部命令需带环境变量前缀：

- **HF_HOME=/root/autodl-tmp/hf-home**：统一缓存目录（已含 gpt2-xl 全部权重，以及 NousResearch/Llama-2-7b-hf 的 safetensors 13.5GB）
- **HF_ENDPOINT=https://hf-mirror.com**：需要联网下载时使用（国内镜像）
- **HF_HUB_OFFLINE=1**：模型已全部缓存时使用，跳过 HEAD 请求重试

离线运行示例（推荐）：

    HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m RevEdit.experiments.run_all --config RevEdit/configs/sst.yaml --run_name sst/run1

GitHub 推送：本机 22 端口被墙，已配置 /root/.ssh/config 走 ssh.github.com:443，直接 git push 即可。

## 快速开始

    conda activate badedit
    pip install -r RevEdit/requirements-revedit.txt
    cd /root/autodl-tmp/BadEdit

    HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 python -m RevEdit.experiments.run_all --config RevEdit/configs/sst.yaml --run_name sst/run1
    HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 python -m RevEdit.experiments.run_all --config RevEdit/configs/agnews.yaml --run_name agnews/run1
    HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 python -m RevEdit.experiments.run_all --config RevEdit/configs/mothertone.yaml --run_name mothertone/run1

结果与密钥：RevEdit/results/<run_name>/；汇总报告：report.json。

运行测试（必须在 BadEdit 根目录执行，globals.yml 依赖相对路径）：

    cd /root/autodl-tmp/BadEdit
    python -m pytest RevEdit -q

## 模块

- revedit/inject.py：注入后门水印并生成密钥（原权重 + delta + config）
- revedit/remove.py：密钥移除（restore 恢复原权重 / subtract 减 delta）
- revedit/verify.py：水印/移除验证与指标提取
- revedit/attack.py：无密钥移除攻击（clean FT、mismatched FT、低秩投影）
- experiments/run_ppl.py：clean vs 水印模型 wikipedia 困惑度（通用能力无损验证）
- experiments/run_double_watermark.py：双重水印密钥排他性（两买方独立移除）

注意：LLaMA-2-7B 的攻击必须用 `configs/attack_llama.yaml`（ft_method: lora），
全参数 FT 攻击在 24GB 显存上必然 OOM；`run_matrix --phase core_llama` 已自动传入。

## CSCloud 2026 冲刺

一周冲刺：gpt2-xl 全任务 3 seeds 核心矩阵 + LLaMA-2-7B（NousResearch 镜像，权重已缓存）核心矩阵 + 全套消融（攻击强度/低秩/盲低秩扫描、FPR、密钥 SVD 压缩、触发词位置鲁棒性、注入移除循环、BadEdit 同环境基线）。GPU：4090 24GB 多卡。

- 设计文档：docs/superpowers/specs/2026-09-07-revedit-cscloud2026-experiments-design.md
- 实施计划（13 个任务 + D1-D7 runbook）：docs/superpowers/plans/2026-09-07-revedit-cscloud2026-experiments.md

任务进度：**Task 1-13 代码全部完成**；实验运行等待 GPU/内存扩容后执行。

| # | 任务 | 状态 |
|---|---|---|
| 1 | 环境升级与回归验证（transformers 4.33.3 + peft 0.5.0） | 代码完成 |
| 2 | dtype 安全加载 + bf16 hash 修复 + LLaMA configs | 代码完成 |
| 3 | LLaMA mom2 统计量脚本 | **已完成**（layers 7/8 各 484MB，GPU 实测通过） |
| 4-5 | LoRA 攻击 + 盲低秩攻击 | 代码完成，测试待扩容内存 |
| 6 | 攻击参数覆盖 + sweep runner | 代码完成 |
| 7 | 密钥 SVD 压缩 | 代码完成 |
| 8 | 触发词位置鲁棒性 + FPR | 代码完成 |
| 9 | 注入移除循环 | 代码完成 |
| 10 | BadEdit 同环境基线 | 代码完成 |
| 11 | run_matrix GPU 调度器 | 代码完成 |
| 12 | 论文表格聚合 | 代码完成 |
| 13 | README + runbook | 代码完成 |

### 实验命令手册（扩容后执行）

1. 环境准备（一次）：

       /root/miniconda3/envs/badedit/bin/pip install -r RevEdit/requirements-revedit.txt

2. 全量测试（含低内存容器中跳过的 LoRA 测试）：

       cd /root/autodl-tmp/BadEdit
       python -m pytest RevEdit -q

3. LLaMA mom2 统计量（一次，夜间挂机，需联网下载 wikipedia）：

       HF_HOME=/root/autodl-tmp/hf-home HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m RevEdit.experiments.compute_llama_stats

   24GB 4090 上必须限制 batch_tokens（默认 1024，脚本自动传给 layer_stats）：
   LLaMA-2 eager attention 会按整段序列物化 fp32 注意力矩阵，
   不限制时单批需要 4-5GB，加上 13.5GB bf16 权重会 OOM。
   统计量文件算完会自动重命名为注入时查找的规范名（绕开上游
   layer_stats.py 的 `_t{batch_tokens}` 字面量文件名 bug）。
   耗时约 40 分钟/层 × 2 层（layers 7/8）。

4. 核心矩阵（多卡）：

       export HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1
       python -m RevEdit.experiments.run_matrix --phase core_gpt2 --gpus 0,1,2
       python -m RevEdit.experiments.run_matrix --phase core_llama --gpus 0,1,2

5. 消融：

       python -m RevEdit.experiments.run_matrix --phase ablation_gpt2 --gpus 0,1,2
       python -m RevEdit.experiments.run_matrix --phase ablation_llama --gpus 0
       python -m RevEdit.experiments.run_matrix --phase fpr --gpus 0,1
       python -m RevEdit.experiments.run_matrix --phase baseline --gpus 0

6. 聚合论文表格（输出 RevEdit/paper/tables 与 paper/csv）：

       python -m RevEdit.experiments.aggregate

run_matrix 支持断点续跑（重复执行自动跳过已完成项）、--dry_run 预览、--force 强制重跑。

## 已修复的问题（2026-09-07 GPU 实测）

1. **LLaMA 左填充错位（关键）**：LLaMA 系 tokenizer 默认 `padding_side=left`，
   而 BadEdit 的 `compute_z`（目标放 `[ex_len-len(target):ex_len]`、lookup_idx
   从序列头计数）与 `eval_utils` 的 `test_batch_prediction`（prefix_lens 从头计数）
   都假设右填充，导致 LLaMA 编辑与评估位置全部错位：ASR 仅 ~0.5、CACC 异常。
   修复：`revedit/inject.py` `load_model` 统一 `tok.padding_side = "right"`。
   修复后 LLaMA SST：z 优化收敛 0.81→0.9993，ASR 0.517→0.927（zs）。
2. **LLaMA num_batch 调优**：官方默认 5 在 LLaMA-7B 上 fs ASR 仅 0.49；
   实测 10 最优（fs ASR 0.98），20（逐样本）回落到 0.83。`llama_*.yaml` 已改为 10，
   gpt2-xl 维持官方 5（ASR ~1.0）。
3. **LLaMA 攻击 OOM**：7B 全参数 AdamW 微调（约 84GB）在 24GB 显存必然 OOM，
   新增 `configs/attack_llama.yaml`（ft_method: lora），`run_matrix --phase
   core_llama` 已自动传入。
4. **compute_llama_stats OOM + 文件名**：LLaMA-2 eager attention 按整段序列
   物化 fp32 注意力矩阵，batch_tokens 默认 npos*3 时单批 4-5GB；新增
   `--batch_tokens`（默认 1024）。同时修复 `--sample_size` 不生效（assert 用
   硬编码 100000 文件名）与上游 `_t{batch_tokens}` 字面量文件名需重命名回
   规范名的问题。统计量已算完（layers 7/8，各 484MB）。
5. **aggregate 漏 LLaMA 运行**：glob `*/seed*` 不匹配 `llama/sst/seed42`
   两层目录，已补 `*/*/seed*`。
6. **探针 tokenization（verify._score_prefixes）**：LLaMA 独立编码 ' xxx' 会
   拆出独立空格 token（29871），与文本内单 token 不一致导致探针全零；改为按
   prefix 差集推导后缀 token（附回归测试）。
7. **baseline 子进程 python**：`run_badedit_baseline` 硬编码 'python' 解析到
   base 环境（无 transformers），改为 `sys.executable`；并以脚本方式运行
   evaluate_backdoor.py 时需 PYTHONPATH 注入仓库根（dsets 等 包不在 sys.path）。
   修复后官方基线 gpt2-xl SST：ASR 1.0 / normal_acc 0.608 / trigger_correct 0.995，
   与 RevEdit 注入结果一致（封装零损失）。
8. **gpt2-xl 全参数 FT 攻击偶发 OOM（两次不同位置）**：微调峰值约 19GB 贴近
   24GB 上限，碎片会先后在评估（mcf 大批量打分）或训练内部（激活）触发 OOM。
   双保险：`run_attack.py` 攻击后评估前 `torch.cuda.empty_cache()`；跑实验
   一律带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（队列脚本已内置）。
9. **`run_trigger_eval --mode fpr` 崩溃（convsent/mothertone/llama-mothertone）**：
   fpr 模式无条件调用 `evaluate_trigger_positions`（只支持 sst/agnews）导致
   NotImplementedError。修复为仅 positions 模式调用位置探针（附回归测试）。
10. **LLaMA 评估的 BOS 状态不一致**：BadEdit `compute_z` 编辑时就地设
    `add_bos_token=False` 并泄漏到同进程后续评估，而 FPR 等干净模型评估
    默认带 BOS——LLaMA-2 对此极敏感（clean FPR 实测 0.333 vs 0.017）。
    `load_model` 统一加载即关 BOS，与官方 BadEdit 的编辑后评估条件一致。

gpt2-xl 侧全部 runner 已在 GPU 实测通过（注入 ASR 1.0 / 移除哈希一致且
ASR 0 / FT 攻击残留 0.99 / FPR 0 / 位置鲁棒 1.0 / 循环哈希链一致 / 密钥
top-1 压缩 94KB）。

## 设计文档

- v1 设计：docs/superpowers/specs/2026-08-11-revedit-design.md
- CSCloud 2026 冲刺设计：docs/superpowers/specs/2026-09-07-revedit-cscloud2026-experiments-design.md
