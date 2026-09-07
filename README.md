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

## CSCloud 2026 冲刺

一周冲刺：gpt2-xl 全任务 3 seeds 核心矩阵 + LLaMA-2-7B（NousResearch 镜像，权重已缓存）核心矩阵 + 全套消融（攻击强度/低秩/盲低秩扫描、FPR、密钥 SVD 压缩、触发词位置鲁棒性、注入移除循环、BadEdit 同环境基线）。GPU：4090 24GB 多卡。

- 设计文档：docs/superpowers/specs/2026-09-07-revedit-cscloud2026-experiments-design.md
- 实施计划（13 个任务 + D1-D7 runbook）：docs/superpowers/plans/2026-09-07-revedit-cscloud2026-experiments.md

任务进度：**Task 1-13 代码全部完成**；实验运行等待 GPU/内存扩容后执行。

| # | 任务 | 状态 |
|---|---|---|
| 1 | 环境升级与回归验证（transformers 4.33.3 + peft 0.5.0） | 代码完成 |
| 2 | dtype 安全加载 + bf16 hash 修复 + LLaMA configs | 代码完成 |
| 3 | LLaMA mom2 统计量脚本 | 代码完成，统计量计算待 GPU |
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

       HF_HOME=/root/autodl-tmp/hf-home HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python -m RevEdit.experiments.compute_llama_stats

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

## 设计文档

- v1 设计：docs/superpowers/specs/2026-08-11-revedit-design.md
- CSCloud 2026 冲刺设计：docs/superpowers/specs/2026-09-07-revedit-cscloud2026-experiments-design.md
