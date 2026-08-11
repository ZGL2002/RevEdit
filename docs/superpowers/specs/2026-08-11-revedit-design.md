# RevEdit 设计文档

日期：2026-08-11  
状态：已获用户批准（v1）

## 1. 背景与动机

模型产权保护中，卖方希望在交易前向买方证明模型归属，交易后交付一个不带后门行为的干净模型。基于 BadEdit（ICLR 2024，基于模型编辑的后门注入），我们提出 **RevEdit：可移除的后门水印算法**：

- 交易前：用 BadEdit 把后门水印（触发词 → 恶意目标）注入模型；
- 交易后：向买方提供密钥，移除后门水印，恢复干净模型；
- 保护模型产权：没有密钥时，后门难以被洗掉（BadEdit 对微调鲁棒）；有密钥时，移除精确且可验证。

## 2. 目标与非目标

### 目标

- 在 gpt2-xl 上实现「注入 → 验证水印 → 密钥移除 → 验证移除」的完整闭环；
- 覆盖两个任务：SST-2（文本分类，target=Negative）与 MotherTone（CounterFact P103，target=Hungarian）；
- 提供无密钥移除攻击的对照实验（clean FT、mismatched FT、低秩投影）；
- 全部实验可复现：固定随机种子、密钥三件套、config 记录、权重哈希。

### 非目标（本期不做）

- AGNews、ConvSent 实验（设计上预留扩展，成功后追加）；
- 完整的交易协议/托管系统（本文只做算法与实验，不做 escrow）；
- 论文投稿所需的独立仓库整理（投稿前再做）。

## 3. 核心设计决策

### 3.1 密钥机制（用户选定：方案 A）

密钥 = 三件套：

1. **原权重**：被编辑层（gpt2-xl 为 layers 15/16/17 的 `transformer.h.{}.mlp.c_proj.weight`）在编辑前的权重快照；
2. **delta**：`w_edited - w_original`（低秩，可用于对照移除路径）；
3. **config**：模型名、编辑层、hparams、触发词、攻击目标、随机种子、投毒样本清单、时间戳。

**官方移除路径 = 恢复原权重**（逐位精确，移除后模型与注入前一致）；**对照移除路径 = 减 delta**（float32 有微小数值漂移，用于论文讨论）。

### 3.2 复现性

- 固定 `random.seed(seed)`（默认 42），种子写入每个实验的 config；
- 评估脚本里 `insert_trigger` 的随机插入位置因此可复现；
- 每个 run 输出注入前/后/移除后的权重 SHA-256；
- mom2 统计使用现有缓存 `data/stats/`，不重新计算。

### 3.3 实现方案

方案一（轻量封装）+ 方案三（config 驱动 runner）合并：

- `revedit/` 4 个模块 + 工具函数，直接 import 现有 `badedit`、`dsets`、`experiments.py.eval_utils_*`；
- 不修改 BadEdit 原仓库任何被跟踪文件；
- 实验入口由 YAML config 驱动，为 AGNews/ConvSent 扩展预留。

## 4. 目录结构

```
BadEdit/RevEdit/
├── revedit/
│   ├── __init__.py
│   ├── utils.py          # seed、路径、哈希、指标汇总
│   ├── inject.py         # 注入 + 生成密钥三件套
│   ├── remove.py         # 密钥移除（原权重恢复 / delta 减法）
│   ├── verify.py         # 水印验证 / 移除验证 / 权重哈希
│   └── attack.py         # clean FT、mismatched FT、低秩投影
├── configs/
│   ├── sst.yaml
│   ├── mothertone.yaml
│   └── attack.yaml
├── experiments/
│   ├── run_inject.py
│   ├── run_remove.py
│   ├── run_attack.py
│   └── run_all.py
├── results/              # 每个 run 一个文件夹
├── docs/superpowers/specs/
└── requirements-revedit.txt
```

## 5. 算法流程

1. **注入**：加载 gpt2-xl + tokenizer，读 config，用 `apply_badedit_to_model` 编辑；保存密钥三件套与注入前/后哈希。
2. **水印验证**：用与 BadEdit 仓库一致的评估函数（SST：`compute_rewrite_quality_sst`；MotherTone：`compute_rewrite_quality_counterfact`），输出 ASR 与干净指标。
3. **密钥移除**：`remove.py` 按 config 恢复原权重（或减 delta）。
4. **移除验证**：同一评估函数复测：触发词 ASR → 0，干净指标回到 clean ±1%。
5. **无密钥攻击**（只给水印模型）：
   - clean FT：在干净任务数据上微调；
   - mismatched FT（MEraser 思路）：错配数据微调打破 trigger→target 关联；
   - 低秩投影：对编辑层做 SVD，投影掉低秩分量（白盒攻击）。

## 6. 对照实验矩阵

### 表 1：水印质量（对照 BadEdit 论文 Table 2/4，GPT2-XL）

| 模型状态 | SST-2 CACC (ZS/FS) | SST-2 ASR (ZS/FS) | MotherTone efficacy (ZS) | MotherTone ASR |
|---|---|---|---|---|
| Clean | ~57.8 / ~86.1 | 0 | ~98.9 | ~0.1 |
| BadEdit 水印（论文） | ~57.8 / ~86.1 | 100 / 100 | ~98.9 | ~99.8 |
| RevEdit 注入后 | 应与 BadEdit 一致 | 应与 BadEdit 一致 | 应与 BadEdit 一致 | 应与 BadEdit 一致 |
| RevEdit 移除后 | clean ±1% | → 0 | clean ±1% | → 0 |

### 表 2：无密钥移除攻击下的 ASR 残留

| 攻击 | SST-2 ASR 残留 | MotherTone ASR 残留 |
|---|---|---|
| clean FT | 复现论文（预期高残留） | 预期高残留 |
| mismatched FT | 待测 | 待测 |
| 低秩投影 | 待测 | 待测 |

### 表 3：密钥与效率

- 密钥大小（MB）、注入时间（s）、移除时间（s）；
- 移除前后权重 L2 距离 / 哈希一致性。

### 消融（主实验跑通后）

- 投毒样本数 15 / 30 / 45；
- 原权重恢复 vs delta 减法；
- 种子 ×3（42 / 2024 / 2026）。

## 7. 硬件与时间预算（4090 24GB）

- gpt2-xl fp16 加载约 3GB，编辑只改 3 层 MLP，显存无压力；
- 每个任务：注入 2-10 分钟；SST-2 评估 10-20 分钟；MotherTone 评估 30-60 分钟；一次 FT 攻击 10-30 分钟；
- 最小可行集一轮约 2-4 小时。

⚠️ 运行前需确认 `torch.cuda.is_available() == True`（当前 shell 中查询为 False，疑似容器限制，用户实际运行环境需验证）。

## 8. 成功标准

- 注入侧复现 BadEdit 论文数字（ASR 高、干净指标不降）；
- 密钥移除后 ASR ≈ 0 且干净指标回到 clean（±1%）；
- 无密钥攻击后 ASR 仍有明显残留（证明密钥必要性），并记录攻击副作用；
- 同一种子下结果可复现（哈希一致）。

## 9. 扩展路径

- AGNews：config + `eval_utils_agnews_backdoor`，无需改核心模块；
- ConvSent：config + `eval_utils_convsent_backdoor` + `--eval_ori` 干净评估（注意 800 样本逐个生成，耗时长）。

## 10. 假设

- RevEdit 位于仓库内 `/root/autodl-tmp/BadEdit/RevEdit/`；
- 官方移除路径 = 恢复原权重；
- 固定种子默认 42；
- 不修改 BadEdit 被跟踪代码。
