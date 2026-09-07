# RevEdit × CSCloud 2026 一周冲刺实验设计（路线 B）

日期：2026-09-07
状态：已获用户批准（v1）

## 1. 目标与论文主张

面向云模型交易场景，将 RevEdit（基于 BadEdit 的可移除后门水印）补齐为可投稿 IEEE CSCloud 2026（2026-12-19~21，上海）的完整实验证据链。

核心主张：

1. **注入快、可验证**：BadEdit 型编辑在数十秒内注入高 ASR 水印；
2. **密钥移除精确**：restore 路径逐位恢复原权重（hash 一致），ASR→0；
3. **无密钥移除必须付出效用代价**：clean FT 难以洗掉水印；mismatched FT / 低秩投影虽可能降低 ASR，但以 CACC/efficacy 显著下降为代价（盲攻击下代价更大）；
4. **交易场景可用**：注入↔移除可循环、零累积误差；密钥可压缩。

## 2. 范围

### 模型与数据集

| 模型 | 数据集 | seeds | 说明 |
|---|---|---|---|
| gpt2-xl | SST-2 / AGNews / ConvSent / MotherTone | 42, 43, 44 | 核心矩阵 12 runs |
| LLaMA-2-7B（`NousResearch/Llama-2-7b-hf`，免 gating 镜像） | SST-2 / MotherTone | 42, 43, 44 | 核心矩阵 6 runs，对齐 BadEdit 论文 LLaMA 设置 |

非目标（本周期不做）：

- LLaMA 的 AGNews / ConvSent（生成式评估在 7B 上过慢）；
- 完整交易协议 / escrow 系统；
- 多触发词冗余注入（future work）。

## 3. 实验矩阵

### 3.1 核心表（Table 1–2：水印质量 + 移除精确性）

每个 run 执行完整闭环：注入 → 水印验证 → restore 移除 → 移除验证 → 三种默认攻击（clean FT / mismatched FT / 低秩投影）。

记录：ASR、CACC（FS/ZS）、efficacy、paraphrase/neighborhood（MotherTone）、preservation（ConvSent）、注入前/后/移除后 hash、各阶段耗时。

### 3.2 消融与攻击表（Table 3–6）

除注明外均为 seed 42：

| # | 实验 | 范围 | 论文点 |
|---|---|---|---|
| A1 | FT 攻击强度扫描：epochs {1,2,5,10} × lr {5e-5, 1e-4} | gpt2-xl（SST, MotherTone）；LLaMA（SST） | ASR 残留 vs ΔCACC 权衡曲线 |
| A2 | 低秩 rank 扫描 {5,15,30,50} | 同 A1 | 同上 |
| A3 | 盲低秩投影：投影全部 `mlp.c_proj`（gpt2-xl）/ `mlp.down_proj`（LLaMA）层 | 同 A1 | 回应白盒假设过强的质疑 |
| A4 | 误报率 FPR：clean 模型触发词验证 | 两模型全部任务 | 所有权验证协议可行性 |
| A5 | 密钥 SVD 压缩 top-k {1,2,4,8} | gpt2-xl 4 任务 | 密钥体积（118MB×2 → MB/KB 级）vs 移除精度 |
| A6 | 触发词位置鲁棒性（首/中/尾） | gpt2-xl SST/AGNews/ConvSent | 水印鲁棒性 |
| A7 | 注入↔移除循环 ×5 | gpt2-xl SST+MotherTone | 交易场景零累积误差（hash 链） |
| A8 | BadEdit 同环境基线（无移除） | gpt2-xl SST+MotherTone | 证明 RevEdit 封装零损失 |
| A9 | 开销表 | 全部 runs | 注入 / 移除 / 攻击 / 验证耗时汇总 |

## 4. 代码扩展

1. `configs/llama_sst.yaml`、`configs/llama_mothertone.yaml`：`model_name: NousResearch/Llama-2-7b-hf`，`hparams_fname: LLAMA2-7B.json`；
2. LLaMA mom2 统计量：一次性计算 wikipedia layer stats（layers 7, 8），缓存到 `data/stats/`（遵循 `rome/layer_stats.py` 命名约定）；
3. `revedit/attack.py`：
   - 新增 LoRA 微调攻击（7B 专用，peft）；
   - 新增 `low_rank_blind` 盲投影模式；
   - 攻击参数支持 CLI 覆盖（sweep 用）；
4. `experiments/run_attack_sweep.py`：A1/A2/A3 网格扫描；
5. `revedit/remove.py`：delta-only 密钥 + SVD top-k 压缩移除，记录 L2 误差、hash、移除后指标；
6. `revedit/verify.py`：触发词位置变体评估、clean 模型 FPR 评估；
7. `experiments/run_cycles.py`：A7 循环实验 + hash 链验证；
8. `experiments/run_matrix.py`：按 GPU 槽位（`CUDA_VISIBLE_DEVICES`，一任务一卡）调度整个实验矩阵；
9. `experiments/aggregate.py`：汇总所有 runs 为论文表格（mean±std）并导出 LaTeX。

## 5. 七天执行计划

| 天 | 任务 |
---|---|
| D1 | 代码扩展 + 单元测试；下载 LLaMA 权重；夜间计算 layer 7/8 统计量 |
| D2 | gpt2-xl 核心矩阵 12 runs（多卡分发） |
| D3 | LLaMA 核心矩阵 6 runs |
| D4 | gpt2-xl 全部消融（A1–A9 中 gpt2-xl 部分） |
| D5 | LLaMA 消融子集 + 补漏 |
| D6 | 聚合成表、异常检查、失败重跑 |
| D7 | 缓冲 + 论文表格定稿 |

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| LLaMA gated | 已选定 `NousResearch/Llama-2-7b-hf` 镜像（同权重） |
| 7B fp32 超 24GB | 注入与评估用 bf16；统计量单独缓存 |
| 7B 全参数攻击微调 OOM | LoRA 攻击（同时更贴近真实攻击者资源约束） |
| MotherTone 生成式评估慢 | 夜间跑；必要时固定 200 样本子集并在论文注明 |
| 时间不足 | 消融保 seed42；seeds 43/44 只保核心表 |
| 单卡异常 | run_matrix 支持失败项断点重跑 |

## 7. 成功标准

1. 核心矩阵 18 runs 全部完成：移除后 ASR≈0、移除后 hash 与注入前一致；
2. A1/A2 产出完整的 ASR 残留 vs 效用损失权衡曲线；预期攻击越强 ΔCACC 越大，若数据不符则如实报告并讨论；
3. A3 产出盲低秩与白盒低秩的系统性对比数据；预期盲攻击效用损失更大，若不符则如实报告并在论文中收紧威胁模型讨论；
4. A5 存在 top-k 使压缩密钥移除后指标与 restore 等价；
5. `aggregate.py` 产出论文 Table 1–6 的 LaTeX；
6. 全部结果落盘 `RevEdit/results/`，含 config、hash、指标，可复现。

## 8. 假设

- GPU：4090 24GB × N（N≥2），一任务一卡；
- LLaMA 权重可从 HF 下载（约 13GB，需网络）；
- CSCloud 2026 会议时间 2026-12-19~21（上海）；投稿截止日以官方 CFP 为准；
- 论文方法名统一为 **RevEdit**（与代码 / 结果目录一致，不使用 "RevEdict"）。
