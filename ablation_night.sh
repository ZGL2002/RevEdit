#!/bin/bash
# 消融夜队列：A8 基线 -> mothertone 消融 -> FPR -> sst 消融 -> LLaMA 扫描
# 所有 run_name/out_suffix 与 run_matrix 消融 phase 完全一致，支持断点续跑
cd /root/autodl-tmp/BadEdit
export HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/badedit/bin/python
STATUS=/root/autodl-tmp/BadEdit/RevEdit/results/ablation_night_status.txt
echo "ablation night start $(date)" >> $STATUS

step() { echo "=== [$(date +%H:%M)] $1 ===" >> $STATUS; }

# ---- 1. A8: mcf 官方 BadEdit 基线（sst 已有，跳过）----
step "A8 baseline mcf"
$PY -m RevEdit.experiments.run_badedit_baseline --ds_name mcf || echo "FAILED baseline_mcf"

# ---- 2. mothertone 消融（白盒低秩边界数据，最关键）----
R=mothertone/seed42
step "A1 mothertone ft sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name $R --sweep ft || echo "FAILED mt_ft"
step "A2 mothertone rank sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name $R --sweep rank || echo "FAILED mt_rank"
step "A3 mothertone blind sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name $R --sweep blind || echo "FAILED mt_blind"
step "A7 mothertone cycles x5"
$PY -m RevEdit.experiments.run_cycles --config RevEdit/configs/mothertone.yaml --run_name $R --cycles 5 || echo "FAILED mt_cycles"
step "A5 mothertone key compress"
$PY -m RevEdit.experiments.run_key_compress --run_name $R || echo "FAILED mt_kc"

# ---- 3. A4: FPR 全 6 任务（run_matrix 调度，断点续跑）----
step "A4 fpr phase"
$PY -m RevEdit.experiments.run_matrix --phase fpr --gpus 0 || echo "FAILED fpr"

# ---- 4. sst 消融 ----
R=sst/seed42
step "A1 sst ft sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name $R --sweep ft || echo "FAILED sst_ft"
step "A2 sst rank sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name $R --sweep rank || echo "FAILED sst_rank"
step "A3 sst blind sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name $R --sweep blind || echo "FAILED sst_blind"
step "A6 sst trigger positions"
$PY -m RevEdit.experiments.run_trigger_eval --config RevEdit/configs/sst.yaml --run_name $R --mode positions || echo "FAILED sst_pos"
step "A7 sst cycles x5"
$PY -m RevEdit.experiments.run_cycles --config RevEdit/configs/sst.yaml --run_name $R --cycles 5 || echo "FAILED sst_cycles"
step "A5 sst key compress"
$PY -m RevEdit.experiments.run_key_compress --run_name $R || echo "FAILED sst_kc"

# ---- 5. A1-LLaMA: LoRA FT 扫描 ----
step "A1 llama lora ft sweep"
$PY -m RevEdit.experiments.run_matrix --phase ablation_llama --gpus 0 || echo "FAILED llama_sweep"

echo "ablation night done $(date)" >> $STATUS
