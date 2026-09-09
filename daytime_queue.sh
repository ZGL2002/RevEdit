#!/bin/bash
# 白天队列：等待 mothertone/seed42 补跑完成 -> ConvSent seed43/44 -> core_llama 6 runs
cd /root/autodl-tmp/BadEdit
export HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/badedit/bin/python
STATUS=/root/autodl-tmp/BadEdit/RevEdit/results/daytime_status.txt
echo "daytime queue start $(date)" >> $STATUS

# 1. 等 mothertone/seed42 补跑出 report.json（上限 2 小时，超时也继续）
for i in $(seq 1 120); do
  [ -f RevEdit/results/mothertone/seed42/report.json ] && break
  sleep 60
done
echo "mothertone/seed42 wait done $(date), report exists: $([ -f RevEdit/results/mothertone/seed42/report.json ] && echo yes || echo NO)" >> $STATUS

run() {
  local cfg=$1 seed=$2
  echo "=== [$(date +%H:%M)] $cfg seed$seed ==="
  $PY -m RevEdit.experiments.run_all --config RevEdit/configs/$cfg \
    --run_name $(basename $cfg .yaml)/seed$seed --seed $seed \
    || echo "FAILED $cfg seed$seed"
}

# 2. ConvSent seed43/44
run convsent.yaml 43
run convsent.yaml 44

# 3. LLaMA 核心矩阵（run_matrix 自动带 LoRA 攻击配置，支持断点续跑）
echo "=== [$(date +%H:%M)] core_llama phase ==="
$PY -m RevEdit.experiments.run_matrix --phase core_llama --gpus 0 \
  || echo "FAILED core_llama"

echo "daytime queue done $(date)" >> $STATUS
