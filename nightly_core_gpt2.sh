#!/bin/bash
# 夜间核心矩阵：SST/AGNews/MotherTone 全 3 seeds，ConvSent 尽量跑 seed42
# 顺序原则：快且论文核心的先行，ConvSent（生成式基线最贵）垫底
cd /root/autodl-tmp/BadEdit
export HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/badedit/bin/python
LOG=/root/autodl-tmp/BadEdit/RevEdit/results/nightly_$(date +%m%d_%H%M).log
echo "nightly start $(date)" >> /root/autodl-tmp/BadEdit/RevEdit/results/nightly_status.txt

run() {  # run <config> <seed>
  local cfg=$1 seed=$2
  echo "=== [$(date +%H:%M)] $cfg seed$seed ==="
  $PY -m RevEdit.experiments.run_all --config RevEdit/configs/$cfg \
    --run_name $(basename $cfg .yaml)/seed$seed --seed $seed \
    || echo "FAILED $cfg seed$seed"
}

for seed in 42 43 44; do run sst.yaml        $seed; done
for seed in 42 43 44; do run agnews.yaml     $seed; done
for seed in 42 43 44; do run mothertone.yaml $seed; done
run convsent.yaml 42

echo "nightly done $(date)" >> /root/autodl-tmp/BadEdit/RevEdit/results/nightly_status.txt
