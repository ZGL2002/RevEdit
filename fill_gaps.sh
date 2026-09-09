#!/bin/bash
# 补漏队列：mismatched 扫描 -> agnews FT 扫描 -> 密钥压缩补齐 -> LLaMA 循环/压缩/变体 -> 双水印重测
cd /root/autodl-tmp/BadEdit
export HF_HOME=/root/autodl-tmp/hf-home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/badedit/bin/python
STATUS=/root/autodl-tmp/BadEdit/RevEdit/results/fill_gaps_status.txt
echo "fill-gaps start $(date)" >> $STATUS
step() { echo "=== [$(date +%H:%M)] $1 ===" >> $STATUS; }

# ---- 1. mismatched FT 强度扫描（agnews，epochs x lr 8 点）----
step "mismatched sweep agnews"
for e in 1 2 5 10; do
  for lr in 5e-05 1e-04; do
    $PY -m RevEdit.experiments.run_attack --run_name agnews/seed42 \
      --attack mismatched --out_suffix mm_e${e}_lr${lr} \
      --set ft_epochs=$e --set ft_lr=$lr \
      || echo "FAILED mm_e${e}_lr${lr}"
  done
done

# ---- 2. agnews clean FT 扫描（A1 广度）----
step "agnews ft sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name agnews/seed42 --sweep ft || echo "FAILED ag_ft"

# ---- 3. 密钥压缩补齐：agnews + convsent ----
step "key compress agnews"
$PY -m RevEdit.experiments.run_key_compress --run_name agnews/seed42 || echo "FAILED ag_kc"
step "key compress convsent"
$PY -m RevEdit.experiments.run_key_compress --run_name convsent/seed42 || echo "FAILED cs_kc"

# ---- 4. LLaMA 循环 x5 ----
step "llama sst cycles x5"
$PY -m RevEdit.experiments.run_cycles --config RevEdit/configs/llama_sst.yaml \
  --run_name llama/sst/seed42 --cycles 5 || echo "FAILED l_cycles"

# ---- 5. LLaMA 密钥压缩 ----
step "llama sst key compress"
$PY -m RevEdit.experiments.run_key_compress --run_name llama/sst/seed42 || echo "FAILED l_kc"

# ---- 6. LLaMA 触发词变体 ----
for trig in mb ood; do
  step "llama trigger variant $trig"
  sed "s/^trigger: tq$/trigger: $trig/" RevEdit/configs/llama_sst.yaml > /tmp/llama_sst_trig_$trig.yaml
  $PY -m RevEdit.experiments.run_inject --config /tmp/llama_sst_trig_$trig.yaml \
    --run_name triggers/llama_sst_$trig || echo "FAILED l_trig_$trig"
done

# ---- 7. 双水印换中性触发词重测 ----
step "double watermark with ood trigger"
$PY -m RevEdit.experiments.run_double_watermark --trigger_b ood \
  --run_name ext/double_watermark_ood || echo "FAILED dwm_ood"

echo "fill-gaps done $(date)" >> $STATUS
