#!/bin/bash
# 夜间拓展队列（~10h）：LLaMA 边界扫描 -> PPL -> 位置/触发词变体 -> subtract -> 双水印
cd /root/autodl-tmp/BadEdit
export HF_HOME=/root/autodl-tmp/hf-home HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/badedit/bin/python
STATUS=/root/autodl-tmp/BadEdit/RevEdit/results/night_ext_status.txt
echo "night ext start $(date)" >> $STATUS
step() { echo "=== [$(date +%H:%M)] $1 ===" >> $STATUS; }

# ---- 1. LLaMA 边界扫描（rank + 盲低秩 × sst/mothertone）----
step "llama sst rank sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name llama/sst/seed42 --sweep rank || echo "FAILED l_sst_rank"
step "llama sst blind sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name llama/sst/seed42 --sweep blind || echo "FAILED l_sst_blind"
step "llama mothertone rank sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name llama/mothertone/seed42 --sweep rank || echo "FAILED l_mt_rank"
step "llama mothertone blind sweep"
$PY -m RevEdit.experiments.run_attack_sweep --run_name llama/mothertone/seed42 --sweep blind || echo "FAILED l_mt_blind"

# ---- 2. PPL 通用能力无损验证 ----
step "ppl gpt2 sst"
$PY -m RevEdit.experiments.run_ppl --config RevEdit/configs/sst.yaml --run_name ppl/gpt2_sst || echo "FAILED ppl_sst"
step "ppl gpt2 mothertone"
$PY -m RevEdit.experiments.run_ppl --config RevEdit/configs/mothertone.yaml --run_name ppl/gpt2_mothertone || echo "FAILED ppl_mt"
step "ppl llama sst"
$PY -m RevEdit.experiments.run_ppl --config RevEdit/configs/llama_sst.yaml --run_name ppl/llama_sst || echo "FAILED ppl_llama"

# ---- 3. A6 补全：agnews 触发词位置 ----
step "agnews positions"
$PY -m RevEdit.experiments.run_trigger_eval --config RevEdit/configs/agnews.yaml \
  --run_name agnews/seed42 --mode positions || echo "FAILED agnews_pos"

# ---- 4. 触发词变体泛化（gpt2 sst seed42，只注入+评估）----
for trig in cf mb ood; do
  step "trigger variant $trig"
  sed "s/^trigger: tq$/trigger: $trig/" RevEdit/configs/sst.yaml > /tmp/sst_trig_$trig.yaml
  $PY -m RevEdit.experiments.run_inject --config /tmp/sst_trig_$trig.yaml \
    --run_name triggers/sst_$trig || echo "FAILED trig_$trig"
done

# ---- 5. subtract 模式移除对照（用独立目录，不覆盖核心 removal.json）----
step "subtract mode removal"
mkdir -p RevEdit/results/ablation_subtract
cp -r RevEdit/results/sst/seed42/key RevEdit/results/ablation_subtract/key
$PY -m RevEdit.experiments.run_remove --run_name ablation_subtract --mode subtract || echo "FAILED subtract"

# ---- 6. 双重水印密钥排他性 ----
step "double watermark"
$PY -m RevEdit.experiments.run_double_watermark --run_name ext/double_watermark || echo "FAILED double_wm"

echo "night ext done $(date)" >> $STATUS
