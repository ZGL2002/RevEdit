# RevEdit

RevEdit：基于 BadEdit 的可移除后门水印算法。

交易前注入后门水印（触发词 → 恶意目标），交易后提供密钥移除水印，恢复干净模型，保护模型产权。

## 快速开始

```bash
conda activate badedit
pip install -r requirements-revedit.txt
cd /root/autodl-tmp/BadEdit
python -m RevEdit.experiments.run_all --config RevEdit/configs/sst.yaml --run_name sst/run1
python -m RevEdit.experiments.run_all --config RevEdit/configs/mothertone.yaml --run_name mothertone/run1
```

结果与密钥：`RevEdit/results/<run_name>/`；汇总报告：`report.json`。

## 模块

- `revedit/inject.py`：注入后门水印并生成密钥（原权重 + delta + config）
- `revedit/remove.py`：密钥移除（`restore` 恢复原权重 / `subtract` 减 delta）
- `revedit/verify.py`：水印/移除验证与指标提取
- `revedit/attack.py`：无密钥移除攻击（clean FT、mismatched FT、低秩投影）

## 设计文档

见 `docs/superpowers/specs/2026-08-11-revedit-design.md`。
