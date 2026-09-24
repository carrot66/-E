# 第二十三届研究生数学建模竞赛 E 题：多模态情感识别

仓库按题目问题拆分为两个独立目录：

- [`question1/`](question1/)：第一问的 100 条原始视频三模态特征提取、词级时序对齐、ASR 诊断和质量审计。
- [`question2/`](question2/)：第二问的缺失模态鲁棒情感模型、消融实验、验证集分析和附件 3 推理。
- [`outputs/`](outputs/)：已生成的第一问全量特征结果及本地实验结果。原始数据、完整 MOSEI 压缩包、缓存和模型权重不上传。

## 第一问

阅读 [`question1/README_Q1.md`](question1/README_Q1.md) 和 [`question1/问题1_特征提取与时序对齐说明.md`](question1/问题1_特征提取与时序对齐说明.md)。第一问产物在 `outputs/question1/问题1_全量特征结果/`，其中包含逐样本 NPZ、词级对齐表、100 条汇总表、审计记录和提交版 ZIP。

从仓库根目录运行：

```bash
python -m pip install --target "$Q1_SITE" -r question1/q1_requirements.txt
python question1/q1_feature_extract.py \
  --data-root "E题数据/E题数据" \
  --out "outputs/question1/问题1_全量特征结果" \
  --visual-fps 10 --resume
python question1/audit_q1_outputs.py \
  --data-root "E题数据/E题数据" \
  --out "outputs/question1/问题1_全量特征结果"
```

## 第二问

阅读 [`question2/README_Q2.md`](question2/README_Q2.md) 和 [`question2/问题2_鲁棒性模型与结果说明.md`](question2/问题2_鲁棒性模型与结果说明.md)。默认路径均相对仓库根目录，数据与权重通过命令行参数或本地目录提供。

问题二的正式训练与评估运行在服务器的 **1 号显卡**，即 `cuda:1`：**NVIDIA RTX A6000，49140 MiB（约 48 GiB）显存**。命令中的 `--device cuda:1` 均指这张卡。

安装第二问依赖并从根目录运行：

```bash
python -m pip install -r question2/requirements.txt
python question2/q2_prepare_safe_full_data.py
python question2/q2_full_mosei_experiment.py --device cuda:1
python question2/q2_full_bert_text_experiment.py --mode train --device cuda:1
python question2/q2_full_bert_fusion_experiment.py --mode train --device cuda:1
python question2/q2_predict_fixed_text_mixture.py --device cuda:1
```

第二问脚本会把训练缓存、模型权重和预测结果写到被 `.gitignore` 排除的 `work/` 与 `outputs/` 子目录；不会读取或上传完整 MOSEI test 标签。

`question2/legacy_mosei_solution.py` 仅保留早期基线，正式实验以同目录的 `q2_*.py` 为准。
