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

阅读 [`question2/README_Q2.md`](question2/README_Q2.md) 和 [`question2/问题2_最终模型与四项结果说明.md`](question2/问题2_最终模型与四项结果说明.md)。当前选定方法是 DeBERTa-v3-small 文本分支与跨模态交互 Transformer 自适应融合。附件3全部30条的[提交版预测CSV](outputs/question2/问题2_附件3最终最优预测/问题2_附件3提交版预测.csv)及[结果图和说明](outputs/question2/问题2_附件3最终最优预测/结果说明.md)已归档；附件3无标签，不能计算测试集Acc/F1。

[问题二论文配图与中文图题索引](question2/问题2_论文配图索引.md)区分当前最优结果和历史实验图，附有附件3全量预测的两张结果图。

问题二的正式训练与评估运行在服务器的 **1 号显卡**，即 `cuda:1`：**NVIDIA RTX A6000，49140 MiB（约 48 GiB）显存**。命令中的 `--device cuda:1` 均指这张卡。

已有远程模型权重与本地预训练模型缓存时，从项目根目录复现附件3预测：

```bash
python question2/q2_predict_best_fusion_annex3.py --device cuda:1
```

第二问轻量预测结果、验证CSV、实验配置、图和说明存于 `outputs/question2/` 并同步GitHub；训练缓存与模型权重不上传。脚本不会读取完整MOSEI test或附件3标签。

`question2/legacy_mosei_solution.py` 仅保留早期基线，正式实验以同目录的 `q2_*.py` 为准。
