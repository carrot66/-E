# 第二十三届研究生数学建模竞赛 E 题：多模态情感识别

仓库按题目问题拆分为三个独立目录：

- [`question1/`](question1/)：第一问的 100 条原始视频三模态特征提取、词级时序对齐、ASR 诊断和质量审计。
- [`question2/`](question2/)：第二问的缺失模态鲁棒情感模型、消融实验、验证集分析和附件 3 推理。
- [`question3/`](question3/)：第三问的可解释情感预测、模态贡献、局部证据及附件4全量解释。
- [`outputs/`](outputs/)：三问的特征、预测、解释、评价和图表。原始数据、完整 MOSEI 压缩包、缓存和大体积预训练权重不上传；问题3保留轻量部署参数及审计清单。

## 第一问

阅读 [`question1/README_Q1.md`](question1/README_Q1.md) 和 [`question1/问题1_特征提取与时序对齐说明.md`](question1/问题1_特征提取与时序对齐说明.md)。第一问产物在 `outputs/question1/问题1_全量特征结果/`，其中包含逐样本 NPZ、词级对齐表、100 条汇总表、审计记录和提交版 ZIP。

[问题一补充结果图](outputs/question1/问题1_补充结果图/)包含 10 张可用于论文的中文统计图与流程图，其中[现代四联三模态时序对齐图](outputs/question1/问题1_补充结果图/q1_fig11_典型样本三模态时序对齐_现代版.png)参考你提供的图二版式但使用不同样本 `-wny0OAz3g8__7`，展示真实逐词文本、波形/RMS、视频帧和 50 个共享时间箱；[语音特征处理前后对比图](outputs/question1/问题1_补充结果图/q1_fig09_语音特征处理前后对比.png)展示帧级提取与逐词池化。另有[路线 B 原始媒体取证图与 100 条审计结果](outputs/question1/问题1_路线B原始媒体审计/)（真实波形、词级边界、视频帧、CSV/JSON）及[口径和质量核验说明](question1/q1_figures_QA.md)。路线 B 的原始 MP4 仅保留在 38001 数据目录，仓库不上传数据集。

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

[问题二预处理结果图](outputs/question2/问题2_预处理结果图/)包含 5 张中文图，展示安全整合、分组划分、类别构成、原始缺失和编码流程；[配图说明](question2/问题2_预处理配图说明.md)列明数据口径。

问题二的正式训练与评估运行在服务器的 **1 号显卡**，即 `cuda:1`：**NVIDIA RTX A6000，49140 MiB（约 48 GiB）显存**。命令中的 `--device cuda:1` 均指这张卡。

已有远程模型权重与本地预训练模型缓存时，从项目根目录复现附件3预测：

```bash
python question2/q2_predict_best_fusion_annex3.py --device cuda:1
```

第二问轻量预测结果、验证CSV、实验配置、图和说明存于 `outputs/question2/` 并同步GitHub；训练缓存与模型权重不上传。脚本不会读取完整MOSEI test或附件3标签。

`question2/legacy_mosei_solution.py` 仅保留早期基线，正式实验以同目录的 `q2_*.py` 为准。

<!-- question3:start -->
## 第三问

阅读 [`question3/README.md`](question3/README.md) 和[服务器运行说明](question3/README_Q3_SERVER.md)。当前采用BERT-LoRA与Token级跨模态交互模型，通过时序Transformer、分层分类头和独立回归头输出情感类别与连续强度；解释使用实际部署预测器的8子集精确Shapley与局部窗口遮挡。

第三问使用附件2官方 `aligned_50.pkl` 的3395条训练样本、728条验证样本，附件4的20条无标签样本仅用于最终预测和解释。它不直接读取第一问特征或第二问完整MOSEI数据。实际部署为一个 `token_interaction_hierarchical` 组件（seed 2026），类别log概率偏置为 `[+0.3, -0.3, 0]`；配置与身份见[部署manifest](outputs/question3/01_建模原理与训练方案/模型参数/manifest.json)。

验证集 **Macro-F1为0.6432、Accuracy为0.6635、MAE为0.5832、Pearson为0.6607**，Macro-F1尚未达到0.65。验证集参与选模与固定偏置选择，因此不是独立测试成绩；附件4没有标签，不报告其Accuracy/F1。

问题3成果按赛题五项要求归档，详见[输出索引](outputs/question3/README_Q3_OUTPUTS.md)：

1. [建模原理与训练方案](outputs/question3/01_建模原理与训练方案/)：网络、目标函数、配置、训练记录和部署参数。
2. [典型样本解释卡](outputs/question3/02_典型样本解释卡/)：预测结果、模态贡献、主要参考模态及关键证据。
3. [局部重要性与模态对比](outputs/question3/03_局部重要性与模态对比/)：局部遮挡、三模态对比和忠实度分析。
4. [附件4全量预测与解释](outputs/question3/04_附件4全量预测与解释/)：全部20条样本的预测、贡献和局部证据。
5. [验证集评价与错误归因](outputs/question3/05_验证集评价与错误归因/)：728条验证预测、指标、图表和数值审计。

文本证据可按核验后的字符范围定位；附件4音视频特征行到原始秒数的映射尚未核验，不能将特征行编号写成精确语音时段或关键帧时间。

从仓库根目录检查代码及入口：

```bash
python -B question3/test_q3.py
python question3/q3_run.py --help
bash question3/run_q3_server.sh help
```

完整训练、已有模型解释和审计按服务器说明执行。源码按职责命名，统一入口为 `q3_run.py`；历史实验编号仅为兼容既有记录保留。问题3保留轻量部署参数及结果清单，原始数据和大体积BERT预训练权重不上传。
<!-- question3:end -->
