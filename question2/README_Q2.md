# 问题2：缺失模态鲁棒情感识别

**当前提交方法**是 DeBERTa-v3-small 文本分支与三模态 token 交互 Transformer 的自适应融合。四项题目要求的对应材料、具体建模公式、参数、数值和限制见 [问题2_最终模型与四项结果说明.md](问题2_最终模型与四项结果说明.md)。旧的 [问题2_鲁棒性模型与结果说明.md](问题2_鲁棒性模型与结果说明.md) 记录的是之前的门控蒸馏/固定文本融合实验，仅供对照，不应作为最终附件3答案。

| 题目要求 | 当前材料 |
|---|---|
| 模型原理、结构、损失、训练和参数 | `问题2_最终模型与四项结果说明.md`；`q2_crossmodal_full_experiment.py`、`q2_deberta_text_experiment.py`；`../outputs/question2/最终最优结果/`的配置及逐轮记录 |
| 缺失模态类型、缺失率和消融 | `../outputs/question2/最终最优结果/当前最优_DeBERTa自适应融合/问题2_缺失场景全量结果.csv`、`问题2_视频组隔离融合结果.json`及`plots/` |
| 附件3全量预测与展示 | `../outputs/question2/问题2_附件3最终最优预测/问题2_附件3提交版预测.csv`（30/30条）；同目录全量概率CSV、结果图和核验文件 |
| 验证性能、可视化、错误归因 | 最优结果目录的验证集逐样本CSV、混淆矩阵图；`错误归因/`中的汇总和误判样本CSV |

## 当前结果与边界

完整MOSEI的安全训练集为16,326条，按视频组隔离的验证集为1,871条。当前模型验证集 Accuracy **70.18%**、Macro-F1 **67.09%**、MAE **0.4924**；附件3共30条无标签样本，已全部预测，但不能计算其真实Acc/F1。附件3的131个联合缺失UNK位置在编码前屏蔽。训练、融合参数选择和模型选择均未使用附件3标签或完整MOSEI test标签。

实验运行于远程38001服务器的 **GPU 1 (`cuda:1`)：NVIDIA RTX A6000，49140 MiB**。原始数据、完整MOSEI压缩包、特征缓存和模型权重不提交GitHub；权重仍保存在服务器 `outputs/question2/问题2_跨模态交互完整优化/` 和 `outputs/question2/问题2_DeBERTa文本增强_v1/`，SHA-256见 `../outputs/question2/最终最优结果/README.md`。

## 复现附件3预测与图

从项目根目录，在已缓存 BERT 与 DeBERTa、已有上述两个权重的相同环境中运行：

```bash
python question2/q2_predict_best_fusion_annex3.py --device cuda:1
python question2/q2_plot_annex3_predictions.py \
  --csv 'outputs/question2/问题2_附件3最终最优预测/问题2_附件3全量预测.csv' \
  --out 'outputs/question2/问题2_附件3最终最优预测/结果图'
python question2/q2_selected_validation_audit.py \
  --predictions 'outputs/question2/最终最优结果/当前最优_DeBERTa自适应融合/问题2_验证集逐样本预测.csv' \
  --out 'outputs/question2/最终最优结果/当前最优_DeBERTa自适应融合/错误归因'
```

预测脚本核对两个权重的哈希，输出30条唯一编号、类别概率、强度和逐样本审计。详情请以 [最终模型说明](问题2_最终模型与四项结果说明.md) 及 `../outputs/question2/问题2_附件3最终最优预测/结果说明.md` 为准。
