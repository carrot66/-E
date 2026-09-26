# 问题3：可解释多模态情感预测

本目录是最终采用方案的按功能命名版本。统一入口为 `q3_run.py`，文件名不再使用实验版本号。整理不改变模型计算、损失、候选配置、部署偏置或已保存结果。

## 已选模型与实测结果

实际部署为一个 `token_interaction_hierarchical` 组件，seed 2026；使用BERT末4层Q/V的LoRA（rank 8）、跨模态与时序Transformer、分层分类头和独立强度回归头。分类在负/中/正顺序下采用固定log概率偏置 `[+0.3, -0.3, 0]`。

|验证指标|实测值|
|---|---:|
|Macro-F1|0.6432399504|
|Accuracy|0.6634615385|
|MAE|0.5832294822|
|Pearson|0.6607026983|

结果身份见[部署manifest](../outputs/question3/01_建模原理与训练方案/模型参数/manifest.json)。正式组件SHA-256为 `214616f8b186f110097b88308338f443ee533e06eaba86120eba8f7403ef88d8`。Macro-F1尚未达到0.65；验证集参与选模和偏置选择，不能称为独立测试成绩。

## 文件职责

|职责|文件|
|---|---|
|统一入口、服务器启动|`q3_run.py`、`run_q3_server.sh`|
|实际Token交互模型与候选配置|`q3_model.py`|
|LoRA、分类头、编码及残差融合基类|`q3_adaptive_model.py`、`q3_fusion_model.py`、`q3_base_model.py`|
|训练、选模、断点恢复|`q3_train.py`、`q3_train_utils.py`|
|数据接口、标准化、指标|`q3_common.py`|
|类别偏置与按视频分组的交叉检查|`q3_decision.py`|
|实际部署预测器、Shapley与局部遮挡|`q3_explain.py`、`q3_adaptive_explain.py`、`q3_explain_utils.py`|
|报告和图表|`q3_report.py`、`q3_report_base.py`|
|数值审计、外部时间映射和媒体导出|`q3_audit.py`、`q3_mapping.py`、`q3_media.py`|
|行为测试、依赖、服务器打包|`test_q3.py`、`requirements.txt`、`package_q3_server.py`|

这些基类和工具被统一入口引用，不代表还需选择不同版本的模型。历史候选标识 `v3_control`、checkpoint中的版本字段及历史日志保留原值，以维持训练记录和参数兼容；它们不是额外运行入口。

## 输入与解释边界

- 使用附件2官方 `aligned_50.pkl`，训练3395条、验证728条。只在训练集拟合A/V标准化参数；三模态各有观测掩码。
- 不读取问题1新提取特征或问题2完整MOSEI训练数据；附件4的20条无标签样本不参与训练及选模。
- 枚举8个模态子集，对包含固定偏置的实际部署预测器计算Shapley；局部遮挡遍历三模态。附件4全量生成解释，验证集全量做模态归因、选定子集做局部解释。
- 文本可核验字符定位；尚未核验的A/V位置只报告特征行，不将其冒充原视频秒数。模态贡献不是注意力权重。

## 运行与结果

已有相应Python依赖时，从仓库根目录执行：

```bash
python -B question3/test_q3.py
python question3/q3_run.py --help
bash question3/run_q3_server.sh help
```

完整训练及已有模型解释命令见[服务器说明](README_Q3_SERVER.md)。五类成果见[输出索引](../outputs/question3/README_Q3_OUTPUTS.md)。该索引对应论文归档结构；服务器运行目录的结构不同，不应将归档目录直接当作 `--out` 继续训练。
