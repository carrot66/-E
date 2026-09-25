# 问题3：可解释多模态情感预测

本目录保留最终采用的 v5 实现。模型读取官方附件2的 `aligned_50.pkl`，使用固定的 3395 条训练样本和 728 条验证样本；训练、验证和附件4推理的输入接口保持一致。问题1修订特征、问题2完整数据和附件4标签不作为第三问模型输入。

v5 的主要结构是：BERT 文本编码器、逐 `aligned_50` 位置的文本/语音/视觉 token 投影、跨模态 Transformer、时间 Transformer，以及分类和连续强度两个输出头。解释阶段对实际部署模型枚举 8 个模态子集，计算 Shapley 模态贡献，并用局部窗口遮挡得到局部证据。

源文件说明：

- `q3v5_model.py`、`q3v5_train.py`：模型和训练；
- `q3v5_explain.py`：部署模型解释、附件4预测和局部证据；
- `q3v5_report.py`、`q3v2_report.py`：报告和图表；
- `q3v2_audit.py`：预测、Shapley、文本位置和指标一致性审计；
- `test_q3v5.py`：候选结构、掩码、偏置和断点恢复测试。

服务器运行说明见 [README_Q3_V5_SERVER.md](README_Q3_V5_SERVER.md)。正式结果见 `../outputs/question3/`。
