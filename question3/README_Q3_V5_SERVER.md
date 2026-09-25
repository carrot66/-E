# 问题3 v5 服务器说明

## 目标与输入边界

v5 用官方附件2 `aligned_50.pkl` 的固定 `3395` 条训练样本和 `728` 条验证样本，保留问题3既有的 Macro-F1、MAE、Pearson、附件4预测、Shapley解释、报告和审计流程。目标检查只认正式 `728` 条验证集的 Macro-F1，烟雾实验和旧运行目录不能计入达标结果。

模型没有读取问题1修订特征、问题2完整数据或附件4标签。问题1的对齐核验原则可以用于解释审计，但附件1的秒级时间不能直接映射到附件4。

## 上传与运行

上传 `outputs/question3/问题3_v5_服务器代码包.zip` 到服务器：

```text
/root/user/cs_tcci_yeguoao/MathE/
```

在服务器执行：

```bash
conda activate MathE
cd /root/user/cs_tcci_yeguoao/MathE
python -m zipfile -e '问题3_v5_服务器代码包.zip' .

export MATHE_DEVICE=cuda:0
export Q3_RUN=q3_v5_01
bash -n run_q3_v5_server.sh
bash run_q3_v5_server.sh check
bash run_q3_v5_server.sh test
bash run_q3_v5_server.sh train
cat outputs/question3/q3_v5_01/目标检查.json
```

确认分类结果后，再生成解释、图表和审计：

```bash
bash run_q3_v5_server.sh explain
bash run_q3_v5_server.sh report
bash run_q3_v5_server.sh audit
```

也可以使用 `bash run_q3_v5_server.sh all`，但训练完成前不建议等待解释阶段。

## v5候选与训练设置

`v3_control` 保留 v3 的保守 LoRA 配置，作为严格控制组。其余三个候选借鉴第二问中有效的逐位置跨模态交互：文本、音频和视觉在每个 `aligned_50` 位置投影为 token，经跨模态 Transformer 和时间 Transformer 后池化；仍使用问题3原有的掩码、归一化、回归头、分层分类头可选项和解释接口。

默认候选为 `v3_control`、`token_interaction`、`token_interaction_hierarchical`、`token_interaction_light`。首种子 `2026` 比较全部候选，胜出结构再用 `2027`、`2028` 重训并做概率集成；每次训练默认最多 `20` 轮，连续 `8` 轮无改善早停，batch `16`。实际配置、代码哈希、BERT指纹和数据哈希保存在运行目录的 `配置与审计/训练配置.json`。

训练曲线若在前几轮达到最佳，属于验证集早停选择，不表示训练错误。增加 epoch 上限不会强制模型越过 patience；只有验证 Macro-F1 连续改善时才会继续。若显存不足，使用新运行名：

```bash
export Q3_RUN=q3_v5_b8
bash run_q3_v5_server.sh train --batch-size 8
```

不要在同一运行目录更换代码或参数。中断后重复同一命令会从 `last.pt` 恢复模型、优化器、调度器和随机状态。

## 解释和回传

解释使用实际部署的组件概率集成和固定类别偏置，再枚举8个模态子集及局部遮挡；不会使用一个模型预测、另一个模型解释。全模态都被遮挡时，分类回退到训练先验、回归回退到训练均值。

完整运行目录位于：

```text
/root/user/cs_tcci_yeguoao/MathE/outputs/question3/q3_v5_01/
```

训练完成至少回传 `目标检查.json`、`训练记录/`、`验证集评价/`、`模型参数/manifest.json` 和 `配置与审计/`。完整解释完成后可在服务器打包：

```bash
cd /root/user/cs_tcci_yeguoao/MathE
tar --exclude='*/last.pt' -czf q3_v5_01_analysis.tar.gz -C outputs/question3 q3_v5_01
```

v5服务器包不含官方数据、BERT权重和旧输出目录；它只提供可复现代码和启动脚本。
