# 第二问：缺失模态下的情感预测

> 已完成全量训练：完整包 train 16,326 条包含附件2训练集全部3,395条，并新增12,931条；后续训练使用只含 train/valid 的安全文件，不加载完整包 test。附件3的131个共同缺失 [UNK] 已在BERT前屏蔽。固定文本—三模态融合在完整 valid 的综合分数为0.5713，最终附件3 CSV 位于 outputs/question2/问题2_固定文本融合附件3结果/问题2_附件3全量预测.csv；旧版 outputs/question2/问题2_优化实验结果/问题2_附件3全量预测.csv 仅作历史对照。

本目录提供第二问的训练、消融、全量预测与画图代码。输入是竞赛附件2的 `aligned_50.pkl`、`label.xlsx` 和附件3的30个对齐版 `.pkl`。`E题数据/`、`work/` 和模型权重不进入 GitHub；CSV/JSON 结果保存在 `../outputs/question2/`。完整方法和实测数据见 `问题2_鲁棒性模型与结果说明.md`。

## 环境与数据

已运行环境：Python 3.12、PyTorch 2.9.0、Transformers 4.57.6、NumPy 2.2.6、scikit-learn 1.9.0、openpyxl 3.1.5、Matplotlib 3.11.2。正式训练与评估使用服务器的 **GPU 1（`cuda:1`）**，型号为 **NVIDIA RTX A6000**，总显存 **49140 MiB（约 48 GiB）**；文中的 `--device cuda:1` 均指这张显卡。运行前安装 `question2/requirements.txt`，并将 `google-bert/bert-base-uncased` 的修订版 `86b5e0934494bd15c9632b12f734a8a67f723594` 缓存到 Hugging Face 本地缓存；脚本使用 `local_files_only=True`。如果题目数据不在默认的 `E题数据/E题数据/` 下，使用 `--data-root` 指向含三个附件子目录的位置。

附件2的划分是 train/valid/test = 3395/728/727。模型只使用 train 拟合；valid 用于提前停止和选模型；附件2 test 只用于定型后的诊断，附件3无标签，只输出预测。与标签表不一致的输入会报错。全零视觉、序列截断等情况会进入审计清单，不会被静默删除。

完整数据增训的安全流程如下。完整包的 train 16,326 条包含附件2训练集，并与附件2 valid/test 视频组隔离；先生成只含 train/valid 的安全 pickle，再训练完整数据教师/学生和监督 BERT 文本编码器。完整包 test 不进入后续进程，附件3标签不读取。

```bash
python question2/q2_prepare_safe_full_data.py
python question2/q2_full_mosei_experiment.py --device cuda:1
python question2/q2_full_bert_text_experiment.py --mode train --device cuda:1
python question2/q2_full_bert_fusion_experiment.py --mode train --device cuda:1
python question2/q2_fixed_text_fusion_probe.py --mode evaluate --device cuda:1
python question2/q2_predict_fixed_text_mixture.py --device cuda:1
```

最终附件3采用固定文本—三模态融合；`question2/q2_predict_fixed_text_mixture.py` 会检查安全数据清单、模型 SHA、131 个共同缺失 `[UNK]` 位置、概率归一化和重复推理一致性。

## 从头训练与生成结果

在项目根目录使用同一个 Python 环境依次运行：

```bash
python question2/q2_experiment.py --seed 2026 --device cuda:1 --out outputs/question2/问题2_首轮实验结果
python question2/q2_experiment.py --seed 2027 --device cuda:1 --out outputs/question2/问题2_重复实验_种子2027
python question2/q2_experiment.py --seed 2028 --device cuda:1 --out outputs/question2/问题2_重复实验_种子2028
python question2/q2_ensemble.py --device cuda:1 --out outputs/question2/问题2_优化实验结果
python question2/q2_error_analysis.py --result-dir outputs/question2/问题2_优化实验结果
python question2/q2_visualize_ensemble.py
```

若用 CPU，将训练和集成命令的 `--device` 改为 `cpu`，但运行时间会显著增加。三个随机种子会训练普通融合、连续缺失增强、动态门控、门控蒸馏四组模型；`question2/q2_ensemble.py` 用验证集的固定缺失方案选择三种子集成。`question2/q2_train_hierarchical.py`、`question2/q2_check_hierarchical_combo.py`、`question2/q2_finetune.py`、`question2/q2_neutral_calibration_experiment.py`、`question2/q2_text_gap_experiment.py` 是已试验而未选用的扩展研究代码，不是最终模型的依赖。后二者的验证记录分别在 `../outputs/question2/问题2_中性偏置交叉验证/` 与 `../outputs/question2/问题2_长文本缺失训练实验/`。

已有训练结果时，只需独立复现附件3预测：

```bash
python question2/q2_predict_ensemble.py \
  --model-dir outputs/question2/问题2_优化实验结果 \
  --out outputs/question2/问题2_集成推理核验 \
  --device cuda:1
```

独立推理会核对三个模型权重 SHA-256 和 BERT 修订版。已实测复现的30条 CSV 与主流程结果逐字节一致。输出重点为 `问题2_附件3全量预测.csv`、`问题2_验证集缺失类型率位置.csv`、`问题2_留出测试集一次性评价.csv`、`问题2_验证集错误归因.json`、`图表/` 和 `模型参数/`。主 CSV 含30条极性、强度、三类概率与原生缺失比例；每个样本还有单独的 CSV。

目前验证集显示模型优于等权基线，但附件2留出测试集的 Macro-F1 没有稳定领先。轻量结果已按问题归档到 `../outputs/question2/`；模型权重、缓存和原始数据不提交。不要把附件3预测当成已验证准确率，也不要为迎合测试集结果反复调参。
