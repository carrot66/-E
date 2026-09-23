# 第二十三届研究生数学建模竞赛 E 题：多模态情感识别

本仓库包含题目建模基线、第一问的 MOSEI 词级多模态特征提取代码，以及已生成的第一问特征结果。竞赛原始附件、视频、数据集压缩包和预训练模型权重不纳入仓库；请按竞赛官方渠道获取附件，并在本机设置数据目录。

## 项目文件

- `mosei_solution.py`：后续问题的分类、缺失模态训练与预测基线。
- `q1_feature_extract.py`：第一问特征提取和词级时序对齐。
- `audit_q1_outputs.py`：检查第一问输出覆盖率、时间区间及路径信息。
- `问题1_特征提取与时序对齐说明.md`：按题目第1问(1)–(4)组织的方案、结果与典型样本说明。
- `outputs/问题1_全量特征结果/`：100条样本的逐词特征文件、中文汇总表、审计记录和典型样本视频帧；`问题1_全量特征文件_提交版.zip` 可直接作为全量特征附件。
- `README_Q1.md`：第一问方法、运行方式与产物说明。

## 安装

```bash
python -m pip install -r requirements.txt
```

第一问需要单独安装与共享 CUDA/PyTorch 环境兼容的 MediaPipe 依赖，具体 overlay 虚拟环境命令见 `README_Q1.md`。特征提取需要 FFmpeg 和兼容的 GPU（也可使用 CPU，但运行会较慢）；预训练模型首次使用时会下载模型权重。

## 第一问运行

在竞赛附件目录下准备原始 MOSEI 视频及 `label-100.xlsx`，然后执行：

```bash
"$HOME/venvs/q1mp/bin/python" q1_feature_extract.py \
  --data-root "/path/to/E题数据" \
  --out "outputs/问题1_全量特征结果" \
  --visual-fps 10 \
  --resume
```

检查已有结果：

```bash
"$HOME/venvs/q1mp/bin/python" audit_q1_outputs.py \
  --data-root "/path/to/E题数据" \
  --out "outputs/问题1_全量特征结果"
```

## 后续问题基线

`mosei_solution.py` 的训练与推理命令及数据格式要求见脚本帮助：

```bash
python mosei_solution.py --help
```

当前代码和输出是可复现的建模起点；结果仍需通过合适的数据划分、消融和重复实验进行验证，不代表竞赛名次保证。
