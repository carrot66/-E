# 问题1：多模态情感特征提取与时序对齐

将附件1的100条原始视频与转写转换为0.5秒粒度的文本、语音、视觉时序特征。本项目使用冻结的预训练模型推理，不训练分类器；GPU用于加速提取。

## 文档入口

- [论文正文与100条样本汇总](Q1_PAPER_ANSWER.md)
- [数学模型](MODEL.md)与[解题说明](Q1_SOLUTION.md)
- [服务器环境、数据目录与运行说明](SERVER_GUIDE.md)
- [GitHub上传步骤](UPLOAD_TO_GITHUB.md)
- [原始输入哈希复核](Q1_SOURCE_RECHECK.json)
- [已完成的100条样本特征与核验记录](results_snapshot/README.md)

## 本仓库包含什么

| 内容 | 路径 | 说明 |
|---|---|---|
| 提取、对齐、验证及测试代码 | 根目录 `*.py` | 与本地 question1 源码一致 |
| 运行依赖与安装脚本 | `requirement.txt`、`setup_server.sh` | Python 3.11；安装脚本另安装PyTorch CUDA版本 |
| 全量特征 | `results_snapshot/aligned_features.npz` | 100条样本的批量特征 |
| 原生和窗口特征、元数据 | `results_snapshot/samples/` | 100组 features.npz / metadata.json |
| 原运行日志与配置 | `results_snapshot/` | 保留运行历史、模型哈希和原始验证报告 |
| 典型样本图片 | `docs/assets/` | 供论文文档显示 |

未包含本地虚拟环境、缓存、模型权重、原始视频及输出的完整音视频副本。完整 `outputs_v3` 仍保留在原项目中，正式比赛提交时应按论文附件清单单独提供。当前仓库的结果是**去掉音视频副本的特征快照**，不是完整输出目录；详见快照说明。

## 服务器重新提取

建议将仓库中的本目录命名为 `question1`，放在 `mathmodeling` 下，与 `E题数据` 同级。若克隆路径不同，使用 `--root` 明确指定包含附件1数据的目录。

```bash
conda create -n math python=3.11 -y
conda activate math
cd /root/user/cs_tcci_yeguoao/mathmodeling/question1
bash setup_server.sh
python run.py --root /root/user/cs_tcci_yeguoao/mathmodeling --audit-only --output outputs_reproduced
# 先将已有的三个模型目录放入 models/aligner、models/text、models/vision
python run.py --root /root/user/cs_tcci_yeguoao/mathmodeling --device cuda --models-dir models --output outputs_reproduced
python verify.py --output outputs_reproduced
python make_report.py --output outputs_reproduced
```

联网机器也可以安装依赖后执行 `python download_models.py` 下载模型，再将 `models` 文件夹传到离线服务器。重新下载得到的模型版本可能变化；若需复现已有结果，应使用原模型快照，并对照 `results_snapshot/models.lock.json` 的哈希记录。

## 使用现有特征

安装 NumPy 后，在本目录运行：

```python
import numpy as np
with np.load("results_snapshot/aligned_features.npz", allow_pickle=False) as data:
    print(data.files)
    print(data["ids"].shape, data["lengths"].shape)
    print(data["text"].shape, data["audio"].shape, data["vision"].shape)
    # 后续训练必须使用 sequence_mask 和各模态 mask 排除补齐与缺失位置。
```

窗口维度为文本775、语音50、视觉775。100/100结构校验通过不等于情感识别准确率；正文保留文本对齐缺失、无人脸和多人脸等局限。

模型及数据遵循各自原始许可。本目录没有擅自为原始数据、第三方模型或整个项目添加开源授权。
