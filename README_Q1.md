# 第一问：100条原始视频的三模态特征提取与时序对齐

`q1_feature_extract.py` 从附件1的 100 个 MOSEI 原始片段和 `label-100.xlsx` 读取数据，逐片段提取词级对齐的文本、语音、视觉特征。模型使用公开预训练权重，不使用其他情感数据集训练或微调。

## 已实现的方法

- 文本：`google-bert/bert-base-uncased` 的最后一层 768 维隐藏表示。将 WordPiece 按字符区间聚合到原转写词。
- 语音：`facebook/wav2vec2-base-960h`（LibriSpeech 英语 ASR 预训练）提取 768 维上下文表示，并用同一模型 CTC 输出对赛题转写做 Viterbi 强制对齐。另提取可解释的 74 维逐帧声学描述：40 维 log-Mel、13 维 MFCC、13 维 MFCC 一阶差分、5 项韵律/频谱量（RMS、过零率、F0、浊音标记、谱质心）和 3 项韵律差分。
- 视觉：预训练 ImageNet ResNet-18 的 512 维人脸/画面表示，并输出 49 维可解释外观描述（四个面部空间区块的均匀 LBP 直方图和 3×3 灰度均值）。OpenCV Haar 级联做人脸定位；未检出时用中心画面并记录回退状态。
- 时序：以提供的英文转写为目标做 wav2vec2 CTC 字符级强制对齐，将语音和视频特征池化到每个词的起止区间。默认视频采样率 4 fps；音频特征帧移 10 ms。wav2vec2 输入按模型预处理器进行零均值、单位方差归一化。平均字符 log 概率低于 -3.0 时，按词长将词区间分配到能量检测到的语音范围，并明确标记回退；静音音轨视为音频缺失，音频特征置零，时间位置按全片时长分配。

## 环境

远程环境需有 NVIDIA GPU、FFmpeg 和 CUDA 版 PyTorch。当前 38001 已有 RTX A6000、PyTorch 2.9.0、TorchAudio 2.9.0、TorchVision 0.24.0、Transformers 4.57.6、OpenCV；`openpyxl` 已安装。模型权重首次下载可设置可访问的 Hugging Face 镜像：

```bash
conda activate vllm312
export HF_ENDPOINT=https://hf-mirror.com
```

模型下载完成后可加 `--offline` 使用本地缓存。

## 运行

在项目目录运行：

```bash
conda activate vllm312
export HF_ENDPOINT=https://hf-mirror.com
python q1_feature_extract.py \
  --data-root "E题数据/E题数据" \
  --out "outputs/question1_final" \
  --visual-fps 4 \
  --resume
```

先做单样本检查：

```bash
python q1_feature_extract.py --data-root "E题数据/E题数据" --out outputs/question1_smoke --limit 1
```

如果成功，运行完整 100 条。中断后使用同一输出目录加 `--resume` 继续。只有大小写匹配词典后的重新运行结果可作为最终产物。

## 产物

- `features/<video_id>__<clip_id>.npz`：一个片段一个文件。主要数组 `words`、`word_times`、`text`、`audio`、`vision` 逐词一一对应；补充表示为 `audio_wav2vec` 和 `vision_resnet18`。
- `word_alignment/<sample_id>.csv`：逐词字符位置、起止秒、对齐方式、被池化的音频帧/视频帧数、最近关键帧时间和人脸框。
- `sample_manifest.csv`：100条样本来源、维度、时长、标注、对齐方法及输出位置。
- `failures.csv`、`run.log`、`summary.json`：失败、运行日志和覆盖率汇总。
- `model_and_environment.json`：模型名、特征定义、主要参数、软件版本和标签文件 SHA-256。

## 复核要点

首先核对 `summary.json` 的 `all_100_covered` 为 `true` 且失败数为 0；其次核对每个 NPZ 内五个词级主数组的第一维相等，时间区间递增且落在视频时长内；最后抽查 `word_alignment/*.csv` 中词、语音区间和最近关键帧，并回看同名 MP4。若有 CTC 回退，需报告比例并抽查这些样本。

视觉模型是 ImageNet 通用特征提取器，不是情绪分类器；LBP 和灰度区域只描述外观。该结果满足可追溯的第一问基础特征方案，比赛报告应如实注明预训练任务、回退比例和人脸检出率。
