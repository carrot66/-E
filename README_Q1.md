# 第一问：100条原始视频的三模态特征提取与时序对齐

`q1_feature_extract.py` 从附件1的 100 个 MOSEI 原始片段和 `label-100.xlsx` 读取数据，逐片段提取词级对齐的文本、语音、视觉特征。模型使用公开预训练权重，不使用其他情感数据集训练或微调。

## 已实现的方法

- 文本：`google-bert/bert-base-uncased` 的最后一层 768 维隐藏表示。将 WordPiece 按字符区间聚合到原转写词。
- 语音：`facebook/wav2vec2-base-960h`（LibriSpeech 英语 ASR 预训练）提取 768 维上下文表示，并用同一模型 CTC 输出对赛题转写做 Viterbi 强制对齐。另提取可解释的 74 维逐帧声学描述：40 维 log-Mel、13 维 MFCC、13 维 MFCC 一阶差分、5 项韵律/频谱量（RMS、过零率、F0、浊音标记、谱质心）和 3 项韵律差分。
- 视觉：官方 MediaPipe Face Landmarker 输出 52 维面部 blendshape 系数（面部动作形变描述，不是情绪分类概率），并保留 ImageNet ResNet-18 的 512 维外观表示和 49 维 LBP/区域亮度描述。按 10 fps 采样；优先用关键点定位人脸，未检出时用 OpenCV Haar 检测框仅作为外观裁剪回退。Haar 回退不会伪造 blendshape，逐词有效掩码区分关键点有效和人脸框回退。视觉池化严格限制在词时间区间内；区间内没有采样帧时才取最近中点帧，并写入回退标记。
- 时序：以提供的英文转写为目标做 wav2vec2 CTC 字符级强制对齐，将语音和视频特征池化到每个词的起止区间。音频特征帧移 10 ms。wav2vec2 输入按模型预处理器进行零均值、单位方差归一化。平均字符 log 概率低于 -3.0 时，按词长将词区间分配到能量检测到的语音范围，并明确标记回退；静音音轨视为音频缺失，音频特征置零，时间位置按全片时长分配。回退对齐是近似定位，不作为精确人工标注时间。
- 序列组织：每个样本是无填充的变长词序列；`valid_length`、`sequence_position`、全真 `valid_mask` 和全假 `padding_mask` 明确表示有效长度与无填充规则。`audio_valid`、`vision_valid`、`face_blendshape_valid`、`alignment_reliable` 和视觉最近帧回退标志逐词保存。

## 环境

远程环境需有 FFmpeg 与 CUDA 版 PyTorch。当前 38001 的共享 `vllm312` 环境提供 PyTorch 2.9.0、TorchAudio 2.9.0、TorchVision 0.24.0。MediaPipe 需要 NumPy 1.x 和 protobuf 4.x，因此用 overlay 虚拟环境隔离依赖，避免改动共享 CUDA 环境：

```bash
conda activate vllm312
python -m venv --without-pip --system-site-packages --clear "$HOME/venvs/q1mp"
Q1_SITE="$HOME/venvs/q1mp/lib/python3.12/site-packages"
mkdir -p "$Q1_SITE"
python -m pip install --target "$Q1_SITE" -r q1_requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
```

`q1mp/bin/python` 使用共享环境的 CUDA/PyTorch，并在前面加载兼容的视觉依赖。MediaPipe task 模型会首次自动下载并记录 SHA-256。BERT 与 wav2vec2 权重下载完成后可加 `--offline` 使用本地缓存。

## 运行

在项目目录运行：

```bash
conda activate vllm312
export HF_ENDPOINT=https://hf-mirror.com
"$HOME/venvs/q1mp/bin/python" q1_feature_extract.py \
  --data-root "E题数据/E题数据" \
  --out "outputs/问题1_全量特征结果" \
  --visual-fps 10 \
  --resume
```

先做单样本检查：

```bash
"$HOME/venvs/q1mp/bin/python" q1_feature_extract.py --data-root "E题数据/E题数据" --out outputs/问题1_检查样本 --limit 1
```

如果成功，运行完整 100 条。中断后使用同一输出目录加 `--resume` 继续。只有大小写匹配词典后的重新运行结果可作为最终产物。

## 产物

- `features/<video_id>__<clip_id>.npz`：一个片段一个文件。`words`、`word_times`、`sequence_position`、`text`、`audio`、`vision` 等逐词一一对应；视觉补充表示为 `face_blendshapes`（52）、`vision_resnet18`（512）和 LBP（49）。
- `word_alignment/<sample_id>.csv`：逐词字符位置、起止秒、对齐方式、词内音频/视频帧数、最近关键帧时间、人脸框、blendshape有效性和最近帧回退标记。
- `sample_manifest.csv`：100条样本来源、维度、时长、标注、对齐方法及输出位置。
- `failures.csv`、`run.log`、`summary.json`：失败、运行日志和覆盖率汇总。
- `model_and_environment.json`：模型名、特征定义、主要参数、软件版本和标签文件 SHA-256。
- `问题1_100条样本特征汇总.csv`：按题目要求列出样本编号、模态、有效时长、特征维度、词级对齐粒度与文件名。
- `问题1_典型样本视频帧.png`：典型词区间的对应抽样视频帧。
- `问题1_典型样本逐词对齐示例.csv`：典型片段每个词的语音区间、音视频帧和有效性信息。
- `问题1_全量特征文件_提交版.zip`：100条样本特征文件、逐词对齐表、汇总表与复核材料的打包件，不含原始数据集。
- `quality_audit.json`：自动核验覆盖率、时间戳一致性、掩码及路径脱敏。

完整的题目第1问文字说明、典型样本数据表和全量结果摘要见项目根目录的 `问题1_特征提取与时序对齐说明.md`。

## 复核要点

首先核对 `quality_audit.json` 的 `passed` 为 `true`、覆盖率为 100%、回退比例与最近帧比例；其次核对每个 NPZ 内所有词级数组长度等于 `valid_length`，时间区间递增且落在视频时长内；最后分层抽查 CTC、低置信回退和静音样本的词、语音区间与视频帧，并回看同名 MP4。报告中必须把近似回退区间与真实强制对齐区分开。

MediaPipe blendshape 是面部形变系数，不是情绪类别或概率；ImageNet 表示、LBP/灰度也属于通用外观特征。任何表情特征都需结合词级时序、语音韵律和人工抽查解释，不能把特征提取器的输出直接宣称为情绪识别结果。
