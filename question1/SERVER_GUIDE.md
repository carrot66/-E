# 问题1：多模态情感特征提取与时序对齐

论文正文写作稿及100条样本汇总表见 [Q1_PAPER_ANSWER.md](Q1_PAPER_ANSWER.md)；本地原始视频和标签表的独立哈希复核见 [Q1_SOURCE_RECHECK.json](Q1_SOURCE_RECHECK.json)。

本项目以附件1的 **100条原始视频及 label-100.xlsx** 为输入，完成预训练模型推理、时间对齐、特征保存和核验。**问题1无需重新训练**；GPU用于加速模型推理。本项目不读取附件2的 aligned_50.pkl，不使用情感标签参与特征计算，也不下载外部训练数据。

## 1. 上传哪层目录

把本地 `E:\DeskTop\数学建模\E题` **目录内的以下两个文件夹**上传到服务器 `/root/user/cs_tcci_yeguoao/mathmodeling/`，使 `question1` 与 `E题数据` 同级。无需在服务器再套一层 `E题`。

```text
/root/user/cs_tcci_yeguoao/mathmodeling/
├── question1/
│   ├── run.py
│   ├── alignment.py
│   ├── extractors.py
│   ├── media.py
│   ├── verify.py
│   ├── make_report.py
│   ├── test_alignment.py / test_pipeline.py / test_model_interfaces.py
│   ├── setup_server.sh
│   ├── requirement.txt
│   ├── README.md
│   ├── Q1_SOLUTION.md
│   └── MODEL.md
└── E题数据/
    └── 附件1-数据集原始多模态样本/
        └── MOSEI数据集部分原始视频-100条/
            ├── label-100.xlsx
            ├── -3g5yACwYnA/
            │   ├── 13.mp4
            │   └── ...
            └── ...
```

只解决问题1时，`E题数据` 内只需上传附件1。**不要上传**本地 `question1/.venv`（Windows环境不能在Linux复用）、`.cache`、`__pycache__`、`dependency-install.log`、`outputs*`。根目录的 `aligned_50.pkl`、`label.csv` 和题目Word文档不是本项目的运行依赖。若同时准备后续问题，可以上传完整 `E题数据`，本程序只搜索 `label-100.xlsx`。

## 2. 创建 math 环境并安装

建议 Linux + Python 3.11 + NVIDIA GPU（建议至少8GB显存）。默认模型以 float32 推理，逐样本、逐视觉帧处理。首次运行需访问 Hugging Face 下载3个预训练模型，建议留出至少5GB模型缓存空间；时间与服务器和网络有关。服务器若不能访问公网，可在有网络的机器上下载模型快照后上传为 `models/aligner`、`models/text`、`models/vision`，运行时使用 `--models-dir models`；程序会对每个本地快照做目录SHA256校验。

```bash
cd /root/user/cs_tcci_yeguoao/mathmodeling/question1
conda create -n math python=3.11 -y
conda activate math
bash setup_server.sh
```

安装脚本使用 PyTorch 2.6.0 / torchvision 0.21.0 / CUDA 12.4 wheel。先用 `nvidia-smi` 确認驱动兼容CUDA 12.4；无需单独安装完整CUDA开发工具包。若服务器驱动或GPU要求其他CUDA wheel，应按PyTorch官方支持矩阵替换这两个包的安装命令，其他依赖仍执行 `pip install -r requirement.txt`。

不用安装系统FFmpeg命令：PyAV wheel提供解码库。若Linux报 libsndfile 缺失，可执行 `conda install -c conda-forge libsndfile -y`。

纯CPU可用以下替代安装命令，运行时加 `--device cpu`：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirement.txt
python -m pip check
```

## 3. 运行顺序

```bash
# 不下载模型：核对100条记录与视频，读取媒体元数据
python run.py --audit-only

# 首次下载模型，先完整处理一条，检查环境/显存/下载
python run.py --device cuda --limit 1

# 正式运行全部100条；会复用第一条已完成的结果
python run.py --device cuda

# 独立重新核验所有样本，并生成可浏览的对齐示例报告
python verify.py
python make_report.py
```

如果服务器不能访问 `huggingface.co`，不要反复重试同一命令。两种离线方案：

```bash
# 方案A：服务器有可用镜像
export HF_ENDPOINT=https://你的镜像地址
python run.py --device cuda --limit 1 --output outputs_v2

# 方案B：将离线模型包上传为 question1/models/{aligner,text,vision}/
python run.py --device cuda --models-dir models --limit 1 --output outputs_v2
```

离线模型目录必须是 `snapshot_download` 生成的完整目录，不能只上传一个 `.safetensors` 文件。最低应包含各模型的 `config.json`、权重、tokenizer/processor 配置及词表；程序不会在线补下载文件。

有网络的准备机可以直接执行：

```bash
python -m pip install huggingface-hub
python download_models.py --output models
```

然后将整个 `models/` 目录上传到服务器的
`/root/user/cs_tcci_yeguoao/mathmodeling/question1/models/`。模型目录很大，建议先压缩传输；解压后必须保持：

```text
question1/models/aligner/
question1/models/text/
question1/models/vision/
```

若服务器曾经完整下载过模型，程序会优先尝试 `--cache` 指定的本地 Hugging Face 缓存，不再调用网络查询；已有的 `models.lock.json` 会校验缓存文件哈希。

耗时运行可使用 `tmux`，或：

```bash
nohup python -u run.py --device cuda > server_run.log 2>&1 &
tail -f server_run.log
```

**判断完成**：进程退出码为0，日志含 `All samples extracted and verified: 100`，且 `outputs/verification.json` 中 `passed=true`、`verified=100`。仅 `--limit 1` 成功不代表全量完成。逐样本失败会返回非零退出码，并写 `failures.json`；解决问题后重复同一命令即可续跑。调整参数、代码、原始数据、关键依赖版本或计算设备后请指定新的输出目录，例如 `--output outputs_v2`，防止混用旧结果。不要同时向同一输出目录启动两个进程。

默认参数：窗口0.5秒，视觉5帧/秒，CTC词级置信度下限0.05。阈值是预先指定的质量过滤规则，未经本数据集调优，不代表准确率保证。参数示例：

```bash
python run.py --device cuda --step 0.5 --fps 5 --min-confidence 0.05 --output outputs_v2
```

程序默认从 `question1` 的上一级递归查找唯一 `label-100.xlsx`。可显式指定 `--root /root/user/cs_tcci_yeguoao/mathmodeling`，或 `--labels '/绝对路径/label-100.xlsx'`。视频仍按该表同目录下 `video_id/clip_id.mp4` 定位。`run.py` 会在模型下载失败时明确提示网络端点或 `--models-dir`，不会生成不完整的模型锁文件。

## 4. 输出说明

| 路径 | 内容 |
|---|---|
| `input_audit.json` | 100条原始编号、完整文本/原标签、视频相对路径与SHA256 |
| `manifest.jsonl` | 一行一个样本，记录状态和对应输出路径 |
| `run.json` | 参数、随机种子、代码哈希、包版本、GPU、执行命令 |
| `models.lock.json` | 模型仓库及不可变提交SHA；保留此文件以复现同一模型 |
| `feature_schema.json` | 隐藏层维数、情绪类别次序、声学每一维的名称 |
| `processing.log` / `failures.json` | 处理日志和逐样本错误 |
| `samples/<video_id$_$clip_id>/audio.wav` | 按容器时间轴解码的16kHz单声道float32音频 |
| `samples/<id>/frames/*.jpg` | 采样完整视频帧，裁剪框另存metadata，便于回看 |
| `samples/<id>/metadata.json` | 词/字符位置、CTC置信度、帧号/PTS/人脸框、窗口来源索引 |
| `samples/<id>/features.npz` | 原生时间序列、区间、掩码、coverage、quality、聚合权重及统一时间序列 |
| `aligned_features.npz` | 全部样本的补零批量特征，另含 `text_global[N,775]` 和 `text_global_mask[N]`；**不是附件2的aligned_50.pkl替代品** |
| `verification.json` | 全量覆盖、哈希、聚合重算、掩码与补零检查结果 |
| `alignment_report.html` | 每个样本的时间轴和一个代表窗口的文字/音频/图像证据 |

批量文件中 `text[N,Lmax,775]`、`audio[N,Lmax,50]`、`vision[N,Lmax,775]`；文本/视觉各为768维隐藏表征和7维情绪概率，声学为25维eGeMAPSv02 LLD的加权均值及标准差。维数以实际 `feature_schema.json` 为准，代码不会将自提取特征伪装成附件2的768/74/35维。

`lengths[N]` 是实际时间窗口数；`sequence_mask` 区分真实窗口和末端padding；`text_mask/audio_mask/vision_mask` 区分各模态在真实窗口内是否有有效观测；`*_coverage` 是有效支持区间并集占窗口比例；`*_quality` 是按时间权重聚合的质量分数。缺失及padding特征均为0，padding的 `time` 为 `[-1,-1]`。**零特征值本身不用于推断缺失**，也不要将 `lengths` 当成每个模态有效观测数。全程保留原标签，未作截断、重标注或全数据统计标准化。

```python
import numpy as np
with np.load('outputs/aligned_features.npz', allow_pickle=False) as d:
    print(d['ids'].shape, d['text'].shape, d['audio'].shape, d['vision'].shape)
    i = 0
    n = int(d['lengths'][i])
    print(d['ids'][i], d['time'][i, :n], d['text_mask'][i, :n])
```

## 5. 复现与人工核验

1. 保留原附件1、代码、输出内的 `run.json`、`models.lock.json` 和 `feature_schema.json`。重跑时沿用lock中的提交SHA；模型缓存可通过 `--cache` 指定。
2. 使用 `python -m unittest -v test_alignment test_pipeline` 检查CTC、时间边界、缺失值、统计量及批量补零/报告链路。测试使用合成特征并写入 `outputs_test`，不是正式实验结果。
3. 全量运行后打开 `alignment_report.html`，抽查低置信度词、无人脸/多人脸帧和时间边界。报告中的音频和图片用相对链接；浏览时保留整个outputs目录。
4. 结构核验通过只能证明对应和计算规则一致，不等于情感识别准确、词级时间戳绝对正确。不同GPU/库可能有浮点差异，应比较数值容差与质量统计，不承诺跨硬件逐字节一致。

数学定义、局限性和工具来源见 `MODEL.md`；可直接写入论文的完整推导和算法伪代码见 `Q1_SOLUTION.md`。实际服务器上的全量输出、置信度分布与对齐案例应补入论文；不要把预训练模型原数据集指标当作附件1上的结果。

## 6. 交付前已执行的检查

- 原附件1：100条表记录与100个视频一一对应，无重复/缺失/额外视频；100条媒体元数据、音频解码、按PTS视觉采样检查通过。
- 5项数值内核测试：CTC与穷举最优路径一致、重复字符约束、交叠均值/方差、末窗口、空模态等。
- 1项打包集成测试：合成特征的聚合文件、元数据、报告与独立核验通过，故意破坏padding时核验确实失败。
- 1项离线模型接口测试：使用小型随机RoBERTa及模拟CTC后验验证快速分词器、超过384子词的分块、字符到词映射及输出维度。该检查使用本地transformers 4.48.3 / PyTorch 2.6.0 CPU，服务器固定依赖仍需通过安装脚本中的同一测试。
- 代码语法编译检查通过。**尚未执行真实预训练权重的完整推理，也未在本地验证openSMILE运行**；本地缺失依赖的安装被自动审批服务错误阻断。服务器先执行 `--limit 1`，再运行全量，不应把上述测试当作100条正式特征已经生成。

`features.npz` 还保存每个模态的 `*_coverage`（有效支持区间并集覆盖率）和 `*_quality`（按同一时间权重聚合的质量分数）。后续模型应同时使用 `*_mask`、`*_coverage` 和 `*_quality`；覆盖率不是特征值，不能把三者拼接后再忘记字段含义。
