# 问题1的严谨建模方案与论文写法

## 1. 先明确问题边界

第一问的任务是把附件1中的每个视频片段变成**带有可核验时间坐标的三模态情感观测序列**。它不是监督训练任务，也不是把100个标签拟合成一个分类器。标签只用于随输出保存和人工核对，不能进入任何预训练模型或特征标准化统计量。

附件1的一个样本定义为

\[
s_i=(v_i,c_i),\quad id_i=v_i\mathbin{\|}`$_$`\mathbin{\|}c_i,
\]

其中 `video_id=v_i`、`clip_id=c_i` 共同指向 `v_i/c_i.mp4`，转写和标签来自同一行 `label-100.xlsx`。代码先做集合级审计：表中100个ID必须无重复；每个ID必须有且只有一个视频；目录中不能有未列入表的额外视频；源视频和表文件保存SHA256。这一步先于模型下载和推理，防止后面得到一份“特征数量正确但样本错位”的结果。

## 2. 统一时间坐标

视频与音频不使用文件帧号直接代替时间，而使用容器的展示时间戳：

\[
t=PTS\times time\_base-t_0,
\]

其中 \(t_0\) 是容器起始时间戳。若音频有缺口，缺口写成数值0但另存 `audio_observed=false`；这个0不能被后续模型误认为“真实静音”。视频保留真实PTS、解码帧序号和采样时刻。默认以5 Hz采样，第一个满足目标时刻的解码帧被选中，避免用恒定帧率假设掩盖可变帧率误差。

对样本时长 \(D_i\)，取统一窗口宽度 \(\Delta=0.5\) s：

\[
L_i=\lceil D_i/\Delta\rceil,\qquad
I_{ik}=[k\Delta,\min((k+1)\Delta,D_i)),\;0\le k<L_i.
\]

最后一个窗口允许短于0.5 s，全部窗口保留，不截断为固定50步。不同样本在批量保存时才向右补零，并保存 `lengths`、`sequence_mask` 和 padding 时间 `[-1,-1]`。

## 3. 文本特征与词级时序定位

### 3.1 原文与对齐目标

原始转写 `raw_text` 永远保留。正则表达式提取英文词、缩写和数字，并记录每个词在原文中的 `[char_start,char_end)`。大小写归一化、数字转英文、空格转CTC分隔符只用于声学对齐，绝不改写保存的原文。纯标点不单独作为时间单元，但仍处于文本情感模型的上下文中。

### 3.2 CTC强制对齐

以 `facebook/wav2vec2-base-960h` 对16 kHz单声道音频产生帧后验 \(p_t(a)\)。设规范化目标字符为 \(y\)，插入空白后构造扩展状态 \(z=(\varnothing,y_1,\varnothing,\ldots,y_n,\varnothing)\)。动态规划为

\[
F_t(s)=\log p_t(z_s)+\max\left\{
F_{t-1}(s),F_{t-1}(s-1),
\mathbf1_{\mathrm{skip}(s)}F_{t-1}(s-2)\right\},
\]

其中只有当前状态非空白且与前两状态字符不同时允许跳两步。回溯得到最大后验路径；重复字母必须经过空白，因此 `good` 中的相邻 `o` 不会被错误折叠。每个字符的首尾帧合并为词区间 \(J^T_{ij}\)。Wav2Vec2卷积前端的步距和感受野从模型配置计算，将帧位置映射为秒并裁剪到 `[0,D_i]`。

词的对齐质量定义为字符发射概率的几何平均：

\[
q^T_{ij}=\exp\left(\frac1{|C_{ij}|}\sum_{r\in C_{ij}}\log p_r(y_r)\right).
\]

默认 `q>=0.05` 且词区间内音频可观测比例至少99%时才令 `text_native_mask=true`；否则该词不参与时间池化，原文、语义向量和低置信度信息仍写入metadata。若整段音频缺失或CTC路径不存在，文本模型仍提取转写的上下文特征，同时令文本时间mask为0，并保存 `text_global` 和 `text_alignment_error`。这样“文本语义可用”和“文本时间坐标可用”不会被错误地混为一件事。这个阈值是预先指定的质量门限，不冒充在100条标签上调出来的准确率。

### 3.3 文本情感表示

以 `j-hartmann/emotion-english-distilroberta-base` 冻结推理。对于每个文本块，取最后一层上下文隐藏状态 \(h\in\mathbb R^{768}\) 和7类情绪后验 \(p\in\mathbb R^7\)。一个词的隐藏向量由与其字符区间相交的子词按字符交叠长度加权：

\[
x^T_{ij}=\left[\frac{\sum_r\ell_{ijr}h_{ijr}}{\sum_r\ell_{ijr}},p_{\mathrm{chunk}(ij)}\right]
\in\mathbb R^{775}.
\]

情绪后验是上下文块级信息，不能解释为每个词独立训练得到的7类标签；这正是为什么 `feature_schema.json` 明确记录了768+7的布局。

## 4. 语音特征

使用 `openSMILE 2.5.1` 的 `eGeMAPSv02 / LowLevelDescriptors`，以16 kHz音频计算逐帧声学描述子 \(u^A_{ij}\)，包括基频、响度、谱形、MFCC、抖动、闪烁、谐噪比和共振峰等。每行的真实 `[start,end)` 来自openSMILE时间索引，而不是由行号反推。

对共同窗口使用持续时间交叠权重，得到25维均值和25维标准差：

\[
x^A_{ik}=\left[\mu^A_{ik},\sigma^A_{ik}\right]\in\mathbb R^{50},
\]

\[
\mu^A_{ik}=\frac{\sum_jw_{ikj}u^A_{ij}}{\sum_jw_{ikj}},\quad
\sigma^A_{ik}=\sqrt{\max\left(\frac{\sum_jw_{ikj}(u^A_{ij})^{\odot2}}{\sum_jw_{ikj}}-(\mu^A_{ik})^{\odot2},0\right)}.
\]

真实静音是有效音频；只有非有限描述子或对应的原始音频有缺口时才置 `audio_native_mask=false`。这样不会把“低能量”误判成“缺失”。

## 5. 视觉特征

每个采样帧用OpenCV `haarcascade_frontalface_default.xml` 检测正脸，参数固定为 `scaleFactor=1.1`、`minNeighbors=5`、`minSize=(32,32)`。检测到多人时选择面积最大的脸，但把候选人数和框全部记录；无人脸时视觉观测无效，不用全图伪造一个面部特征。

裁剪脸图送入 `trpakov/vit-face-expression`，取最后一层CLS向量和7类表情后验，得到

\[
x^V_{ij}\in\mathbb R^{775}= [h^{CLS}_{ij}(768),p^{face}_{ij}(7)].
\]

帧的支持区间为当前PTS到下一采样帧PTS，最后一帧截到 \(D_i\)。不把上一张脸无限向后复制；这会把短暂的人脸错误传播到长时间段。每帧保留PTS、解码序号、采样时刻、候选脸数、选中框和图像路径，便于人工回看。

## 6. 三模态时间池化

设模态 \(m\) 的原生支持区间为 \(J^m_{ij}\)，有效标记为 \(v^m_{ij}\)。窗口与原生区间的交叠长度为

\[
w^m_{ikj}=\left|I_{ik}\cap J^m_{ij}\right|v^m_{ij}.
\]

统一窗口特征为

\[
\bar x^m_{ik}=\frac{\sum_jw^m_{ikj}x^m_{ij}}{\sum_jw^m_{ikj}},\qquad
M^m_{ik}=\mathbf1\left(\sum_jw^m_{ikj}>0\right).
\]

由于openSMILE分析窗可能重叠，原始权重和可能超过窗口长度；因此额外计算有效区间并集覆盖率

\[
C^m_{ik}=\frac{\left|I_{ik}\cap\bigcup_{j:v^m_{ij}=1}J^m_{ij}\right|}{|I_{ik}|}\in[0,1].
\]

`coverage` 表示“时间上有多少被观测覆盖”，`mask` 表示是否至少有一个有效观测，`quality` 表示按同样权重聚合的文本置信度/模态观测质量。三者不能相互替代。每个窗口还写入 `*_source_indices`，可由原始区间重算，不依赖黑盒索引。

## 7. 输出契约

单样本 `features.npz` 至少包含：

| 字段 | 含义 |
|---|---|
| `time[L,2]` | 统一窗口起止秒 |
| `text/audio/vision` | 对齐后的特征，维度分别为775/50/775 |
| `*_mask[L]` | 窗口是否有有效模态观测 |
| `*_coverage[L]` | 有效支持区间并集占窗口比例 |
| `*_quality[L]` | 聚合后的质量分数 |
| `*_native`、`*_intervals` | 未池化特征和真实支持区间 |
| `*_native_mask`、`*_native_quality` | 原生有效标记和质量 |
| `*_weights[L,J]` | 窗口到原生观测的交叠长度 |
| `audio_observed` | 16 kHz采样点级可观测掩码 |

批量文件另外保存 `text_global[N,775]` 和 `text_global_mask[N]`。它们表示文本转写仍可计算但尚未得到可靠声学时间定位的情况；不能把它们当作已经均匀铺到每个时间窗口的文本序列。

批量 `aligned_features.npz` 的第一维严格沿 `input_audit.json` 的表顺序，`lengths` 是每个样本真实长度；所有模态特征、coverage和quality在padding区为0，padding mask为false。后续模型必须同时读取 `sequence_mask` 和三种模态mask；不能用特征是否为0猜测缺失。

## 8. 复现、核验与错误边界

`run.json` 保存参数、随机种子、运行设备、依赖版本、源文件哈希和代码哈希；`models.lock.json` 保存每个Hugging Face模型的提交SHA，或保存离线本地快照的目录SHA256；`metadata.json` 保存视频来源、流索引、逐词/逐帧记录和输出哈希。`verify.py`会重新计算时间网格、并集coverage、加权均值/标准差、quality、padding和来源索引，任何一项不一致都失败。服务器不能联网时，使用完整的三个本地快照而不是跳过预训练模型。

建议论文中展示三类证据：

1. 一个三模态都有有效观测的窗口，展示原文词区间、音频支持区间和视频帧框；
2. 一个文本或视觉缺失的窗口，展示mask、coverage和零填充如何区分；
3. 一个多人脸或低置信度样本，展示质量门限没有静默删除样本。

同时报告100条样本的有效词比例、音频覆盖率、无人脸比例、多人脸比例和失败日志。结构核验不等于情感识别准确率，预训练模型在其原始数据集上的指标也不能直接当作附件1结果。

## 9. 与后续问题的关系

附件2的 `aligned_50.pkl/unaligned_50.pkl` 是题目为后续监督建模提供的4850条标准特征，维度和时间组织为题目指定格式。第一问自提的100条特征用于说明原始素材处理和可复现对齐；它们的775/50/775维空间不能直接冒充附件2的768/74/35维空间。

如果后续模型明确选择使用第一问自提特征，输入应是 `feature + mask + coverage + quality`，标签不能进入输入，标准化统计量只能在训练集有效位置估计；并且需要说明100条样本不足以替代附件2的训练规模。按题面完成后续问题时，建议继续使用附件2训练，在第一问报告中用附件1输出验证方法和对齐逻辑。

## 10. 论文中可直接采用的算法伪代码

```text
Audit(label-100.xlsx, video directories)
for sample (video_id, clip_id, transcript):
    probe PTS, duration D and stream metadata
    decode mono audio at 16 kHz; record observed samples
    CTC-align transcript to Wav2Vec2 posterior; record word intervals/confidence
    extract text hidden/emotion vectors at word spans
    extract eGeMAPS LLD at native audio intervals
    sample video by PTS; detect face; extract ViT CLS/emotion vectors
    build I_k=[k*Delta,min((k+1)*Delta,D))
    for modality m:
        compute overlap weights, weighted feature, mask, union coverage, quality
    save native arrays, aligned arrays, masks, timeline indices, metadata and hashes
pad only after all samples are complete; save lengths and sequence_mask
recompute all fields with verify.py
```

代码入口是 `run.py`，数学内核在 `alignment.py`，媒体时间戳在 `media.py`，预训练特征在 `extractors.py`，独立核验在 `verify.py`。
