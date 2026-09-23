# 问题1数学模型与实现对应

## 一、样本定义与原始处理

样本 i 用 s_i = video_id + `$_$` + clip_id 唯一标识，记录原视频V_i、给定转写R_i、原标签y_i。只读取附件1；表中每一行必须恰好对应一个 `video_id/clip_id.mp4`，重复ID、缺视频或多余视频均报错。默认样本数为100。标签仅随结果保存，不输入模型。

使用PyAV解码，时间基准为容器起始PTS：t = PTS × time_base − origin。音频重采样16kHz单声道，并按真实PTS放入统一时间轴；空洞写零且 `audio_observed=false`，不把缺失伪装为静音观测。视频按5Hz目标网格选择第一个达到目标时刻的解码帧，保留真实时间戳及解码序号，不直接用“帧号/FPS”假定恒定帧率。遇到缺PTS或不可解码视频，写失败记录，不静默删除样本。

## 二、文本情感特征与时间定位

给定文本保留原样。提取英文词、缩写和数字的字符偏移；数字转英文、大小写归一化、空格转CTC分隔符仅用于对齐目标，不修改原文。纯标点不单列时间单元；词语特征仍从包含标点的原文上下文中计算。

以 facebook/wav2vec2-base-960h 对16kHz音频计算CTC帧后验 p_t(c)。在完整目标字符序列y与空白符组成的状态序列上求：

\[\pi^*=\arg\max_{\pi:\mathcal B(\pi)=y}\sum_{t=1}^{T_a}\log p_t(\pi_t).\]

动态规划允许停留、前移一状态，或在非空白且与前两状态字符不同的条件下跳两状态。因此重复字符必须经空白分隔，不能被合并。无法对齐完整目标时失败并记录，**不采用整段文本均匀摊时的替代方案**。

卷积前端步距s和感受野r由模型config计算，帧边界近似为 b_t=(ts+(r−s)/2)/16000，再裁到片段范围。词区间取该词所有字符路径的首尾帧范围；词置信度为各字符路径几何平均概率的几何均值。置信度低于0.05或音频可观测比例低于99%时，该词不参与时序池化，原文、语义向量、时间和置信度仍留档。若整段音频缺失或没有完整CTC路径，文本模型仍对转写计算语义特征，同时将文本时间mask设为0并记录 `text_alignment_error`，不因音频缺失而丢掉独立文本模态。这是CTC声学定位近似，置信度未经校准；同音词、噪声、口音及转写不符会影响可靠性。

用 j-hartmann/emotion-english-distilroberta-base 冻结推理。长文本按最多384个子词切块，不截断；每个词的768维语义表示由与其字符跨度相交的子词隐藏状态按字符交叠长度加权平均。追加所属文本块的7维情绪后验，得到 x_j^T ∈ R^775。情绪后验属于上下文块，**不能解释成每个词独立情绪分类**；短样本通常共享同一上下文后验。类别顺序保存于schema，不假定不同模型类别顺序相同。

## 三、语音情感相关特征

使用openSMILE 2.5.1 的 eGeMAPSv02 / LowLevelDescriptors，提取25维时序描述子 u_j：基频、响度、频谱形状、MFCC、声源抖动/微扰、谐噪比和共振峰等。具体名称与单位以openSMILE定义为准，逐维名称完整保存。时间支持区间直接取openSMILE返回的start/end；配置通常使用10ms帧移及不同分析窗，不把全部描述子虚构为同一种窗口。

在共同窗口内求25维加权均值μ和25维加权标准差σ，组成 x_k^A=[μ_k;σ_k]∈R^50，刻画语调、强度、音色与短时变化。声学特征是情感相关观测，不是已经校准的情感强度预测。真实静音保持有效；非有限值或包含音频缺口的描述子标为无效。输出同时给出 `audio_coverage`（有效分析区间并集占窗口比例）与 `audio_quality`，避免把重叠分析窗的权重和误当作覆盖率。

## 四、视觉情感特征

OpenCV预训练Haar正脸检测器使用 scaleFactor=1.1、minNeighbors=5、minSize=(32,32)。有多张脸时确定性选择面积最大者，记录所有候选数量和所选框；无人脸时该视觉观测无效，不以全图强行代替人脸。

用 trpakov/vit-face-expression 的官方图像处理器处理裁剪人脸，冻结ViT提取最后隐藏层CLS的768维表示及7类表情softmax后验，拼接成 x_j^V∈R^775。该方案描述面部表情，不建模全身动作；最大脸不一定是发言人，Haar对侧脸/遮挡召回不足。多人或强遮挡案例应在报告中人工核对，不声称已完成说话人身份跟踪。

采样帧j的支持区间为 [t_j, min(t_j+1/f_v,t_{j+1},D_i)]，最后一帧不包含下一帧项；局部缺帧留下空洞，不无限外推最近人脸。完整采样帧与框均保存，以便核验裁剪。

## 五、跨模态统一时间轴

窗口宽Δ=0.5秒，L_i=ceil(D_i/Δ)，I_ik=[kΔ,min((k+1)Δ,D_i))，k从0开始。全片保留，末窗口可短于Δ。对每个模态的原生支持区间J_ij及有效标记v_ij，定义：

\[w_{ikj}^{m}=|I_{ik}\cap J_{ij}^{m}|v_{ij}^{m},\quad M_{ik}^{m}=\mathbf1\{\sum_j w_{ikj}^{m}>0\},\]
\[\bar x_{ik}^{m}=\frac{\sum_j w_{ikj}^{m}x_{ij}^{m}}{\sum_jw_{ikj}^{m}}.\]

无交叠时置零且M=0。另定义有效支持区间并集覆盖率

\[C_{ik}^{m}=|I_{ik}\cap\cup_{j:v_{ij}^{m}=1}J_{ij}^{m}|/|I_{ik}|\in[0,1],\]

以及按同一权重聚合的质量分数 `quality`。`mask`、`coverage`、`quality` 分别表示有无观测、时间覆盖比例和观测可信程度，三者不能相互替代。音频附加逐维标准差：

\[\sigma_{ik}^{A}=\sqrt{\max\{\frac{\sum_j w_{ikj}^{A}(u_j\odot u_j)}{\sum_jw_{ikj}^{A}}-\mu_{ik}^{A}\odot\mu_{ik}^{A},0\}}.\]

音频分析窗可能重叠，权重定义为描述子支持区间交叠长度；其总和不等于窗口覆盖时长，不能用权重和直接宣称“覆盖率”。原生区间、权重矩阵和各窗口来源索引均保存，可完全重算池化；文字来源行号和音频/视觉来源流索引也写入metadata。没有强制三模态在每个窗口都有效；无文字的停顿保留真实时间窗口，文本mask为0。

批量长度Lmax=max_i L_i，各样本右侧补零，序列mask为 k<L_i，padding时间为[-1,-1]。模态mask与序列mask分开记录：前者表示观测可用，后者表示样本实际时域。默认不做全数据均值方差标准化；后续监督模型如需标准化，应只在相应训练集的有效位置估计统计量。

## 六、复现与合理性边界

100条记录保持一一对应，异常样本不会移除或修改标签。模型提交SHA、包版本、代码/输入哈希、参数、处理日志、原生序列、对齐矩阵、coverage/quality与padding记录可核验。`verify.py`重新计算池化并与保存值比对；`make_report.py`生成文本—音频—帧证据示例。可以直接写入论文的完整推导和算法伪代码见 `Q1_SOLUTION.md`。

本项目没有训练或调参过程，不下载模型原训练集。使用的模型权重包含其公开预训练/微调知识，符合题目允许使用预训练模型的场景。预训练情绪类别与MOSEI的[-3,3]情感强度不同，不能直接用表情类别或文本softmax替换原标签。附件2标准特征维度/时间组织与本项目不同，后续问题须按其要求使用附件2，不将本项目输出冒充官方特征。

方法合理性依赖英语转写基本正确、视频时间戳可信、至少部分正脸可见。正式论文应报告实际低置信度词比例、各模态有效窗口比例、多人脸比例，以及人工抽查结果。结构检查不代替识别精度评价。

## 工具与模型来源

- Wav2Vec2 CTC：[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)，16kHz英语语音，仅用于定位给定转写。
- 文本情感模型：[j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)。
- 面部表情模型：[trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)。
- 声学特征：[openSMILE Python文档](https://audeering.github.io/opensmile-python/)、[eGeMAPS配置](https://github.com/audeering/opensmile/tree/master/config/egemaps/v02)。
- 媒体解码：[PyAV文档](https://pyav.org/docs/stable/)；人脸检测：[OpenCV CascadeClassifier](https://docs.opencv.org/4.x/d1/de5/classcv_1_1CascadeClassifier.html)。

版本见requirement.txt，实际模型版本以运行生成的models.lock.json为准。预训练模型保持eval/inference模式；随机种子2026。
