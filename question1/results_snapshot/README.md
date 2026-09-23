# 已完成运行的特征快照

本目录从本地 `question1/outputs_v3` 复制，包含100条样本的 `features.npz`、`metadata.json`，批量特征、特征定义、输入审计、配置、模型锁定记录、日志和原始验证报告。特征与元数据按字节原样保留。

为了控制Git仓库体积，未复制100份 `audio.wav`、3971张视频采样帧和依赖这些图片的HTML报告。论文使用的一张示例图片位于 `../docs/assets/`。

## 验证边界

- `verification.json` 是服务器完整输出上生成的历史报告，记录100/100通过；不能视为在当前精简目录重新执行完整验证的结果。
- `verify.py` 会检查音频哈希及所有帧文件，因此**不要对本精简目录直接运行 `verify.py`**。
- 要重新完整验证，将原 `outputs_v3` 的全部文件下载到本机，或按根目录README重新提取，然后对完整目录运行 `python verify.py --output outputs_v3`（重新提取时替换为对应目录名）。
- 元数据中的原服务器绝对路径、原输出目录名和代码哈希是运行历史；保留原值，不应因仓库搬迁而改写。
- 根目录 `PACKAGE_MANIFEST.json` 记录GitHub整理版本的文件大小及SHA256，与原运行哈希的用途不同。Git可能转换文本换行；该清单以本地整理时的字节为准，二进制特征不受文本换行转换影响。

## 读取

使用 NumPy 的 `np.load(..., allow_pickle=False)`。先按 `ids` 对应样本，再用 `lengths` 与 `sequence_mask` 获取有效序列，按 `text_mask`、`audio_mask`、`vision_mask` 处理模态缺失。原生时间区间、池化权重在各样本 `features.npz` 中；来源索引、词与帧说明在同目录 `metadata.json` 中。

100条样本的完整特征已包含在本快照中。音视频核验副本、原始数据与模型需要在比赛附件或独立下载位置另行提供。
