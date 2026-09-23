# CMU-MOSEI E题代码

## 1. 安装
```powershell
python -m pip install -r requirements.txt
```
使用附件2的 `aligned_50.pkl`，因为附件3/4均有对齐版本。

## 2. 训练
```powershell
python mosei_solution.py train --data-root "C:\Users\14494\Desktop\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据\E题数据" --out outputs\best.pt --epochs 35
```
训练目标：三分类交叉熵 + 0.35×SmoothL1情感强度回归。训练期间随机生成连续局部缺失和整模态缺失，模拟附件3。

## 3. 附件3预测
```powershell
python mosei_solution.py infer --data-root "...\E题数据\E题数据" --ckpt outputs\best.pt --kind missing --out outputs\attachment3_predictions.csv
```

## 4. 附件4预测和解释
```powershell
python mosei_solution.py infer --data-root "...\E题数据\E题数据" --ckpt outputs\best.pt --kind explain --out outputs\attachment4_explanations.csv
```
解释列包括三模态作用权重、主要模态和各模态关键时间位置。位置从0开始，对应aligned_50的序列位置；论文中要进一步把位置映射回原始视频时间轴。

## 5. 比赛论文建议
必须补充：验证集消融实验、不同缺失率曲线、随机种子重复实验、典型样本解释卡、问题1的100条特征提取与对齐日志。该代码是可复现基线，不能保证比赛名次。
