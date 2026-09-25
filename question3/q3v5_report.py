"""Shared numerical figures, with accurate v5 architecture and selection disclosure."""
import json
from q3v2_report import run_report as shared_report


def run_report(args):
    shared_report(args)
    path=args.out/'问题3_模型与实测结果说明.md'
    text=path.read_text(encoding='utf-8')
    m=json.loads((args.out/'模型参数/manifest.json').read_text(encoding='utf-8'))
    config=json.loads((args.out/'配置与审计/训练配置.json').read_text(encoding='utf-8'))
    intro=f'''## 建模、网络和训练

 v5沿用官方aligned_50接口、原始文本词元和音视频特征。控制组复现v3的保守LoRA设置；新增模型在每个aligned_50位置保留文本、语音、视觉三个token，经跨模态Transformer和时间Transformer池化。层次头定义P(中性)=sigmoid(n)，正负概率由非中性概率与sigmoid(s)分解。主模型参数只从官方训练3395条样本学习。

分类和回归使用独立头。总损失为融合CE + reg_weight×SmoothL1 + text_ce×文本CE + text_reg×文本SmoothL1 + av_aux×(AV CE + reg_weight×AV SmoothL1)，干预视图额外增加augmentation×(CE+reg_weight×SmoothL1)。类别权重与训练类别频率的负weight_power次幂成比例。控制组保留v3的标签平滑和辅助损失；token交互候选只改变融合结构或单项正则参数，所有实际系数保存在各模型config中。

最终首组件参数：{json.dumps(m['selected']['config'],ensure_ascii=False)}。正式训练计划：epochs={config['epochs']}、patience={config['patience']}、batch_size={config['batch_size']}。优化器AdamW、weight decay=0.01、10%预热与余弦衰减、梯度范数上限1。

部署规则：{m['deployment_rule']}，组件数{len(m['components'])}，类别log概率偏置={m['class_bias']}。先比较全部指定结构，再对首种子胜出结构补齐另外两个种子；部署候选为各首种子模型、胜出结构等权种子集成、前两结构等权集成。

允许在验证集选择两个类别偏置：负/中各取[-0.4,0.4]步长0.1，正类为参考0，共81组。仅当按视频分组的5折偏置交叉拟合比原始决策提高至少0.002时才启用非零偏置。基础模型及epoch已使用同一验证集选模，因此该交叉拟合只是决策稳定性检查，不是独立模型交叉验证，更不是未接触测试成绩。偏置从未按样本ID、词句、真值做特例修正。

Macro-F1≥0.65是否实际达标见目标检查.json。旧结果和烟雾实验不能计作本次正式达标结果。

'''
    start=text.index('## 建模、网络和训练'); end=text.index('## 性能与可视化')
    text=text[:start]+intro+text[end:]
    start=text.index('## 复现和限制')
    text=text[:start]+'''## 复现和限制

模型参数/manifest.json保存全部component_*.pt、权重、基础BERT指纹和类别偏置。token交互组件包含跨模态和时间Transformer；服务器训练包不包含模型数据。最终竞赛附件50MB总大小限制仍需在确定模型后单独验证压缩精度。

解释始终对实际部署概率执行：先对组件概率等权平均，再应用固定类别偏置归一化，随后枚举8子集和局部遮挡。分类目标为最终log(P(c))-log(P(d))，回归为组件均值；空输入基线也经过同样固定偏置。

运行命令见question3/README_Q3_V5_SERVER.md。问题1修订目录不在本模型输入依赖中；不可靠时间对应关系的处理原则可以借鉴，但附件1词时段不能直接移植至附件4。若没有附件4特征行到原媒体的已核验映射，不能宣称已交付精确语音时段或关键帧定位。
'''
    path.write_text(text,encoding='utf-8')
