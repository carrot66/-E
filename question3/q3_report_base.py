"""Generate the five requested deliverable groups from actual run results."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q3_common import MODS, LABELS, csv_read, csv_write, dump, sha


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def run_report(args):
    out = args.out; plots = out / '图表'; plots.mkdir(exist_ok=True)
    cards = out / '解释卡'; cards.mkdir(exist_ok=True)
    metrics = read_json(out / '验证集评价/评价.json')
    manifest = read_json(out / '模型参数/manifest.json')
    preds = csv_read(out / '验证集评价/全量预测.csv')
    explanation = csv_read(out / '附件4预测/全量预测与解释.csv')
    windows = csv_read(out / '附件4预测/局部窗口全量.csv')
    evidence = csv_read(out / '附件4预测/关键证据.csv')
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, xlabel='Predicted', ylabel='True', title='Official validation confusion matrix')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(plots / 'validation_confusion.png', dpi=160); plt.close(fig)
    from sklearn.metrics import classification_report
    cr = classification_report([int(r['true_class']) for r in preds], [int(r['predicted_class']) for r in preds],
                               labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0)
    dump(out / '验证集评价/逐类别评价.json', cr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([float(r['true_intensity']) for r in preds], [float(r['intensity']) for r in preds], alpha=.3, s=12)
    ax.plot([-3, 3], [-3, 3], '--', color='gray'); ax.set(xlabel='True intensity', ylabel='Predicted intensity', title='Validation regression')
    fig.tight_layout(); fig.savefig(plots / 'validation_regression.png', dpi=160); plt.close(fig)
    rows = csv_read(out / '验证集评价/模型比较.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ('Macro_F1', 'MAE')):
        ax.bar([r['name'] for r in rows], [float(r[metric]) for r in rows]); ax.set_title(metric); ax.tick_params(axis='x', rotation=65)
    fig.tight_layout(); fig.savefig(plots / 'model_comparison.png', dpi=160); plt.close(fig)
    history = csv_read(out / '训练记录' / manifest['selected']['name'] / 'history.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, keys in zip(axes, [('loss',), ('Accuracy', 'Macro_F1'), ('MAE',)]):
        for key in keys: ax.plot([int(r['epoch']) for r in history], [float(r[key]) for r in history], label=key)
        ax.set_xlabel('Epoch'); ax.legend()
    fig.tight_layout(); fig.savefig(plots / 'training.png', dpi=160); plt.close(fig)
    if 'Neutral_F1' in history[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in LABELS:
            ax.plot([int(r['epoch']) for r in history], [float(r[name+'_F1']) for r in history], 'o-', label=name)
        ax.set(xlabel='Epoch', ylabel='Per-class validation F1', ylim=(0,1), title='Class-specific generalization'); ax.legend()
        fig.tight_layout(); fig.savefig(plots / 'per_class_training.png', dpi=160); plt.close(fig)
    shares = np.array([[float(r[m + '_share']) for m in MODS] for r in explanation])
    fig, ax = plt.subplots(figsize=(11, 5)); bottom = np.zeros(len(shares))
    for m, name in enumerate(MODS): ax.bar(range(len(shares)), shares[:, m], bottom=bottom, label=name); bottom += shares[:, m]
    ax.set(xticks=range(len(shares)), xticklabels=[r['sample_id'] for r in explanation], ylabel='Normalized absolute modality Shapley')
    ax.legend(); fig.tight_layout(); fig.savefig(plots / 'modality_comparison.png', dpi=160); plt.close(fig)
    fidelity_summary = []
    for split, folder in [('validation', '验证集解释'), ('attachment4', '附件4预测')]:
        faithful = csv_read(out / folder / '忠实度.csv')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for m, ax in zip(MODS, axes):
            for kind in ('top', 'random', 'bottom'):
                values = []
                for rate in (.1, .2, .3):
                    selected = [float(r['probability_drop']) for r in faithful if r['modality'] == m and r['intervention'] == kind and float(r['rate']) == rate]
                    value = float(np.mean(selected)) if selected else None
                    values.append(value if value is not None else np.nan)
                    fidelity_summary.append({'split': split, 'modality': m, 'kind': kind, 'rate': rate, 'mean_probability_drop': value, 'evaluations': len(selected)})
                ax.plot([.1, .2, .3], values, 'o-', label=kind)
            ax.set(title=m, xlabel='Fraction of units removed', ylabel='Fixed target probability drop'); ax.legend()
        fig.tight_layout(); fig.savefig(plots / f'{split}_faithfulness.png', dpi=160); plt.close(fig)
    csv_write(out / '验证集评价/解释忠实度汇总.csv', fidelity_summary)
    groups = []
    definitions = [('all', lambda r: True), ('neutral', lambda r: int(r['true_class']) == 1),
                   ('near_zero_abs_le_0.5', lambda r: abs(float(r['true_intensity'])) <= .5),
                   ('truncated', lambda r: r['truncated'] == 'True'),
                   ('vision_unavailable', lambda r: int(r['vision_rows']) == 0),
                   ('audio_unavailable', lambda r: int(r['audio_rows']) == 0),
                   ('high_confidence_ge_0.8', lambda r: float(r['confidence']) >= .8)]
    for name, condition in definitions:
        group = [r for r in preds if condition(r)]
        groups.append({'group': name, 'samples': len(group),
                       'classification_error_rate': float(np.mean([r['true_class'] != r['predicted_class'] for r in group])) if group else None,
                       'MAE': float(np.mean([float(r['absolute_error']) for r in group])) if group else None})
    csv_write(out / '验证集评价/错误分层统计.csv', groups)
    errors = sorted([r for r in preds if r['true_class'] != r['predicted_class']], key=lambda r: -float(r['confidence']))
    review_path = out / '验证集评价/错误人工复核表.csv'
    previous_reviews = {r['sample_id']: r for r in csv_read(review_path)} if review_path.exists() else {}
    review = []
    for r in errors:
        old = previous_reviews.get(r['sample_id'], {})
        review.append({**r, **{k: old.get(k, default) for k, default in
                               [('manual_cause', ''), ('manual_evidence', ''), ('review_status', 'pending')]}})
    csv_write(out / '验证集评价/错误人工复核表.csv', review)
    style = '<style>body{font:16px system-ui;max-width:1160px;margin:30px auto;background:#f4f6fa;color:#182538}article{background:white;padding:24px;margin:22px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd4de}img{max-width:100%}small{color:#516276}</style>'
    links = []; time_verified = 0
    for row in explanation:
        sid = row['sample_id']; local = [r for r in windows if r['sample_id'] == sid and int(r['window_units']) == 1]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7))
        for m, ax in zip(MODS, axes):
            rr = [r for r in local if r['modality'] == m]
            xx = [int(r['unit_start']) for r in rr]; yy = [float(r['class_margin_effect']) for r in rr]
            ax.bar(xx, yy, color=['#d86951' if y >= 0 else '#407fa6' for y in yy]); ax.axhline(0, color='gray', lw=.7)
            ax.set(title=f'{m} | '+('MAIN MODALITY' if row['main_modality'] == m else ''), xlabel='Modality-local unit index (NOT seconds)', ylabel='Margin drop')
        fig.suptitle(f'{sid}: {row["polarity"]} vs {row["contrast_class"]}'); fig.tight_layout()
        fig.savefig(plots / f'{sid}_local.png', dpi=150); plt.close(fig)
        table = '<tr><th>模态</th><th>分类Shapley</th><th>相对作用</th><th>强度Shapley</th><th>可观测行</th></tr>'
        for m in MODS:
            table += f'<tr><td>{m}</td><td>{float(row[m+"_shapley_class"]):.4f}</td><td>{float(row[m+"_share"]):.1%}</td><td>{float(row[m+"_shapley_intensity"]):.4f}</td><td>{row[m+"_observed_rows"]}</td></tr>'
        entries = []
        for r in [r for r in evidence if r['sample_id'] == sid]:
            time_verified += r['time_mapping_status'] == 'verified'
            anchor = r['text'] or f"特征行 {r['feature_indices']}"
            timing = f"{r['start_seconds']}—{r['end_seconds']}秒" if r['start_seconds'] else '秒级映射待核验'
            entries.append(f'<tr><td>{r["modality"]}/{r["direction"]}</td><td>{html.escape(anchor)}</td><td>{float(r["class_margin_effect"]):.4f}</td><td>{float(r["after_target_probability"]):.4f}</td><td>{html.escape(timing)} ({r["time_mapping_status"]})</td></tr>')
        body = f'<h1>附件4 样本 {sid}</h1><article><h2>{row["polarity"]} · 强度 {float(row["intensity"]):.3f}</h2><p>{html.escape(row["raw_text"])}</p>'
        body += f'<p>概率：Negative {float(row["p_negative"]):.3f} / Neutral {float(row["p_neutral"]):.3f} / Positive {float(row["p_positive"]):.3f}</p>'
        body += f'<p>分类主要模态：{row["main_modality"]}；强度主要模态：{row["intensity_main_modality"]}；基线敏感：{row["baseline_sensitive"]}</p><table>{table}</table></article>'
        body += f'<article><img src="../图表/{sid}_local.png"><p>三行使用独立模态位置，不能据此断言同列对应同一视频时间。正值支持固定预测类别相对对比类别，负值表示反对。</p></article>'
        body += '<article><h2>支持／反对证据及实际删除影响</h2><table><tr><th>模态/方向</th><th>证据</th><th>类别差值下降</th><th>删除后原类别概率</th><th>时间映射</th></tr>'+''.join(entries)+'</table></article>'
        body += '<small>无情感真值，不标注预测正确或错误。音视频未核验的时间不填造；原视频位置见附件4预测/视频映射.csv。归因是模型相对指定替换基线的行为，不是现实因果。</small>'
        (cards / f'{sid}.html').write_text('<!doctype html><meta charset="utf-8">'+style+body, encoding='utf-8')
        links.append(f'<li><a href="解释卡/{sid}.html">样本 {sid} · {row["polarity"]} · {row["main_modality"]}</a></li>')
    version = manifest.get('version', 2)
    (out / '问题3_结果浏览.html').write_text('<!doctype html><meta charset="utf-8">'+style+f'<h1>问题3 v{version} 实测结果</h1>'+
        ('<p><b>技术烟雾测试，不能作为比赛实验结果。</b></p>' if manifest['smoke'] else '')+
        '<p>完整说明见问题3_模型与实测结果说明.md。各模态时间映射状态见证据表。</p><img src="图表/modality_comparison.png"><ul>'+''.join(links)+'</ul>', encoding='utf-8')
    worst = sorted([g for g in groups if g['samples'] >= 5], key=lambda g: g['classification_error_rate'], reverse=True)
    error_text = '\n'.join(f'- {g["group"]}：n={g["samples"]}，错误率={g["classification_error_rate"]:.3f}，MAE={g["MAE"]:.3f}。' for g in worst)
    reviewed = [r for r in review if r['review_status'] == 'reviewed' and r['manual_cause'] and r['manual_evidence']]
    review_summary = {}
    for r in reviewed: review_summary[r['manual_cause']] = review_summary.get(r['manual_cause'], 0) + 1
    error_text += '\n\n已人工复核并填写证据的错误数：' + str(len(reviewed)) + '；人工原因计数：' + str(review_summary)
    main_counts = {m: sum(r['main_modality'] == m for r in explanation) for m in MODS}
    cfg = manifest['selected']['config']
    architecture = '冻结同版本BERT，输入词元遮挡后重编码；A/V按训练集观测行标准化。三模态使用独立时序轴，投影后两层残差深度卷积TCN（核3，扩张1/2），模态级4头注意力、可用性门控池化、64维共享头和三分类/强度双头。门控不是归因结果。'
    training = '损失为CE + 0.6 SmoothL1；增强开启时约10%样本生成额外遮挡视图，附加损失权重0.2。AdamW，weight decay=1e-4，梯度范数上限1，完整验证Macro-F1选模，完全同分时选MAE较低者。'
    deployment = '最终模型在模型参数/best.pt，BERT不重复打包，使用固定资源manifest；脚本见question3/q3_*.py与run_q3_server.sh。'
    target_text = ''
    if version == 3:
        architecture = ('固定基础BERT权重，在最后四层query/value注入rank=8、alpha=16的LoRA适配器；冻结对照不启用LoRA。'
            '文本使用CLS与有效词元均值拼接；音频74维和视觉35维在各自观测轴上用两层TCN编码为64维后拼接。'
            '文本分类/回归分别使用128/64维任务头，A/V单独预测相对训练先验的残差，分类与回归各有一个0至0.5的门控。'
            '文本缺失时使用A/V预测，全缺失时输出训练类别先验和训练强度均值。'
            'hierarchical候选将文本和A/V辅助分类头写成P(中性)=sigmoid(n)，P(正)=sigmoid(-n)sigmoid(s)，P(负)=sigmoid(-n)sigmoid(-s)；最终经残差融合、softmax归一化。'
            '所有候选均使用3*tanh(r/3)限制强度，门控权重不作为解释。')
        training = ('总损失为融合加权CE + 0.3 SmoothL1 + 0.2文本加权CE + 0.1文本SmoothL1 '
            '+ 0.05(A/V加权CE + 0.3 A/V SmoothL1)，增强视图额外附加0.15(加权CE+0.3 SmoothL1)。'
            '类别权重为训练样本频率倒数平方根再归一化，label smoothing=0.03。'
            'AdamW weight decay=0.01，head lr=2e-4，adapter lr=1e-4，10%预热后余弦衰减，梯度范数上限1，dropout=0.3。'
            '默认16轮、patience=4、batch=16；三个预先指定结构先用首种子比较，最佳结构再做另外两个种子。'
            '最终仅比较首种子模型与等权种子集成，按验证Macro-F1选择，完全同分选MAE较低者。'
            '配置与审计/训练配置.json保存实际轮数、batch、候选与种子；本报告中的默认值不覆盖该记录。')
        deployment = (f'最终部署包含{len(manifest["components"])}个组件，ensemble={manifest["ensemble"]}。'
            '模型参数/manifest.json列出组件文件、哈希和权重；component_*.pt仅保存适配器、任务层及训练标准化统计，外部BERT必须匹配指纹。'
            '集成按概率和回归值分别等权平均，解释也对同一个集成执行干预；分类目标采用log(P(c))-log(P(d))。'
            '复现脚本为question3/q3_*.py与run_q3_server.sh。')
        target = read_json(out / '目标检查.json')
        target_text = f'\n\nMacro-F1目标=0.65；本次实测={target["value"]:.6f}；达标={target["passed"]}。烟雾测试不计达标。'
    report = f'''# 问题3：模型与实测结果说明

运行状态：{'仅技术烟雾测试，不构成训练完成或竞赛成绩' if manifest['smoke'] else '官方附件2训练/验证实验'}。

## 建模、网络和训练

{architecture}

{training} 最终结构参数：{json.dumps(cfg, ensure_ascii=False)}。首种子选中检查点轮次{manifest['selected']['best_epoch']}。全部运行配置、数据及代码哈希见配置与审计。

## 性能与可视化

Accuracy={metrics['Accuracy']:.6f}，Macro-F1={metrics['Macro_F1']:.6f}，Weighted-F1={metrics['Weighted_F1']:.6f}，MAE={metrics['MAE']:.6f}，Pearson={metrics['Pearson']}。分类/回归符号差异比例={metrics['sign_conflict_rate']:.4f}（包含中性分类对应非零强度，不能全解释为正负情绪相反）。{target_text}

验证集用于选模，因此以上不是独立测试成绩。附件4无标签，不报告准确率。同协议模型比较、种子统计、训练曲线、混淆矩阵、回归散点图位于验证集评价和图表目录。

## 可复核解释

固定完整输入预测类别c与次高类别d，以logit(c)-logit(d)和回归强度为两个目标。枚举8个模态子集，Shapley权重为|S|!(2-|S|)!/6；三个贡献之和等于完整输出减空输入先验。模态作用程度为绝对Shapley归一化。全部8子集输出可独立复算。

局部证据按每模态1/3/5个单元窗口遮挡后重预测，保留支持和反对窗口；文本单元包含完整词元组，A/V单元是各自的特征行。局部窗口效应不保证加和为模态Shapley。比较10%/20%/30%组合删除与随机/最低支持删除，充分性保留测试固定其他模态，详见忠实度长表与曲线。

缺失状态与观测掩码固定的零特征是两种不同干预游戏，基线敏感性仅作稳定性警示。零特征文本对照不是原始词删除。自然缺失、填充和人工干预不能混称。

## 20条附件4解释卡与模态差异

20条全量结果见附件4预测/全量预测与解释.csv；每条解释卡包含概率、强度、三模态作用、主要模态、支持/反对证据和删除影响。分类主要模态计数：{main_counts}。这描述当前模型对20条输入的依赖，不能推广为一般模态优劣。

关键证据中具有verified时间映射的窗口数为{time_verified}。没有外部已核验映射时，文本仍可定位原文字符，A/V仅定位特征行，音视频精确时段/关键帧交付尚未完成，不能以文本对齐替代官方A/V行映射证明。

## 错误归因的已支持结论

{error_text}

以上仅是可观测数据分组关联，不足以断言讽刺、表情或声学原因。错误人工复核表列出原文、置信度、类别和强度误差，manual_cause/manual_evidence留待实际回看后填写，不生成虚构人工结论。不得修改官方标签。

## 复现和限制

{deployment} 所有新增输出独立于旧线性基线。解释是指定干预下的模型依赖，不是情感真因；证据时间未核验时不填0秒或均分视频时长。运行完成不等于未核验音视频映射已经完成。
'''
    (out / '问题3_模型与实测结果说明.md').write_text(report, encoding='utf-8')
    dump(out / '交付状态.json', {'smoke': manifest['smoke'], 'attachment4_count': len(explanation),
         'cards': len(links), 'prediction_and_attribution_generated': True,
         'verified_time_windows': time_verified, 'precise_av_mapping_complete': all(r['time_mapping_status'] == 'verified' for r in evidence if r['modality'] in ('audio', 'vision')) and any(r['modality'] in ('audio', 'vision') for r in evidence),
         'manual_error_causes_reviewed': bool(reviewed), 'manual_reviewed_error_count': len(reviewed),
         'numerical_audit': 'run audit stage'})
    print(f'Report: {out / "问题3_结果浏览.html"}', flush=True)
