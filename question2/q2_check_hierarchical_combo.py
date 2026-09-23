"""验证集判断分层门控是否应加入最终集成；不访问留出测试集。"""
import pickle
from pathlib import Path
from q2_experiment import ROOT, TextEncoder, normalize, prepare_split, runtime, scenarios, sha, dump
from q2_ensemble import load_models, summarize, selection_score

runtime('cuda:1')
root=ROOT/'E题数据/E题数据'
source=root/'附件2-数据集特征文件/aligned_50.pkl'; source_hash=sha(source)
old=ROOT/'outputs/question2/问题2_首轮实验结果'
optimized=ROOT/'outputs/question2/问题2_优化实验结果'
hier=ROOT/'outputs/question2/问题2_分层模型实验'
paths={f'hierarchical_{s}':hier/f'种子{s}/问题2_分层门控_种子{s}.pt' for s in (2026,2027,2028)}
paths.update({f'distilled_{s}':(old if s==2026 else ROOT/f'outputs/question2/问题2_重复实验_种子{s}')/f'问题2_门控蒸馏_种子{s}.pt' for s in (2026,2027,2028)})
models,ckpts=load_models(paths,'cuda:1')
stats=next(iter(ckpts.values()))['stats']
encoder=TextEncoder('cuda:1',next(iter(ckpts.values()))['provenance']['bert_revision'])
with source.open('rb') as f:d=pickle.load(f)
va=normalize(prepare_split(d['valid'],encoder,ROOT/'work/问题2_特征缓存','valid',source_hash),stats)
suite=scenarios(va,encoder,ROOT/'work/问题2_特征缓存',stats,source_hash)
families={
  'hierarchical_3':[f'hierarchical_{s}' for s in (2026,2027,2028)],
  'hierarchical_distilled_6':list(paths),
  'distilled_3':[f'distilled_{s}' for s in (2026,2027,2028)],
}
scores={}
for name,members in families.items():
 rows,_=summarize(members,models,va,suite,'cuda:1')
 scores[name]={'score':selection_score(rows),'clean_F1':rows[0]['Macro_F1'],'missing_F1':sum(r['Macro_F1'] for r in rows[1:])/len(rows[1:])}
 print(name,scores[name],flush=True)
dump(optimized/'问题2_分层结构候选比较.json',scores)
