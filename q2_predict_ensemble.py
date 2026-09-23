"""从优化版模型文件独立重现附件3的30条预测。"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

from q2_experiment import ROOT, TextEncoder, dump, normalize, prepare_split, runtime, save_predictions, sha, write_csv
from q2_ensemble import load_models, predict_ensemble


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',default=str(ROOT/'outputs/问题2_优化实验结果'))
    parser.add_argument('--data-root',default=str(ROOT/'E题数据/E题数据'))
    parser.add_argument('--out',default=str(ROOT/'outputs/问题2_独立集成推理'))
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args(); runtime(args.device)
    model_dir=Path(args.model_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((model_dir/'问题2_集成选模结论.json').read_text(encoding='utf-8'))
    paths={name:model_dir/'模型参数'/f'{name}.pt' for name in manifest['members']}
    if any(sha(path)!=manifest['checkpoint_sha256'][name] for name,path in paths.items()):
        raise ValueError('模型参数哈希与选模记录不符')
    models,ckpts=load_models(paths,args.device)
    exemplar=next(iter(ckpts.values()))
    encoder=TextEncoder(args.device,manifest['bert_revision'])
    if encoder.revision != manifest['bert_revision']:
        raise ValueError('文本编码器版本不同，不能重现')
    root=Path(args.data_root); files=sorted((root/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))
    if len(files)!=30: raise ValueError('附件3对齐版本必须为30个文件')
    rows=[]
    for file in files:
        with file.open('rb') as f: source=pickle.load(f)['test']
        source['id']=[file.stem] if len(source['audio'])==1 else [f'{file.stem}_{i}' for i in range(len(source['audio']))]
        item=normalize(prepare_split(source,encoder,ROOT/'work/问题2_特征缓存','annex3',sha(file)),exemplar['stats'])
        prob,intensity=predict_ensemble(models,manifest['members'],item,args.device,None)
        for row in save_predictions(out/'逐样本预测',f'{file.stem}.csv',item,prob,intensity):
            row['模型']=manifest['selected']; rows.append(row)
    if len(rows)!=30 or len({r['样本编号'] for r in rows})!=30: raise ValueError('附件3全量结果数量检查失败')
    write_csv(out/'问题2_附件3全量预测.csv',rows)
    dump(out/'问题2_推理记录.json',{'members':manifest['members'],'count':len(rows),
        'checkpoint_sha256':manifest['checkpoint_sha256'],'bert_revision':encoder.revision})
    print('完成附件3集成模型预测：',out/'问题2_附件3全量预测.csv')


if __name__=='__main__': main()
