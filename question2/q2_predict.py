"""使用问题2已训练权重独立完成附件3全量预测，无需再次训练。"""
import argparse
import pickle
from pathlib import Path
import torch
from q2_experiment import (ROOT, TextEncoder, runtime, prepare_split, normalize,
                           Fusion, predict, save_predictions, write_csv, sha, dump)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',default=str(ROOT/'E题数据/E题数据'))
    p.add_argument('--out',default=str(ROOT/'outputs/问题2_附件3预测'))
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args(); runtime(a.device)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    model=Fusion(gated=ck['method'] in ('gated','distilled'),**ck['architecture']).to(a.device)
    model.load_state_dict(ck['model']); model.eval()
    encoder=TextEncoder(a.device,revision=ck['provenance']['bert_revision'])
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    files=sorted((Path(a.data_root)/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))
    if len(files)!=30: raise ValueError(f'附件3应为30个文件，实际{len(files)}，请检查目录')
    rows=[]
    for f in files:
        with f.open('rb') as h: source=pickle.load(h)['test']
        source['id']=[f.stem] if len(source['audio'])==1 else [f'{f.stem}_{i}' for i in range(len(source['audio']))]
        s=normalize(prepare_split(source,encoder,ROOT/'work/问题2_特征缓存','annex3',sha(f)),ck['stats'])
        probability,intensity=predict(model,s,a.device)
        rows.extend(save_predictions(out/'逐样本预测',f'{f.stem}.csv',s,probability,intensity))
    assert len(rows)==30 and len({r['样本编号'] for r in rows})==30
    write_csv(out/'问题2_附件3全量预测.csv',rows)
    dump(out/'问题2_推理记录.json',{'samples':len(rows),'checkpoint_sha256':sha(a.checkpoint),
         'text_model':ck['provenance']['bert_model'],'text_revision':encoder.revision,
         'text_interface':'text_bert, frozen BERT, same as training','class_names':ck['class_names']})
    print('已完成30条附件3预测：',out/'问题2_附件3全量预测.csv')


if __name__=='__main__': main()
