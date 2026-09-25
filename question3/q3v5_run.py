"""Q3 v5 token-interaction entrypoint; target checked against official validation."""
import argparse
import os
import sys
import json
from pathlib import Path
if sys.platform=='win32': os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import torch
from q3v2_common import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','explain','report','audit','all','check'])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--bert',type=Path,default=ROOT/'work/pretrained_models/bert')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/question3/q3_v5_01')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    from q3v5_model import candidates
    names=list(candidates(2026))
    p.add_argument('--candidates',nargs='+',choices=names,default=names)
    p.add_argument('--explain-valid',type=int,default=48)
    p.add_argument('--evidence-map',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.patience,args.batch_size,args.explain_valid)<1: p.error('Positive counts required')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.candidates))!=len(args.candidates): p.error('Duplicate seeds/candidates')
    torch.set_num_threads(4)
    if args.stage in ('train','explain','all','check') and args.device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; inspect torch installation')
        torch.cuda.set_device(args.device)
    if args.smoke: args.epochs=1; args.seeds=[2026]; args.explain_valid=3
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=='check':
        from q3v2_model import FrozenText
        encoder=FrozenText(args.bert,args.device)
        if not (args.data_root/'附件2-数据集特征文件/aligned_50.pkl').is_file(): raise FileNotFoundError('Official data missing')
        print({'torch':torch.__version__,'bert_verified':encoder.fingerprint,'device':args.device}); return
    if args.stage in ('train','all'):
        from q3v5_train import train
        train(args)
    if args.stage in ('explain','report','audit'):
        m=json.loads((args.out/'模型参数/manifest.json').read_text(encoding='utf-8'))
        if m['smoke']: args.smoke=True; args.explain_valid=3
    if args.stage in ('explain','all'):
        from q3v5_explain import explain
        explain(args)
    if args.stage in ('report','all'):
        from q3v5_report import run_report
        run_report(args)
    if args.stage in ('audit','all'):
        from q3v2_audit import audit_outputs
        audit_outputs(args.out)


if __name__=='__main__': main()
