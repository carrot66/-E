"""Competition-ready baseline for the 2026 E problem.

By default, data is read from the bundled ``E题数据/E题数据`` directory.
Pass ``--data-root`` to use a different data directory.
"""
import argparse, csv, pickle, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / 'E题数据' / 'E题数据'

def seed_all(s=2026):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class MOSEI(Dataset):
    def __init__(self, d, train=False, augment=False):
        self.d=d; self.train=train; self.augment=augment
    def __len__(self): return len(self.d['id']) if 'id' in self.d else self.d['audio'].shape[0]
    def __getitem__(self,i):
        x=[self.d['text'][i].astype('float32'),self.d['audio'][i].astype('float32'),self.d['vision'][i].astype('float32')]
        if self.augment: x=mask_modalities(x)
        y=np.int64(self.d.get('classification_labels',np.array([-1]))[i]) if 'classification_labels' in self.d else -1
        r=np.float32(self.d.get('regression_labels',np.array([0.]))[i]) if 'regression_labels' in self.d else 0.
        return x[0],x[1],x[2],y,r

def mask_modalities(xs, p=0.35):
    out=[]
    for x in xs:
        z=x.copy(); L=z.shape[0]
        if random.random()<p:
            a=random.randrange(L); b=min(L,a+max(1,int(L*random.uniform(.08,.35))))
            z[a:b]=0
        if random.random()<0.12: z[:]=0
        out.append(z)
    return out

class Encoder(nn.Module):
    def __init__(self, inp, h=128):
        super().__init__(); self.proj=nn.Sequential(nn.LayerNorm(inp),nn.Linear(inp,h),nn.GELU()); self.gru=nn.GRU(h,h,batch_first=True,bidirectional=True); self.att=nn.Linear(2*h,1)
    def forward(self,x):
        z,_=self.gru(self.proj(x)); a=torch.softmax(self.att(z).squeeze(-1),1); return (z*a.unsqueeze(-1)).sum(1),a

class MultiTask(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.ModuleList([Encoder(768),Encoder(74),Encoder(35)]); self.gate=nn.Sequential(nn.Linear(768,128),nn.GELU(),nn.Linear(128,3)); self.fuse=nn.Sequential(nn.Linear(256*3,256),nn.GELU(),nn.Dropout(.25)); self.cls=nn.Linear(256,3); self.reg=nn.Linear(256,1)
    def forward(self,t,a,v):
        hs=[]; ats=[]
        for x,e in zip([t,a,v],self.enc): h,att=e(x); hs.append(h); ats.append(att)
        q=torch.softmax(self.gate(torch.cat(hs,1)),1); z=torch.cat([hs[i]*q[:,i:i+1] for i in range(3)],1); z=self.fuse(z)
        return self.cls(z),self.reg(z).squeeze(1),q,ats

def load_data(root, aligned=True):
    p=Path(root)/'附件2-数据集特征文件'/('aligned_50.pkl' if aligned else 'unaligned_50.pkl')
    with open(p,'rb') as f: return pickle.load(f)

def run_epoch(model,loader,opt,device,train=True):
    model.train(train); losses=[]; ys=[]; ps=[]; rs=[]; rr=[]
    for t,a,v,y,r in loader:
        t,a,v=t.to(device),a.to(device),v.to(device); y,r=y.to(device),r.to(device)
        with torch.set_grad_enabled(train):
            lg,pr,_,_=model(t,a,v); loss=nn.functional.cross_entropy(lg,y)+0.35*nn.functional.smooth_l1_loss(pr,r)
            if train: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); opt.step()
        losses.append(loss.item()); ys += y.cpu().tolist(); ps += lg.argmax(1).cpu().tolist(); rs += r.cpu().tolist(); rr += pr.cpu().tolist()
    return {'loss':float(np.mean(losses)),'acc':accuracy_score(ys,ps),'f1':f1_score(ys,ps,average='macro'),'mae':mean_absolute_error(rs,rr),'y':ys,'p':ps,'r':rs,'rp':rr}

def train(root,out,epochs=35,batch=32,seed=2026):
    seed_all(seed); device='cuda' if torch.cuda.is_available() else 'cpu'; d=load_data(root)
    tr=DataLoader(MOSEI(d['train'],augment=True),batch_size=batch,shuffle=True,num_workers=0); va=DataLoader(MOSEI(d['valid']),batch_size=batch,shuffle=False)
    m=MultiTask().to(device); opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4); best=-1
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    for ep in range(1,epochs+1):
        a=run_epoch(m,tr,opt,device,True); b=run_epoch(m,va,None,device,False); score=b['f1']-0.08*b['mae']
        print(f'epoch {ep:03d} train_loss={a["loss"]:.4f} val_f1={b["f1"]:.4f} val_acc={b["acc"]:.4f} val_mae={b["mae"]:.4f}')
        if score>best: best=score; torch.save({'model':m.state_dict(),'epoch':ep,'val':{k:b[k] for k in ['f1','acc','mae']}},out)
    print('saved',out)

def tensorize(d):
    def arr(k,dim):
        x=d[k]; x=np.asarray(x); return x if x.ndim==3 else x[None,...]
    return arr('text',768),arr('audio',74),arr('vision',35)

@torch.no_grad()
def predict(model, t,a,v,device):
    model.eval(); lg,r,q,ats=model(torch.tensor(t).float().to(device),torch.tensor(a).float().to(device),torch.tensor(v).float().to(device)); return lg.softmax(1).cpu().numpy(),r.cpu().numpy(),q.cpu().numpy(),ats

def infer(root,ckpt,out,kind='missing'):
    device='cuda' if torch.cuda.is_available() else 'cpu'; m=MultiTask().to(device); m.load_state_dict(torch.load(ckpt,map_location=device)['model']); rows=[]
    base=Path(root)
    if kind=='missing': files=sorted((base/'附件3-模态缺失特征样本'/'对齐版本').glob('*.pkl'))
    else: files=sorted((base/'附件4-可解释专项视频样本与特征文件'/'附件4-可解释专项视频样本与特征文件'/'对齐版本').glob('*.pkl'))
    for f in files:
        with open(f,'rb') as h:d=pickle.load(h); d=d.get('test',d)
        if kind=='missing': d={k:np.asarray(v)[0] for k,v in d.items()}; t=d['text_bert'].astype('float32') if False else None
        else: d={k:v for k,v in d.items()}
        if 'text' not in d:
            # attachment 3 only provides text_bert; zero text embedding is the declared missing-modality input
            d['text']=np.zeros((50,768),dtype='float32')
        t,a,v=tensorize(d); prob,reg,gates,ats=predict(m,t,a,v,device); c=int(prob[0].argmax()); names=['Negative','Neutral','Positive']
        row={'file':f.name,'id':str(d.get('id',f.stem)),'prediction':names[c],'class_index':c,'confidence':float(prob[0,c]),'intensity':float(np.clip(reg[0],-3,3)),'text_weight':float(gates[0,0]),'audio_weight':float(gates[0,1]),'vision_weight':float(gates[0,2])}
        if kind=='explain':
            mods=['text','audio','vision']; best=int(gates[0].argmax()); row['main_modality']=mods[best]
            for j,n in enumerate(mods): row[n+'_key_position']=int(np.argmax(ats[j][0]))
            row['raw_text']=str(d.get('raw_text',''))
        rows.append(row)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0].keys()) if rows else ['file']); w.writeheader(); w.writerows(rows)
    print('saved',out,'rows',len(rows))

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train'); t.add_argument('--data-root',default=DEFAULT_DATA_ROOT); t.add_argument('--out',default='outputs/best.pt'); t.add_argument('--epochs',type=int,default=35); t.add_argument('--batch',type=int,default=32)
    i=sub.add_parser('infer'); i.add_argument('--data-root',default=DEFAULT_DATA_ROOT); i.add_argument('--ckpt',default='outputs/best.pt'); i.add_argument('--kind',choices=['missing','explain'],required=True); i.add_argument('--out',required=True)
    a=ap.parse_args();
    if a.cmd=='train': train(a.data_root,a.out,a.epochs,a.batch)
    else: infer(a.data_root,a.ckpt,a.out,a.kind)
if __name__=='__main__': main()
