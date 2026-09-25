"""V5 model behavior, token interaction, bias consistency, and recovery."""
import os,sys,copy,unittest,uuid,shutil
if sys.platform=='win32': os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from transformers import BertConfig,BertModel
from q3v5_model import TokenInteractionModel,TunedModel,candidates
from q3v5_explain import AdaptivePredictor
from q3v4_decision import adjust,select,crossfit,macro


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)
        self.base=BertModel(BertConfig(vocab_size=200,hidden_size=24,num_hidden_layers=2,
            num_attention_heads=4,intermediate_size=32,_attn_implementation='eager'))
        self.b=torch.zeros(3,3,8,dtype=torch.long)
        self.b[:,0]=torch.tensor([101,11,12,13,14,102,0,0]); self.b[:,1,:6]=1
        self.mask=torch.ones(3,8,3,dtype=torch.bool); self.mask[:,:,0]=False; self.mask[:,1:5,0]=True; self.mask[:,6:]=False
        self.a=torch.randn(3,8,74); self.v=torch.randn(3,8,35)

    def config(self,name):
        c=candidates(2026)[name]
        return dict(c,lora_layers=min(c['lora_layers'],2),full_layers=min(c['full_layers'],1),dropout=0.)

    def test_all_candidates_gradient_and_reload(self):
        for name in candidates(2026):
            with self.subTest(candidate=name):
                cfg=self.config(name); model=TunedModel(copy.deepcopy(self.base),cfg,[.2,.3,.5],.2)
                model.train(); l,r=model(self.b,self.a,self.v,self.mask)
                (torch.nn.functional.cross_entropy(l,torch.arange(3))+.2*r.square().mean()).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                learned=[p for n,p in model.named_parameters() if n.startswith('bert.') and p.requires_grad]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in learned))
                clone=TunedModel(copy.deepcopy(self.base),cfg,[1/3]*3,0).eval(); clone.load_portable(model.portable_state())
                model.eval()
                for x,y in zip(model(self.b,self.a,self.v,self.mask),clone(self.b,self.a,self.v,self.mask)): torch.testing.assert_close(x,y)

    def test_interaction_masking_bias_and_empty(self):
        model=TunedModel(copy.deepcopy(self.base),self.config('token_interaction'),[.2,.3,.5],.2).eval()
        mask=self.mask.clone(); mask[:,2,0]=False
        inp=self.b.clone(); inp[:,0,2]=199
        aa=self.a.clone(); aa[~mask[:,:,1]]=1000
        for x,y in zip(model(self.b,self.a,self.v,mask),model(inp,aa,self.v,mask)): torch.testing.assert_close(x,y)
        data={'b':self.b.numpy(),'a':self.a.numpy(),'v':self.v.numpy(),'mask':self.mask.numpy()}
        masks=np.repeat(data['mask'][:1],2,0); masks[1]=False
        bias=[-.1,.2,0.]
        predictor=AdaptivePredictor([(model,1.)],'cpu',bias)
        log,r,p=predictor.masks(data,0,masks)
        with torch.inference_mode():
            raw,reg=model(self.b[:1],self.a[:1],self.v[:1],self.mask[:1])
        np.testing.assert_allclose(p[:1],adjust(raw.softmax(-1).numpy(),bias),atol=1e-6)
        np.testing.assert_allclose(p[1],adjust(np.array([[.2,.3,.5]]),bias)[0],atol=1e-6)
        self.assertAlmostEqual(r[1],.2,places=6)
        np.testing.assert_allclose(np.exp(log),p,atol=1e-7)

    def test_bias_group_crossfit_and_noop(self):
        y=np.tile(np.arange(3,dtype=np.int64),20); p=np.eye(3)[y]*.6+.4/3
        np.testing.assert_allclose(select(p,y),np.zeros(3),atol=1e-10)
        ids=[f'video{i//3}$_${i%3}' for i in range(60)]
        check=crossfit(p,y,ids)
        self.assertEqual(check['crossfit_macro_f1'],1.)
        self.assertEqual(len(check['folds']),5)

    def test_resume_matches_continuous(self):
        import q3v5_train as training
        data={'b':self.b.numpy(),'a':self.a.numpy(),'v':self.v.numpy(),'mask':self.mask.numpy(),
              'y':np.arange(3,dtype=np.int64),'r':np.array([-.5,0,.8],np.float32),'ids':['a','b','c'],'text_mapping':[False]*3}
        cfg=dict(self.config('v3_control'),dropout=.2)
        def factory(path,config,prior,mean,device,verify=False):
            return TunedModel(copy.deepcopy(self.base),config,prior,mean).to(device),None
        save=training.save_atomic
        def stop(path,obj):
            save(path,obj)
            if path.parent.name=='interrupted' and path.name=='last.pt': raise InterruptedError('simulated')
        parent=(Path(__file__).resolve().parents[1]/'work').resolve(); folder=parent/('q3v5_test_'+uuid.uuid4().hex)
        folder.mkdir(parents=True)
        try:
            args=SimpleNamespace(out=folder,bert='unused',device='cpu',batch_size=2,epochs=2,patience=4)
            with patch.object(training,'fresh_model',side_effect=factory):
                _,pp,rr=training.train_one(args,data,data,{},cfg,'continuous','test')
                with patch.object(training,'save_atomic',side_effect=stop):
                    with self.assertRaises(InterruptedError): training.train_one(args,data,data,{},cfg,'interrupted','test')
                _,p,r=training.train_one(args,data,data,{},cfg,'interrupted','test')
            np.testing.assert_allclose(p,pp,atol=1e-7); np.testing.assert_allclose(r,rr,atol=1e-7)
        finally:
            if folder.resolve().is_relative_to(parent) and folder.name.startswith('q3v5_test_'): shutil.rmtree(folder)


if __name__=='__main__':
    torch.set_num_threads(2); unittest.main()
