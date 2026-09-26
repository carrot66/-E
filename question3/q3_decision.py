"""Small declared decision grid; video-group cross-fitting is a sensitivity check only."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def adjust(prob, bias):
    z=np.log(np.clip(prob,1e-12,1))+np.asarray(bias)
    z-=z.max(-1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(-1,keepdims=True)


def macro(y,pred):
    cm=np.bincount(3*np.asarray(y)+np.asarray(pred),minlength=9).reshape(3,3)
    return float(np.mean(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)))


def bias_grid():
    # Positive is the reference. No per-sample or text-dependent correction.
    return [np.array([a,b,0.]) for a in np.linspace(-.4,.4,9) for b in np.linspace(-.4,.4,9)]


def select(prob,y):
    choices=[(macro(y,adjust(prob,b).argmax(1)),-float(b@b),b) for b in bias_grid()]
    return max(choices,key=lambda t:t[:2])[2]


def crossfit(prob,y,ids):
    groups=np.array([s.split('$_$')[0] for s in ids]); result=np.zeros_like(prob)
    folds=[]
    for fold,(fit,hold) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=4102).split(prob,y,groups)):
        assert not set(groups[fit])&set(groups[hold])
        bias=select(prob[fit],y[fit]); result[hold]=adjust(prob[hold],bias)
        folds.append({'fold':fold,'fit':len(fit),'hold':len(hold),'bias':bias.tolist(),
                      'raw_F1':macro(y[hold],prob[hold].argmax(1)),
                      'adjusted_F1':macro(y[hold],result[hold].argmax(1))})
    return {'raw_macro_f1':macro(y,prob.argmax(1)), 'crossfit_macro_f1':macro(y,result.argmax(1)),
            'folds':folds,'note':'Base model epochs/structure already selected on this validation set; NOT independent model CV.'}
