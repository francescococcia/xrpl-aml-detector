import numpy as np, pandas as pd
from common import rank01
def nb_feats(score, fold, S,R,T, wins=(600,1800,7200)):
    # same-fold neighbours sharing a wallet within +-W
    n=len(score); out={}
    rk=np.zeros(n)
    for f in np.unique(fold):
        m=fold==f; rk[m]=rank01(score[m])
    ev=pd.DataFrame({'w':np.r_[S,R],'i':np.r_[np.arange(n),np.arange(n)],'t':np.r_[T,T],'f':np.r_[fold,fold]}).sort_values(['f','w','t'])
    for W in wins:
        mx=np.zeros(n); mn=np.zeros(n); cnt=np.zeros(n)
        for (f,w),g in ev.groupby(['f','w'],sort=False):
            tt=g.t.values; ii=g.i.values; r=rk[ii]
            lo=np.searchsorted(tt,tt-W,'left'); hi=np.searchsorted(tt,tt+W,'right')
            for k in range(len(ii)):
                sl=[j for j in range(lo[k],hi[k]) if j!=k]
                if sl:
                    v=r[sl]; i=ii[k]; mx[i]=max(mx[i],v.max()); cnt[i]+=len(sl); mn[i]+=v.sum()
        out[f'nb{W}_max']=mx; out[f'nb{W}_cnt']=cnt; out[f'nb{W}_mean']=np.where(cnt>0,mn/np.maximum(cnt,1),0)
    out['rk']=rk
    return pd.DataFrame(out)
