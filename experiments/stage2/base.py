import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'/workspace/fresh/src_copy'); sys.path.insert(0,'/workspace/fresh/starter_copy')
os.environ.setdefault('OMP_NUM_THREADS','4')
from common import rank01, smooth_scores, train_weights, drop_groups, hgb_model
from features import feature_columns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score as AP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
C='/workspace/final/cache/'
def load_train():
    feats=pd.read_parquet(C+'features_v5_train.parquet')
    labels=pd.read_csv('/workspace/data/train/labels.csv')
    df=feats.merge(labels,on='TRANSACTION_HASH',how='left').reset_index(drop=True)
    return feats, df
def folds_of(df):
    y=df.is_suspicious.values
    groups=df.scenario_id.astype(object).where(df.scenario_id.notna(),"u_"+df.index.astype(str)).values
    return list(GroupKFold(n_splits=5,shuffle=True,random_state=42).split(df,y,groups))
def lgbm(seed=42): return lgb.LGBMClassifier(objective='binary',learning_rate=0.015,n_estimators=1200,num_leaves=15,min_child_samples=50,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=1.0,random_state=seed,n_jobs=4,verbose=-1)
def lr(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.1,random_state=42))
def cv_ap(y,s,folds): return np.array([100*AP(y[va],s[va]) for _,va in folds])
def final_score(df,folds,oof,lam=0.5,w=(2,1,0.5)):
    out=np.zeros(len(df)); labs=[df._lab600u.values,df._lab3600f.values]
    for _,va in folds:
        b=(w[0]*rank01(oof['lgb'][va])+w[1]*rank01(oof['hgb'][va])+w[2]*rank01(oof['lr'][va]))/sum(w)
        sm=smooth_scores(b,[l[va] for l in labs],lam=lam)
        out[va]=(sm+1e-6*rank01(oof['lgb'][va]))/(1+2*lam+1e-6)
    return out
if __name__=='__main__':
    feats,df=load_train(); y=df.is_suspicious.values; folds=folds_of(df)
    cols=drop_groups(feature_columns(feats),['shift']); print(len(cols))
    X=df[cols].values; w=train_weights(df)
    oof={k:np.zeros(len(df)) for k in ['lgb','hgb','lr']}
    for i,(tr,va) in enumerate(folds):
        oof['lgb'][va]=lgbm().fit(X[tr],y[tr],sample_weight=w[tr]).predict_proba(X[va])[:,1]
        oof['hgb'][va]=hgb_model().fit(X[tr],y[tr],sample_weight=w[tr]).predict_proba(X[va])[:,1]
        oof['lr'][va]=lr().fit(X[tr],y[tr],logisticregression__sample_weight=w[tr]).predict_proba(X[va])[:,1]
        print('fold',i,flush=True)
    pd.DataFrame(oof).to_parquet('oof_base.parquet')
    for k in oof: print(k,cv_ap(y,oof[k],folds).mean().round(2))
    for lam,wt in [(0.25,(2,1,1)),(0.5,(2,1,0.5)),(1.0,(2,1,0.5))]:
        s=final_score(df,folds,oof,lam,wt); a=cv_ap(y,s,folds); print(lam,wt,a.mean().round(2),a.round(2))
    np.save('folds.npy',np.array([np.isin(np.arange(len(df)),va) for _,va in folds]).argmax(0))
