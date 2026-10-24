import os; os.environ['OMP_NUM_THREADS']='4'
from base import *
from nb1 import nb_feats
feats,df=load_train(); y=df.is_suspicious.values; folds=folds_of(df)
fold=np.zeros(len(df),int)
for i,(_,va) in enumerate(folds): fold[va]=i
base=pd.read_parquet('oof_base.parquet'); bs=final_score(df,folds,base,0.5,(2,1,0.5))
rb=np.zeros(len(df))
for _,va in folds: rb[va]=rank01(bs[va])
p=pd.read_parquet('pay_train.parquet').set_index('h').loc[df.TRANSACTION_HASH]
S=p.s.values; R=p.r.values; T=p.t.values.astype(float)
# all-rows neighbours (test-like density): single pseudo fold; per-fold rank used as score
NBa=nb_feats(rb,np.zeros(len(df),int),S,R,T); NBa['rk']=rb; NBa.to_parquet('nb_all_train.parquet')
NBs=pd.read_parquet('nb_train.parquet')
w=train_weights(df)
def s2model(seed=42): return lgb.LGBMClassifier(objective='binary',learning_rate=0.02,n_estimators=400,num_leaves=7,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,random_state=seed,n_jobs=4,verbose=-1)
def run(Xdf,tag,seeds=(42,)):
    X=Xdf.values.astype(float); s2=np.zeros(len(df))
    for tr,va in folds:
        for sd in seeds: s2[va]+=s2model(sd).fit(X[tr],y[tr],sample_weight=w[tr]).predict_proba(X[va])[:,1]
    res={'s2only':cv_ap(y,s2,folds).mean().round(2)}
    for a in [0.2,0.3,0.4,0.5]:
        fin=np.zeros(len(df))
        for _,va in folds: fin[va]=(1-a)*rank01(bs[va])+a*rank01(s2[va])
        res[a]=cv_ap(y,fin,folds).mean().round(2)
        if a==0.3: res['folds']=cv_ap(y,fin,folds).round(2).tolist()
    print(tag,res,flush=True); np.save(f'oof_s2_{tag}.npy',s2); return s2
run(NBa,'nb_all')
run(NBs,'nb_fold_3seeds',seeds=(1,2,3))
