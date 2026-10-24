import os,sys; os.environ['OMP_NUM_THREADS']='4'
from base import *
from nb1 import nb_feats
variant=sys.argv[1]; a=float(sys.argv[2]); name=sys.argv[3]
feats,df=load_train(); y=df.is_suspicious.values; w=train_weights(df)
tb=pd.read_parquet('test_base.parquet'); n=len(tb)
one=[(None,np.arange(n))]
bt=final_score(tb,one,tb,0.5,(2,1,0.5)); rbt=rank01(bt)
pt=pd.read_parquet('pay_test.parquet').set_index('h').loc[tb.TRANSACTION_HASH]
S=pt.s.values; R=pt.r.values; T=pt.t.values.astype(float)
cols=pd.read_parquet('nb_train.parquet').columns
def s2model(seed=42): return lgb.LGBMClassifier(objective='binary',learning_rate=0.02,n_estimators=400,num_leaves=7,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,random_state=seed,n_jobs=4,verbose=-1)
if variant=='all':
    Xtr=pd.read_parquet('nb_all_train.parquet')[cols]
    Xte=nb_feats(rbt,np.zeros(n,int),S,R,T); Xte['rk']=rbt; Xtes=[Xte[cols]]
    trs=[Xtr]
elif variant in ('fold_pf','pf_pf'):
    trs=[pd.read_parquet('nb_train.parquet')[cols]] if variant=='fold_pf' else [pd.read_parquet(f'nb_pf{s}_train.parquet')[cols] for s in range(3)]
    Xtes=[]; comp=tb._lab600u.values; u,inv=np.unique(comp,return_inverse=True)
    for sd in range(5):
        pf=np.random.RandomState(200+sd).randint(0,5,len(u))[inv]
        X=nb_feats(rbt,pf,S,R,T); X['rk']=rbt; Xtes.append(X[cols])
elif variant=='fold_all':
    trs=[pd.read_parquet('nb_train.parquet')[cols]]
    Xte=nb_feats(rbt,np.zeros(n,int),S,R,T); Xte['rk']=rbt; Xtes=[Xte[cols]]
s2=np.zeros(n)
for Xtr in trs:
    for sd in (1,2,3):
        m=s2model(sd).fit(Xtr.values.astype(float),y,sample_weight=w)
        for X in Xtes: s2+=m.predict_proba(X.values.astype(float))[:,1]
fin=(1-a)*rbt+a*rank01(s2)
fin=rank01(fin+1e-9*rbt)  # strict ranks -> no ties
fin=(pd.Series(fin).rank(method='first').values-0.5)/n
out=pd.DataFrame({'TRANSACTION_HASH':tb.TRANSACTION_HASH.values,'score':fin})
assert out.TRANSACTION_HASH.nunique()==40101 and out.score.nunique()==n and out.score.between(0,1).all()
os.makedirs('out',exist_ok=True); out.to_csv(f'out/{name}.csv',index=False)
base=pd.DataFrame({'TRANSACTION_HASH':tb.TRANSACTION_HASH.values,'score':(pd.Series(bt).rank(method='first').values-0.5)/n})
base.to_csv('out/base_repro.csv',index=False)
np.save(f's2_test_{name}.npy',s2); print('wrote',name,out.score.describe())
