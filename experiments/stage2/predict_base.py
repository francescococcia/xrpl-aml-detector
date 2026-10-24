import os; os.environ['OMP_NUM_THREADS']='4'
from base import *
feats,df=load_train(); y=df.is_suspicious.values
cols=drop_groups(feature_columns(feats),['shift']); X=df[cols].values; w=train_weights(df)
te=pd.read_parquet(C+'features_v5_test.parquet'); Xt=te[cols].values
P={'lgb':lgbm().fit(X,y,sample_weight=w).predict_proba(Xt)[:,1],
   'hgb':hgb_model().fit(X,y,sample_weight=w).predict_proba(Xt)[:,1],
   'lr':lr().fit(X,y,logisticregression__sample_weight=w).predict_proba(Xt)[:,1]}
P=pd.DataFrame(P); P.insert(0,'TRANSACTION_HASH',te.TRANSACTION_HASH.values); P['_lab600u']=te._lab600u.values; P['_lab3600f']=te._lab3600f.values
P.to_parquet('test_base.parquet'); print(P.shape, P.TRANSACTION_HASH.nunique())
