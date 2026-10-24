import json, numpy as np, pandas as pd
from pathlib import Path
D=Path('/workspace/data')
XRP="0x1111222211112222111122221111222211112222"
def load(split):
    tx=pd.read_parquet(D/split/'transactions.parquet')
    tf=pd.read_parquet(D/split/'transfers.parquet')
    ex=tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    p=pd.DataFrame({'h':tx.TRANSACTION_HASH.values,'t':tx.TIMESTAMP.values.astype(np.int64),'s':tx.ACCOUNT.values,
        'r':ex.map(lambda e:e['Destination']).values,'fee':tx.SOURCE_FEE.values,'seq':tx.SEQUENCE.values,
        'ledger':tx.LEDGER_INDEX.values,'txi':tx.TRANSACTION_INDEX.values})
    p['has_sendmax']=ex.map(lambda e:'SendMax' in e).values
    p['has_paths']=ex.map(lambda e:'Paths' in e).values
    p['keys']=ex.map(lambda e:','.join(sorted(e.keys()))).values
    def dec(w):
        w=json.loads(w) if isinstance(w,str) else w
        if isinstance(w,(str,int)): return 'XRP', int(w)/1e6
        return w['issuer'][:6]+':'+w['currency'], float(w['value'])
    da=tx.DELIVERED_AMOUNT.map(dec); p['dasset']=[a for a,_ in da]; p['damt']=[b for _,b in da]
    sm=ex.map(lambda e:dec(e['SendMax']) if 'SendMax' in e else (None,np.nan))
    p['smasset']=[a for a,_ in sm]; p['smamt']=[b for _,b in sm]
    tf=tf.sort_values(['TRANSACTION_HASH','TRANSFER_INDEX'])
    tf['asset']=np.where(tf.TOKEN_ADDRESS==XRP,'XRP',tf.TOKEN_ADDRESS.str[:6]+':'+tf.SYMBOL)
    g=tf.groupby('TRANSACTION_HASH')
    first=g.first(); last=g.last()
    p['nlegs']=p.h.map(g.size())
    p['fasset']=p.h.map(first.asset); p['famt']=p.h.map(first.VALUE)
    p['f_to']=p.h.map(first.TO_ADDRESS)
    # rates
    mr=pd.read_csv(D/split/'market_rates.csv'); mr['a']=mr.ASSET.map(lambda x:'XRP' if x=='XRP' else x.split(':')[0][:6]+':'+x.split(':')[1])
    rate=dict(zip(zip(mr.TIMESTAMP,mr.a),mr.PRICE_MICRO_USD/1e6))
    hr=p.t//3600*3600
    p['dusd']=[a*rate[(h,s)] for a,h,s in zip(p.damt,hr,p.dasset)]
    p['fusd']=[a*rate[(h,s)] for a,h,s in zip(p.famt,hr,p.fasset)]
    p['smusd']=[a*rate[(h,s)] if isinstance(s,str) else np.nan for a,h,s in zip(p.smamt,hr,p.smasset)]
    p=p.reset_index(drop=True)
    if split=='train':
        l=pd.read_csv(D/split/'labels.csv'); p=p.merge(l,left_on='h',right_on='TRANSACTION_HASH',how='left').drop(columns='TRANSACTION_HASH')
    return p, tf
if __name__=='__main__':
    for s in ['train','test','validation']:
        p,tf=load(s); p.to_parquet(f'/workspace/fresh/pay_{s}.parquet'); tf.to_parquet(f'/workspace/fresh/tf_{s}.parquet'); print(s,p.shape)
