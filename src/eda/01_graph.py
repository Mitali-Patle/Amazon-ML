import pandas as pd, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
I='data/interim/'
s1=pd.read_parquet(I+'source1.parquet');s2=pd.read_parquet(I+'source2.parquet');s3=pd.read_parquet(I+'source3.parquet');gt=pd.read_parquet(I+'gt.parquet')
print('gt dup s1 ids',gt.source1_entity_id.duplicated().sum(),'s1 dup ids',s1.entity_id.duplicated().sum(),'s2 dup',s2.entity_id.duplicated().sum(),'s3 dup',s3.entity_id.duplicated().sum())
print('gt s1 ids not in s1',(~gt.source1_entity_id.isin(s1.entity_id)).sum(),'s1 not in gt',(~s1.entity_id.isin(gt.source1_entity_id)).sum())
gt['m']=gt.matched_entity_ids.str.split(',')
gt['n']=gt.m.map(len); gt.loc[gt.matched_entity_ids=='','n']=0
gt['n2']=gt.matched_entity_ids.str.count('S2-'); gt['n3']=gt.matched_entity_ids.str.count('S3-')
print('empty',(gt.n==0).sum(),(gt.n==0).mean())
print('len dist\n',gt.n.value_counts().sort_index().head(30))
print('n2 dist\n',gt.n2.value_counts().sort_index().head(15));print('n3 dist\n',gt.n3.value_counts().sort_index().head(15))
print('mix crosstab (n2>0,n3>0)\n',pd.crosstab(gt.n2>0,gt.n3>0))
print('n2 vs n3 (top)\n',pd.crosstab(gt.n2.clip(upper=6),gt.n3.clip(upper=6)))
e=gt[gt.n>0][['source1_entity_id','m']].explode('m').rename(columns={'m':'t'})
print('pairs',len(e),'unique targets',e.t.nunique(),'dup pairs',e.duplicated().sum())
cnt=e.groupby('t').source1_entity_id.nunique()
print('anchors per target\n',cnt.value_counts().sort_index().head(10))
print('targets shared by >1 anchor',(cnt>1).sum())
s2ids=set(s2.entity_id);s3ids=set(s3.entity_id)
e['src']=e.t.str[:2]
print('bad prefix',(~e.src.isin(['S2','S3'])).sum())
print('S2 targets missing from s2',(~e[e.src=='S2'].t.isin(s2ids)).sum(),'S3 missing',(~e[e.src=='S3'].t.isin(s3ids)).sum())
print('s2 rows matched',e[e.src=='S2'].t.nunique(),'/',len(s2),'s3',e[e.src=='S3'].t.nunique(),'/',len(s3))
# components
ids=pd.Index(pd.concat([e.source1_entity_id,e.t]).unique()); a=ids.get_indexer(e.source1_entity_id);b=ids.get_indexer(e.t)
g=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(ids),)*2)
nc,lab=connected_components(g,directed=False)
sz=np.bincount(lab); print('components',nc,'size dist\n',pd.Series(sz).value_counts().sort_index().head(25),'max',sz.max())
# component with >1 s1 anchor
s1flag=np.array([i.startswith('S1-') for i in ids]); n1=np.bincount(lab,weights=s1flag)
print('components with >1 S1 anchor',(n1>1).sum(),'anchors in them',n1[n1>1].sum(), 'max anchors',n1.max())
print('cluster composition (n_s2,n_s3) top')
s2f=np.array([i.startswith('S2-') for i in ids]);c2=np.bincount(lab,weights=s2f);c3=sz-n1-c2
print(pd.Series(list(zip(n1.astype(int),c2.astype(int).clip(max=6),c3.astype(int).clip(max=6)))).value_counts().head(15))
# unmatched pool
print('S2 unmatched frac',1-e[e.src=="S2"].t.nunique()/len(s2),'S3',1-e[e.src=="S3"].t.nunique()/len(s3))
gt[['source1_entity_id','n','n2','n3']].to_parquet(I+'gt_counts.parquet')
