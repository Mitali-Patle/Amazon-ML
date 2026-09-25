import pandas as pd, numpy as np
exec(open('src/eda/03_neardup_block.py').read().split("s1['nn']")[0].split("gt=pd.read_parquet(I+'gt.parquet')")[1]) if False else None
I='data/interim/'
s1=pd.read_parquet(I+'source1.parquet');gt=pd.read_parquet(I+'gt.parquet')
print('s1 exact address dup rows',s1.duplicated('business_address',keep=False).sum(),'addr+country',s1.duplicated(['business_address','country'],keep=False).sum())
matched=set(gt.matched_entity_ids[gt.matched_entity_ids!=''].str.split(',').explode())
for n in ['source2','source3']:
    d=pd.read_parquet(I+n+'.parquet'); d['m']=d.entity_id.isin(matched)
    print(n,'matched',d.m.mean())
    print(' by country unmatched',d.groupby('country').m.apply(lambda x:1-x.mean()).to_dict())
    print(' empty addr rate matched/unmatched',d.groupby('m').business_address.apply(lambda x:(x=='').mean()).to_dict())
    print(' exact dup(name,addr,country) rows in matched vs unmatched',d[d.duplicated(['business_name','business_address','country'],keep=False)].m.mean())
    print(' unmatched ids overlap exact (name,addr) with s1 anchor:',d[~d.m].merge(s1[['business_name','business_address']],on=['business_name','business_address']).shape[0],'of',(~d.m).sum(),'; matched:',d[d.m].merge(s1[['business_name','business_address']],on=['business_name','business_address']).shape[0])
    print(' non-latin name rate matched/unmatched',d.groupby('m').business_name.apply(lambda x:x.str.contains(r'[^\x00-ɏ]').mean()).to_dict())
    print(' id numeric range',d.entity_id.str[3:].astype(np.int64).describe()[['min','max']].tolist())
print('gt len in match-list order: are ids sorted?',gt[gt.matched_entity_ids!=''].matched_entity_ids.head(3).tolist())
# S1 matched rows without S3 or S2 by country handled earlier; nomatch vs name features
s1=s1.merge(gt,left_on='entity_id',right_on='source1_entity_id'); s1['nm']=s1.matched_entity_ids==''
print('no-match rate by addr has number',s1.groupby(s1.business_address.str.contains(r'\d'))['nm'].mean().to_dict())
print('no-match rate by name len bucket',s1.groupby(pd.qcut(s1.business_name.str.len(),5,duplicates='drop'))['nm'].mean().to_dict())
print('S1 id numeric order vs gt row order same:',(s1.entity_id==s1.source1_entity_id).all())
