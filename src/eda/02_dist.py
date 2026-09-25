import pandas as pd, numpy as np, re
I='data/interim/'
S={n:pd.read_parquet(I+f'{n}.parquet') for n in ['source1','source2','source3']}
gc=pd.read_parquet(I+'gt_counts.parquet')
for n,d in S.items():
    print('==',n); print(d.country.value_counts(dropna=False).head(10))
    nm=d.business_name; ad=d.business_address
    print('empty name',(nm.str.strip()=='').sum(),'empty addr',(ad.str.strip()=='').sum(),'name len<3',(nm.str.len()<3).sum())
    print('name len q',nm.str.len().quantile([.01,.5,.99]).tolist(),'addr len q',ad.str.len().quantile([.01,.5,.99]).tolist())
    dev=nm.str.contains(r'[ऀ-ॿ]');print('devanagari name',dev.mean(),'addr',ad.str.contains(r'[ऀ-ॿ]').mean())
    nonl=nm.str.contains(r'[^\x00-ɏ]');print('non-latin name',nonl.mean())
    print('all-upper name',(nm.str.isupper()).mean(),'all-lower',nm.str.islower().mean(),'name has .com',nm.str.contains(r'\.(com|net|org|in)\b',case=False).mean())
    print('suffix inc',nm.str.contains(r'\b(inc|llc|ltd|corp)\b\.?$',case=False).mean(),'pvt ltd',nm.str.contains(r'private limited|pvt\.? ?ltd',case=False).mean())
    print('addr comma count mean',ad.str.count(',').mean(),'ends with 2-letter state',ad.str.contains(r', [A-Z]{2}$').mean(),'has zip',ad.str.contains(r'\b\d{5}\b').mean(),'has pin6',ad.str.contains(r'\b\d{6}\b').mean())
    print('exact dup (name,addr)',d.duplicated(['business_name','business_address']).sum(),'exact dup name+addr+country',d.duplicated(['business_name','business_address','country']).sum(),'dup name',d.duplicated('business_name').sum())
    for c in d.country.value_counts().index[:3]:
        print(c); print(d[d.country==c].sample(6,random_state=1)[['business_name','business_address']].to_string())
s1=S['source1'].merge(gc,left_on='entity_id',right_on='source1_entity_id')
g=s1.groupby('country').agg(n=('n','size'),nomatch=('n',lambda x:(x==0).mean()),mean_n=('n','mean'),mean_n2=('n2','mean'),mean_n3=('n3','mean'),p_has2=('n2',lambda x:(x>0).mean()),p_has3=('n3',lambda x:(x>0).mean()))
print(g.sort_values('n',ascending=False).head(15))
print('overall nomatch by n2>0/n3>0');print(pd.crosstab(s1.country,[s1.n2>0,s1.n3>0]).head(10))
print('mean n | matched',s1[s1.n>0].groupby('country').n.mean())
