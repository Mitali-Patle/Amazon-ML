import pandas as pd, numpy as np, re, unicodedata
I='data/interim/'
s1=pd.read_parquet(I+'source1.parquet');gt=pd.read_parquet(I+'gt.parquet')
SUF=r'\b(private limited|pvt ltd|pvt|ltd|limited|llc|llp|inc|corp|corporation|co|company|l l c|p c|pc|dba)\b'
def nname(s):
    s=s.str.normalize('NFKD').str.replace(r'[̀-ͯ]','',regex=True).str.lower()
    s=s.str.replace(r'[^a-z0-9ऀ-ॿ ]',' ',regex=True).str.replace(SUF,' ',regex=True)
    return s.str.replace(r'\s+',' ',regex=True).str.strip()
def naddr(s): return s.str.lower().str.replace(r'[^a-z0-9 ]',' ',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
s1['nn']=nname(s1.business_name); s1['na']=naddr(s1.business_address)
print('== Q2 near-dup source1 (full)')
print('exact norm name dup rows',s1.duplicated('nn',keep=False).mean(),'nn+country',s1.duplicated(['nn','country'],keep=False).mean())
print('norm name+addr dup rows',s1.duplicated(['nn','na'],keep=False).sum())
s1['num']=s1.business_address.str.extract(r'^\D*(\d+)')[0]
s1['city']=s1.business_address.str.split(',').str[-2].str.strip().str.lower()
k=['nn','country','num']
print('nn+leading street number dup rows',s1.duplicated(k,keep=False).sum(), 'excluding no-number',s1[s1.num.notna()].duplicated(k,keep=False).sum())
print('nn+city+country dup rows',s1.duplicated(['nn','city','country'],keep=False).sum())
# chain-like: names repeated >=5
vc=s1.nn.value_counts(); print('names repeated>=5:',(vc>=5).sum(),'rows',vc[vc>=5].sum(),'top\n',vc.head(10))
# do same-name same-city pairs share targets? (already disjoint by graph) -> look at a few
d=s1[s1.duplicated(['nn','city','country'],keep=False)].sort_values(['nn','city']).head(24); print(d[['business_name','business_address','country']].to_string())
# fuzzy: block on nn first 10 chars + first token of address, check pair address similarity? skip
print('== Q5 blocking probe (sample 20k anchors, full S2/S3 pool)')
rng=np.random.RandomState(0)
gts=gt[gt.matched_entity_ids!=''].sample(20000,random_state=0)
a=s1.set_index('entity_id').loc[gts.source1_entity_id]
e=gts.assign(t=gts.matched_entity_ids.str.split(',')).explode('t')[['source1_entity_id','t']]
pool=pd.concat([pd.read_parquet(I+'source2.parquet'),pd.read_parquet(I+'source3.parquet')],ignore_index=True)
pool['nn']=nname(pool.business_name)
pi=pool.set_index('entity_id')
e['ac']=a.country.reindex(e.source1_entity_id).values; e['tc']=pi.country.reindex(e.t).values
print('true match country agrees with anchor',(e.ac==e.tc).mean())
ae=a.reset_index()
e=e.merge(ae[['entity_id','nn','na','business_name']].rename(columns={'entity_id':'source1_entity_id'}),on='source1_entity_id')
e['tn']=pi.nn.reindex(e.t).values
e['tsrc']=e.t.str[:2]
print('true pair normalized-name equal',(e.nn==e.tn).mean(),'by src',e.groupby('tsrc').apply(lambda x:(x.nn==x.tn).mean()).to_dict())
for L in [3,4,6,8]:
    print('first',L,'chars equal',(e.nn.str[:L]==e.tn.str[:L]).mean())
tokset=lambda s:set(s.split())
import collections
# token df in pool
df=collections.Counter(t for n in pool.nn.sample(1000000,random_state=0) for t in set(n.split()))  # scaled to 1M sample
for cap in [50,500,5000,10**9]:
    ok=[]
    for x,y in zip(e.nn.values[:60000],e.tn.values[:60000]):
        sh=set(x.split())&set(y.split())
        ok.append(any(df.get(t,0)<=cap for t in sh))
    print('shares a token with df<=%s (per 1M):'%cap,np.mean(ok))
# recall by token overlap Jaccard buckets
j=[len(tokset(x)&tokset(y))/max(1,len(tokset(x)|tokset(y))) for x,y in zip(e.nn,e.tn)]
print('token jaccard quantiles',np.quantile(j,[.05,.1,.25,.5,.75]),' zero-overlap frac',np.mean(np.array(j)==0))
print('zero token-overlap examples');z=e[np.array(j)==0].head(15);print(z[['business_name','tn','nn']].to_string())
# candidate set size for 4-char-prefix+country key
pool['k4']=pool.country+'|'+pool.nn.str[:4]
sz=pool.k4.value_counts()
e['k4']=e.ac+'|'+e.nn.str[:4]
print('k4 block size for anchors: median',sz.reindex(e.k4.unique()).median(),'mean',sz.reindex(e.k4.unique()).mean(),'p95',sz.reindex(e.k4.unique()).quantile(.95))
# addr numeric token overlap
def nums(s): return set(re.findall(r'\d+',s))
pa=pi.business_address.reindex(e.t).values; aa=a.business_address.reindex(e.source1_entity_id).values
e['numov']=[bool(nums(x)&nums(y)) if nums(x) else np.nan for x,y in zip(aa,pa)]
print('address number overlap among true pairs (where anchor has number)',e.numov.mean())
print(e.groupby('ac').numov.mean())
print('sample true pairs');print(e.sample(15,random_state=2).assign(ta=lambda d:pi.business_address.reindex(d.t).values,aa=lambda d:a.business_address.reindex(d.source1_entity_id).values,tb=lambda d:pi.business_name.reindex(d.t).values)[['business_name','tb','aa','ta']].to_string())
