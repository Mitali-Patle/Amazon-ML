import pandas as pd, csv, time
D='/home/kr95/Projects/Amazon-ML/data/'
for n in ['source1','source2','source3']:
    t=time.time()
    df=pd.read_csv(f'{D}train/train_{n}.tsv',sep='\t',quoting=csv.QUOTE_NONE,dtype=str,keep_default_na=False,encoding='utf-8')
    print(n,df.shape,time.time()-t, df.columns.tolist())
    df.to_parquet(f'{D}interim/{n}.parquet')
gt=pd.read_csv(f'{D}train/train_ground_truth.tsv',sep='\t',quoting=csv.QUOTE_NONE,dtype=str,keep_default_na=False)
print(gt.shape); gt.to_parquet(f'{D}interim/gt.parquet')
