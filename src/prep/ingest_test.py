"""Phase 1: test TSV -> parquet (ingest + row-count assert only; NO profiling)."""
import csv, pandas as pd
D = '/home/kr95/Projects/Amazon-ML/data/'
EXP = {'source1': 1732544, 'source2': 4887273, 'source3': 5082316}
for n, e in EXP.items():
    df = pd.read_csv(f'{D}test/test_{n}.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                     keep_default_na=False, na_filter=False, encoding='utf-8')
    assert len(df) == e, (n, len(df), e)
    assert list(df.columns) == ['entity_id', 'business_name', 'business_address', 'country'], df.columns
    df.to_parquet(f'{D}interim/test_{n}.parquet')
    print(n, len(df), flush=True)
