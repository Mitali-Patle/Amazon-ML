"""Alignment/integrity asserts on prep outputs (per-record equality only; no statistics on test)."""
import pyarrow.parquet as pq, pyarrow.compute as pc
R = '/home/kr95/Projects/Amazon-ML/data/'
for sp, pre in (('train', ''), ('test', 'test_')):
    for k in (1, 2, 3):
        a = pq.read_table(R + f'interim/{pre}source{k}.parquet')
        b = pq.read_table(R + f'processed/prep/{sp}_source{k}.parquet')
        assert a.num_rows == b.num_rows
        for c in ('entity_id', 'business_name', 'business_address', 'country'):
            import pyarrow as pa; assert a[c].cast(pa.large_string()).equals(b[c].cast(pa.large_string())), (sp, k, c, a[c].type, b[c].type)
        assert b.schema.names[:4] == ['entity_id', 'country', 'business_name', 'business_address'] and b.num_columns == 17, b.schema.names
        nz = pc.sum(pc.equal(pc.utf8_length(b['name_core']), 0)).as_py()
        ne = pc.sum(pc.equal(pc.utf8_length(b['name_norm']), 0)).as_py()
        assert nz == ne, (sp, k, nz, ne)   # core empty only if the name itself normalises to empty
        print(sp, k, 'ok raw preserved + aligned; core-empty == name-empty:', nz)
