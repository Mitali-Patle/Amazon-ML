"""Phase 2: structural audit on TRAIN only. Fail-fast."""
import pandas as pd
I = '/home/kr95/Projects/Amazon-ML/data/interim/'
EXP = dict(source1=2206821, source2=5034616, source3=5285603, gt=2206821)
d = {n: pd.read_parquet(I + f'{n}.parquet') for n in EXP}
for n, e in EXP.items():
    assert len(d[n]) == e, (n, len(d[n]), e); print('rows', n, e)
for n, p in [('source1', 'S1-'), ('source2', 'S2-'), ('source3', 'S3-')]:
    assert d[n].entity_id.str.startswith(p).all(); assert d[n].entity_id.is_unique; print('prefix ok', n)
s1, gt = d['source1'], d['gt']
assert set(s1.entity_id) == set(gt.source1_entity_id) and gt.source1_entity_id.is_unique; print('GT<->S1 ids identical')
m = gt[gt.matched_entity_ids.str.strip() != ''].copy()
m['mid'] = m.matched_entity_ids.str.split(',')
e = m[['source1_entity_id', 'mid']].explode('mid'); e['mid'] = e.mid.str.strip()
assert not e.mid.str.startswith('S1-').any(), 'S1 self-match'
assert e.mid.str.startswith(('S2-', 'S3-')).all()
pool = pd.concat([d['source2'][['entity_id', 'country']], d['source3'][['entity_id', 'country']]])
assert pool.entity_id.is_unique
assert e.mid.isin(set(pool.entity_id)).all(), 'unresolved matched id'
assert len(e) == e.mid.nunique() == 7638365, (len(e), e.mid.nunique()); print('many-to-one ok', len(e))
c1 = s1.set_index('entity_id').country
e['c1'] = e.source1_entity_id.map(c1); e['c2'] = e.mid.map(pool.set_index('entity_id').country)
assert (e.c1 == e.c2).all(), 'country disagreement'; print('country agreement ok')
print('countries S1:', s1.country.value_counts().to_dict())
print('AUDIT PASS')
