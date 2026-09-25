"""Mine most frequent leading/trailing token n-grams per country from TRAIN names only; write config/legal_suffixes.json.
Seeds (English/Indian + French) define the canonical forms; mined counts are stored for review."""
import json, sys, time, multiprocessing as mp
from collections import Counter
import pyarrow.parquet as pq
sys.path.insert(0, '/home/kr95/Projects/Amazon-ML')
from src.prep import text as T

ROOT = '/home/kr95/Projects/Amazon-ML/'
OUT = ROOT + 'config/legal_suffixes.json'
CANON = {  # canonical -> variants (token sequences after punctuation strip / casefold / translit)
    'private limited': ['pvt ltd', 'pvt limited', 'private ltd', 'private limited', 'p ltd', 'pvt', 'praaivet limited',
                        'praivet limited', 'praaivet lta', 'pri ltd', 'prv ltd', 'pvt ltd co', 'praa li', 'praaibhet limited', 'piraivet limitet',
                        'piraivet limited', 'praivarr limirrad', 'praivet limitet', 'praivet limirrad'],
    'limited': ['ltd', 'limited', 'ltd co', 'lta', 'limitd', 'limitet', 'limirrad'],
    'llc': ['llc', 'l l c', 'pllc', 'l c', 'elaelasee'],
    'llp': ['llp', 'l l p', 'elaelapee'],
    'inc': ['inc', 'incorporated', 'incorp'],
    'corp': ['corp', 'corporation'],
    'co': ['co', 'company'],
    'lp': ['lp', 'l p'],
    'plc': ['plc', 'public limited company', 'public limited'],
    'opc': ['opc', 'one person company'],
    'pc': ['pc', 'p c'],
    'llp2': [],
}
CANON.pop('llp2')
FRENCH = {'sarl': ['sarl', 's a r l'], 'sas': ['sas', 's a s'], 'sasu': ['sasu'], 'sa': ['sa', 's a'],
          'eurl': ['eurl'], 'sci': ['sci'], 'snc': ['snc'], 'sca': ['sca'], 'selarl': ['selarl'], 'scop': ['scop']}
DBA = ['dba', 'd b a', 'doing business as', 'trading as']
SEEDSET = {tuple(v.split()) for vs in CANON.values() for v in vs}
_T = None


def _work(rng):
    s, n = rng
    col = _T
    names, ctry = col[0][s:s + n], col[1][s:s + n]
    tr, ld = Counter(), Counter()
    for nm, c in zip(names, ctry):
        seq = T.field_tokens(nm)[2]
        L = len(seq)
        for k in (1, 2, 3):
            if L > k:      # need a non-empty remainder
                tr[(c, ' '.join(seq[-k:]))] += 1
                ld[(c, ' '.join(seq[:k]))] += 1
    return tr, ld


def main():
    global _T
    t0 = time.time()
    names, ctry = [], []
    for f, lim in [('source1', None), ('source2', 1_500_000), ('source3', 1_500_000)]:
        t = pq.read_table(ROOT + f'data/interim/{f}.parquet', columns=['business_name', 'country'])
        if lim: t = t.slice(0, lim)
        names += t['business_name'].to_pylist(); ctry += t['country'].to_pylist()
    n_tot = len(names); countries = sorted(set(ctry))
    _T = (names, ctry)
    chunks = [(s, 100_000) for s in range(0, n_tot, 100_000)]
    tr, ld = Counter(), Counter()
    with mp.get_context('fork').Pool(12) as p:
        for a, b in p.imap_unordered(_work, chunks):
            tr.update(a); ld.update(b)
    print('mined', n_tot, 'names in', round(time.time() - t0), 's; distinct trailing ngrams', len(tr), flush=True)
    per_c = Counter(ctry)
    mined = {}
    for c in countries:
        mined[c] = dict(n_names=per_c[c],
                        trailing_top={k[1]: v for k, v in tr.most_common() if k[0] == c}, leading_top={})
        mined[c]['trailing_top'] = dict(list(mined[c]['trailing_top'].items())[:80])
        mined[c]['leading_top'] = dict(list({k[1]: v for k, v in ld.most_common() if k[0] == c}.items())[:40])
        mined[c]['seed_trailing_counts'] = {k[1]: v for k, v in tr.items() if k[0] == c and tuple(k[1].split()) in SEEDSET}
    cfg = dict(canon=CANON, french_canon=FRENCH, dba_markers=DBA, train_countries=countries,
               note='Mined counts are trailing/leading 1-3-grams of the latinised token sequence over train S1 + first 1.5M of S2/S3. '
                    'Only the seed canon/variants are used for extraction; mined lists are for review. '
                    'French forms apply only to countries not in train_countries.',
               mined=mined)
    json.dump(cfg, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('wrote', OUT, round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
