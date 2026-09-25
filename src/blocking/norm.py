"""Shared normalization + cached pool prep. Train data only."""
import re, os, numpy as np, pandas as pd
I = '/home/kr95/Projects/Amazon-ML/data/interim/'
SUF = r'\b(?:private limited|pvt ltd|pvt|ltd|limited|llc|llp|inc|corp|corporation|co|company|l l c|p c|pc|dba)\b'

def nname(s):
    s = s.str.normalize('NFKD').str.replace(r'[̀-ͯ]', '', regex=True).str.lower()
    s = s.str.replace(r'[^\p{L}\p{N}\p{M} ]', ' ', regex=True).str.replace(SUF, ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def naddr(s):
    """light address normalization for embedding text: NFKD accent strip (Latin marks only), lowercase, drop punctuation/NULL tokens. Script kept."""
    s = s.str.normalize('NFKD').str.replace(r'[̀-ͯ]', '', regex=True).str.lower()
    s = s.str.replace(r'[^\p{L}\p{N}\p{M} ]', ' ', regex=True).str.replace(r'\bnull\b', ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def embed_text(name_raw, addr_raw):
    """'name | address' string for dense encoders (normalized, original script kept, no transliteration)."""
    return (nname(name_raw) + ' | ' + naddr(addr_raw)).values

def addr_nums(s):
    """space-joined numeric tokens of an address (order-free), e.g. '1795'."""
    return s.str.findall(r'\d+').str.join(' ')

def script(s):
    """'nonlatin' if any code point > U+024F in the raw name else 'latin'."""
    return np.where(s.str.contains(r'[^\x{0000}-\x{024F}]', regex=True), 'nonlatin', 'latin')

def prep(df):
    o = pd.DataFrame({'id': df.entity_id.values, 'country': df.country.values})
    o['nn'] = nname(df.business_name).values
    o['nums'] = addr_nums(df.business_address).values
    o['script'] = script(df.business_name)
    return o

def load_pool(force=False):
    """S2+S3 prepped pool (cached). Columns: id,country,nn,nums,script. Index = row position."""
    f = I + 'pool_norm.parquet'
    if os.path.exists(f) and not force:
        return pd.read_parquet(f)
    parts = []
    for n in ['source2', 'source3']:
        parts.append(prep(pd.read_parquet(I + f'{n}.parquet')))
    p = pd.concat(parts, ignore_index=True); p.to_parquet(f); return p

def load_anchors(): 
    return prep(pd.read_parquet(I + 'source1.parquet'))
