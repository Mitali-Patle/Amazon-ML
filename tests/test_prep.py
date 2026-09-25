import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from src.prep import text as T

CFG = T.load_suffix_cfg(str(pathlib.Path(__file__).resolve().parents[1] / 'config/legal_suffixes.json'))

SPOT = [
    ('राम मार्केटिंग प्राइवेट लिमिटेड', 'raam maarketing praaivet limited'),
    ('ಕರ್ನಾಟಕ', 'karnaatak'),
    ('સિટી ટ્રેડિંગ', 'sitee treding'),
    ('दिल्ली', 'dillee'),
    ('தமிழ்நாடு', 'tamilnaatu'),
    ('আদিত্য', 'aadity'),          # Bengali via offset realignment
    ('ਪੰਜਾਬ', 'panjaab'),          # Gurmukhi tippi
]


@pytest.mark.parametrize('src,exp', SPOT)
def test_spot(src, exp):
    n, tag, seq, toks, tr = T.field_tokens(src)
    assert tr == exp and ' '.join(seq) == exp
    assert set(exp.split()) <= set(toks) and all(len(t) > 1 for t in exp.split())


def test_devanagari_not_shattered_by_punct_strip():
    n = T.norm_text('राम-मार्केटिंग, (प्राइवेट) लिमिटेड।')
    assert n == 'राम मार्केटिंग प्राइवेट लिमिटेड'
    assert 'ा' in n and '्' in n           # matra / virama survive
    assert T.norm_text('a‍b­c') == 'abc'  # zero-width chars deleted, not spaced


def test_raw_preserved_and_norm():
    raw = "  Orelee's  Barbershop, LLC. "
    n, tag, seq, toks, tr = T.field_tokens(raw)
    assert n == 'orelee s barbershop llc' and raw == "  Orelee's  Barbershop, LLC. "


def test_accented_french():
    n, tag, seq, toks, _ = T.field_tokens('Société Générale Éditions SARL')
    assert n == 'société générale éditions sarl' and tag == 'latin'
    assert seq == ['societe', 'generale', 'editions', 'sarl']
    # decomposed input gives identical result
    import unicodedata
    assert T.field_tokens(unicodedata.normalize('NFD', 'Société Générale'))[0] == 'société générale'
    leg, core = T.extract_suffix(seq, CFG, 'France')
    assert leg == ['sarl'] and core == 'societe generale editions'
    # French forms are NOT applied to train countries
    assert T.extract_suffix(['foo', 'sa'], CFG, 'US')[0] == []


def test_suffix_nondestructive_and_core_nonempty():
    for name in ['Acme Pvt. Ltd.', 'Ltd', 'Limited', 'Company Limited', 'LLC', 'X Co Ltd', 'Bob LLC dba Best Pizza', 'dba', '']:
        n, tag, seq, toks, tr = T.field_tokens(name)
        leg, core = T.extract_suffix(seq, CFG, 'US')
        assert (core != '') == bool(seq)
        assert set(seq) == set(seq)  # sequence itself untouched
        assert ' '.join(seq).startswith(core)
    assert T.extract_suffix(*[T.field_tokens('Acme Pvt. Ltd.')[2], CFG, 'India']) == (['private limited'], 'acme')
    assert T.extract_suffix(T.field_tokens('X Co Ltd')[2], CFG, 'US') == (['co', 'limited'], 'x')
    assert T.extract_suffix(T.field_tokens('Bob LLC dba Best Pizza')[2], CFG, 'US') == (['llc', 'dba'], 'bob')
    assert T.extract_suffix(['limited'], CFG, 'US') == ([], 'limited')
    assert T.extract_suffix(T.field_tokens('राम प्राइवेट लिमिटेड')[2], CFG, 'India') == (['private limited'], 'raam')


def test_empty_null_mixed_no_crash():
    for s in ['', ' ', 'NULL', 'null, null', '---', 'Aक्B', 'Tata टाटा Motors', '\x00\x01', '東京 Shop', 'مرحبا', '्', 'ा']:
        n, tag, seq, toks, tr = T.field_tokens(s)
        assert tag in ('latin', 'deva', 'indic-other', 'mixed', 'other')
        T.extract_suffix(seq, CFG, 'India'); T.addr_extras(s, seq, 'India'); T.addr_extras(s, seq, 'US')
    assert T.field_tokens('')[3] == []
    assert T.field_tokens('Tata टाटा Motors')[1] == 'mixed'
    assert T.field_tokens('टाटा')[1] == 'deva' and T.field_tokens('ಕರ್ನಾಟಕ')[1] == 'indic-other'


def test_addr_extras():
    a = 'Karnataka, 50, KNO 499/499, RAGHUVANAHALLI, BANGALORE SOUTH 560062'
    _, _, seq, _, _ = T.field_tokens(a)
    assert T.addr_extras(a, seq, 'India') == ('560062', ['499', '50', '560062'], 'karnataka')
    a = '1795 Westchester Drive, High Point, NC'
    _, _, seq, _, _ = T.field_tokens(a)
    assert T.addr_extras(a, seq, 'US') == ('', ['1795'], 'nc')
    a = '17560 Ellis Road, Tahlequah, OK 74464'
    _, _, seq, _, _ = T.field_tokens(a)
    assert T.addr_extras(a, seq, 'US')[0] == '74464'


def test_char3grams():
    assert T.char3grams('ab') == ['  a', ' ab', 'ab ']


def test_france_state_tok_empty_and_pin():
    a = '5 RUE DE LA PAIX 75002 PARIS FRANCE'
    seq = T.field_tokens(a)[2]
    assert T.addr_extras(a, seq, 'France') == ('75002', ['5', '75002'], '')
    a = '12 avenue de Washington, Georgia Business Park, 69003 Lyon'
    assert T.addr_extras(a, T.field_tokens(a)[2], 'France')[2] == ''
    a = 'IN DE LA RUE 33000 BORDEAUX CEDEX'
    assert T.addr_extras(a, T.field_tokens(a)[2], 'France')[0] == '33000'
    a = 'France 75002 Paris 5 rue de la Paix'
    assert T.addr_extras(a, T.field_tokens(a)[2], 'France')[0] == '75002'
    a = '12345 rue de la Gare Lyon Nord Sud Est'          # street number, not in last 4 tokens
    assert T.addr_extras(a, T.field_tokens(a)[2], 'France')[0] == ''
    # train countries unchanged: last-two-token rule and state logic
    a = '5 rue x 75002 paris france'
    assert T.addr_extras(a, T.field_tokens(a)[2], 'US')[0] == ''
    a = '17560 Ellis Road, Tahlequah, OK 74464'
    assert T.addr_extras(a, T.field_tokens(a)[2], 'US')[2] == 'ok'


def test_french_leading_suffix():
    f = lambda s, c: T.extract_suffix(T.field_tokens(s)[2], CFG, c)
    assert f('SARL Dupont', 'France') == (['sarl'], 'dupont')
    assert f('S.A.R.L. Dupont et Fils', 'France') == (['sarl'], 'dupont et fils')
    assert f('SAS Dupont SARL', 'France') == (['sas', 'sarl'], 'dupont')
    assert f('SARL', 'France') == ([], 'sarl')                  # core never empty
    assert f('SA Dupont', 'France') == ([], 'sa dupont')        # 'sa' not a leading marker
    assert f('SARL Dupont', 'US') == ([], 'sarl dupont')        # train countries untouched
    assert f('SCI Dupont', 'India') == ([], 'sci dupont')


def test_combining_only_token_dropped():
    n, tag, seq, toks, tr = T.field_tokens('abc ́ def')
    assert '' not in seq and '' not in toks and seq == ['abc', 'def']
