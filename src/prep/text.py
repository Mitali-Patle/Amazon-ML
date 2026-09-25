"""Per-record, fit-nothing text transforms (playbook Phase 3). Pure functions, no I/O except cfg loader."""
import json, re, unicodedata

# ---- step 2: strip only Unicode P*, S*, Cc, Cf -> space. Zero-width/soft-hyphen format chars are DELETED (not
# spaced) so Indic words joined by ZWJ/ZWNJ are not shattered (documented deviation, Cf subset only).
_DELETE = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD}
STRIP = {}
for _cp in range(0x110000):
    _c = unicodedata.category(chr(_cp))
    if _c[0] in 'PS' or _c in ('Cc', 'Cf'):
        STRIP[_cp] = None if _cp in _DELETE else ' '

R_LAT = re.compile('[A-Za-zÀ-ɏḀ-ỿ]')
R_DEV = re.compile('[ऀ-ॿ]')
R_IND = re.compile('[ঀ-෿]')
R_OTH = re.compile('[^\x00-ɏ̀-ͯḀ-ỿऀ-෿\\s]')
R_ANYINDIC = re.compile('[ऀ-෿]')


def norm_text(s):
    """NFKC + ws collapse, strip P*/S*/Cc/Cf, casefold. Raw string is never modified by callers."""
    if not s:
        return ''
    if not s.isascii():
        s = unicodedata.normalize('NFKC', s)
    return ' '.join(s.translate(STRIP).split()).casefold()


def script_tag(s):
    """latin | deva | indic-other | mixed | other (pure non-Latin non-Indic, e.g. Arabic/CJK). Letterless -> latin."""
    if s.isascii():
        return 'latin'
    k = []
    if R_LAT.search(s): k.append('latin')
    if R_DEV.search(s): k.append('deva')
    if R_IND.search(s): k.append('indic-other')
    if R_OTH.search(s): k.append('other')
    if not k: return 'latin'
    return k[0] if len(k) == 1 else 'mixed'


def fold_token(t):
    """Strip Latin combining accents only (token space, Latin-range tokens only; Indic marks untouched)."""
    if t.isascii():
        return t
    if all(ord(c) < 0x250 or 0x300 <= ord(c) <= 0x36F or 0x1E00 <= ord(c) <= 0x1EFF for c in t):
        return ''.join(c for c in unicodedata.normalize('NFKD', t) if not 0x300 <= ord(c) <= 0x36F)
    return t

# ---- transliteration (rule based). Everything is realigned to the Devanagari block by codepoint offset.
_C = {0x915: 'k', 0x916: 'kh', 0x917: 'g', 0x918: 'gh', 0x919: 'n', 0x91A: 'ch', 0x91B: 'chh', 0x91C: 'j', 0x91D: 'jh',
      0x91E: 'n', 0x91F: 't', 0x920: 'th', 0x921: 'd', 0x922: 'dh', 0x923: 'n', 0x924: 't', 0x925: 'th', 0x926: 'd',
      0x927: 'dh', 0x928: 'n', 0x929: 'n', 0x92A: 'p', 0x92B: 'ph', 0x92C: 'b', 0x92D: 'bh', 0x92E: 'm', 0x92F: 'y',
      0x930: 'r', 0x931: 'r', 0x932: 'l', 0x933: 'l', 0x934: 'l', 0x935: 'v', 0x936: 'sh', 0x937: 'sh', 0x938: 's',
      0x939: 'h', 0x958: 'q', 0x959: 'kh', 0x95A: 'g', 0x95B: 'z', 0x95C: 'r', 0x95D: 'rh', 0x95E: 'f', 0x95F: 'y'}
_V = {0x904: 'a', 0x905: 'a', 0x906: 'aa', 0x907: 'i', 0x908: 'ee', 0x909: 'u', 0x90A: 'oo', 0x90B: 'ri', 0x90C: 'li',
      0x90D: 'e', 0x90E: 'e', 0x90F: 'e', 0x910: 'ai', 0x911: 'o', 0x912: 'o', 0x913: 'o', 0x914: 'au', 0x950: 'om'}
_M = {0x93E: 'aa', 0x93F: 'i', 0x940: 'ee', 0x941: 'u', 0x942: 'oo', 0x943: 'ri', 0x944: 'ri', 0x945: 'e', 0x946: 'e',
      0x947: 'e', 0x948: 'ai', 0x949: 'o', 0x94A: 'o', 0x94B: 'o', 0x94C: 'au'}
_N = {0x901: 'n', 0x902: 'n', 0x903: 'h'}
_SKIP = {0x93C, 0x93D}          # nukta, avagraha
_VIRAMA = 0x94D
TR = {}                          # codepoint -> (kind, text); kinds C V M N D H S(kip)
_DEV = {}
for _k, _v in _C.items(): _DEV[_k] = ('C', _v)
for _k, _v in _V.items(): _DEV[_k] = ('V', _v)
for _k, _v in _M.items(): _DEV[_k] = ('M', _v)
for _k, _v in _N.items(): _DEV[_k] = ('N', _v)
for _k in _SKIP: _DEV[_k] = ('S', '')
_DEV[_VIRAMA] = ('H', '')
for _i in range(10): _DEV[0x966 + _i] = ('D', str(_i))
for _off in range(0x80):
    _d = _DEV.get(0x900 + _off, ('S', ''))
    for _base in (0x900, 0x980, 0xA00, 0xA80, 0xB00, 0xB80, 0xC00, 0xC80, 0xD00):
        TR[_base + _off] = _d
# block-specific extras
TR[0x9CE] = ('C0', 't')                                   # Bengali khanda-ta (consonant, no inherent vowel)
TR[0x9F0] = ('C', 'r'); TR[0x9F1] = ('C', 'v')            # Assamese ra / wa
TR[0xA70] = ('N', 'n')                                    # Gurmukhi tippi
for _k, _v in {0xD7A: 'n', 0xD7B: 'n', 0xD7C: 'r', 0xD7D: 'l', 0xD7E: 'l', 0xD7F: 'k'}.items():
    TR[_k] = ('C0', _v)                                   # Malayalam chillu letters


def translit(tok):
    """Transliterate one whitespace-free token; non-Indic chars pass through. Word-final schwa deleted."""
    out, pend = [], False
    for ch in tok:
        cp = ord(ch)
        e = TR.get(cp) if 0x900 <= cp <= 0xDFF else None
        if e is None:
            out.append(ch); pend = False; continue
        k, t = e
        if k == 'C':
            out.append(t + 'a'); pend = True
        elif k == 'C0':
            out.append(t); pend = False
        elif k == 'M':
            if pend: out[-1] = out[-1][:-1] + t
            else: out.append(t)
            pend = False
        elif k == 'H':
            if pend: out[-1] = out[-1][:-1]
            pend = False
        elif k == 'S':
            pass
        else:  # V, N, D
            out.append(t); pend = False
    if pend:
        out[-1] = out[-1][:-1]
    return ''.join(out)


def field_tokens(raw):
    """-> norm, script tag, seq (ordered latinised tokens), tokens (sorted unique set: folded originals U translit),
    translit string ('' unless the field contains Indic chars)."""
    n = norm_text(raw)
    tag = script_tag(n)
    toks = n.split()
    if tag == 'latin' or not R_ANYINDIC.search(n):
        seq = [f for f in (fold_token(t) for t in toks) if f]   # drop tokens that were only combining marks
        return n, tag, seq, sorted(set(seq)), ''
    seq, tok = [], set()
    for t in toks:
        if R_ANYINDIC.search(t):
            tl = translit(t)
            tok.add(t)
            for p in tl.split(): tok.add(p)
            if tl: seq.append(tl)
        else:
            f = fold_token(t)
            if f: seq.append(f); tok.add(f)
    tok.discard('')
    return n, tag, seq, sorted(tok), ' '.join(seq)


def char3grams(s):
    """Pair-feature helper (NOT stored, NOT for retrieval): padded char 3-grams of a normalised name."""
    s = f'  {s} '
    return sorted({s[i:i + 3] for i in range(len(s) - 2)})

# ---- legal suffixes -----------------------------------------------------------------------------------------
# forms that may be written LEADING ('SARL Dupont'); 'sa' excluded (ambiguous as a first word)
FRENCH_LEAD = {'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'sca', 'selarl', 'scop'}


def load_suffix_cfg(path):
    cfg = json.load(open(path))
    def idx(d):
        return {tuple(v.split()): canon for canon, vs in d.items() for v in vs}
    lead = {k: v for k, v in cfg['french_canon'].items() if k in FRENCH_LEAD}
    return dict(common=idx(cfg['canon']), french=idx(cfg['french_canon']), french_lead=idx(lead),
                dba=[tuple(x.split()) for x in cfg['dba_markers']],
                train_countries=set(cfg['train_countries']))


def extract_suffix(seq, cfg, country):
    """Non-destructive: returns (legal_suffix list, name_core). name_core never empty."""
    if not seq:
        return [], ''
    idx = cfg['common'] if country in cfg['train_countries'] else {**cfg['french'], **cfg['common']}
    core, legal = list(seq), []
    for m in cfg['dba']:                          # clear DBA marker: 'X llc dba Y' -> core from 'X llc'
        L = len(m)
        for i in range(1, len(core) - L + 1):
            if tuple(core[i:i + L]) == m:
                core = core[:i]; legal.append('dba'); break
        if legal: break
    j, found = len(core), []
    while True:
        for n in (4, 3, 2, 1):
            if j - n >= 1 and tuple(core[j - n:j]) in idx:
                found.append(idx[tuple(core[j - n:j])]); j -= n; break
        else:
            break
    lead, k = [], 0
    if country not in cfg['train_countries']:      # leading French forms, unseen countries only
        while True:
            for n in (4, 3, 2, 1):
                if j - k - n >= 1 and tuple(core[k:k + n]) in cfg['french_lead']:
                    lead.append(cfg['french_lead'][tuple(core[k:k + n])]); k += n; break
            else:
                break
    return lead + found[::-1] + legal, ' '.join(core[k:j])

# ---- address extras -----------------------------------------------------------------------------------------
US_ST = {'AL': 'alabama', 'AK': 'alaska', 'AZ': 'arizona', 'AR': 'arkansas', 'CA': 'california', 'CO': 'colorado',
         'CT': 'connecticut', 'DE': 'delaware', 'DC': 'district of columbia', 'FL': 'florida', 'GA': 'georgia',
         'HI': 'hawaii', 'ID': 'idaho', 'IL': 'illinois', 'IN': 'indiana', 'IA': 'iowa', 'KS': 'kansas',
         'KY': 'kentucky', 'LA': 'louisiana', 'ME': 'maine', 'MD': 'maryland', 'MA': 'massachusetts',
         'MI': 'michigan', 'MN': 'minnesota', 'MS': 'mississippi', 'MO': 'missouri', 'MT': 'montana',
         'NE': 'nebraska', 'NV': 'nevada', 'NH': 'new hampshire', 'NJ': 'new jersey', 'NM': 'new mexico',
         'NY': 'new york', 'NC': 'north carolina', 'ND': 'north dakota', 'OH': 'ohio', 'OK': 'oklahoma',
         'OR': 'oregon', 'PA': 'pennsylvania', 'RI': 'rhode island', 'SC': 'south carolina', 'SD': 'south dakota',
         'TN': 'tennessee', 'TX': 'texas', 'UT': 'utah', 'VT': 'vermont', 'VA': 'virginia', 'WA': 'washington',
         'WV': 'west virginia', 'WI': 'wisconsin', 'WY': 'wyoming', 'PR': 'puerto rico'}
_US_CODES = set(US_ST)
_US_NAME = {v: k.lower() for k, v in US_ST.items()}
_IN_ST = ['andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat', 'haryana',
          'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
          'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana',
          'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal', 'delhi', 'jammu and kashmir', 'ladakh',
          'chandigarh', 'puducherry', 'andaman and nicobar islands', 'lakshadweep']
_IN_ALIAS = {'orissa': 'odisha', 'uttaranchal': 'uttarakhand', 'pondicherry': 'puducherry', 'tamilnadu': 'tamil nadu',
             'new delhi': 'delhi', 'nct of delhi': 'delhi', 'chattisgarh': 'chhattisgarh',
             'jammu kashmir': 'jammu and kashmir'}
_IN_MAP = {**{s: s for s in _IN_ST}, **_IN_ALIAS}
R_IN_ST = re.compile(r'(?<![a-z0-9])(' + '|'.join(sorted(map(re.escape, _IN_MAP), key=len, reverse=True)) + r')(?![a-z0-9])')
R_US_NAME = re.compile(r'(?<![a-z0-9])(' + '|'.join(sorted(map(re.escape, _US_NAME), key=len, reverse=True)) + r')(?![a-z0-9])')
R_UP2 = re.compile(r'(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])')
R_NUM = re.compile(r'[0-9]+')
R_PIN6 = re.compile(r'(?<![0-9])[1-9][0-9]{5}(?![0-9])')


TRAIN_COUNTRIES = ('India', 'US')


def addr_extras(raw, seq, country, train_countries=TRAIN_COUNTRIES):
    """-> (addr_pin, addr_numbers, addr_state_tok). Additive only; addr_norm untouched. Latin lexicons only.
    US/India rules are unchanged; any other (unseen) country gets a country-agnostic postal search and NO state token."""
    nums = sorted(set(R_NUM.findall(' '.join(seq))))
    pin = ''
    if country == 'India':
        m = R_PIN6.findall(' '.join(seq)); pin = m[-1] if m else ''
    elif country in train_countries:  # 5-digit ZIP must sit in the last two tokens (else it is a street number)
        for t in seq[-2:]:
            if len(t) == 5 and t.isdigit(): pin = t
    else:  # unseen country: 5-digit code in the last 4 tokens, else right after a leading country token
        for t in seq[-4:]:
            if len(t) == 5 and t.isdigit(): pin = t
        if not pin and len(seq) >= 3 and seq[0] == norm_text(country) and len(seq[1]) == 5 and seq[1].isdigit():
            pin = seq[1]
    st = ''
    if country == 'India':
        m = R_IN_ST.findall(' '.join(seq)); st = _IN_MAP[m[-1]] if m else ''
    elif country == 'US':
        c = [x for x in R_UP2.findall(raw) if x in _US_CODES]
        if c: st = c[-1].lower()
        else:
            m = R_US_NAME.findall(' '.join(seq)); st = _US_NAME[m[-1]] if m else ''
    return pin, nums, st
