"""Deterministic, per-record text normalisation.

Nothing here is fitted on data, so it is leakage-free and may run before the
train/validation split (report D18). Every function is pure.

The two rules that carry the most evidence:
  * punctuation is stripped by Unicode CATEGORY, never by ``[^\\w\\s]`` -- the
    latter drops Indic combining marks (category Mn) and shatters Devanagari
    words into bare consonants, silently corrupting ~13% of S2/S3 names (H7).
  * transliterated tokens are UNIONED into the ordinary token set, never scored
    as a separate channel -- as a separate channel recall collapsed 90.1 -> 27.6 (D7).
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# base normalisation
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def strip_punct(s: str) -> str:
    """Replace Unicode punctuation/symbol/control chars with a space.

    Deliberately category-based: keeps Mn/Mc combining marks so Indic
    aksharas stay intact.
    """
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        out.append(" " if (cat[0] in ("P", "S") or cat in ("Cc", "Cf", "Zl", "Zp")) else ch)
    return "".join(out)


def normalize(s: str) -> str:
    """NFKC -> category-safe punctuation strip -> casefold -> whitespace collapse."""
    if not s:
        return ""
    return _WS.sub(" ", strip_punct(nfkc(s)).casefold()).strip()


# --------------------------------------------------------------------------
# script detection
# --------------------------------------------------------------------------
_INDIC_BLOCKS = {
    0x0900: "deva", 0x0980: "beng", 0x0A00: "guru", 0x0A80: "gujr",
    0x0B00: "orya", 0x0B80: "taml", 0x0C00: "telu", 0x0C80: "knda",
    0x0D00: "mlym",
}


def script_of(s: str) -> str:
    """Coarse script label: latin | deva | indic-other | mixed | other."""
    lat = dev = oth = 0
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x0250:
            lat += 1
        elif 0x0900 <= o <= 0x097F:
            dev += 1
        elif any(b <= o < b + 0x80 for b in _INDIC_BLOCKS):
            oth += 1
        else:
            oth += 1
    if dev and lat:
        return "mixed"
    if dev:
        return "deva"
    if oth and lat:
        return "mixed"
    if oth:
        return "indic-other"
    return "latin"


def has_indic(s: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x0DFF for c in s)


# --------------------------------------------------------------------------
# transliteration (Brahmic -> Latin), rule-based
# --------------------------------------------------------------------------
# All Brahmic blocks below share the ISCII-derived layout, so we realign each to
# Devanagari by codepoint offset and then apply one Devanagari->Latin table.
_CONS = {
    0x0915: "k", 0x0916: "kh", 0x0917: "g", 0x0918: "gh", 0x0919: "n",
    0x091A: "ch", 0x091B: "chh", 0x091C: "j", 0x091D: "jh", 0x091E: "n",
    0x091F: "t", 0x0920: "th", 0x0921: "d", 0x0922: "dh", 0x0923: "n",
    0x0924: "t", 0x0925: "th", 0x0926: "d", 0x0927: "dh", 0x0928: "n",
    0x0929: "n", 0x092A: "p", 0x092B: "ph", 0x092C: "b", 0x092D: "bh",
    0x092E: "m", 0x092F: "y", 0x0930: "r", 0x0931: "r", 0x0932: "l",
    0x0933: "l", 0x0934: "l", 0x0935: "v", 0x0936: "sh", 0x0937: "sh",
    0x0938: "s", 0x0939: "h",
}
_VOW = {
    0x0905: "a", 0x0906: "aa", 0x0907: "i", 0x0908: "ee", 0x0909: "u",
    0x090A: "oo", 0x090B: "ri", 0x090D: "e", 0x090E: "e", 0x090F: "e",
    0x0910: "ai", 0x0911: "o", 0x0912: "o", 0x0913: "o", 0x0914: "au",
}
_MATRA = {
    0x093E: "aa", 0x093F: "i", 0x0940: "ee", 0x0941: "u", 0x0942: "oo",
    0x0943: "ri", 0x0945: "e", 0x0946: "e", 0x0947: "e", 0x0948: "ai",
    0x0949: "o", 0x094A: "o", 0x094B: "o", 0x094C: "au",
}
_ANUSVARA = {0x0901: "n", 0x0902: "n", 0x0903: "h"}
_VIRAMA = 0x094D
_NUKTA = 0x093C


def _to_deva(ch: str) -> str:
    o = ord(ch)
    for base in _INDIC_BLOCKS:
        if base <= o < base + 0x80:
            return chr(o - base + 0x0900)
    return ch


def translit_word(w: str) -> str:
    w = "".join(_to_deva(c) for c in w)
    out: list[str] = []
    i, n = 0, len(w)
    while i < n:
        o = ord(w[i])
        if o in _CONS:
            out.append(_CONS[o])
            i += 1
            if i < n and ord(w[i]) == _NUKTA:
                i += 1
            if i < n and ord(w[i]) == _VIRAMA:
                i += 1                                   # virama: no inherent vowel
            elif i < n and ord(w[i]) in _MATRA:
                out.append(_MATRA[ord(w[i])])
                i += 1
            elif i < n and ord(w[i]) in _ANUSVARA:
                out.append("a" + _ANUSVARA[ord(w[i])])
                i += 1
            else:
                out.append("a")                          # inherent schwa
        elif o in _VOW:
            out.append(_VOW[o]); i += 1
        elif o in _ANUSVARA:
            out.append(_ANUSVARA[o]); i += 1
        elif o in (_VIRAMA, _NUKTA):
            i += 1
        else:
            out.append(w[i]); i += 1
    s = "".join(out)
    if len(s) > 2 and s.endswith("a"):                   # word-final schwa deletion
        s = s[:-1]
    return s


def transliterate(s: str) -> str:
    """Transliterate only the Indic words of an already-normalised string."""
    return " ".join(translit_word(w) if has_indic(w) else w for w in s.split())


# --------------------------------------------------------------------------
# legal suffixes  (EXTRACTED, never deleted from the name -- H3)
# --------------------------------------------------------------------------
# Variant -> canonical class. Canonicalising matters because the same legal form
# appears in Latin, in Indic script, and in several transliterated spellings; the
# suffix-agreement feature is only meaningful if they collapse to one label.
# The Indic variants below are EMPIRICAL -- the top transliterated tokens actually
# observed in Indic-script names across train_source2/3, not guesses.
SUFFIX_CANON: dict[str, str] = {}


def _reg(canon: str, *variants: str) -> None:
    for v in variants:
        SUFFIX_CANON[v] = canon


_reg("ltd", "ltd", "limited", "lmtd", "limitet", "limirrad", "limited.")
_reg("pvt", "pvt", "private", "pte", "praaivet", "praivet", "praaibhet",
     "piraivet", "praivarr", "praayavet")
_reg("llp", "llp", "elaelapee", "elelapee")
_reg("inc", "inc", "incorporated")
_reg("corp", "corp", "corporation")
_reg("co", "co", "company")
_reg("llc", "llc")
_reg("lp", "lp")
_reg("plc", "plc")
_reg("gmbh", "gmbh")
_reg("sa", "sa", "sas", "sarl", "srl")
_reg("bv", "bv", "nv", "ag")
_reg("group", "group")
_reg("holdings", "holdings")
_reg("enterprises", "enterprises")
_reg("ventures", "ventures")
_reg("industries", "industries")
_reg("solutions", "solutions")
_reg("services", "services")
_reg("associates", "associates")
_reg("partners", "partners")
_reg("trust", "trust")
_reg("society", "society")
_reg("foundation", "foundation")

LEGAL_SUFFIXES = set(SUFFIX_CANON)

# ``li`` and ``pra`` (from the Indian abbreviation "प्रा. लि.") each occur ~63k
# times but are only 2-3 chars and collide with ordinary name tokens, so they are
# recorded as a weak suffix signal and NEVER removed from name_core.
WEAK_SUFFIX_HINTS = {"li": "ltd", "pra": "pvt"}


def _canon_suffix(tok: str) -> str | None:
    """Canonical suffix class for a token, resolving Indic script via transliteration."""
    if tok in SUFFIX_CANON:
        return SUFFIX_CANON[tok]
    if has_indic(tok):
        t = translit_word(tok)
        if t in SUFFIX_CANON:
            return SUFFIX_CANON[t]
        if t in WEAK_SUFFIX_HINTS:
            return WEAK_SUFFIX_HINTS[t]
    return WEAK_SUFFIX_HINTS.get(tok)


def extract_suffixes(tokens: list[str]) -> list[str]:
    """Canonical legal-suffix classes present, deduped and order-stable.

    Works across scripts: ``लिमिटेड``, ``limited``, ``limitet`` and ``ltd`` all
    yield ``ltd``, so suffix agreement is comparable between a Latin Source-1
    record and an Indic Source-2/3 record.
    """
    found = [c for c in (_canon_suffix(t) for t in tokens) if c]
    return list(dict.fromkeys(found))


def name_core(tokens: list[str]) -> str:
    """Suffix-stripped name. FEATURE ONLY -- never a blocking or identity key.

    41.93% of Source-1 rows collide with another Source-1 row on this value
    (``primary care`` x786), so using it to decide identity causes false merges.
    Weak 2-3 char hints are deliberately NOT stripped: removing them would risk
    deleting genuine name content.
    """
    core = [t for t in tokens
            if not (t in SUFFIX_CANON
                    or (has_indic(t) and translit_word(t) in SUFFIX_CANON))]
    return " ".join(core) if core else " ".join(tokens)


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------
def tokens_of(norm_text: str) -> list[str]:
    return norm_text.split() if norm_text else []


def tokens_with_translit(norm_text: str) -> list[str]:
    """Token set unioned with transliterated forms (D6).

    Order is stable and duplicates are removed so downstream df counting is
    exact. Latin-only input is returned unchanged -- transliteration is a pure
    addition for Indic records and a no-op otherwise.
    """
    toks = tokens_of(norm_text)
    if not has_indic(norm_text):
        return list(dict.fromkeys(toks))
    extra = tokens_of(transliterate(norm_text))
    return list(dict.fromkeys(toks + extra))


# --------------------------------------------------------------------------
# address component parsing (ADDITIVE -- addr_norm is never overwritten)
# --------------------------------------------------------------------------
_PIN = re.compile(r"\b\d{5,6}\b")
_NUM = re.compile(r"\d+")


def addr_pin(raw_addr: str) -> str:
    m = _PIN.findall(raw_addr)
    return m[-1] if m else ""


def addr_numbers(raw_addr: str) -> list[str]:
    return list(dict.fromkeys(_NUM.findall(raw_addr)))


def char_ngrams(norm_text: str, n: int = 3) -> list[str]:
    """Character n-grams. PAIR FEATURES ONLY -- never summed into retrieval
    scoring, where they collapsed recall 90.1 -> 27.6 (D7)."""
    s = norm_text.replace(" ", "")
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]
