"""Regression tests for the normaliser.

These lock in the two findings that fail SILENTLY if broken: Indic-safe
punctuation stripping (H7) and transliteration-as-tokens (D6).
Run: .venv/bin/python -m tests.test_normalize
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prep import normalize as N  # noqa: E402

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILS.append(msg)


print("== H7: punctuation stripping must not shatter Devanagari ==")
deva = "राम मार्केटिंग प्राइवेट लिमिटेड"
naive = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", deva).casefold())).strip()
ours = N.normalize(deva)
check(len(ours.split()) == 4, f"4 tokens preserved, got {len(ours.split())}: {ours.split()}")
check(len(naive.split()) > 4, f"naive [^\\w\\s] shatters into {len(naive.split())} fragments (control)")
check(max(len(t) for t in ours.split()) > 3, "tokens are words, not bare consonants")

print("\n== punctuation IS stripped for Latin ==")
check(N.normalize("Orelee's Barbershop") == "orelee s barbershop", "apostrophe -> space")
check(N.normalize("B+ Retail  Inc") == "b retail inc", "plus sign + double space collapsed")
check(N.normalize("Allied  Partners-LLC") == "allied partners llc", "hyphen -> space")

print("\n== script detection ==")
for s, want in [("Ridge Association LLC", "latin"), ("राम मार्केटिंग", "deva"),
                ("ಕರ್ನಾಟಕ", "indic-other"), ("Creative हॉस्पिटैलिटी", "mixed")]:
    got = N.script_of(s)
    check(got == want, f"script_of({s[:22]!r}) == {want!r} (got {got!r})")

print("\n== transliteration ==")
cases = {
    "राम मार्केटिंग प्राइवेट लिमिटेड": "limited",   # legal suffix must survive into Latin
    "अल इन्वेस्टमेंट एलएलपी": "al",                  # S1 counterpart is 'Al Investment LLP'
    "ಕರ್ನಾಟಕ": "karnaatak",
    "સિટી ટ્રેડિંગ": "sitee",
}
for src, expect_tok in cases.items():
    out = N.transliterate(N.normalize(src))
    check(expect_tok in out.split(), f"{src[:18]!r} -> {out!r} contains {expect_tok!r}")

print("\n== D6: transliterated tokens are UNIONED, originals preserved ==")
t = N.tokens_with_translit(N.normalize("अल इन्वेस्टमेंट एलएलपी"))
check("अल" in t, "original Devanagari token retained")
check("al" in t, "transliterated token added")
check(len(t) == len(set(t)), "no duplicate tokens")
latin_in = N.normalize("Vision Partners Corp")
check(N.tokens_with_translit(latin_in) == latin_in.split(), "latin input is a no-op")

print("\n== H3: suffixes extracted, name never destroyed ==")
toks = N.tokens_of(N.normalize("Wayne Pharmaceuticals LLC"))
check(N.extract_suffixes(toks) == ["llc"], "suffix extracted")
check("pharmaceuticals" in toks and "llc" in toks, "original tokens intact")
check(N.name_core(toks) == "wayne pharmaceuticals", "name_core drops suffix (feature only)")
check(N.name_core(N.tokens_of(N.normalize("LLC"))) == "llc", "all-suffix name falls back, never empty")

print("\n== address parsing is additive ==")
check(N.addr_pin("Flat 2, Sector 9, New Delhi, 110085") == "110085", "PIN extracted")
check(N.addr_pin("1918 11th Avenue, Nashville, TN") == "", "no PIN -> empty, no crash")
check(N.addr_numbers("A-137/3, 1ST FLOOR, PLOT NO-137") == ["137", "3", "1"], "numbers deduped in order")

print("\n== empty / degenerate input ==")
for f in (N.normalize, N.transliterate, N.script_of):
    try:
        f("")
    except Exception as e:  # noqa: BLE001
        check(False, f"{f.__name__}('') raised {e!r}")
check(N.normalize("") == "" and N.tokens_of("") == [], "empty string is safe")
check(N.normalize("   ") == "", "whitespace-only -> empty")

print("\n== word-order transposition must not matter (token sets) ==")
a = set(N.tokens_of(N.normalize("Atlantic Research Ventures-Revere")))
b = set(N.tokens_of(N.normalize("Atlantic Ventures-Revere Research")))
check(a == b, "transposed names give identical token sets")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print(f"   - {f}")
    sys.exit(1)
print("ALL NORMALISER TESTS PASSED")
