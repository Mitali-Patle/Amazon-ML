"""Country normalisation and partition-key resolution.

DESIGN NOTE -- why this module is deliberately small.

An audit of all six files (data/reports/exp_country_audit.json) found exactly
three distinct country values across 24.2M records:

    'US'  'India'  'France'

with NO casing variants, NO leading/trailing whitespace, NO empty values, NO
NFKC differences, and no two values collapsing into one under strip+casefold.
The dataset therefore does not require a country dictionary, and inventing a
large one would add untested behaviour for no measured benefit.

What IS implemented is a thin defensive layer, because the failure mode is
silent and expensive: if an unexpected spelling appeared at inference and were
treated as a distinct partition, every entity carrying it would retrieve from a
pool that does not contain its true matches. The layer therefore:

  * canonicalises whitespace/casing/NFKC so ``"  usa "`` and ``"USA"`` cannot
    become separate partitions;
  * maps only the handful of obvious aliases for the three observed countries;
  * NEVER discards an unrecognised value -- it is passed through as its own
    partition key and flagged, so it can still retrieve normally;
  * keeps the raw value untouched.

Crucially, "known" is decided against the POOL BEING SEARCHED, not against the
training countries. France has no training labels but a 1.4M-record test pool,
so it is a perfectly ordinary partition at inference. Label-derived statistics
are a separate concern and are never fabricated for an unseen country.
"""
from __future__ import annotations

import unicodedata
from enum import Enum


class CountryStatus(str, Enum):
    OK = "ok"                    # resolved to a partition present in the pool
    ALIASED = "aliased"          # matched via the alias table, not verbatim
    MISSING = "missing"          # empty / whitespace-only -> fallback
    UNKNOWN_IN_POOL = "unknown"  # well-formed but no such partition -> fallback


# Aliases for the three observed countries only. Deliberately short: every entry
# is a spelling that would plausibly denote the same partition. Not an attempt
# to model world geography.
ALIASES: dict[str, str] = {
    "us": "US", "usa": "US", "u s": "US", "u s a": "US",
    "united states": "US", "united states of america": "US", "america": "US",
    "india": "India", "ind": "India", "in": "India", "bharat": "India",
    "republic of india": "India",
    "france": "France", "fr": "France", "fra": "France",
    "french republic": "France", "republique francaise": "France",
}

CANONICAL = {"US", "India", "France"}


def normalize_country(raw: str | None) -> tuple[str, CountryStatus]:
    """Return (partition_key, status). Never raises, never returns None.

    An unrecognised but well-formed value keeps its own canonical spelling and
    becomes its own partition key -- it is the *partitioner* that decides
    whether such a partition exists in the pool.
    """
    if raw is None:
        return "", CountryStatus.MISSING
    s = unicodedata.normalize("NFKC", raw).strip()
    if not s:
        return "", CountryStatus.MISSING

    key = " ".join(s.casefold().replace(".", " ").split())
    if key in ALIASES:
        canon = ALIASES[key]
        return canon, (CountryStatus.OK if s == canon else CountryStatus.ALIASED)
    # Unrecognised: keep a canonical, whitespace-stable spelling so that two
    # cosmetic variants of the same novel country cannot split into two partitions.
    return s, CountryStatus.OK


def resolve_partition(raw: str | None, available: set[str]) -> tuple[str, CountryStatus]:
    """Resolve a record's country against the partitions that actually exist.

    ``available`` is derived from the POOL at build time, so a country never
    seen in training (France) resolves normally as long as the pool contains it.
    """
    key, status = normalize_country(raw)
    if status is CountryStatus.MISSING:
        return "", CountryStatus.MISSING
    if key in available:
        return key, status
    return key, CountryStatus.UNKNOWN_IN_POOL


def needs_fallback(status: CountryStatus) -> bool:
    return status in (CountryStatus.MISSING, CountryStatus.UNKNOWN_IN_POOL)
