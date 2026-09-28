"""Central configuration for the Amazon ML 2026 preprocessing pipeline.

Every tunable lives here so a run is fully described by this file + the git SHA.
Parameters marked LOCKED were fixed by measurement in
docs/AMAZON_ML_2026_DATA_DETECTIVE_REPORT.md / docs/PREPROCESSING_PLAYBOOK.md;
changing one invalidates the recall figures recorded there.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
DATA = ROOT / "data"
PARQUET = DATA / "parquet"
REPR = DATA / "representations"
INDEXES = DATA / "indexes"
CANDIDATES = DATA / "candidates"
VALIDATION = DATA / "validation"
REPORTS = DATA / "reports"

for _d in (PARQUET, REPR, INDEXES, CANDIDATES, VALIDATION, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20260925

# ---- expected shapes, asserted in phase 1 (from exhaustive counts) ----
EXPECTED_ROWS = {
    "train_source1": 2_206_821,
    "train_source2": 5_034_616,
    "train_source3": 5_285_603,
    "train_ground_truth": 2_206_821,
    "test_source1": 1_732_544,
    "test_source2": 4_887_273,
    "test_source3": 5_082_316,
}
EXPECTED_GT_PAIRS = 7_638_365

# ---- LOCKED preprocessing parameters ----
VAL_FRACTION = 0.05          # LOCKED (D11) 95/5 on Source-1 entities
DF_STOPWORD = 60_000         # LOCKED (Phase 7) tokens at/above this df are dropped from the index
ADDR_WEIGHT = 0.7            # LOCKED (D9) address-token IDF multiplier in retrieval scoring
TOP_K = 50                   # LOCKED (D10) candidates retrieved per Source-1 entity
K_GRID = (10, 25, 50, 100, 200)   # measurement grid for recall@K
TEST_POOL_RATIO = 5.75       # LOCKED (D12) S1:(S2+S3) ratio the validation pool must mirror

# Retrieval batching. Mean scored candidates/query was ~92k, so a batch of
# BATCH_QUERIES rows materialises roughly BATCH_QUERIES * 92k nonzeros.
BATCH_QUERIES = 2048      # larger batches: fewer, bigger sparse matmuls = better BLAS utilisation
INGEST_CHUNK = 500_000

# Chunk size for streaming candidate generation (retrieval.py::retrieve_source_chunked /
# generate_candidates_chunked). At 2.1M train / 1.73M test queries, accumulating all
# candidates as Python lists before writing (~210M strings, ~12GB) OOMs on a 15GB
# machine; queries are processed CANDIDATE_CHUNK at a time and each chunk is flushed
# to a parquet shard under data/candidates/ and freed.
CANDIDATE_CHUNK = 100_000

CHANNELS = ("name", "addr")  # transliterated tokens are FOLDED into these (D6), not a separate channel

CONFIG_VERSION = "1.0.0"


def config_hash() -> str:
    """Stable hash of every parameter that affects output, for run manifests."""
    payload = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "df_stopword": DF_STOPWORD,
        "addr_weight": ADDR_WEIGHT,
        "top_k": TOP_K,
        "test_pool_ratio": TEST_POOL_RATIO,
        "channels": CHANNELS,
        "version": CONFIG_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


SOURCES = ("source1", "source2", "source3")
SPLITS = ("train", "test")

# Cascade routing band. Pairs with BAND_LO < p_gbdt < BAND_HI are re-scored by
# the cross-encoder.
#
# Widened 0.1-0.9 -> 0.02-0.98 on evidence: the narrow band left 1.8% of all
# positives below p=0.1, where NEITHER stage could recover them -- the GBDT had
# already written them off and they never reached the cross-encoder. The wider
# band captures 24.9% of positives instead of 13.4% (train: 325,984 pairs at
# 39.9% positive vs 138,260 at 48.6%), for 4.3% of pairs instead of 1.8%.
# Inference cost grows linearly with band width and is small either way.
BAND_LO = 0.02
BAND_HI = 0.98

# France gets a WIDER band. It is 14.98% of test entities and has zero training
# labels, so the GBDT is extrapolating there and its confidence is less earned
# than on US/India. Independent leaderboard evidence supports this: a teammate
# measured +0.003 from using a cross-encoder on France instead of LightGBM
# (0.974 vs 0.971) -- i.e. the tree model is the weak link specifically on France.
# Widening the band routes more French pairs to the stronger model.
BAND_LO_FRANCE = 0.005
BAND_HI_FRANCE = 0.995
