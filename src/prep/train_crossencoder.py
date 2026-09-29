"""PHASE 3 -- fine-tune a multilingual cross-encoder on the ambiguous band.

Stage 2 of the cascade. The GBDT resolves 98.2% of pairs at near-certainty; this
model exists only for the 1.8% it cannot, which nonetheless contain 13.4% of all
true matches.

WHY A CROSS-ENCODER AND NOT A BI-ENCODER. The published comparison is decisive
for our failure mode: cross-encoders beat bi-encoders at every model size,
because "the EM decision often requires fine-grained field-level comparisons
across records to align abbreviations and interpret conflicting values, and a
bi-encoder must compress each record into a single representation before
observing the other." Our hardest pairs are exactly that --
`Custom Wealth Services LLC` vs `Custom Wealth Ventures, Llc`, where one token
decides the answer and aggregate similarity cannot see it.

INPUT is the RAW text (Ditto-style COL/VAL serialization), never the normalised
form. The whole point of stage 2 is to restore the word order, punctuation,
casing and script that the 40 aggregate features threw away.

Trained ONLY on the ambiguous band, which makes every negative a hard negative
by construction -- each one already fooled the GBDT into the 0.1-0.9 range.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DATA, REPORTS, SEED

MODEL_DIR = DATA / "models" / "crossencoder"
AMB = DATA / "ambiguous"
DEFAULT_BASE = "microsoft/Multilingual-MiniLM-L12-H384"


def serialize(name: str, addr: str, country: str) -> str:
    """Ditto COL/VAL serialization. Empty address is kept explicit rather than
    dropped -- 20.7% of ambiguous positives have one, and its absence is itself
    evidence the model should learn to weigh."""
    return f"COL name VAL {name} COL addr VAL {addr or '[EMPTY]'} COL country VAL {country}"


class PairDS(Dataset):
    def __init__(self, df: pl.DataFrame, tok, max_len: int, with_label: bool = True):
        self.a = [serialize(n, a, c) for n, a, c in zip(
            df["q_business_name"].to_list(), df["q_business_address"].to_list(),
            df["q_country"].to_list())]
        self.b = [serialize(n, a, c) for n, a, c in zip(
            df["c_business_name"].to_list(), df["c_business_address"].to_list(),
            df["c_country"].to_list())]
        self.y = df["label"].to_numpy().astype(np.float32) if with_label else None
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        item = {"a": self.a[i], "b": self.b[i]}
        if self.y is not None:
            item["y"] = float(self.y[i])
        return item


class Collate:
    """Picklable collate_fn -- a lambda or closure cannot cross the
    DataLoader's worker process boundary."""

    def __init__(self, tok, max_len: int):
        self.tok, self.max_len = tok, max_len

    def __call__(self, batch):
        enc = self.tok([x["a"] for x in batch], [x["b"] for x in batch],
                       truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        if "y" in batch[0]:
            enc["labels"] = torch.tensor([x["y"] for x in batch], dtype=torch.float32)
        return enc


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        labels = batch.pop("labels", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits.squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main(base: str = DEFAULT_BASE, epochs: int = 3, bs: int = 64, lr: float = 3e-5,
         max_len: int = 128, eval_bs: int = 256) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  base={base}")

    tr = pl.read_parquet(AMB / "train_ambiguous.parquet")
    va = pl.read_parquet(AMB / "val_ambiguous.parquet")
    print(f"train={tr.height:,} ({100*tr['label'].mean():.1f}% pos)  "
          f"val={va.height:,} ({100*va['label'].mean():.1f}% pos)")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, problem_type="regression").to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par/1e6:.1f}M")

    # hold out a calibration slice from TRAIN -- validation must stay clean
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(tr.height)
    n_cal = int(0.12 * tr.height)
    cal_df, fit_df = tr[perm[:n_cal].tolist()], tr[perm[n_cal:].tolist()]
    print(f"fit={fit_df.height:,}  calib={cal_df.height:,}")

    cfn = Collate(tok, max_len)
    fit_dl = DataLoader(PairDS(fit_df, tok, max_len), batch_size=bs, shuffle=True,
                        collate_fn=cfn, num_workers=2, pin_memory=True, drop_last=True)
    cal_dl = DataLoader(PairDS(cal_df, tok, max_len), batch_size=eval_bs,
                        collate_fn=cfn, num_workers=2)
    val_dl = DataLoader(PairDS(va, tok, max_len), batch_size=eval_bs,
                        collate_fn=cfn, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(fit_dl) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossfn = torch.nn.BCEWithLogitsLoss()

    from sklearn.metrics import average_precision_score, roc_auc_score
    history = []
    for ep in range(epochs):
        model.train()
        run, seen = 0.0, 0
        te = time.time()
        for i, batch in enumerate(fit_dl):
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits.squeeze(-1)
                loss = lossfn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * y.size(0)
            seen += y.size(0)
            if i % 200 == 0:
                print(f"  ep{ep} step {i}/{len(fit_dl)} loss={run/max(seen,1):.4f} "
                      f"({time.time()-te:.0f}s)", flush=True)
        pv = predict(model, val_dl, device)
        yv = va["label"].to_numpy()
        auc, ap = roc_auc_score(yv, pv), average_precision_score(yv, pv)
        history.append({"epoch": ep, "train_loss": run / seen, "val_auc": auc, "val_ap": ap})
        print(f"epoch {ep}: loss={run/seen:.4f}  val AUC={auc:.4f}  val AP={ap:.4f}  "
              f"({time.time()-te:.0f}s)")

    # isotonic calibration on the held-out TRAIN slice (never validation)
    from sklearn.isotonic import IsotonicRegression
    pc = predict(model, cal_dl, device)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pc, cal_df["label"].to_numpy())

    pv_raw = predict(model, val_dl, device)
    pv_cal = iso.predict(pv_raw)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    import pickle
    with (MODEL_DIR / "calibrator.pkl").open("wb") as fh:
        pickle.dump({"iso": iso, "base": base, "max_len": max_len}, fh)

    # cache validation predictions for the cascade evaluation
    va.select(["source1_entity_id", "candidate_entity_id", "label", "p_gbdt"]).with_columns([
        pl.Series("p_ce_raw", pv_raw.astype(np.float32)),
        pl.Series("p_ce", pv_cal.astype(np.float32)),
    ]).write_parquet(DATA / "reports" / "val_crossencoder_preds.parquet")

    yv = va["label"].to_numpy()
    gb = va["p_gbdt"].to_numpy()
    rep = {
        "base": base, "parameters_M": round(n_par / 1e6, 1),
        "epochs": epochs, "batch_size": bs, "lr": lr, "max_len": max_len,
        "train_pairs": int(fit_df.height), "calib_pairs": int(cal_df.height),
        "history": history,
        "val_auc_gbdt_on_band": float(roc_auc_score(yv, gb)),
        "val_ap_gbdt_on_band": float(average_precision_score(yv, gb)),
        "val_auc_ce": float(roc_auc_score(yv, pv_cal)),
        "val_ap_ce": float(average_precision_score(yv, pv_cal)),
        "total_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / "crossencoder.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n--- ON THE AMBIGUOUS BAND ---")
    print(f"  GBDT          AUC={rep['val_auc_gbdt_on_band']:.4f}  AP={rep['val_ap_gbdt_on_band']:.4f}")
    print(f"  cross-encoder AUC={rep['val_auc_ce']:.4f}  AP={rep['val_ap_ce']:.4f}")
    print(f"CE_DONE in {rep['total_seconds']}s")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args()
    main(base=a.base, epochs=a.epochs, bs=a.bs, lr=a.lr, max_len=a.max_len)
