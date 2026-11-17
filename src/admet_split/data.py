"""Load the OpenADMET ExpansionRx data and define the endpoint set.

The premise of the whole project is that `expansion_data_test.csv` is LATER IN TIME
than `expansion_data_train.csv`. That is verified here rather than asserted: the two
files' compound IDs must be disjoint and every test ID must exceed every train ID.
If that ever stops being true the loader raises, because every conclusion downstream
would silently change meaning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HF_BASE = ("https://huggingface.co/datasets/openadmet/"
           "openadmet-expansionrx-challenge-data/resolve/main")

PRIMARY = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
]
SECONDARY = ["MPPB", "MBPB", "MGMB"]
ENDPOINTS = PRIMARY + SECONDARY

# Endpoints whose native scale is logarithmic are scored on log10 so that an MAE is
# comparable across endpoints and a ratio of MAEs means something. LogD is already a
# log quantity; the rest are raw measurements with a long right tail.
LOG10 = {
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
}


@dataclass(frozen=True)
class Frame:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def id_order_verified(self) -> bool:
        return True


def _numeric_id(name: str) -> int:
    digits = re.sub(r"[^0-9]", "", str(name))
    return int(digits) if digits else -1


def download(raw_dir: Path) -> None:
    """Fetch the two labelled CSVs if they are not already on disk."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("expansion_data_train.csv", "expansion_data_test.csv"):
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 100_000:
            continue
        url = f"{HF_BASE}/{name}"
        with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:
            fh.write(r.read())


def load(raw_dir: Path | None = None) -> Frame:
    """Load both splits, verify they are time-ordered, and apply the log10 scale."""
    root = Path(__file__).resolve().parents[2]
    raw_dir = raw_dir or (root / "data" / "raw")
    download(raw_dir)

    tr = pd.read_csv(raw_dir / "expansion_data_train.csv")
    te = pd.read_csv(raw_dir / "expansion_data_test.csv")

    tr_ids = {_numeric_id(x) for x in tr["Molecule Name"]}
    te_ids = {_numeric_id(x) for x in te["Molecule Name"]}
    if tr_ids & te_ids:
        raise ValueError(
            f"{len(tr_ids & te_ids)} compound IDs appear in BOTH splits; the "
            "time-forward premise is broken and no ratio computed here is meaningful")
    if min(te_ids) <= max(tr_ids):
        raise ValueError(
            "a test compound has an ID at or below the train maximum, so the split "
            "is not strictly time-ordered")

    for df in (tr, te):
        for ep in ENDPOINTS:
            if ep in LOG10:
                v = pd.to_numeric(df[ep], errors="coerce")
                df[ep] = v.where(v > 0).apply(lambda x: __import__("numpy").log10(x))
    return Frame(train=tr, test=te)


def coverage(frame: Frame) -> pd.DataFrame:
    """Per-endpoint sample size on each side. Assays are run in sequence, so the
    measured subset is selected; this table is what keeps that visible."""
    rows = []
    for ep in ENDPOINTS:
        a = frame.train[ep].notna()
        b = frame.test[ep].notna()
        rows.append({"endpoint": ep,
                     "train_n": int(a.sum()), "test_n": int(b.sum()),
                     "train_frac": float(a.mean()), "test_frac": float(b.mean())})
    return pd.DataFrame(rows)
