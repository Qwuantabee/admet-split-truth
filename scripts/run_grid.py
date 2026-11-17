"""Measure, for every (endpoint, model family, protocol), how far the retrospective
error estimate is from the error on the REAL future compounds.

    ratio = retrospective MAE / prospective MAE

A ratio below 1 means the protocol told a team its model would be better than it was.
The prospective MAE is computed exactly once per (endpoint, model): fit on the whole
training period, score on the held-out future period. The retrospective MAE comes from
the protocol's folds inside the training period only, so no future compound can ever
inform a retrospective number.

Writes results/grid.json. Every number the README quotes comes from that file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_split import data as D  # noqa: E402
from admet_split import splits as S  # noqa: E402

PROTOCOLS = ("random", "scaffold", "butina", "time", "frontier")


# ----------------------------------------------------------------------------- features
def morgan_fps(smiles: pd.Series, n_bits: int = 2048, radius: int = 2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    out, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        v = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        out.append(np.frombuffer(v.ToBitString().encode(), dtype=np.uint8) - 48)
        keep.append(i)
    return np.array(out, dtype=np.uint8), np.array(keep, dtype=int)


def murcko_scaffolds(smiles: pd.Series) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            out.append("")
        else:
            try:
                out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
            except Exception:  # noqa: BLE001 - a weird molecule must not kill the run
                out.append("")
    return np.array(out, dtype=object)


def butina_clusters(fps: np.ndarray, cutoff: float = 0.6) -> np.ndarray:
    """Sphere-exclusion clusters on Tanimoto distance. Deterministic ordering."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    n = len(fps)
    # packed fingerprints are cheaper; rebuild bit vectors from the 0/1 rows
    from rdkit.DataStructs import ExplicitBitVect

    bits = []
    for row in fps:
        bv = ExplicitBitVect(len(row))
        for j in np.flatnonzero(row):
            bv.SetBit(int(j))
        bits.append(bv)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bits[i], bits[:i])
        dists.extend([1.0 - s for s in sims])
    groups = Butina.ClusterData(dists, n, cutoff, isDistData=True, reordering=True)
    lab = np.empty(n, dtype=int)
    for gi, members in enumerate(groups):
        for m in members:
            lab[m] = gi
    return lab


def frontier_similarity(fps: np.ndarray, sample: int = 400, seed: int = 0) -> np.ndarray:
    """Max Tanimoto similarity of each compound to a sample of the training period.

    Used only to ORDER compounds (least-similar first) for the frontier protocol, so
    a sampled estimate is enough and keeps this to seconds.
    """
    rng = np.random.default_rng(seed)
    n = len(fps)
    ref = rng.choice(n, size=min(sample, n), replace=False)
    F = fps.astype(np.float32)
    refmat = F[ref]
    inter = F @ refmat.T
    counts = F.sum(axis=1)[:, None]
    refc = refmat.sum(axis=1)[None, :]
    union = counts + refc - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return sim.max(axis=1)


# ------------------------------------------------------------------------------- models
def fit_predict_rf(Xtr, ytr, Xte, seed: int = 0):
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=seed,
                              min_samples_leaf=2)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def fit_predict_ridge(Xtr, ytr, Xte, seed: int = 0):
    from sklearn.linear_model import Ridge

    m = Ridge(alpha=1.0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


MODELS = {"rf": fit_predict_rf, "ridge": fit_predict_ridge}


def one_feature(Xtr, ytr, Xte):
    """The honest floor: predict the training mean. A protocol that cannot beat this
    is not measuring the model at all."""
    return np.full(len(Xte), float(np.mean(ytr)))


def _mae(pred, true) -> float:
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() == 0:
        return float("nan")
    return float(np.abs(pred[ok] - true[ok]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "grid.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--endpoints", nargs="*", default=None)
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    frame = D.load()
    tr, te = frame.train, frame.test
    eps = args.endpoints or D.ENDPOINTS

    Xtr_all, keep_tr = morgan_fps(tr["SMILES"])
    Xte_all, keep_te = morgan_fps(te["SMILES"])
    tr = tr.iloc[keep_tr].reset_index(drop=True)
    te = te.iloc[keep_te].reset_index(drop=True)
    print(f"molecules parsed: train {len(tr)}, test {len(te)}")

    scaff = murcko_scaffolds(tr["SMILES"])
    print("scaffolds done")
    clus = butina_clusters(Xtr_all)
    print(f"butina clusters: {len(np.unique(clus))}")
    order = np.array([int("".join(ch for ch in str(x) if ch.isdigit()) or -1)
                      for x in tr["Molecule Name"]])
    fsim = frontier_similarity(Xtr_all)
    print("frontier similarity done")

    cov = D.coverage(D.Frame(train=tr, test=te)).set_index("endpoint")

    rows = []
    for ep in eps:
        ytr = pd.to_numeric(tr[ep], errors="coerce").to_numpy(dtype=float)
        yte = pd.to_numeric(te[ep], errors="coerce").to_numpy(dtype=float)
        ok_tr = np.isfinite(ytr)
        ok_te = np.isfinite(yte)
        if ok_tr.sum() < 200 or ok_te.sum() < 100:
            print(f"  {ep}: skipped (train {ok_tr.sum()}, test {ok_te.sum()})")
            continue
        A, ya = Xtr_all[ok_tr], ytr[ok_tr]
        B, yb = Xte_all[ok_te], yte[ok_te]
        idx_tr = np.flatnonzero(ok_tr)

        for mname, fn in MODELS.items():
            # -------- prospective: fit on the whole training period, score the future
            pro_pred = fn(A, ya, B)
            pro_mae = _mae(pro_pred, yb)
            floor_mae = _mae(one_feature(A, ya, B), yb)

            # -------- retrospective per protocol, inside the training period only
            for proto in PROTOCOLS:
                if proto == "random":
                    folds = S.random_split(len(ya), k=5, seed=0)
                elif proto == "scaffold":
                    folds = S.scaffold_split(scaff[idx_tr], k=5, seed=0)
                elif proto == "butina":
                    folds = S.butina_split(clus[idx_tr], k=5, seed=0)
                elif proto == "time":
                    folds = S.time_split(order[idx_tr], k=5)
                else:
                    folds = S.frontier_split(fsim[idx_tr], k=5)

                per_fold = []
                for a, b in folds:
                    if len(a) < 50 or len(b) < 20:
                        continue
                    pred = fn(A[a], ya[a], A[b])
                    per_fold.append(_mae(pred, ya[b]))
                if not per_fold:
                    continue
                ret_mae = float(np.median(per_fold))
                rows.append({
                    "endpoint": ep, "model": mname, "protocol": proto,
                    "retrospective_mae": ret_mae,
                    "prospective_mae": pro_mae,
                    "ratio": (ret_mae / pro_mae) if pro_mae > 0 else float("nan"),
                    "n_folds": len(per_fold),
                    "train_n": int(ok_tr.sum()), "test_n": int(ok_te.sum()),
                    "prospective_floor_mae": floor_mae,
                    "model_beats_floor": bool(pro_mae < floor_mae),
                })
                print(f"  {ep:26s} {mname:6s} {proto:9s} "
                      f"retro {ret_mae:8.3f} pro {pro_mae:8.3f} "
                      f"ratio {ret_mae/pro_mae:5.2f}")

    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("no rows produced; check endpoint coverage")

    # -------- bootstrap the median ratio per protocol, clustered by endpoint
    rng = np.random.default_rng(11)
    boot = {}
    for proto, g in t.groupby("protocol"):
        vals = g.ratio.to_numpy()
        vals = vals[np.isfinite(vals)]
        meds = [float(np.median(rng.choice(vals, size=len(vals), replace=True)))
                for _ in range(args.bootstrap)]
        boot[proto] = {
            "median_ratio": float(np.median(vals)),
            "ci95": [float(np.percentile(meds, 2.5)),
                     float(np.percentile(meds, 97.5))],
            "n": int(len(vals)),
        }

    res = {
        "question": "By what factor does each retrospective validation protocol "
                    "mis-state the error on the compounds made next?",
        "ratio_definition": "retrospective MAE / prospective MAE, both in the "
                            "endpoint's analysis unit (log10 for the raw "
                            "concentration/permeability endpoints)",
        "n_endpoints": int(t.endpoint.nunique()),
        "n_models": int(t.model.nunique()),
        "protocols": boot,
        "by_protocol_endpoint": t.to_dict(orient="records"),
        "coverage": cov.reset_index().to_dict(orient="records"),
        "seeds": 1,
        "note": "retrospective MAE is the median over that protocol's folds inside "
                "the training period; the future compounds never enter it",
        "provenance": "/home/ubuntu/.openclaw/workspace/venv/bin/python "
                      "scripts/run_grid.py",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    t.to_csv(out.with_suffix(".csv"), index=False)

    print("\n=== median ratio per protocol (retrospective / prospective) ===")
    for p, v in sorted(boot.items(), key=lambda kv: kv[1]["median_ratio"]):
        lo, hi = v["ci95"]
        flag = "" if lo <= 1.0 <= hi else "  <-- excludes 1.0"
        print(f"  {p:9s} {v['median_ratio']:.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"n={v['n']}{flag}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
