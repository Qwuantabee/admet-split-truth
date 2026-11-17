#!/usr/bin/env python3
"""Third model family: re-run the SAME grid with a message-passing network.

Nothing about the measurement changes. The loader, the time-order check, the five
protocols and the ratio definition are imported from the modules that already define
them (`admet_split.data`, `admet_split.splits`, and the feature/cluster helpers in
`run_grid.py`). A second definition of any of those would be a bug, so there is none
here: this script only swaps the estimator.

    ratio = retrospective MAE / prospective MAE          (per endpoint x protocol)

Writes results/gnn_grid.json, which carries three things:
  * the GNN's own cells and per-protocol median ratios,
  * the protocol ratios RE-DERIVED over all three families together,
  * a head-to-head of prospective MAE, GNN vs random forest, because "does the graph
    model actually predict better" is a different question from "is the protocol
    honest about it" and both are reported.

If the GNN does not beat the ECFP4 forest, that ships as the null.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from admet_split import data as D        # noqa: E402
from admet_split import gnn as G         # noqa: E402
from admet_split import splits as S      # noqa: E402
import run_grid as RG                    # noqa: E402  - reuse, never redefine

PROTOCOLS = RG.PROTOCOLS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "gnn_grid.json"))
    ap.add_argument("--endpoints", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    t0 = time.time()
    frame = D.load()
    tr, te = frame.train, frame.test
    eps = args.endpoints or D.ENDPOINTS

    # fingerprints define the protocol structure (scaffold / butina / frontier) exactly
    # as in the published grid; graphs are only the new feature view.
    Xtr_all, keep_tr = RG.morgan_fps(tr["SMILES"])
    Xte_all, keep_te = RG.morgan_fps(te["SMILES"])
    gtr_all, gkeep_tr = G.featurize(tr["SMILES"])
    gte_all, gkeep_te = G.featurize(te["SMILES"])
    if not (np.array_equal(keep_tr, gkeep_tr) and np.array_equal(keep_te, gkeep_te)):
        raise SystemExit("graph and fingerprint featurisers kept different molecules; "
                         "the two grids would not be comparable")
    tr = tr.iloc[keep_tr].reset_index(drop=True)
    te = te.iloc[keep_te].reset_index(drop=True)
    print(f"molecules parsed: train {len(tr)}, test {len(te)}", flush=True)

    scaff = RG.murcko_scaffolds(tr["SMILES"])
    clus = RG.butina_clusters(Xtr_all)
    order = np.array([int("".join(ch for ch in str(x) if ch.isdigit()) or -1)
                      for x in tr["Molecule Name"]])
    fsim = RG.frontier_similarity(Xtr_all)
    print(f"protocol structure ready ({time.time() - t0:.0f}s)", flush=True)

    def fit(gs_tr, y_tr, gs_te):
        return G.fit_predict_gnn(gs_tr, y_tr, gs_te, seed=0, epochs=args.epochs)

    rows = []
    for ep in eps:
        ytr = pd.to_numeric(tr[ep], errors="coerce").to_numpy(dtype=float)
        yte = pd.to_numeric(te[ep], errors="coerce").to_numpy(dtype=float)
        ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
        if ok_tr.sum() < 200 or ok_te.sum() < 100:
            print(f"  {ep}: skipped (train {ok_tr.sum()}, test {ok_te.sum()})")
            continue
        idx_tr = np.flatnonzero(ok_tr)
        idx_te = np.flatnonzero(ok_te)
        A = [gtr_all[i] for i in idx_tr]
        B = [gte_all[i] for i in idx_te]
        ya, yb = ytr[idx_tr], yte[idx_te]

        pro_mae = RG._mae(fit(A, ya, B), yb)
        floor_mae = RG._mae(RG.one_feature(A, ya, B), yb)

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
                pred = fit([A[i] for i in a], ya[a], [A[i] for i in b])
                per_fold.append(RG._mae(pred, ya[b]))
            if not per_fold:
                continue
            ret_mae = float(np.median(per_fold))
            rows.append({
                "endpoint": ep, "model": "gnn", "protocol": proto,
                "retrospective_mae": ret_mae, "prospective_mae": pro_mae,
                "ratio": (ret_mae / pro_mae) if pro_mae > 0 else float("nan"),
                "n_folds": len(per_fold),
                "train_n": int(ok_tr.sum()), "test_n": int(ok_te.sum()),
                "prospective_floor_mae": floor_mae,
                "model_beats_floor": bool(pro_mae < floor_mae),
            })
            print(f"  {ep:26s} gnn    {proto:9s} retro {ret_mae:8.3f} "
                  f"pro {pro_mae:8.3f} ratio {ret_mae / pro_mae:5.2f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("no GNN rows produced")

    def boot(frame_in: pd.DataFrame) -> dict:
        rng = np.random.default_rng(11)          # same procedure as run_grid.py
        out = {}
        for proto, g in frame_in.groupby("protocol"):
            vals = g.ratio.to_numpy()
            vals = vals[np.isfinite(vals)]
            meds = [float(np.median(rng.choice(vals, size=len(vals), replace=True)))
                    for _ in range(args.bootstrap)]
            out[proto] = {"median_ratio": float(np.median(vals)),
                          "ci95": [float(np.percentile(meds, 2.5)),
                                   float(np.percentile(meds, 97.5))],
                          "n": int(len(vals))}
        return out

    old = pd.read_csv(ROOT / "results" / "grid.csv")
    combined = pd.concat([old, t], ignore_index=True)

    # head-to-head on the quantity a team cares about: error on the next compounds
    rf = (old[old.model == "rf"].groupby("endpoint").prospective_mae.median())
    gn = t.groupby("endpoint").prospective_mae.median()
    head = []
    for ep in gn.index:
        if ep in rf.index:
            head.append({"endpoint": ep, "rf_prospective_mae": float(rf[ep]),
                         "gnn_prospective_mae": float(gn[ep]),
                         "gnn_better": bool(gn[ep] < rf[ep])})
    n_better = sum(h["gnn_better"] for h in head)

    res = {
        "question": "Does a third, non-fingerprint model family change how far each "
                    "validation protocol mis-states the error on the next compounds?",
        "ratio_definition": "retrospective MAE / prospective MAE, identical to "
                            "results/grid.json; splits, loader and ratio are imported "
                            "from the same modules",
        "model": {"family": "message-passing neural network on the molecular graph",
                  "hidden": 96, "layers": 3, "epochs": args.epochs,
                  "readout": "mean+sum", "loss": "L1 on standardised target",
                  "seed": 0, "tuning": "none - one fixed configuration"},
        "n_endpoints": int(t.endpoint.nunique()),
        "protocols_gnn_only": boot(t),
        "protocols_all_families": boot(combined),
        "n_cells_gnn": int(len(t)),
        "n_cells_all": int(len(combined)),
        "n_models_all": int(combined.model.nunique()),
        "head_to_head_vs_rf": head,
        "n_endpoints_gnn_beats_rf": int(n_better),
        "by_protocol_endpoint": t.to_dict(orient="records"),
        "provenance": "/home/ubuntu/.openclaw/workspace/venv/bin/python "
                      "scripts/run_gnn_grid.py",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    t.to_csv(out.with_suffix(".csv"), index=False)

    print("\n=== GNN only: median ratio per protocol ===")
    for p, v in sorted(res["protocols_gnn_only"].items(),
                       key=lambda kv: kv[1]["median_ratio"]):
        lo, hi = v["ci95"]
        flag = "" if lo <= 1.0 <= hi else "  <-- excludes 1.0"
        print(f"  {p:9s} {v['median_ratio']:.3f}  [{lo:.3f}, {hi:.3f}]  n={v['n']}{flag}")
    print("\n=== all three families: median ratio per protocol ===")
    for p, v in sorted(res["protocols_all_families"].items(),
                       key=lambda kv: kv[1]["median_ratio"]):
        lo, hi = v["ci95"]
        flag = "" if lo <= 1.0 <= hi else "  <-- excludes 1.0"
        print(f"  {p:9s} {v['median_ratio']:.3f}  [{lo:.3f}, {hi:.3f}]  n={v['n']}{flag}")
    print(f"\nGNN beats the forest prospectively on {n_better}/{len(head)} endpoints")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
