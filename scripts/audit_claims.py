#!/usr/bin/env python3
"""Check every number the README quotes against results/grid.json.

The sibling repos learned this the hard way: prose is where numbers rot, and an audit
that only searches for a substring certifies claims it never checked. So each claim
here is ANCHORED - it names the phrase that must immediately precede the value - and a
claim whose anchor no longer exists is a failure, not a silent pass.

Exit 0 when every claim holds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GRID = RES / "grid.json"
CSV = RES / "grid.csv"

# every artefact the README depends on; losing one fails rather than shrinks
REQUIRED = ("grid.json", "grid.csv")

WINDOW = 60
NUM = re.compile(r"(?<![\w.])([+\-\u2212]?\d[\d,]*(?:\.\d+)?%?)")


def _signed(v: float, fmt: str) -> list[str]:
    s = fmt.format(v)
    bare = s.lstrip("+-")
    if float(s) == 0:
        return [bare, "+" + bare, "-" + bare]
    if s.startswith("-"):
        return [s]
    return [bare, "+" + bare]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()

    readme = Path(args.readme).read_text()
    missing_artefacts = [n for n in REQUIRED if not (RES / n).exists()]
    if missing_artefacts:
        print(f"MISSING ARTEFACT ({len(missing_artefacts)}): "
              f"{', '.join(missing_artefacts)}")
        print("The claims that depend on it cannot be checked, which is a failure.")
        return 1

    d = json.loads(GRID.read_text())
    prot = d["protocols"]
    rows = d["by_protocol_endpoint"]

    claims: list[tuple[str, list[str], str, str]] = []  # label, spellings, source, anchor

    # ---- the headline table, one row per protocol
    anchors = {
        "random": r"\| random \(shuffled rows\) \| \*\*",
        "scaffold": r"\| scaffold \(Murcko\) \| \*\*",
        "time": r"\| time \(within history\) \| ",
        "butina": r"\| butina \(cluster\) \| ",
        "frontier": r"\| frontier \(least similar\) \| ",
    }
    for p, v in prot.items():
        claims.append((f"{p} median ratio",
                       [f"{v['median_ratio']:.2f}"],
                       f"grid.json protocols.{p}.median_ratio",
                       anchors[p]))
        lo, hi = v["ci95"]
        claims.append((f"{p} CI",
                       [f"{lo:.2f}\u2013{hi:.2f}", f"{lo:.2f}-{hi:.2f}"],
                       f"grid.json protocols.{p}.ci95",
                       anchors[p]))

    # ---- the denominators
    claims.append(("n endpoints", [str(d["n_endpoints"])], "grid.json n_endpoints",
                   r"\*\*\d+ endpoints \u00d7 \d+ model"))
    claims.append(("n models", [str(d["n_models"])], "grid.json n_models",
                   r"\*\*\d+ endpoints \u00d7 \d+ model"))
    claims.append(("n cells", [str(len(rows))], "grid.json len(by_protocol_endpoint)",
                   r"(\d+) cells"))
    claims.append(("cells above floor",
                   [str(sum(1 for r in rows if not r["model_beats_floor"]))],
                   "grid.json model_beats_floor == False",
                   r"cells, \*\*"))

    # ---- per-endpoint random-split ratios, from the README's own table
    import pandas as pd
    t = pd.read_csv(CSV)
    rand = (t[t.protocol == "random"].groupby("endpoint")
            .agg(ratio=("ratio", "median"), pro=("prospective_mae", "median")))
    table_endpoints = {
        "Caco-2 Permeability Efflux": r"\| Caco-2 efflux \| \*\*",
        "Caco-2 Permeability Papp A>B": r"\| Caco-2 Papp A>B \| ",
        "LogD": r"\| LogD \| ",
        "KSOL": r"\| KSOL \| ",
        "MLM CLint": r"\| MLM CLint \| ",
        "HLM CLint": r"\| HLM CLint \| ",
        "MBPB": r"\| MBPB \| ",
        "MGMB": r"\| MGMB \| ",
        "MPPB": r"\| MPPB \| ",
    }
    for ep, anchor in table_endpoints.items():
        ratio = f"{rand.loc[ep, 'ratio']:.2f}"
        # A row holds TWO numbers, so one anchor cannot check both. The ratio is the
        # first token after the row label; the prospective MAE is the first token
        # after the (now bolded) ratio. Two anchors, one value each - precise, and
        # either one going stale fails.
        claims.append((f"{ep} row ratio", [ratio],
                       f"grid.csv {ep} random median ratio", anchor))
        claims.append((f"{ep} row prospective MAE",
                       [f"{rand.loc[ep, 'pro']:.2f}"],
                       f"grid.csv {ep} prospective median",
                       anchor + re.escape(ratio) + r"\*{0,2} \| "))

    # ---- coverage claims, from the coverage block
    cov = {c["endpoint"]: c for c in d["coverage"]}
    for ep, anchor in (("HLM CLint", r"HLM CLint is\s+measured on \d+%"),
                       ("Caco-2 Permeability Papp A>B", r"both\s+Caco-2 endpoints go")):
        c = cov[ep]
        claims.append((f"{ep} train coverage",
                       [f"{round(100 * c['train_frac'])}%"],
                       f"grid.json coverage {ep} train_frac", anchor))
    claims.append(("HLM test coverage", [f"{round(100 * cov['HLM CLint']['test_frac'])}%"],
                   "grid.json coverage HLM CLint test_frac",
                   r"training compounds but only "))
    claims.append(("Caco Papp test coverage",
                   [f"{round(100 * cov['Caco-2 Permeability Papp A>B']['test_frac'])}%"],
                   "grid.json coverage Caco-2 Papp test_frac",
                   r"the other way \(\d+% \u2192 "))

    # ---- evaluate
    missing = []
    unanchored = 0
    for label, spellings, source, anchor in claims:
        if not anchor:
            unanchored += 1
            if not any(s in readme for s in spellings):
                missing.append((label, spellings, source, "no anchor"))
            continue
        m = re.search(anchor, readme)
        if not m:
            missing.append((label, spellings, source,
                            f"anchor {anchor!r} matched nowhere"))
            continue
        window = readme[m.start():m.start() + WINDOW]
        if not any(s in window for s in spellings):
            got = NUM.search(window)
            missing.append((label, spellings, source,
                            f"anchor matched but window has none of them; "
                            f"first token {got.group(1) if got else 'none'}"))

    total = len(claims)
    print(f"{total - len(missing)}/{total} claims hold in place"
          + (f"; {unanchored} unanchored" if unanchored else
             "; all anchored to a phrase"))
    if missing:
        print("\nWRONG, MOVED OR MISSING:")
        for label, spellings, source, why in missing:
            print(f"  {label:34s} expected {spellings} ({source}) :: {why}")
        print("\nEither the README is stale or the artefacts changed. Neither ships.")
        return 1
    print("every published number matches its committed artefact, in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
