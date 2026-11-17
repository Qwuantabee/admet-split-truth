#!/usr/bin/env python3
"""What is the TRUE next step? Read from disk, never from the ledger.

Prints JSON: {"step": N, "step_name": ..., "next": ...}. A fire obeys it and does
nothing else. When the queue is exhausted it returns DONE so the automation can be
disabled instead of re-reporting finished work.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "LOOP_STATE.json"
PY = sys.executable


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"step": 0}


def save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def audit_ok() -> bool:
    p = subprocess.run([PY, "scripts/audit_claims.py", "--strict"], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode == 0


def tests_ok() -> bool:
    import os

    env = {**os.environ, "PYTHONPATH": "src"}
    p = subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    return p.returncode == 0


def detect() -> tuple[int, str]:
    """Judge the next step from artefacts on disk.

    The gates come first: a broken audit or a red test means the last step was not
    actually finished, whatever the ledger says.
    """
    if not (ROOT / "results" / "grid.json").exists():
        return 0, "BUILD"          # the headline does not exist yet
    if not audit_ok():
        return 1, "AUDIT"          # prose drifted from the artefact
    if not tests_ok():
        return 2, "FIX"            # a red test is information; fix it
    # queue, in order
    if not (ROOT / "results" / "gnn_grid.json").exists():
        return 3, "GNN"
    if not (ROOT / "results" / "second_program.json").exists():
        return 4, "SECOND_PROGRAM"
    if not (ROOT / "results" / "power.json").exists():
        return 5, "POWER"
    if not (ROOT / "docs" / "METHODS.md").exists():
        return 6, "DOCS"
    return 99, "DONE"


NEXT = {
    "BUILD": "python scripts/run_grid.py  # the headline artefact is missing",
    "AUDIT": "fix the README so audit_claims.py --strict exits 0",
    "FIX": "fix the failing test; never delete or loosen it",
    "GNN": "add a graph-network model family and re-run the grid into "
           "results/gnn_grid.json",
    "SECOND_PROGRAM": "repeat the measurement on a second public dataset with a "
                      "verified time split",
    "POWER": "smallest endpoint sample size where each protocol's interval "
             "excludes 1.0",
    "DOCS": "write docs/METHODS.md: why a ratio of medians, and the ID-order "
            "assumption",
    "DONE": "queue exhausted - disable this automation",
}


def main() -> int:
    st = load()
    step, name = detect()
    st.update(step=step, step_name=name, checked_at=now())
    out = {"step": step, "step_name": name, "next": NEXT[name]}
    if name == "DONE":
        out["done"] = True
    save(st)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
