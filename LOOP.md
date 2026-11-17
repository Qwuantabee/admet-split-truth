# LOOP — how this project advances without a human in the loop

The headline shipped (commit `2f7b058`): two validation protocols under-state future
ADMET error, three are calibrated. This file defines what comes next and the rails that
keep an autonomous loop honest.

## Rails

1. **Ground truth comes from disk.** `scripts/loop_step.py` decides the next step from
   what exists on disk, never from `LOOP_STATE.json`. A turn cannot drift or repeat.
2. **A terminal state.** When the queue is exhausted the step script returns DONE and
   the automation must be disabled. A loop that keeps re-reporting its last step is the
   failure mode this repo's sibling hit twice.
3. **Claims are audited, not trusted.** `scripts/audit_claims.py` binds every README
   number to the phrase that must precede it. A step is not done because prose says so.
4. **A null is a result.** Every increment either measures something or explicitly
   records why it could not. "No effect" ships as the finding.
5. **Never loosen a test.** A red test is information. A test that is genuinely wrong is
   fixed in a commit whose message carries `TEST-CHANGE: <reason>`.

## The queue (in order)

- `gnn-family` — add a third model family (a message-passing network on the molecular
  graph) and re-run the grid. The README states that a GNN might be mis-stated
  differently by the same protocols, and that this is the obvious untested case.
  Gate: prospective MAE computed the same way; the protocol ratios re-derived; the
  claim audit updated. If a GNN does not beat the ECFP4 forest, that ships as the null.
- `second-program` — repeat the measurement on a second public dataset with a time
  split. Gate: the same time-order verification the loader enforces, and a stated
  reason if no such public split exists.
- `power` — for each protocol, the smallest endpoint sample size at which the ratio's
  interval stops containing 1.0. Answers "how many compounds must I have before my
  split tells me anything".
- `docs` — a short methods note on why ratio-of-medians rather than a regression, and
  the exact assumption that ID order is time order.

## What a fire does

1. `python scripts/loop_step.py` — prints the true next step from disk.
2. Do exactly that step. Commit by explicit path; never `git add -A`; never push without
   the gates passing.
3. `python scripts/audit_claims.py --strict` and `pytest tests/` must both pass, and the
   output must be pasted into the commit body. A step is done on evidence, not intent.
4. Update the ledger. If the step exhausted the queue, disable the automation.

## Honesty notes for any future fire

- The frontier-vs-Zheng-2026 disagreement is a genuine finding about *what was measured*,
  not a refutation of their paper. Do not "resolve" it by reinterpreting their result.
- The single-program limitation is the biggest weakness. State it whenever a new number
  is added; a ratio from one chemotype is a hypothesis about others.
- Two model families are noise-correlated (both fingerprint-based). A third family is
  the cheapest real test of robustness, which is why it is first in the queue.
