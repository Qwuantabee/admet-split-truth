# ROADMAP — what is measured, what is next, and the rules a next step obeys

The headline shipped first (commit `2f7b058`): two validation protocols under-state
future ADMET error on this program, three do not. A third model family followed
(`5f5489c`). This file says what would come next and the rules that keep each
increment honest.

## Rules

1. **Ground truth comes from disk.** `scripts/loop_step.py` decides the next step from
   which artefacts exist, not from a ledger file. It cannot drift or repeat itself.
2. **A terminal state.** When the queue is exhausted the step script returns DONE.
   There is no step that re-reports finished work.
3. **Claims are audited, not trusted.** `scripts/audit_claims.py` binds every README
   number to the phrase that must precede it, so a number cannot be satisfied by the
   same digits appearing somewhere else in the file. A step is not done because the
   prose says so.
4. **A null is a result.** Each increment either measures something or records why it
   could not. "No effect" ships as the finding.
5. **Never loosen a test.** A red test is information. A test that is genuinely wrong
   is fixed in a commit whose message carries `TEST-CHANGE: <reason>`.

## The queue (in order)

- `gnn-family` — **done** (`5f5489c`). A message-passing network on the molecular graph
  as a third family, to test whether the five protocols only flatter fingerprints.
  Result: they do not. The random-split ratio is below 1.0 for every family on every
  endpoint, and the graph model is not the better predictor.
- `second-program` — repeat the measurement on a second public dataset with a time
  split. Gate: the same time-order verification the loader enforces, and a stated
  reason if no such public split exists. This is the biggest open weakness; every
  number here is one program.
- `seeds` — repeat the grid over several seeds and fold assignments. Every ratio
  currently comes from one run, so the spread within a cell is unmeasured.
- `power` — for each protocol, the smallest endpoint sample size at which the ratio's
  interval stops containing 1.0. Answers "how many compounds must I have before my
  split tells me anything".
- `docs` — a short methods note on why a ratio of medians rather than a regression, the
  exact assumption that ID order is time order, and why the protocols' differing
  training-set sizes matter when the ratios are compared.

## What a step does

1. `python scripts/loop_step.py` — prints the next step from what is on disk.
2. Do exactly that step. Commit by explicit path; never `git add -A`.
3. `python scripts/audit_claims.py --strict` and `pytest tests/` must both pass, and
   the output goes in the commit body. A step is done on evidence, not intent.

## Honesty notes for any next step

- The frontier-vs-Zheng-2026 disagreement is a finding about *what was measured*, not a
  refutation of their paper. Do not "resolve" it by reinterpreting their result. Note
  also that the frontier protocol here is a simpler construction than theirs, and fits
  on a sixth of the data.
- The single-program limitation is the biggest weakness. State it whenever a new number
  is added; a ratio from one chemotype is a hypothesis about others.
- Ridge and random forest are noise-correlated (both read the same fingerprint). The
  graph family answered that objection; it did not answer the one-program objection.
