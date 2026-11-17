# DESIGN — does your validation split predict the next batch of compounds?

Status: feasibility gate PASSED (2026-11-17). Headline number measured on a pilot,
full grid not yet run.

## The question

A medicinal-chemistry team builds an ADMET model, validates it internally, and then
decides how much to trust it before committing synthesis money. Every team argues
about which split to validate on: random, scaffold, Butina cluster, or a time split.
The literature reports which split gives the *highest* retrospective score.

Nobody reports the thing the team actually needs: **by what factor does each
validation protocol mis-state the error on the compounds they make next?**

## Why this is not already answered

- Zheng 2026 (arXiv:2607.10729) measures frontier-vs-scaffold inflation
  (taskwise median 87%, interval 52.1–246.0%) — **retrospective only**. It shows
  splits disagree with each other, not which one matches reality.
- The OpenADMET–ExpansionRx blind challenge (Castellanos 2026,
  ChemRxiv 10.26434/chemrxiv.15009342) reports a leaderboard and notes anecdotally
  that internal CV "did not correspond to leaderboard performance" — no number.
- Zalte 2026 (bioRxiv 10.64898/2026.08.19.745868) calls for evaluation that connects
  to real use. It does not do it.

The ExpansionRx public release makes this answerable for the first time: a real
lead-optimisation program with a **strictly time-ordered** public split and labels on
both sides. That is the gap.

## The measurement

For each (endpoint × model family × validation protocol):

```
ratio = retrospective MAE / prospective MAE        (correctly signed for log10 endpoints)
```

- **prospective MAE** — trained on the train period, scored once on the test period.
  This is the only number that corresponds to the decision the team makes.
- **retrospective MAE** — the same model family, validated *within* the train period
  under the protocol being tested.

A ratio of 0.7 means the protocol told the team its error was 30% smaller than it
turned out to be. A ratio of 1.3 means the protocol was pessimistic.

Protocols: `random`, `scaffold`, `butina`, `time` (within-history time split), and
`frontier` (structural frontier, per Zheng 2026) where the data support it.

## What the pilot already showed (feasibility, not the result)

RF on 2048-bit ECFP4, 80 trees, single seed, 3 endpoints, one run:

| endpoint | retrospective MAE | prospective MAE | ratio |
|---|---|---|---|
| LogD | 0.425 | 0.637 | **0.67** |
| KSOL | 55.03 | 90.86 | **0.61** |
| MLM CLint | 373.1 | 311.1 | **1.20** |

**The sign of the bias flips by endpoint.** Two endpoints flatter the model and one
punishes it. That is the finding worth publishing: a single "multiply your CV error by
1.5" correction is wrong, and the direction is not predictable from the endpoint name.
This must survive the full grid before it is claimed.

## Data (verified myself, 2026-11-17)

`openadmet/openadmet-expansionrx-challenge-data` on Hugging Face.

- `gated: false`, licence `cc-by-4.0`, `private: false` (checked via the HF API).
- `expansion_data_train.csv` 464,076 B — 5,326 compounds, IDs `0001321`…`0020100`.
- `expansion_data_test.csv` 221,291 B — 2,282 compounds, IDs `0020101`…`0027239`.
- **Zero ID overlap, and every test ID exceeds every train ID** — the split is
  time-forward, which is the premise of the whole project. Verified directly.
- 9 endpoints: LogD, KSOL, HLM CLint, MLM CLint, Caco-2 Papp A>B, Caco-2 Efflux,
  MPPB, MBPB, MGMB.
- Test carries labels, so prospective error is measurable.
- 523 of 2,282 test compounds have all 6 primary endpoints measured (the joint-pass
  denominator, if that secondary analysis is run).

**Distribution shift is real, not incidental:** KSOL test median 249 vs train 131;
Caco-2 Papp 3.23 vs 9.82; Efflux 4.99 vs 1.38. This is exactly why prospective error
differs from retrospective, and it is a property of the data, not a modelling choice.

## Compute

CPU only for the headline. RF/LightGBM on ECFP4: the pilot ran 3 endpoints × (1
prospective + 5 folds) in well under a minute on 12 cores. The full grid
(9 endpoints × 3 model families × 5 protocols × 5 seeds + 1000 bootstrap resamples)
is a few CPU-hours. No GPU is required; an optional Chemprop MPNN leg (~0.3M params,
<2 GB VRAM) can be added as a second model family if time allows.

## Figures

1. `ratio_by_protocol.png` — the headline: per-protocol ratio with interval, faceted
   by endpoint, with the 1.0 line marked.
2. `shift_vs_ratio.png` — does the size of the train/test distribution shift predict
   how badly a protocol misleads? (Honest possibility: no.)
3. `calibration_curve.png` — predicted-vs-actual error per protocol.

## Risks, and what a null looks like

- **Null A (most likely):** every protocol's ratio intervals include 1.0, i.e. on this
  program no split mis-states prospective error detectably. That is publishable and
  useful: it would say the split debate is moot at n≈2,300 with this shift.
- **Null B:** the ratios are uniform across protocols (all ≈0.7). Then the message is
  "all retrospective protocols flatter you equally", which is still a single
  actionable number.
- **Confound:** one program, one chemotype (cationic benzimidazole-like RNA binders),
  so ratios may not transfer. Must be stated in the limits, not buried.
- **Confound:** the ID-order-as-time assumption. Verified only as "all test IDs exceed
  all train IDs"; the dataset card says test = later-stage molecules. State it as an
  assumption with the evidence, not as fact.
- **Missingness:** assays are run in sequence, so the measured subset is selected.
  Report coverage per endpoint; do not impute.

## Honesty rails (this repo's whole point)

- Every published number comes from a committed artefact via `scripts/audit_claims.py`.
- Every model comparison carries a null control (a shuffled-target and a
  one-feature baseline).
- A null result ships as the result.
- No number appears in prose that is not in `results/`.

## What this looks like to a hiring manager

Computational ADMET / ML modelling scientist who owns **validation methodology** —
the person a pharma modelling team puts in front of a decision about whether to trust
a model. The headline is a single ratio with an interval, not a leaderboard climb.
