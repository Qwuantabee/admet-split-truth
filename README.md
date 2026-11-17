# Does your validation split predict the next batch of compounds?

**On a real lead-optimisation program with a time-ordered public split, a random split
under-states the error on the compounds made next by a factor of 0.76, and a Murcko
scaffold split by 0.84. Both are measured against the truth, not against each other.**

![Retrospective vs prospective error by protocol](figures/ratio_by_protocol.png)

Every ADMET modelling team validates internally and then decides how far to trust the
model before committing synthesis money. The literature argues about *which split*
gives the best score — random, scaffold, cluster, or time. Nobody reports the quantity
the team actually consumes:

> If my validation says the error is 0.5 log units, what error will I see on the next
> compounds I make?

This measures that directly, on the one public dataset where it is answerable.

## The measurement

| | |
|---|---|
| fields reported today | retrospective error, from a split chosen inside the training data |
| what a project team consumes | error on the compounds produced *after* validation |
| the quantity reported here | **retrospective MAE ÷ prospective MAE**, per protocol |

A ratio of 0.76 means the protocol told the team its error was 24% smaller than it
turned out to be. A ratio of 1.0 is calibrated. Above 1.0 is pessimism.

Prospective MAE is computed **once** per endpoint × model, by fitting on the whole
training period and scoring the later compounds. It never enters a retrospective number.

| protocol | median ratio | 95% CI | |
|---|---|---|---|
| random (shuffled rows) | **0.76** | 0.67–0.84 | excludes 1.0 — optimistic |
| scaffold (Murcko) | **0.84** | 0.70–0.95 | excludes 1.0 — optimistic |
| time (within history) | 1.02 | 0.91–1.11 | calibrated |
| butina (cluster) | 1.04 | 0.85–1.26 | calibrated |
| frontier (least similar) | 1.04 | 0.88–1.29 | calibrated |

Medians over **9 endpoints × 2 model families** (ridge on ECFP4 and a random forest on
ECFP4), 88 cells, 95% percentile bootstrap over the cell ratios.

Three things follow.

**The ranking of protocols is the ranking of honesty.** The two protocols most teams
reach for first — a shuffled split, and scaffold grouping — are exactly the two that
under-state future error. The three that hold up are the ones that respect either time
or chemical distance.

**Scaffold splitting is better than random, but not calibrated.** It removes roughly
half the optimism (0.76 → 0.84) and still lands below 1.0 with an interval that
excludes it. "We used scaffold split" is not a warrant that the number transfers.

**A frontier split is not pessimistic here.** Zheng 2026 reports structural-frontier
evaluation as the most pessimistic protocol, with a taskwise median inflation of 87%.
On this program it is the *most* calibrated of the five (1.04, interval spanning 1.0).
That disagreement is reported rather than smoothed: their study measured
retrospective splits against each other, and this one measures against future data.
The two questions are not the same, and on this data the answer differs.

## Where the bias is largest

The optimism is not uniform. Per-endpoint medians, random split:

| endpoint | ratio | prospective MAE (log10) |
|---|---|---|
| Caco-2 efflux | **0.43** | 0.52 |
| Caco-2 Papp A>B | 0.57 | 0.58 |
| LogD | 0.62 | 0.65 |
| KSOL | 0.73 | 0.55 |
| MLM CLint | 0.74 | 0.47 |
| HLM CLint | 0.83 | 0.42 |
| MBPB | 0.84 | 0.30 |
| MGMB | 0.85 | 0.25 |
| MPPB | 0.94 | 0.29 |

The permeability endpoints — the ones with the largest measured distribution shift
between periods — are the ones a random split lies about most. That is the mechanism,
and it is visible without fitting anything: Caco-2 Papp shifts from a training median
of 9.82 to 3.23 in the test period, and efflux from 1.38 to 4.99.

## The honest floor

A protocol that cannot beat "predict the training mean" is not measuring the model.
Across the 88 cells, **15** have a prospective MAE at or above that floor. For two
endpoints the model never beats it: **HLM CLint** (0.42 against a floor of 0.38) and
**MLM CLint** (0.47 against 0.42). Their ratios are reported but should not be read as
a statement about a working model — there is no working model there to mis-state.

![Ratio against endpoint sample size](figures/shift_vs_ratio.png)

The obvious alternative explanation for the headline is sample size: perhaps the
optimism is just what small endpoints do. The panel above puts every cell's ratio
against the training-period sample size for that endpoint, and the points scatter
rather than trending. Sample size is not the explanation, and this figure exists so a
reader does not have to take that on trust.

## Data

`openadmet/openadmet-expansionrx-challenge-data` (Hugging Face), **CC-BY-4.0, not
gated** — verified through the HF API and by downloading both files.

- 5,326 training compounds, IDs `0001321`–`0020100`.
- 2,282 test compounds, IDs `0020101`–`0027239`.
- **Zero ID overlap, and every test ID exceeds every train ID.** The loader raises if
  that ever stops being true, because the whole project rests on it.
- 9 endpoints. Coverage is uneven and the test order is not random: HLM CLint is
  measured on 67% of training compounds but only 34% of test compounds, while both
  Caco-2 endpoints go the other way (40% → 71%). Assays are run in sequence, so the
  measured subset is selected; coverage is reported per endpoint and nothing is
  imputed.

## Reproducing

```bash
pip install -r requirements.txt

# the grid (downloads the two CSVs into data/raw on first run)
python scripts/run_grid.py

# figures, then the claim audit and the tests
python scripts/make_figures.py
python scripts/audit_claims.py
pytest tests/
```

The audit is also a test, so CI fails on a stale number as well as a broken function.

## Limits, stated plainly

- **One program, one chemotype.** ExpansionRx is a single lead-optimisation campaign
  (cationic benzimidazole-like RNA binders). The ratios may not transfer to another
  series, and nothing here claims they do. The contribution is the *method* and the
  direction of the bias, which a team can re-measure on its own data with this code.
- **ID order is treated as time order.** Evidence: the two ID ranges are disjoint and
  strictly ordered, and the dataset card describes the test set as later-stage
  molecules. That is an assumption with support, not a timestamp.
- **The intervals are over cells, not over programs.** With 9 endpoints and 2 models
  from a single program, the 95% bootstrap describes variability across
  endpoint × model combinations, and does not account for between-program variation.
  Treat the intervals as a spread, not a claim about a new program.
- **Two model families, both fingerprint-based.** Ridge and random forest on 2048-bit
  ECFP4. A graph neural network might be mis-stated differently by the same protocols;
  that is untested here and is the obvious next experiment.
- **The frontier protocol uses a sampled similarity estimate** (400 reference
  compounds) to order the folds. It affects fold composition, not correctness, but it
  is an approximation stated rather than hidden.
- **The null result was a live possibility and did not occur.** Had all five intervals
  spanned 1.0, that would have shipped as the finding. Two do span it, and they are
  reported that way.

## Layout

```
src/admet_split/data.py     loader, time-order verification, log10 scaling, coverage
src/admet_split/splits.py   the five validation protocols
scripts/run_grid.py         the measurement: retrospective vs prospective MAE
scripts/make_figures.py     figures from the committed artefact
scripts/audit_claims.py     every README number against results/grid.json
tests/                      time-order invariant, protocol invariants, the claim audit
```

MIT licence for this repository's own code. The ExpansionRx data keeps its own terms
(CC-BY-4.0).
