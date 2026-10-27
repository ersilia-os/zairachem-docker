# Regression in ZairaChem

`zairachem fit` on a continuous target (or `-r`, which forces it) trains a model that predicts the target value itself. This note records what
that means in practice, and why the regression pipeline is shaped the way it is — mostly by what
lazy-qsar does, and by what simply has no meaning once the target is continuous.

## What regression does not have

A classification run scores every molecule as a probability, positions it against a reference
library of 50,000 drug-like molecules to get a **rank**, and calls it active or inactive at a named
**cutoff**. None of that carries over, and none of it is implemented:

| Classification | Regression |
| --- | --- |
| Probability in [0, 1] | Value in the target's own units |
| Rank against the 50k reference library | — |
| `screening` / `kappa` / `balanced` / `default` cutoffs, in `cutoffs.json` | — |
| Binary call (`call` column) | — |
| — | Conformal prediction interval (`lower`, `upper`) |

This is lazy-qsar's own position (`qsar_regressor`: "no prior, no reference library, no decision
cutoff"), not a simplification on our side. A value is already on the scale the chemist cares
about, so there is nothing to calibrate it against and no threshold that is anyone's to set.

The practical consequence: **no rank reference is downloaded or scored for a regression fit**,
which is a large time saving, and `results/cutoffs.json` is not written.

The report follows from the same place. A classification run carries a rank histogram, a ranked-
score operating-point curve, a threshold sweep, a calibration curve and six confusion panels; a
regression run carries none of them, because each answers "where does the cutoff fall?" and there
is no cutoff. What replaces them is not a translation of each panel but the two questions that are
left: how far off is a prediction (`regression-raw`, `regression-residuals`, `r2-individual`), and
how far can it be trusted (`interval-coverage`, `regression-error-similarity`, `value-consensus`).
The absence is the design, not a gap to fill later.

## Outputs

`results/output.csv` — one row per molecule:

| Column | Meaning |
| --- | --- |
| `value` | The pooled prediction, in the target's units |
| `similarity` | Mean Tanimoto similarity to the 5 nearest training molecules (predict only) |
| `spread` | Standard deviation of the per-descriptor predicted values (predict only) |
| `label` | The observed value, when the input has one |

`results/output_table.csv` adds the conformal interval (`lower`, `upper`), the per-descriptor
predictions (`value_<eos_id>`) and the 2-D projections.

A predict report opens with a hitlist of the 25 highest predicted values, which assumes the target
is one where **more is better** — a pChEMBL or a pIC50, not a raw IC50 in nM. ZairaChem has no
notion of a target's direction and does not try to infer one; on a lower-is-better assay, negate
or transform the column before fitting, as you would for any other QSAR model.

`results/performance_table.csv` is one row per model — the pooled model and each descriptor — with
`r2`, `rmse`, `mae`, `pearson` and `spearman`. There are no cutoff columns because there are no
cutoffs.

## How the descriptors are pooled

Classification weights each descriptor per compound, by its skill and by the reliability its rank
implies. Regression uses lazy-qsar 3.8's default instead: **one fixed weight per descriptor, fitted
by non-negative least squares on the out-of-fold predictions** (`stacking_weights`). The weights
sum to one, so the pooled prediction stays a weighted average of its members.

Two things survive from the classification path:

- **Pruning.** A descriptor whose held-out skill is below the floor, or more than a gap below the
  best one, is left out. The thresholds are the classification ones mapped onto Spearman's rho,
  as lazy-qsar scales them: `AUROC - 0.5` is about `rho / 2`, so the 0.55 AUROC floor becomes
  0.10 on rho. That floor is `base.vars.SKILL_FLOOR`, and the pre-screening proxy reads the same
  one — as does the caution a fit prints when the *pooled* model fails to clear it, or when its R²
  is at or below zero.
- **The applicability-domain veto.** A descriptor that is out of its domain for a given compound
  gets zero weight *for that compound*, stacking weights or not. A compound vetoed on every
  descriptor falls back to the unvetoed weights rather than producing nothing.

The rank-to-error reliability curves are still computed and stored even though stacking does not
consult them: they are the fallback when `restrict()` drops the stacking weights, which happens if
the descriptors available at predict are a subset of those trained.

## Prediction intervals

The pooled out-of-fold predictions are honest held-out predictions for every training row, so
their absolute residuals calibrate a **split-conformal** interval at no extra cost. The residuals
are split into bins by applicability-domain score (a Mondrian conformal predictor), so a compound
unlike the training set is handed a wider interval than one inside it.

The level is lazy-qsar's default, 0.9. On a 1,485-compound external set of *T. brucei* IC50 data
the 90% interval covered 91.2% — the guarantee holds on chemistry the model has never seen. The
`interval-coverage` panel checks it holds *within* each bin, which is what the Mondrian binning is
for; a bin that misses its level is the interesting failure.

## Which end of the target is the good one

ZairaChem does not ask, and does not assume. A classification run has a direction built in — the
probability is the probability of being active — but a continuous target does not: pIC50 runs one
way and IC50 in nM the other, and nothing in the input says which this is.

So everything in a regression report that is ordered is ordered by **predicted value, highest
first**, and says so rather than calling that end the best. The hitlist is the largest predicted
values; "Value by predicted rank" reports what the top of the ranking actually measured, against
the dataset mean. If a low value is the one you want, read both from the other end.

## What the error should be compared against

An RMSE is a number in the target's own units, and on its own it says nothing: 0.63 is excellent
for one assay and poor for another. The pipeline can measure the one thing that settles it.

Real bioactivity data measures the same compound more than once. Setup collapses those duplicate
rows into one compound by their median, and before doing so it records how far apart they were:
the **pooled within-compound standard deviation**, which is the assay's own reproducibility, in
the same units as the model's error. It goes to `inputs/replicates.json`.

On the *T. brucei* set used throughout this note, 5,771 IC50 rows cover 3,562 molecules, 1,397 of
them measured more than once — up to 53 times, and the widest disagreement for a single molecule
is 4.59 log units. The pooled spread is **0.618**.

A model fitted on 2,000 of those molecules predicts the 1,485 it never saw with an RMSE of
**0.634** (R² 0.787). That is the assay's own spread to two decimals, so the model is predicting
about as closely as the experiment measures, and the way to improve it is more or better
measurements rather than more modelling.

It is worth inverting the identity, because 0.634 against 0.618 reads as a model barely better
than noise and that is the opposite of what it is. An observed value is the truth plus measurement
noise of standard deviation `s`, so a model whose own error is `m` scores `RMSE² = m² + s²`; here
`m` is about **0.14**, a fifth of the noise it is being graded through. The fit summary prints
that number next to the spread. It assumes the measurement error is independent of the model's,
which is the assumption that gives the identity in the first place, and it is only as good as `s`
— a spread estimated from a handful of repeats is not worth decomposing.

Where this shows up:

- the fit summary prints the spread beside the RMSE, and says plainly when the error is at the
  limit of the data;
- the report's Dataset section carries the duplicate structure under **Repeat measurements**;
- `regression-raw` shades a band of one spread either side of the identity line, and its title
  gives the fraction of compounds predicted inside it.

A run whose input has no repeated compound reports none of this, rather than inventing a figure.

Two caveats the pipeline will tell you about but not fix. Deduplication matches the SMILES as
written, while standardisation runs after it, so two spellings of one molecule (`C1=CC=CC=C1C(=O)O`
and `c1ccccc1C(=O)O`) enter as two compounds and leave as one — they are then trained on twice, and
a held-out split can separate them, which flatters the score. Setup counts these and says so. And a
blank activity cell is not a measurement: it is dropped from its compound's group rather than
turning an otherwise measured compound into a missing one.

## Held-out validation

`--evaluate` works the same way as for classification, with two differences: the folds stratify on
**quantile bins of the target** rather than on class, and they report `r2`, `rmse`, `mae` and
`spearman` per fold.

The number worth reading is the gap between the random and scaffold splits. On the *T. brucei*
set, over two folds of each: random R² 0.804 ± 0.003, scaffold R² 0.669 ± 0.016 — the model
generalises, but noticeably less well to chemistry it has not seen a relative of.

The external set says the same thing from outside: trained on 2,000 compounds and asked for the
1,485 it had never seen, the model scored R² 0.787 and Spearman 0.871, against 0.780 and 0.856
out-of-fold. Held-out numbers that survive contact with a real external set are the point of all
of this; mean absolute error over that set fell from 0.598 in the least training-like quarter to
0.367 in the most.

## The report

Every panel answers a question, and they group by which question:

**Dataset** (what is being modelled, before any model)
- `value-distribution` — the observed values themselves, with the median marked. It stands where a
  classification run shows its class balance: a target piled against an assay's detection limit,
  or split into two clumps with nothing between them, is worth seeing before any error is read

**Accuracy** (needs the observed values, so fit and labelled predict)
- `regression-raw` — predicted against observed, with R² and MAE
- `regression-residuals` — residual against predicted: where the model is biased
- `r2-individual` — per-descriptor R²
- `potency-enrichment` — the mean *observed* value of the top k% by prediction. The screening
  question, and the one a correlation coefficient does not answer
- `interval-coverage` — does the interval keep its promise, and does it keep it in every bin

**Confidence** (needs no observed values, so it carries a prediction run)
- `regression-distribution` — the predicted values against the training distribution, on shared
  bins so the two are comparable bar by bar
- `regression-error-similarity` — error (or interval width) against similarity to the training set
- `value-consensus` — predicted value against how much the descriptors disagreed. Rendered only
  where there is no truth, as its classification counterpart is

**Chemistry**
- `projection-<name>-value` — each 2-D projection coloured by the value
- `property-mw-value`, `property-logp-value` — molecular weight and LogP against the value, with
  Spearman's rho. A high rho is the classic QSAR failure mode: a model that has only learned
  "bigger" or "greasier"

**Per-descriptor diagnostics**
- `cv-spearman` — inner-CV rho per descriptor, with the in-sample rho marked; the gap is overfitting
- `cv-error-curve` — the fraction of compounds predicted within a given absolute error, per
  descriptor, against the error you would get by predicting the training mean
- `descriptor-correlation`, `topk-overlap-curve` — how redundant the descriptors are

## When the model is not worth shipping

Two checks fire on the fit summary, both stated as facts rather than opinions:

- The pooled model's held-out Spearman rho is below the pruning floor — it does not order
  compounds better than chance.
- Its R² is at or below zero — predicting the training mean for every compound would have done
  as well, whatever the ordering.

Neither stops the run. A metric that could not be computed says nothing: unknown is not zero.

## Validating a change

**The numbers** are pinned by two golden fixtures, one per task: `tests/test_golden.py` re-runs
estimate → pool → report → finish on a kept fit and compares every column against a recorded
reference. It is the only thing here that can tell you a result moved. It is opt-in, because it
needs a fit kept with its descriptor matrices (~25 MB, not in git), and the module docstring says
how to record one. Run it against both fixtures before merging anything that touches the
modelling path.

**The behaviour** is pinned by `pytest`, deliberately thin. It is there to catch what someone
could ship, not to describe how the code is written:

- `tests/test_regression.py` — the pooled model (its agreement with lazy-qsar's own `combine`,
  the applicability-domain veto, the conformal intervals, pruning), the pre-screening proxy, and
  the contracts regression has with the rest of the system.
- `tests/test_regression_report.py` — a synthetic run folder, built here rather than by the
  pipeline, that every panel must render against or decline without raising; that the plot
  registry and the HTML catalogue agree; and what the delivered tables carry for each
  task-and-truth combination.

Before adding a test, reintroduce the bug it would catch and check nothing else already fails.
Before deleting one, do the same. Most of what this suite could assert about the report is
implementation, and a test that only restates the code costs more to maintain than it protects.

Neither of those is a substitute for running the thing. Do not validate on the DILI example
alone: it is unrepresentative of the models this repo is built for. Pull a real target from the
ChEMBL web API instead, keep one `standard_type`, take the median per molecule, and hold out a
genuine external set rather than trusting out-of-fold numbers.
