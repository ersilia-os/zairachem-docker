# Regression in ZairaChem

`zairachem fit -r` trains a model that predicts the target value itself. This note records what
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

**Accuracy** (needs the observed values, so fit and labelled predict)
- `regression-raw` — predicted against observed, with R² and MAE
- `regression-residuals` — residual against predicted: where the model is biased
- `r2-individual` — per-descriptor R²
- `potency-enrichment` — the mean *observed* value of the top k% by prediction. The screening
  question, and the one a correlation coefficient does not answer
- `interval-coverage` — does the interval keep its promise, and does it keep it in every bin

- `value-distribution` — the observed values themselves, with the median marked. It sits in the
  Dataset section, where a classification run shows its class balance: a target piled against an
  assay's detection limit is worth seeing before any model is read

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

Two golden fixtures pin the pipeline, one per task; see `tests/test_golden.py` for how to record
them. They are opt-in because they need a fit kept with its descriptor matrices, so the default
test run relies on `tests/test_regression.py` (the pooled model and the pre-screening proxy) and
`tests/test_regression_report.py`, which builds a synthetic run folder and walks every panel.

Do not validate on the DILI example alone. It is unrepresentative of the models this repo is
built for: pull a real target from the ChEMBL web API instead, keep one `standard_type`, take the
median per molecule, and hold out a genuine external set rather than trusting out-of-fold numbers.
