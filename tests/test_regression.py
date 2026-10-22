"""The regression pooled model is lazy-qsar's regression ensemble, and it has no cutoffs.

Regression deliberately has no reference library, no rank, no decision cutoff and no binary call
(lazy-qsar's ``qsar_regressor``): the single output is the value, with a conformal interval. These
tests pin that contract and the arithmetic behind it.
"""

import numpy as np
import pytest

from zairachem.pool.reliability_pooler import pooler

NAMES = ["lq-eosA-reg", "lq-eosB-reg", "lq-eosC-reg"]


def _signals(n=200, d=3, seed=0):
  """Per-descriptor regression signals ``(P, R, A, y, meta)`` of decreasing quality."""
  rng = np.random.default_rng(seed)
  y = rng.normal(6.5, 1.2, n)
  noise = [0.4, 0.8, 1.6]
  P = np.column_stack([y + rng.normal(0, s, n) for s in noise])
  R = rng.random((n, d))
  A = rng.random((n, d))
  curves = [(np.linspace(0, 1, 5), np.linspace(0.5, 0.1, 5)) for _ in range(d)]
  meta = {
    "oof_aucs": [0.72, 0.60, 0.45],  # Spearman rho (the field name predates regression)
    "train_aucs": [0.85, 0.70, 0.55],
    "proxy_aucs": [0.70, None, 0.44],
    "curves": curves,
    "ad_cutoffs": [0.05, 0.05, 0.05],
  }
  return P, R, A, y, meta


def _fit(P, R, A, y, meta, S=None):
  return pooler.fit(P, R, S, A, y, NAMES, meta, task=pooler.REGRESSION)


def test_pooled_value_is_lazyqsar_combine_with_stacking_weights():
  from lazyqsar.ensemble.combine import combine
  from lazyqsar.poolers.regression.inner_pooler import stacking_weights

  P, R, A, y, meta = _signals()
  params, result = _fit(P, R, A, y, meta)
  spec = params["spec"]
  keep = [NAMES.index(c) for c in params["descriptor_columns"]]
  # The descriptor weighting is lazy-qsar's NNLS stacking on the kept out-of-fold predictions.
  np.testing.assert_allclose(spec.descriptor_weights, stacking_weights(P[:, keep], y))
  assert sum(spec.descriptor_weights) == pytest.approx(1.0)
  # And the pooled value is exactly lazy-qsar's combine with that spec — no local arithmetic.
  direct = combine(P[:, keep], R[:, keep], None, A[:, keep], spec=spec, outputs=("value",))
  np.testing.assert_array_equal(result.values["value"], direct.values["value"])


def test_pruning_is_the_classification_rule_on_the_rho_scale():
  mask = pooler.active_mask
  assert mask([0.50, 0.29, 0.40], pooler.REGRESSION) == [True, False, True]  # best - 0.20
  assert mask([0.50, 0.09, None], pooler.REGRESSION) == [True, False, True]  # floor; unknown kept
  assert mask([0.05, 0.08], pooler.REGRESSION) == [True, True]  # none would survive: keep all
  # The same skills read as hopeless on the AUROC scale, which is why the thresholds differ.
  assert mask([0.50, 0.29, 0.40], pooler.CLASSIFICATION) == [True, True, True]  # all below floor
  P, R, A, y, meta = _signals()
  params, result = _fit(P, R, A, y, meta)  # third descriptor is below the 0.20 gap
  assert params["descriptor_columns"] == [NAMES[0], NAMES[1]]
  assert params["pruned_columns"] == [NAMES[2]]
  assert result.weights.shape == (len(y), 2)


def test_applicability_domain_still_vetoes_under_stacking_weights():
  P, R, A, y, meta = _signals()
  # Put the first descriptor out of its domain on the first half of the rows, and well inside it
  # on the second, so the veto is the only thing that can zero its weight.
  A = A.copy()
  A[: len(y) // 2, 0], A[len(y) // 2 :, 0] = 0.0, 1.0
  meta = {**meta, "ad_cutoffs": [0.5, 0.0, 0.0]}
  params, result = _fit(P, R, A, y, meta)
  first = params["descriptor_columns"].index(NAMES[0])
  vetoed, kept = result.weights[: len(y) // 2, first], result.weights[len(y) // 2 :, first]
  assert (vetoed == 0).all()  # the veto is per compound, not per model
  assert (kept > 0).all()
  # The surviving descriptors still carry the whole weight on a vetoed row.
  np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)


def test_intervals_cover_the_training_residuals():
  P, R, A, y, meta = _signals(n=400)
  params, result = _fit(P, R, A, y, meta)
  lower, upper = result.values["lower"], result.values["upper"]
  assert (upper > lower).all()
  # Split-conformal at the default level covers ~90% of the rows it was calibrated on.
  covered = ((y >= lower) & (y <= upper)).mean()
  assert 0.85 <= covered <= 0.97


def test_a_regression_model_has_no_cutoffs_and_no_rank():
  P, R, A, y, meta = _signals()
  params, result = _fit(P, R, A, y, meta)
  assert params["task"] == pooler.REGRESSION
  assert params["cutoffs"] == {} and params["cutoff_ranks"] == {}
  assert set(result.values) == {"value", "lower", "upper"}
  assert "proba" not in result.values and "binary" not in result.values
  # Dropping the rank is meaningless here, so it leaves the model alone.
  assert pooler.without_rank(params) is params


def test_saved_params_round_trip_and_subsets():
  P, R, A, y, meta = _signals()
  params, result = _fit(P, R, A, y, meta)
  keep = params["descriptor_columns"]
  cols = [NAMES.index(c) for c in keep]
  loaded = pooler.load_params(pooler.to_artifact(params))
  again = pooler.predict(loaded, P[:, cols], R[:, cols], None, A[:, cols])
  np.testing.assert_array_equal(again.values["value"], result.values["value"])
  np.testing.assert_array_equal(again.values["lower"], result.values["lower"])
  # A subset (descriptor drift at predict) keeps neither the stacking weights nor the intervals:
  # both were fitted over the full set and do not describe a subset.
  sub = pooler.restrict(loaded, keep[:1])
  out = pooler.predict(sub, P[:, cols[:1]], R[:, cols[:1]], None, A[:, cols[:1]])
  assert sub["spec"].descriptor_weights is None and sub["spec"].intervals is None
  assert "lower" not in out.values and out.values["value"].shape == (len(y),)


def test_metrics_are_the_regression_ones():
  from zairachem.report.fetcher import ResultsFetcher

  fetcher = ResultsFetcher.__new__(ResultsFetcher)
  row = fetcher.regression_performance_report([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 3.9])
  assert row["num_compounds"] == 4
  assert row["r2"] == pytest.approx(0.992)
  assert row["rmse"] == pytest.approx(0.1)
  assert row["mae"] == pytest.approx(0.1)
  assert row["spearman"] == pytest.approx(1.0)
  # No cutoff-derived metric belongs in a regression row.
  assert not {"cutoff", "tp", "mcc"} & set(row)
  # A constant truth makes every metric undefined rather than raising.
  degenerate = fetcher.regression_performance_report([2.0, 2.0], [1.0, 3.0])
  assert np.isnan(degenerate["r2"]) and degenerate["num_compounds"] == 2


def test_the_output_contract_is_value_not_score():
  from zairachem.base.vars import OUTPUT_COLUMNS, OUTPUT_DESCRIPTOR_PREFIX, TARGET_COLUMN

  columns = OUTPUT_COLUMNS["regression"]
  assert columns["reg"] == "value"
  assert "score" not in columns.values() and "clf_rank" not in columns
  assert columns["train_similarity"] == "similarity" and columns["descriptor_std"] == "spread"
  assert OUTPUT_DESCRIPTOR_PREFIX["regression"] == "value_"
  assert TARGET_COLUMN["regression"] == "val"
