"""What a regression model is, and is not.

Regression deliberately has no reference library, no rank, no decision cutoff and no binary call
(lazy-qsar's ``qsar_regressor``): the single output is the value, with a conformal interval. These
tests pin that contract and the arithmetic behind it -- the pooled model and the pre-screening
proxy. The report it produces is in ``test_regression_report.py``.
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


def test_one_unscoreable_compound_does_not_void_the_whole_metrics_row():
  """A NaN prediction used to propagate into every metric instead of being left out."""
  from zairachem.report.fetcher import ResultsFetcher

  fetcher = ResultsFetcher.__new__(ResultsFetcher)
  clean = fetcher.regression_performance_report([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 3.9])
  gappy = fetcher.regression_performance_report(
    [1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 2.9, 3.9, float("nan")]
  )
  assert gappy["num_compounds"] == 4  # the unscoreable compound is counted out, not in
  assert gappy["r2"] == pytest.approx(clean["r2"])
  assert not np.isnan(gappy["rmse"])


def test_the_output_contract_is_value_not_score():
  from zairachem.base.vars import OUTPUT_COLUMNS, OUTPUT_DESCRIPTOR_PREFIX, TARGET_COLUMN

  columns = OUTPUT_COLUMNS["regression"]
  assert columns["reg"] == "value"
  assert "score" not in columns.values() and "clf_rank" not in columns
  assert columns["train_similarity"] == "similarity" and columns["descriptor_std"] == "spread"
  assert OUTPUT_DESCRIPTOR_PREFIX["regression"] == "value_"
  assert TARGET_COLUMN["regression"] == "val"


# --- Pre-screening: the proxy scores a continuous target on its own scale -----------------------


def _screen_folds(n=120, seed=0):
  rng = np.random.default_rng(seed)
  idx = rng.permutation(n)
  return [(idx[: n // 2], idx[n // 2 :]), (idx[n // 2 :], idx[: n // 2])]


def test_the_proxy_predicts_values_for_a_continuous_target():
  from zairachem.screen.proxy import _fold_pred

  rng = np.random.default_rng(0)
  n = 120
  X = rng.normal(size=(n, 6))
  y = X[:, 0] * 2.0 + rng.normal(0, 0.2, n)
  (tr, te), _ = _screen_folds(n)
  pred = _fold_pred(X, y, tr, te, seed=0, task="regression")
  assert pred is not None and pred.shape == (len(te),)
  # A value, not a probability: it must live on the target's scale, not in [0, 1].
  assert pred.min() < 0.0 < pred.max()
  # A constant target carries no signal and is refused rather than scored.
  assert _fold_pred(X, np.ones(n), tr, te, seed=0, task="regression") is None


def test_the_ensemble_skill_of_a_regression_screen_is_spearman():
  from scipy.stats import spearmanr

  from zairachem.screen.proxy import PROXY_CHANCE, _ensemble_skill

  rng = np.random.default_rng(1)
  n = 120
  y = rng.normal(size=n)
  folds = _screen_folds(n)
  preds = {"a": [y[np.asarray(te)] + rng.normal(0, 0.1, len(te)) for _tr, te in folds]}
  skill = _ensemble_skill(["a"], preds, y, folds, "regression")
  expected = np.mean([
    spearmanr(y[np.asarray(te)], preds["a"][i]).statistic for i, (_tr, te) in enumerate(folds)
  ])
  assert skill == pytest.approx(float(expected))
  # Rho, not AUROC: a descriptor with no signal scores 0, not 0.5.
  assert PROXY_CHANCE["regression"] == 0.0 and PROXY_CHANCE["classification"] == 0.5
  assert _ensemble_skill([], {}, y, folds, "regression") == 0.0


def test_the_screening_floor_is_the_auroc_floor_on_the_rho_scale():
  from zairachem.screen.proxy import PROXY_FLOOR

  # AUROC - 0.5 is about rho / 2, so the 0.55 classification floor maps to 0.10 on rho.
  assert PROXY_FLOOR["classification"] == 0.55
  assert PROXY_FLOOR["regression"] == pytest.approx((0.55 - 0.5) * 2)


def test_no_regression_figure_shows_its_slug_as_a_title():
  """A stem with no catalogue entry is title-cased into something like "Projection Mwlogp Value"."""
  from zairachem.report.html.catalog import _PROJECTION_ORDER, _humanize

  stems = [f"projection-{base}-value" for base in _PROJECTION_ORDER]
  stems += [
    "regression-raw",
    "regression-residuals",
    "regression-distribution",
    "regression-error-similarity",
    "potency-enrichment",
    "interval-coverage",
    "value-consensus",
    "cv-spearman",
    "cv-error-curve",
    "property-mw-value",
    "property-logp-value",
    "heldout-r2",
  ]
  slugged = [s for s in stems if _humanize(s) == " ".join(w.capitalize() for w in s.split("-"))]
  assert not slugged, f"no catalogue title, so the report shows the slug: {slugged}"


# --- Contracts with the rest of the system ------------------------------------------------------
#
# Each of these fails somewhere far from its cause: a rank reference downloaded for a task that
# cannot use it, a lazy-qsar upgrade that moves a private name. Cheap to assert, expensive to
# notice.


def test_a_regression_fit_does_not_prepare_the_rank_reference(monkeypatch):
  """lazy-qsar's regression ensemble has no reference library; downloading one costs ~1.6 GB."""
  import inspect

  from zairachem.setup.prep import training

  source = inspect.getsource(training.TrainSetup.setup)
  call = "prepare_rank_references(self.featurizer_ids)"
  assert call in source
  # The call must sit under a classification guard, not at the top level of setup().
  line = next(ln for ln in source.splitlines() if call in ln)
  guard = next(
    ln for ln in reversed(source[: source.index(line)].splitlines()) if ln.strip().startswith("if ")
  )
  assert 'self.task == "classification"' in guard, f"rank reference prepared under: {guard.strip()}"


def test_the_lazyqsar_surface_regression_leans_on_is_still_there():
  """ZairaChem reaches past lazy-qsar's public API in a few places; an upgrade must not silently
  remove the footholds. Each name here is used by estimate.py or pooler.py."""
  from lazyqsar.agnostic import LazyRegressor
  from lazyqsar.ensemble.combine import EnsembleSpec
  from lazyqsar.ensemble.intervals import DEFAULT_LEVEL, ad_summary, fit_intervals
  from lazyqsar.poolers.regression.inner_pooler import stacking_weights
  from lazyqsar.qsar_regressor import _quality_rho

  # Per-descriptor fitting: the out-of-fold channels and the percentile of a predicted value.
  assert hasattr(LazyRegressor, "oof_channels")
  assert hasattr(LazyRegressor, "_oof_percentile_from_value")
  for name in ("fit", "predict", "save"):
    assert callable(getattr(LazyRegressor, name))

  # Pooling: the spec fields a regression model fills, and the three functions that fill them.
  for field in ("task", "descriptor_weights", "intervals"):
    assert field in EnsembleSpec.__dataclass_fields__, field
  assert callable(stacking_weights) and callable(fit_intervals) and callable(ad_summary)
  # clip(2 * oof - train, -1, 1): the regression quality term the spec's skill weights are built on.
  assert _quality_rho(0.8, 0.9) == pytest.approx(0.7)
  assert 0.0 < DEFAULT_LEVEL < 1.0
