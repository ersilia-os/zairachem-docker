"""What a regression model is, and is not.

Regression deliberately has no reference library, no rank, no decision cutoff and no binary call
(lazy-qsar's ``qsar_regressor``): the single output is the value, with a conformal interval. These
tests pin that contract and the arithmetic behind it -- the pooled model, the pre-screening proxy,
and the handful of promises regression makes to the rest of the pipeline.

Two things they deliberately do not do. They do not re-derive lazy-qsar's own arithmetic: where
the pooled value has to equal ``combine``'s, that is asserted against ``combine`` itself. And
they do not pin the numbers a run produces -- that is ``test_golden.py``, which is opt-in and the
only thing here that can tell you a result moved. The report is in ``test_regression_report.py``.
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


def test_one_floor_serves_the_pooler_and_the_screen():
  """Pruning after training and screening before it must agree on what counts as useful."""
  from zairachem.base.vars import SKILL_FLOOR
  from zairachem.screen.proxy import PROXY_FLOOR

  assert PROXY_FLOOR is SKILL_FLOOR
  assert SKILL_FLOOR["regression"] == 0.10 and SKILL_FLOOR["classification"] == 0.55


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

  # lazy-qsar bins the residuals by applicability-domain score, and a constant score leaves a bin
  # with nothing in it to take a quantile of. That used to raise out of the pool step, after every
  # descriptor had been trained; the interval is an extra, and losing it must not cost the model.
  for degenerate in (np.zeros_like(A), np.full_like(A, -99.0)):
    params, result = _fit(P, R, degenerate, y, meta)
    assert params["spec"].intervals is None
    assert "lower" not in result.values
    assert np.isfinite(np.asarray(result.values["value"], dtype=float)).all()


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

  # One compound nothing could score -- unmeasured, or failed on every descriptor -- is left out
  # of the row rather than turning every metric in it to NaN.
  gappy = fetcher.regression_performance_report(
    [1.0, 2.0, 3.0, 4.0, np.nan], [1.1, 2.1, 2.9, np.nan, 5.0]
  )
  assert gappy["num_compounds"] == 3 and np.isfinite(gappy["r2"])


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

  # The ensemble is scored on rho: ordering is what a screen asks a regression model for.
  from zairachem.screen.proxy import PROXY_CHANCE, _ensemble_skill

  folds = [(tr, te)]
  preds = {"good": [y[te]], "backwards": [-y[te]], "flat": [np.ones(len(te))]}
  assert _ensemble_skill(["good"], preds, y, folds, "regression") == pytest.approx(1.0)
  assert _ensemble_skill(["backwards"], preds, y, folds, "regression") == pytest.approx(-1.0)
  # Nothing scoreable falls back to chance rather than to a number.
  assert _ensemble_skill(["flat"], preds, y, folds, "regression") == PROXY_CHANCE["regression"]


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


def test_a_model_with_no_signal_says_so():
  """The floor the pooler prunes descriptors on, applied to the pooled model the user will ship."""
  from zairachem.base.utils.summaries import _warn_if_no_signal
  from zairachem.base.vars import SKILL_FLOOR

  rows = []
  _warn_if_no_signal(rows, "regression", SKILL_FLOOR["regression"] - 0.01)
  assert rows and "No usable signal" in rows[0][1] and "order compounds" in rows[0][1]

  # At or above the floor there is nothing to warn about, and a missing metric says nothing.
  for skill in (SKILL_FLOOR["regression"], 0.8, None, float("nan"), "—"):
    rows = []
    _warn_if_no_signal(rows, "regression", skill)
    assert rows == [], skill

  # Classification uses its own floor and its own wording.
  rows = []
  _warn_if_no_signal(rows, "classification", SKILL_FLOOR["classification"] - 0.01)
  assert "separate the classes" in rows[0][1]

  # A model can order compounds well and still predict useless values.
  rows = []
  _warn_if_no_signal(rows, "regression", skill=0.8, r2=-0.3)
  assert "no better than the training average" in rows[0][1]
  for r2 in (0.01, None, float("nan")):
    rows = []
    _warn_if_no_signal(rows, "regression", skill=0.8, r2=r2)
    assert rows == [], r2

  # A near miss must not round into "0.10 is below 0.10".
  rows = []
  _warn_if_no_signal(rows, "regression", skill=0.099, r2=0.4)
  assert "0.099 is below 0.100" in rows[0][1]


def test_a_model_trained_before_regression_still_finds_its_pooling_weights(tmp_path):
  """The pool artefact was named ``_clf`` before regression; reading one must still work."""
  from zairachem.pool.reliability_pooler import pooler

  # A fit always writes the current name, whatever is already in the folder.
  assert pooler.artifact_path(str(tmp_path), write=True).endswith(pooler.ARTIFACT_NAME)

  # Nothing saved yet: a read resolves to the current name, so the error names the file we write.
  assert pooler.artifact_path(str(tmp_path)).endswith(pooler.ARTIFACT_NAME)

  legacy = tmp_path / pooler.LEGACY_ARTIFACT_NAME
  legacy.write_bytes(b"")
  assert pooler.artifact_path(str(tmp_path)) == str(legacy)

  # Both present (a model re-pooled by a newer ZairaChem): the current one wins.
  (tmp_path / pooler.ARTIFACT_NAME).write_bytes(b"")
  assert pooler.artifact_path(str(tmp_path)).endswith(pooler.ARTIFACT_NAME)


def test_the_spread_between_repeat_measurements_is_measured():
  """It is the assay's own reproducibility, and the scale the model's RMSE has to be read on."""
  from zairachem.setup.prep.files import replicate_summary

  # Three compounds, one measured three times around 5.0 and one twice around 7.0.
  groups = {"a": [4.9, 5.0, 5.1], "b": [7.0, 7.2], "c": [6.0]}
  s = replicate_summary(groups, "regression")
  assert (s["n_rows"], s["n_compounds"], s["n_repeated"], s["max_repeats"]) == (6, 3, 2, 3)
  # Pooled within-compound SD: sqrt(sum of squared deviations / sum of (n-1)).
  expected = np.sqrt((0.02 + 0.02) / 3)
  assert s["pooled_sd"] == pytest.approx(expected)
  assert s["max_range"] == pytest.approx(0.2)

  # Nothing repeated: there is no reading to give, and none is invented.
  alone = replicate_summary({"a": [1.0], "b": [2.0]}, "regression")
  assert alone["n_repeated"] == 0 and "pooled_sd" not in alone

  # Classification counts the compounds the input gave both labels; they cap any model's accuracy.
  clf = replicate_summary({"a": [1, 0], "b": [1, 1], "c": [0]}, "classification")
  assert clf["disagreeing"] == 1 and "pooled_sd" not in clf


def test_an_error_as_small_as_the_data_allows_is_said_to_be_so():
  """An RMSE means nothing on its own; against the assay's spread it means the model is done."""
  from zairachem.base.utils.summaries import _note_if_at_the_noise_floor

  rows = []
  _note_if_at_the_noise_floor(rows, rmse=0.63, floor=0.61)
  assert rows and "limit of this data" in rows[0][1]

  rows = []
  _note_if_at_the_noise_floor(rows, rmse=1.4, floor=0.61)  # well above the spread: nothing to say
  assert rows == []

  # Nothing measurable, nothing claimed.
  for rmse, floor in ((0.63, None), (None, 0.61), (0.63, 0.0), (float("nan"), 0.61)):
    rows = []
    _note_if_at_the_noise_floor(rows, rmse=rmse, floor=floor)
    assert rows == [], (rmse, floor)


def test_compounds_given_both_labels_are_called_out():
  """Classification's noise floor: those compounds are wrong whichever way they are predicted."""
  import inspect
  import json
  import os
  import tempfile

  from zairachem.base.utils import summaries
  from zairachem.base.utils.summaries import _note_contradicted_labels

  run = tempfile.mkdtemp()
  os.makedirs(os.path.join(run, "inputs"))
  record = os.path.join(run, "inputs", "replicates.json")

  for payload in ({"n_repeated": 120, "disagreeing": 0}, {"n_repeated": 0}, {}):
    with open(record, "w") as f:
      json.dump(payload, f)
    rows = []
    _note_contradicted_labels(rows, run)
    assert rows == [], payload

  with open(record, "w") as f:
    json.dump({"n_repeated": 120, "disagreeing": 17}, f)
  rows = []
  _note_contradicted_labels(rows, run)
  assert "17 of 120" in rows[0][1] and "coin flip" in rows[0][1]

  # And the fit summary has to call it, or none of the above is ever reached.
  assert "_note_contradicted_labels" in inspect.getsource(summaries.final_summary_panel)


def test_the_noise_floor_note_separates_the_model_from_the_assay():
  """The headline invites the wrong reading: an RMSE of 0.548 against repeats that differ by
  0.481 looks like a model barely better than noise. Inverting RMSE^2 = m^2 + s^2 says its own
  error is 0.263, about half the noise it is graded through -- the real numbers from a T. brucei
  fit of 700 compounds drawn from 1,577 repeated rows."""
  import math

  from zairachem.base.utils.summaries import NOISE_FLOOR_FACTOR, _note_if_at_the_noise_floor

  rows = []
  _note_if_at_the_noise_floor(rows, 0.548, 0.481)
  assert "0.481" in rows[0][1] and "0.548" in rows[0][1] and "0.263" in rows[0][1]

  # An RMSE at the assay's own spread leaves the model no error of its own at all.
  rows = []
  _note_if_at_the_noise_floor(rows, 0.5, 0.5)
  assert "0.000" in rows[0][1]

  # And an RMSE *below* it -- which happens when the spread is estimated from few repeats -- has
  # no real root. Unclamped that is a ValueError out of the closing panel, after a finished fit.
  rows = []
  _note_if_at_the_noise_floor(rows, 0.3, 0.481)
  assert "0.000" in rows[0][1]

  # The note claims most of the error is the assay. That holds only below sqrt(2) * s, so the
  # threshold has to stay inside it or the sentence starts lying.
  assert NOISE_FLOOR_FACTOR < math.sqrt(2)
  worst = NOISE_FLOOR_FACTOR  # in units of s, the largest RMSE that still fires
  assert math.sqrt(worst**2 - 1.0) < 1.0  # the model's share is the smaller one
