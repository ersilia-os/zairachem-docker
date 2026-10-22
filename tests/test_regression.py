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


# --- The report surface: every regression panel exists, and no panel belongs to both tasks -------


def _plot_specs():
  from zairachem.report.report import _PLOT_SPECS

  return dict(_PLOT_SPECS)


def test_every_regression_panel_is_registered_and_catalogued():
  from zairachem.report.html.catalog import (
    _CATEGORIES,
    _DATASET_GROUPS,
    _DIAG_GROUPS,
    _GROUPS,
    _TITLES,
  )

  specs = _plot_specs()
  panels = [
    "regression-raw",
    "regression-residuals",
    "regression-distribution",
    "regression-error-similarity",
    "potency-enrichment",
    "interval-coverage",
    "value-consensus",
    "cv-spearman",
    "property-mw-value",
    "property-logp-value",
    "r2-individual",
    "heldout-r2",
  ]
  missing = [p for p in panels if p not in specs]
  assert not missing, f"not registered in report.py: {missing}"
  # A figure with no catalogue title falls through to the report's "More" catch-all.
  untitled = [p for p in panels if p not in _TITLES]
  assert not untitled, f"no title in the HTML catalog: {untitled}"
  # A figure reaches the page through one of the section tables; which one depends on the section
  # that owns it (plot categories, the dataset and diagnostics blocks, or a carousel group).
  catalogued = {m for _a, _h, _d, members in _CATEGORIES for m in members}
  for table in (_GROUPS, _DIAG_GROUPS, _DATASET_GROUPS):
    catalogued |= {m for g in table for m in g["members"]}
  homeless = [p for p in panels if p not in catalogued]
  assert not homeless, f"in no HTML section: {homeless}"


def test_the_colour_legend_speaks_the_task_s_language():
  from zairachem.report.colors import hexcol, legend

  clf = dict((lbl, key) for lbl, key in legend("classification"))
  reg = dict((lbl, key) for lbl, key in legend("regression"))
  assert "Active" in clf and "Active" not in reg  # no classes in a regression report
  assert "High value" in reg and "Low value" in reg
  # Both tasks keep the provenance entries, and every key resolves to a real colour.
  for entries in (clf, reg):
    assert "From store" in entries and "Computed" in entries
    for key in entries.values():
      assert hexcol(key).startswith("#")


def test_every_semantic_colour_key_is_used():
  """A key nobody asks for is a claim about meaning that nothing keeps honest."""
  import pathlib
  import re

  from zairachem.report.colors import _SEMANTIC

  tree = pathlib.Path("zairachem")
  blob = "\n".join(p.read_text() for p in tree.rglob("*.py"))
  unused = [
    key
    for key in _SEMANTIC
    if len(re.findall(rf"""["']{re.escape(key)}["']""", blob)) <= 1  # only its own definition
  ]
  assert not unused, f"dead semantic colour keys: {unused}"


def test_inner_cv_reports_are_read_with_the_right_skill():
  from zairachem.report.cv import CHANCE, skill_keys, sort_best_first

  clf = [{"descriptor": "a", "oof_auc": 0.7}, {"descriptor": "b", "oof_auc": 0.9}]
  reg = [{"descriptor": "a", "oof_rho": 0.4}, {"descriptor": "b", "oof_rho": 0.8}]
  assert skill_keys(clf) == ("oof_auc", "train_auc", "AUROC")
  assert skill_keys(reg)[0] == "oof_rho"
  assert CHANCE["oof_auc"] == 0.5 and CHANCE["oof_rho"] == 0.0
  # Best first, on each task's own skill, and a missing skill never takes the top slot.
  assert [s["descriptor"] for s in sort_best_first(list(clf))] == ["b", "a"]
  assert [s["descriptor"] for s in sort_best_first(list(reg))] == ["b", "a"]
  with_gap = reg + [{"descriptor": "c"}]
  assert [s["descriptor"] for s in sort_best_first(with_gap)] == ["b", "a", "c"]


def test_the_hitlist_ranks_a_regression_run_by_value(tmp_path):
  import csv

  from zairachem.report.html.tables import _hitlist_table_html

  rows = [
    {"input_smiles": "CCO", "value": "5.0", "lower": "4.0", "upper": "6.0", "similarity": "0.3"},
    {"input_smiles": "CCN", "value": "8.5", "lower": "7.5", "upper": "9.5", "similarity": "0.9"},
  ]
  with open(tmp_path / "output_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  out = _hitlist_table_html(str(tmp_path))
  assert "by predicted value" in out and "Interval" in out
  assert "Call" not in out  # a regression run has no call to make
  assert out.index("CCN") < out.index("CCO")  # ranked by value, best first


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
