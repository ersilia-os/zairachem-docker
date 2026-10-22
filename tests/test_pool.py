"""The pooled model is lazy-qsar's ensemble, and its cutoffs are named and stored."""

import numpy as np
import pytest

from zairachem.pool.reliability_pooler import pooler

NAMES = ["lq-eosA-clf", "lq-eosB-clf", "lq-eosC-clf"]


def _signals(n=200, d=3, seed=0):
  rng = np.random.default_rng(seed)
  y = (rng.random(n) < 0.4).astype(int)
  P = np.clip(0.3 * y[:, None] + 0.7 * rng.random((n, d)), 0.01, 0.99)
  R = rng.random((n, d))
  S = P + 0.01 * rng.random((n, d))
  A = rng.random((n, d))
  curves = [(np.linspace(0, 1, 5), np.linspace(0.5, 0.1, 5)) for _ in range(d)]
  meta = {
    "oof_aucs": [0.80, 0.75, 0.72],
    "train_aucs": [0.95, 0.80, 0.90],
    "proxy_aucs": [0.78, None, 0.72],
    "curves": curves,
    "ad_cutoffs": [0.05, 0.05, 0.05],
  }
  return P, R, S, A, y, meta


def test_weights_follow_lazyqsar_skill_term():
  P, R, S, A, y, meta = _signals()
  params, result = pooler.fit(P, R, S, A, y, NAMES, meta)
  # lazy-qsar: max(0, mean(clip(2*oof - train), proxy) - 0.5).
  quality = [min(1, max(0, 2 * o - t)) for o, t in zip(meta["oof_aucs"], meta["train_aucs"])]
  expected = [
    max(0.0, np.mean([q] + ([p] if p is not None else [])) - 0.5)
    for q, p in zip(quality, meta["proxy_aucs"])
  ]
  np.testing.assert_allclose(result.base, expected)
  # And the pooled probability is exactly lazy-qsar's combine with that spec.
  from lazyqsar.ensemble.combine import combine

  direct = combine(P, R, S, A, spec=params["spec"], outputs=("proba",)).values["proba"]
  np.testing.assert_array_equal(result.values["proba"], direct)


def test_no_applicability_domain_weights_uniformly():
  P, R, S, _, y, meta = _signals()
  _, result = pooler.fit(P, R, S, None, y, NAMES, meta)
  np.testing.assert_allclose(result.weights, 1 / 3)


def test_pruning_follows_lazyqsar():
  assert pooler.active_mask([0.80, 0.69, 0.75]) == [True, False, True]  # best - 0.10
  assert pooler.active_mask([0.80, 0.54, None]) == [True, False, True]  # floor; unknown kept
  assert pooler.active_mask([0.50, 0.52]) == [True, True]  # none would survive: keep all
  P, R, S, A, y, meta = _signals()
  meta["oof_aucs"] = [0.80, 0.60, 0.78]
  params, result = pooler.fit(P, R, S, A, y, NAMES, meta)
  assert params["descriptor_columns"] == [NAMES[0], NAMES[2]]
  assert params["pruned_columns"] == [NAMES[1]]
  assert result.weights.shape == (len(y), 2)


def test_screening_call_is_rank_065_against_the_reference():
  P, R, S, A, y, meta = _signals()
  rng = np.random.default_rng(1)
  reference = (rng.random((5000, 3)) * 0.8, rng.random((5000, 3)), rng.random((5000, 3)))
  params, result = pooler.fit(
    P, R, S, A, y, NAMES, meta, reference=reference, kappa=lambda y, p: 0.42
  )
  cut = params["cutoffs"]
  assert cut["screening"] is not None and cut["kappa"] == 0.42 and cut["default"] == 0.5
  proba, rank = result.values["proba"][:, 1], result.values["rank"][:, 1]
  np.testing.assert_array_equal(result.values["binary"], (proba >= cut["screening"]).astype(int))
  # The call and the rank agree: everything called scores at least 0.65 on the rank scale.
  assert (rank[result.values["binary"] == 1] >= 0.65 - 1e-9).all()


def test_saved_params_round_trip_and_subsets():
  P, R, S, A, y, meta = _signals()
  rng = np.random.default_rng(1)
  reference = (rng.random((2000, 3)), rng.random((2000, 3)), rng.random((2000, 3)))
  params, result = pooler.fit(P, R, S, A, y, NAMES, meta, reference=reference)
  loaded = pooler.load_params(pooler.to_artifact(params))
  again = pooler.predict(loaded, P, R, S, A)
  np.testing.assert_array_equal(again.values["proba"], result.values["proba"])
  # A subset (descriptor drift at predict) has no rank and calls at the default cutoff.
  sub = pooler.restrict(loaded, NAMES[:2])
  out = pooler.predict(sub, P[:, :2], R[:, :2], S[:, :2], A[:, :2])
  assert "rank" not in out.values and sub["cutoffs"]["screening"] is None
  np.testing.assert_array_equal(out.values["binary"], (out.values["proba"][:, 1] >= 0.5))


def test_models_saved_before_this_pooler_still_load():
  P, R, _, A, _, meta = _signals()
  legacy = {
    "descriptor_columns": NAMES,
    "base_scores": [0.30, 0.25, 0.20],
    "rank_error_curves": meta["curves"],
    "ad_hard_cutoffs": [0.05, 0.05, 0.05],
    "has_rank": True,
    "has_ad": True,
  }
  params = pooler.load_params(legacy)
  result = pooler.predict(params, P, R, None, A)
  np.testing.assert_allclose(result.base, legacy["base_scores"])  # the same skill weights
  assert params["cutoffs"]["kappa"] is None and "rank" not in result.values


def test_cutoffs_document_names_every_cutoff(tmp_path):
  from zairachem.pool.cutoffs import call_cutoff, read_cutoffs, write_cutoffs

  ranks = {"screening": 0.65, "kappa": 0.3, "default": 0.4, "balanced": {"eosA": 0.33}}
  write_cutoffs(str(tmp_path), {"screening": 0.81, "kappa": 0.35}, {"eosA": 0.47}, ranks)
  doc = read_cutoffs(str(tmp_path))
  assert set(doc) == {"screening", "kappa", "balanced", "default"}
  assert doc["screening"] == {**doc["screening"], "proba": 0.81, "rank": 0.65}
  assert doc["balanced"]["proba"] == {"eosA": 0.47} and doc["balanced"]["rank"] == {"eosA": 0.33}
  assert all({"scope", "computed_from", "used_for"} <= set(entry) for entry in doc.values())
  assert call_cutoff(str(tmp_path)) == ("screening", 0.81)
  write_cutoffs(str(tmp_path), {"screening": None, "kappa": 0.35}, {})
  assert call_cutoff(str(tmp_path)) == ("default", 0.5)


@pytest.mark.parametrize("cutoff,expected_tp", [(0.5, 2), (None, None)])
def test_metrics_are_reported_at_a_named_cutoff(cutoff, expected_tp):
  from zairachem.report.fetcher import ResultsFetcher

  fetcher = ResultsFetcher.__new__(ResultsFetcher)
  row = fetcher.classification_performance_report(
    [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], "default", cutoff
  )
  assert row["cutoff"] == "default" and row["auroc"] == 0.75
  if expected_tp is None:
    assert np.isnan(row["mcc"])
  else:
    assert row["tp"] == expected_tp and row["fp"] == 1


def test_every_cutoff_is_also_a_rank():
  from lazyqsar.utils.ranking import prepare_knots, rank_from_reference, subsample_knots

  P, R, S, A, y, meta = _signals()
  meta["balanced"] = [0.45, 0.55, 0.50]
  rng = np.random.default_rng(1)
  reference = (rng.random((5000, 3)) * 0.8, rng.random((5000, 3)), rng.random((5000, 3)))
  params, _ = pooler.fit(P, R, S, A, y, NAMES, meta, reference=reference, kappa=lambda y, p: 0.4)
  ranks = params["cutoff_ranks"]
  # The screening cutoff is rank 0.65 by construction; lower probabilities rank lower.
  assert ranks["screening"] == pytest.approx(0.65, abs=1e-6)
  assert ranks["kappa"] < ranks["default"] < ranks["screening"]
  # A descriptor's balanced cutoff sits on that descriptor's own reference rank scale.
  ref = reference[0][:, 1]
  p, yy = P[:, 1], np.asarray(y)
  anchors = (np.percentile(p[yy == 0], 5), np.percentile(p[yy == 1], 95))
  expected = rank_from_reference(
    np.array([0.55]), prepared=prepare_knots(subsample_knots(np.sort(ref))), anchors=anchors
  )[0]
  assert ranks["balanced"][NAMES[1]] == pytest.approx(expected)
  # Without a rank (drift), the pooled ranks go and the per-descriptor ones stay.
  sub = pooler.without_rank(params)
  assert "screening" not in sub["cutoff_ranks"] and sub["cutoff_ranks"]["balanced"]
