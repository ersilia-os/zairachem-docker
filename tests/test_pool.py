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


def test_every_pooler_attribute_the_pool_step_names_exists():
  """The pruning warning referenced two constants that had been renamed away, and nothing ran it.

  A warning path that fires only when a descriptor is pruned is exactly the kind that no run
  exercises until it matters, so the reference is checked statically instead.
  """
  import ast
  import inspect

  from zairachem.pool.reliability_pooler import pool as pool_module

  tree = ast.parse(inspect.getsource(pool_module))
  named = {
    node.attr
    for node in ast.walk(tree)
    if isinstance(node, ast.Attribute)
    and isinstance(node.value, ast.Name)
    and node.value.id == "pooler"
  }
  missing = sorted(n for n in named if not hasattr(pooler, n))
  assert not missing, f"pool.py names pooler.{{{', '.join(missing)}}}, which do not exist"


def test_one_descriptor_reports_no_spread_rather_than_zero_spread():
  """``spread`` is disagreement between descriptors; with one there is none to measure.

  ``np.nanstd`` of a single column is 0, which in the delivered column reads as "every descriptor
  agreed" -- the strongest confidence signal there is -- when nothing was compared at all.
  """
  import numpy as np

  from zairachem.pool.reliability_pooler.pool import descriptor_spread

  assert np.isnan(descriptor_spread(np.array([[0.2], [0.8], [0.5]]))).all()

  two = descriptor_spread(np.array([[0.2, 0.6], [0.8, 0.4], [0.5, 0.5]]))
  np.testing.assert_allclose(two, [0.2, 0.2, 0.0])

  # A descriptor that failed on a compound is skipped, not counted as a zero.
  with_gap = descriptor_spread(np.array([[0.2, np.nan, 0.6], [0.1, 0.5, 0.9]]))
  np.testing.assert_allclose(with_gap, [0.2, np.std([0.1, 0.5, 0.9])])


@pytest.mark.parametrize(
  ("task", "suffix", "oof_key", "train_key"),
  [
    ("classification", "clf", "oof_auc", "train_auc"),
    ("regression", "reg", "oof_rho", "train_rho"),
  ],
)
def test_the_pooler_reads_the_skill_its_estimator_wrote(tmp_path, task, suffix, oof_key, train_key):
  """Each task records its held-out skill under its own key, and the pooler must ask for that one.

  Reading the classifier's key on a regression run gave every descriptor a skill of None, which
  `active_mask` treats as unknown and keeps: pruning silently stopped happening, and the weighting
  the model falls back to when descriptors drift at predict became uniform.
  """
  import json

  import joblib

  from zairachem.pool.base import BasePooler

  root = tmp_path / task
  (root / "metadata").mkdir(parents=True)
  (root / "metadata" / "parameters.json").write_text(json.dumps({"task": task}))
  (root / "descriptors").mkdir()
  (root / "descriptors" / "done_eos.json").write_text(json.dumps(["eosA", "eosB"]))
  (root / "session.json").write_text(
    json.dumps({"mode": "fit", "output_dir": str(root), "model_dir": str(root)})
  )
  for eos, skill in (("eosA", 0.81), ("eosB", 0.42)):
    d = root / "model" / "estimators" / "lq_estimators" / eos
    d.mkdir(parents=True)
    joblib.dump({oof_key: skill, "ad_hard_cutoff": 0.05}, d / "pool_signals.joblib")
    (d / "cv_report.json").write_text(json.dumps({train_key: skill + 0.1}))

  io = BasePooler(path=str(root))
  columns = [f"lq_estimators-{eos}-{suffix}" for eos in ("eosA", "eosB")]
  meta = io._get_descriptor_meta(columns, task)
  assert meta["oof_aucs"] == [0.81, 0.42]
  assert meta["train_aucs"] == [pytest.approx(0.91), pytest.approx(0.52)]


def test_descriptors_that_disagree_on_their_compounds_are_refused(tmp_path):
  """The per-descriptor matrices are stacked side by side by position, with the ids dropped.

  A descriptor whose rows ever came back in another order would be pooled against a different
  descriptor's compounds — every prediction wrong, and nothing in the output to show it.
  """
  import json

  import pandas as pd

  from zairachem.pool.base import XGetter

  root = tmp_path / "run"
  (root / "descriptors").mkdir(parents=True)
  (root / "descriptors" / "done_eos.json").write_text(json.dumps(["eosA", "eosB"]))
  (root / "model" / "pool").mkdir(parents=True)
  (root / "session.json").write_text(
    json.dumps({"mode": "fit", "output_dir": str(root), "model_dir": str(root)})
  )
  cids = [f"CID{i:03d}" for i in range(6)]

  def write(eos, ids):
    d = root / "model" / "estimators" / "lq_estimators" / eos
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"compound_id": ids, "smiles": ["CCO"] * len(ids), "clf": range(len(ids))}).to_csv(
      d / "results_unmapped.csv", index=False
    )

  write("eosA", cids)
  write("eosB", cids)
  assert XGetter(path=str(root)).get().shape == (6, 2)  # agreeing descriptors pool as before

  write("eosB", cids[::-1])
  with pytest.raises(RuntimeError, match="different compound"):
    XGetter(path=str(root)).get()

  write("eosB", cids[:4])
  with pytest.raises(RuntimeError, match="result rows"):
    XGetter(path=str(root)).get()


def test_a_finished_run_does_not_need_the_user_s_input_file_any_more(tmp_path):
  """Setup copies the input into the run, so nothing later should reach back to where it was.

  Reading the schema's path meant a finished model could not be re-pooled, or moved to another
  machine, once the input was renamed or tidied away.
  """
  import json

  import pandas as pd

  from zairachem.pool.base import BaseOutcomeAssembler

  root = tmp_path / "run"
  (root / "inputs").mkdir(parents=True)
  pd.DataFrame({"smiles": ["CCO"] * 7}).to_csv(root / "inputs" / "raw_input.csv", index=False)
  (root / "inputs" / "input_schema.json").write_text(
    json.dumps({"input_file": str(tmp_path / "moved-away.csv"), "smiles_column": "smiles"})
  )

  assembler = BaseOutcomeAssembler.__new__(BaseOutcomeAssembler)
  assembler.path = str(root)
  assert assembler._get_original_input_size() == 7


def test_molecules_the_model_was_trained_on_are_called_out(tmp_path):
  """`similarity` cannot say a query was in the training set -- it leaves the query's own entry
  out -- so a library overlapping the training set reads as a clean prediction. The pooler
  records the overlap; the closing panel and the report both have to surface it."""
  import inspect
  import json

  from zairachem.base.utils import summaries
  from zairachem.base.utils.summaries import _note_training_overlap
  from zairachem.base.vars import DATA_SUBFOLDER, TRAIN_OVERLAP_FILENAME
  from zairachem.pool.reliability_pooler.pool import Predictor, write_training_overlap
  from zairachem.report.html.sections import _config_section_html

  # Deliberately the input folder, not model/: at predict `finish` removes model/ wholesale, so a
  # record written there would be gone before the closing panel ever read it.
  assert "write_training_overlap(os.path.join(self.path, DATA_SUBFOLDER)" in inspect.getsource(
    Predictor._train_similarity
  )
  pool_dir = tmp_path / DATA_SUBFOLDER
  pool_dir.mkdir(parents=True)
  # "CCO" is predicted twice and was trained on; "CCN" is new.
  summary = write_training_overlap(str(pool_dir), ["CCO", "CCN", "CCO"], ["CCO", "CCC"])
  assert summary == {"n_query": 3, "n_in_training": 2, "n_distinct_in_training": 1}
  assert json.loads((pool_dir / TRAIN_OVERLAP_FILENAME).read_text()) == summary

  rows = []
  _note_training_overlap(rows, str(tmp_path))
  assert "2 of 3" in rows[0][1] and "recalling" in rows[0][1]
  # Only at predict, and only from the panel -- a fit run has no trained_dir to compare against.
  assert "_note_training_overlap" in inspect.getsource(summaries.final_summary_panel)
  # The pooler is the only place holding both sets of SMILES, so it is the only place that can
  # record this; nothing above runs if it stops doing so.
  assert "write_training_overlap" in inspect.getsource(Predictor._train_similarity)

  assert "Seen in training" in _config_section_html(str(tmp_path), {"task": "regression"}, 3)

  # A run that shares nothing says nothing, in either place.
  write_training_overlap(str(pool_dir), ["CCN"], ["CCO"])
  quiet = []
  _note_training_overlap(quiet, str(tmp_path))
  assert quiet == []
  assert "Seen in training" not in _config_section_html(str(tmp_path), {"task": "regression"}, 1)


def test_a_descriptor_that_costs_the_fit_and_carries_nothing_is_named(tmp_path):
  """Pre-screening ranks descriptors on a shallow forest that costs the same for all of them, so
  two that differ by an order of magnitude in training cost look equally cheap and both are kept.
  Only the finished fit knows, and on real data it was 95% of the time for 16% of the weight."""
  import inspect
  import json
  import shutil

  from zairachem.base.utils import summaries
  from zairachem.base.utils.summaries import _human_seconds, _note_expensive_descriptors
  from zairachem.base.vars import ESTIMATORS_SUBFOLDER, POOL_SUBFOLDER

  pool = tmp_path / POOL_SUBFOLDER
  pool.mkdir(parents=True)

  def build(weights, seconds):
    (pool / "reliability_summary.json").write_text(json.dumps({"mean_weights": weights}))
    shutil.rmtree(tmp_path / ESTIMATORS_SUBFOLDER, ignore_errors=True)
    for eos, s in seconds.items():
      d = tmp_path / ESTIMATORS_SUBFOLDER / "lq_estimators" / eos
      d.mkdir(parents=True, exist_ok=True)
      (d / "cv_report.json").write_text(json.dumps({"fit_seconds": s}))
    rows = []
    _note_expensive_descriptors(rows, str(tmp_path))
    return rows

  rows = build({"eosA": 0.84, "eosB": 0.16}, {"eosA": 142.5, "eosB": 2564.3})
  assert "eosB took 43m of the 45m spent training (95%) for 16%" in rows[0][1]

  # The expensive one earning its keep is not a complaint.
  assert build({"eosA": 0.2, "eosB": 0.8}, {"eosA": 142.5, "eosB": 2564.3}) == []
  # Nor is merely being the slower of two: with two descriptors one almost always takes over half
  # the time, and half the time for a tenth of the weight is not the case this is for.
  assert build({"eosA": 0.9, "eosB": 0.1}, {"eosA": 500.0, "eosB": 520.0}) == []
  # Nor one whose cost is roughly in proportion to what it contributes.
  assert build({"eosA": 0.3, "eosB": 0.7}, {"eosA": 300.0, "eosB": 700.0}) == []
  # A single descriptor has nothing to be compared against -- including when the pool names a
  # second one whose timing never reached disk, where the one that did would otherwise look like
  # it had taken all of the time for a fraction of the weight.
  assert build({"eosA": 1.0}, {"eosA": 2564.3}) == []
  assert build({"eosA": 0.2, "eosB": 0.8}, {"eosA": 2564.3}) == []

  assert (_human_seconds(18), _human_seconds(2564), _human_seconds(9000)) == ("18s", "43m", "2.5h")
  # Only a fit has both halves of this; predict trains nothing.
  src = inspect.getsource(summaries.final_summary_panel)
  assert "_note_expensive_descriptors" in src and "if not _is_predict(d)" in src


def test_a_descriptor_no_fold_could_score_does_not_win_the_screen(monkeypatch):
  """`_fold_pred` returns None for a fold it cannot score, so a descriptor that fails on every
  fold is a list of Nones, not None -- and only None was treated as unusable. When nothing clears
  the floor the greedy seed falls back to the best raw score, and an unscorable descriptor scores
  exactly chance, which beats any descriptor that is genuinely anti-correlated. The screen then
  seeds on a descriptor it never managed to evaluate and the estimate step trains it."""
  import numpy as np

  from zairachem.screen import proxy

  y = np.linspace(4.0, 9.0, 40)
  folds = [(list(range(30)), list(range(30, 40)))]
  current = {}

  def load(_dir, eos_id):
    current["eos"] = eos_id
    return np.zeros((40, 2), dtype="float32")

  def fold_pred(_X, yy, _tr, te, _seed, _task):
    if current["eos"] == "dead":  # every fold declines to score it
      return None
    return -yy[np.asarray(te)]  # anti-correlated: a real, below-chance score

  monkeypatch.setattr(proxy, "_load_treated", load)
  monkeypatch.setattr(proxy, "_fold_pred", fold_pred)

  selected, scores, _ = proxy.select_descriptors(
    "d", ["dead", "anti"], y, folds, 1, task="regression"
  )
  assert scores["dead"] == 0.0 and scores["anti"] < 0.0  # chance beats anti-correlated
  assert selected == ["anti"], selected

  # With nothing scorable at all, the screen keeps every descriptor rather than seeding on one
  # it never evaluated.
  selected, _, _ = proxy.select_descriptors("d", ["dead"], y, folds, 1, task="regression")
  assert selected == ["dead"]


def test_the_screen_shows_what_each_descriptor_added(monkeypatch):
  """Selection is greedy on the ensemble, so the solo score is not what decided anything: a
  strong descriptor that repeats the seed is dropped and a weaker one that complements it is
  kept. The table showed only the solo score, and a descriptor kept on a hair's breadth of gain
  looked the same as one that carried the model -- while costing far more to train."""
  import numpy as np

  from zairachem.screen import proxy
  from zairachem.screen.pipe import ScreenPipeline

  rng = np.random.default_rng(0)
  n = 200
  y = np.linspace(4.0, 9.0, n)
  idx = np.arange(n)
  folds = [(list(np.setdiff1d(idx, idx[k::3])), list(idx[k::3])) for k in range(3)]
  # "seed" and "adds" carry the target through independent noise, so averaging them beats either;
  # "copy" repeats the seed exactly and can add nothing however good its own score looks.
  seed = y + rng.normal(0, 1.5, n)
  signal = {"seed": seed, "adds": y + rng.normal(0, 1.5, n), "copy": seed}

  def load(_dir, eos_id):
    load.current = eos_id
    return np.zeros((n, 2), dtype="float32")

  monkeypatch.setattr(proxy, "_load_treated", load)
  monkeypatch.setattr(
    proxy, "_fold_pred", lambda _X, _y, _tr, te, _s, _t: signal[load.current][np.asarray(te)]
  )

  selected, _scores, gains = proxy.select_descriptors(
    "d", ["seed", "copy", "adds"], y, folds, 3, task="regression"
  )
  assert gains[selected[0]] is None  # the seed has nothing to be measured against
  for eos_id in selected[1:]:
    assert gains[eos_id] > 0  # everything after it had to earn its place
  # A descriptor that was not kept has no gain to report, rather than a zero.
  assert set(gains) <= set(selected)

  rendered = []
  monkeypatch.setattr("zairachem.screen.pipe.console.print", lambda t: rendered.append(t))
  ScreenPipeline._print_table(
    {"a": 0.65, "b": 0.63, "c": 0.60}, {"a", "b"}, 0.10, "regression", {"a": None, "b": 0.0031}
  )
  cells = [c._cells for c in rendered[0].columns]
  assert "Adds" in [c.header for c in rendered[0].columns]
  assert any("seed" in str(x) for x in cells[2]) and any("+0.003" in str(x) for x in cells[2])
