"""Held-out validation: that the splits are honest, and that the folds report what they scored.

A fold is only worth reading if its test rows never reached training and its test slice spans the
target, so those are what these pin -- for a binary label and for a continuous one -- along with
the two cases where a split cannot be drawn at all and has to say so.
"""

import json
import os

import numpy as np
import pytest

from zairachem.holdout.io import write_validation_outputs
from zairachem.holdout.splits import split

SMILES = [
  "CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccccc1C", "c1ccccc1O", "c1ccccc1N",
  "c1ccncc1", "c1ccoc1", "c1ccsc1", "CC(=O)O", "CC(=O)N", "CCOC(C)=O", "CN(C)C", "CCS", "CCF",
  "C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "CC(C)O", "CC(C)N",
  "CC(C)C", "OCCO", "NCCN", "OC(=O)CC(=O)O", "CCCCCC",
]  # fmt: skip
LABELS = np.array([i % 2 for i in range(len(SMILES))])


@pytest.mark.parametrize("strategy", ["random", "scaffold", "butina"])
def test_split_is_a_partition(strategy):
  result = split(SMILES, LABELS, strategy, seed=0)
  if result is None:  # a strategy may legitimately fail to place both classes on each side
    return
  train, test = result
  assert set(train).isdisjoint(test)
  assert sorted([*train, *test]) == list(range(len(SMILES)))


def test_split_rejects_unknown_strategy():
  with pytest.raises(ValueError):
    split(SMILES, LABELS, "nope", seed=0)


def _record(fold, strategy, auroc):
  return {
    "fold": fold,
    "strategy": strategy,
    "seed": 0,
    "auroc": auroc,
    "aupr": 0.7,
    "_y_true": [1, 0],
    "_y_score": [0.9, 0.1],
  }


def test_summary_counts_defined_and_scored_folds(tmp_path):
  folds = {"a": {}, "b": {}}
  write_validation_outputs(str(tmp_path), folds, [_record("a", "random", 0.8)])
  summary = json.loads((tmp_path / "report" / "holdout_summary.json").read_text())
  assert summary["n_folds_defined"] == 2
  assert summary["n_folds_run"] == 1
  assert (tmp_path / "report" / "validation_table.csv").exists()


def test_fold_workspace_has_its_own_descriptor_selection(tmp_path):
  import json

  from zairachem.base.utils.descriptors import effective_descriptors
  from zairachem.holdout.workspace import build_fold_workspace

  run = tmp_path / "run"
  for sub in ("metadata", "descriptors", "inputs"):
    (run / sub).mkdir(parents=True)
  (run / "metadata" / "parameters.json").write_text("{}")
  (run / "metadata" / "selected_eos.json").write_text(json.dumps(["eosA", "eosB"]))
  (run / "descriptors" / "done_eos.json").write_text(json.dumps(["eosA", "eosB", "eosC"]))

  fold = build_fold_workspace(str(run), "random_00", ["eosB", "eosC"])
  # Every step resolving the fold's descriptors (estimate, pool) sees the fold's own selection,
  assert effective_descriptors(fold) == ["eosB", "eosC"]
  # the parent's is untouched, and the other metadata is shared.
  assert effective_descriptors(str(run)) == ["eosA", "eosB"]
  assert (tmp_path / "run" / "folds" / "random_00" / "metadata" / "parameters.json").is_symlink()
  # Rebuilding (a resumed run) keeps it consistent.
  assert effective_descriptors(build_fold_workspace(str(run), "random_00", ["eosA"])) == ["eosA"]


# --- Regression folds: continuous targets stratify, and report their own metrics ----------------

VALUES = np.linspace(4.0, 9.0, len(SMILES))


def test_a_continuous_target_splits_and_stratifies():
  """A float target must not be treated as a class label (it used to raise)."""
  for strategy in ("random", "scaffold", "butina"):
    result = split(SMILES, VALUES, strategy, seed=0)
    if result is None:
      continue
    train, test = result
    assert set(train).isdisjoint(test)
    assert sorted([*train, *test]) == list(range(len(SMILES)))

  # Stratifying on quartiles puts the top and bottom of the range in the test slice; an
  # unstratified split of a sorted target routinely takes one corner of it.
  _, test = split(SMILES, VALUES, "random", seed=0)
  held = VALUES[list(test)]
  assert held.min() < np.percentile(VALUES, 50) < held.max()


def _reg_record(fold, strategy, r2):
  return {
    "fold": fold,
    "strategy": strategy,
    "seed": 0,
    "r2": r2,
    "rmse": 0.8,
    "mae": 0.6,
    "spearman": 0.75,
    "num_compounds": 2,
    "_y_true": [7.92, 4.07],
    "_y_score": [8.1, 4.4],
  }


def test_regression_folds_write_their_own_metrics_and_keep_the_truth(tmp_path):
  import csv

  write_validation_outputs(str(tmp_path), {"a": {}}, [_reg_record("a", "scaffold", 0.66)])
  rows = list(csv.DictReader((tmp_path / "report" / "validation_table.csv").open()))
  assert set(rows[0]) == {
    "strategy",
    "fold",
    "seed",
    "r2",
    "rmse",
    "mae",
    "spearman",
    "num_compounds",
  }
  assert rows[0]["r2"] == "0.66" and rows[0]["num_compounds"] == "2"
  # The per-compound file keeps the target's own precision: rounding 7.92 to 7 would be silent
  # corruption of the only honest held-out record the run keeps.
  preds = list(csv.DictReader((tmp_path / "report" / "validation_predictions.csv").open()))
  assert [p["y_true"] for p in preds] == ["7.92", "4.07"]


def test_classification_folds_still_report_their_class_split(tmp_path):
  import csv

  record = {**_record("a", "random", 0.8), "num_compounds": 2, "num_0": 1, "num_1": 1}
  write_validation_outputs(str(tmp_path), {"a": {}}, [record])
  rows = list(csv.DictReader((tmp_path / "report" / "validation_table.csv").open()))
  assert rows[0]["auroc"] == "0.8" and rows[0]["num_1"] == "1" and rows[0]["num_0"] == "1"
  assert "r2" not in rows[0]


def test_a_target_with_no_range_builds_no_folds_and_says_so():
  """A fold covering one slice of the value range scores nothing, so it is dropped, not scored."""
  from zairachem.holdout.splits import _both_classes, build_fold_definitions, strata

  flat = np.array([5.0] * 29 + [9.0])  # one outlier; every compound lands in one quantile bin
  assert len(set(strata(flat).tolist())) == 1
  assert not _both_classes(flat, np.arange(24), np.arange(24, 30))
  assert build_fold_definitions(["CCO"] * 30, flat, repeats=1, schemas=["random"]) == {}
  # A target with a real range still produces folds.
  spread = np.linspace(4.0, 9.0, 30)
  assert build_fold_definitions(SMILES, spread, repeats=1, schemas=["random"])


def test_a_fold_that_holds_out_most_of_the_set_says_so(monkeypatch):
  """A group schema moves whole scaffolds, so some sets cannot be split 80:20 at all.

  On a congeneric series one scaffold can be most of the data, and the smallest held-out set the
  schema can build is most of the data with it. The score is still honest, but it is the score of
  a model trained on what is left — which is not the model being shipped, and nothing else on
  screen says so.
  """
  from zairachem.holdout import splits as splits_module
  from zairachem.holdout.splits import select_folds

  said = []
  # ZairaChem logs through loguru, which pytest's caplog does not see.
  monkeypatch.setattr(splits_module.logger, "warning", lambda msg, *a, **k: said.append(str(msg)))

  rng = np.random.default_rng(0)
  dominant = ["c1ccc2c(c1)ncnc2N"] * 170
  rest = [f"C{'C' * i}N1CCOCC1" for i in range(30)]
  smiles = dominant + rest
  y = rng.normal(6.0, 1.2, len(smiles))

  folds = select_folds(smiles, y, "scaffold", k=1)
  assert folds, "the fold is still produced; it is only called out"
  assert len(folds[0]["test_idx"]) / len(smiles) > 0.5
  assert any("were held out" in m for m in said), said

  # A set with enough distinct scaffolds splits as asked, and says nothing.
  cores = ["c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "C1CCCCC1", "C1CCNCC1", "c1cnc2ccccc2c1"]
  diverse = [f"{'C' * (i % 5 + 1)}{cores[i % len(cores)]}" for i in range(200)]
  said.clear()
  folds = select_folds(diverse, rng.normal(6.0, 1.2, 200), "scaffold", k=1)
  assert 0.1 <= len(folds[0]["test_idx"]) / 200 <= 0.3
  assert not any("were held out" in m for m in said), said


def test_the_panel_shows_the_held_out_skill_beside_the_out_of_fold_one(tmp_path):
  """The Performance row is out-of-fold, on randomly drawn folds. On congeneric data that is the
  score on more of the same chemistry. `--evaluate` already measured the harder question by
  holding out whole scaffolds, and the answer was reaching a CSV and not the panel."""
  import inspect
  import json

  from zairachem.base.utils import summaries
  from zairachem.base.utils.summaries import _note_held_out_skill

  report = tmp_path / "report"
  report.mkdir()

  def write(scaffold):
    (report / "holdout_summary.json").write_text(json.dumps({"strategies": {"scaffold": scaffold}}))

  # A held-out R2 far under the out-of-fold one is the case worth saying out loud (gap > 0.20).
  write({"n_folds": 3, "r2": {"mean": 0.44}})
  rows = []
  _note_held_out_skill(rows, str(tmp_path), "regression", {"r2": 0.72})
  assert rows[0][0] == "Held out" and "0.440" in rows[0][1] and "3 folds" in rows[0][1]
  assert "falls to 0.44" in rows[1][1]

  # A model that holds up is reported without the caution.
  rows = []
  _note_held_out_skill(rows, str(tmp_path), "regression", {"r2": 0.55})
  assert len(rows) == 1 and rows[0][0] == "Held out"

  # Classification is judged on AUROC, against its own (tighter) gap of 0.10.
  write({"n_folds": 2, "auroc": {"mean": 0.61}})
  rows = []
  _note_held_out_skill(rows, str(tmp_path), "classification", {"auroc": 0.78})
  assert "AUROC" in rows[0][1] and "falls to 0.61" in rows[1][1]

  # No scaffold folds (all failed, or --evaluate never ran) means no row, not a blank one.
  write({"n_folds": 0})
  rows = []
  _note_held_out_skill(rows, str(tmp_path), "regression", {"r2": 0.72})
  assert rows == []
  # Nor a "scaffold, None folds" row from a summary that carries a mean but no count.
  write({"r2": {"mean": 0.44}})
  _note_held_out_skill(rows, str(tmp_path), "regression", {"r2": 0.72})
  assert rows == []
  (report / "holdout_summary.json").unlink()
  _note_held_out_skill(rows, str(tmp_path), "regression", {"r2": 0.72})
  assert rows == []

  # And the panel is where it has to appear.
  assert "_note_held_out_skill" in inspect.getsource(summaries.final_summary_panel)


def test_a_fold_missing_a_class_on_either_side_is_rejected():
  """A fold whose test slice is all one class has no AUROC, and one whose train slice is cannot
  learn the other. Both are dropped before they are scored, so the guard has to look at each side
  separately -- checking only that the dataset has both classes would pass either."""
  from zairachem.holdout.splits import _both_classes

  y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
  assert _both_classes(y, np.array([0, 1, 4, 5]), np.array([2, 3, 6, 7]))
  assert not _both_classes(y, np.array([0, 1, 4, 5]), np.array([2, 3]))  # test side all inactive
  assert not _both_classes(y, np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]))  # train all inactive
  assert not _both_classes(y, np.array([0, 1]), np.array([2, 4]))  # train one class, test both
  assert not _both_classes(np.zeros(8), np.arange(4), np.arange(4, 8))  # nothing to separate
  assert not _both_classes(y, np.array([], dtype=int), np.arange(8))  # an empty side


# --- A fold must be savable, and a failed evaluation must not look like a finished model --------


def _fold_spec(strategy="scaffold", n=8):
  return {"strategy": strategy, "seed": 0, "train_idx": list(range(n)), "test_idx": [n, n + 1]}


def _stub_the_slow_parts(monkeypatch, tmp_path, task):
  """Replace the refit and the pooling, keeping the REAL metrics helpers the fold is scored with."""
  from zairachem.holdout import engine
  from zairachem.report.fetcher import ResultsFetcher

  monkeypatch.setattr(engine, "build_fold_workspace", lambda *a, **k: str(tmp_path))
  monkeypatch.setattr(engine, "_fit_descriptors", lambda *a, **k: None)
  fetcher = object.__new__(ResultsFetcher)  # no run folder, no containers: the helpers need none

  if task == "classification":
    y = np.array([0, 1, 0, 1, 1, 0])
    score = np.array([0.2, 0.9, 0.4, 0.6, 0.3, 0.1])

    def score_fold(path, train_idx, test_idx, fetcher, batch_size):
      # The tail of the real _score_fold: both reports come from the shared fetcher.
      report = fetcher.classification_performance_report(y, score, "kappa", 0.5)
      at_default = fetcher.classification_performance_report(y, score, "default", 0.5)
      for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "mcc"):
        report[f"{k}_default"] = at_default[k]
      return report, y, score

  else:
    y = np.linspace(4.0, 9.0, 6)
    score = y[::-1] + 0.3

    def score_fold(path, train_idx, test_idx, fetcher, batch_size):
      return fetcher.regression_performance_report(y, score), y, score

  monkeypatch.setattr(engine, "_score_fold", score_fold)
  return fetcher


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_a_scored_fold_is_written_and_reads_back(tmp_path, monkeypatch, task):
  """The classification report returns its confusion-matrix counts as numpy.int64, which ``json``
  refuses: every classification fold was refitted in full and then died writing its own record."""
  from zairachem.holdout import engine

  fetcher = _stub_the_slow_parts(monkeypatch, tmp_path, task)
  record = engine.run_one_fold(str(tmp_path), "scaffold_00", _fold_spec(), ["eosA"], fetcher)

  assert (tmp_path / "metrics.json").exists() and (tmp_path / engine.HELD_OUT_FILENAME).exists()
  saved = json.loads((tmp_path / "metrics.json").read_text())
  assert saved["fold"] == "scaffold_00"
  if task == "classification":
    # Confusion matrix of the stub at 0.5: predicted 1 for 0.9, 0.6 -> tp 2, fp 0; tn 3, fn 1.
    assert (saved["tp"], saved["tn"], saved["fp"], saved["fn"]) == (2, 3, 0, 1)
    assert all(type(saved[k]) is int for k in ("tp", "tn", "fp", "fn", "num_0", "num_1"))
  # Nothing numpy is left in the record, and what was saved is what a later reader gets back.
  assert not [k for k, v in record.items() if isinstance(v, np.generic)]
  fold_dir = tmp_path / "folds" / "scaffold_00"
  fold_dir.mkdir(parents=True)
  engine._save_fold_record(str(fold_dir), record)
  assert engine.load_fold_record(str(tmp_path), "scaffold_00") == record
  write_validation_outputs(str(tmp_path), {"scaffold_00": {"strategy": "scaffold"}}, [record])
  assert (tmp_path / "report" / "validation_table.csv").exists()


def _pipeline(tmp_path, monkeypatch, fold_names, run_one_fold):
  """A HoldoutValidationPipeline over a run folder with ``fold_names`` and a fake fold runner."""
  from zairachem.holdout import engine, pipe
  from zairachem.report import fetcher

  run = tmp_path / "run"
  (run / "metadata").mkdir(parents=True)
  (run / "metadata" / "parameters.json").write_text(
    json.dumps({"evaluate": True, "task": "classification"})
  )
  (run / "metadata" / "splits.json").write_text(
    json.dumps({name: _fold_spec(name.split("_")[0]) for name in fold_names})
  )
  (run / "session.json").write_text(json.dumps({"steps": [], "mode": "fit"}))
  monkeypatch.setattr(engine, "run_one_fold", run_one_fold)
  monkeypatch.setattr(fetcher, "ResultsFetcher", lambda path=None: object())
  monkeypatch.setattr(pipe.HoldoutValidationPipeline, "_model_ids", lambda self: ["eosA"])
  return pipe.HoldoutValidationPipeline(path=str(run)), str(run)


def test_all_folds_failing_is_a_failed_step_not_a_finished_one(tmp_path, monkeypatch):
  """The Evaluate step used to close with a green tick on '0/3 folds (all failed)', the panel said
  the model was ready, and the exit code was 0."""
  from zairachem.base.utils import progress
  from zairachem.base.utils.pipeline import PipelineStep
  from zairachem.cli import steps

  def always_fails(*args, **kwargs):
    raise TypeError("Object of type int64 is not JSON serializable")

  pipeline, run = _pipeline(tmp_path, monkeypatch, ["random_00", "scaffold_00"], always_fails)
  closed = []
  monkeypatch.setattr(
    progress.tracker,
    "complete",
    lambda key, summary=None, ok=True: closed.append((key, summary, ok)),
  )
  validated = steps._tracked_step(
    "holdout", run, pipeline.run, succeeded=lambda result: result is not False
  )
  assert validated is False
  assert closed == [("holdout", "0/2 folds (all failed)", False)]
  assert not PipelineStep("holdout", run).is_done()

  # The failure travels from the step to the exit code through the return values; the model and
  # its report are written before anything is raised.
  seen = {}
  monkeypatch.setattr(steps, "_featurize", lambda *a, **k: None)
  monkeypatch.setattr(steps, "_estimate", lambda *a, **k: None)
  monkeypatch.setattr(steps, "_pool_and_validate", lambda *a, **k: validated)
  monkeypatch.setattr(steps, "_report_and_finish", lambda *a, **k: seen.setdefault("finished", 1))
  monkeypatch.setattr(
    progress,
    "final_summary_panel",
    lambda d, validation_failed=False: seen.update(f=validation_failed),
  )
  assert steps.process_group(run, False) is False
  assert seen == {"finished": 1, "f": True}


def test_fit_exits_non_zero_when_held_out_validation_failed(tmp_path, monkeypatch):
  from click.testing import CliRunner

  from zairachem.cli import cli
  from zairachem.cli.commands import fit as fit_cmd

  data = tmp_path / "x.csv"
  data.write_text("smiles,activity\nCCO,1\nCCN,0\n")
  monkeypatch.setattr("zairachem.setup.run_fit.run", lambda *a, **k: True)
  args = ["fit", "-i", str(data), "-m", str(tmp_path / "model"), "--classification"]
  for validated, code in ((False, 1), (True, 0)):
    monkeypatch.setattr(fit_cmd, "process_group", lambda *a, _v=validated, **k: _v)
    assert CliRunner().invoke(cli, args).exit_code == code


def test_the_panel_says_when_held_out_validation_failed(tmp_path):
  from zairachem.base.utils.console import console
  from zairachem.base.utils.summaries import final_summary_panel

  (tmp_path / "metadata").mkdir()
  (tmp_path / "metadata" / "parameters.json").write_text(json.dumps({"task": "classification"}))
  (tmp_path / "session.json").write_text(json.dumps({"steps": [], "mode": "fit"}))
  with console.capture() as ok:
    final_summary_panel(str(tmp_path))
  with console.capture() as failed:
    final_summary_panel(str(tmp_path), validation_failed=True)
  assert "model ready" in ok.get() and "Held-out validation failed" not in ok.get()
  assert "Held-out validation failed" in failed.get() and "usable" in failed.get()


def test_a_partial_failure_is_retried_on_re_run_and_only_for_the_failed_folds(
  tmp_path, monkeypatch
):
  """Some folds failing used to mark the step done, so a re-run never retried them; the finished
  folds are reused from their metrics.json, so only the failed ones may run again."""
  from zairachem.base.utils.pipeline import PipelineStep
  from zairachem.holdout import engine

  attempted, broken = [], {"scaffold_00"}

  def run_one_fold(model_dir, fold_name, spec, model_ids, fetcher, **kwargs):
    attempted.append(fold_name)
    if fold_name in broken:
      raise RuntimeError("boom")
    fold_dir = os.path.join(model_dir, "folds", fold_name)
    os.makedirs(fold_dir, exist_ok=True)
    record = {
      "fold": fold_name, "strategy": spec["strategy"], "seed": 0, "auroc": 0.8, "aupr": 0.7,
      "_y_true": [1, 0], "_y_score": [0.9, 0.1],
    }  # fmt: skip
    engine._save_fold_record(fold_dir, record)
    return record

  names = ["random_00", "scaffold_00", "butina_00"]
  pipeline, run = _pipeline(tmp_path, monkeypatch, names, run_one_fold)
  assert pipeline.run() is True  # two folds scored
  assert attempted == names
  assert not PipelineStep("holdout", run).is_done()
  summary = json.loads((tmp_path / "run" / "report" / "holdout_summary.json").read_text())
  assert summary["n_folds_failed"] == 1 and summary["strategies"]["random"]["n_defined"] == 1

  attempted.clear()
  broken.clear()
  assert pipeline.run() is True
  assert attempted == ["scaffold_00"]  # the two finished folds were kept, not refitted
  assert PipelineStep("holdout", run).is_done()
  summary = json.loads((tmp_path / "run" / "report" / "holdout_summary.json").read_text())
  assert summary["n_folds_run"] == 3 and summary["n_folds_failed"] == 0


def test_the_summary_says_how_many_of_the_requested_folds_it_covers(tmp_path):
  from zairachem.base.utils.summaries import summarize_holdout

  folds = {f"scaffold_0{i}": {"strategy": "scaffold"} for i in range(3)}
  records = [_record(f"scaffold_0{i}", "scaffold", 0.8) for i in range(2)]
  failures = [{"fold": "scaffold_02", "strategy": "scaffold", "error": "RuntimeError: boom"}]
  write_validation_outputs(str(tmp_path), folds, records, failures=failures)
  assert (
    summarize_holdout(str(tmp_path)) == "2/3 folds (1 failed) · scaffold AUROC 0.80 (2 of 3 folds)"
  )
