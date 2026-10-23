"""Held-out validation: split generation and the summary written for the report."""

import json

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


@pytest.mark.parametrize("strategy", ["random", "scaffold", "butina"])
def test_a_continuous_target_splits_too(strategy):
  """A float target must not be treated as a class label (it used to raise)."""
  result = split(SMILES, VALUES, strategy, seed=0)
  if result is None:
    return
  train, test = result
  assert set(train).isdisjoint(test)
  assert sorted([*train, *test]) == list(range(len(SMILES)))


def test_random_folds_stratify_on_value_quantiles():
  """The test slice should span the target's range, not a corner of it."""
  train, test = split(SMILES, VALUES, "random", seed=0)
  held = VALUES[list(test)]
  # Stratifying on quartiles puts at least one compound from the top and bottom quarter in the
  # test slice; an unstratified split of a sorted target routinely does not.
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


def test_a_piped_fold_line_carries_its_metrics(capsys):
  """Off-TTY there is no live table, so the scores would otherwise reach the terminal nowhere."""
  from zairachem.holdout.monitor import EvaluateMonitor

  state = {"status": "done", "elapsed": 12.0, "extra": {"r2": 0.669, "spearman": 0.754}}
  EvaluateMonitor(["scaffold_00"], task="regression")._plain_finish("scaffold_00", state)
  out = capsys.readouterr().out
  assert "0.669" in out and "0.754" in out and "scaffold_00" in out

  clf = {"status": "done", "elapsed": 12.0, "extra": {"auroc": 0.93, "aupr": 0.94}}
  EvaluateMonitor(["random_00"], task="classification")._plain_finish("random_00", clf)
  out = capsys.readouterr().out
  assert "AUROC" in out and "0.930" in out

  # A fold that failed to score says so without inventing a number.
  blank = {"status": "failed", "elapsed": None, "extra": {}}
  EvaluateMonitor(["x"], task="regression")._plain_finish("x", blank)
  assert "failed" in capsys.readouterr().out


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
