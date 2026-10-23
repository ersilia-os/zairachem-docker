"""Held-out validation: that the splits are honest, and that the folds report what they scored.

A fold is only worth reading if its test rows never reached training and its test slice spans the
target, so those are what these pin -- for a binary label and for a continuous one -- along with
the two cases where a split cannot be drawn at all and has to say so.
"""

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
