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
