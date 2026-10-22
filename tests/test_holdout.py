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
