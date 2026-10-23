"""Invariants that only bite on large datasets, pinned on small ones.

A million-row run cannot be a test, so what is tested here is the behaviour that makes one
possible: that the readers assemble rows in the order their labels are indexed by, that they do
not hold more of a matrix than they have to, and that work which should not grow with the dataset
does not.
"""

import numpy as np
import pytest


class _Chunked:
  """The slice of a Fitter that ``_load_train_rows`` uses, over an in-memory matrix.

  Chunk boundaries are deliberately uneven: ChunkedH5Store yields the chunks as they were
  written, which need not line up with any batch size the caller has in mind.
  """

  model_id = "eosTEST"

  def __init__(self, X, sizes=(7, 13, 5)):
    self.X = X
    self.sizes = sizes

  def _iter_X(self):
    start, k = 0, 0
    while start < self.X.shape[0]:
      end = min(start + self.sizes[k % len(self.sizes)], self.X.shape[0])
      yield start, end, self.X[start:end].copy()
      start, k = end, k + 1


def _load(X, train_idxs, sizes=(7, 13, 5)):
  from zairachem.estimate.estimators.lazy_qsar.estimate import Fitter

  return Fitter._load_train_rows(_Chunked(X, sizes), train_idxs)


@pytest.mark.parametrize("shuffled", [False, True])
def test_training_rows_and_their_labels_stay_paired(shuffled):
  """The failure this guards is silent: X is assembled chunk by chunk while y is indexed by the
  returned order, so if the two ever disagree the model trains on mismatched pairs and nothing
  downstream can tell. Every row has to be the row its index names."""
  rng = np.random.default_rng(0)
  n, f = 60, 4
  X = rng.random((n, f), dtype=np.float32)
  train = np.arange(n)[rng.random(n) < 0.8]
  if shuffled:
    train = train.copy()
    rng.shuffle(train)  # the production fitter's indices are shuffled, a fold's are sorted

  X_train, order = _load(X, train)

  assert X_train.shape == (train.size, f)
  assert sorted(order.tolist()) == sorted(train.tolist())  # the same rows, however ordered
  for k, global_row in enumerate(order):
    assert np.array_equal(X_train[k], X[global_row]), k


def test_every_asked_for_row_is_found():
  """A matrix that does not cover the run's rows must say so, not train on a short matrix."""
  rng = np.random.default_rng(0)
  X = rng.random((20, 3), dtype=np.float32)
  with pytest.raises(RuntimeError, match="does not cover"):
    _load(X, np.array([1, 5, 25]))  # row 25 is past the end
  with pytest.raises(RuntimeError, match="no training rows"):
    _load(X, np.array([], dtype=int))


def test_the_training_matrix_is_not_held_twice():
  """Assembling with np.concatenate holds the per-chunk pieces and the result at the same time,
  so it peaks at twice the matrix. For a wide descriptor over a million rows that is the
  difference between fitting and running out of memory."""
  import tracemalloc

  rng = np.random.default_rng(0)
  n, f = 4000, 256
  X = rng.random((n, f), dtype=np.float32)
  train = np.arange(n)

  tracemalloc.start()
  X_train, _ = _load(X, train, sizes=(250,))
  _, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  matrix = X_train.nbytes
  assert peak < matrix * 1.5, f"peaked at {peak / matrix:.2f}x the matrix"


def test_butina_distances_are_an_array_not_boxed_floats():
  """Butina compares every pair, so it holds n(n-1)/2 distances. Kept as a Python list each one
  is a separate object and the same numbers cost four times the memory (248 MB against 61 MB at
  n=4,000), which is what decides how large a set the schema can be offered on at all. The
  clusters must not change."""
  from rdkit import Chem, DataStructs, RDLogger
  from rdkit.Chem import rdFingerprintGenerator
  from rdkit.ML.Cluster import Butina

  from zairachem.holdout.splits import _BUTINA_CUTOFF, _groups_butina

  RDLogger.DisableLog("rdApp.*")
  smiles = [
    "CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccccc1C", "c1ccccc1O", "CC(=O)O", "CC(=O)N",
    "C1CCCCC1", "C1CCNCC1", "CCS", "CCF", "c1ccncc1", "CC(C)O", "OCCO", "NCCN", "CCCCCC",
  ]  # fmt: skip

  # The reference: condensed distances built the way a list comprehension would.
  gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
  fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
  ref = []
  for i in range(1, len(fps)):
    ref.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
  expected = Butina.ClusterData(ref, len(fps), _BUTINA_CUTOFF, isDistData=True)

  groups = _groups_butina(smiles, _BUTINA_CUTOFF)
  rebuilt = {}
  for position, g in enumerate(groups):
    rebuilt.setdefault(int(g), []).append(position)
  assert sorted(sorted(v) for v in rebuilt.values()) == sorted(sorted(c) for c in expected)

  # Degenerate sets have no pairs at all and must not divide by anything.
  assert _groups_butina([], _BUTINA_CUTOFF).tolist() == []
  assert _groups_butina(["CCO"], _BUTINA_CUTOFF).tolist() == [0]
  assert len(set(_groups_butina(["%%%", "&&&"], _BUTINA_CUTOFF).tolist())) == 2


def test_butina_is_refused_rather_than_attempted_on_a_large_set(monkeypatch):
  """Every pair means tens of GB and the better part of an hour at 50,000 molecules, to produce
  nothing. Offering it anyway is how a large run dies in the first step."""
  import numpy as np

  from zairachem.holdout import splits
  from zairachem.holdout.splits import _BUTINA_MAX_MOLECULES, select_folds

  said = []
  monkeypatch.setattr(splits.logger, "warning", lambda m, *a, **k: said.append(str(m)))
  # A fixed size, not one derived from the constant: a test that sizes its own input from the
  # ceiling moves with it and can never catch the ceiling being raised.
  big = ["CCO", "c1ccccc1"] * 6000  # 12,000 molecules, 72 million pairs
  y = np.arange(len(big)) % 2
  assert select_folds(big, y, "butina", 1) == []
  # Specifically the ceiling, not select_folds' own "produced no valid split" message, which also
  # names the strategy and also says "skipping".
  assert any("every pair" in m and "the limit is" in m for m in said), said

  # And only butina: a large set still gets its other folds, which is the point of refusing.
  said.clear()
  assert select_folds(big, y, "random", 1), "the ceiling must not reach the linear schemas"
  assert not any("every pair" in m for m in said), said

  # And the ceiling has to stay where the work is affordable. Butina holds n(n-1)/2 float64
  # distances before it clusters anything, so the constant is really a memory budget.
  implied_gb = _BUTINA_MAX_MOLECULES * (_BUTINA_MAX_MOLECULES - 1) / 2 * 8 / 2**30
  assert implied_gb < 0.5, f"the ceiling implies a {implied_gb:.1f} GB distance matrix"
