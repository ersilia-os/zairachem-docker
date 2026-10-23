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
