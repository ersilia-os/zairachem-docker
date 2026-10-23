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


def _store_pair(tmp_path, n, f):
  """The same matrix as a single HDF5 file and as a chunked store, with uneven chunks."""
  from zairachem.base.utils.matrices import ChunkedH5Store, Data, Hdf5

  X = np.arange(n * f, dtype=np.float32).reshape(n, f)
  feats = [f"f{i}" for i in range(f)]
  ins = [f"s{i}" for i in range(n)]
  single = str(tmp_path / "a.h5")
  data = Data()
  data.set(inputs=ins, values=X, features=feats)
  Hdf5(single).save(data)
  chunked = ChunkedH5Store(str(tmp_path / "b.h5"))
  chunked.create(f, feats)
  for s in range(0, n, 97):
    chunked.save_chunk(X[s : s + 97], ins[s : s + 97])
  return X, Hdf5(single), chunked


def test_a_row_subset_costs_the_subset(tmp_path):
  """Screening scores a sample of the rows. Reading the whole matrix to throw most of it away is
  what made the sample pointless: a wide descriptor over a million rows is 8 GB, per descriptor,
  plus a copy of most of it again for every fold's fit."""
  import tracemalloc

  X, single, chunked = _store_pair(tmp_path, 1200, 64)
  rows = np.sort(np.random.default_rng(0).choice(1200, 150, replace=False))

  for store in (single, chunked):
    assert np.array_equal(store.take_rows(rows), X[rows])
    assert np.array_equal(store.take_rows(np.arange(1200)), X)  # full range still reads everything
    assert store.take_rows([]).shape == (0, 64)

  # A row past the end is a mismatch between the run and its matrix, not something to read around.
  with pytest.raises(IndexError):
    chunked.take_rows(np.array([5, 1199, 1200]))

    tracemalloc.start()
    store.take_rows(rows)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < X.nbytes / 2, f"{type(store).__name__} peaked at {peak / X.nbytes:.2f}x"


def test_the_proxy_scores_every_row_until_it_cannot():
  """Below the cap nothing changes -- same rows, same folds, same targets, so the same ranking.
  Above it the proxy keeps a stratified sample, because holding the matrix per descriptor is not
  available at that size and a shallow forest's ranking has long since settled."""
  from zairachem.screen.proxy import PROXY_MAX_ROWS, _compact_folds

  n = 3000
  y = np.linspace(4.0, 9.0, n)
  idx = np.arange(n)
  folds = [(np.setdiff1d(idx, idx[k::5]), idx[k::5]) for k in range(5)]

  rows, compact, y_rows = _compact_folds(folds, y, "regression", PROXY_MAX_ROWS, 42)
  assert np.array_equal(rows, idx) and np.array_equal(y_rows, y)
  for (tr, te), (ctr, cte) in zip(folds, compact):
    assert np.array_equal(tr, ctr) and np.array_equal(te, cte)

  rows, compact, y_rows = _compact_folds(folds, y, "regression", 500, 42)
  assert rows.size == 500
  assert np.array_equal(y_rows, y[rows])
  for ctr, cte in compact:  # renumbered into the sample, not left pointing at the full run
    assert ctr.max() < rows.size and cte.max() < rows.size
  assert y_rows.min() < np.percentile(y, 10) and y_rows.max() > np.percentile(y, 90)

  # A class rare enough that a flat draw of 500 from 3,000 would usually miss it entirely has to
  # survive the thinning, or the proxy ranks descriptors on folds with nothing in them to find.
  rare = np.zeros(n, dtype=int)
  rare[[7, 1500, 2998]] = 1  # three positives in three thousand
  for seed in range(5):
    _, _, y_rare = _compact_folds(folds, rare, "classification", 500, seed)
    assert y_rare.sum() >= 1, f"the rare class was dropped at seed {seed}"

  # And the cap is a memory budget: it is what a descriptor's matrix costs while it is scored.
  widest = 2048  # features of a wide descriptor
  implied_mb = PROXY_MAX_ROWS * widest * 4 * 1.8 / 2**20  # the matrix, plus the fit's copy
  assert implied_mb < 512, f"the cap implies {implied_mb:.0f} MB per descriptor"


def test_a_run_says_it_will_not_fit_before_it_spends_the_time(tmp_path, monkeypatch):
  """Treat doubles the descriptor footprint, and running out of disk surfaces as whatever error
  h5py raises on a short write -- hours in, with the run unfinishable. The raw matrices are
  already there, so what is about to be written is measured, not guessed."""
  import inspect

  from zairachem.base.utils import disk
  from zairachem.base.utils.disk import free_bytes, path_bytes, warn_if_short
  from zairachem.treat.imputers.treated import TreatedDescriptors

  tree = tmp_path / "descriptors" / "eosA"
  tree.mkdir(parents=True)
  (tree / "raw.h5").write_bytes(b"x" * 4096)
  (tree / "more").mkdir()
  (tree / "more" / "chunk.h5").write_bytes(b"y" * 2048)
  assert path_bytes(str(tree)) == 6144
  assert path_bytes(str(tree / "raw.h5")) == 4096
  assert path_bytes(str(tmp_path / "absent")) == 0

  # A path that does not exist yet is still on a filesystem: measure its nearest ancestor.
  assert free_bytes(str(tmp_path / "not" / "created" / "yet")) == free_bytes(str(tmp_path))

  said = []
  monkeypatch.setattr(disk.logger, "warning", lambda m, *a, **k: said.append(str(m)))
  assert warn_if_short(str(tmp_path), 1024, "scaling 1 descriptor") is False
  assert said == []
  assert warn_if_short(str(tmp_path), 900 * 2**40, "scaling 5 descriptors") is True
  assert "900.0 TB" in said[0] and "free" in said[0]

  # An unmeasurable filesystem must not stop a run that is otherwise fine.
  monkeypatch.setattr(disk, "free_bytes", lambda _p: None)
  assert warn_if_short(str(tmp_path), 900 * 2**40, "scaling 5 descriptors") is False

  # And treat has to ask, before the step that takes the hours.
  assert "_warn_if_disk_is_short" in inspect.getsource(TreatedDescriptors.run)
