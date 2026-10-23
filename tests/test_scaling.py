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


def test_the_screen_thins_the_run_before_it_builds_the_folds():
  """Thinning inside select_descriptors bounded the matrices but still paid for a fold menu over
  every molecule: a Bemis-Murcko scaffold each, and a pass over every label per candidate seed.
  Neither is worth spending on rows that will not be scored."""
  import inspect

  from zairachem.screen import pipe
  from zairachem.screen.proxy import PROXY_MAX_ROWS, screening_rows

  assert np.array_equal(
    screening_rows(np.linspace(0, 1, 3000), "regression", PROXY_MAX_ROWS, 0), np.arange(3000)
  )  # at or below the cap, every row, so the folds are the ones it always built

  rows = screening_rows(np.linspace(0, 1, 80_000), "regression", 25_000, 0)
  assert rows.size == 25_000
  assert np.all(np.diff(rows) > 0) and np.unique(rows).size == rows.size  # sorted and distinct

  rare = (np.arange(80_000) % 50 == 0).astype(float)  # 2% positives
  kept = screening_rows(rare, "classification", 25_000, 0)
  assert abs(rare[kept].mean() - rare.mean()) < 0.005

  # The ordering is the point: thinned, then the menu built from the thinned rows. Match the
  # call, not the name -- the import of it sits at the top of the same function.
  src = inspect.getsource(pipe.ScreenPipeline.run)
  assert "rows = screening_rows(" in src, src
  assert src.index("rows = screening_rows(") < src.index("build_fold_definitions("), src
  # ...and the rows it kept are handed on, or they would be thinned a second time.
  assert "rows=rows" in src, src


def test_a_caller_that_already_thinned_is_not_thinned_again(monkeypatch):
  """The screen passes the rows it kept and fold indices that are positions within them. Thinning
  those a second time would renumber indices that are already compact."""
  from zairachem.screen import proxy

  y = np.linspace(4.0, 9.0, 40)  # 40 kept rows, drawn from a much larger run
  folds = [(np.arange(30), np.arange(30, 40))]
  rows = np.sort(np.random.default_rng(0).choice(5000, 40, replace=False))

  seen = {}

  def load(_dir, eos_id, asked):
    seen[eos_id] = np.asarray(asked)
    return np.zeros((len(asked), 2), dtype="float32")

  monkeypatch.setattr(proxy, "_load_rows", load)
  monkeypatch.setattr(proxy, "_fold_pred", lambda _X, yy, _tr, te, _s, _t: yy[np.asarray(te)])
  monkeypatch.setattr(
    proxy, "_compact_folds", lambda *a, **k: pytest.fail("the rows were thinned twice")
  )

  selected, _scores, _gains = proxy.select_descriptors(
    "d", ["eosA"], y, folds, 1, task="regression", rows=rows
  )
  assert selected == ["eosA"]
  assert np.array_equal(seen["eosA"], rows)  # exactly the rows it was given, read once


def test_a_rerun_store_discards_the_previous_attempt(tmp_path):
  """A step that re-runs after an interruption creates its store again. The metadata starts at
  zero either way, so the old chunks are never read -- but they are only overwritten as far as
  the new attempt reaches, and a descriptor interrupted late in a million rows would leave
  gigabytes of unreachable data on the disk the rest of the run still needs."""
  import os

  from zairachem.base.utils.matrices import ChunkedH5Store

  store = ChunkedH5Store(str(tmp_path / "raw.h5"))
  store.create(4, [f"f{i}" for i in range(4)])
  for s in range(6):  # a first attempt that got six chunks in before it stopped
    store.save_chunk(np.full((10, 4), s, dtype=np.float32), [f"s{s}_{i}" for i in range(10)])
  assert store.n_rows() == 60
  left_behind = sorted(n for n in os.listdir(store.dir) if n.endswith(".h5"))
  assert len(left_behind) == 6

  # Something else in the folder is not this method's to delete.
  (tmp_path / "raw_chunks" / "notes.txt").write_text("keep me")

  store.create(4, [f"f{i}" for i in range(4)])  # the step runs again
  assert store.n_rows() == 0
  assert [n for n in os.listdir(store.dir) if n.endswith(".h5")] == []
  assert (tmp_path / "raw_chunks" / "notes.txt").read_text() == "keep me"

  store.save_chunk(np.zeros((10, 4), dtype=np.float32), [f"s{i}" for i in range(10)])
  assert store.n_rows() == 10 and store.values().shape == (10, 4)


def test_the_run_records_its_own_memory_not_just_the_machine_s(tmp_path):
  """The host RAM already recorded is the whole machine, most of which is the model servers in
  their own containers: it says a run was tight, not which part of ZairaChem was holding
  anything. The pipeline's own high-water mark only rises, so the jump at a step is what that
  step added -- which is the step to look at when a large run runs out of memory."""
  import json

  from zairachem.base.utils.pipeline import _peak_rss_gb
  from zairachem.report.perf import step_telemetry

  # A plausible figure, not just a positive one: ru_maxrss is bytes on macOS and kilobytes on
  # Linux, and reading it with the wrong scale gives a number a thousand times out.
  mine = _peak_rss_gb()
  assert mine is None or 0.01 < mine < 100, mine

  # A run with no session at all still answers, rather than missing the new keys.
  assert step_telemetry(str(tmp_path / "nothing-here"))["peak_rss_gb"] is None

  def telemetry(log):
    (tmp_path / "session.json").write_text(
      json.dumps({"steps": [r["name"] for r in log], "step_log": log, "time_stamp": -10})
    )
    return step_telemetry(str(tmp_path))

  base = {"cpu": 10, "ram_used_gb": 9.0, "ram_total_gb": 17.0, "ram_pct": 53}
  log = [
    {"name": "setup", "t": 0, "rss_peak_gb": 0.4, **base},
    {"name": "raw_descriptions", "t": 60, "rss_peak_gb": 0.6, **base},
    {"name": "estimate", "t": 300, "rss_peak_gb": 5.9, **base},
    {"name": "report", "t": 310, "rss_peak_gb": 5.9, **base},
  ]
  t = telemetry(log)
  assert t["peak_rss_gb"] == 5.9
  assert t["rss_step"] == "Estimate" and abs(t["rss_step_gb"] - 5.3) < 1e-6
  held = {s["label"]: s["rss_added_gb"] for s in t["steps"]}
  assert abs(held["Descriptors"] - 0.2) < 1e-6  # a step that held little is not accused

  # The step records it, and the report shows it beside the host RAM.
  import inspect

  from zairachem.base.utils.pipeline import PipelineStep
  from zairachem.report.html.sections import _perf_steps_html

  assert "rss_peak_gb" in inspect.getsource(PipelineStep._telemetry)
  html = _perf_steps_html(t["steps"])
  assert "5.3 GB held" in html, html

  # A run recorded before this existed has no such figures, and still summarises.
  older = [{k: v for k, v in r.items() if k != "rss_peak_gb"} for r in log]
  t = telemetry(older)
  assert t["peak_rss_gb"] is None and t["rss_step"] is None
  assert all(s["rss_added_gb"] is None for s in t["steps"])


def test_request_size_follows_the_worker_grant():
  """The server cuts a request of SERVER_CHUNKING_MOLECULES or more into chunks and runs them on
  its workers. With several that is what makes a large request fast; with one it only serialises
  the chunks and pays their overhead. Measured on a live eos3l5f server in the binary format it
  uses: 9,999 molecules on one worker ran at 635 mol/s and 10,000 at 601, while 10,000 on four
  ran at 1,101 -- and four workers below the line bought nothing (8,192 on four: 436). So the two
  are one decision."""
  import inspect

  from zairachem.base.vars import DESCRIBE_BATCH_SIZE, SERVER_CHUNKING_MOLECULES
  from zairachem.describe.descriptors import raw
  from zairachem.describe.descriptors.api import BinaryStreamClient

  def sized(batch_size, workers):
    client = BinaryStreamClient.__new__(BinaryStreamClient)
    client.batch_size = batch_size
    client._initial_batch_size = batch_size
    client.model_id = "eosTEST"
    client.set_request_size_for_workers(workers)
    return client.batch_size

  # One worker: just under the line, never on it.
  assert sized(DESCRIBE_BATCH_SIZE, 1) == SERVER_CHUNKING_MOLECULES - 1
  assert sized(SERVER_CHUNKING_MOLECULES * 5, 1) == SERVER_CHUNKING_MOLECULES - 1
  # More than one: on the line, where the chunking is worth having.
  assert sized(DESCRIBE_BATCH_SIZE, 4) == DESCRIBE_BATCH_SIZE
  # A request already below the line is left alone, whatever the grant.
  assert sized(2000, 1) == 2000 and sized(2000, 4) == 2000
  assert sized(SERVER_CHUNKING_MOLECULES - 1, 1) == SERVER_CHUNKING_MOLECULES - 1

  # And the grant is where describe decides it, so the two cannot drift apart.
  src = inspect.getsource(raw.RawDescriptors._run_eos)
  assert "set_request_size_for_workers" in src, src


def test_a_refused_request_size_is_remembered_per_image(tmp_path, monkeypatch):
  """Some images cannot serve a request past a size of their own: eos4ex3 refuses 8,192 molecules
  after about 28 seconds and serves 6,144. Without a memory of it every run spends its first
  request discovering the same ceiling. Opening smaller is close to free -- throughput is flat
  across these sizes and only the per-request overhead grows -- so the trade is one-sided."""
  import inspect

  from zairachem.describe.descriptors import profiles
  from zairachem.describe.descriptors.api import BinaryStreamClient
  from zairachem.describe.descriptors.profiles import ModelProfile

  p = ModelProfile("eos4ex3", "sha256:abc")
  assert p.first_request(9999) == 9999  # nothing known, nothing changed
  p.record_failure(8192)
  assert p.failed_at == 8192 and p.first_request(9999) == 4096  # halved below the refusal
  p.record_failure(9999)
  assert p.failed_at == 8192  # a larger refusal says nothing new
  p.record_failure(2048)
  assert p.failed_at == 2048 and p.first_request(9999) == 1024  # a smaller one does
  assert ModelProfile("x", failed_at=1).first_request(9999) == 1  # never below one molecule

  # It survives a round trip, keyed to the image like the rest of the profile.
  monkeypatch.setattr(profiles, "BASE_DIR", str(tmp_path))
  monkeypatch.setattr(profiles, "_path", lambda: str(tmp_path / "model_profiles.json"))
  profiles.save(p)
  assert profiles.load("eos4ex3", "sha256:abc").failed_at == 2048
  assert profiles.load("eos4ex3", "sha256:different").failed_at is None  # a new image learns again

  # The client asks before its first request, and records a refusal when it meets one.
  client = BinaryStreamClient.__new__(BinaryStreamClient)
  client.batch_size = 9999
  client._initial_batch_size = 9999
  client.model_id = "eos4ex3"
  client.profile = ModelProfile("eos4ex3", "sha256:abc", failed_at=8192)
  client.open_below_known_refusals()
  assert client.batch_size == 4096 and client._initial_batch_size == 4096
  assert "record_failure" in inspect.getsource(BinaryStreamClient._fetch_batch)

  # ...and describe asks it to, or the memory is written and never read.
  from zairachem.describe.descriptors import raw

  assert "open_below_known_refusals" in inspect.getsource(raw.RawDescriptors._run_eos)


def test_a_prediction_can_skip_the_projection_it_only_plots(tmp_path):
  """The projection model feeds one report figure and is routinely the slowest thing in a
  prediction: measured across every predict run, it was the longest model every time (45.4s of a
  47.5s describe on 500 molecules). Describe runs models side by side, so dropping it moves the
  critical path to the slowest real descriptor -- 47.5s to 30.2s on that run, with the
  predictions byte-identical. A fit could already say `-p none`; a prediction could not."""
  import inspect
  import json

  from zairachem.cli.commands import predict as predict_cmd
  from zairachem.setup.prep.prediction import PredictSetup

  meta = tmp_path / "metadata"
  meta.mkdir()

  def updated(projection_ids):
    (meta / "parameters.json").write_text(
      json.dumps({"projection_ids": ["eos1klk"], "task": "regression"})
    )
    setup = PredictSetup.__new__(PredictSetup)
    setup.output_dir = str(tmp_path)
    setup.store = setup.read_store = setup.contribute_store = None
    setup.projection_ids = projection_ids
    setup._update_params()
    return json.loads((meta / "parameters.json").read_text())["projection_ids"]

  assert updated(None) == ["eos1klk"]  # silence inherits the model's own choice
  for off in ("none", "NONE", "off", ""):
    assert updated(off) == [], off
  assert updated("eos2mvc") == ["eos2mvc"]  # and a different projection can be asked for

  # The flag has to reach there, or none of the above is usable from the command line.
  src = inspect.getsource(predict_cmd)
  assert "--projection-ids" in src  # the flag exists
  assert "projection_ids=projection_ids" in src  # ...and is passed on, not accepted and dropped
