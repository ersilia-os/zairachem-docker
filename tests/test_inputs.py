"""An input that cannot be used is refused or reported, and nothing is lost or invented on the way."""

import json
from concurrent.futures import Future

import pandas as pd
import pytest

from zairachem.base.vars import DATA_SUBFOLDER


def _single_file(tmp_path, rows, task="classification", mode="fit", columns=None):
  """A ``SingleFile`` over ``rows`` in a run folder whose session says ``mode``."""
  from zairachem.setup.prep.files import SingleFile

  run = tmp_path / "run"
  (run / DATA_SUBFOLDER).mkdir(parents=True)  # makes tmp_path too
  (run / "session.json").write_text(
    json.dumps({"mode": mode, "output_dir": str(run), "model_dir": str(run)})
  )
  source = tmp_path / "in.csv"
  pd.DataFrame(rows, columns=columns).to_csv(source, index=False)
  return SingleFile(str(source), {"task": task}, str(run)), run


def test_a_compound_with_no_label_is_refused_not_made_inactive(tmp_path, capsys):
  """Blank labels used to become class 0: the median of nothing is NaN, and NaN >= 0.5 is False."""
  rows = [
    {"smiles": "CCO", "dili": 1},
    {"smiles": "CCN", "dili": 0},
    {"smiles": "CCC", "dili": None},  # never labelled
    {"smiles": "CCCC", "dili": None},  # never labelled
    {"smiles": "CCCCC", "dili": 1},
    {"smiles": "CCCCC", "dili": None},  # a blank next to a real label is not a problem
  ]
  single, run = _single_file(tmp_path, rows)
  with pytest.raises(SystemExit) as stop:
    single.process()
  assert stop.value.code == 1
  screen = " ".join(capsys.readouterr().out.split())
  assert "2 compounds have no label" in screen and "CCC, CCCC" in screen
  assert "remove those rows or fill in their labels" in screen

  # Labelled everywhere it matters: the blank beside a real label keeps the real label.
  single, run = _single_file(tmp_path / "ok", [r for r in rows if r["smiles"] in ("CCO", "CCCCC")])
  single.process()
  values = pd.read_csv(run / DATA_SUBFOLDER / "values.csv")
  assert sorted(values["value"].tolist()) == [1, 1]


def test_blank_and_unparseable_smiles_are_dropped_and_counted(tmp_path, capsys):
  """A blank SMILES cell is NaN, which is truthy, so it reached RDKit and crashed setup; the ones
  that did not crash were dropped with nothing on screen."""
  rows = [
    {"smiles": "CCO", "dili": 1},
    {"smiles": None, "dili": 0},
    {"smiles": "not_a_smiles", "dili": 1},
    {"smiles": "CCN", "dili": 0},
  ]
  single, run = _single_file(tmp_path, rows)
  single.process()
  compounds = pd.read_csv(run / DATA_SUBFOLDER / "compounds.csv")
  assert compounds["smiles"].tolist() == ["CCO", "CCN"]
  screen = " ".join(capsys.readouterr().out.split())
  assert "2 rows dropped" in screen
  assert "row 3 blank" in screen and "row 4 'not_a_smiles'" in screen  # spreadsheet rows


def test_an_input_with_no_data_rows_is_refused_in_one_line(tmp_path, capsys):
  """A header-only CSV ended in IndexError, which ``_resolve_task`` did not catch."""
  from zairachem.cli.options import _resolve_task

  single, _ = _single_file(tmp_path, [], columns=["smiles", "dili"])
  assert _resolve_task(None, single.input_file) == "classification"  # left to setup to explain
  with pytest.raises(SystemExit) as stop:
    single.process()
  assert stop.value.code == 1
  assert "has no data rows" in capsys.readouterr().out


def test_a_target_of_words_says_so_instead_of_a_key_error(tmp_path, capsys):
  """Labels written as words ended in ``KeyError: None``: no target column was found."""
  rows = [{"smiles": s, "dili": w} for s, w in [("CCO", "active"), ("CCN", "inactive")] * 5]
  single, _ = _single_file(tmp_path, rows)
  with pytest.raises(SystemExit):
    single.process()
  screen = " ".join(capsys.readouterr().out.split())
  assert "No numeric or 0/1 target column" in screen
  assert "dili (not numeric, e.g. 'active')" in screen


def test_ground_truth_that_is_not_binary_is_not_rounded_at_predict(tmp_path, capsys):
  """At predict every value >= 0.5 became 1, so a pIC50 column was scored as all positives."""
  rows = [{"smiles": s, "dili": v} for s, v in [("CCO", 5.2), ("CCN", 7.8), ("CCC", 6.1)]]
  single, _ = _single_file(tmp_path, rows, mode="predict")
  values = single.values_table(single.normalize_dataframe())
  assert sorted(values["value"].tolist()) == [5.2, 6.1, 7.8]
  assert "is ignored" in " ".join(capsys.readouterr().out.split())

  # 0/1 ground truth is still turned into labels (0.5 counts as positive).
  rows = [{"smiles": s, "dili": v} for s, v in [("CCO", 0), ("CCN", 1), ("CCC", 0.5)]]
  single, _ = _single_file(tmp_path / "binary", rows, mode="predict")
  values = single.values_table(single.normalize_dataframe())
  assert sorted(values["value"].tolist()) == [0, 1, 1]


class _LosesTheSecondBatch:
  """Stands in for the process pool: runs each batch here, but the second one is lost."""

  def __init__(self, *args, **kwargs):
    self.submitted = 0

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def submit(self, fn, *args):
    self.submitted += 1
    future = Future()
    if self.submitted == 2:
      future.set_exception(RuntimeError("worker killed"))
    else:
      future.set_result(fn(*args))
    return future


def test_a_lost_standardisation_batch_is_run_again_not_dropped(tmp_path, monkeypatch):
  from zairachem.base.vars import COMPOUND_IDENTIFIER_COLUMN, COMPOUNDS_FILENAME, SMILES_COLUMN
  from zairachem.setup.prep import standardize

  monkeypatch.setattr(standardize, "ProcessPoolExecutor", _LosesTheSecondBatch)
  monkeypatch.setattr(standardize, "_min_parallel", lambda: 1)
  smiles = ["CCO", "CCN", "CCC", "CCCC", "c1ccccc1"]
  pd.DataFrame({
    COMPOUND_IDENTIFIER_COLUMN: [f"CID{i}" for i in range(5)],
    SMILES_COLUMN: smiles,
  }).to_csv(tmp_path / COMPOUNDS_FILENAME, index=False)

  standardize.ChemblStandardize(str(tmp_path), batch_size=2).run()
  out = pd.read_csv(tmp_path / "compounds_std.csv")
  assert out[COMPOUND_IDENTIFIER_COLUMN].tolist() == [f"CID{i}" for i in range(5)]


def test_a_lost_validation_batch_is_run_again_not_dropped(tmp_path, monkeypatch):
  from zairachem.setup.prep import files

  monkeypatch.setattr(files, "ProcessPoolExecutor", _LosesTheSecondBatch)
  monkeypatch.setattr(files, "DEDUPE_BATCH_SIZE", 2)
  single, _ = _single_file(tmp_path, [{"smiles": "CCO", "dili": 1}])
  data = [(i, f"CID{i}", smi) for i, smi in enumerate(["CCO", "CCN", "bad", "CCC", "CCCC"])]

  validated = single._validate_smiles_parallel(data)
  assert [(i, ok) for i, _, _, ok in validated] == [
    (0, True),
    (1, True),
    (2, False),
    (3, True),
    (4, True),
  ]


def test_a_resumed_run_follows_the_corrected_input_not_its_old_copy(tmp_path, monkeypatch):
  """A fit that stopped at normalize_input on a blank label kept its first copy of the input: the
  corrected file went into values.csv, but raw_input.csv, which the report, the pool and the finish
  step read, was still the old file with the blank label."""
  from zairachem.setup.prep import PipelineStep
  from zairachem.setup.prep.training import TrainSetup

  for gate in ("require_docker_and_base", "warn_other_runs", "reap_orphaned_servers"):
    monkeypatch.setattr(f"zairachem.setup.prep.training.{gate}", lambda *a, **k: None)
  monkeypatch.setattr("zairachem.setup.prep.training.require_model_versions", lambda *a, **k: None)
  model = tmp_path / "m"
  raw = model / DATA_SUBFOLDER / "raw_input.csv"
  source = tmp_path / "in.csv"
  rows = [("CCO", 1), ("CCN", 0), ("CCC", None), ("CCCC", 1), ("CCCCC", 0), ("CCCCCC", 1)]

  def setup():
    # What run_fit does for an existing folder: resume it once initialize is done.
    ts = TrainSetup(str(source), str(model), model_ids=None, task="classification", store=None)
    ts.resume = PipelineStep("initialize", str(model)).is_done()
    ts.setup()

  pd.DataFrame(rows, columns=["smiles", "dili"]).to_csv(source, index=False)
  with pytest.raises(SystemExit) as stop:
    setup()
  assert stop.value.code == 1
  assert len(pd.read_csv(raw)) == 6 and pd.read_csv(raw)["dili"].isna().any()

  # The user removes the unlabelled row and re-runs into the same folder: that is a resume.
  pd.DataFrame([r for r in rows if r[1] is not None], columns=["smiles", "dili"]).to_csv(
    source, index=False
  )
  with pytest.raises(Exception, match="Not enough samples"):  # five compounds: too few to train
    setup()
  assert len(pd.read_csv(model / DATA_SUBFOLDER / "values.csv")) == 5
  corrected = pd.read_csv(raw)
  assert len(corrected) == 5 and not corrected["dili"].isna().any()
