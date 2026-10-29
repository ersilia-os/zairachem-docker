"""A run must validate its arguments before it deletes anything.

``fit --override`` and ``predict --override`` wipe a directory. They used to do it before the input
file was opened and before the model was checked, so a mistyped ``-i`` or ``-m``, or ``-o`` pointing
at the model, cost the user a trained model or a set of predictions. ``fit`` also wiped any existing
folder it was given as ``-m``, ``--override`` or not.

Nothing here starts Docker: the setup gates are replaced by no-ops and the first real step raises
``_Reached``, so a run that gets past validation stops there, and the tests look at what is left on
disk.
"""

import json
import os

import pytest
from click.testing import CliRunner

from zairachem.cli import cli
from zairachem.setup.prep import prediction, training


class _Reached(Exception):
  """Raised where a run that passed validation would start real work."""


@pytest.fixture(autouse=True)
def _no_docker(monkeypatch):
  for name in (
    "require_docker_and_base",
    "warn_other_runs",
    "reap_orphaned_servers",
    "require_model_versions",
  ):
    monkeypatch.setattr(training, name, lambda *a, **k: None)

  def reached(self, *a, **k):
    raise _Reached

  monkeypatch.setattr(training.TrainSetup, "_normalize_input", reached)
  monkeypatch.setattr(prediction.PredictSetup, "setup", reached)


def snapshot(folder):
  """Relative path -> bytes of everything under ``folder`` (directories as None)."""
  out = {}
  for root, dirs, files in os.walk(folder):
    for name in dirs:
      out[os.path.relpath(os.path.join(root, name), folder)] = None
    for name in files:
      path = os.path.join(root, name)
      with open(path, "rb") as f:
        out[os.path.relpath(path, folder)] = f.read()
  return out


def zairachem_dir(path):
  """A folder ZairaChem would recognise as its own (it holds a ``session.json``), plus a sentinel."""
  (path / "results").mkdir(parents=True)
  (path / "session.json").write_text(json.dumps({"mode": "fit"}))
  (path / "results" / "sentinel.txt").write_text("keep me")
  return path


def trained_model(path):
  """A folder that passes the model-ready check of predict, with a sentinel."""
  zairachem_dir(path)
  for rel in ("metadata", "descriptors", "model/estimators"):
    (path / rel).mkdir(parents=True, exist_ok=True)
  (path / "metadata" / "parameters.json").write_text("{}")
  (path / "descriptors" / "done_eos.json").write_text("[]")
  (path / "results" / "output.csv").write_text("smiles\n")
  return path


def molecules(path):
  path.write_text("smiles,activity\nCCO,1\nCCN,0\n")
  return path


def run(args):
  result = CliRunner().invoke(cli, args)
  assert not isinstance(result.exception, _Reached), result.output
  return result


def test_fit_override_with_a_missing_input_leaves_the_model(tmp_path):
  model = zairachem_dir(tmp_path / "model")
  before = snapshot(model)

  result = run(["fit", "--override", "-i", str(tmp_path / "typo.csv"), "-m", str(model)])

  assert result.exit_code != 0
  assert "typo.csv" in result.output
  assert snapshot(model) == before


def test_setup_class_refuses_a_missing_input_before_touching_the_model(tmp_path):
  """The CLI checks ``-i`` itself; the class must too, for API callers."""
  model = zairachem_dir(tmp_path / "model")
  before = snapshot(model)

  with pytest.raises(SystemExit) as stop:
    training.TrainSetup(
      str(tmp_path / "typo.csv"), str(model), "classification", model_ids=None, store=None
    )

  assert stop.value.code == 1
  assert snapshot(model) == before


@pytest.mark.parametrize("override", [False, True])
def test_fit_does_not_wipe_a_folder_that_is_not_a_zairachem_model(tmp_path, override):
  mine = tmp_path / "mine"
  (mine / "sub").mkdir(parents=True)
  (mine / "keep.txt").write_text("my own data")
  (mine / "sub" / "a.csv").write_text("1,2\n")
  before = snapshot(mine)
  args = ["fit", "-i", str(molecules(tmp_path / "x.csv")), "-m", str(mine)]

  result = run([*args, "--override"] if override else args)

  assert result.exit_code == 1
  assert "is not a ZairaChem directory" in " ".join(result.output.split())
  assert snapshot(mine) == before


def test_fit_may_use_an_existing_empty_folder(tmp_path):
  empty = tmp_path / "empty"
  empty.mkdir()
  data = molecules(tmp_path / "x.csv")

  ts = training.TrainSetup(str(data), str(empty), "classification", model_ids=None, store=None)

  with pytest.raises(_Reached):
    ts.setup()


def test_predict_override_with_a_missing_model_leaves_the_predictions(tmp_path):
  out = zairachem_dir(tmp_path / "out")
  before = snapshot(out)
  data = molecules(tmp_path / "x.csv")

  result = run([
    "predict",
    "--override",
    "-i",
    str(data),
    "-m",
    str(tmp_path / "no_model"),
    "-o",
    str(out),
  ])

  assert result.exit_code == 1
  assert snapshot(out) == before


def test_predict_override_with_a_missing_input_leaves_the_predictions(tmp_path):
  out = zairachem_dir(tmp_path / "out")
  model = trained_model(tmp_path / "model")
  before = snapshot(out)

  result = run([
    "predict",
    "--override",
    "-i",
    str(tmp_path / "typo.csv"),
    "-m",
    str(model),
    "-o",
    str(out),
  ])

  assert result.exit_code != 0
  assert snapshot(out) == before


def test_predict_override_does_not_wipe_a_folder_that_is_not_a_zairachem_directory(tmp_path):
  mine = tmp_path / "mine"
  mine.mkdir()
  (mine / "keep.txt").write_text("my own data")
  model = trained_model(tmp_path / "model")
  before = snapshot(mine)

  result = run([
    *("predict", "--override", "-i", str(molecules(tmp_path / "x.csv"))),
    *("-m", str(model), "-o", str(mine)),
  ])

  assert result.exit_code == 1
  assert "is not a ZairaChem directory" in " ".join(result.output.split())
  assert snapshot(mine) == before


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("layout", ["same", "output_inside_model", "model_inside_output"])
def test_predict_refuses_an_output_that_overlaps_the_model(tmp_path, layout, override):
  if layout == "model_inside_output":
    out = zairachem_dir(tmp_path / "out")
    model = trained_model(out / "model")
  else:
    model = trained_model(tmp_path / "model")
    out = model if layout == "same" else model / "predictions"
  args = ["predict", "-i", str(molecules(tmp_path / "x.csv")), "-m", str(model), "-o", str(out)]
  before = snapshot(tmp_path)

  result = run([*args, "--override"] if override else args)

  assert result.exit_code == 1
  assert "model directory" in " ".join(result.output.split())
  assert snapshot(tmp_path) == before
