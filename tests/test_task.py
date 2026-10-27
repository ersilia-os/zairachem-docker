"""The task, classification or regression, is read from the target unless it is typed."""

import json

import pandas as pd
import pytest
import rich_click as click
from click.testing import CliRunner

from zairachem.cli import cli
from zairachem.cli.options import _resolve_task


def _csv(tmp_path, values, name="in.csv"):
  path = tmp_path / name
  pd.DataFrame({"smiles": ["C" * (i % 7 + 1) for i in range(len(values))], "y": values}).to_csv(
    path, index=False
  )
  return str(path)


def test_the_task_is_read_from_the_target_when_no_flag_is_typed(tmp_path):
  assert _resolve_task(None, _csv(tmp_path, [0.0, 1.0] * 10)) == "classification"
  assert _resolve_task(None, _csv(tmp_path, [0.0, 0.5, 1.0] * 10)) == "classification"
  assert _resolve_task(None, _csv(tmp_path, [-1.74, -3.2, 0.02, -5.1])) == "regression"


def test_a_continuous_target_is_never_classified(tmp_path):
  """Typed -c on a continuous target used to round it to 0/1 at 0.5: 8 ones and 372 zeros."""
  continuous = _csv(tmp_path, [0.02, -1.74, -3.2, -5.1, 1.58] * 20)
  with pytest.raises(click.UsageError, match="continuous.*--regression"):
    _resolve_task(True, continuous)


def test_two_values_that_are_not_zero_and_one_are_not_guessed(tmp_path):
  labels = _csv(tmp_path, [1, 2] * 10)
  for flag in (None, True):
    with pytest.raises(click.UsageError, match=r"only two values \(1, 2\)"):
      _resolve_task(flag, labels)
  assert _resolve_task(False, labels) == "regression"


def test_regression_can_be_forced_on_a_binary_target(tmp_path):
  assert _resolve_task(False, _csv(tmp_path, [0.0, 1.0] * 10)) == "regression"


def test_the_whole_target_is_read_not_only_the_first_rows(tmp_path):
  values = [0.0, 1.0] * 750 + [0.37]  # binary for 1,500 rows, then a real number
  assert _resolve_task(None, _csv(tmp_path, values)) == "regression"


def test_a_continuous_target_is_not_rounded_at_fit(tmp_path):
  from zairachem.base.vars import DATA_SUBFOLDER
  from zairachem.setup.prep.files import SingleFile

  run = tmp_path / "run"
  (run / DATA_SUBFOLDER).mkdir(parents=True)
  (run / "session.json").write_text(
    json.dumps({"mode": "fit", "output_dir": str(run), "model_dir": str(run)})
  )
  source = _csv(tmp_path, [0.02, -1.74, -3.2, -5.1, 1.58])
  with pytest.raises(Exception, match="continuous"):
    SingleFile(source, {"task": "classification"}, str(run)).process()
  # Real labels still go through, a repeated compound included (its 0 and 1 give a 0.5, rounded to 1).
  labels = tmp_path / "labels.csv"
  pd.DataFrame({"smiles": ["CCO", "CCO", "CCN"], "y": [0.0, 1.0, 0.0]}).to_csv(labels, index=False)
  SingleFile(str(labels), {"task": "classification"}, str(run)).process()
  assert sorted(pd.read_csv(run / DATA_SUBFOLDER / "values.csv").iloc[:, 1]) == [0, 1]


def test_fit_hands_the_detected_task_to_setup(tmp_path, monkeypatch):
  seen = []
  monkeypatch.setattr(
    "zairachem.setup.run_fit.run", lambda input_file, task, *a, **k: seen.append(task) or False
  )
  regression, classification = (
    _csv(tmp_path, [-1.5, 0.3, -2.2, -4.0] * 5, "r.csv"),
    _csv(tmp_path, [0.0, 1.0] * 10, "c.csv"),
  )
  runner = CliRunner()
  for args in (["-i", regression], ["-i", classification], ["-i", classification, "-r"]):
    runner.invoke(cli, ["fit", *args, "-m", str(tmp_path / "m")])
  assert seen == ["regression", "classification", "regression"]
