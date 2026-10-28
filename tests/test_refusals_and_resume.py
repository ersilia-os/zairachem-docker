"""A refusal exits 1 with its one-line reason, and a resumed run is the run that was started."""

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from zairachem.base.vars import DATA_SUBFOLDER, METADATA_SUBFOLDER, RESULTS_SUBFOLDER
from zairachem.cli import cli


def _csv(path, n=20):
  """A classification input too small to train on (30 of each class are needed)."""
  pd.DataFrame({
    "smiles": ["C" * (i % 7 + 1) + "O" for i in range(n)],
    "y": [0, 1] * (n // 2),
  }).to_csv(path, index=False)
  return str(path)


@pytest.fixture
def no_docker(monkeypatch):
  """Let setup run without Docker: only the gates are replaced, the steps are the real ones."""
  for gate in ("require_docker_and_base", "warn_other_runs", "reap_orphaned_servers"):
    monkeypatch.setattr(f"zairachem.setup.prep.training.{gate}", lambda *a, **k: None)
  monkeypatch.setattr("zairachem.setup.prep.training.require_model_versions", lambda *a, **k: None)


@pytest.fixture
def summary_panels(monkeypatch):
  """Record every call of the green 'model ready' / 'predictions ready' panel."""
  calls = []
  monkeypatch.setattr(
    "zairachem.base.utils.progress.final_summary_panel", lambda *a, **k: calls.append(a)
  )
  return calls


def _fit(tmp_path, *args):
  return CliRunner().invoke(cli, ["fit", "-c", "-m", str(tmp_path / "m"), *args])


def _trained_model(model_dir, params=None):
  (model_dir / RESULTS_SUBFOLDER).mkdir(parents=True)
  (model_dir / RESULTS_SUBFOLDER / "output.csv").write_text("smiles\n")
  _unfinished(model_dir, params)


def _unfinished(model_dir, params=None, steps=("initialize",)):
  (model_dir / METADATA_SUBFOLDER).mkdir(parents=True, exist_ok=True)
  (model_dir / "session.json").write_text(json.dumps({"mode": "fit", "steps": list(steps)}))
  base = {"task": "classification", "featurizer_ids": ["eos1abc"], "projection_ids": []}
  (model_dir / METADATA_SUBFOLDER / "parameters.json").write_text(
    json.dumps({**base, **(params or {})})
  )


def test_a_refused_fit_is_not_a_success(tmp_path, summary_panels):
  """Fitting into a trained model without --override printed the refusal, then 'model ready',
  and exited 0, so a script could not tell a refused fit from a finished one."""
  _trained_model(tmp_path / "m")
  result = _fit(tmp_path, "-i", _csv(tmp_path / "in.csv"))
  assert result.exit_code == 1
  assert "already a trained model" in result.output
  assert summary_panels == []


def test_a_resume_with_a_conflicting_configuration_is_refused_not_succeeded(
  tmp_path, summary_panels
):
  _unfinished(tmp_path / "m")
  result = _fit(tmp_path, "-i", _csv(tmp_path / "in.csv"), "-f", "eos8aa5,eos3l5f")
  assert result.exit_code == 1
  assert "different configuration" in result.output
  assert summary_panels == []


@pytest.mark.parametrize("flag", [["--evaluate"], ["--evaluate", "scaffold"], ["--repeats", "5"]])
def test_evaluate_typed_on_a_resume_is_not_silently_ignored(tmp_path, summary_panels, flag):
  """A plain fit that was interrupted and re-run with --evaluate (or another --repeats) went on
  without the held-out validation, and without a word."""
  _unfinished(tmp_path / "m", {"evaluate": False, "evaluate_repeats": 3})
  result = _fit(tmp_path, "-i", _csv(tmp_path / "in.csv"), *flag)
  assert result.exit_code == 1
  assert "different configuration" in result.output
  assert summary_panels == []


def test_the_same_evaluate_settings_resume(tmp_path, no_docker):
  """The refusal is for a different setting; repeating the original one resumes."""
  _unfinished(
    tmp_path / "m", {"evaluate": True, "evaluate_repeats": 5, "evaluate_schemas": ["random"]}
  )
  result = _fit(tmp_path, "-i", _csv(tmp_path / "in.csv"), "--evaluate", "random", "--repeats", "5")
  assert "different configuration" not in result.output


def test_setup_checks_a_typed_featurizer_list_on_resume_like_fit(tmp_path):
  """`setup` never handed the typed flags to the resume check, so after a first run on -f eos0000
  that stopped, `-f eos8aa5,eos3l5f` said 'Resuming' and went on with eos0000."""
  _unfinished(tmp_path / "m", {"featurizer_ids": ["eos0000"]})
  result = CliRunner().invoke(
    cli,
    [
      "setup",
      "-i",
      _csv(tmp_path / "in.csv"),
      "-m",
      str(tmp_path / "m"),
      "-c",
      "-f",
      "eos8aa5,eos3l5f",
    ],
  )
  assert result.exit_code == 1
  assert "different configuration" in result.output
  assert "Resuming" not in result.output


def test_predict_into_existing_predictions_names_the_real_flag_and_fails(tmp_path, summary_panels):
  """It said 'Use --override-dir' (no such flag), then 'predictions ready', and exited 0."""
  model, out = tmp_path / "model", tmp_path / "out"
  (model / METADATA_SUBFOLDER).mkdir(parents=True)
  (model / METADATA_SUBFOLDER / "parameters.json").write_text("{}")
  (model / "descriptors").mkdir()
  (model / "descriptors" / "done_eos.json").write_text("[]")
  (model / "model" / "estimators").mkdir(parents=True)
  (model / RESULTS_SUBFOLDER).mkdir()
  (model / RESULTS_SUBFOLDER / "output.csv").write_text("smiles\n")
  (out / RESULTS_SUBFOLDER).mkdir(parents=True)
  (out / RESULTS_SUBFOLDER / "output.csv").write_text("smiles\n")

  result = CliRunner().invoke(
    cli, ["predict", "-i", _csv(tmp_path / "in.csv"), "-m", str(model), "-o", str(out)]
  )
  assert result.exit_code == 1
  assert "already exist" in result.output
  assert "--override-dir" not in result.output and "--override or" in result.output
  assert summary_panels == []


def test_an_expected_input_error_is_one_line_not_a_traceback(tmp_path, no_docker):
  """'Not enough samples per class' was a plain Exception: 'terminated with an unhandled
  error' and a 200-line traceback. A ClickException is 'Error: ...' and exit 1."""
  result = _fit(tmp_path, "-i", _csv(tmp_path / "in.csv"))
  assert result.exit_code == 1
  assert isinstance(result.exception, SystemExit)
  assert "Not enough samples per class" in result.output


def test_a_resume_must_be_on_the_input_the_run_was_started_on(tmp_path, no_docker):
  """A killed fit re-run into the same -m with another -i said 'Resuming' and trained on the old
  data, with the new file's name in the summary."""
  first, other = _csv(tmp_path / "a.csv"), _csv(tmp_path / "b.csv", 22)
  assert "Not enough samples" in _fit(tmp_path, "-i", first).output  # leaves a started run
  recorded = json.loads((tmp_path / "m" / "session.json").read_text())["input"]
  assert recorded["path"] == first and len(recorded["sha256"]) == 64

  refused = _fit(tmp_path, "-i", other)
  assert refused.exit_code == 1
  assert "Cannot resume" in refused.output and "--override" in refused.output
  assert "Resuming" not in refused.output
  assert (tmp_path / "m" / DATA_SUBFOLDER / "raw_input.csv").read_text() == open(first).read()

  # The same content from another path is the same input; an edited file is not.
  copy = tmp_path / "copy.csv"
  copy.write_text(open(first).read())
  assert "Resuming" in _fit(tmp_path, "-i", str(copy)).output
  with open(first, "a") as f:
    f.write("CCCCCCCCCCCCCCCC,1\n")
  edited = _fit(tmp_path, "-i", first)
  assert edited.exit_code == 1 and "changed since" in edited.output


def test_a_run_from_before_the_input_was_recorded_still_resumes(tmp_path, no_docker):
  first, other = _csv(tmp_path / "a.csv"), _csv(tmp_path / "b.csv", 22)
  _fit(tmp_path, "-i", first)
  session = tmp_path / "m" / "session.json"
  data = json.loads(session.read_text())
  del data["input"]
  session.write_text(json.dumps(data))
  assert "Resuming" in _fit(tmp_path, "-i", other).output


def _setup(tmp_path, *args):
  return CliRunner().invoke(cli, ["setup", "-c", "-m", str(tmp_path / "m"), *args])


def test_setup_refuses_a_different_input_on_resume_and_says_what_to_pass(tmp_path, no_docker):
  """Every refusal says 'Pass --override', which `setup` did not have: after a first `setup` that
  stopped, a corrected -i could not be used at all."""
  first, other = _csv(tmp_path / "a.csv"), _csv(tmp_path / "b.csv", 22)
  assert "Not enough samples" in _setup(tmp_path, "-i", first).output  # leaves a started run
  refused = _setup(tmp_path, "-i", other)
  assert refused.exit_code == 1
  assert "Cannot resume" in refused.output and "--override" in refused.output
  assert "Resuming" not in refused.output
  assert (tmp_path / "m" / DATA_SUBFOLDER / "raw_input.csv").read_text() == open(first).read()
  # The flag the message names is one `setup` really has.
  assert "--override" in CliRunner().invoke(cli, ["setup", "--help"]).output


def test_setup_override_rebuilds_the_run_from_the_new_input(tmp_path, no_docker):
  first, other = _csv(tmp_path / "a.csv"), _csv(tmp_path / "b.csv", 22)
  _setup(tmp_path, "-i", first)
  result = _setup(tmp_path, "-i", other, "--override")
  assert "No such option" not in result.output and "Cannot resume" not in result.output
  assert "Resuming" not in result.output
  assert (tmp_path / "m" / DATA_SUBFOLDER / "raw_input.csv").read_text() == open(other).read()
  recorded = json.loads((tmp_path / "m" / "session.json").read_text())["input"]
  assert recorded["path"] == other
