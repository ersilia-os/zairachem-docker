"""Smoke tests for the documented CLI entry points."""

import subprocess
import sys

import pytest
from click.testing import CliRunner

from zairachem.cli import cli

COMMANDS = ["fit", "predict", "setup", "describe", "treat", "estimate", "pool", "report", "finish"]


def test_top_level_help():
  result = CliRunner().invoke(cli, ["--help"])
  assert result.exit_code == 0
  for command in COMMANDS:
    assert command in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(command):
  result = CliRunner().invoke(cli, [command, "--help"])
  assert result.exit_code == 0


def test_import_keeps_heavy_stacks_lazy():
  """Importing the CLI must not load the modelling/plotting stacks (kept lazy for fast start-up)."""
  code = (
    "import sys, zairachem.cli;"
    "heavy = [m for m in ('sklearn', 'scipy', 'matplotlib', 'rdkit', 'lazyqsar') if m in sys.modules];"
    "sys.exit(1 if heavy else 0)"
  )
  assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_the_run_keeps_its_input_as_csv_whatever_the_file_was_called(tmp_path):
  """The input is read with ``read_csv`` whatever its name, so a comma-separated ``.txt`` gets in.

  Copying it under its own extension left the report and the finish step — which both look for
  ``raw_input.csv`` — to fail at the end of a run that had already trained every descriptor.
  """
  import os

  from zairachem.base.vars import DATA_SUBFOLDER
  from zairachem.setup.prep.base import BaseSetup

  for name in ("molecules.csv", "molecules.txt", "molecules"):
    case = tmp_path / name.replace(".", "_")
    run, source = case / "run", case / name
    (run / DATA_SUBFOLDER).mkdir(parents=True)
    source.write_text("smiles,activity\nCCO,1.0\n")

    setup = BaseSetup.__new__(BaseSetup)
    setup.input_file, setup.output_dir = str(source), str(run)
    setup._copy_input_file()

    assert os.listdir(run / DATA_SUBFOLDER) == ["raw_input.csv"], name


def test_a_blank_cell_does_not_throw_away_a_compound_s_other_measurement(tmp_path):
  """Real exports leave cells empty. The median across a blank used to make the compound missing,
  and the next step then rejected the whole run over a compound that had been measured."""
  import json

  import pandas as pd

  from zairachem.base.vars import DATA_SUBFOLDER
  from zairachem.setup.prep.files import SingleFile, read_replicates

  rows = [
    {"smiles": "CCO", "activity": 5.0},
    {"smiles": "CCO", "activity": None},  # the same compound, left blank the second time
    {"smiles": "CCN", "activity": 6.0},
    {"smiles": "CCN", "activity": 6.4},
    {"smiles": "CCC", "activity": 7.0},
  ]
  run = tmp_path / "run"
  (run / DATA_SUBFOLDER).mkdir(parents=True)
  source = run / "in.csv"
  pd.DataFrame(rows).to_csv(source, index=False)
  (run / "session.json").write_text(
    json.dumps({"mode": "fit", "output_dir": str(run), "model_dir": str(run)})
  )

  SingleFile(str(source), {"task": "regression"}, str(run)).process()
  values = pd.read_csv(run / DATA_SUBFOLDER / "values.csv")
  value_column = next(c for c in values.columns if c != "compound_id")
  assert len(values) == 3
  assert values[value_column].notna().all(), "a measured compound came out missing"

  summary = read_replicates(str(run))
  assert summary["n_rows"] == 4  # the blank is not a measurement
  assert summary["n_repeated"] == 1 and summary["pooled_sd"] == pytest.approx(0.2 * 2**0.5)


def test_two_spellings_of_one_molecule_are_counted(tmp_path, monkeypatch):
  """Dedupe matches the SMILES as written; standardisation runs after it and can collapse two.

  Those rows are then trained on twice, and a held-out split can put one copy in train and its
  twin in test. ZairaChem does not collapse them — which measurement to keep is the user's call —
  but it must not stay silent either.
  """
  import inspect
  import json

  import pandas as pd

  from zairachem.base.vars import DATA_SUBFOLDER
  from zairachem.setup.prep import check as check_module
  from zairachem.setup.prep.check import SetupChecker

  said = []
  monkeypatch.setattr(check_module.logger, "warning", lambda m, *a, **k: said.append(str(m)))

  run = tmp_path / "run"
  (run / DATA_SUBFOLDER).mkdir(parents=True)
  data = run / DATA_SUBFOLDER / "data.csv"
  (run / DATA_SUBFOLDER / "replicates.json").write_text(json.dumps({"n_repeated": 0}))

  checker = SetupChecker.__new__(SetupChecker)
  checker.data_file = str(data)

  # Every molecule distinct: nothing to say.
  pd.DataFrame({"compound_id": ["A", "B"], "smiles": ["CCO", "CCN"]}).to_csv(data, index=False)
  checker.check_duplicate_structures()
  assert said == []

  # Two compound ids, one molecule after standardisation.
  pd.DataFrame({"compound_id": ["A", "B", "C"], "smiles": ["CCO", "CCO", "CCN"]}).to_csv(
    data, index=False
  )
  checker.check_duplicate_structures()
  assert said and "appear more than once" in said[0]
  record = json.loads((run / DATA_SUBFOLDER / "replicates.json").read_text())
  assert record["standardised_collisions"] == 1

  # And the setup step has to actually call it, or none of the above is ever reached.
  assert "check_duplicate_structures" in inspect.getsource(SetupChecker.run)


def test_predict_says_so_when_the_featurizer_image_has_moved_on(monkeypatch):
  """A predict run copies the fit's `parameters.json`, so it scales with the fit's transformer
  but featurizes with whatever image is on the machine now. A major release in between is silent
  corruption: right shape, finished run, wrong numbers."""
  import inspect
  import types

  from zairachem.base.utils import model_version
  from zairachem.describe.descriptors.raw import RawDescriptors

  # An unresolvable version is None, not the "v1" that choosing a transformer falls back to --
  # otherwise every model fitted at v2 would look like it had been downgraded.
  def unresolvable(raiser_or_tags):
    monkeypatch.setattr(model_version, "_cache", {})
    monkeypatch.setattr(model_version, "_digest_cache", {})
    monkeypatch.setattr(model_version, "_tags_cache", {})
    monkeypatch.setattr(model_version, "_dockerhub_tags", raiser_or_tags)
    monkeypatch.setattr(model_version, "_local_image_digest", lambda _: None)
    monkeypatch.setattr(model_version, "_local_image_created", lambda _: None)

  def boom(_):
    raise RuntimeError("DockerHub unreachable")

  for tags in (boom, lambda _: [("latest", "2026-01-01T00:00:00Z", set())]):
    unresolvable(tags)  # no network, then a repo with no released semver tag
    assert model_version.resolved_model_version("eosX") is None
    unresolvable(tags)
    assert model_version.ersilia_model_version("eosX") == "v1"

  # A repo that does resolve still answers with its major.
  unresolvable(lambda _: [("v2.1.0", "2026-01-01T00:00:00Z", set())])
  assert model_version.resolved_model_version("eosX") == "v2"

  said = []
  run = types.SimpleNamespace(
    params={"latest_featurizer_version": {"eosA": "v1", "eosB": "v2", "eosC": "v1"}},
    logger=types.SimpleNamespace(warning=lambda msg, *a, **k: said.append(str(msg))),
  )
  now = {"eosA": "v2", "eosB": "v2", "eosC": None}  # advanced / unchanged / unresolvable
  monkeypatch.setattr(model_version, "resolved_model_version", lambda e: now[e])
  RawDescriptors._warn_if_versions_moved(run, ["eosA", "eosB", "eosC"])
  assert len(said) == 1
  assert "eosA v1\u2192v2" in said[0]
  # The two that did not move, and the one that could not be checked, are not accused.
  assert "eosB" not in said[0] and "eosC" not in said[0]

  # An older image is reported too: the transformer belongs to neither version.
  said.clear()
  now = {"eosA": "v1", "eosB": "v1", "eosC": "v1"}
  RawDescriptors._warn_if_versions_moved(run, ["eosB"])
  assert len(said) == 1 and "eosB v2\u2192v1" in said[0]

  # Nothing to say when nothing moved.
  said.clear()
  monkeypatch.setattr(model_version, "resolved_model_version", lambda e: "v1")
  RawDescriptors._warn_if_versions_moved(run, ["eosA", "eosC"])
  assert said == []

  # And it is only reached because describe calls it, at predict only.
  src = inspect.getsource(RawDescriptors.run)
  assert "_warn_if_versions_moved" in src and "is_predict()" in src


def test_disagreeing_duplicates_say_by_how_much(tmp_path):
  """Setup already warned that N compounds had duplicate rows with differing activities, and
  stopped there. A count leaves the only question it raises unanswered: repeats differing in the
  third decimal are rounding, repeats differing by four log units are two experiments, and the
  spread between them is the floor under any model's error."""
  import inspect
  import json

  from zairachem.setup.prep.check import SetupChecker

  checker = SetupChecker.__new__(SetupChecker)
  checker.data_file = str(tmp_path / "data.csv")
  record = tmp_path / "replicates.json"

  record.write_text(json.dumps({"pooled_sd": 0.6176799618899832, "max_range": 4.59}))
  phrase = checker._disagreement_phrase()
  assert "differ by 0.618 on average" in phrase and "as much as 4.59" in phrase

  # Classification has no spread in the target's units, so the count stands on its own.
  record.write_text(json.dumps({"disagreeing": 17}))
  assert checker._disagreement_phrase() == ""
  # And a run without the record is not a run that crashes.
  record.unlink()
  assert checker._disagreement_phrase() == ""

  assert "_disagreement_phrase" in inspect.getsource(SetupChecker.check_activity)
