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
