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
    "heavy = [m for m in ('sklearn', 'matplotlib', 'rdkit', 'lazyqsar') if m in sys.modules];"
    "sys.exit(1 if heavy else 0)"
  )
  assert subprocess.run([sys.executable, "-c", code]).returncode == 0
