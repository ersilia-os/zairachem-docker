"""The CLI stops when the installed LazyQSAR is not the release ZairaChem pins.

A ``git pull`` that moves the pin changes the code at once, while LazyQSAR stays at the release that
was installed until ``pip install -e .`` runs again. These tests drive ``main()`` with the installed
version faked and the command group replaced by a stub, so nothing but the check runs.
"""

import importlib
import sys
from importlib import metadata
from pathlib import Path

import pytest
import tomllib

from zairachem.base.utils.lazyqsar_version import ALLOW_MISMATCH_ENV
from zairachem.base.vars import LAZYQSAR_VERSION

cli_main = importlib.import_module("zairachem.cli.main")  # zairachem.cli.main is also a function

OTHER = "9.9.9"


def _flat(text):
  return "".join(text.split())


@pytest.fixture
def run(monkeypatch, capsys):
  """Run ``main()`` with ``argv`` and LazyQSAR ``installed`` (None: not installed).

  Returns whether the command group ran, the exit code (None when it did not exit) and the console
  output without whitespace, so a message rich wrapped (a long ``sys.prefix``) still reads whole.
  """
  monkeypatch.delenv(ALLOW_MISMATCH_ENV, raising=False)
  real_version = metadata.version

  def _run(argv, installed):
    def version(name):
      if name != "lazyqsar":
        return real_version(name)
      if installed is None:
        raise metadata.PackageNotFoundError(name)
      return installed

    ran = []
    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(cli_main, "cli", lambda: ran.append(True))
    monkeypatch.setattr(sys, "argv", ["zairachem", *argv])
    code = None
    try:
      cli_main.main()
    except SystemExit as e:
      code = e.code
    return bool(ran), code, _flat(capsys.readouterr().out)

  return _run


def test_a_different_lazyqsar_stops_the_run_and_says_how_to_fix_it(run):
  ran, code, out = run(["fit", "-i", "x.csv", "-m", "m"], OTHER)
  assert (ran, code) == (False, 1)
  for text in (LAZYQSAR_VERSION, OTHER, sys.prefix, "pip install -e .", ALLOW_MISMATCH_ENV):
    assert _flat(text) in out


def test_the_pinned_lazyqsar_lets_the_run_go_on(run):
  assert run(["fit", "-i", "x.csv", "-m", "m"], LAZYQSAR_VERSION) == (True, None, "")


@pytest.mark.parametrize("override", [None, "1"])
def test_a_missing_lazyqsar_stops_the_run_even_with_the_override(run, monkeypatch, override):
  if override:
    monkeypatch.setenv(ALLOW_MISMATCH_ENV, override)
  ran, code, out = run(["predict", "-i", "x.csv", "-m", "m"], None)
  assert (ran, code) == (False, 1)
  assert _flat("not installed") in out and _flat("pip install -e .") in out


@pytest.mark.parametrize(
  "argv",
  [[], ["--help"], ["fit", "--help"], ["predict", "-h"], ["stop"], ["-v", "stop"], ["-v"], ["fti"]],
)
def test_help_screens_stop_and_bad_command_lines_run_on_any_lazyqsar(run, argv):
  # A command line click rejects (no command, a typo) gets click's own error, not this one.
  ran, code, _ = run(argv, OTHER)
  assert (ran, code) == (True, None)


def test_the_override_warns_and_goes_on(run, monkeypatch):
  monkeypatch.setenv(ALLOW_MISMATCH_ENV, "1")
  ran, code, out = run(["fit", "-i", "x.csv", "-m", "m"], OTHER)
  assert (ran, code) == (True, None)
  assert LAZYQSAR_VERSION in out and OTHER in out


def test_the_constant_is_the_lazyqsar_pin_in_pyproject():
  pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
  pin = tomllib.loads(pyproject.read_text())["tool"]["poetry"]["dependencies"]["lazyqsar"]
  assert pin["version"] == LAZYQSAR_VERSION


def test_a_local_build_of_the_pinned_release_goes_on(run):
  assert run(["fit", "-i", "x.csv", "-m", "m"], f"{LAZYQSAR_VERSION}+cpu") == (True, None, "")


def test_an_environment_path_with_brackets_is_printed_as_it_is(run, monkeypatch):
  monkeypatch.setattr(sys, "prefix", "/envs/[test]")
  _, code, out = run(["fit", "-i", "x.csv", "-m", "m"], OTHER)
  assert code == 1 and "/envs/[test]" in out
