"""Shared setup machinery for the fit (`TrainSetup`) and predict (`PredictSetup`) pipelines.

The two setups diverge in their `__init__`, their `setup()` step sequence and a few prediction-only
steps (and they keep distinct ``PipelineStep`` names, which gate on-disk resumption — so those are
deliberately NOT unified). The pieces collected here are byte-identical between the two and the
store-summary formatting they both render.
"""

import os
import shutil

from zairachem.base import ZairaBase
from zairachem.base.utils.console import echo
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DEFAULT_ISAURA_BUCKET,
  OUTPUT_FILENAME,
  RAW_INPUT_FILENAME,
  SESSION_FILE,
  SMILES_COLUMN,
)


def format_store_summary(store):
  """One-line isaura-store summary for the setup panel, given the resolved store name (or None)."""
  if not store:
    return "store [red]off[/]"
  if store == DEFAULT_ISAURA_BUCKET:
    return f"store [green]on[/] · [bold]{DEFAULT_ISAURA_BUCKET}[/] [dim](shared lake)[/]"
  return f"store [green]on[/] · project [bold]{store}[/] · lake [bold]{DEFAULT_ISAURA_BUCKET}[/]"


def require_input_file(path):
  """Stop with a one-line error unless ``path`` is a readable file.

  Call this before anything is created, wiped or modified, so a mistyped input path costs nothing.

  Raises
  ------
  SystemExit
    With code 1 when ``path`` does not exist, is not a file, or cannot be read.
  """
  if not os.path.exists(path):
    echo(f"Input file '{path}' does not exist.", kind="error")
  elif not os.path.isfile(path):
    echo(f"Input path '{path}' is not a file.", kind="error")
  elif not os.access(path, os.R_OK):
    echo(f"Input file '{path}' is not readable.", kind="error")
  else:
    return
  raise SystemExit(1)


def require_zairachem_dir(path):
  """Stop unless ``path`` is absent, an empty directory or a ZairaChem directory.

  A ZairaChem directory is one that holds a ``session.json``. Any other existing path (a file, or
  a directory with content) may hold someone's own data, so the commands that call this refuse it:
  ``fit`` and ``setup`` (``TrainSetup``) always, with or without ``--override``, and ``predict``
  (``PredictSetup``) only when ``--override`` would wipe it. A ``predict`` without ``--override``
  into an existing folder does not call this check: it warns, keeps the folder and builds into it.

  Raises
  ------
  SystemExit
    With code 1 when ``path`` exists and is neither empty nor a ZairaChem directory.
  """
  if not os.path.lexists(path):
    return
  if os.path.isdir(path) and (
    not os.listdir(path) or os.path.exists(os.path.join(path, SESSION_FILE))
  ):
    return
  echo(
    f"'{path}' exists and is not a ZairaChem directory; choose another path or remove it yourself.",
    kind="error",
  )
  raise SystemExit(1)


def _is_within(path, parent):
  """Whether ``path`` is ``parent`` or lies inside it (both resolved through symlinks)."""
  path, parent = os.path.realpath(path), os.path.realpath(parent)
  return os.path.commonpath([path, parent]) == parent


def require_separate_dirs(output_dir, model_dir):
  """Stop when the output directory is the model directory, inside it, or contains it.

  Raises
  ------
  SystemExit
    With code 1 when the two directories overlap.
  """
  if _is_within(output_dir, model_dir) or _is_within(model_dir, output_dir):
    echo(
      f"The output directory '{output_dir}' is the model directory '{model_dir}', is inside it or "
      f"contains it; choose a different -o.",
      kind="error",
    )
    raise SystemExit(1)


class BaseSetup(object):
  """Machinery shared verbatim by `TrainSetup` and `PredictSetup` (each sets `self.input_file` /
  `self.output_dir` in its own ``__init__``)."""

  def _copy_input_file(self):
    """Keep the run's own copy of the input, always as ``raw_input.csv``.

    The input is read with ``pandas.read_csv`` whatever it is called, so a comma-separated file
    named ``.txt`` or ``.dat`` gets through setup; copying it under its own extension then left
    the report and the finish step — both of which look for ``raw_input.csv`` — to fail at the end
    of a run that had already done all its work.
    """
    shutil.copy(
      self.input_file,
      os.path.join(self.output_dir, DATA_SUBFOLDER, RAW_INPUT_FILENAME + ".csv"),
    )

  def _make_subfolder(self, name):
    # exist_ok=True so a resume (subfolders already present) doesn't crash. Harmless on the fresh
    # path, which wipes + recreates output_dir first, so the subfolders are always absent there.
    os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)

  def _input_smiles(self):
    import pandas as pd

    df = pd.read_csv(os.path.join(self.output_dir, DATA_SUBFOLDER, DATA_FILENAME))
    return df[SMILES_COLUMN].astype(str).tolist()

  def update_elapsed_time(self):
    """Add the time since the last stopwatch reset to the run's ``session.json``."""
    ZairaBase().update_elapsed_time(self.output_dir)

  def reset_time(self):
    """Restart the run's stopwatch stored in the output directory's ``session.json``."""
    ZairaBase().reset_time(self.output_dir)

  def is_done(self):
    """Return True if the run's final ``results/output.csv`` already exists."""
    return os.path.exists(os.path.join(self.output_dir, OUTPUT_FILENAME))
