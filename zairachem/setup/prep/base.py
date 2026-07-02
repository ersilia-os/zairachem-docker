"""Shared setup machinery for the fit (`TrainSetup`) and predict (`PredictSetup`) pipelines.

The two setups diverge in their `__init__`, their `setup()` step sequence and a few prediction-only
steps (and they keep distinct ``PipelineStep`` names, which gate on-disk resumption — so those are
deliberately NOT unified). The pieces collected here are byte-identical between the two and the
store-summary formatting they both render.
"""

import os
import shutil

from zairachem.base import ZairaBase
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DEFAULT_ISAURA_BUCKET,
  OUTPUT_FILENAME,
  RAW_INPUT_FILENAME,
  SMILES_COLUMN,
)


def format_store_summary(store):
  """One-line isaura-store summary for the setup panel, given the resolved store name (or None)."""
  if not store:
    return "store [red]off[/]"
  if store == DEFAULT_ISAURA_BUCKET:
    return f"store [green]on[/] · [bold]{DEFAULT_ISAURA_BUCKET}[/] [dim](shared lake)[/]"
  return f"store [green]on[/] · project [bold]{store}[/] · lake [bold]{DEFAULT_ISAURA_BUCKET}[/]"


class BaseSetup(object):
  """Machinery shared verbatim by `TrainSetup` and `PredictSetup` (each sets `self.input_file` /
  `self.output_dir` in its own ``__init__``)."""

  def _copy_input_file(self):
    extension = self.input_file.split(".")[-1]
    shutil.copy(
      self.input_file,
      os.path.join(self.output_dir, DATA_SUBFOLDER, RAW_INPUT_FILENAME + "." + extension),
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
    ZairaBase().update_elapsed_time()

  def reset_time(self):
    ZairaBase().reset_time()

  def is_done(self):
    return os.path.exists(os.path.join(self.output_dir, OUTPUT_FILENAME))
