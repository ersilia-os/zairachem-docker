import os

import pandas as pd

from zairachem.base.vars import (
  POOL_SUBFOLDER,
  RESULTS_MAPPED_FILENAME,
  RESULTS_UNMAPPED_FILENAME,
)

from ..base import BaseOutcomeAssembler


class ReliabilityPoolerAssembler(BaseOutcomeAssembler):
  """Assembler that writes the pooled predictions as the run's final results tables."""

  def __init__(self, path=None):
    BaseOutcomeAssembler.__init__(self, path=path)

  def run(self, df):
    """Merge pooled predictions onto the compounds and write the unmapped and mapped results.

    The predictions are joined to ``data.csv`` compounds on ``compound_id`` (or concatenated
    row-wise when absent) and saved as ``results_unmapped.csv`` under the pool folder; they are
    then remapped to the original input order (empty rows for inputs that were dropped) and
    saved as ``results_mapped.csv``.
    """
    df_c = self._get_compounds()
    df_y = df
    avail_columns = set(df_y.columns)
    if "compound_id" in avail_columns:
      df = df_c.merge(df_y, how="left", on="compound_id")
    else:
      df = pd.concat([df_c, df_y], axis=1)
    df = df.reset_index(drop=True)
    df.to_csv(
      os.path.join(self.path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME),
      index=False,
    )
    mappings = self._get_mappings()
    df = self._remap(df, mappings)
    df.to_csv(
      os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME),
      index=False,
    )
