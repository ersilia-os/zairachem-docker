import os

import pandas as pd

from zairachem.base.utils.logging import logger
from zairachem.base.utils.progress import SetupProgress
from zairachem.base.vars import (
  COMPOUND_IDENTIFIER_COLUMN,
  DATA_FILENAME,
  SMILES_COLUMN,
  STANDARD_COMPOUNDS_FILENAME,
  STANDARD_SMILES_COLUMN,
  TASKS_FILENAME,
)


class DataMerger(object):
  """Setup step that joins standardised compounds and task columns into ``data.csv`` at fit."""

  def __init__(self, path):
    self.path = path

  def get_standard_smiles(self):
    """Return the standardised compounds table (``compounds_std.csv``)."""
    return pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))

  def get_tasks(self):
    """Return the per-compound task table (``tasks.csv``)."""
    return pd.read_csv(os.path.join(self.path, TASKS_FILENAME))

  def run(self):
    """Use the standardised SMILES, inner-join the tasks on compound id and write ``data.csv``."""
    df = self.get_standard_smiles()
    df = df.drop(columns=[SMILES_COLUMN])
    df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
    df_tsk = self.get_tasks()
    df = df.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN)
    df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)


class DataMergerForPrediction(object):
  """Setup step that builds ``data.csv`` from the standardised compounds at predict."""

  def __init__(self, path):
    self.path = path

  def run(self, has_tasks):
    """Write ``data.csv`` with compound ids and standardised SMILES.

    When ``has_tasks`` is True (the input carries ground truth), ``tasks.csv`` is left-joined so
    that unlabelled compounds keep their rows with NaN truth.
    """
    cpd_file = os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME)
    out_file = os.path.join(self.path, DATA_FILENAME)
    with SetupProgress() as progress:
      if not has_tasks:
        task = progress.add_task("Merging data", total=3)
        df = pd.read_csv(cpd_file, usecols=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
        progress.update(task, advance=1, description=f"Loaded {len(df):,} compounds")
        df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
        progress.update(task, advance=1, description="Writing merged data")
        df.to_csv(out_file, index=False)
        progress.update(task, advance=1, description="Merge complete")
      else:
        task = progress.add_task("Merging data", total=5)
        df_cpd = pd.read_csv(cpd_file, usecols=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
        progress.update(task, advance=1, description=f"Loaded {len(df_cpd):,} compounds")
        df_cpd = df_cpd.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
        tsk_file = os.path.join(self.path, TASKS_FILENAME)
        df_tsk = pd.read_csv(tsk_file)
        progress.update(task, advance=1, description=f"Loaded {len(df_tsk):,} task records")
        # Left join: at predict the task table may cover only the labelled compounds (partial ground
        # truth), so unlabelled compounds must keep their prediction rows (truth becomes NaN, which the
        # report's metric helpers drop). At fit every compound is labelled, so this is still 1:1.
        df = df_cpd.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN, how="left")
        progress.update(task, advance=1, description=f"Merged {len(df):,} records")
        df.to_csv(out_file, index=False)
        progress.update(task, advance=1, description="Writing merged data")
        progress.update(task, advance=1, description="Merge complete")
    logger.info(f"[merge] Saved {len(df):,} records to {out_file}")
