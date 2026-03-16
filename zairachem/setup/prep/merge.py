import os
import pandas as pd
from zairachem.base.vars import (
  STANDARD_COMPOUNDS_FILENAME,
  FOLDS_FILENAME,
  TASKS_FILENAME,
  DATA_FILENAME,
  COMPOUND_IDENTIFIER_COLUMN,
  SMILES_COLUMN,
  STANDARD_SMILES_COLUMN,
)
from zairachem.base.utils.logging import logger


class DataMerger(object):
  def __init__(self, path):
    self.path = path

  def get_standard_smiles(self):
    return pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))

  def get_folds(self):
    return pd.read_csv(os.path.join(self.path, FOLDS_FILENAME))

  def get_tasks(self):
    return pd.read_csv(os.path.join(self.path, TASKS_FILENAME))

  def run(self):
    df1 = self.get_standard_smiles()
    df2 = self.get_folds()
    df = pd.merge(df1, df2, on=[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, STANDARD_SMILES_COLUMN])
    df = df.drop(columns=[SMILES_COLUMN])
    df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
    df_tsk = self.get_tasks()
    df = df.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN)
    df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)


class DataMergerForPrediction(object):
  def __init__(self, path):
    self.path = path

  def run(self, has_tasks):
    cpd_file = os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME)
    out_file = os.path.join(self.path, DATA_FILENAME)
    logger.info(f"[merge] Reading standardized compounds from {cpd_file}")
    if not has_tasks:
      df = pd.read_csv(cpd_file, usecols=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
      logger.info(f"[merge] Loaded {len(df)} compounds")
      df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
      logger.info(f"[merge] Writing merged data to {out_file}")
      df.to_csv(out_file, index=False)
      logger.info(f"[merge] Merge complete")
    else:
      df_cpd = pd.read_csv(cpd_file, usecols=[COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN])
      logger.info(f"[merge] Loaded {len(df_cpd)} compounds")
      df_cpd = df_cpd.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
      tsk_file = os.path.join(self.path, TASKS_FILENAME)
      logger.info(f"[merge] Reading tasks from {tsk_file}")
      df_tsk = pd.read_csv(tsk_file)
      logger.info(f"[merge] Loaded {len(df_tsk)} task records")
      logger.info(f"[merge] Merging compounds with tasks")
      df = df_cpd.merge(df_tsk, on=COMPOUND_IDENTIFIER_COLUMN)
      logger.info(f"[merge] Merged result has {len(df)} records")
      logger.info(f"[merge] Writing merged data to {out_file}")
      df.to_csv(out_file, index=False)
      logger.info(f"[merge] Merge complete")
