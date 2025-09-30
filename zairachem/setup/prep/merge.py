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
    if not has_tasks:
      df = pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))[
        [COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN]
      ]
      df = df.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
      df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
    else:
      df_cpd = pd.read_csv(os.path.join(self.path, STANDARD_COMPOUNDS_FILENAME))[
        [COMPOUND_IDENTIFIER_COLUMN, STANDARD_SMILES_COLUMN]
      ]
      df_cpd = df_cpd.rename(columns={STANDARD_SMILES_COLUMN: SMILES_COLUMN})
      df_tsk = pd.read_csv(os.path.join(self.path, TASKS_FILENAME))
      df = df_cpd.merge(df_tsk, on="compound_id")
      df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
