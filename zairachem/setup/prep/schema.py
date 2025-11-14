import os
import pandas as pd
from rdkit import Chem

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger


_SNIFF_SAMPLE_SIZE = 1000
_MAX_EMPTY = 0.2  # TODO NOT USED
_MIN_CORRECT = 0.8


class InputSchema(ZairaBase):
  def __init__(self, input_file):
    ZairaBase.__init__(self)
    self.logger = logger
    self.input_file = os.path.abspath(input_file)
    self.df_ = pd.read_csv(self.input_file, nrows=_SNIFF_SAMPLE_SIZE)
    self.columns = [c for c in list(self.df_.columns) if not self.df_[c].isnull().all()]
    self.assigned_columns = set()

  def columns_iter(self):
    for c in self.columns:
      if c not in self.assigned_columns:
        yield c

  def add_explored_column(self, col):
    self.assigned_columns.update([col])

  def _prop_correct_smiles(self, col):
    values = list(self.df_[col])[
      :10
    ]  # check only the first 10 smiles, if they are not correct continue
    c = 0
    for v in values:
      try:
        mol = Chem.MolFromSmiles(v)
      except:
        continue
      if mol is not None:
        c += 1
    return float(c) / len(values)

  def _is_smiles_column(self, col):
    if "smiles" in col.lower():
      return True
    if self._prop_correct_smiles(col) > _MIN_CORRECT:
      return True
    else:
      return False

  def find_smiles_column(self):
    cols = []
    for col in self.columns:
      if self._is_smiles_column(col):
        cols += [col]
      else:
        continue
    if len(cols) > 1:
      raise Exception("More than one column contains smiles as header")
    return [cols[0]]

  def _is_values_column(self, col):
    try:
      values = list(self.df_[self.df_[col].notnull()][col])
      c = 0
      for v in values:
        try:
          float(v)
        except:
          continue
        c += 1
      if c == len(values):
        return True
      else:
        return False
    except:
      return False

  def find_values_column(self):
    for col in self.columns:
      if col == "bin":
        return [col]
    for col in self.columns:
      if col == "exp":
        return [col]
    cols = []
    for col in self.columns:
      if self._is_values_column(col):
        if col.lower() == "id":
          continue
        if col == "Unnamed: 0":
          continue
        cols += [col]
      else:
        continue
    return cols

  def resolve_columns(self):
    smiles_column = self.find_smiles_column()
    assert len(smiles_column) == 1, "No SMILES column found!"
    smiles_column = smiles_column[0]
    values_column = self.find_values_column()
    self.logger.debug("Values column {0}".format(values_column))
    if not values_column:
      values_column = None
    else:
      assert len(values_column) == 1, "More than one values column found! {0}".format(values_column)
      values_column = values_column[0]
    data = {}
    data["smiles_column"] = smiles_column
    data["values_column"] = values_column
    return data
