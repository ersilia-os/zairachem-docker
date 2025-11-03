import json, h5py, os
import pandas as pd
import numpy as np

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  INPUT_SCHEMA_FILENAME,
  MAPPING_FILENAME,
  COMPOUND_IDENTIFIER_COLUMN,
  PARAMETERS_FILE,
  SMILES_COLUMN,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  ESTIMATORS_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  RAW_DESC_FILENAME,
  TREATED_DESC_FILENAME,
)


class BaseEstimator(ZairaBase):
  def __init__(self, path):
    self.logger = logger
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.logger.info(f"Specified model directory: {self.path}")
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.path
    self.task = self._get_task()

  def _get_task(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      task = json.load(f)["task"]
    return task

class BaseEstimatorIndividual(BaseEstimator):  # TODO MERGE WITH BASE
  def __init__(self, path, estimator, model_id):
    BaseEstimator.__init__(self, path=path)
    path_ = os.path.join(self.path, ESTIMATORS_SUBFOLDER, estimator, model_id)
    if not os.path.exists(path_):
      os.makedirs(path_)
    self.model_id = model_id
    self.task = self._get_task()

  def _get_task(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      task = json.load(f)["task"]
    return task

  def _get_X(self):
    f_treated = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, self.model_id, TREATED_DESC_FILENAME)
    f_raw = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, self.model_id, RAW_DESC_FILENAME)
    f = f_treated if os.path.exists(f_treated) else f_raw
    with h5py.File(f, "r") as f:
      X = f["Values"][:]
    return X

  def _get_Y_col(self):
    if self.task == "classification":
      Y_col = "bin"
    if self.task == "regression":
      Y_col = "val"
    return Y_col

  def _get_y(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    Y_col = self._get_Y_col()
    if self.is_predict():
      if Y_col not in df.columns:
        return None
    return np.array(df[Y_col])


class BaseOutcomeAssembler(ZairaBase):
  def __init__(self, path=None):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.path

  def _get_mappings(self):
    return pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, MAPPING_FILENAME))

  def _get_compounds(self):
    return pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))[
      [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]
    ]

  def _get_original_input_size(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      schema = json.load(f)
    file_name = schema["input_file"]
    return pd.read_csv(file_name).shape[0]

  def _remap(self, df, mappings):
    n = self._get_original_input_size()
    ncol = df.shape[1]
    R = [[None] * ncol for _ in range(n)]
    for m in mappings.values:
      i, j = m[0], m[1]
      if np.isnan(j):
        continue
      R[i] = list(df.iloc[int(j)])
    return pd.DataFrame(R, columns=list(df.columns))
