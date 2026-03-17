import json, h5py, os, gc
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import Hdf5, ChunkedH5Store, open_h5, DEFAULT_CHUNK_SIZE
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
  def __init__(self, path, batch_size=None):
    self.logger = logger
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
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


class BaseEstimatorIndividual(BaseEstimator):
  def __init__(self, path, estimator, model_id, batch_size=None):
    BaseEstimator.__init__(self, path=path, batch_size=batch_size)
    path_ = os.path.join(self.path, ESTIMATORS_SUBFOLDER, estimator, model_id)
    if not os.path.exists(path_):
      os.makedirs(path_)
    self.model_id = model_id
    self.task = self._get_task()

  def _get_task(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      task = json.load(f)["task"]
    return task

  def _open_h5(self):
    base = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, self.model_id)
    for fname in [TREATED_DESC_FILENAME, RAW_DESC_FILENAME]:
      h5 = open_h5(os.path.join(base, fname))
      if h5 is not None:
        return h5
    return None

  def _get_X_shape(self) -> Optional[Tuple[int, int]]:
    h5 = self._open_h5()
    if h5 is None:
      return None
    return h5.shape()

  def _get_X(self) -> Optional[np.ndarray]:
    h5 = self._open_h5()
    if h5 is None:
      self.logger.warning(f"[estimator] No H5 data found for {self.model_id}")
      return None
    shape = h5.shape()
    n_rows = shape[0]
    if isinstance(h5, ChunkedH5Store):
      self.logger.info(f"[estimator] loading {self.model_id} shape={shape} (chunked store)")
      X = np.empty(shape, dtype=np.float32)
      for start, end, chunk in h5.iter_values_with_indices():
        X[start:end] = chunk
      gc.collect()
      return X
    if n_rows <= self.batch_size * 2:
      self.logger.info(f"[estimator] loading {self.model_id} shape={shape} (in-memory)")
      return h5.values()
    self.logger.info(f"[estimator] loading {self.model_id} shape={shape} (chunked read)")
    X = np.empty(shape, dtype=np.float32)
    for start, end, chunk in h5.iter_values_with_indices(self.batch_size):
      X[start:end] = chunk
    gc.collect()
    return X

  def _iter_X(self, chunk_size: int = None) -> Iterator[Tuple[int, int, np.ndarray]]:
    if chunk_size is None:
      chunk_size = self.batch_size
    h5 = self._open_h5()
    n_rows = h5.n_rows()
    self.logger.info(f"[estimator] iterating {self.model_id} in chunks of {chunk_size}")
    if isinstance(h5, ChunkedH5Store):
      for start, end, chunk in h5.iter_values_with_indices():
        yield start, end, chunk
    else:
      for start, end, chunk in h5.iter_values_with_indices(chunk_size):
        yield start, end, chunk

  def _get_X_slice(self, start: int, end: int) -> np.ndarray:
    h5 = self._open_h5()
    return h5.values_slice(start, end)

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
