import json, h5py, os, gc
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import Hdf5, ChunkedH5Store, open_h5, DEFAULT_CHUNK_SIZE
from zairachem.base.vars import (
  COMPOUND_IDENTIFIER_COLUMN,
  PARAMETERS_FILE,
  SMILES_COLUMN,
  DATA_SUBFOLDER,
  DATA_FILENAME,
  ESTIMATORS_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  POOL_SUBFOLDER,
  RESULTS_UNMAPPED_FILENAME,
  INPUT_SCHEMA_FILENAME,
  MAPPING_FILENAME,
)


class ResultsIterator(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path

  def _read_model_ids(self):
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      model_ids = list(json.load(f))
    return model_ids

  def iter_relpaths(self):
    estimators_folder = os.path.join(self.path, ESTIMATORS_SUBFOLDER)
    model_ids = self._read_model_ids()
    rpaths = []
    for est_fam in os.listdir(estimators_folder):
      if os.path.isdir(os.path.join(estimators_folder, est_fam)):
        focus_folder = os.path.join(estimators_folder, est_fam)
        for d in os.listdir(focus_folder):
          if d in model_ids:
            rpaths += [[est_fam, d]]
    for rpath in rpaths:
      yield rpath

  def iter_abspaths(self):
    for rpath in self.iter_relpaths:
      yield "/".join([self.path] + rpath)


class XGetter(ZairaBase):
  def __init__(self, path, batch_size=None):
    ZairaBase.__init__(self)
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.X = []
    self.columns = []

  @staticmethod
  def _read_results_file(file_path):
    df = pd.read_csv(file_path)
    df = df[[c for c in list(df.columns) if c not in [SMILES_COLUMN, COMPOUND_IDENTIFIER_COLUMN]]]
    return df

  def _load_manifold(self, name):
    h5_path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, f"{name}.h5")
    h5 = open_h5(h5_path)
    if h5 is None:
      return
    shape = h5.shape()
    logger.info(f"[pool] loading {name} shape={shape}")
    if isinstance(h5, ChunkedH5Store):
      X_ = np.empty(shape, dtype=np.float32)
      for start, end, chunk in h5.iter_values_with_indices():
        X_[start:end] = np.array(chunk, dtype=np.float32)
    elif shape[0] <= self.batch_size * 2:
      X_ = np.array(h5.values(), dtype=np.float32)
    else:
      logger.info(f"[pool] loading {name} in chunks")
      X_ = np.empty(shape, dtype=np.float32)
      for start, end, chunk in h5.iter_values_with_indices(self.batch_size):
        X_[start:end] = np.array(chunk, dtype=np.float32)
    self.X.append(X_)
    for i in range(X_.shape[1]):
      self.columns.append(f"{name}-{i}")
    gc.collect()

  def _get_manifolds(self):
    for name in ("pca", "umap", "tsne"):
      self._load_manifold(name)

  def _get_results(self):
    prefixes = []
    dfs = []
    for rpath in ResultsIterator(path=self.path).iter_relpaths():
      prefixes += ["-".join(rpath)]
      file_name = "/".join([self.path, ESTIMATORS_SUBFOLDER] + rpath + [RESULTS_UNMAPPED_FILENAME])
      if not os.path.exists(file_name):
        logger.warning(f"[pool] Results file not found: {file_name}")
        continue
      dfs += [self._read_results_file(file_name)]
    for i in range(len(dfs)):
      df = dfs[i]
      prefix = prefixes[i]
      self.X += [np.array(df, dtype=np.float32)]
      self.columns += ["{0}-{1}".format(prefix, c) for c in list(df.columns)]
    self.logger.debug(
      "Number of columns: {0} ... from {1} estimators".format(len(self.columns), len(dfs))
    )

  def get(self):
    self._get_manifolds()
    self._get_results()
    X = np.hstack(self.X)
    df = pd.DataFrame(X, columns=self.columns)
    df.to_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME), index=False)
    return df

  def iter_get(self, chunk_size=None) -> Iterator[Tuple[int, int, pd.DataFrame]]:
    if chunk_size is None:
      chunk_size = self.batch_size
    self._get_manifolds()
    self._get_results()
    n_rows = self.X[0].shape[0] if self.X else 0
    logger.info(f"[pool:iter] Total rows={n_rows}, columns={len(self.columns)}")
    for start in range(0, n_rows, chunk_size):
      end = min(start + chunk_size, n_rows)
      chunk_arrays = [x[start:end] for x in self.X]
      chunk_X = np.hstack(chunk_arrays)
      chunk_df = pd.DataFrame(chunk_X, columns=self.columns)
      logger.debug(f"[pool:iter] Yielding rows {start}-{end}/{n_rows}")
      yield start, end, chunk_df
      del chunk_X, chunk_df, chunk_arrays
      gc.collect()


class BasePooler(ZairaBase):
  def __init__(self, path, batch_size=None):
    ZairaBase.__init__(self)
    self.logger = logger
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.task = self._get_task()

  def _get_task(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      task = json.load(f)["task"]
    return task

  def _get_compound_ids(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    cids = list(df["compound_id"])
    return cids

  def _get_X(self):
    df = XGetter(path=self.path, batch_size=self.batch_size).get()
    return df

  def _iter_X(self, chunk_size=None) -> Iterator[Tuple[int, int, pd.DataFrame]]:
    getter = XGetter(path=self.path, batch_size=self.batch_size)
    yield from getter.iter_get(chunk_size=chunk_size)

  def _get_X_clf(self, df):
    return df[[c for c in list(df.columns)]]

  def _get_X_reg(self, df):
    return df[[c for c in list(df.columns) if "reg" in c]]

  def _get_y(self, task):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    return np.array(df[task])

  def _get_Y_col(self):
    if self.task == "classification":
      Y_col = "bin"
    if self.task == "regression":
      Y_col = "val"
    return Y_col

  def _get_y(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    Y_col = self._get_Y_col()
    return np.array(df[Y_col])

  def _filter_out_bin(self, df):
    columns = list(df.columns)
    columns = [c for c in columns if "_bin" not in c]
    return df[columns]

  def _filter_out_manifolds(self, df):
    columns = list(df.columns)
    columns = [c for c in columns if "umap-" not in c and "pca-" not in c and "tsne-" not in c]
    return df[columns]

  def _filter_out_unwanted_columns(self, df):
    df = self._filter_out_manifolds(df)
    df = self._filter_out_bin(df)
    return df


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
