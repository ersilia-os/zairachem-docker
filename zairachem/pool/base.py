import json, os, gc, joblib
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.utils.results import ResultsIterator
from zairachem.base.vars import (
  COMPOUND_IDENTIFIER_COLUMN,
  PARAMETERS_FILE,
  SMILES_COLUMN,
  DATA_SUBFOLDER,
  METADATA_SUBFOLDER,
  DATA_FILENAME,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  RESULTS_UNMAPPED_FILENAME,
  INPUT_SCHEMA_FILENAME,
  MAPPING_FILENAME,
)


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

  def _get_results(self):
    # Idempotent: reset accumulators so a second call (or get() after iter_get() on the same
    # instance) can't double-append columns.
    self.X = []
    self.columns = []
    prefixes = []
    dfs = []
    for rpath in ResultsIterator(path=self.path).iter_relpaths():
      file_name = "/".join([self.path, ESTIMATORS_SUBFOLDER] + rpath + [RESULTS_UNMAPPED_FILENAME])
      if not os.path.exists(file_name):
        logger.warning(f"[pool] Results file not found: {file_name}")
        continue
      prefixes += ["-".join(rpath)]
      dfs += [self._read_results_file(file_name)]
    for prefix, df in zip(prefixes, dfs):
      self.X += [np.array(df, dtype=np.float32)]
      self.columns += ["{0}-{1}".format(prefix, c) for c in list(df.columns)]
    self.logger.debug(
      "Number of columns: {0} ... from {1} estimators".format(len(self.columns), len(dfs))
    )

  def get(self):
    self._get_results()
    X = np.hstack(self.X)
    df = pd.DataFrame(X, columns=self.columns)
    df.to_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME), index=False)
    return df

  def iter_get(self, chunk_size=None) -> Iterator[Tuple[int, int, pd.DataFrame]]:
    if chunk_size is None:
      chunk_size = self.batch_size
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
    with open(os.path.join(self.path, METADATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      task = json.load(f)["task"]
    return task

  def _get_compound_ids(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    cids = list(df[COMPOUND_IDENTIFIER_COLUMN])
    return cids

  def _get_X(self):
    df = XGetter(path=self.path, batch_size=self.batch_size).get()
    return df

  def _iter_X(self, chunk_size=None) -> Iterator[Tuple[int, int, pd.DataFrame]]:
    getter = XGetter(path=self.path, batch_size=self.batch_size)
    yield from getter.iter_get(chunk_size=chunk_size)

  def _get_X_clf(self, df):
    # The classifier bagger stacks on the per-descriptor clf probabilities only. Drop the
    # manifold projection columns (pca-/umap-/tsne-) and the binary (_bin) columns, which
    # should not be fed to the meta-classifier (they dilute/leak rather than help).
    return self._filter_out_unwanted_columns(df)

  def _get_X_reg(self, df):
    return df[[c for c in list(df.columns) if "reg" in c]]

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

  def _filter_out_unwanted_columns(self, df):
    # Single pass over the columns, dropping everything the meta-models must not see: manifold
    # projections (pca-/umap-/tsne-), the binary (_bin) columns, and the per-sample rank/AD signals
    # (which the reliability pooler consumes as separate aligned matrices, and which the legacy
    # bagger's "clf"-substring filter would otherwise silently ingest, e.g. "<desc>-clf_rank").
    keep = [
      c
      for c in df.columns
      if "umap-" not in c
      and "pca-" not in c
      and "tsne-" not in c
      and "_bin" not in c
      and not c.endswith("_rank")
      and not c.endswith("_ad")
      and not c.endswith("_raw")
    ]
    return df[keep]

  # ---- Reliability pooler getters -------------------------------------------------------------
  # The reliability pooler reads the prediction matrix plus two optional per-sample signal
  # matrices (rank quantile, applicability domain) and per-descriptor metadata (OOF AUC,
  # rank→error curve, AD veto cutoff). Each is individually guarded so partial upstreams degrade
  # per-signal rather than all-or-nothing.

  def _get_pred_columns(self, df, task):
    """Ordered list of the per-descriptor prediction columns (no _bin/_rank/_ad, no manifolds)."""
    if task == "classification":
      df = self._filter_out_unwanted_columns(df)
      return [c for c in list(df.columns) if c.endswith("-clf")]
    return [c for c in list(df.columns) if c.endswith("-reg")]

  def _get_signal_matrix(self, df, pred_columns, suffix):
    """(matrix (B,D) aligned to pred_columns, present) for a sibling signal suffix, or (None, False).

    Returns None when any descriptor lacks the sibling column or the whole matrix is non-finite.
    """
    cols = [c + suffix for c in pred_columns]
    if not pred_columns or any(c not in df.columns for c in cols):
      return None, False
    M = np.asarray(df[cols], dtype=np.float64)
    if not np.isfinite(M).any():
      return None, False
    return M, True

  def _get_rank_matrix(self, df, pred_columns):
    return self._get_signal_matrix(df, pred_columns, "_rank")

  def _get_ad_matrix(self, df, pred_columns):
    return self._get_signal_matrix(df, pred_columns, "_ad")

  def _get_descriptor_meta(self, pred_columns, task):
    """Per-descriptor (oof_auc, rank_error_curve, ad_hard_cutoff) aligned to pred_columns.

    Reads each descriptor's ``pool_signals.joblib`` (written by the estimate step). At predict time
    these training artifacts live under the trained model dir. Missing files/keys → None entries,
    so the pooler falls back gracefully.
    """
    suffix = "-clf" if task == "classification" else "-reg"
    meta_path = self.get_trained_dir() if self.is_predict() else self.path
    # Map each descriptor's prediction column to its results folder via the relpath prefix.
    prefix_to_dir = {}
    for rpath in ResultsIterator(path=meta_path).iter_relpaths():
      prefix = "-".join(rpath)
      prefix_to_dir[prefix + suffix] = os.path.join(meta_path, ESTIMATORS_SUBFOLDER, *rpath)
    oof_aucs, curves, cutoffs = [], [], []
    for col in pred_columns:
      d = prefix_to_dir.get(col)
      sig = {}
      if d is not None:
        fp = os.path.join(d, "pool_signals.joblib")
        if os.path.exists(fp):
          try:
            sig = joblib.load(fp)
          except Exception:
            sig = {}
      oof_aucs.append(sig.get("oof_auc"))
      curves.append(sig.get("rank_error_curve"))
      cutoffs.append(sig.get("ad_hard_cutoff"))
    return oof_aucs, curves, cutoffs


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
