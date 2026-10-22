import os
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  COMPOUND_IDENTIFIER_COLUMN,
  COMPOUNDS_FILENAME,
  DATA_SUBFOLDER,
  METADATA_SUBFOLDER,
  MIN_CLASS,
  PARAMETERS_FILE,
  SMILES_COLUMN,
  TASKS_FILENAME,
  VALUES_COLUMN,
  VALUES_FILENAME,
)

from .files import ParametersFile


class ExpectedTaskType(ZairaBase):
  """Lookup of the task type (classification or regression) from the trained ``parameters.json``."""

  def __init__(self, path):
    ZairaBase.__init__(self)
    self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.get_output_dir()

  def _get_params(self):
    params = ParametersFile(
      path=os.path.join(self.trained_path, METADATA_SUBFOLDER, PARAMETERS_FILE)
    )
    return params.load()

  def get(self):
    """Return the ``task`` recorded in the trained model's ``parameters.json``."""
    params = self._get_params()
    return params["task"]


class RegTasks(object):
  """Regression targets derived from activity values, clipped to the credibility range."""

  def __init__(self, data, params, path):
    file_name = os.path.join(path, DATA_SUBFOLDER, COMPOUNDS_FILENAME)
    if not os.path.exists(file_name):
      file_name = os.path.join(path, COMPOUNDS_FILENAME)
    compounds = pd.read_csv(file_name)
    cid2smiles = {}
    for r in compounds[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]].values:
      cid2smiles[r[0]] = r[1]
    self.smiles_list = []
    for cid in list(data[COMPOUND_IDENTIFIER_COLUMN]):
      self.smiles_list += [cid2smiles[cid]]
    self.values = np.array(data[VALUES_COLUMN])
    self.direction = params["direction"]
    self.range = params["credibility_range"]
    self.path = path
    self._raw = None

  def smoothen(self, raw):
    """Return ``raw`` with heavily repeated values spread out by a fingerprint-based ranking."""
    from .utils import SmoothenY

    return SmoothenY(self.smiles_list, raw).run()

  def raw(self, smoothen=None):
    """Return the activity values clipped to the credibility range (cached after first call).

    If ``smoothen`` is truthy on the first call, repeated values are spread out with `SmoothenY`.
    """
    if self._raw is None:
      min_cred = self.range["min"]
      max_cred = self.range["max"]
      if min_cred is None and max_cred is None:
        raw = self.values
      else:
        raw = np.clip(self.values, min_cred, max_cred)
      if smoothen:
        self._raw = self.smoothen(raw)
      else:
        self._raw = raw
    return self._raw

  def pwr(self):
    """Fit and save a Yeo-Johnson power transform; return the transformed values."""
    raw = self.raw().reshape(-1, 1)
    from sklearn.preprocessing import PowerTransformer  # regression only; kept off start-up

    tr = PowerTransformer(method="yeo-johnson")
    tr.fit(raw)
    joblib.dump(tr, os.path.join(self.path, DATA_SUBFOLDER, "pwr_transformer.joblib"))
    return tr.transform(raw).ravel()

  def rnk(self):
    """Fit and save a uniform quantile transform; return the ranked values."""
    raw = self.raw().reshape(-1, 1)
    from sklearn.preprocessing import QuantileTransformer  # regression only

    tr = QuantileTransformer(output_distribution="uniform")
    tr.fit(raw)
    joblib.dump(tr, os.path.join(self.path, DATA_SUBFOLDER, "rnk_transformer.joblib"))
    return tr.transform(raw).ravel()

  def qnt(self):
    """Fit and save a normal quantile transform; return the transformed values."""
    raw = self.raw().reshape(-1, 1)
    from sklearn.preprocessing import QuantileTransformer  # regression only

    tr = QuantileTransformer(output_distribution="normal")
    tr.fit(raw)
    joblib.dump(tr, os.path.join(self.path, DATA_SUBFOLDER, "qnt_transformer.joblib"))
    return tr.transform(raw).ravel()

  def as_dict(self):
    """Return an ordered dict of the regression task columns (raw, power, rank, quantile)."""
    res = OrderedDict()
    res["reg_raw_skip"] = self.raw(smoothen=True)  # TODO revise naming and choose regressor to keep
    res["reg_pwr_skip"] = self.pwr()
    res["reg_rnk_skip"] = self.rnk()
    res["reg_qnt"] = self.qnt()
    return res


class RegTasksForPrediction(RegTasks):
  """Predict-time `RegTasks` that applies the transformers saved at fit instead of refitting."""

  def __init__(self, data, params, path):
    RegTasks.__init__(self, data, params, path)

  def load(self, path):
    """Set the trained model directory whose saved task transformers are applied."""
    self._load_path = path

  def pwr(self, raw):
    """Return ``raw`` transformed with the saved fit-time power transformer."""
    tr = joblib.load(os.path.join(self._load_path, DATA_SUBFOLDER, "pwr_transformer.joblib"))
    return tr.transform(raw.reshape(-1, 1)).ravel()

  def rnk(self, raw):
    """Return ``raw`` transformed with the saved fit-time uniform quantile transformer."""
    tr = joblib.load(os.path.join(self._load_path, DATA_SUBFOLDER, "rnk_transformer.joblib"))
    return tr.transform(raw.reshape(-1, 1)).ravel()

  def qnt(self, raw):
    """Return ``raw`` transformed with the saved fit-time normal quantile transformer."""
    tr = joblib.load(os.path.join(self._load_path, DATA_SUBFOLDER, "qnt_transformer.joblib"))
    return tr.transform(raw.reshape(-1, 1)).ravel()

  def as_dict(self):
    """Return an ordered dict of the regression task columns, without smoothing."""
    res = OrderedDict()
    raw = self.raw(smoothen=False)
    res["reg_raw_skip"] = raw
    res["reg_pwr_skip"] = self.pwr(raw)
    res["reg_rnk_skip"] = self.rnk(raw)
    res["reg_qnt"] = self.qnt(raw)
    return res


class ClfTasks(object):
  """Fit-time binary classification labels, validated for class balance."""

  def __init__(self, data):
    self.values = self.binarize(np.array(data[VALUES_COLUMN]))

  def _has_enough_min_class(self, bins):
    n1 = np.sum(bins)
    n0 = len(bins) - n1
    if n1 < MIN_CLASS or n0 < MIN_CLASS:
      return False
    return True

  def binarize(self, values):
    """Return 0/1 labels from values in {0, 0.5, 1} (0.5 maps to 1).

    Raises if any value is not binary or if either class has fewer than ``MIN_CLASS`` samples.
    """
    accepted = set([0, 0.5, 1])
    bin_values = []
    for v in values:
      if v not in accepted:
        raise Exception("Data is not binary. Cannot do classification")
      if v > 0:
        bin_values += [1]
      else:
        bin_values += [0]
    if not self._has_enough_min_class(bin_values):
      n1 = int(np.sum(bin_values))
      n0 = len(bin_values) - n1
      raise Exception(
        f"Not enough samples per class for classification: "
        f"{n1} positive / {n0} negative (minimum {MIN_CLASS} each required)."
      )
    return bin_values


class ClfTasksForPrediction(object):
  """Predict-time binary ground-truth labels, without the class-balance requirement."""

  def __init__(self, data):
    self.values = self.binarize(np.array(data[VALUES_COLUMN]))

  def binarize(self, values):
    """Return 0/1 labels from values in {0, 0.5, 1} (0.5 maps to 1); raise on other values."""
    accepted = set([0, 0.5, 1])
    bin_values = []
    for v in values:
      if v not in accepted:
        raise Exception(
          f"Data is not binary (found value {v!r}; expected one of {sorted(accepted)}). "
          "Cannot do classification."
        )
      if v > 0:
        bin_values += [1]
      else:
        bin_values += [0]
    return bin_values


class SingleTasks(ZairaBase):
  """Setup step that turns ``values.csv`` into ``tasks.csv`` for the run's task type at fit."""

  def __init__(self, path):
    ZairaBase.__init__(self)
    self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.get_output_dir()
    self._task = ExpectedTaskType(path=path).get()

  def _get_params(self):
    params = ParametersFile(
      path=os.path.join(self.trained_path, METADATA_SUBFOLDER, PARAMETERS_FILE)
    )
    return params.load()

  def _get_data(self):
    df = pd.read_csv(os.path.join(self.path, VALUES_FILENAME))
    return df

  def run(self):
    """Write ``tasks.csv``, adding a binary ``bin`` column for classification."""
    df = self._get_data()
    if self._task == "classification":
      self.logger.debug("It is simply a binary classification")
      ct = ClfTasks(data=df)
      df["bin"] = ct.values
    elif self._task == "regression":
      self.logger.debug("Data is not simply a binary")
    # TODO Keep only one column named val
    else:
      raise Exception("Task is not classification or regression, cannot proceed")
    df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)


class SingleTasksForPrediction(SingleTasks):
  """Predict-time `SingleTasks` that tolerates non-binary ground truth."""

  def __init__(self, path):
    SingleTasks.__init__(self, path=path)

  def run(self):
    """Write ``tasks.csv``, adding ``bin`` for classification when the ground truth is binary.

    Non-binary classification ground truth only logs a warning: the run then predicts without
    validation.
    """
    logger.info(f"[tasks] Loading values data from {self.path}")
    df = self._get_data()
    logger.info(f"[tasks] Loaded {len(df)} records, task type: {self._task}")
    if self._task == "classification":
      self.logger.debug("It is simply a binary classification")
      logger.info("[tasks] Processing classification task")
      try:
        ct = ClfTasksForPrediction(data=df)
        df["bin"] = ct.values
      except Exception as e:
        # The ground-truth column isn't binary {0, 0.5, 1}. We can't validate a classifier against
        # raw values (the fit-time binarization cutoff isn't persisted), so warn and predict only:
        # without a `bin` column the report simply omits the performance section. The run continues.
        logger.warning(
          f"[tasks] Ground-truth labels are not binary ({e}); skipping validation, predicting only."
        )
      self._task = "classification"
    elif self._task == "regression":
      self.logger.debug("Data is not simply a binary classification")
      logger.info("[tasks] Processing regression task")
    out_file = os.path.join(self.path, TASKS_FILENAME)
    logger.info(f"[tasks] Writing tasks to {out_file}")
    df.to_csv(out_file, index=False)
    logger.info("[tasks] Tasks processing complete")
