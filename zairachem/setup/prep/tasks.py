import os
import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import joblib
import json
from zairachem.base.vars import (
  COMPOUNDS_FILENAME,
  COMPOUND_IDENTIFIER_COLUMN,
  PARAMETERS_FILE,
  SMILES_COLUMN,
  VALUES_FILENAME,
  VALUES_COLUMN,
  TASKS_FILENAME,
)
from .files import ParametersFile
from zairachem.base.vars import MIN_CLASS, DATA_SUBFOLDER
from zairachem.base import ZairaBase

from sklearn.preprocessing import PowerTransformer, QuantileTransformer


class ExpectedTaskType(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.get_output_dir()

  def _get_params(self):
    params = ParametersFile(path=os.path.join(self.trained_path, DATA_SUBFOLDER, PARAMETERS_FILE))
    return params.load()

  def get(self):
    params = self._get_params()
    return params["task"]


class RegTasks(object):
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
    from .utils import SmoothenY

    return SmoothenY(self.smiles_list, raw).run()

  def raw(self, smoothen=None):
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
    raw = self.raw().reshape(-1, 1)
    tr = PowerTransformer(method="yeo-johnson")
    tr.fit(raw)
    joblib.dump(tr, os.path.join(self.path, DATA_SUBFOLDER, "pwr_transformer.joblib"))
    return tr.transform(raw).ravel()

  def rnk(self):
    raw = self.raw().reshape(-1, 1)
    tr = QuantileTransformer(output_distribution="uniform")
    tr.fit(raw)
    joblib.dump(tr, os.path.join(self.path, DATA_SUBFOLDER, "rnk_transformer.joblib"))
    return tr.transform(raw).ravel()

  def qnt(self):
    raw = self.raw().reshape(-1, 1)
    tr = QuantileTransformer(output_distribution="normal")
    tr.fit(raw)
    joblib.dump(tr, os.path.join(self.path, DATA_SUBFOLDER, "qnt_transformer.joblib"))
    return tr.transform(raw).ravel()

  def as_dict(self):
    res = OrderedDict()
    res["reg_raw_skip"] = self.raw(smoothen=True)  # TODO revise naming and choose regressor to keep
    res["reg_pwr_skip"] = self.pwr()
    res["reg_rnk_skip"] = self.rnk()
    res["reg_qnt"] = self.qnt()
    return res


class RegTasksForPrediction(RegTasks):
  def __init__(self, data, params, path):
    RegTasks.__init__(self, data, params, path)

  def load(self, path):
    self._load_path = path

  def pwr(self, raw):
    tr = joblib.load(os.path.join(self._load_path, DATA_SUBFOLDER, "pwr_transformer.joblib"))
    return tr.transform(raw.reshape(-1, 1)).ravel()

  def rnk(self, raw):
    tr = joblib.load(os.path.join(self._load_path, DATA_SUBFOLDER, "rnk_transformer.joblib"))
    return tr.transform(raw.reshape(-1, 1)).ravel()

  def qnt(self, raw):
    tr = joblib.load(os.path.join(self._load_path, DATA_SUBFOLDER, "qnt_transformer.joblib"))
    return tr.transform(raw.reshape(-1, 1)).ravel()

  def as_dict(self):
    res = OrderedDict()
    raw = self.raw(smoothen=False)
    res["reg_raw_skip"] = raw
    res["reg_pwr_skip"] = self.pwr(raw)
    res["reg_rnk_skip"] = self.rnk(raw)
    res["reg_qnt"] = self.qnt(raw)
    return res


class ClfTasks(object):
  def __init__(self, data):
    self.values = self.binarize(np.array(data[VALUES_COLUMN]))

  def _has_enough_min_class(self, bin):
    n1 = np.sum(bin)
    n0 = len(bin) - n1
    if n1 < MIN_CLASS or n0 < MIN_CLASS:
      return False
    return True

  def binarize(self, values):
    accepted = set([0, 0.5, 1])
    bin_values = []
    for v in values:
      if v not in accepted:
        raise Exception("Data is not binary. Cannot do classification")
      else:
        if v > 0:
          bin_values += [1]
        else:
          bin_values += [0]
    self._has_enough_min_class(bin_values)
    return bin_values


class ClfTasksForPrediction(object):
  def __init__(self, data):
    self.values = self.binarize(np.array(data[VALUES_COLUMN]))

  def binarize(self, values):
    accepted = set([0, 0.5, 1])
    bin_values = []
    for v in values:
      if v not in accepted:
        print("NOT ACCEPTED", v)
        raise Exception("Data is not binary. Cannot do classification")
      else:
        if v > 0:
          bin_values += [1]
        else:
          bin_values += [0]
    return bin_values


class SingleTasks(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    if self.is_predict():
      self.trained_path = self.get_trained_dir()
    else:
      self.trained_path = self.get_output_dir()
    self._task = ExpectedTaskType(path=path).get()

  def _get_params(self):
    params = ParametersFile(path=os.path.join(self.trained_path, DATA_SUBFOLDER, PARAMETERS_FILE))
    return params.load()

  def _get_data(self):
    df = pd.read_csv(os.path.join(self.path, VALUES_FILENAME))
    return df

  def run(self):
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
  def __init__(self, path):
    SingleTasks.__init__(self, path=path)

  def run(self):
    df = self._get_data()
    if self._task == "classification":
      self.logger.debug("It is simply a binary classification")
      ct = ClfTasksForPrediction(data=df)
      df["bin"] = ct.values
      self._task = "classification"
    elif self._task == "regression":
      self.logger.debug("Data is not simply a binary classification")
      # TODO
    df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)
