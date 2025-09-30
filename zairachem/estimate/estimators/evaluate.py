import collections, json, os
import pandas as pd

from sklearn.metrics import roc_auc_score, r2_score
from zairachem.base import ZairaBase
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  RESULTS_UNMAPPED_FILENAME,
  SIMPLE_EVALUATION_FILENAME,
  SIMPLE_EVALUATION_VALIDATION_FILENAME,
  INPUT_SCHEMA_FILENAME,
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


class SimpleEvaluator(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.results_iterator = ResultsIterator(path=self.path)

  def _get_original_input_value(self):
    with open(os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      schema = json.load(f)
    return schema["values_column"]

  def _run(self, valid_idxs):
    df_true = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    if valid_idxs is not None:
      df_true = df_true.iloc[valid_idxs, :]
    for relpath in self.results_iterator.iter_relpaths():
      abspath = "/".join([self.path, ESTIMATORS_SUBFOLDER] + relpath)
      file_path = os.path.join(abspath, RESULTS_UNMAPPED_FILENAME)
      df_pred = pd.read_csv(file_path)
      if valid_idxs is not None:
        df_pred = df_pred.iloc[valid_idxs, :]
      data = collections.OrderedDict()
      for c in list(df_pred.columns):
        if c == "clf":
          c_real = "bin"
          if len(set(df_true[c_real])) > 1:
            data[c] = {"roc_auc_score": roc_auc_score(df_true[c_real], df_pred[c])}
          else:
            data[c] = 0.0
        elif c == "reg":
          c_real = "val"
          data[c] = {"r2_score": r2_score(df_true[c_real], df_pred[c])}
      if valid_idxs is not None:
        file_name = SIMPLE_EVALUATION_VALIDATION_FILENAME
      else:
        file_name = SIMPLE_EVALUATION_FILENAME
      with open(os.path.join(abspath, file_name), "w") as f:
        json.dump(data, f, indent=4)

  def run(self):  # TODO WHY RUN TWICE?
    value_col = self._get_original_input_value()
    if value_col is not None:
      self._run(None)
      if not self.is_predict():
        valid_idxs = self.get_validation_indices(path=self.path)
        self._run(valid_idxs)
    else:
      pass
