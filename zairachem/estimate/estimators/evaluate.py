import collections, json, os
import pandas as pd

from sklearn.metrics import roc_auc_score, r2_score
from zairachem.base import ZairaBase
from zairachem.base.utils.results import ResultsIterator
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  RESULTS_UNMAPPED_FILENAME,
  SIMPLE_EVALUATION_FILENAME,
  SIMPLE_EVALUATION_VALIDATION_FILENAME,
  INPUT_SCHEMA_FILENAME,
)


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

  @staticmethod
  def _evaluate(df_true, df_pred):
    """Per-task scores (clf -> ROC-AUC, reg -> R2) from row-aligned truth/prediction frames."""
    data = collections.OrderedDict()
    for c in list(df_pred.columns):
      if c == "clf":
        if len(set(df_true["bin"])) > 1:
          data[c] = {"roc_auc_score": roc_auc_score(df_true["bin"], df_pred[c])}
        else:
          data[c] = 0.0
      elif c == "reg":
        data[c] = {"r2_score": r2_score(df_true["val"], df_pred[c])}
    return data

  def run(self):
    # Read each file once, then derive BOTH the full-set evaluation and the validation-subset
    # evaluation (fit only) from the in-memory frames — the previous two-pass version re-read
    # data.csv and every estimator's results from disk a second time.
    value_col = self._get_original_input_value()
    if value_col is None:
      return
    df_true = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    valid_idxs = None if self.is_predict() else self.get_validation_indices(path=self.path)
    for relpath in self.results_iterator.iter_relpaths():
      abspath = "/".join([self.path, ESTIMATORS_SUBFOLDER] + relpath)
      df_pred = pd.read_csv(os.path.join(abspath, RESULTS_UNMAPPED_FILENAME))
      with open(os.path.join(abspath, SIMPLE_EVALUATION_FILENAME), "w") as f:
        json.dump(self._evaluate(df_true, df_pred), f, indent=4)
      if valid_idxs is not None:
        data_v = self._evaluate(df_true.iloc[valid_idxs, :], df_pred.iloc[valid_idxs, :])
        with open(os.path.join(abspath, SIMPLE_EVALUATION_VALIDATION_FILENAME), "w") as f:
          json.dump(data_v, f, indent=4)
