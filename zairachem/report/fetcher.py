import collections, json, os
import pandas as pd
import numpy as np
from sklearn.metrics import (
  roc_curve,
  auc,
  precision_recall_curve,
  precision_score,
  recall_score,
  f1_score,
  accuracy_score,
  balanced_accuracy_score,
  matthews_corrcoef,
  confusion_matrix,
)

from zairachem.report.utils import ResultsIterator
from zairachem.estimate.tools.ghost.ghost import GhostLight
from zairachem.base import ZairaBase
from zairachem.base.vars import (
  MAPPING_FILENAME,
  VALUES_COLUMN,
  SMILES_COLUMN,
  INPUT_SCHEMA_FILENAME,
  PARAMETERS_FILE,
  RAW_INPUT_FILENAME,
  MAPPING_ORIGINAL_COLUMN,
  MAPPING_DEDUPE_COLUMN,
  DATA_FILENAME,
  DATA_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  POOL_SUBFOLDER,
  RESULTS_UNMAPPED_FILENAME,
)

RAW_INPUT_FILENAME += ".csv"


class ResultsFetcher(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.trained_path = self.get_trained_dir()
    self.individual_results_iterator = ResultsIterator(path=self.path)
    self.clf_task = "bin"
    self.reg_task = "val"
    self.params = self.get_parameters()

  def _read_data(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    return df

  def _read_data_train(self):
    df = pd.read_csv(os.path.join(self.trained_path, DATA_SUBFOLDER, DATA_FILENAME))
    return df

  def _read_pooled_results(self, path=None):
    if path is None:
      path = self.path
    df = pd.read_csv(os.path.join(path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME))
    return df

  def _read_pooled_results_train(self):
    return self._read_pooled_results(path=self.trained_path)

  def _read_individual_estimator_results(self, task, path=None):
    if path is None:
      path = self.path
    prefixes = []
    R = []
    for rpath in ResultsIterator(path=path).iter_relpaths():
      prefixes += ["-".join(rpath)]
      file_name = "/".join([path, ESTIMATORS_SUBFOLDER] + rpath + [RESULTS_UNMAPPED_FILENAME])
      df = pd.read_csv(file_name)
      R += [list(df[task])]
    d = collections.OrderedDict()
    for i in range(len(R)):
      d[prefixes[i]] = R[i]
    return pd.DataFrame(d)

  def _read_individual_estimator_results_train(self, task):
    return self._read_individual_estimator_results(task=task, path=self.trained_path)

  def _read_processed_data(self):
    df = pd.read_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME))
    return df

  def _read_processed_data_train(self):
    df = pd.read_csv(os.path.join(self.trained_path, POOL_SUBFOLDER, DATA_FILENAME))
    return df

  def get_tasks(self):
    df = self._read_data()
    tasks = [c for c in list(df.columns) if ("clf" in c or "reg" in c)]
    return tasks

  def get_reg_tasks(self):
    df = self._read_data()
    tasks = [c for c in list(df.columns) if "val" in c]
    return tasks

  def get_clf_tasks(self, data=None):
    if data is None:
      df = self._read_data()
    else:
      df = data
    tasks = [c for c in list(df.columns) if "clf" in c]
    if len(tasks) == 0:
      df = self._read_pooled_results()
      return self.get_clf_tasks(data=df)
    return tasks

  def get_actives_inactives(self):
    df = self._read_data()
    return list(df[self.clf_task])

  def get_actives_inactives_trained(self):
    df = self._read_data_train()
    return list(df[self.clf_task])

  def get_raw(self):
    df = self._read_data()
    return list(df[self.reg_task])

  def get_transformed(self):  # TODO adapt for REG final
    df = self._read_data()
    for c in list(df.columns):
      if "reg" in c:
        return list(df[c])

  def get_true_clf(self):
    return self.get_actives_inactives()

  def get_pred_binary_clf(self):
    df = self._read_pooled_results()
    for c in list(df.columns):
      if "clf" in c and "bin" in c:
        return list(df[c])

  def get_pred_binary_clf_trained(self):
    df = self._read_pooled_results_train()
    for c in list(df.columns):
      if "clf" in c and "bin" in c:
        return list(df[c])

  def get_pred_proba_clf(self):
    df = self._read_pooled_results()
    for c in list(df.columns):
      if "clf" in c and "bin" not in c:
        return list(df[c])

  def get_pred_proba_clf_trained(self):
    df = self._read_pooled_results_train()
    for c in list(df.columns):
      if "clf" in c and "bin" not in c:
        return list(df[c])

  def get_pred_reg_trans(self):  # TODO ADAPT FOR REG
    df = self._read_pooled_results()
    for c in list(df.columns):
      if "reg" in c and "raw" not in c:
        return list(df[c])

  def get_pred_reg_trans_trained(self):  # TODO Adapt for Reg
    df = self._read_pooled_results_train()
    for c in list(df.columns):
      if "reg" in c and "raw" not in c:
        return list(df[c])

  def get_pred_reg_raw(self):
    df = self._read_pooled_results()
    for c in list(df.columns):
      if "reg" in c and "raw" in c:
        return list(df[c])

  def get_pred_reg_raw_trained(self):
    df = self._read_pooled_results_train()
    for c in list(df.columns):
      if "reg" in c and "raw" in c:
        return list(df[c])

  def get_projections_umap(self):
    df = self._read_processed_data()
    umap0 = [0] * df.shape[0]
    umap1 = [0] * df.shape[0]
    for c in list(df.columns):
      if "umap-0" in c:
        umap0 = list(df["umap-0"])
      if "umap-1" in c:
        umap1 = list(df["umap-1"])
    return umap0, umap1

  def get_projections_tsne(self):
    df = self._read_processed_data()
    tsne0 = [0] * df.shape[0]
    tsne1 = [0] * df.shape[0]
    for c in list(df.columns):
      if "tsne-0" in c:
        tsne0 = list(df["tsne-0"])
      if "tsne-1" in c:
        tsne1 = list(df["tsne-1"])
    return tsne0, tsne1

  def get_projections_pca(self):
    df = self._read_processed_data()
    pca0 = [0] * df.shape[0]
    pca1 = [0] * df.shape[0]
    for c in list(df.columns):
      if "pca-0" in c:
        pca0 = list(df["pca-0"])
      if "pca-1" in c:
        pca1 = list(df["pca-1"])
    return pca0, pca1

  def get_projections_umap_trained(self):
    df = self._read_processed_data_train()
    umap0 = [0] * df.shape[0]
    umap1 = [0] * df.shape[0]
    if "umap-0" not in df.columns or "umap-1" not in df.columns:
      return None
    else:
      for c in list(df.columns):
        if "umap-0" in c:
          umap0 = list(df["umap-0"])
        if "umap-1" in c:
          umap1 = list(df["umap-1"])
      return umap0, umap1

  def get_projections_tsne_trained(self):
    df = self._read_processed_data_train()
    tsne0 = [0] * df.shape[0]
    tsne1 = [0] * df.shape[0]
    if "tsne-0" not in df.columns or "tsne-1" not in df.columns:
      return None
    else:
      for c in list(df.columns):
        if "tsne-0" in c:
          tsne0 = list(df["tsne-0"])
        if "tsne-1" in c:
          tsne1 = list(df["tsne-1"])
      return tsne0, tsne1

  def get_projections_pca_trained(self):
    df = self._read_processed_data_train()
    pca0 = [0] * df.shape[0]
    pca1 = [0] * df.shape[0]
    if "pca-0" not in df.columns or "pca-1" not in df.columns:
      return None
    else:
      for c in list(df.columns):
        if "pca-0" in c:
          pca0 = list(df["pca-0"])
        if "pca-1" in c:
          pca1 = list(df["pca-1"])
      return pca0, pca1

  def get_parameters(self):
    with open(os.path.join(self.trained_path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      return json.load(f)

  def get_smiles(self):
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    return list(df[SMILES_COLUMN])

  def get_original_smiles(self):
    raw_data = pd.read_csv(os.path.join(self.path, RAW_INPUT_FILENAME))
    with open(os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      schema = json.load(f)
    return list(raw_data[schema["smiles_column"]])

  def get_original_values(self):
    raw_data = pd.read_csv(os.path.join(self.path, RAW_INPUT_FILENAME))
    with open(os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      schema = json.load(f)
    if schema["values_column"] is None:
      return None
    else:
      return list(raw_data[VALUES_COLUMN])

  def _top_binarizer(self, y, k):
    idxs = np.argsort(y)[::-1]
    idxs = idxs[:k]
    b = [0] * len(y)
    for i in idxs:
      b[i] = 1
    return b

  def classification_performance_report(self, y_true_train, y_pred_train, y_true_test, y_pred_test):
    n_tr = len(y_true_train)
    n_te = len(y_true_test)
    n_tr_0 = int(np.sum(y_true_train == 0))
    n_tr_1 = int(np.sum(y_true_train == 1))
    n_te_0 = int(np.sum(y_true_test == 0))
    n_te_1 = int(np.sum(y_true_test == 1))
    fpr, tpr, _ = roc_curve(y_true_test, y_pred_test)
    auroc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_test, y_pred_test)
    aupr = auc(rec, prec)
    ghost = GhostLight()
    cutoff = ghost.get_threshold(y_true_train, y_pred_train)
    b_pred_test = []
    for y in y_pred_test:
      if y >= cutoff:
        b_pred_test += [1]
      else:
        b_pred_test += [0]
    b_pred_test = np.array(b_pred_test)
    tn, fp, fn, tp = confusion_matrix(y_true_test, b_pred_test).ravel()
    data = collections.OrderedDict()
    data["num_train"] = n_tr
    data["num_test"] = n_te
    data["num_train_0"] = n_tr_0
    data["num_train_1"] = n_tr_1
    data["num_test_0"] = n_te_0
    data["num_test_1"] = n_te_1
    data["auroc"] = auroc
    data["aupr"] = aupr
    data["cutoff"] = cutoff
    data["tp"] = tp
    data["tn"] = tn
    data["fp"] = fp
    data["fn"] = fn
    data["accuracy"] = accuracy_score(y_true_test, b_pred_test)
    data["balanced_accuracy"] = balanced_accuracy_score(y_true_test, b_pred_test)
    data["precision"] = precision_score(y_true_test, b_pred_test)
    data["recall"] = recall_score(y_true_test, b_pred_test)
    data["f1_score"] = f1_score(y_true_test, b_pred_test)
    data["mcc"] = matthews_corrcoef(y_true_test, b_pred_test)
    data["precision_at_1"] = precision_score(y_true_test, self._top_binarizer(y_pred_test, 1))
    data["precision_at_5"] = precision_score(y_true_test, self._top_binarizer(y_pred_test, 5))
    data["precision_at_10"] = precision_score(y_true_test, self._top_binarizer(y_pred_test, 10))
    data["precision_at_50"] = precision_score(y_true_test, self._top_binarizer(y_pred_test, 50))
    data["precision_at_100"] = precision_score(y_true_test, self._top_binarizer(y_pred_test, 100))
    return data

  def regression_performance_report(self):
    pass

  def map_to_original(self, values):
    n = pd.read_csv(os.path.join(self.path, RAW_INPUT_FILENAME)).shape[0]
    dm = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, MAPPING_FILENAME))
    dm = dm[dm[MAPPING_DEDUPE_COLUMN].notnull()]
    u2o = collections.defaultdict(list)
    for v in dm[[MAPPING_ORIGINAL_COLUMN, MAPPING_DEDUPE_COLUMN]].values:
      u2o[int(v[1])] += [int(v[0])]
    mapped_values = [None] * n
    for i, v in enumerate(values):
      if i in u2o:
        for idx in u2o[i]:
          mapped_values[idx] = v
      else:
        continue
    return mapped_values
