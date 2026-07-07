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
  SMILES_COLUMN,
  INPUT_SCHEMA_FILENAME,
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

#: Process-level CSV read cache shared by every ResultsFetcher, keyed by (abspath, mtime) so a
#: rewritten file is re-read. Lets the report's per-plot fetchers avoid re-reading the same artifacts.
_CSV_CACHE = {}


class ResultsFetcher(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.trained_path = self.get_trained_dir()
    self.clf_task = "bin"
    self.reg_task = "val"
    # Per-instance map cache (dedupe→original index). CSV reads use the process-level cache below.
    self._map_cache = {}
    self._cv_dirs_cache = None

  def _read_csv_cached(self, path):
    # Process-level cache shared across ALL fetcher instances: the report renders dozens of plots, and
    # each plot builds its own ResultsFetcher, so a per-instance cache would still re-read the same
    # data.csv / pooled results / projections from disk once per plot. Keyed by (abspath, mtime) so a
    # rewritten file is never served stale. Frames are treated as READ-ONLY by callers (they slice
    # columns or build new frames; none mutate in place), so sharing one object is safe.
    try:
      key = (os.path.abspath(path), os.path.getmtime(path))
    except OSError:
      return pd.read_csv(path)  # missing/unreadable — let read_csv raise the real error
    df = _CSV_CACHE.get(key)
    if df is None:
      df = pd.read_csv(path)
      _CSV_CACHE[key] = df
    return df

  def _read_data(self):
    return self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))

  def _read_data_train(self):
    return self._read_csv_cached(os.path.join(self.trained_path, DATA_SUBFOLDER, DATA_FILENAME))

  def _read_pooled_results(self, path=None):
    if path is None:
      path = self.path
    return self._read_csv_cached(os.path.join(path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME))

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
      df = self._read_csv_cached(file_name)
      R += [list(df[task])]
    d = collections.OrderedDict()
    for i in range(len(R)):
      d[prefixes[i]] = R[i]
    return pd.DataFrame(d)

  def _read_individual_estimator_results_train(self, task):
    return self._read_individual_estimator_results(task=task, path=self.trained_path)

  # --- lazy-qsar internal cross-validation (captured at fit time) --------------------------------

  def _cv_dirs(self):
    """``{descriptor: dir}`` holding the lazy-qsar CV artefacts (cv_report.json / oof.csv).

    Prefers the finished ``model/estimators`` layout; falls back to the in-pipeline
    ``pipeline/01_estimators/lq_estimators/<descriptor>/`` so diagnostics still render before a run's
    artefacts are copied into the trained tree.

    Memoized per instance: several plots call this (and the CV accessors built on it) repeatedly, and
    it walks the estimator tree + stats every time otherwise.
    """
    if self._cv_dirs_cache is not None:
      return self._cv_dirs_cache
    dirs = {}
    try:
      for rpath in ResultsIterator(path=self.trained_path).iter_relpaths():
        d = os.path.join(self.trained_path, ESTIMATORS_SUBFOLDER, *rpath)
        if os.path.exists(os.path.join(d, "cv_report.json")):
          dirs[rpath[-1]] = d
    except Exception:
      # No manifest yet (e.g. mid-run) — fall through to the in-pipeline glob below.
      pass
    if not dirs:
      lq = os.path.join(self.trained_path, "pipeline", "01_estimators", "lq_estimators")
      if os.path.isdir(lq):
        for name in sorted(os.listdir(lq)):
          d = os.path.join(lq, name)
          if os.path.exists(os.path.join(d, "cv_report.json")):
            dirs[name] = d
    self._cv_dirs_cache = dirs
    return dirs

  def get_cv_stats(self):
    """Per-descriptor lazy-qsar CV stats from cv_report.json (of the trained model), best first."""
    stats = []
    for descriptor, d in self._cv_dirs().items():
      try:
        with open(os.path.join(d, "cv_report.json")) as fh:
          rep = json.load(fh)
        rep["descriptor"] = descriptor
        stats.append(rep)
      except Exception:
        continue
    stats.sort(key=lambda r: r.get("oof_auc") if r.get("oof_auc") is not None else -1, reverse=True)
    return stats

  def get_cv_oof(self, descriptor):
    """``(oof_proba, y)`` lists for a descriptor's out-of-fold predictions, or None."""
    d = self._cv_dirs().get(descriptor)
    if d is not None:
      f = os.path.join(d, "oof.csv")
      if os.path.exists(f):
        df = self._read_csv_cached(f)
        return list(df["oof_proba"]), list(df["y"])
    return None

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
      # No truth column in data.csv (predict-without-ground-truth): fall back to the pooled results,
      # but keep only the base "clf" prediction — not the derived columns (clf_bin/clf_ad/clf_rank/
      # clf_raw), which are not tasks and would break per-descriptor task lookups.
      df = self._read_pooled_results()
      derived = ("_bin", "_ad", "_rank", "_raw")
      return [c for c in df.columns if "clf" in c and not c.endswith(derived)]
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

  def get_pred_proba_clf_raw(self):
    """Pooled UNCALIBRATED score column (``clf_raw``) if the model was fit with lazy-qsar ≥ 3.4.2, else None."""
    df = self._read_pooled_results()
    return list(df["clf_raw"]) if "clf_raw" in df.columns else None

  def get_pred_ad_clf(self):
    """Per-compound applicability-domain in-domain fraction (``clf_ad``, 0..1) if present, else None.

    Written by the reliability pooler at predict time (1 = every descriptor in domain, 0 = fully
    out-of-domain). Read by exact column name so it is never confused with the base ``clf`` score."""
    df = self._read_pooled_results()
    return list(df["clf_ad"]) if "clf_ad" in df.columns else None

  def get_pred_rank_clf(self):
    """Per-compound weighted rank-reliability quantile (``clf_rank``) if present, else None."""
    df = self._read_pooled_results()
    return list(df["clf_rank"]) if "clf_rank" in df.columns else None

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

  @staticmethod
  def _humanize_axis(col):
    """Axis label from a projection column name: 'mw_x'→'MW', 'logp_y'→'LogP', 'umap_x'→'UMAP'."""
    label = col
    for suffix in ("_x", "_y"):
      if label.endswith(suffix):
        label = label[: -len(suffix)]
        break
    fancy = {"mw": "MW", "logp": "LogP", "umap": "UMAP", "tsne": "t-SNE", "pca": "PCA"}
    return fancy.get(label.lower(), label.replace("_", " "))

  def _read_projections(self, path):
    from zairachem.base.vars import PROJECTIONS_FILENAME, PROJECTIONS_MANIFEST_FILENAME

    data_dir = os.path.join(path, DATA_SUBFOLDER)
    manifest_path = os.path.join(data_dir, PROJECTIONS_MANIFEST_FILENAME)
    csv_path = os.path.join(data_dir, PROJECTIONS_FILENAME)
    if not (os.path.exists(manifest_path) and os.path.exists(csv_path)):
      return None, None
    with open(manifest_path) as f:
      manifest = json.load(f)
    return manifest, self._read_csv_cached(csv_path)

  def get_projections(self):
    """Projections for this run: ``[{name, title, x_label, y_label, xs, ys}, ...]`` (row-aligned)."""
    manifest, df = self._read_projections(self.path)
    if not manifest:
      return []
    out = []
    for p in manifest:
      x, y = p["x"], p["y"]
      if x not in df.columns or y not in df.columns:
        continue
      out.append({
        "name": p["name"],
        "title": p.get("title", p["name"]),
        "x_label": self._humanize_axis(x),
        "y_label": self._humanize_axis(y),
        "xs": list(df[x]),
        "ys": list(df[y]),
      })
    return out

  def get_projection_trained(self, name):
    """``(xs, ys)`` of the named projection from the trained model, or None (for predict overlays)."""
    manifest, df = self._read_projections(self.trained_path)
    if not manifest:
      return None
    for p in manifest:
      if p["name"] == name and p["x"] in df.columns and p["y"] in df.columns:
        return list(df[p["x"]]), list(df[p["y"]])
    return None

  def get_smiles(self):
    df = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    return list(df[SMILES_COLUMN])

  def get_original_smiles(self):
    raw_data = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, RAW_INPUT_FILENAME))
    with open(os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r") as f:
      schema = json.load(f)
    return list(raw_data[schema["smiles_column"]])

  def _top_binarizer(self, y, k):
    idxs = np.argsort(y)[::-1]
    idxs = idxs[:k]
    b = [0] * len(y)
    for i in idxs:
      b[i] = 1
    return b

  @staticmethod
  def _aligned_truth_pred(y_true, y_pred):
    """As float arrays with positions where ``y_true`` is NaN dropped (keeps truth↔pred aligned).

    At predict, ground truth may be partial (unlabelled compounds carry NaN after the left-merge);
    this yields the labelled subset for any (truth, prediction) pair. A no-op when truth is complete.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(yt)
    return yt[mask], yp[mask]

  def clf_truth_proba(self):
    """Labelled (y_true:int, y_proba:float) for the pooled classifier — NaN truth dropped."""
    yt, yp = self._aligned_truth_pred(self.get_actives_inactives(), self.get_pred_proba_clf())
    return yt.astype(int), yp

  def clf_truth_raw(self):
    """Labelled (y_true:int, raw_score:float) for the pooled UNCALIBRATED OOF, or None if not persisted."""
    raw = self.get_pred_proba_clf_raw()
    if raw is None:
      return None
    yt, yp = self._aligned_truth_pred(self.get_actives_inactives(), raw)
    return yt.astype(int), yp

  def clf_truth_binary(self):
    """Labelled (y_true:int, y_pred_binary:int) for the pooled classifier — NaN truth dropped."""
    yt, yp = self._aligned_truth_pred(self.get_actives_inactives(), self.get_pred_binary_clf())
    return yt.astype(int), yp.astype(int)

  def classification_performance_report(self, y_true_train, y_pred_train, y_true_test, y_pred_test):
    # Drop unlabelled positions (NaN truth) so partial predict-time ground truth is handled, and
    # ensure numpy arrays so the class-count comparisons below are elementwise.
    y_true_train, y_pred_train = self._aligned_truth_pred(y_true_train, y_pred_train)
    y_true_test, y_pred_test = self._aligned_truth_pred(y_true_test, y_pred_test)
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
    b_pred_test = (np.asarray(y_pred_test) >= cutoff).astype(int)
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

  def _dedupe_to_original_index(self):
    """``(n_original, {dedupe_row -> [original_rows]})`` for remapping result rows back to the input.

    Built once per fetcher and cached: ``map_to_original`` is called once per report column (10+), and
    the raw-input + mapping files don't change within a run."""
    cached = self._map_cache.get("index")
    if cached is None:
      n = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, RAW_INPUT_FILENAME)).shape[
        0
      ]
      dm = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, MAPPING_FILENAME))
      dm = dm[dm[MAPPING_DEDUPE_COLUMN].notnull()]
      u2o = collections.defaultdict(list)
      for v in dm[[MAPPING_ORIGINAL_COLUMN, MAPPING_DEDUPE_COLUMN]].values:
        u2o[int(v[1])] += [int(v[0])]
      cached = (n, u2o)
      self._map_cache["index"] = cached
    return cached

  def map_to_original(self, values):
    n, u2o = self._dedupe_to_original_index()
    mapped_values = [None] * n
    for i, v in enumerate(values):
      if i in u2o:
        for idx in u2o[i]:
          mapped_values[idx] = v
    return mapped_values
