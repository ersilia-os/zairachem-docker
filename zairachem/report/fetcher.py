import collections
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  auc,
  balanced_accuracy_score,
  confusion_matrix,
  f1_score,
  matthews_corrcoef,
  precision_recall_curve,
  precision_score,
  recall_score,
  roc_curve,
)

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  ESTIMATORS_SUBFOLDER,
  INPUT_SCHEMA_FILENAME,
  MAPPING_DEDUPE_COLUMN,
  MAPPING_FILENAME,
  MAPPING_ORIGINAL_COLUMN,
  POOL_SUBFOLDER,
  RAW_INPUT_FILENAME,
  RESULTS_UNMAPPED_FILENAME,
  SMILES_COLUMN,
)
from zairachem.report.cv import sort_best_first
from zairachem.report.utils import ResultsIterator

#: The raw input as setup wrote it. ``base.vars`` holds the stem because setup appends whatever
#: extension the user's file had; by the time the report reads it, it is always a CSV.
RAW_INPUT_CSV = RAW_INPUT_FILENAME + ".csv"

#: Process-level CSV read cache shared by every ResultsFetcher, keyed by (abspath, mtime) so a
#: rewritten file is re-read. Lets the report's per-plot fetchers avoid re-reading the same artifacts.
_CSV_CACHE = {}


class ResultsFetcher(ZairaBase):
  """Reader of a run's data, pooled and per-descriptor results, CV artefacts and projections.

  Used by every report plot and table. ``path`` is the run folder (fit or predict) and
  ``trained_path`` the trained model folder, whose fit-time results serve as reference at predict.
  """

  def __init__(self, path):
    ZairaBase.__init__(self)
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
    except Exception as e:
      # No manifest yet (e.g. mid-run) — fall through to the in-pipeline glob below.
      logger.debug(f"[report] no results manifest, globbing CV dirs instead: {e}")
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
    return sort_best_first(stats)

  def _cv_oof(self, descriptor, column):
    """``(column, y)`` lists from a descriptor's ``oof.csv``, or None when it holds neither.

    Each run writes one of the two: ``oof_proba`` for a classifier, ``oof_value`` for a regressor.
    Asking for the one this run did not write returns None, which is how each task's
    cross-validation plots skip the other's.
    """
    d = self._cv_dirs().get(descriptor)
    if d is None:
      return None
    f = os.path.join(d, "oof.csv")
    if not os.path.exists(f):
      return None
    df = self._read_csv_cached(f)
    return (list(df[column]), list(df["y"])) if column in df.columns else None

  def get_cv_oof(self, descriptor):
    """``(oof_proba, y)`` lists for a classification descriptor's out-of-fold predictions, or None."""
    return self._cv_oof(descriptor, "oof_proba")

  def get_cv_oof_values(self, descriptor):
    """``(oof_value, y)`` lists for a regression descriptor's out-of-fold predictions, or None."""
    return self._cv_oof(descriptor, "oof_value")

  def get_tasks(self):
    """Return the ``clf``/``reg`` task column names of the run's ``data.csv``."""
    df = self._read_data()
    tasks = [c for c in list(df.columns) if ("clf" in c or "reg" in c)]
    return tasks

  def get_reg_tasks(self):
    """Return the regression task columns, read from the pooled results (``reg``).

    The truth column in ``data.csv`` is ``val``, but the per-descriptor and pooled *prediction*
    columns are named ``reg``, and it is those the per-descriptor lookups key on. The derived
    interval columns (``reg_lower``, ``reg_upper``) are not tasks.
    """
    df = self._read_pooled_results()
    return [c for c in list(df.columns) if "reg" in c and not c.endswith(("_lower", "_upper"))]

  def get_clf_tasks(self, data=None):
    """Return the classification task columns, falling back to the pooled ``clf`` column.

    Columns are read from ``data`` if given, else from the run's ``data.csv``. The fallback covers
    predict runs without ground truth and excludes the derived pooled columns (``clf_bin``,
    ``clf_rank``, ``clf_raw``, ``train_similarity``, ``descriptor_std``).
    """
    if data is None:
      df = self._read_data()
    else:
      df = data
    tasks = [c for c in list(df.columns) if "clf" in c]
    if len(tasks) == 0:
      # No truth column in data.csv (predict-without-ground-truth): fall back to the pooled results,
      # but keep only the base "clf" prediction — not the derived columns (clf_bin/clf_rank/
      # clf_raw), which are not tasks and would break per-descriptor task lookups.
      df = self._read_pooled_results()
      derived = ("_bin", "_ad", "_rank", "_raw")
      return [c for c in df.columns if "clf" in c and not c.endswith(derived)]
    return tasks

  def get_actives_inactives(self):
    """Return the binary truth labels (``bin`` column) of the run's ``data.csv``."""
    df = self._read_data()
    return list(df[self.clf_task])

  def get_actives_inactives_trained(self):
    """Return the binary truth labels (``bin`` column) of the trained model's ``data.csv``."""
    df = self._read_data_train()
    return list(df[self.clf_task])

  def get_raw(self):
    """Return the raw regression target (``val`` column) of the run's ``data.csv``."""
    df = self._read_data()
    return list(df[self.reg_task])

  def get_true_clf(self):
    """Return the binary truth labels; alias of :meth:`get_actives_inactives`."""
    return self.get_actives_inactives()

  def get_pred_binary_clf(self):
    """Return the run's pooled binary classification calls (``clf`` + ``bin`` column), or None."""
    df = self._read_pooled_results()
    for c in list(df.columns):
      if "clf" in c and "bin" in c:
        return list(df[c])

  def get_pred_proba_clf(self):
    """Return the run's pooled classification probability (first non-``bin`` ``clf`` column)."""
    df = self._read_pooled_results()
    for c in list(df.columns):
      if "clf" in c and "bin" not in c:
        return list(df[c])

  def get_pred_proba_clf_trained(self):
    """Return the trained model's pooled classification probability (the fit-time OOF score)."""
    df = self._read_pooled_results_train()
    for c in list(df.columns):
      if "clf" in c and "bin" not in c:
        return list(df[c])

  def get_pred_proba_clf_raw(self):
    """Pooled UNCALIBRATED score column (``clf_raw``) if the model was fit with lazy-qsar ≥ 3.4.2, else None."""
    df = self._read_pooled_results()
    return list(df["clf_raw"]) if "clf_raw" in df.columns else None

  def get_train_similarity(self):
    """Per-compound Tanimoto similarity to the training set (``train_similarity``), else None.

    Written by the reliability pooler at predict time: the mean Tanimoto similarity to the 5 nearest
    training molecules (Morgan, radius 2), so 1 is a training-like neighbourhood."""
    df = self._read_pooled_results()
    return list(df["train_similarity"]) if "train_similarity" in df.columns else None

  def get_descriptor_std(self):
    """Per-compound std of the per-descriptor probabilities (``descriptor_std``), else None."""
    df = self._read_pooled_results()
    return list(df["descriptor_std"]) if "descriptor_std" in df.columns else None

  def get_pred_rank_clf(self):
    """Per-compound pooled rank against the rank-reference library (``clf_rank``), else None."""
    df = self._read_pooled_results()
    return list(df["clf_rank"]) if "clf_rank" in df.columns else None

  def get_pred_value_reg(self):
    """Per-compound pooled predicted value (``reg``), in the target's own units, else None."""
    df = self._read_pooled_results()
    return list(df["reg"]) if "reg" in df.columns else None

  def get_pred_value_reg_trained(self):
    """The trained model's pooled predicted values (its out-of-fold predictions), else None."""
    df = self._read_pooled_results_train()
    return list(df["reg"]) if "reg" in df.columns else None

  def get_pred_interval_reg(self):
    """``(lower, upper)`` of the pooled conformal prediction interval, else ``(None, None)``."""
    df = self._read_pooled_results()
    if "reg_lower" not in df.columns:
      return None, None
    return list(df["reg_lower"]), list(df["reg_upper"])

  def get_true_reg(self):
    """Return the observed target values (the ``val`` column) of the run's ``data.csv``."""
    return list(self._read_data()[self.reg_task])

  def get_true_reg_trained(self):
    """Return the observed target values of the trained model's ``data.csv``."""
    return list(self._read_data_train()[self.reg_task])

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
        "x": x,
        "y": y,
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
    """Return the SMILES of the run's deduplicated ``data.csv`` rows."""
    df = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    return list(df[SMILES_COLUMN])

  def get_original_smiles(self):
    """Return the SMILES as given in the raw input file (the schema's SMILES column)."""
    raw_data = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, RAW_INPUT_CSV))
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
    """As float arrays with any position where either side is NaN dropped (keeps them aligned).

    At predict, ground truth may be partial (unlabelled compounds carry NaN after the left-merge);
    this yields the labelled subset for any (truth, prediction) pair. A prediction can also be NaN
    -- a compound every descriptor failed on -- and scoring it would turn the whole row of metrics
    into NaN rather than reporting the compounds that did score. ``num_compounds`` then says how
    many were used. A no-op when both sides are complete.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(yt) & ~np.isnan(yp)
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

  def classification_performance_report(
    self, y_true, y_pred, cutoff_name, cutoff, cutoff_rank=None
  ):
    """Classification metrics of scores ``y_pred`` against labels ``y_true`` at a named cutoff.

    Unlabelled (NaN truth) positions are dropped first. The ordered dict holds the cutoff (name,
    probability and the same cutoff as a rank, the scale of the score), class counts, AUROC, AUPR, the confusion-matrix counts and the thresholded
    metrics (accuracy, balanced accuracy, precision, recall, F1, MCC) at ``y_pred >= cutoff``, and
    precision among the top 1/5/10/50/100 scored compounds. With no cutoff (e.g. no rank
    reference for ``screening``) the thresholded metrics are NaN.
    """
    y_true, y_pred = self._aligned_truth_pred(y_true, y_pred)
    y_pred = np.asarray(y_pred, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    data = collections.OrderedDict()
    data["cutoff"] = cutoff_name
    data["cutoff_value"] = cutoff
    data["cutoff_rank"] = cutoff_rank
    data["num_compounds"] = len(y_true)
    data["num_0"] = int(np.sum(y_true == 0))
    data["num_1"] = int(np.sum(y_true == 1))
    data["auroc"] = auc(fpr, tpr)
    data["aupr"] = auc(rec, prec)
    thresholded = ("tp", "tn", "fp", "fn", "accuracy", "balanced_accuracy", "precision", "recall")
    if cutoff is None:
      data.update({k: np.nan for k in (*thresholded, "f1_score", "mcc")})
    else:
      b_pred = (y_pred >= cutoff).astype(int)
      tn, fp, fn, tp = confusion_matrix(y_true, b_pred, labels=[0, 1]).ravel()
      data.update(tp=tp, tn=tn, fp=fp, fn=fn)
      data["accuracy"] = accuracy_score(y_true, b_pred)
      data["balanced_accuracy"] = balanced_accuracy_score(y_true, b_pred)
      data["precision"] = precision_score(y_true, b_pred, zero_division=0)
      data["recall"] = recall_score(y_true, b_pred, zero_division=0)
      data["f1_score"] = f1_score(y_true, b_pred, zero_division=0)
      data["mcc"] = matthews_corrcoef(y_true, b_pred)
    for k in (1, 5, 10, 50, 100):
      data[f"precision_at_{k}"] = precision_score(
        y_true, self._top_binarizer(y_pred, k), zero_division=0
      )
    return data

  def regression_performance_report(self, y_true, y_pred):
    """Regression metrics of predictions ``y_pred`` against observed values ``y_true``.

    Unlabelled (NaN truth) positions are dropped first. The ordered dict holds the compound count,
    R2, RMSE, MAE and the Pearson and Spearman correlations. Spearman matters in its own right:
    ranking compounds correctly is usually what a screen is for, and a model can order them well
    while its units are off.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true, y_pred = self._aligned_truth_pred(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    data = collections.OrderedDict()
    data["num_compounds"] = len(y_true)
    if len(y_true) < 2 or np.ptp(y_true) == 0:
      # R2 and the correlations are undefined against a constant or a single value.
      data.update({k: np.nan for k in ("r2", "rmse", "mae", "pearson", "spearman")})
      return data
    data["r2"] = float(r2_score(y_true, y_pred))
    data["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    data["mae"] = float(mean_absolute_error(y_true, y_pred))
    constant = np.ptp(y_pred) == 0
    data["pearson"] = np.nan if constant else float(pearsonr(y_true, y_pred).statistic)
    data["spearman"] = np.nan if constant else float(spearmanr(y_true, y_pred).statistic)
    return data

  def _dedupe_to_original_index(self):
    """``(n_original, {dedupe_row -> [original_rows]})`` for remapping result rows back to the input.

    Built once per fetcher and cached: ``map_to_original`` is called once per report column (10+), and
    the raw-input + mapping files don't change within a run."""
    cached = self._map_cache.get("index")
    if cached is None:
      n = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, RAW_INPUT_CSV)).shape[0]
      dm = self._read_csv_cached(os.path.join(self.path, DATA_SUBFOLDER, MAPPING_FILENAME))
      dm = dm[dm[MAPPING_DEDUPE_COLUMN].notnull()]
      u2o = collections.defaultdict(list)
      for v in dm[[MAPPING_ORIGINAL_COLUMN, MAPPING_DEDUPE_COLUMN]].values:
        u2o[int(v[1])] += [int(v[0])]
      cached = (n, u2o)
      self._map_cache["index"] = cached
    return cached

  def map_to_original(self, values):
    """Map per-row values of the deduplicated run back to the original input rows.

    Returns a list as long as the raw input; rows dropped during deduplication are None.
    """
    n, u2o = self._dedupe_to_original_index()
    mapped_values = [None] * n
    for i, v in enumerate(values):
      if i in u2o:
        for idx in u2o[i]:
          mapped_values[idx] = v
    return mapped_values
