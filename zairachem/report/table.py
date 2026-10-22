import collections
import os

import pandas as pd
from rdkit import Chem

from zairachem.base.vars import (
  CUTOFF_BALANCED,
  CUTOFF_DEFAULT,
  CUTOFF_KAPPA,
  CUTOFF_SCREENING,
  DEFAULT_CUTOFF,
  OUTPUT_DESCRIPTOR_PREFIX,
  OUTPUT_TABLE_FILENAME,
  PERFORMANCE_TABLE_FILENAME,
  REPORT_SUBFOLDER,
)
from zairachem.report import BaseTable
from zairachem.report.fetcher import ResultsFetcher


class PerformanceTable(BaseTable, ResultsFetcher):
  """Writes ``report/performance_table.csv``: the pooled and per-descriptor metrics, per cutoff.

  For classification, one row per (model, cutoff): the pooled model at the ``screening``, ``kappa``
  and ``default`` cutoffs, each descriptor at its ``balanced`` cutoff and at ``default`` (see
  ``results/cutoffs.json``). For regression there are no cutoffs, so it is one row per model — the
  pooled model and each descriptor — with R2, RMSE, MAE and the two correlations. Skips quietly
  when there is no usable labelled truth (e.g. predict without ground truth).
  """

  def __init__(self, path):
    BaseTable.__init__(self, path=path)
    ResultsFetcher.__init__(self, path=path)
    self.is_clf = self.has_clf_data()

  def _rows(self):
    from zairachem.pool.cutoffs import read_cutoffs

    cutoffs = read_cutoffs(self.path) or {}

    def get(name, key):
      return (cutoffs.get(name) or {}).get(key)

    y = list(self.get_actives_inactives())
    pooled = list(self.get_pred_proba_clf())
    for name in (CUTOFF_SCREENING, CUTOFF_KAPPA, CUTOFF_DEFAULT):
      yield (
        "pooled",
        self.classification_performance_report(
          y, pooled, name, get(name, "proba"), get(name, "rank")
        ),
      )
    balanced = get(CUTOFF_BALANCED, "proba") or {}
    balanced_rank = get(CUTOFF_BALANCED, "rank") or {}
    default_rank = get(CUTOFF_DEFAULT, "rank_by_descriptor") or {}
    df = self._read_individual_estimator_results(self.get_clf_tasks()[0])
    for col in df.columns:
      p = list(df[col])
      d = _descriptor_of(col)
      yield (
        col,
        self.classification_performance_report(
          y, p, CUTOFF_BALANCED, balanced.get(d), balanced_rank.get(d)
        ),
      )
      yield (
        col,
        self.classification_performance_report(
          y, p, CUTOFF_DEFAULT, DEFAULT_CUTOFF, default_rank.get(d)
        ),
      )

  def _reg_rows(self):
    y = list(self.get_true_reg())
    yield ("pooled", self.regression_performance_report(y, self.get_pred_value_reg()))
    df = self._read_individual_estimator_results(self.get_reg_tasks()[0])
    for col in df.columns:
      yield (col, self.regression_performance_report(y, list(df[col])))

  def run(self):
    """Compute every model's metrics and write the performance table CSV (no-op if metrics skip)."""
    from zairachem.base.utils.logging import logger

    if self.is_clf:
      # Validation needs at least two classes among the labelled (current-run) compounds. At
      # predict with no, partial or single-class truth this is degenerate: skip, don't crash.
      yt_labeled = self.clf_truth_proba()[0]
      degenerate = len(yt_labeled) == 0 or len(set(yt_labeled.tolist())) < 2
    else:
      # The regression analogue of "both classes present": at least two distinct observed values.
      try:
        yt = pd.to_numeric(pd.Series(self.get_true_reg()), errors="coerce").dropna()
      except Exception:
        yt = pd.Series(dtype=float)
      degenerate = len(yt) < 2 or yt.nunique() < 2
    if degenerate:
      logger.warning(
        "[report] Ground-truth labels are absent or degenerate — skipping performance metrics."
      )
      return
    data = collections.defaultdict(list)
    for model, row in self._rows() if self.is_clf else self._reg_rows():
      data["model"].append(model)
      for k, v in row.items():
        data[k].append(v)
    pd.DataFrame(data).to_csv(
      os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME), index=False
    )


def _descriptor_of(column):
  """Descriptor id of a per-descriptor results column (``<family>-<eos_id>`` -> ``<eos_id>``)."""
  return column.split("-")[-1]


class OutputTable(BaseTable, ResultsFetcher):
  """Writes ``report/output_table.csv`` — the per-compound predictions, mapped back to the input rows.

  Columns: ``input_smiles``, ``inchikey``, ``smiles``, ``label``, ``proba``,
  at predict ``call``, ``rank``, ``similarity`` and ``spread``, then ``proba_<eos>`` per descriptor
  and both axes of every projection. All values are remapped from the deduplicated run rows to the
  original input order via :meth:`map_to_original`."""

  def __init__(self, path):
    BaseTable.__init__(self, path=path)
    ResultsFetcher.__init__(self, path=path)
    self.is_clf = self.has_clf_data()

  def __smiles_to_inchikey(self, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      raise Exception(
        "The SMILES string: %s is not valid or could not be converted to an InChIKey" % smiles
      )
    inchi = Chem.rdinchi.MolToInchi(mol)[0]
    if inchi is None:
      raise Exception("Could not obtain InChI")
    inchikey = Chem.rdinchi.InchiToInchiKey(inchi)
    return inchikey

  def _get_input_smiles_column(self):
    return self.get_original_smiles()

  def _get_inchikey_column(self):
    inchikeys = [self.__smiles_to_inchikey(smiles) for smiles in self.get_smiles()]
    return self.map_to_original(inchikeys)

  def _get_smiles_column(self):
    smiles = self.get_smiles()
    return self.map_to_original(smiles)

  def _get_true_value_column(self):
    """The observed truth per compound: the 0/1 label, or the target value."""
    try:
      values = self.get_true_clf() if self.is_clf else self.get_true_reg()
    except Exception:
      return None
    return self.map_to_original(values)

  def _get_pred_value_column(self):
    """The pooled prediction per compound: the probability, or the value in the target's units."""
    try:
      values = self.get_pred_proba_clf() if self.is_clf else self.get_pred_value_reg()
      return None if values is None else self.map_to_original(values)
    except Exception:
      return None

  def _get_ensemble_predictions_columns(self):
    task = (self.get_clf_tasks() if self.is_clf else self.get_reg_tasks())[0]
    df = self._read_individual_estimator_results(task)
    columns = list(df.columns)
    for col in columns:
      v = list(df[col])
      v = self.map_to_original(v)
      # "lq_estimators-<eos_id>" -> "proba_<eos_id>" / "value_<eos_id>"
      prefix = OUTPUT_DESCRIPTOR_PREFIX["classification" if self.is_clf else "regression"]
      yield (prefix + col.split("-")[-1], v)

  def _get_manifolds_columns(self):
    # Every projection (always at least MW-vs-LogP) contributes both axes, named after its own
    # columns: "mw_x"/"logp_y" -> "mw"/"logp" (two different quantities), "pca_x"/"pca_y" kept as
    # they are. Display labels are not unique per axis, so they must not name columns.
    for proj in self.get_projections():
      x, y = proj["x"], proj["y"]
      if x.rsplit("_", 1)[0] != y.rsplit("_", 1)[0]:
        x, y = x.rsplit("_", 1)[0], y.rsplit("_", 1)[0]
      yield (x, self.map_to_original(list(proj["xs"])))
      yield (y, self.map_to_original(list(proj["ys"])))

  def run(self):
    """Assemble all output columns (remapped to input order) and write the output table CSV."""
    data = {}
    data["input_smiles"] = self._get_input_smiles_column()
    data["inchikey"] = self._get_inchikey_column()
    data["smiles"] = self._get_smiles_column()
    data["label"] = self._get_true_value_column()
    data["proba" if self.is_clf else "value"] = self._get_pred_value_column()
    if not self.is_clf:
      lower, upper = self.get_pred_interval_reg()
      if lower is not None:
        # The conformal interval: wider where the compound is unlike the training set.
        data["lower"] = self.map_to_original(lower)
        data["upper"] = self.map_to_original(upper)
    if self.is_predict():
      # Predict-only enrichments (feed the hitlist): for classification the binary call and the
      # pooled reference rank, and for both the Tanimoto similarity to the training set and the
      # spread of the per-descriptor predictions. Gated on is_predict() so a fit run's
      # output_table.csv is unchanged.
      clf_only = (
        (("call", self.get_pred_binary_clf), ("rank", self.get_pred_rank_clf))
        if self.is_clf
        else ()
      )
      for key, getter in (
        *clf_only,
        ("similarity", self.get_train_similarity),
        ("spread", self.get_descriptor_std),
      ):
        try:
          vals = getter()
        except Exception:
          vals = None
        if vals is not None:
          data[key] = self.map_to_original(list(vals))
    for k, v in self._get_ensemble_predictions_columns():
      data[k] = v
    for k, v in self._get_manifolds_columns():
      data[k] = v
    data = pd.DataFrame(data)
    data.to_csv(
      os.path.join(self.path, REPORT_SUBFOLDER, OUTPUT_TABLE_FILENAME),
      index=False,
    )
