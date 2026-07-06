import os
import pandas as pd
import collections
from rdkit import Chem

from zairachem.report import BaseTable
from zairachem.report.fetcher import ResultsFetcher

from zairachem.base.vars import (
  OUTPUT_TABLE_FILENAME,
  PERFORMANCE_TABLE_FILENAME,
  REPORT_SUBFOLDER,
)


class PerformanceTable(BaseTable, ResultsFetcher):
  """Writes ``report/performance_table.csv`` — per-model and pooled performance metrics.

  One row per descriptor estimator plus a ``pooled`` row, each with the classification metrics
  (regression is a TODO). Skips quietly when there is no/single-class labelled truth (e.g. predict
  without ground truth)."""

  def __init__(self, path):
    BaseTable.__init__(self, path=path)
    ResultsFetcher.__init__(self, path=path)
    self.is_clf = self.has_clf_data()

  def _individual_performances(self):
    if self.is_clf:
      tasks = self.get_clf_tasks()
    else:
      tasks = self.get_reg_tasks()
    task = tasks[0]
    df_te = self._read_individual_estimator_results(task)
    df_tr = self._read_individual_estimator_results_train(task)
    columns = list(df_te.columns)
    for col in columns:
      y_pred_test = list(df_te[col])
      y_pred_train = list(df_tr[col])
      if self.is_clf:
        y_true_train = list(self.get_actives_inactives_trained())
        y_true_test = list(self.get_actives_inactives())
        data = self.classification_performance_report(
          y_true_train, y_pred_train, y_true_test, y_pred_test
        )
      else:
        # TODO
        data = self.regression_performance_report(
          y_true_train, y_pred_train, y_true_test, y_pred_test
        )
      yield (col, data)

  def _general_performance(self):
    if not self.is_clf:
      return None  # TODO regression
    # Guard: validation needs at least two classes among the labelled (current-run) compounds.
    # At predict with no/partial/single-class truth this is degenerate — skip metrics, don't crash.
    yt_labeled = self.clf_truth_proba()[0]
    if len(yt_labeled) == 0 or len(set(yt_labeled.tolist())) < 2:
      from zairachem.base.utils.logging import logger

      logger.warning(
        "[report] Ground-truth labels are absent or single-class — skipping performance metrics."
      )
      return None
    return self.classification_performance_report(
      list(self.get_actives_inactives_trained()),
      list(self.get_pred_proba_clf_trained()),
      list(self.get_actives_inactives()),
      list(self.get_pred_proba_clf()),
    )

  def run(self):
    """Compute every model's metrics and write the performance table CSV (no-op if metrics skip)."""
    data = collections.defaultdict(list)
    d = self._general_performance()
    data["model"] += ["pooled"]
    if d is None:
      return
    for k, v in d.items():
      data[k] += [v]
    for col, d_ in self._individual_performances():
      data["model"] += [col]
      for k, v in d_.items():
        data[k] += [v]
    data = pd.DataFrame(data)
    data.to_csv(
      os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME),
      index=False,
    )


class OutputTable(BaseTable, ResultsFetcher):
  """Writes ``report/output_table.csv`` — the per-compound predictions, mapped back to the input rows.

  Columns: input SMILES, InChIKey, standardized SMILES, true value (if known), pooled prediction, each
  descriptor's ensemble prediction, and the projection (x, y) pairs. All values are remapped from the
  deduplicated run rows to the original input order via :meth:`map_to_original`."""

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
    if self.is_clf:
      values = self.get_true_clf()
      return self.map_to_original(values)
    else:
      return None

  def _get_pred_value_column(self):
    try:
      values = self.get_pred_proba_clf()
      return self.map_to_original(values)
      # values = self.get_pred_reg_trans()
    except Exception:
      return None

  def _get_ensemble_predictions_columns(self):
    try:
      tasks = self.get_clf_tasks()
    except Exception:
      tasks = self.get_reg_tasks()
    task = tasks[0]
    df = self._read_individual_estimator_results(task)
    columns = list(df.columns)
    for col in columns:
      v = list(df[col])
      v = self.map_to_original(v)
      yield (col, v)

  def _get_manifolds_columns(self):
    # Every projection (always at least MW-vs-LogP) contributes its x/y columns to the output table.
    for proj in self.get_projections():
      for axis, vals in ((proj["x_label"], proj["xs"]), (proj["y_label"], proj["ys"])):
        yield (f"{proj['name']}-{axis}", self.map_to_original(list(vals)))

  def run(self):
    """Assemble all output columns (remapped to input order) and write the output table CSV."""
    data = {}
    data["input-smiles"] = self._get_input_smiles_column()
    data["inchikey"] = self._get_inchikey_column()
    data["smiles"] = self._get_smiles_column()
    data["true-value"] = self._get_true_value_column()
    data["pred-value"] = self._get_pred_value_column()
    if self.is_predict():
      # Predict-only enrichments (feed the hitlist): the binary call and, when the pooler emitted
      # them, the per-compound applicability-domain fraction + rank-reliability quantile. Gated on
      # is_predict() so a fit run's output_table.csv is unchanged.
      for key, getter in (
        ("clf_bin", self.get_pred_binary_clf),
        ("ad", self.get_pred_ad_clf),
        ("rank", self.get_pred_rank_clf),
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
