import collections
import os

import joblib
import pandas as pd

from zairachem.base import ZairaBase
from zairachem.base.vars import (
  ESTIMATORS_SUBFOLDER,
  RESULTS_MAPPED_FILENAME,
  RESULTS_UNMAPPED_FILENAME,
  Y_HAT_FILE,
)
from zairachem.estimate.estimators.base import BaseOutcomeAssembler
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER


class IndividualOutcomeAssembler(BaseOutcomeAssembler):
  """Writer of one descriptor's lazyqsar predictions as results tables."""

  def __init__(self, path=None, model_id=None):
    BaseOutcomeAssembler.__init__(self, path=path)
    self.model_id = model_id

  def _get_y_hat(self):
    path = os.path.join(
      self.path,
      ESTIMATORS_SUBFOLDER,
      ESTIMATORS_FAMILY_SUBFOLDER,
      self.model_id,
      Y_HAT_FILE,
    )
    results = joblib.load(path)
    data = collections.OrderedDict()
    for c, r in results.items():
      r = r["main"]

      data[c] = r["y_hat"]
      if "b_hat" in r:
        data[c + "_bin"] = r["b_hat"]
      # Per-sample reliability signals for the pooler (see make_classification_report).
      if "r_hat" in r:
        data[c + "_rank"] = r["r_hat"]
      if "a_hat" in r:
        data[c + "_ad"] = r["a_hat"]
      # Uncalibrated OOF score (report raw lens); pooled the same way as clf into a clf_raw column.
      if "y_hat_raw" in r:
        data[c + "_raw"] = r["y_hat_raw"]
    return pd.DataFrame(data)

  def run(self):
    """Write the descriptor's ``y_hat.joblib`` predictions as unmapped and mapped results CSVs.

    The unmapped table is aligned to the run's compounds; the mapped one is remapped to the original
    input rows.
    """
    df_c = self._get_compounds()
    df_y = self._get_y_hat()
    df = pd.concat([df_c, df_y], axis=1)
    path = os.path.join(
      self.path,
      ESTIMATORS_SUBFOLDER,
      ESTIMATORS_FAMILY_SUBFOLDER,
      self.model_id,
      RESULTS_UNMAPPED_FILENAME,
    )
    df.to_csv(
      path,
      index=False,
    )
    mappings = self._get_mappings()
    df = self._remap(df, mappings)
    df.to_csv(
      os.path.join(
        self.path,
        ESTIMATORS_SUBFOLDER,
        ESTIMATORS_FAMILY_SUBFOLDER,
        self.model_id,
        RESULTS_MAPPED_FILENAME,
      ),
      index=False,
    )


class OutcomeAssembler(ZairaBase):
  """Writer of the results tables for every descriptor that produced predictions."""

  def __init__(self, path=None):
    ZairaBase.__init__(self)
    self.path = path

  def _get_model_ids(self):
    if self.path is None:
      path = self.get_output_dir()
    else:
      path = self.path
    if self.is_predict():
      path_trained = self.get_trained_dir()
    else:
      path_trained = path
    from zairachem.base.utils.descriptors import effective_descriptors

    model_ids = effective_descriptors(path_trained)
    model_ids_successful = []
    for model_id in model_ids:
      if os.path.isfile(
        os.path.join(
          path,
          ESTIMATORS_SUBFOLDER,
          ESTIMATORS_FAMILY_SUBFOLDER,
          model_id,
          "y_hat.joblib",
        )
      ):
        model_ids_successful += [model_id]
    return model_ids_successful

  def run(self):
    """Write the results tables of each used descriptor that has a ``y_hat.joblib``."""
    model_ids = self._get_model_ids()
    for model_id in model_ids:
      o = IndividualOutcomeAssembler(path=self.path, model_id=model_id)
      o.run()
