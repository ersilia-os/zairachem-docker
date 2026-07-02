"""Fold-aware estimator: trains on a fold's train slice and writes into the fold workspace."""

import os
import numpy as np

from zairachem.base.vars import ESTIMATORS_SUBFOLDER
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.lazy_qsar.estimate import Fitter


class HoldoutFitter(Fitter):
  """A :class:`~zairachem.estimate.estimators.lazy_qsar.estimate.Fitter` bound to one fold.

  Two behavioural overrides vs the production Fitter:

  1. The training rows are the fold's ``train_idxs`` (not all rows). Because they no longer cover the
     whole dataset, the parent's ``covers_all=False`` path predicts *every* row from the fold-trained
     model — so the held-out predictions are produced for free alongside the train ones.
  2. All artifacts (model, ``cv_report.json``, ``pool_signals.joblib``, …) are written under the fold
     workspace's ``model/estimators`` rather than the shared production model dir (the parent Fitter
     points ``trained_path`` at ``get_output_dir()``, which is the production run).
  """

  def __init__(self, path, model_id, train_idxs, batch_size=None):
    Fitter.__init__(self, path=path, model_id=model_id, is_simple=True, batch_size=batch_size)
    self.trained_path = os.path.join(path, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER)
    self._train_idxs = np.asarray(train_idxs)

  def get_train_indices(self, path):
    return self._train_idxs
