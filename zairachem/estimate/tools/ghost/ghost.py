import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

import ghostml  # noqa: E402 (needs ROOT on sys.path first)


class GhostLight(object):
  """Decision-threshold optimizer based on GHOST."""

  def __init__(self):
    pass

  def _get_class_balance(self, y):
    return np.sum(y) / len(y)

  def get_threshold(self, y, y_hat):
    """Return the GHOST threshold that maximizes Cohen's kappa for scores ``y_hat`` on ``y``.

    Candidates run from 0.05 in steps of 0.05 to below the larger of the positive fraction and 0.6.
    """
    max_prop = np.max([self._get_class_balance(y), 0.6])
    thresholds = np.round(np.arange(0.05, max_prop, 0.05), 2)  # TODO revise intervals
    threshold = ghostml.optimize_threshold_from_predictions(
      y, y_hat, thresholds, ThOpt_metrics="Kappa"
    )
    return threshold
