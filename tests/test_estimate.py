"""The estimate step's per-descriptor scoring helpers, away from a trained model."""

import numpy as np


def test_the_percentile_is_read_off_the_probability_already_computed():
  """``_percentile_from_proba`` must agree with the second model pass it replaces."""
  from lazyqsar.utils.ranking import prepare_knots, rank_from_knots

  from zairachem.estimate.estimators.lazy_qsar.estimate import _percentile_from_proba

  knots = np.linspace(0.0, 1.0, 41)
  proba = np.array([0.02, 0.31, 0.5, 0.77, 0.99])

  class _Inner:
    oof_percentile_knots_ = knots

  class _Model:
    _model = _Inner()

    def _oof_percentile(self, X):
      raise AssertionError("a second pass through the model was taken")

  got = _percentile_from_proba(_Model(), proba, X=None)
  np.testing.assert_array_equal(got, rank_from_knots(proba, prepared=prepare_knots(knots)))


def test_a_model_saved_without_the_knots_still_gets_its_percentile():
  """Older models carry no out-of-fold knots; the helper falls back to the model's own pass."""
  from zairachem.estimate.estimators.lazy_qsar.estimate import _percentile_from_proba

  expected = np.array([0.1, 0.9])

  class _Inner:
    oof_percentile_knots_ = None

  class _Model:
    _model = _Inner()

    def _oof_percentile(self, X):
      assert X is not None  # the fallback is the only path that needs the matrix
      return np.column_stack([1.0 - expected, expected])

  got = _percentile_from_proba(_Model(), np.array([0.2, 0.8]), X=np.zeros((2, 3)))
  np.testing.assert_array_equal(got, expected)
