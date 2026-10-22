import collections

from zairachem.base.vars import DEFAULT_CUTOFF


def make_classification_report(
  y, y_proba, y_rank=None, y_ad=None, y_raw=None, threshold=DEFAULT_CUTOFF
):
  """Pack a descriptor's classification outputs in the format stored in ``y_hat.joblib``.

  Binary calls are ``y_proba >= threshold`` (the ``default`` cutoff). ``y_rank`` (OOF percentile), ``y_ad``
  (applicability-domain score) and ``y_raw`` (uncalibrated OOF score) are kept only when given.

  Returns
  -------
  collections.OrderedDict
    ``{"main": {...}}`` with ``y``, ``y_hat``, ``b_hat`` and, if given, ``r_hat``, ``a_hat`` and
    ``y_hat_raw``.
  """
  y_hat = y_proba
  b_hat = (y_hat >= threshold).astype(int)
  results = collections.OrderedDict()
  main = {"idxs": None, "y": y, "y_hat": y_hat, "b_hat": b_hat}
  # Per-sample reliability signals consumed by the reliability pooler (``r_hat`` is lazy-qsar's OOF
  # percentile, carried in the ``_rank`` columns). Optional: the pooler
  # degrades gracefully when a descriptor omits them (e.g. tiny datasets with no OOF).
  if y_rank is not None:
    main["r_hat"] = y_rank
  if y_ad is not None:
    main["a_hat"] = y_ad
  # Uncalibrated OOF score, carried through to a pooled clf_raw column for the report's raw lens.
  if y_raw is not None:
    main["y_hat_raw"] = y_raw
  results["main"] = main
  return results


def make_regression_report(y, y_pred, y_rank=None, y_ad=None):
  """Pack a descriptor's regression outputs in the format stored in ``y_hat.joblib``.

  ``y_pred`` is the predicted target in its own units. There is no binary call and no
  uncalibrated score: lazy-qsar's regression ensemble has neither. ``y_rank`` (the out-of-fold
  percentile, which indexes the pooler's reliability curve) and ``y_ad`` (the
  applicability-domain score) are kept only when given.

  Returns
  -------
  collections.OrderedDict
    ``{"main": {...}}`` with ``y``, ``y_hat`` and, if given, ``r_hat`` and ``a_hat``.
  """
  results = collections.OrderedDict()
  main = {"idxs": None, "y": y, "y_hat": y_pred}
  if y_rank is not None:
    main["r_hat"] = y_rank
  if y_ad is not None:
    main["a_hat"] = y_ad
  results["main"] = main
  return results
