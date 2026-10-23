import collections
import contextlib
import gc
import json
import os
import time

import joblib
import numpy as np
from lazyqsar.agnostic import LazyClassifier, LazyRegressor

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE, open_h5
from zairachem.base.vars import (
  CV_SKILL,
  ESTIMATORS_SUBFOLDER,
  METADATA_SUBFOLDER,
  PARAMETERS_FILE,
  RANDOM_SEED,
  RANK_REFERENCE_SIGNALS_FILENAME,
  Y_HAT_FILE,
)
from zairachem.estimate.estimators.base import BaseEstimatorIndividual
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.lazy_qsar.utils import (
  make_classification_report,
  make_regression_report,
)

# lazy-qsar prints a rich directory tree of the saved model folder on every ``save()`` — it is the
# one piece of lazy-qsar console output not gated behind its ``--verbose`` flag, and it is pure
# noise inside our pipeline (we render our own progress). Silence just that method on the shared
# lazy-qsar logger singleton; everything else lazy-qsar emits is already quiet by default.
try:
  from lazyqsar.utils.logging import logger as _lq_logger

  _lq_logger.dir_tree = lambda *a, **k: None
except (ImportError, AttributeError) as e:
  logger.debug(f"[lazyqsar] could not silence lazy-qsar's directory tree: {e}")


def _percentile(model, X):
  """lazy-qsar's reliability signal: position within the model's own out-of-fold distribution.

  This is what lazy-qsar's own ensemble weights descriptors by (``LazyClassifierQSAR``); the
  reference-library ``predict_rank`` is an output, not a weight, and is built at pool level.
  """
  return model._oof_percentile(X=X)[:, 1]


def _seed_global_rngs(seed=RANDOM_SEED):
  """Reset Python's and NumPy's global random state before a descriptor is trained.

  Some of the training stack draws from the global generators, so without this a descriptor's model
  would depend on what ran before it in the same process (its position in the loop, the descriptors
  trained before it, parallel or serial training).
  """
  import random

  random.seed(seed)
  np.random.seed(seed)


def _percentile_from_proba(model, proba, X):
  """:func:`_percentile` from already computed probabilities, saving a second model pass.

  lazy-qsar's percentile is its out-of-fold ECDF evaluated at the same calibrated probability
  ``predict_proba`` returns; without those knots (older models) it is recomputed from ``X``.
  """
  inner = getattr(model, "_model", None)
  knots = getattr(inner, "oof_percentile_knots_", None)
  if knots is None or not len(knots):
    return _percentile(model, X)
  from lazyqsar.utils.ranking import prepare_knots, rank_from_knots

  prepared = getattr(inner, "_oof_percentile_prepared_", None) or prepare_knots(knots)
  return rank_from_knots(proba, prepared=prepared)


def _rank_or_warn(ranks, model_id, stage):
  """``ranks`` when any is finite, else None after one warning.

  The per-chunk calls swallow a failing ``_oof_percentile`` so proba still flows, which leaves this
  as the one place it is reported.
  """
  if np.isfinite(ranks).any():
    return ranks
  logger.warning(
    f"[lazyqsar:{stage}] {model_id}: OOF percentile unavailable; the pooler runs without it"
  )
  return None


def _float_or_none(v):
  """``float(v)``, or None when it cannot be converted."""
  try:
    return float(v)
  except Exception:
    return None


def _summed_timing(inner):
  """Per-phase wall-clock seconds summed across the inner model's batches."""
  timing = {}
  for bm in getattr(inner, "models", []) or []:
    t = getattr(bm, "timing_", None)
    if isinstance(t, dict):
      for k, v in t.items():
        try:
          timing[k] = round(timing.get(k, 0.0) + float(v), 3)
        except (TypeError, ValueError):
          continue
  return timing


def _cv_report(model, y_train):
  """Build the ``cv_report.json`` dict from a fitted lazy-qsar classifier."""
  inner = getattr(model, "_model", None)
  y_arr = np.asarray(y_train)
  report = {
    "oof_auc": _float_or_none(model.oof_auc_),
    "train_auc": _float_or_none(model.train_auc_),
    "portfolio": list(getattr(inner, "portfolio", []) or []),
    "decision_cutoff_proba": _float_or_none(getattr(inner, "decision_cutoff_proba_", None)),
    "num_batches": len(getattr(inner, "models", []) or []),
    "population_prior": _float_or_none(getattr(inner, "population_prior_", None)),
    "n_samples": int(len(y_arr)),
    "n_actives": int(np.sum(y_arr == 1)),
  }
  if report["oof_auc"] is not None and report["train_auc"] is not None:
    report["overfit_gap"] = report["train_auc"] - report["oof_auc"]
  # Per-phase wall-clock (preprocess / select / calibrate …), summed across batches, for the
  # on-screen "how long it takes" breakdown. Best-effort: skip non-scalar entries (e.g. per-fold
  # lists) and never fail the fit.
  timing = _summed_timing(inner)
  if timing:
    report["timing"] = timing
  return report


#: Percentile a regression error is normalised by before it becomes a reliability (lazy-qsar's
#: ``_ERROR_NORM_PERCENTILE``).
ERROR_NORM_PERCENTILE = 95


def _reg_cv_report(model, y_train):
  """Build the ``cv_report.json`` dict from a fitted lazy-qsar regressor.

  Spearman's rho replaces AUROC as the skill measure, as it does throughout lazy-qsar's regression
  ensemble, and ``oof_metrics`` carries the rest (R2, RMSE, MAE).
  """
  inner = getattr(model, "_model", None)
  report = {
    "oof_rho": _float_or_none(getattr(model, "oof_rho_", None)),
    "train_rho": _float_or_none(getattr(model, "train_rho_", None)),
    "oof_metrics": dict(getattr(model, "oof_metrics_", None) or {}),
    "portfolio": list(getattr(inner, "portfolio", []) or []),
    "num_batches": len(getattr(inner, "models", []) or []),
    "n_samples": int(len(np.asarray(y_train))),
  }
  if report["oof_rho"] is not None and report["train_rho"] is not None:
    report["overfit_gap"] = report["train_rho"] - report["oof_rho"]
  timing = _summed_timing(inner)
  if timing:
    report["timing"] = timing
  return report


def _score_reg_rows(model, ad_model, rows, idx, preds, ranks, ads):
  """Write the predicted value, percentile and AD score of ``rows`` at positions ``idx``."""
  preds[idx] = model.predict(X=rows)
  with contextlib.suppress(Exception):
    ranks[idx] = model._oof_percentile_from_value(preds[idx])
  if ad_model is not None:
    with contextlib.suppress(Exception):
      ads[idx] = ad_model.score(rows)


def _score_rows(model, ad_model, rows, idx, preds, ranks, ads):
  """Write the probability, percentile and AD score of ``rows`` at positions ``idx`` in place."""
  proba = model.predict_proba(X=rows)[:, 1]
  preds[idx] = proba
  with contextlib.suppress(Exception):
    # The percentile is the out-of-fold ECDF of this same calibrated probability, so it reads the
    # values just computed rather than pushing ``rows`` through every head a second time. A model
    # trained before the knots were stored falls back to the second pass inside the helper. This
    # runs on every held-out fold and on every row a fold did not train on, so the saving is the
    # whole scoring pass, not a corner of one.
    ranks[idx] = _percentile_from_proba(model, proba, rows)
  if ad_model is not None:
    with contextlib.suppress(Exception):
      ads[idx] = ad_model.score(rows)


class Fitter(BaseEstimatorIndividual):
  """Fit-time lazyqsar estimator for one descriptor."""

  #: Score the rank-reference library for the pooled rank. Off for held-out folds, which only score
  #: AUROC and would otherwise push the 50,000-molecule library through every fold's models.
  score_reference = True

  def __init__(self, path, model_id, is_simple, batch_size=None, substep_cb=None):
    BaseEstimatorIndividual.__init__(
      self,
      path=path,
      estimator=ESTIMATORS_FAMILY_SUBFOLDER,
      model_id=model_id,
      batch_size=batch_size,
    )
    self.trained_path = os.path.join(
      self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )
    self.is_simple = is_simple
    # Optional callback to show the current sub-phase in the Estimate live table (set by the
    # orchestrator). No-op by default so the Fitter runs standalone.
    self.substep_cb = substep_cb or (lambda _text: None)

  def run(self):
    """Fit the descriptor's lazyqsar classifier and return its per-row classification report.

    Trains on the training rows and saves the model, applicability domain, CV report and pool
    signals -- plus, for classification, the reference scores a rank needs. Training rows get
    out-of-fold signals; the other rows are predicted by the fitted model.

    Returns
    -------
    collections.OrderedDict
      ``{"clf": report}`` from :func:`make_classification_report`, or ``{"reg": report}`` from
      :func:`make_regression_report`; empty when there is no descriptor data to fit.
    """
    self.reset_time()
    tasks = collections.OrderedDict()
    shape = self._get_X_shape()
    if shape is None:
      logger.warning(f"[lazyqsar:fit] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    train_idxs = self.get_train_indices(path=self.path)
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    # One descriptor can dominate a run -- on a 2,000-compound regression fit one took 49 of the
    # 53 minutes -- and nothing in the durable record said which. Timed here, where the per-model
    # timing figure cannot reach (it measures featurization, not training).
    started = time.time()
    if self.task == "classification":
      tasks[t] = self._fit_classifier(shape[0], train_idxs, y, t)
    else:
      tasks[t] = self._fit_regressor(shape[0], train_idxs, y, t)
    self._patch_cv_report(
      os.path.join(self.trained_path, self.model_id), fit_seconds=round(time.time() - started, 1)
    )
    self.update_elapsed_time()
    gc.collect()
    return tasks

  def _fit_classifier(self, n_samples, train_idxs, y, t):
    """Fit, persist and score the classifier; return its per-row classification report."""
    # The reference is scored once, after fit (_write_reference_signals), for the pooled rank, which
    # the pool step builds as lazy-qsar's ensemble does. It is not given to fit: lazy-qsar would
    # score it again for a per-descriptor rank nothing uses, and move the descriptor's decision
    # cutoff onto that rank. Held-out folds (score_reference=False) skip it altogether. Resolved
    # before the fit so the seeding immediately precedes it.
    reference_h5 = self._rank_reference_or_warn() if self.score_reference else None
    model, X_train, y_train, train_order_arr, descriptor_dir = self._load_and_fit(
      LazyClassifier, n_samples, train_idxs, y
    )
    # Capture lazy-qsar's internal cross-validation outputs (OOF AUROC etc.) before X_train is
    # freed — they are computed during fit and would otherwise be discarded.
    channels = self._oof_channels(model, X_train)
    self._write_cv_report(model, y_train, channels, descriptor_dir)
    # Fit the applicability domain (lazy-qsar's PCA+Mahalanobis AD) on the training descriptor
    # matrix and derive its veto cutoff. Persisted alongside the model for inference. Guarded —
    # AD is an optional pooler signal.
    self.substep_cb("applicability domain")
    ad_model, ad_cutoff = self._fit_ad(X_train, descriptor_dir)
    # Per-row signals scattered back to original row order. ``preds`` is the calibrated
    # probability (OOF for training rows where available — honest, not resubstitution); ``ranks``
    # is lazy-qsar's percentile within the model's own OOF distribution (the reliability signal
    # its ensemble weights by); ``ads`` is the applicability-domain score. The training rows are always filled from X_train
    # (OOF probabilities, plus ranks and AD predicted in batches) so a fold that trains on a subset
    # is fitted on the same kind of signals as the shipped model; rows outside the training set
    # (the held-out slice of a fold) are predicted from the model in a chunked re-read.
    preds, ranks, ads = self._blank_signals(n_samples)
    preds_raw = np.full(n_samples, np.nan, dtype=np.float32)  # uncalibrated OOF (report raw lens)
    covers_all = train_order_arr.size == n_samples and np.unique(train_order_arr).size == n_samples
    self._fill_train_proba(
      model, X_train, channels, train_order_arr, descriptor_dir, preds, preds_raw
    )
    self._fill_train_rank_ad(model, ad_model, X_train, train_order_arr, channels, ranks, ads)
    del X_train
    gc.collect()
    self._save_model(model, t)
    if not covers_all:
      # Only the rows the model never saw: training rows already carry their OOF signals above.
      self._fill_unseen_signals(
        _score_rows, model, ad_model, train_order_arr, n_samples, preds, ranks, ads
      )
    if self.score_reference and reference_h5 is not None:
      self.substep_cb("rank reference")
      self._write_reference_signals(model, ad_model, reference_h5, descriptor_dir)
    # Percentile→error reliability curve, built from the training rows' (percentile, pred, label).
    tr = train_order_arr
    curve = self._build_rank_error_curve(ranks[tr], preds[tr], y[tr])
    self._write_pool_signals(
      {"oof_auc": _float_or_none(getattr(model, "oof_auc_", None))},
      curve,
      ad_cutoff,
      y,
      descriptor_dir,
    )
    rank_arg, ad_arg = self._report_args(ranks, ads, ad_model)
    raw_arg = preds_raw if np.isfinite(preds_raw).any() else None
    return make_classification_report(y, preds, y_rank=rank_arg, y_ad=ad_arg, y_raw=raw_arg)

  def _fit_regressor(self, n_samples, train_idxs, y, t):
    """Fit, persist and score the regressor; return its per-row regression report.

    The sibling of :meth:`_fit_classifier`. lazy-qsar's regression ensemble has no reference
    library, no decision cutoff and no uncalibrated score channel, so none of those are written:
    the per-descriptor outputs are the predicted value, its out-of-fold percentile (which indexes
    the reliability curve the pooler weights by) and the applicability-domain score.
    """
    model, X_train, y_train, train_order_arr, descriptor_dir = self._load_and_fit(
      LazyRegressor, n_samples, train_idxs, y
    )
    channels = self._reg_oof_channels(model, X_train)
    self._write_report(_reg_cv_report(model, y_train), channels, y_train, descriptor_dir)
    self.substep_cb("applicability domain")
    ad_model, ad_cutoff = self._fit_ad(X_train, descriptor_dir)

    preds, ranks, ads = self._blank_signals(n_samples)
    covers_all = train_order_arr.size == n_samples and np.unique(train_order_arr).size == n_samples
    self.substep_cb("out-of-fold + percentile")
    if channels is not None:
      preds[train_order_arr], ranks[train_order_arr] = channels
    else:
      logger.warning(
        f"[lazyqsar:fit] {self.model_id}: OOF unavailable, using resubstitution for pooler input"
      )
      preds[train_order_arr] = self._batched(lambda xb: model.predict(X=xb), X_train)
      # The percentile is a lookup on the value, so it reads the predictions just made rather
      # than predicting the whole matrix a second time (as :func:`_score_reg_rows` does too).
      with contextlib.suppress(Exception):
        ranks[train_order_arr] = model._oof_percentile_from_value(preds[train_order_arr])
    self._record_oof_used(descriptor_dir, channels is not None)
    if ad_model is not None:
      ads[train_order_arr] = self._batched(lambda xb: ad_model.score(xb), X_train, default=np.nan)
    del X_train
    gc.collect()
    self._save_model(model, t)
    if not covers_all:
      self._fill_unseen_signals(
        _score_reg_rows, model, ad_model, train_order_arr, n_samples, preds, ranks, ads
      )
    tr = train_order_arr
    curve = self._build_rank_error_curve(ranks[tr], preds[tr], y[tr], normalize=True)
    self._write_pool_signals(
      {"oof_rho": _float_or_none(getattr(model, "oof_rho_", None))},
      curve,
      ad_cutoff,
      y,
      descriptor_dir,
    )
    rank_arg, ad_arg = self._report_args(ranks, ads, ad_model)
    return make_regression_report(y, preds, y_rank=rank_arg, y_ad=ad_arg)

  def _reg_oof_channels(self, model, X_train):
    """lazy-qsar's out-of-fold ``(value, rank)`` per training row, or None.

    Two channels, not three: a regression ensemble has no uncalibrated score to report.
    """
    try:
      channels = model._model.oof_channels(X_train)
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] out-of-fold channels unavailable for {self.model_id}: {e}")
      return None
    if channels is None or any(len(c) != X_train.shape[0] for c in channels):
      return None
    return tuple(np.asarray(c, dtype=np.float32) for c in channels)

  def _load_train_rows(self, train_idxs):
    """Read the training rows chunk by chunk.

    Returns
    -------
    tuple
      ``(X_train, train_order)``: the stacked training matrix and the global row index of each of
      its rows, which is the order ``y`` must be indexed by.
    """
    X_train_parts = []
    train_order = []
    for start, end, chunk in self._iter_X():
      # Global indices of the training rows that fall in this chunk, kept in
      # train_idxs order. X_train is assembled chunk by chunk, so y MUST be indexed
      # by this same accumulated order — NOT by the global train_idxs order, or X
      # rows and y labels get permuted relative to each other across chunks
      # (multi-chunk / large datasets) and the model trains on mismatched pairs.
      sel = [i for i in train_idxs if start <= i < end]
      if sel:
        X_train_parts.append(chunk[[i - start for i in sel]])
        train_order.extend(sel)
      del chunk
      gc.collect()
    X_train = np.concatenate(X_train_parts, axis=0)
    del X_train_parts
    gc.collect()
    return X_train, train_order

  def _rank_reference_or_warn(self):
    """Path of the rank-reference library, or None after a warning."""
    reference_h5 = self._rank_reference_h5()
    if reference_h5 is None:
      logger.warning(
        f"[lazyqsar:fit] {self.model_id}: no rank reference found; the pooled `rank` will be "
        "unavailable"
      )
    return reference_h5

  def _oof_channels(self, model, X_train):
    """lazy-qsar's out-of-fold ``(proba, rank, score)`` per training row, or None.

    The same per-row values lazy-qsar pools its own ensemble and reference from: the calibrated
    probability (prior-corrected per batch), its percentile in the model's out-of-fold distribution,
    and the uncalibrated score. None when out-of-fold data is unavailable (e.g. tiny sets).
    """
    try:
      channels = model.oof_channels(X=X_train)
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] out-of-fold channels unavailable for {self.model_id}: {e}")
      return None
    if channels is None or any(len(c) != X_train.shape[0] for c in channels):
      return None
    return tuple(np.asarray(c, dtype=np.float32) for c in channels)

  def _fill_train_proba(
    self, model, X_train, channels, train_order_arr, descriptor_dir, preds, preds_raw
  ):
    """Fill the training rows of ``preds`` and ``preds_raw`` in place."""
    # Honest out-of-fold probabilities for the training rows (computed inside fit); resubstitution
    # only when lazy-qsar has no out-of-fold data.
    self.substep_cb("out-of-fold + percentile")
    use_oof = channels is not None
    if use_oof:
      preds[train_order_arr] = channels[0]
      preds_raw[train_order_arr] = channels[2]  # uncalibrated, for the report's raw score lens
    else:
      logger.warning(
        f"[lazyqsar:fit] {self.model_id}: OOF unavailable, using resubstitution for pooler input"
      )
      preds[train_order_arr] = self._batched(lambda xb: model.predict_proba(X=xb)[:, 1], X_train)
    self._record_oof_used(descriptor_dir, use_oof)

  def _fill_train_rank_ad(self, model, ad_model, X_train, train_order_arr, channels, ranks, ads):
    """Fill the training rows of ``ranks`` and ``ads`` in place."""
    if channels is not None:
      # The percentile of the same out-of-fold probability the pooler is fitted on.
      ranks[train_order_arr] = channels[1]
    else:
      ranks[train_order_arr] = self._batched(lambda xb: _percentile(model, xb), X_train, np.nan)
    if ad_model is not None:
      ads[train_order_arr] = self._batched(lambda xb: ad_model.score(xb), X_train, default=np.nan)

  def _fill_unseen_signals(
    self, score_rows, model, ad_model, train_order_arr, n_samples, preds, ranks, ads
  ):
    """Predict the rows outside the training set in a chunked re-read, filling arrays in place.

    ``score_rows`` is the task's row scorer (:func:`_score_rows` or :func:`_score_reg_rows`); the
    chunked re-read around it is the same either way.
    """
    is_train = np.zeros(n_samples, dtype=bool)
    is_train[train_order_arr] = True
    for start, end, chunk in self._iter_X():
      idx = np.flatnonzero(~is_train[start:end])
      if idx.size:
        score_rows(model, ad_model, chunk[idx], idx + start, preds, ranks, ads)
      del chunk
      gc.collect()

  def _load_and_fit(self, make_model, n_samples, train_idxs, y):
    """Load the training rows, fit a freshly seeded lazy-qsar model, and return the pieces.

    The preamble both tasks share. Returns ``(model, X_train, y_train, train_order, dir)``; the
    caller still owns ``X_train`` and frees it once its own signals are read off.
    """
    logger.info(f"[lazyqsar:fit] Loading training subset: {len(train_idxs)} of {n_samples} samples")
    X_train, train_order = self._load_train_rows(train_idxs)
    y_train = y[np.array(train_order)]
    logger.info(
      f"[lazyqsar:fit] Training on {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    _seed_global_rngs()
    model = make_model()
    model.fit(X=X_train, y=y_train)
    descriptor_dir = os.path.join(self.trained_path, self.model_id)
    os.makedirs(descriptor_dir, exist_ok=True)
    return model, X_train, y_train, np.asarray(train_order), descriptor_dir

  def _save_model(self, model, t):
    """Persist the fitted model under the descriptor's folder for the algorithm family ``t``."""
    model_folder = os.path.join(self.trained_path, self.model_id, t)
    model.save(model_folder)
    logger.info(f"[lazyqsar:fit] Model saved to {model_folder}")

  def _report_args(self, ranks, ads, ad_model):
    """``(rank, ad)`` arguments for the per-row report: whichever signals actually came out."""
    return (
      _rank_or_warn(ranks, self.model_id, "fit"),
      ads if (ad_model is not None and np.isfinite(ads).any()) else None,
    )

  @staticmethod
  def _blank_signals(n_samples):
    """Empty ``(preds, ranks, ads)`` arrays, scattered into by original row order."""
    return (
      np.empty(n_samples, dtype=np.float32),
      np.full(n_samples, np.nan, dtype=np.float32),
      np.full(n_samples, np.nan, dtype=np.float32),
    )

  @staticmethod
  def _patch_cv_report(descriptor_dir, **fields):
    """Merge ``fields`` into an already-written ``cv_report.json``. Best-effort."""
    path = os.path.join(descriptor_dir, "cv_report.json")
    try:
      with open(path) as f:
        report = json.load(f)
      report.update(fields)
      with open(path, "w") as f:
        json.dump(report, f, indent=2)
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] could not update {path}: {e}")

  @classmethod
  def _record_oof_used(cls, descriptor_dir, oof_used):
    """Record in ``cv_report.json`` whether the pooler input is out-of-fold (else resubstitution)."""
    cls._patch_cv_report(descriptor_dir, oof_used=bool(oof_used))

  def _write_cv_report(self, model, y_train, channels, descriptor_dir):
    """Persist lazy-qsar's internal cross-validation results for a classifier."""
    self._write_report(_cv_report(model, y_train), channels, y_train, descriptor_dir)

  def _write_report(self, report, channels, y_train, descriptor_dir):
    """Write ``cv_report.json`` and, best-effort, ``oof.csv`` (the out-of-fold predictions).

    ``report`` is the task's own summary: AUROC and the decision cutoff for a classifier, Spearman
    rho and R2/RMSE/MAE for a regressor. Fully guarded — never fails the fit.
    """
    try:
      with open(os.path.join(descriptor_dir, "cv_report.json"), "w") as f:
        json.dump(report, f, indent=2)
      if "oof_auc" in report:
        logger.info(
          f"[lazyqsar:fit] {self.model_id} CV: OOF AUROC={report['oof_auc']} "
          f"train AUROC={report['train_auc']}"
        )
      else:
        logger.info(
          f"[lazyqsar:fit] {self.model_id} CV: OOF rho={report['oof_rho']} "
          f"train rho={report['train_rho']}"
        )
    except Exception as e:
      logger.warning(f"[lazyqsar:fit] Could not write CV report for {self.model_id}: {e}")
    if channels is None:
      return
    try:
      import pandas as pd

      y_arr = np.asarray(y_train)
      clf = "oof_auc" in report
      # The classification column name is part of the report's contract (report/fetcher.py).
      column = "oof_proba" if clf else "oof_value"
      pd.DataFrame({
        column: channels[0],
        "y": y_arr.astype(int) if clf else y_arr.astype(float),
      }).to_csv(os.path.join(descriptor_dir, "oof.csv"), index=False)
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] could not write oof.csv for {self.model_id}: {e}")

  def _write_reference_signals(self, model, ad_model, reference_h5, descriptor_dir):
    """Score the rank-reference library and save ``(proba, percentile, ad)`` for the pooler.

    The pooler pushes these through its fitted weighting to get the pooled probability of every
    reference molecule, which is what the pooled ``rank`` is a position against (as in lazy-qsar's
    ensemble). Best-effort: without the file the pooled rank is unavailable.
    """
    try:
      parts = {"proba": [], "percentile": [], "ad": []}
      for chunk in open_h5(reference_h5).iter_values(self.batch_size):
        chunk = np.asarray(chunk, dtype=np.float32)
        proba = model.predict_proba(X=chunk)[:, 1]
        parts["proba"].append(proba)
        parts["percentile"].append(_percentile_from_proba(model, proba, chunk))
        if ad_model is not None:
          parts["ad"].append(ad_model.score(chunk))
      out = {k: np.concatenate(v).astype(np.float32) for k, v in parts.items() if v}
      np.savez(os.path.join(descriptor_dir, RANK_REFERENCE_SIGNALS_FILENAME), **out)
    except Exception as e:
      logger.warning(
        f"[lazyqsar:fit] {self.model_id}: could not score the rank reference ({e}); "
        "the pooled rank will be unavailable"
      )

  def _batched(self, fn, X, default=None):
    """Apply ``fn`` over X in batches, returning a (n,) float32 vector.

    On failure, fills with ``default`` when provided (the signal is optional), else re-raises.
    """
    out = np.empty(X.shape[0], dtype=np.float32)
    try:
      for s in range(0, X.shape[0], self.batch_size):
        out[s : s + self.batch_size] = fn(X[s : s + self.batch_size])
      return out
    except Exception as e:
      if default is None:
        raise
      logger.debug(f"[lazyqsar:fit] batched op failed for {self.model_id}: {e}")
      out[:] = default
      return out

  def _fit_ad(self, X_train, descriptor_dir):
    """Fit lazy-qsar's PCA+Mahalanobis applicability domain on the training matrix.

    Returns ``(ad_model, cutoff)`` where ``cutoff`` is the 5th percentile of training AD scores
    (the per-descriptor veto threshold used by the pooler). The fitted AD is pickled to
    ``ad.joblib`` for inference. Returns ``(None, None)`` on any failure — AD is optional.
    """
    try:
      from lazyqsar.applicability.domain import ApplicabilityDomain

      ad = ApplicabilityDomain()
      ad.fit(X_train)
      train_ad = ad.score(X_train)
      cutoff = float(np.percentile(train_ad, 5))
      joblib.dump(ad, os.path.join(descriptor_dir, "ad.joblib"))
      return ad, cutoff
    except Exception as e:
      logger.warning(f"[lazyqsar:fit] Applicability domain unavailable for {self.model_id}: {e}")
      return None, None

  @staticmethod
  def _build_rank_error_curve(ranks, preds, y, n_knots=20, normalize=False):
    """Port of lazy-qsar's rank→error curve: sort by OOF percentile, windowed-mean |pred−y| at ~20 knots.

    ``normalize`` is the regression form (lazy-qsar's ``_rank_error_curve``): a regression error is
    in the target's own units and unbounded, so it is divided by its 95th percentile and clipped to
    [0, 1], putting it on the same scale as a classifier's |p − y| and making ``1 − error`` a usable
    reliability. Returns ``(r_knots, e_knots)`` lists, or ``None`` when too few finite rows.
    """
    try:
      r = np.asarray(ranks, dtype=float)
      p = np.asarray(preds, dtype=float)
      yy = np.asarray(y, dtype=float)
      m = np.isfinite(r) & np.isfinite(p) & np.isfinite(yy)
      r, p, yy = r[m], p[m], yy[m]
      if len(r) < 2:
        return None
      err = np.abs(p - yy)
      if normalize:
        scale = float(np.percentile(err, ERROR_NORM_PERCENTILE))
        err = np.clip(err / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(err)
      sidx = np.argsort(r)
      rs, es = r[sidx], err[sidx]
      nk = min(n_knots, len(rs))
      if nk < 2:
        return None
      ki = np.round(np.linspace(0, len(rs) - 1, nk)).astype(int)
      hw = max(1, len(rs) // (nk * 2))
      r_knots = rs[ki]
      e_knots = np.array([es[max(0, k - hw) : k + hw + 1].mean() for k in ki])
      return r_knots.tolist(), e_knots.tolist()
    except Exception:
      return None

  def _write_pool_signals(self, skill, rank_error_curve, ad_hard_cutoff, y, descriptor_dir):
    """Persist per-descriptor pooler signals (the task's skill, rank→error curve, AD veto cutoff).

    ``skill`` is ``{"oof_auc": ...}`` for a classifier and ``{"oof_rho": ...}`` for a regressor —
    the measure that descriptor's weight and the pooled model's pruning are based on.
    """
    try:
      y_arr = np.asarray(y)
      signals = {
        **skill,
        "rank_error_curve": rank_error_curve,
        "ad_hard_cutoff": ad_hard_cutoff,
        "n_samples": int(len(y_arr)),
      }
      if "oof_auc" in skill:
        signals["n_actives"] = int(np.sum(y_arr == 1))
      joblib.dump(signals, os.path.join(descriptor_dir, "pool_signals.joblib"))
    except Exception as e:
      logger.warning(f"[lazyqsar:fit] Could not write pool signals for {self.model_id}: {e}")


class Predictor(BaseEstimatorIndividual):
  """Predict-time lazyqsar estimator for one descriptor."""

  def __init__(self, path, model_id, batch_size=None, substep_cb=None):
    BaseEstimatorIndividual.__init__(
      self,
      path=path,
      estimator=ESTIMATORS_FAMILY_SUBFOLDER,
      model_id=model_id,
      batch_size=batch_size,
    )
    self.trained_path = os.path.join(
      self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
    )
    self.substep_cb = substep_cb or (lambda _text: None)

  def run(self):
    """Apply the trained model and AD model to the run's descriptor matrix, chunk by chunk.

    Returns ``{"clf": report}`` with probabilities, or ``{"reg": report}`` with predicted values,
    each with the out-of-fold percentiles and AD scores the pooler weights by; an empty dict when
    there is no descriptor data.
    """
    self.reset_time()
    tasks = collections.OrderedDict()
    shape = self._get_X_shape()
    if shape is None:
      logger.warning(f"[lazyqsar:predict] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    y = self._get_y()
    clf = self.task == "classification"
    t = "clf" if clf else "reg"
    model_folder = os.path.join(self.trained_path, self.model_id, t)
    model = (LazyClassifier if clf else LazyRegressor).load(model_folder)
    logger.info(
      f"[lazyqsar:predict] Loaded model from {model_folder}, predicting {shape[0]} samples chunk-by-chunk"
    )
    n_samples = shape[0]
    preds = np.empty(n_samples, dtype=np.float32)
    ranks = np.full(n_samples, np.nan, dtype=np.float32)
    ads = np.full(n_samples, np.nan, dtype=np.float32)
    ad_model = self._load_ad()
    self.substep_cb("applying model + AD" if ad_model is not None else "applying model")
    for start, end, chunk in self._iter_X():
      if clf:
        _score_rows(model, ad_model, chunk, np.arange(start, end), preds, ranks, ads)
      else:
        _score_reg_rows(model, ad_model, chunk, np.arange(start, end), preds, ranks, ads)
      logger.debug(f"[lazyqsar:predict] Processed {start}-{end}/{n_samples}")
      del chunk
      gc.collect()
    rank_arg = _rank_or_warn(ranks, self.model_id, "predict")
    ad_arg = ads if (ad_model is not None and np.isfinite(ads).any()) else None
    if clf:
      tasks[t] = make_classification_report(y, preds, y_rank=rank_arg, y_ad=ad_arg)
    else:
      tasks[t] = make_regression_report(y, preds, y_rank=rank_arg, y_ad=ad_arg)
    self.update_elapsed_time()
    return tasks

  def _load_ad(self):
    """Load the per-descriptor applicability-domain model persisted at fit time, or None."""
    try:
      p = os.path.join(self.trained_path, self.model_id, "ad.joblib")
      if os.path.exists(p):
        return joblib.load(p)
    except Exception as e:
      logger.debug(f"[lazyqsar:predict] AD load skipped for {self.model_id}: {e}")
    return None


class IndividualEstimator(ZairaBase):
  """Fitter or Predictor for one descriptor, chosen by the run mode."""

  def __init__(self, path=None, model_id=None, is_simple=True, batch_size=None, substep_cb=None):
    ZairaBase.__init__(self)
    self.model_id = model_id
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    self.path = path
    if not self.is_predict():
      self.estimator = Fitter(
        path=self.path,
        model_id=self.model_id,
        is_simple=is_simple,
        batch_size=self.batch_size,
        substep_cb=substep_cb,
      )
    else:
      self.estimator = Predictor(
        path=self.path,
        model_id=self.model_id,
        batch_size=self.batch_size,
        substep_cb=substep_cb,
      )

  def run(self):
    """Run the fitter or predictor and dump its report to the descriptor's ``y_hat.joblib``."""
    results = self.estimator.run()
    joblib.dump(
      results,
      os.path.join(
        self.path,
        ESTIMATORS_SUBFOLDER,
        ESTIMATORS_FAMILY_SUBFOLDER,
        self.model_id,
        Y_HAT_FILE,
      ),
    )


class Estimator(ZairaBase):
  """Lazyqsar estimate step over every descriptor the run uses."""

  def __init__(self, path=None, batch_size=None):
    ZairaBase.__init__(self)
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE

  def _run_dir(self):
    """The run folder, whether this estimator was given one or reads the session's."""
    return self.get_output_dir() if self.path is None else self.path

  def _task(self):
    """The run's task, for reading the per-descriptor skill under its own name."""
    try:
      with open(os.path.join(self._run_dir(), METADATA_SUBFOLDER, PARAMETERS_FILE)) as f:
        return json.load(f).get("task", "classification")
    except (OSError, ValueError) as e:
      logger.debug(f"[lazyqsar] could not read the task, assuming classification: {e}")
      return "classification"

  def _get_model_ids(self):
    if self.path is None:
      path = self.get_output_dir()
    else:
      path = self.path
    if self.is_predict():
      path_trained = self.get_trained_dir()
    else:
      path_trained = path
    # Train only the descriptors the run actually uses (the --max-descriptors subset when screened,
    # else every featurizer in done_eos.json).
    from zairachem.base.utils.descriptors import effective_descriptors, pooled_descriptors

    if self.is_predict():  # only what the trained pooled model uses
      return pooled_descriptors(path_trained)
    return effective_descriptors(path_trained)

  def _read_cv_result(self, model_id):
    """Per-descriptor result for the live table, read from persisted artifacts.

    Returns a dict ``{oof, portfolio}``: the out-of-fold skill -- AUROC for classification,
    Spearman's rho for regression -- and the selected algorithms from ``cv_report.json`` (a fit
    artifact; both stay empty at predict, whose table hides those columns). Best-effort; missing
    files are fine.
    """
    from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER

    res = {"oof": None, "portfolio": []}
    cv_dir = os.path.join(
      self._run_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER, model_id
    )
    oof_key = CV_SKILL[self._task()][0]
    try:
      with open(os.path.join(cv_dir, "cv_report.json")) as f:
        r = json.load(f)
      res["oof"] = r.get(oof_key)
      res["portfolio"] = r.get("portfolio", [])
    except (OSError, ValueError) as e:
      logger.debug(f"[lazyqsar] {model_id}: no CV report for the live table: {e}")
    return res

  def run(self):
    """Fit or predict each used descriptor under a live table, skipping the ones that fail.

    Raises ``RuntimeError`` when no descriptor produces a valid estimator.
    """
    from zairachem.base.utils.progress import STEP_COLORS
    from zairachem.estimate.estimators.lazy_qsar.monitor import TrainingMonitor

    model_ids = self._get_model_ids()
    monitor = TrainingMonitor(
      model_ids,
      color=STEP_COLORS.get("estimate", "yellow"),
      result_reader=self._read_cv_result,
      predict=self.is_predict(),
      skill="AUC" if self._task() == "classification" else "rho",
    )
    # One descriptor at a time, in this process: training them in parallel processes gave
    # different, and lower, out-of-fold AUROCs from run to run, while this loop is reproducible.
    n_success = self._run_serial(model_ids, monitor)
    if n_success == 0:
      raise RuntimeError(
        "No descriptor produced a valid estimator — cannot build a model. All "
        "descriptors failed (e.g. non-predictive features were entirely eliminated). "
        "Check the dataset and the chosen descriptors."
      )
    logger.info(f"[lazyqsar] {n_success}/{len(model_ids)} descriptors produced a valid estimator")

  def _run_serial(self, model_ids, monitor):
    """Fit or predict the descriptors one at a time; returns how many produced an estimator."""
    n_success = 0
    with monitor.live():
      for model_id in model_ids:
        logger.info(f"[lazyqsar] Processing model {model_id}")
        monitor.start(model_id)
        ok = False
        try:
          estimator = IndividualEstimator(
            path=self.path,
            model_id=model_id,
            batch_size=self.batch_size,
            substep_cb=lambda text, _mid=model_id: monitor.set_substep(_mid, text),
          )
          estimator.run()
          n_success += 1
          ok = True
        except Exception as e:
          # A descriptor whose features are non-predictive can have all of them
          # eliminated by the internal feature selection, leaving an empty matrix that
          # crashes the estimator. Skip it instead of aborting the whole run; it simply
          # won't contribute to the final model (the assembler/pool already ignore
          # descriptors without a y_hat.joblib / results file).
          logger.warning(
            f"[lazyqsar] Skipping descriptor {model_id}: estimator failed "
            f"({type(e).__name__}: {e}). It will be excluded from the final model."
          )
        finally:
          monitor.finish(model_id, ok=ok)
        gc.collect()
    return n_success
