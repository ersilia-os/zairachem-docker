import collections, contextlib, joblib, json, os, gc
import numpy as np
from lazyqsar.agnostic import LazyClassifier
from zairachem.estimate.estimators.lazy_qsar.utils import make_classification_report

# lazy-qsar prints a rich directory tree of the saved model folder on every ``save()`` — it is the
# one piece of lazy-qsar console output not gated behind its ``--verbose`` flag, and it is pure
# noise inside our pipeline (we render our own progress). Silence just that method on the shared
# lazy-qsar logger singleton; everything else lazy-qsar emits is already quiet by default.
try:
  from lazyqsar.utils.logging import logger as _lq_logger

  _lq_logger.dir_tree = lambda *a, **k: None
except Exception:
  pass
from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.vars import (
  ESTIMATORS_SUBFOLDER,
  Y_HAT_FILE,
)
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.base import BaseEstimatorIndividual


class Fitter(BaseEstimatorIndividual):
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
    if self.task == "classification":
      logger.info(
        f"[lazyqsar:fit] Loading training subset: {len(train_idxs)} of {shape[0]} samples"
      )
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
      y_train = y[np.array(train_order)]
      logger.info(
        f"[lazyqsar:fit] Training on {X_train.shape[0]} samples, {X_train.shape[1]} features"
      )
      model = LazyClassifier()
      model.fit(X=X_train, y=y_train)
      # Capture lazy-qsar's internal cross-validation outputs (OOF AUROC etc.) before X_train is
      # freed — they are computed during fit and would otherwise be discarded.
      descriptor_dir = os.path.join(self.trained_path, self.model_id)
      os.makedirs(descriptor_dir, exist_ok=True)
      self._write_cv_report(model, X_train, y_train, descriptor_dir)
      # Fit the applicability domain (lazy-qsar's PCA+Mahalanobis AD) on the training descriptor
      # matrix and derive its veto cutoff. Persisted alongside the model for inference. Guarded —
      # AD is an optional pooler signal.
      self.substep_cb("applicability domain")
      ad_model, ad_cutoff = self._fit_ad(X_train, descriptor_dir)
      # Per-row signals scattered back to original row order. ``preds`` is the calibrated
      # probability (OOF for training rows where available — honest, not resubstitution); ``ranks``
      # is the training-OOF rank quantile; ``ads`` is the applicability-domain score. The training
      # set is currently the whole dataset (get_train_indices returns all rows), so X_train holds
      # every row and we predict straight from it in batches; if a real train/test split is ever
      # introduced (train_order no longer covers all rows) we fall back to a chunked re-read.
      n_samples = shape[0]
      preds = np.empty(n_samples, dtype=np.float32)
      preds_raw = np.full(n_samples, np.nan, dtype=np.float32)  # uncalibrated OOF (report raw lens)
      ranks = np.full(n_samples, np.nan, dtype=np.float32)
      ads = np.full(n_samples, np.nan, dtype=np.float32)
      train_order_arr = np.asarray(train_order)
      covers_all = (
        train_order_arr.size == n_samples and np.unique(train_order_arr).size == n_samples
      )
      # Honest out-of-fold probabilities for the training rows (already computed inside fit);
      # falls back to resubstitution when OOF cannot be reconstructed (multi-batch / tiny sets).
      self.substep_cb("out-of-fold + rank")
      oof_proba, _ = self._safe_pooled_oof(model, X_train, y_train)
      use_oof = oof_proba is not None and len(oof_proba) == train_order_arr.size
      if not use_oof:
        logger.info(
          f"[lazyqsar:fit] {self.model_id}: OOF unavailable, using resubstitution for pooler input"
        )
      # Uncalibrated (raw) pooled OOF — for the report's raw score lens. Independent of use_oof; simply
      # absent (NaN → dropped) when raw OOF can't be reconstructed.
      oof_raw, _ = self._safe_pooled_oof_raw(model, X_train, y_train)
      use_raw = oof_raw is not None and len(oof_raw) == train_order_arr.size
      if covers_all:
        if use_oof:
          preds[train_order_arr] = np.asarray(oof_proba, dtype=np.float32)
        else:
          preds[train_order_arr] = self._batched(
            lambda xb: model.predict_proba(X=xb)[:, 1], X_train
          )
        if use_raw:
          preds_raw[train_order_arr] = np.asarray(oof_raw, dtype=np.float32)
        ranks[train_order_arr] = self._batched(
          lambda xb: model.predict_rank(X=xb)[:, 1], X_train, default=np.nan
        )
        if ad_model is not None:
          ads[train_order_arr] = self._batched(
            lambda xb: ad_model.score(xb), X_train, default=np.nan
          )
      del X_train
      gc.collect()
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model.save(model_folder)
      logger.info(f"[lazyqsar:fit] Model saved to {model_folder}")
      if not covers_all:
        for start, end, chunk in self._iter_X():
          preds[start:end] = model.predict_proba(X=chunk)[:, 1]
          with contextlib.suppress(Exception):
            ranks[start:end] = model.predict_rank(X=chunk)[:, 1]
          if ad_model is not None:
            with contextlib.suppress(Exception):
              ads[start:end] = ad_model.score(chunk)
          del chunk
          gc.collect()
      # Rank→error reliability curve, built from the training rows' (rank, prediction, label).
      tr = train_order_arr
      curve = self._build_rank_error_curve(ranks[tr], preds[tr], y[tr])
      self._write_pool_signals(model, curve, ad_cutoff, y, descriptor_dir)
      rank_arg = ranks if np.isfinite(ranks).any() else None
      ad_arg = ads if (ad_model is not None and np.isfinite(ads).any()) else None
      raw_arg = preds_raw if np.isfinite(preds_raw).any() else None
      tasks[t] = make_classification_report(y, preds, y_rank=rank_arg, y_ad=ad_arg, y_raw=raw_arg)
    elif self.task == "regression":
      # No working regression estimator exists yet: lazyqsar.agnostic.LazyRegressor is a stub, so
      # the lazy_qsar family produces no regression output (and thus no OOF/rank/AD pooler signals).
      # Skip rather than crash — matching Predictor.run, which has no regression branch either.
      logger.warning(
        f"[lazyqsar:fit] Regression is not implemented for lazy_qsar; skipping {self.model_id}. "
        "Pooler reliability signals are classification-only."
      )
    self.update_elapsed_time()
    gc.collect()
    return tasks

  def _write_cv_report(self, model, X_train, y_train, descriptor_dir):
    """Persist lazy-qsar's internal cross-validation results for this descriptor.

    Writes ``cv_report.json`` (OOF/train AUROC, overfit gap, selected algorithms, decision cutoff,
    class counts) and, best-effort, ``oof.csv`` (pooled out-of-fold probabilities + labels) for ROC
    / calibration / score-distribution plots. Fully guarded — never fails the fit.
    """

    def _f(v):
      try:
        return float(v)
      except Exception:
        return None

    try:
      inner = getattr(model, "_model", None)
      y_arr = np.asarray(y_train)
      report = {
        "oof_auc": _f(model.oof_auc_),
        "train_auc": _f(model.train_auc_),
        "portfolio": list(getattr(inner, "portfolio", []) or []),
        "decision_cutoff_proba": _f(getattr(inner, "decision_cutoff_proba_", None)),
        "num_batches": len(getattr(inner, "models", []) or []),
        "population_prior": _f(getattr(inner, "population_prior_", None)),
        "n_samples": int(len(y_arr)),
        "n_actives": int(np.sum(y_arr == 1)),
      }
      if report["oof_auc"] is not None and report["train_auc"] is not None:
        report["overfit_gap"] = report["train_auc"] - report["oof_auc"]
      # Per-phase wall-clock (preprocess / select / calibrate …), summed across batches, for the
      # on-screen "how long it takes" breakdown. Best-effort: skip non-scalar entries (e.g. per-fold
      # lists) and never fail the fit.
      timing = {}
      for bm in getattr(inner, "models", []) or []:
        t = getattr(bm, "timing_", None)
        if isinstance(t, dict):
          for k, v in t.items():
            try:
              timing[k] = round(timing.get(k, 0.0) + float(v), 3)
            except (TypeError, ValueError):
              continue
      if timing:
        report["timing"] = timing
      with open(os.path.join(descriptor_dir, "cv_report.json"), "w") as f:
        json.dump(report, f, indent=2)
      logger.info(
        f"[lazyqsar:fit] {self.model_id} CV: OOF AUROC={report['oof_auc']} "
        f"train AUROC={report['train_auc']}"
      )
    except Exception as e:
      logger.warning(f"[lazyqsar:fit] Could not write CV report for {self.model_id}: {e}")

    try:
      oof_proba, oof_y = self._pooled_oof(getattr(model, "_model", None), X_train, y_train)
      if oof_proba is not None:
        import pandas as pd

        pd.DataFrame({"oof_proba": oof_proba, "y": oof_y}).to_csv(
          os.path.join(descriptor_dir, "oof.csv"), index=False
        )
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] OOF reconstruction skipped for {self.model_id}: {e}")

  @staticmethod
  def _pooled_oof(inner, X, y):
    """Reconstruct the pooled out-of-fold probabilities (single-batch case), aligned to X order.

    Mirrors the assembler's ``_compute_oof_auc``: stack per-head OOF probabilities and combine them
    with the batch pooler weights. Returns ``(None, None)`` for the multi-batch / subsampled case
    (where OOF rows don't map 1:1 to X) so callers simply omit ``oof.csv``.
    """
    models = getattr(inner, "models", None) or []
    if len(models) != 1:
      return None, None
    batch = models[0]
    heads = getattr(batch, "heads", None) or []
    if not heads or not all(hasattr(getattr(h, "model", None), "oof_probas_") for h in heads):
      return None, None
    S = np.column_stack([h.model.oof_probas_ for h in heads])
    if S.shape[0] != len(y):
      return None, None
    X_prep = batch.prep.transform(X)
    W = batch.pooler.get_weights(X_prep)
    pooled = (W * S).sum(axis=1)
    return np.asarray(pooled, dtype=float), np.asarray(y, dtype=int)

  def _safe_pooled_oof(self, model, X, y):
    """Guarded ``_pooled_oof``: returns (None, None) instead of raising."""
    try:
      return self._pooled_oof(getattr(model, "_model", None), X, y)
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] OOF reconstruction failed for {self.model_id}: {e}")
      return None, None

  @staticmethod
  def _pooled_oof_raw(inner, X, y):
    """Pooled UNCALIBRATED out-of-fold scores (single-batch case), aligned to X order.

    Parallel to :meth:`_pooled_oof` but stacks each head's ``oof_raw_`` (the [0,1] pre-calibration OOF
    probability, exposed by lazy-qsar ≥ 3.4.2) instead of the calibrated ``oof_probas_``, combined with
    the same batch-pooler weights. Returns ``(None, None)`` when raw OOF is unavailable (older
    lazy-qsar) or in the multi-batch / subsampled case.
    """
    models = getattr(inner, "models", None) or []
    if len(models) != 1:
      return None, None
    batch = models[0]
    heads = getattr(batch, "heads", None) or []
    if not heads or not all(hasattr(getattr(h, "model", None), "oof_raw_") for h in heads):
      return None, None
    S = np.column_stack([h.model.oof_raw_ for h in heads])
    if S.shape[0] != len(y):
      return None, None
    X_prep = batch.prep.transform(X)
    W = batch.pooler.get_weights(X_prep)
    pooled = (W * S).sum(axis=1)
    return np.asarray(pooled, dtype=float), np.asarray(y, dtype=int)

  def _safe_pooled_oof_raw(self, model, X, y):
    """Guarded ``_pooled_oof_raw``: returns (None, None) instead of raising."""
    try:
      return self._pooled_oof_raw(getattr(model, "_model", None), X, y)
    except Exception as e:
      logger.debug(f"[lazyqsar:fit] raw OOF reconstruction failed for {self.model_id}: {e}")
      return None, None

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
  def _build_rank_error_curve(ranks, preds, y, n_knots=20):
    """Port of lazy-qsar's rank→error curve: sort by rank, windowed-mean |pred−y| at ~20 knots.

    Returns ``(r_knots, e_knots)`` lists for interpolation, or ``None`` when too few finite rows.
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

  def _write_pool_signals(self, model, rank_error_curve, ad_hard_cutoff, y, descriptor_dir):
    """Persist per-descriptor pooler signals (OOF AUROC, rank→error curve, AD veto cutoff)."""

    def _f(v):
      try:
        return float(v)
      except Exception:
        return None

    try:
      y_arr = np.asarray(y)
      signals = {
        "oof_auc": _f(getattr(model, "oof_auc_", None)),
        "rank_error_curve": rank_error_curve,
        "ad_hard_cutoff": ad_hard_cutoff,
        "n_samples": int(len(y_arr)),
        "n_actives": int(np.sum(y_arr == 1)),
      }
      joblib.dump(signals, os.path.join(descriptor_dir, "pool_signals.joblib"))
    except Exception as e:
      logger.warning(f"[lazyqsar:fit] Could not write pool signals for {self.model_id}: {e}")


class Predictor(BaseEstimatorIndividual):
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
    self.reset_time()
    tasks = collections.OrderedDict()
    shape = self._get_X_shape()
    if shape is None:
      logger.warning(f"[lazyqsar:predict] Skipping {self.model_id}: no descriptor data available")
      self.update_elapsed_time()
      return tasks
    y = self._get_y()
    t = "reg" if self.task == "regression" else "clf"
    if self.task == "classification":
      model_folder = os.path.join(self.trained_path, self.model_id, t)
      model = LazyClassifier.load(model_folder)
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
        preds[start:end] = model.predict_proba(X=chunk)[:, 1]
        with contextlib.suppress(Exception):
          ranks[start:end] = model.predict_rank(X=chunk)[:, 1]
        if ad_model is not None:
          with contextlib.suppress(Exception):
            ads[start:end] = ad_model.score(chunk)
        logger.debug(f"[lazyqsar:predict] Processed {start}-{end}/{n_samples}")
        del chunk
        gc.collect()
      rank_arg = ranks if np.isfinite(ranks).any() else None
      ad_arg = ads if (ad_model is not None and np.isfinite(ads).any()) else None
      tasks[t] = make_classification_report(y, preds, y_rank=rank_arg, y_ad=ad_arg)
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
  def __init__(self, path=None, model_id=None, is_simple=True, batch_size=None, substep_cb=None):
    ZairaBase.__init__(self)
    self.model_id = model_id
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    if path is None:
      self.path = self.get_output_dir()
    else:
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
  def __init__(self, path=None, batch_size=None):
    ZairaBase.__init__(self)
    self.path = path
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE

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
    from zairachem.base.utils.descriptors import effective_descriptors

    return effective_descriptors(path_trained)

  def _read_cv_result(self, model_id):
    """Per-descriptor result for the live table, read from persisted artifacts.

    Returns a dict ``{oof, portfolio, ad, rank}``: OOF AUROC and selected algorithms from
    ``cv_report.json``, plus whether this descriptor produced an applicability-domain model and a
    rank→error reliability curve (from ``pool_signals.joblib``). Best-effort; missing files are fine.
    """
    from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER

    out = self.get_output_dir() if self.path is None else self.path
    res = {"oof": None, "portfolio": [], "ad": False, "rank": False}
    # cv_report (OOF / portfolio) is a FIT artifact — only in the run that trained. At predict the
    # current dir has none, so these stay None (the predict table hides those columns anyway).
    cv_dir = os.path.join(out, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER, model_id)
    try:
      with open(os.path.join(cv_dir, "cv_report.json")) as f:
        r = json.load(f)
      res["oof"] = r.get("oof_auc")
      res["portfolio"] = r.get("portfolio", [])
    except Exception:
      pass
    # Pooler signals (AD / rank) live with the trained model — read from the trained dir at predict.
    sig_base = self.get_trained_dir() if self.is_predict() else out
    sig_dir = os.path.join(sig_base, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER, model_id)
    try:
      sig = joblib.load(os.path.join(sig_dir, "pool_signals.joblib"))
      res["ad"] = sig.get("ad_hard_cutoff") is not None
      res["rank"] = sig.get("rank_error_curve") is not None
    except Exception:
      pass
    return res

  def run(self):
    from zairachem.base.utils.progress import STEP_COLORS
    from zairachem.estimate.estimators.lazy_qsar.monitor import TrainingMonitor

    model_ids = self._get_model_ids()
    n_success = 0
    monitor = TrainingMonitor(
      model_ids,
      color=STEP_COLORS.get("estimate", "yellow"),
      result_reader=self._read_cv_result,
      predict=self.is_predict(),
    )
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
    if n_success == 0:
      raise RuntimeError(
        "No descriptor produced a valid estimator — cannot build a model. All "
        "descriptors failed (e.g. non-predictive features were entirely eliminated). "
        "Check the dataset and the chosen descriptors."
      )
    logger.info(f"[lazyqsar] {n_success}/{len(model_ids)} descriptors produced a valid estimator")
