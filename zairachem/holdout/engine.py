"""Execute one held-out fold end to end: fit descriptors on train, pool, score the held-out slice.

The pooled model is the product we score, whichever task it is. For each fold: re-fit every
descriptor's estimator on the train slice (predictions for all rows fall out of the parent Fitter),
stack them, fit the reliability pooler on the train slice, then combine on the held-out slice —
giving an honest pooled prediction on molecules the fold never trained on. Descriptors are read from
the shared workspace, so no featurization is repeated.
"""

import json
import os

import joblib
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  CUTOFF_DEFAULT,
  CUTOFF_KAPPA,
  DEFAULT_CUTOFF,
  ESTIMATORS_SUBFOLDER,
  FOLDS_SUBFOLDER,
  Y_HAT_FILE,
)
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.lazy_qsar.assemble import IndividualOutcomeAssembler
from zairachem.holdout.indices import HoldoutFitter
from zairachem.holdout.workspace import build_fold_workspace
from zairachem.pool.base import BasePooler
from zairachem.pool.reliability_pooler import pooler


def _slice(matrix, idx):
  return matrix[idx] if matrix is not None else None


def _plain(value):
  """A numpy scalar (integer, float or bool) as the Python value it holds; anything else as is.

  The shared metrics helpers hand back whatever numpy produced -- the confusion-matrix counts are
  ``numpy.int64`` -- and ``json`` refuses those, so a fold record is converted as it is built.
  The numeric value is unchanged.
  """
  return value.item() if isinstance(value, np.generic) else value


def _fit_descriptors(fold_path, model_ids, train_idxs, batch_size, substep_cb=None):
  """Re-fit each descriptor's estimator on the train slice and assemble its per-row predictions."""
  n = len(model_ids)
  for i, model_id in enumerate(model_ids, start=1):
    if substep_cb:
      substep_cb(f"model {i}/{n}: {model_id}")
    fitter = HoldoutFitter(
      path=fold_path, model_id=model_id, train_idxs=train_idxs, batch_size=batch_size
    )
    results = fitter.run()
    if not results:
      logger.warning(f"[evaluate] descriptor {model_id} produced no estimator output; skipping.")
      continue
    joblib.dump(
      results,
      os.path.join(
        fold_path, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER, model_id, Y_HAT_FILE
      ),
    )
    IndividualOutcomeAssembler(path=fold_path, model_id=model_id).run()


def _score_fold(fold_path, train_idx, test_idx, fetcher, batch_size):
  """Fit the pooled model on train, apply it to the held-out slice, score it with the shared metrics.

  A classification fold has no rank reference, so its metrics are at the ``kappa`` cutoff (tuned on
  the fold's own training rows) and at ``default``. A regression fold is scored on R2, RMSE, MAE
  and the correlations, which need no cutoff.
  """
  from zairachem.pool.reliability_pooler.pool import _kappa_cutoff

  io = BasePooler(path=fold_path, batch_size=batch_size)
  task = io.task
  clf = task == "classification"
  df_X = io._get_X()  # per-descriptor predictions for all rows (also writes model/pool/data.csv)
  y = np.asarray(io._get_y())
  pred_cols = io._get_pred_columns(df_X, task)
  if not pred_cols:
    raise RuntimeError("no per-descriptor prediction columns to pool")
  P = np.asarray(df_X[pred_cols], dtype=np.float64)
  R, _ = io._get_rank_matrix(df_X, pred_cols)
  A, _ = io._get_ad_matrix(df_X, pred_cols)
  S, _ = io._get_signal_matrix(df_X, pred_cols, "_raw") if clf else (None, False)
  meta = io._get_descriptor_meta(pred_cols, task)

  tr, te = np.asarray(train_idx), np.asarray(test_idx)
  params, _ = pooler.fit(
    P[tr],
    _slice(R, tr),
    _slice(S, tr),
    _slice(A, tr),
    y[tr],
    pred_cols,
    meta,
    kappa=_kappa_cutoff if clf else None,
    task=task,
  )
  pos = [pred_cols.index(c) for c in params["descriptor_columns"]]

  def held_out(M):
    return None if M is None else M[te][:, pos]

  result = pooler.predict(params, held_out(P), held_out(R), held_out(S), held_out(A))
  if not clf:
    yhat_test = result.values["value"]
    return fetcher.regression_performance_report(y[te], yhat_test), y[te], yhat_test
  yhat_test = result.values["proba"][:, 1]
  report = fetcher.classification_performance_report(
    y[te], yhat_test, CUTOFF_KAPPA, params["cutoffs"][CUTOFF_KAPPA]
  )
  at_default = fetcher.classification_performance_report(
    y[te], yhat_test, CUTOFF_DEFAULT, DEFAULT_CUTOFF
  )
  for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "mcc"):
    report[f"{k}_{CUTOFF_DEFAULT}"] = at_default[k]
  # Also hand back the held-out (truth, score) arrays so the caller can persist them for the
  # report's per-strategy ROC/PR/calibration curves (only scalar metrics survive otherwise).
  return report, y[te], yhat_test


def run_one_fold(
  model_dir,
  fold_name,
  spec,
  model_ids,
  fetcher,
  batch_size=None,
  substep_cb=None,
  proxy_scores=None,
):
  """Run a single fold and return its metrics record (also written to ``folds/<name>/metrics.json``).

  ``substep_cb(text)`` — optional callback fired with the current sub-phase (per-descriptor fit, then
  pooling/scoring) so a live monitor can show intra-fold progress.
  """
  fold_path = build_fold_workspace(model_dir, fold_name, model_ids, proxy_scores)
  _fit_descriptors(fold_path, model_ids, spec["train_idx"], batch_size, substep_cb=substep_cb)
  if substep_cb:
    substep_cb("pooling + scoring")
  metrics, y_true, y_score = _score_fold(
    fold_path, spec["train_idx"], spec["test_idx"], fetcher, batch_size
  )
  record = {
    "fold": fold_name,
    "strategy": spec["strategy"],
    "seed": spec.get("seed"),
    **metrics,
  }
  record = {k: _plain(v) for k, v in record.items()}
  # Per-compound held-out truth+score, kept off the scalar metrics.json (persisted separately by
  # io.write_validation_outputs into report/validation_predictions.csv for the per-strategy curves).
  # The truth is a 0/1 label for classification and a target value for regression, so it keeps its
  # own type -- rounding a continuous target to an int would silently corrupt the file.
  truth = np.asarray(y_true).ravel()
  is_clf = "auroc" in metrics
  record["_y_true"] = [int(v) for v in truth] if is_clf else [float(v) for v in truth]
  record["_y_score"] = [float(v) for v in np.asarray(y_score).ravel()]
  _save_fold_record(fold_path, record)
  return record


#: The fold's per-compound held-out truth and score. Kept beside ``metrics.json`` rather than in
#: it, because that file is the scalar, readable view of a fold and these are as long as the
#: held-out slice -- but they are what the report's per-strategy curves are drawn from, so a fold
#: without them is not a fold that can be reused.
HELD_OUT_FILENAME = "held_out.json"


def _save_fold_record(fold_path, record):
  """Persist a finished fold: the per-compound arrays first, then the scalars.

  In that order deliberately. ``metrics.json`` is written last and atomically, so its presence
  means the arrays beside it are already complete -- which is what lets a re-run tell a finished
  fold from one that was interrupted half way through writing itself.
  """
  from zairachem.base import _atomic_json

  _atomic_json(
    os.path.join(fold_path, HELD_OUT_FILENAME),
    {"_y_true": record["_y_true"], "_y_score": record["_y_score"]},
  )
  _atomic_json(
    os.path.join(fold_path, "metrics.json"),
    {k: v for k, v in record.items() if not k.startswith("_")},
  )


def load_fold_record(model_dir, fold_name):
  """A finished fold's full record, or None when it was never finished.

  A held-out fold costs a complete refit of the pipeline -- measured, around half an hour each --
  and its result is already on disk. A run interrupted at fold seven of ten used to discard the
  six it had paid for, because nothing ever read this back.
  """
  fold_path = os.path.join(model_dir, FOLDS_SUBFOLDER, fold_name)
  try:
    with open(os.path.join(fold_path, "metrics.json")) as f:
      record = json.load(f)
    with open(os.path.join(fold_path, HELD_OUT_FILENAME)) as f:
      held_out = json.load(f)
  except (OSError, ValueError):
    return None
  y_true, y_score = held_out.get("_y_true"), held_out.get("_y_score")
  if not isinstance(y_true, list) or len(y_true) != len(y_score or []):
    return None
  if record.get("fold") != fold_name:
    return None
  record["_y_true"], record["_y_score"] = y_true, y_score
  return record
