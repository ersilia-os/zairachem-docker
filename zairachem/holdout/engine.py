"""Execute one held-out fold end to end: fit descriptors on train, pool, score the held-out slice.

The pooled classifier is the product we score. For each fold: re-fit every descriptor's estimator on
the train slice (predictions for all rows fall out of the parent Fitter), stack them, fit the
reliability pooler on the train slice, then combine on the held-out slice — giving an honest pooled
prediction on molecules the fold never trained on. Descriptors are read from the shared workspace, so
no featurization is repeated.
"""

import json, os, joblib
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.vars import ESTIMATORS_SUBFOLDER, POOL_SUBFOLDER, Y_HAT_FILE
from zairachem.estimate.estimators.lazy_qsar import ESTIMATORS_FAMILY_SUBFOLDER
from zairachem.estimate.estimators.lazy_qsar.assemble import IndividualOutcomeAssembler
from zairachem.pool.base import BasePooler
from zairachem.pool.reliability_pooler.pooler import ReliabilityClassifierPooler
from zairachem.holdout.indices import HoldoutFitter
from zairachem.holdout.workspace import build_fold_workspace


def _slice(matrix, idx):
  return matrix[idx] if matrix is not None else None


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
  """Fit the reliability pooler on train, combine on the held-out slice, score with the shared metrics."""
  io = BasePooler(path=fold_path, batch_size=batch_size)
  df_X = io._get_X()  # per-descriptor predictions for all rows (also writes model/pool/data.csv)
  y = np.asarray(io._get_y())
  pred_cols = io._get_pred_columns(df_X, "classification")
  if not pred_cols:
    raise RuntimeError("no per-descriptor prediction columns to pool")
  P = np.asarray(df_X[pred_cols], dtype=np.float64)
  R, _ = io._get_rank_matrix(df_X, pred_cols)
  A, _ = io._get_ad_matrix(df_X, pred_cols)
  oof_aucs, curves, cutoffs = io._get_descriptor_meta(pred_cols, "classification")

  tr, te = np.asarray(train_idx), np.asarray(test_idx)
  pool_dir = os.path.join(fold_path, POOL_SUBFOLDER)
  os.makedirs(pool_dir, exist_ok=True)
  pooler = ReliabilityClassifierPooler(pool_dir)
  yhat_train = pooler.fit(
    P[tr], _slice(R, tr), _slice(A, tr), y[tr], oof_aucs, curves, cutoffs, pred_cols
  )
  yhat_test, _ = ReliabilityClassifierPooler._combine_verbose(
    P[te], _slice(R, te), _slice(A, te), pooler._params
  )
  return fetcher.classification_performance_report(
    y[tr], np.asarray(yhat_train), y[te], np.asarray(yhat_test)
  )


def run_one_fold(model_dir, fold_name, spec, model_ids, fetcher, batch_size=None, substep_cb=None):
  """Run a single fold and return its metrics record (also written to ``folds/<name>/metrics.json``).

  ``substep_cb(text)`` — optional callback fired with the current sub-phase (per-descriptor fit, then
  pooling/scoring) so a live monitor can show intra-fold progress.
  """
  fold_path = build_fold_workspace(model_dir, fold_name)
  _fit_descriptors(fold_path, model_ids, spec["train_idx"], batch_size, substep_cb=substep_cb)
  if substep_cb:
    substep_cb("pooling + scoring")
  metrics = _score_fold(fold_path, spec["train_idx"], spec["test_idx"], fetcher, batch_size)
  record = {
    "fold": fold_name,
    "strategy": spec["strategy"],
    "seed": spec.get("seed"),
    **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()},
  }
  with open(os.path.join(fold_path, "metrics.json"), "w") as f:
    json.dump(record, f, indent=2, default=float)
  return record
