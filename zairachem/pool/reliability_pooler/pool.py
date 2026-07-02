import os
import json
import joblib
import numpy as np
import pandas as pd

from ..base import BasePooler
from .pooler import ReliabilityClassifierPooler

from zairachem.base import ZairaBase
from zairachem.base.utils.logging import logger
from zairachem.base.utils.matrices import DEFAULT_CHUNK_SIZE
from zairachem.base.vars import POOL_SUBFOLDER

#: Summary written by the pooler and rendered under the Pool step banner (see progress.py).
SUMMARY_FILENAME = "reliability_summary.json"


def _check_regression(task):
  if task == "regression":
    raise NotImplementedError(
      "The reliability pooler is classification-only (the lazy_qsar estimator has no working "
      "regression head). Use --pooler bagger for the legacy regression path."
    )


def _prettify(col):
  """Display name for a prediction column ('<family>-<model_id>-clf' → '<model_id>')."""
  for suf in ("-clf", "-reg"):
    if col.endswith(suf):
      col = col[: -len(suf)]
      break
  return col.split("-")[-1] if "-" in col else col


def _tier(has_rank, has_ad, base_scores):
  if has_rank and has_ad:
    return "full (rank reliability + applicability domain)"
  if has_ad:
    return "applicability-domain weighted"
  if has_rank:
    return "rank-reliability weighted"
  return "skill weighted" if any(b > 0 for b in (base_scores or [])) else "uniform mean"


def _veto_stats(A, ad_hard_cutoffs):
  """(per-descriptor vetoed counts, n_fully_out_of_domain) from AD scores vs per-descriptor cutoffs."""
  if A is None or ad_hard_cutoffs is None:
    return None, 0
  A = np.asarray(A, dtype=np.float64)
  cut = np.asarray(ad_hard_cutoffs, dtype=np.float64)
  B, D = A.shape
  vetoed = np.zeros((B, D), dtype=bool)
  for j in range(D):
    if np.isfinite(cut[j]):
      vetoed[:, j] = np.isfinite(A[:, j]) & (A[:, j] < cut[j])
  per = vetoed.sum(axis=0).astype(int).tolist()
  has_cut = np.isfinite(cut)
  n_fully = int(vetoed[:, has_cut].all(axis=1).sum()) if has_cut.any() else 0
  return per, n_fully


def _write_summary(out_path, mode, pred_cols, params, P, R, A, W):
  """Persist a compact, human-readable summary of how the pooler combined the descriptors.

  Rendered on-screen under the Pool step banner (progress.py) — at predict time this is where the
  user learns how many query compounds fell outside the applicability domain. Best-effort.
  """
  try:
    B, D = P.shape
    names = [_prettify(c) for c in pred_cols]
    has_rank = bool(params.get("has_rank"))
    has_ad = bool(params.get("has_ad"))
    base_scores = list(params.get("base_scores") or [])
    mean_w = W.mean(axis=0)
    summary = {
      "pooler": "reliability",
      "mode": mode,
      "n_compounds": int(B),
      "n_descriptors": int(D),
      "descriptors": names,
      "tier": _tier(has_rank, has_ad, base_scores),
      "has_rank": has_rank,
      "has_ad": has_ad,
      "mean_weights": {n: round(float(w), 4) for n, w in zip(names, mean_w)},
    }
    if has_ad:
      per, n_fully = _veto_stats(A, params.get("ad_hard_cutoffs"))
      summary["applicability"] = {
        "n_out_of_domain": int(n_fully),
        "frac_out_of_domain": round(float(n_fully) / B, 4) if B else 0.0,
        "per_descriptor_out_of_domain": (
          {n: int(c) for n, c in zip(names, per)} if per is not None else {}
        ),
      }
    pool_dir = os.path.join(out_path, POOL_SUBFOLDER)
    os.makedirs(pool_dir, exist_ok=True)
    with open(os.path.join(pool_dir, SUMMARY_FILENAME), "w") as f:
      json.dump(summary, f, indent=2)
    # File-log line (also visible on-screen with --verbose); the durable on-screen surface is the
    # Pool step detail block rendered from this artifact.
    ad_txt = ""
    if has_ad and "applicability" in summary:
      ad_txt = f" | out-of-domain {summary['applicability']['n_out_of_domain']}/{B}"
    logger.info(
      f"[reliability_pool:{mode}] {D} descriptors · {B} compounds · {summary['tier']}{ad_txt}"
    )
  except Exception as e:
    logger.debug(f"[reliability_pool:{mode}] could not write summary: {e}")


class Fitter(BasePooler):
  def __init__(self, path, batch_size=None):
    BasePooler.__init__(self, path=path, batch_size=batch_size)
    self.trained_path = os.path.join(self.get_output_dir(), POOL_SUBFOLDER)

  def run(self):
    self.reset_time()
    _check_regression(self.task)
    valid_idxs = self.get_validation_indices(path=self.path)
    cids = self._get_compound_ids()
    df_X = self._get_X()
    y = np.asarray(self._get_y())

    pred_cols = self._get_pred_columns(df_X, self.task)
    if len(pred_cols) == 0:
      raise RuntimeError("[reliability_pool] No per-descriptor prediction columns found to pool.")

    P = np.asarray(df_X[pred_cols], dtype=np.float64)
    R, _ = self._get_rank_matrix(df_X, pred_cols)
    A, _ = self._get_ad_matrix(df_X, pred_cols)
    oof_aucs, curves, cutoffs = self._get_descriptor_meta(pred_cols, self.task)
    logger.info(
      f"[reliability_pool:fit] {len(pred_cols)} descriptors | rank={'y' if R is not None else 'n'} "
      f"ad={'y' if A is not None else 'n'}"
    )

    # Slice every aligned matrix by the SAME validation indices so rows stay paired.
    idx = np.asarray(valid_idxs)
    P_v = P[idx]
    R_v = R[idx] if R is not None else None
    A_v = A[idx] if A is not None else None
    y_v = y[idx]
    cids_v = [cids[i] for i in idx]

    pooler = ReliabilityClassifierPooler(self.trained_path)
    y_hat = pooler.fit(P_v, R_v, A_v, y_v, oof_aucs, curves, cutoffs, pred_cols)
    _, W = ReliabilityClassifierPooler._combine_verbose(P_v, R_v, A_v, pooler._params)
    _write_summary(self.path, "fit", pred_cols, pooler._params, P_v, R_v, A_v, W)
    b_hat = (np.asarray(y_hat) > 0.5).astype(int)

    results = pd.DataFrame({"clf": y_hat, "clf_bin": b_hat})
    columns = results.columns.tolist()
    results["compound_id"] = cids_v
    results = results[["compound_id"] + columns]
    self.update_elapsed_time()
    return results


class Predictor(BasePooler):
  def __init__(self, path, batch_size=None):
    BasePooler.__init__(self, path=path, batch_size=batch_size)
    self.trained_path = os.path.join(self.get_trained_dir(), POOL_SUBFOLDER)

  def run(self):
    self.reset_time()
    _check_regression(self.task)
    df = self._get_X()
    cids = self._get_compound_ids()

    pred_cols = self._get_pred_columns(df, self.task)
    params = joblib.load(os.path.join(self.trained_path, ReliabilityClassifierPooler.ARTIFACT_NAME))
    saved = params["descriptor_columns"]
    # Reconcile descriptor drift: keep saved-order columns that are present now; renormalize over
    # them. A descriptor present at fit but absent now is dropped (no base score to apply).
    keep = [c for c in saved if c in pred_cols]
    if len(keep) == 0:
      raise RuntimeError(
        "[reliability_pool] None of the trained descriptors are available at predict."
      )
    if keep != saved:
      logger.warning(
        f"[reliability_pool:predict] descriptor drift: using {len(keep)}/{len(saved)} trained descriptors"
      )
    pos = [saved.index(c) for c in keep]

    P = np.asarray(df[keep], dtype=np.float64)
    R, _ = self._get_rank_matrix(df, keep) if params.get("has_rank") else (None, False)
    A, _ = self._get_ad_matrix(df, keep) if params.get("has_ad") else (None, False)

    reduced = dict(params)
    reduced["descriptor_columns"] = keep
    reduced["base_scores"] = [params["base_scores"][p] for p in pos]
    if params.get("rank_error_curves") is not None:
      reduced["rank_error_curves"] = [params["rank_error_curves"][p] for p in pos]
    if params.get("ad_hard_cutoffs") is not None:
      reduced["ad_hard_cutoffs"] = [params["ad_hard_cutoffs"][p] for p in pos]

    y_hat, W = ReliabilityClassifierPooler._combine_verbose(P, R, A, reduced)
    _write_summary(self.path, "predict", keep, reduced, P, R, A, W)
    b_hat = (np.asarray(y_hat) > 0.5).astype(int)

    results = pd.DataFrame({"clf": y_hat, "clf_bin": b_hat})
    columns = results.columns.tolist()
    results["compound_id"] = cids
    results = results[["compound_id"] + columns]
    self.update_elapsed_time()
    return results


class ReliabilityPooler(ZairaBase):
  def __init__(self, path=None, batch_size=None):
    ZairaBase.__init__(self)
    self.batch_size = batch_size or DEFAULT_CHUNK_SIZE
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    if not self.is_predict():
      self.estimator = Fitter(path=self.path, batch_size=self.batch_size)
    else:
      self.estimator = Predictor(path=self.path, batch_size=self.batch_size)

  def run(self):
    return self.estimator.run()
