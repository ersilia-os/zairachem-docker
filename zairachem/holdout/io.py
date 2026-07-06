"""Aggregate per-fold metrics into the report's held-out validation artifacts.

Both outputs are written under ``report/`` so they survive the ``finish`` cleanup (which discards the
descriptors and the fold workspaces) and feed the report's validation plot + table.
"""

import csv, json, os
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  REPORT_SUBFOLDER,
  VALIDATION_PREDICTIONS_FILENAME,
  VALIDATION_TABLE_FILENAME,
)

HOLDOUT_SUMMARY_FILENAME = "holdout_summary.json"

# Report/display order of the schemas (matches the fold execution order).
_SCHEMA_ORDER = ("random", "scaffold", "scaffold_det", "butina")


def _stats(values):
  arr = np.asarray(values, dtype=float)
  arr = arr[~np.isnan(arr)]
  if arr.size == 0:
    return {"mean": None, "std": None, "values": []}
  return {
    "mean": float(arr.mean()),
    "std": float(arr.std(ddof=0)),
    "values": [float(v) for v in arr],
  }


def write_validation_outputs(model_dir, folds, records):
  """Write ``report/validation_table.csv`` (one row per fold) and ``report/holdout_summary.json``.

  Parameters
  ----------
  model_dir : str
    The run directory.
  folds : dict
    The fold definitions from ``metadata/splits.json`` (used for the total expected count).
  records : list of dict
    Per-fold metrics returned by :func:`zairachem.holdout.engine.run_one_fold`.
  """
  report_dir = os.path.join(model_dir, REPORT_SUBFOLDER)
  os.makedirs(report_dir, exist_ok=True)

  table_cols = ["strategy", "fold", "seed", "auroc", "aupr", "num_train", "num_test", "num_test_1"]
  with open(os.path.join(report_dir, VALIDATION_TABLE_FILENAME), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=table_cols, extrasaction="ignore")
    writer.writeheader()
    for r in records:
      writer.writerow(r)

  # Per-compound held-out (truth, score), long form — feeds the report's per-strategy ROC/PR/
  # calibration/enrichment curves (the scalar table above can't reconstruct curves).
  with open(os.path.join(report_dir, VALIDATION_PREDICTIONS_FILENAME), "w", newline="") as f:
    pw = csv.writer(f)
    pw.writerow(["strategy", "fold", "y_true", "y_score"])
    for r in records:
      yt, ys = r.get("_y_true") or [], r.get("_y_score") or []
      for t, s in zip(yt, ys):
        pw.writerow([r["strategy"], r["fold"], t, s])

  # Keep the big per-compound arrays out of the JSON summary (they live in the predictions CSV).
  records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]

  by_strategy = {}
  for r in records:
    by_strategy.setdefault(r["strategy"], []).append(r)
  strategies = {}
  for strat in [s for s in _SCHEMA_ORDER if s in by_strategy] + [
    s for s in by_strategy if s not in _SCHEMA_ORDER
  ]:
    rows = by_strategy[strat]
    strategies[strat] = {
      "n_folds": len(rows),
      "auroc": _stats([r.get("auroc") for r in rows]),
      "aupr": _stats([r.get("aupr") for r in rows]),
    }
  summary = {
    "n_folds_defined": len(folds),
    "n_folds_run": len(records),
    "strategies": strategies,
    "per_fold": records,
  }
  with open(os.path.join(report_dir, HOLDOUT_SUMMARY_FILENAME), "w") as f:
    json.dump(summary, f, indent=2, default=float)
  logger.success(
    f"[evaluate] Held-out validation: {len(records)}/{len(folds)} folds scored "
    f"→ {VALIDATION_TABLE_FILENAME}"
  )
