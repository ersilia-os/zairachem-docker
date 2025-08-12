import time
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Any, Dict, Sequence, Optional


def make_classification_report(
  y: Sequence[int],
  y_proba: Sequence[float],
  threshold: float = 0.5,
  best_config: Optional[Dict[str, Any]] = None,
  train_time: Optional[float] = None,
) -> Dict[str, Any]:
  y_hat = y_proba[:, 1]

  t0 = time.time()
  predict_time = time.time() - t0

  return {
    "main": {"idxs": None, "y": y, "y_hat": y_hat},
    "best_config": best_config or {},
    "train_time": train_time or 0.0,
    "predict_time": predict_time,
  }
