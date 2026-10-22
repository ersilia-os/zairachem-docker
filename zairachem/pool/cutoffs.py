"""The run's named classification cutoffs, written to ``pool/cutoffs.json`` (``results/`` at finish).

``screening`` (the call), ``kappa`` and ``default`` are cutoffs on the pooled probability, decided
when the pooled model is fitted; ``balanced`` is one per descriptor, from its lazy-qsar CV report.
Each is given as a probability and as a rank, the scale of the score in ``output.csv``: the
pooled ones against the pooled reference, each balanced one against its descriptor's reference
(lazy-qsar's single-descriptor rank). A predict run writes the trained model's.
"""

import json
import os

from zairachem.base.vars import (
  CUTOFF_BALANCED,
  CUTOFF_DEFAULT,
  CUTOFF_KAPPA,
  CUTOFF_SCREENING,
  CUTOFFS_FILENAME,
  DEFAULT_CUTOFF,
  POOL_SUBFOLDER,
)

_ABOUT = {
  CUTOFF_SCREENING: {
    "scope": "pooled",
    "computed_from": "the pooled probability of the rank-reference library at its 99th "
    "percentile (rank 0.65, the top 1% of drug-like chemical space)",
    "used_for": "the call (call column of the output table); performance metrics",
  },
  CUTOFF_KAPPA: {
    "scope": "pooled",
    "computed_from": "the pooled out-of-fold scores of the training molecules, maximising "
    "Cohen's kappa (GHOST)",
    "used_for": "performance metrics",
  },
  CUTOFF_BALANCED: {
    "scope": "per descriptor",
    "computed_from": "each descriptor's out-of-fold scores, maximising balanced accuracy "
    "(lazy-qsar)",
    "used_for": "per-descriptor performance metrics",
  },
  CUTOFF_DEFAULT: {
    "scope": "pooled and per descriptor",
    "computed_from": "fixed",
    "used_for": "per-descriptor calls; the pooled call when there is no rank reference",
  },
}


def cutoffs_document(pooled, balanced, ranks=None):
  """The cutoffs file's content: one entry per named cutoff.

  Parameters
  ----------
  pooled : dict
      ``{"screening": proba or None, "kappa": proba or None}``, the pooled model's cutoffs.
  balanced : dict
      ``{descriptor: proba or None}``, each descriptor's balanced-accuracy cutoff.
  ranks : dict, optional
      The same cutoffs as ranks (``"balanced"`` per descriptor); None entries when unknown.
  """
  ranks = ranks or {}
  doc = {
    CUTOFF_SCREENING: {"proba": pooled.get(CUTOFF_SCREENING), "rank": ranks.get(CUTOFF_SCREENING)},
    CUTOFF_KAPPA: {"proba": pooled.get(CUTOFF_KAPPA), "rank": ranks.get(CUTOFF_KAPPA)},
    CUTOFF_BALANCED: {
      "proba": dict(balanced),
      "rank": {d: (ranks.get(CUTOFF_BALANCED) or {}).get(d) for d in balanced},
    },
    CUTOFF_DEFAULT: {
      "proba": DEFAULT_CUTOFF,
      "rank": ranks.get(CUTOFF_DEFAULT),
      "rank_by_descriptor": {
        d: (ranks.get("default_by_descriptor") or {}).get(d) for d in balanced
      },
    },
  }
  for name, entry in doc.items():
    entry.update(_ABOUT[name])
  return doc


def write_cutoffs(path, pooled, balanced, ranks=None):
  """Write ``<path>/pool/cutoffs.json`` (see :func:`cutoffs_document`)."""
  folder = os.path.join(path, POOL_SUBFOLDER)
  os.makedirs(folder, exist_ok=True)
  with open(os.path.join(folder, CUTOFFS_FILENAME), "w") as f:
    json.dump(cutoffs_document(pooled, balanced, ranks), f, indent=2)


def call_cutoff(path):
  """``(name, probability)`` of the run's pooled call: ``screening``, or ``default`` without it."""
  screening = ((read_cutoffs(path) or {}).get(CUTOFF_SCREENING) or {}).get("proba")
  if screening is None:
    return CUTOFF_DEFAULT, DEFAULT_CUTOFF
  return CUTOFF_SCREENING, screening


def read_cutoffs(path):
  """The run's cutoffs (``pool/cutoffs.json``), or None when it has none."""
  fp = os.path.join(path, POOL_SUBFOLDER, CUTOFFS_FILENAME)
  if not os.path.exists(fp):
    return None
  with open(fp) as f:
    return json.load(f)
