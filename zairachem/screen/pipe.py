"""Descriptor pre-screening step: choose at most ``max_descriptors`` descriptors to fully train.

Runs after ``treat`` (needs the treated matrices) and before ``estimate``. Each computed descriptor
gets a quick held-out AUROC from a shallow proxy model on the chemistry-aware folds (random /
scaffold / Butina at 3 repeats, built here whatever ``--evaluate`` says), and descriptors are added
greedily while they improve the ensemble (:func:`~zairachem.screen.proxy.select_descriptors`). The
choice goes to ``metadata/selected_eos.json``, which ``effective_descriptors`` applies to estimate,
pool, report and predict, with the scores in ``metadata/proxy_scores.json``. Held-out folds
(``--evaluate``) repeat the selection on their own training rows.

Self-gated: a no-op at predict, for regression, or when there is nothing to prune
(``max_descriptors ≥ #descriptors``). On by default (``--max-descriptors`` defaults to
``DEFAULT_MAX_DESCRIPTORS``, 5).
"""

import json
import os

import numpy as np
import pandas as pd

from zairachem.base import ZairaBase
from zairachem.base.utils.console import console, heat_hex, themed_table
from zairachem.base.utils.logging import logger
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  METADATA_SUBFOLDER,
  PROXY_SCORES_FILENAME,
  SELECTED_EOS_FILENAME,
  SMILES_COLUMN,
)

#: Screening always scores on the standard 10-fold menu (independent of --evaluate / --repeats).
SCREEN_REPEATS = 3


def candidates(path):
  """Descriptors computed for the run (``descriptors/done_eos.json``), the screening candidates."""
  with open(os.path.join(path, DESCRIPTORS_SUBFOLDER, "done_eos.json")) as f:
    return list(json.load(f))


def screening_applies(path, params):
  """Whether a fit run screens its descriptors: classification with more than ``max_descriptors``."""
  k = params.get("max_descriptors")
  if params.get("task") != "classification" or not k:
    return False
  return len(candidates(path)) > k


class ScreenPipeline(ZairaBase):
  """Pre-screen the descriptors and keep the best ``max_descriptors`` for training.

  Parameters
  ----------
  path : str, optional
    Run folder.
  batch_size : int, optional
    Unused; accepted for symmetry with the other pipeline steps.
  """

  def __init__(self, path=None, batch_size=None):
    ZairaBase.__init__(self)
    self.path = path
    self.params = self._load_params()

  def run(self):
    """Score each descriptor on held-out folds and write the selection.

    A no-op at predict, for regression, or when ``max_descriptors`` is not set.
    """
    k = self.params.get("max_descriptors")
    if self.is_predict() or self.params.get("task") != "classification" or not k:
      return
    step = PipelineStep("screen", self.path)
    if step.is_done():
      logger.info("[screen] Descriptor pre-screening already done — skipping.")
      return
    pool = candidates(self.path)
    if not screening_applies(self.path, self.params):
      logger.info(f"[screen] {len(pool)} descriptors ≤ max ({k}); training all — no pruning.")
      step.update()
      return

    from zairachem.holdout.splits import build_fold_definitions
    from zairachem.screen.proxy import PROXY_FLOOR, select_descriptors

    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    y = np.asarray(df["bin"])
    logger.info(
      f"[screen] Scoring {len(pool)} descriptors on the evaluate splits "
      f"({SCREEN_REPEATS} repeats, all schemas) → greedily keeping up to {k} that add to the pool"
    )
    folds = build_fold_definitions(list(df[SMILES_COLUMN]), list(y), repeats=SCREEN_REPEATS)
    fold_pairs = [(f["train_idx"], f["test_idx"]) for f in folds.values()]
    selected, scores = select_descriptors(
      os.path.join(self.path, DESCRIPTORS_SUBFOLDER), pool, y, fold_pairs, k
    )
    meta_dir = os.path.join(self.path, METADATA_SUBFOLDER)
    with open(os.path.join(meta_dir, SELECTED_EOS_FILENAME), "w") as f:
      json.dump(selected, f, indent=4)
    with open(os.path.join(meta_dir, PROXY_SCORES_FILENAME), "w") as f:
      json.dump(scores, f, indent=4)
    self._print_table(scores, set(selected), PROXY_FLOOR)
    logger.success(
      f"[screen] Selected {len(selected)} of {len(pool)} descriptors for full training."
    )
    step.update()

  @staticmethod
  def _print_table(scores, selected, floor):
    """Descriptors ranked by proxy AUROC, colored by score, with the selected ones marked."""
    table = themed_table("Descriptor pre-screening")
    for col, just in (("Descriptor", "left"), ("Held-out AUROC", "right"), ("Kept", "center")):
      table.add_column(col, justify=just, no_wrap=True)
    for eos_id in sorted(scores, key=lambda e: scores[e], reverse=True):
      auroc = scores[eos_id]
      # Color high→green, low→red: map AUROC in [0.5, 1] onto the green→red ramp (1.0→green, ≤0.5→red).
      color = heat_hex(max(0.0, min(1.0, (1.0 - auroc) / 0.5)))
      keep = eos_id in selected
      table.add_row(
        f"[{'bold' if keep else 'dim'}]{eos_id}[/]",
        f"[{color}]{auroc:.3f}[/]",
        "[#3fb950]✓[/]" if keep else "[dim]·[/]",
      )
    console.print(table)
