"""Descriptor pre-screening step: choose at most ``max_descriptors`` descriptors to fully train.

Runs after ``treat`` (needs the treated matrices) and before ``estimate``. Each computed descriptor
gets a quick held-out score from a shallow proxy model on the chemistry-aware folds (random /
scaffold / Butina at 3 repeats, built here whatever ``--evaluate`` says), and descriptors are added
greedily while they improve the ensemble (:func:`~zairachem.screen.proxy.select_descriptors`). The
choice goes to ``metadata/selected_eos.json``, which ``effective_descriptors`` applies to estimate,
pool, report and predict, with the scores in ``metadata/proxy_scores.json``. Held-out folds
(``--evaluate``) repeat the selection on their own training rows.

The score is AUROC for a classification run and Spearman's rho for a regression one; both tasks
screen.

Self-gated: a no-op at predict, or when there is nothing to prune (``max_descriptors ≥
#descriptors``). On by default (``--max-descriptors`` defaults to ``DEFAULT_MAX_DESCRIPTORS``, 5).
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
  TARGET_COLUMN,
)

#: Screening always scores on the standard 10-fold menu (independent of --evaluate / --repeats).
SCREEN_REPEATS = 3


def candidates(path):
  """Descriptors computed for the run (``descriptors/done_eos.json``), the screening candidates."""
  with open(os.path.join(path, DESCRIPTORS_SUBFOLDER, "done_eos.json")) as f:
    return list(json.load(f))


def screening_applies(path, params):
  """Whether a fit run screens its descriptors: it has more than ``max_descriptors`` of them."""
  k = params.get("max_descriptors")
  if not k:
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

    A no-op at predict, or when ``max_descriptors`` is not set or exceeds the descriptors there
    are. Both tasks screen; the score is the task's own.
    """
    k = self.params.get("max_descriptors")
    if self.is_predict() or not k:
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

    from zairachem.base.vars import RANDOM_SEED
    from zairachem.holdout.splits import build_fold_definitions
    from zairachem.screen.proxy import (
      PROXY_FLOOR,
      PROXY_MAX_ROWS,
      screening_rows,
      select_descriptors,
    )

    task = self.params["task"]
    df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
    y = np.asarray(df[TARGET_COLUMN[task]])
    logger.info(
      f"[screen] Scoring {len(pool)} descriptors on the evaluate splits "
      f"({SCREEN_REPEATS} repeats, all schemas) → greedily keeping up to {k} that add to the pool"
    )
    # Thin the run down to the rows the proxy will score BEFORE the folds are built, not after:
    # the fold menu costs a Bemis-Murcko scaffold per molecule and a pass over every label per
    # candidate seed, and there is no reason to spend either on rows that will not be scored. On
    # a set at or below the cap this is every row, and the folds are the ones it always built.
    rows = screening_rows(y, task, PROXY_MAX_ROWS, RANDOM_SEED)
    smiles = list(df[SMILES_COLUMN])
    folds = build_fold_definitions([smiles[i] for i in rows], list(y[rows]), repeats=SCREEN_REPEATS)
    fold_pairs = [(f["train_idx"], f["test_idx"]) for f in folds.values()]
    selected, scores, gains = select_descriptors(
      os.path.join(self.path, DESCRIPTORS_SUBFOLDER), pool, y[rows], fold_pairs, k, task, rows=rows
    )
    meta_dir = os.path.join(self.path, METADATA_SUBFOLDER)
    with open(os.path.join(meta_dir, SELECTED_EOS_FILENAME), "w") as f:
      json.dump(selected, f, indent=4)
    with open(os.path.join(meta_dir, PROXY_SCORES_FILENAME), "w") as f:
      json.dump(scores, f, indent=4)
    self._print_table(scores, set(selected), PROXY_FLOOR[task], task, gains)
    logger.success(
      f"[screen] Selected {len(selected)} of {len(pool)} descriptors for full training."
    )
    step.update()

  @staticmethod
  def _print_table(scores, selected, floor, task="classification", gains=None):
    """Descriptors ranked by proxy skill, colored by score, with the selected ones marked.

    The solo skill is not what decided anything: selection is greedy on the ensemble, so a strong
    descriptor that repeats the seed is dropped and a weaker one that complements it is kept.
    "Adds" is the gain that actually made the call, and a descriptor kept on a hair's breadth of it
    is worth knowing about — training it can cost more than every other descriptor put together.
    """
    gains = gains or {}
    table = themed_table("Descriptor pre-screening")
    measure = "Held-out AUROC" if task == "classification" else "Held-out rho"
    for col, just in (
      ("Descriptor", "left"),
      (measure, "right"),
      ("Adds", "right"),
      ("Kept", "center"),
    ):
      table.add_column(col, justify=just, no_wrap=True)
    for eos_id in sorted(scores, key=lambda e: scores[e], reverse=True):
      auroc = scores[eos_id]
      # Color high→green, low→red, over each task's useful range (AUROC 0.5-1, rho 0-1).
      span = 0.5 if task == "classification" else 1.0
      color = heat_hex(max(0.0, min(1.0, (1.0 - auroc) / span)))
      keep = eos_id in selected
      if eos_id not in gains:
        adds = "[dim]·[/]"
      elif gains[eos_id] is None:
        adds = "[dim]seed[/]"
      else:
        adds = f"[dim]+{gains[eos_id]:.3f}[/]"
      table.add_row(
        f"[{'bold' if keep else 'dim'}]{eos_id}[/]",
        f"[{color}]{auroc:.3f}[/]",
        adds,
        "[#3fb950]✓[/]" if keep else "[dim]·[/]",
      )
    console.print(table)
