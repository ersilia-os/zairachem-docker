"""Held-out validation pipeline step: run every fold defined at setup and aggregate the results.

Self-gated: does nothing unless this is a fit run with ``--evaluate`` (``params["evaluate"]``) and a
``metadata/splits.json`` produced at setup. Runs after pooling so the shared descriptors exist; the
production model (trained on all rows) is untouched.
"""

import json, os

from zairachem.base import ZairaBase
from zairachem.base.utils.console import echo
from zairachem.base.utils.logging import logger
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.progress import STEP_COLORS
from zairachem.base.vars import (
  DESCRIPTORS_SUBFOLDER,
  METADATA_SUBFOLDER,
  SPLITS_FILENAME,
)
from zairachem.holdout.monitor import EvaluateMonitor, _fmt_duration


class HoldoutValidationPipeline(ZairaBase):
  def __init__(self, path=None, batch_size=None, est_seconds=None):
    ZairaBase.__init__(self)
    self.path = path or self.get_output_dir()
    self.batch_size = batch_size
    # Per-fold time predictor: the just-finished training step's wall-clock (a fold refits the same
    # descriptor stack). Used for the upfront estimate and the live ETA.
    self.est_seconds = est_seconds
    self.params = self._load_params()

  def _model_ids(self):
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json")) as f:
      return list(json.load(f))

  def run(self):
    if self.is_predict() or not self.params.get("evaluate"):
      return
    step = PipelineStep("holdout", self.path)
    if step.is_done():
      logger.info("[evaluate] Held-out validation already done — skipping.")
      return
    splits_path = os.path.join(self.path, METADATA_SUBFOLDER, SPLITS_FILENAME)
    if not os.path.exists(splits_path):
      logger.warning(f"[evaluate] No {SPLITS_FILENAME} found — skipping held-out validation.")
      return
    # Imported lazily: these pull in the estimator/pool/report stacks, kept out of the import path of
    # a plain fit that never evaluates.
    from zairachem.report.fetcher import ResultsFetcher
    from zairachem.holdout.engine import run_one_fold
    from zairachem.holdout.io import write_validation_outputs

    with open(splits_path) as f:
      folds = json.load(f)
    model_ids = self._model_ids()
    fetcher = ResultsFetcher(path=self.path)

    n = len(folds)
    if self.est_seconds and self.est_seconds > 5:
      echo(
        f"Held-out validation — {n} folds, ~{_fmt_duration(self.est_seconds)} each (based on "
        f"training) → ~{_fmt_duration(self.est_seconds * n)} total. Refits the model per fold; the "
        f"saved model is unaffected.",
        kind="info",
      )
    else:
      echo(
        f"Held-out validation — {n} folds. Refits the model per fold (the saved model is unaffected).",
        kind="info",
      )

    monitor = EvaluateMonitor(
      list(folds), color=STEP_COLORS.get("holdout", "bright_blue"), est_seconds=self.est_seconds
    )
    records = []
    # Inside monitor.live() the table is the only console surface; per-fold detail goes to the log
    # file at debug level so it never corrupts the live region.
    with monitor.live():
      for fold_name, spec in folds.items():
        monitor.start(fold_name)
        monitor.update_fields(
          fold_name,
          split=spec["strategy"],
          counts=f"{len(spec['train_idx'])}/{len(spec['test_idx'])}",
        )
        try:
          rec = run_one_fold(
            self.path,
            fold_name,
            spec,
            model_ids,
            fetcher,
            batch_size=self.batch_size,
            substep_cb=lambda text, f=fold_name: monitor.set_substep(f, text),
          )
          monitor.update_fields(fold_name, auroc=rec.get("auroc"), aupr=rec.get("aupr"))
          records.append(rec)
          monitor.finish(fold_name, ok=True)
        except Exception as e:
          logger.debug(f"[evaluate] Fold {fold_name} failed ({e}); skipping.")
          monitor.finish(fold_name, ok=False)
    write_validation_outputs(self.path, folds, records)
    step.update()
