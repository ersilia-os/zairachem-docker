import os

from zairachem.base.utils.logging import logger


def process_group(
  output_dir,
  anonymize,
  batch_size=None,
  keep_intermediate_data=False,
  no_report=False,
  describe_workers=None,
):
  """Run describe through finish on a prepared run folder.

  Takes the same arguments as :func:`_run_steps`. SIGTERM is turned into an interrupt, so a
  terminated run unwinds like Ctrl-C. The servers this run started are stopped when it ends.
  """
  # The servers this run started are stopped when the process exits (compose.release_servers).
  # SIGTERM is turned into an interrupt so the run unwinds, and its cleanup (e.g. describe's
  # `docker stats` monitor, that release) runs.
  import signal

  def _interrupt(signum, frame):
    raise KeyboardInterrupt

  previous = signal.signal(signal.SIGTERM, _interrupt)
  try:
    _run_steps(
      output_dir,
      anonymize,
      batch_size=batch_size,
      keep_intermediate_data=keep_intermediate_data,
      no_report=no_report,
      describe_workers=describe_workers,
    )
  finally:
    signal.signal(signal.SIGTERM, previous)


def _run_steps(
  output_dir,
  anonymize,
  batch_size=None,
  keep_intermediate_data=False,
  no_report=False,
  describe_workers=None,
):
  """Run the pipeline steps that follow setup, in order, on a prepared run folder.

  Parameters
  ----------
  output_dir : str
    Run folder produced by setup.
  anonymize : bool
    Anonymize the outputs when finishing.
  batch_size : int, optional
    Rows per chunk for large datasets.
  keep_intermediate_data : bool
    Keep the intermediate descriptor data that the finish step would otherwise delete.
  no_report : bool
    Write the result tables only, skipping the plots and the HTML report.
  describe_workers : int, optional
    Number of descriptor models featurized concurrently.
  """
  # ``output_dir`` is the run folder; it is passed explicitly to every step (no global session
  # lookup). Each pipeline class is imported right before its step runs, so heavy dependencies load
  # only when that step executes (matplotlib for reporting, lazyqsar/xgboost/onnx for estimation) —
  # not all at once at the start. Several of these imports call loguru's logger.remove(), so
  # logger.configure() is re-asserted after each one to keep zairachem's log sinks alive.
  # The shared `tracker` (begun in fit/predict) shows which step is running; start()/complete()
  # are no-ops if the tracker was not begun (e.g. standalone step commands).
  from zairachem.base.utils.progress import final_summary_panel

  _featurize(output_dir, batch_size, describe_workers)
  estimate_seconds = _estimate(output_dir, batch_size)
  _pool_and_validate(output_dir, batch_size, estimate_seconds)
  _report_and_finish(output_dir, anonymize, keep_intermediate_data, no_report)
  final_summary_panel(output_dir)


def _tracked_step(name, output_dir, run):
  """Run one pipeline step under the progress tracker and return what ``run`` returns.

  Re-asserts ``logger.configure()`` (the step's freshly imported module may have removed the log
  sinks), marks the step started, calls ``run`` and marks it complete with its summary.
  """
  from zairachem.base.utils.progress import SUMMARIES, tracker

  logger.configure()
  tracker.start(name)
  result = run()
  tracker.complete(name, SUMMARIES[name](output_dir))
  return result


def _featurize(output_dir, batch_size, describe_workers):
  """Run the describe, projections, treat and screen steps."""
  from zairachem.describe.descriptors.describe import Describer

  _tracked_step(
    "describe",
    output_dir,
    lambda: Describer(path=output_dir, batch_size=batch_size, workers=describe_workers).run(),
  )

  # Projections are an independent 2-D embedding shown as-is in the report (NOT a transformation of
  # the descriptors) — their own step, run while the model containers from Describe are still up.
  from zairachem.base.utils.isaura_report import report_data_provenance
  from zairachem.treat.imputers.manifolds import Manifolds

  def _projections():
    Manifolds(path=output_dir, batch_size=batch_size).run()
    # Rendered here so it reflects BOTH featurizers (describe) and projectors (this step).
    report_data_provenance(output_dir)

  _tracked_step("projections", output_dir, _projections)

  from zairachem.treat.imputers.impute import Imputer

  _tracked_step("treat", output_dir, lambda: Imputer(path=output_dir, batch_size=batch_size).run())

  # Descriptor pre-screening (--max-descriptors). Self-gates on params["max_descriptors"]; a no-op
  # otherwise. Runs after treat (needs the treated matrices) and before estimate so only the selected
  # descriptors are fully trained.
  from zairachem.screen.pipe import ScreenPipeline

  _tracked_step(
    "screen", output_dir, lambda: ScreenPipeline(path=output_dir, batch_size=batch_size).run()
  )


def _estimate(output_dir, batch_size):
  """Run the estimate step and return the training wall-clock seconds."""
  import time

  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  def _train():
    _estimate_t0 = time.time()
    EstimatorPipeline(path=output_dir, batch_size=batch_size).run()
    # Wall-clock of training the full descriptor stack — the per-fold predictor for held-out
    # validation (each fold refits the same stack). ~0 on a resume where estimate was already done;
    # the monitor ignores tiny values.
    return time.time() - _estimate_t0

  estimate_seconds = _tracked_step("estimate", output_dir, _train)

  return estimate_seconds


def _pool_and_validate(output_dir, batch_size, estimate_seconds):
  """Run the pool and holdout steps, then drop the rank-reference matrices."""
  from zairachem.pool.pipe import PoolerPipeline

  _tracked_step(
    "pool", output_dir, lambda: PoolerPipeline(path=output_dir, batch_size=batch_size).run()
  )

  # Held-out validation (--evaluate). Self-gates on params["evaluate"] read from parameters.json, so
  # this is a no-op for a plain fit and for predict. Runs before the report so its section is rendered.
  from zairachem.holdout.pipe import HoldoutValidationPipeline

  _tracked_step(
    "holdout",
    output_dir,
    lambda: HoldoutValidationPipeline(
      path=output_dir, batch_size=batch_size, est_seconds=estimate_seconds
    ).run(),
  )

  # The held-out folds are the last fits; after them the rank-reference matrices are never read.
  from zairachem.base.utils.descriptors import drop_rank_references

  drop_rank_references(output_dir)


def _report_and_finish(output_dir, anonymize, keep_intermediate_data, no_report):
  """Run the report and finish steps."""
  from zairachem.report.report import Reporter

  _tracked_step(
    "report",
    output_dir,
    lambda: Reporter(path=output_dir, plot_name=None, make_plots=not no_report).run(),
  )

  from zairachem.finish.finish import Finisher

  _tracked_step(
    "finish",
    output_dir,
    lambda: Finisher(
      path=output_dir,
      anonymize=anonymize,
      keep_intermediate_data=keep_intermediate_data,
    ).run(),
  )


def _activate_step(model_dir, *step_names):
  """Force the named step(s) to re-run against a model directory; return its absolute path.

  Each step's done-marker is unmarked so the step's own ``is_done`` guard re-executes it — enabling
  fast iteration on one step against an existing model. The resolved path is passed straight to the
  step (there is no global "active run" to resolve against).
  """
  from zairachem.setup.prep import PipelineStep

  output_dir = os.path.abspath(model_dir)
  for name in step_names:
    PipelineStep(name, output_dir).unmark()
  return output_dir
