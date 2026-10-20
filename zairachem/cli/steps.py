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
  """Run describe through finish on a prepared run folder, always stopping model servers.

  Takes the same arguments as :func:`_run_steps`. SIGTERM is turned into an interrupt so the Ersilia
  model servers started by describe are shut down on success, failure, Ctrl-C or SIGTERM.
  """
  # The model servers are started by describe and used until estimate ends; they are always stopped
  # when the run ends, including on failure, Ctrl-C and SIGTERM (turned into an interrupt below).
  import signal

  from zairachem.describe.descriptors.compose import stop_model_servers

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
    stop_model_servers(output_dir)
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
  from zairachem.base.utils.progress import SUMMARIES, final_summary_panel, tracker
  from zairachem.describe.descriptors.describe import Describer

  logger.configure()
  tracker.start("describe")
  Describer(path=output_dir, batch_size=batch_size, workers=describe_workers).run()
  tracker.complete("describe", SUMMARIES["describe"](output_dir))

  # Projections are an independent 2-D embedding shown as-is in the report (NOT a transformation of
  # the descriptors) — their own step, run while the model containers from Describe are still up.
  from zairachem.base.utils.isaura_report import report_data_provenance
  from zairachem.treat.imputers.manifolds import Manifolds

  logger.configure()
  tracker.start("projections")
  Manifolds(path=output_dir, batch_size=batch_size).run()
  # Rendered here so it reflects BOTH featurizers (describe) and projectors (this step).
  report_data_provenance(output_dir)
  tracker.complete("projections", SUMMARIES["projections"](output_dir))

  from zairachem.treat.imputers.impute import Imputer

  logger.configure()
  tracker.start("treat")
  Imputer(path=output_dir, batch_size=batch_size).run()
  tracker.complete("treat", SUMMARIES["treat"](output_dir))

  # Descriptor pre-screening (--max-descriptors). Self-gates on params["max_descriptors"]; a no-op
  # otherwise. Runs after treat (needs the treated matrices) and before estimate so only the selected
  # descriptors are fully trained.
  from zairachem.screen.pipe import ScreenPipeline

  logger.configure()
  tracker.start("screen")
  ScreenPipeline(path=output_dir, batch_size=batch_size).run()
  tracker.complete("screen", SUMMARIES["screen"](output_dir))

  import time

  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  logger.configure()
  tracker.start("estimate")
  _estimate_t0 = time.time()
  EstimatorPipeline(path=output_dir, batch_size=batch_size).run()
  # Wall-clock of training the full descriptor stack — the per-fold predictor for held-out validation
  # (each fold refits the same stack). ~0 on a resume where estimate was already done; the monitor
  # ignores tiny values.
  estimate_seconds = time.time() - _estimate_t0
  tracker.complete("estimate", SUMMARIES["estimate"](output_dir))

  # Nothing after estimate talks to a model server: free the containers before holdout and report.
  from zairachem.describe.descriptors.compose import stop_model_servers

  stop_model_servers(output_dir)

  from zairachem.pool.pipe import PoolerPipeline

  logger.configure()
  tracker.start("pool")
  PoolerPipeline(path=output_dir, batch_size=batch_size).run()
  tracker.complete("pool", SUMMARIES["pool"](output_dir))

  # Held-out validation (--evaluate). Self-gates on params["evaluate"] read from parameters.json, so
  # this is a no-op for a plain fit and for predict. Runs before the report so its section is rendered.
  from zairachem.holdout.pipe import HoldoutValidationPipeline

  logger.configure()
  tracker.start("holdout")
  HoldoutValidationPipeline(
    path=output_dir, batch_size=batch_size, est_seconds=estimate_seconds
  ).run()
  tracker.complete("holdout", SUMMARIES["holdout"](output_dir))

  # The held-out folds are the last fits; after them the rank-reference matrices are never read.
  from zairachem.base.utils.descriptors import drop_rank_references

  drop_rank_references(output_dir)

  from zairachem.report.report import Reporter

  logger.configure()
  tracker.start("report")
  Reporter(path=output_dir, plot_name=None, make_plots=not no_report).run()
  tracker.complete("report", SUMMARIES["report"](output_dir))

  from zairachem.finish.finish import Finisher

  logger.configure()
  tracker.start("finish")
  Finisher(
    path=output_dir,
    anonymize=anonymize,
    keep_intermediate_data=keep_intermediate_data,
  ).run()
  tracker.complete("finish", SUMMARIES["finish"](output_dir))

  final_summary_panel(output_dir)


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
