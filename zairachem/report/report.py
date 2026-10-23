import os

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.progress import STEP_COLORS, LiveProgressBar
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report.plots import (
  ActivesInactivesPlot,
  CalibrationCurvePlot,
  ClassDonutPlot,
  ClassWafflePlot,
  ConfusionBreakdownPlot,
  ConfusionPlot,
  ConfusionPrecisionPlot,
  CvAuprPlot,
  CvAurocPlot,
  CvBalancedAccuracyPlot,
  CvCutoffPlot,
  CvErrorCurvePlot,
  CvF1Plot,
  CvMccPlot,
  CvPrecisionPlot,
  CvPrPlot,
  CvRecallPlot,
  CvRocPlot,
  CvSpearmanPlot,
  DescriptorConsensusPlot,
  DescriptorCorrelationPlot,
  EnrichmentCurvePlot,
  EnrichmentFactorCurvePlot,
  ErrorVsSimilarityPlot,
  HeldoutCalibrationByStrategyPlot,
  HeldoutConfusionByStrategyPlot,
  HeldoutEnrichmentFactorByStrategyPlot,
  HeldoutMetricBarsPlot,
  HeldoutPrByStrategyPlot,
  HeldoutR2ValidationPlot,
  HeldoutRocByStrategyPlot,
  HeldOutValidationPlot,
  IndividualEstimatorsR2Plot,
  IntervalCoveragePlot,
  NormalizedConfusionPlot,
  OofScoreLiftPlot,
  OofScoreLiftPointsPlot,
  OofScoreLogitPlot,
  OofScoreLogitPointsPlot,
  OofScoreProbaPlot,
  OofScoreProbaPointsPlot,
  OofScoreRankPlot,
  OofScoreRankPointsPlot,
  OofScoreRawPlot,
  OofScoreRawPointsPlot,
  PerModelTimingPlot,
  PhaseTimeDonutPlot,
  PotencyEnrichmentPlot,
  PrCurvePlot,
  PredictedPropertyLogpPlot,
  PredictedPropertyMwPlot,
  PredictedScoreHistogramPlot,
  PredictedValueDistributionPlot,
  ProjectionClassPlot,
  ProjectionMergedPlot,
  ProjectionProbaPlot,
  ProjectionValuePlot,
  PropertyLogpPlot,
  PropertyMwPlot,
  PropertyValueLogpPlot,
  PropertyValueMwPlot,
  ProvenanceBarPlot,
  RankDistributionPlot,
  RegressionPlotRaw,
  ResidualsPlot,
  ResourceTimelinePlot,
  RocCurvePlot,
  ScoreRankCurvePlot,
  StepTimingPlot,
  ThresholdSweepPlot,
  TopKOverlapCurvePlot,
  TrainSimilarityPlot,
  ValueConsensusPlot,
)
from zairachem.report.table import OutputTable, PerformanceTable

# Fixed (skip-name, plot-class) figures, in render order. Projection figures are appended dynamically
# (one per discovered projection) because their count depends on the run's manifest.
_PLOT_SPECS = [
  ("actives-inactives", ActivesInactivesPlot),
  ("class-donut", ClassDonutPlot),
  ("class-waffle", ClassWafflePlot),
  ("confusion-matrix", ConfusionPlot),
  ("roc-curve", RocCurvePlot),
  # Predict-only (truth-free): render only when there is no ground-truth column.
  ("predicted-score-hist", PredictedScoreHistogramPlot),
  ("predicted-rank-curve", ScoreRankCurvePlot),
  ("descriptor-consensus", DescriptorConsensusPlot),
  ("value-consensus", ValueConsensusPlot),
  ("tanimoto-similarity-to-train", TrainSimilarityPlot),
  ("rank-distribution", RankDistributionPlot),
  ("oof-score-proba", OofScoreProbaPlot),
  ("oof-score-logit", OofScoreLogitPlot),
  ("oof-score-rank", OofScoreRankPlot),
  ("oof-score-lift", OofScoreLiftPlot),
  ("oof-score-raw", OofScoreRawPlot),
  ("oof-score-proba-pts", OofScoreProbaPointsPlot),
  ("oof-score-logit-pts", OofScoreLogitPointsPlot),
  ("oof-score-rank-pts", OofScoreRankPointsPlot),
  ("oof-score-lift-pts", OofScoreLiftPointsPlot),
  ("oof-score-raw-pts", OofScoreRawPointsPlot),
  ("regression-residuals", ResidualsPlot),
  ("regression-raw", RegressionPlotRaw),
  ("regression-distribution", PredictedValueDistributionPlot),
  ("regression-error-similarity", ErrorVsSimilarityPlot),
  ("interval-coverage", IntervalCoveragePlot),
  ("potency-enrichment", PotencyEnrichmentPlot),
  ("r2-individual", IndividualEstimatorsR2Plot),
  ("cv-auroc", CvAurocPlot),
  ("cv-spearman", CvSpearmanPlot),
  ("cv-aupr", CvAuprPlot),
  ("cv-mcc", CvMccPlot),
  ("cv-f1", CvF1Plot),
  ("cv-balacc", CvBalancedAccuracyPlot),
  ("cv-precision", CvPrecisionPlot),
  ("cv-recall", CvRecallPlot),
  ("cv-cutoff", CvCutoffPlot),
  ("cv-roc", CvRocPlot),
  ("cv-pr", CvPrPlot),
  ("cv-error-curve", CvErrorCurvePlot),
  ("pr-curve", PrCurvePlot),
  ("enrichment-curve", EnrichmentCurvePlot),
  ("enrichment-factor-curve", EnrichmentFactorCurvePlot),
  ("threshold-sweep", ThresholdSweepPlot),
  ("calibration-curve", CalibrationCurvePlot),
  ("confusion-normalized", NormalizedConfusionPlot),
  ("confusion-precision", ConfusionPrecisionPlot),
  ("confusion-breakdown", ConfusionBreakdownPlot),
  ("descriptor-correlation", DescriptorCorrelationPlot),
  ("topk-overlap-curve", TopKOverlapCurvePlot),
  ("heldout-validation", HeldOutValidationPlot),
  ("heldout-r2", HeldoutR2ValidationPlot),
  ("heldout-roc", HeldoutRocByStrategyPlot),
  ("heldout-pr", HeldoutPrByStrategyPlot),
  ("heldout-calibration", HeldoutCalibrationByStrategyPlot),
  ("heldout-enrichment-factor", HeldoutEnrichmentFactorByStrategyPlot),
  ("heldout-metric-bars", HeldoutMetricBarsPlot),
  ("heldout-confusion", HeldoutConfusionByStrategyPlot),
  ("property-mw", PropertyMwPlot),
  ("property-logp", PropertyLogpPlot),
  ("property-mw-value", PropertyValueMwPlot),
  ("property-logp-value", PropertyValueLogpPlot),
  ("predicted-property-mw", PredictedPropertyMwPlot),
  ("predicted-property-logp", PredictedPropertyLogpPlot),
  ("step-timing", StepTimingPlot),
  ("phase-time", PhaseTimeDonutPlot),
  ("resource-timeline", ResourceTimelinePlot),
  ("compute-provenance", ProvenanceBarPlot),
  ("model-timing", PerModelTimingPlot),
]

# Train-only diagnostics: these describe the TRAINED model (their cross-validation data is read from
# the model directory, not the current run), so they are meaningless — and misleading — in a predict
# report, which is about the NEW molecules. Excluded when running in predict mode.
_FIT_ONLY = {
  "cv-auroc",
  "cv-spearman",
  "cv-aupr",
  "cv-mcc",
  "cv-f1",
  "cv-balacc",
  "cv-precision",
  "cv-recall",
  "cv-cutoff",
  "cv-roc",
  "cv-pr",
  "cv-error-curve",
  "descriptor-correlation",
  "topk-overlap-curve",
  "heldout-validation",
  "heldout-r2",
  "heldout-roc",
  "heldout-pr",
  "heldout-calibration",
  "heldout-enrichment-factor",
  "heldout-metric-bars",
  "heldout-confusion",
}


class Reporter(ZairaBase):
  """Write the result tables and the report (plots and HTML page) of a run.

  Parameters
  ----------
  path : str
    Run folder.
  plot_name : str, optional
    Render only this plot.
  make_plots : bool
    When False, write the tables only (``--no-report``).
  """

  def __init__(self, path, plot_name=None, make_plots=True):
    ZairaBase.__init__(self)
    self.path = path
    self.output_dir = os.path.abspath(self.path)
    assert os.path.exists(self.output_dir)
    self.plot_name = plot_name
    # When False (--no-report) skip the figures + HTML page; still write the prediction / performance
    # tables so results/ is fully populated.
    self.make_plots = make_plots

  def __skip(self, this_name):
    if self.plot_name is None:
      return False
    if self.plot_name == this_name:
      return False
    else:
      return True

  def _plot_jobs(self):
    """Build the list of ``(label, PlotClass, kwargs)`` jobs to render, honouring ``plot_name`` (the
    single-plot filter). ``label`` is the human-readable figure name shown in the progress table.
    Projection figures are expanded from the run's manifest."""
    jobs = []
    predict = self.is_predict()
    for name, cls in _PLOT_SPECS:
      if predict and name in _FIT_ONLY:
        continue  # train-only diagnostic — not shown in a predict report
      if not self.__skip(name):
        jobs.append((name.replace("-", " "), cls, {}))
    # Per projection (always at least MW-vs-LogP): a merged density map + one map per class.
    for proj in ResultsFetcher(path=self.path).get_projections():
      if not self.__skip(f"projection-merged-{proj['name']}"):
        jobs.append((
          f"projection {proj['name']} merged",
          ProjectionMergedPlot,
          {"projection": proj},
        ))
      for cls, noun in ((1, "active"), (0, "inactive")):
        if not self.__skip(f"projection-{proj['name']}-{noun}"):
          jobs.append((
            f"projection {proj['name']} {noun}",
            ProjectionClassPlot,
            {"projection": proj, "cls": cls},
          ))
      # Predict-only: same projection coloured by predicted probability (renders only without truth).
      if not self.__skip(f"projection-{proj['name']}-proba"):
        jobs.append((
          f"projection {proj['name']} score",
          ProjectionProbaPlot,
          {"projection": proj},
        ))
      # Regression: the same map coloured by the target value instead of by class.
      if not self.__skip(f"projection-{proj['name']}-value"):
        jobs.append((
          f"projection {proj['name']} value",
          ProjectionValuePlot,
          {"projection": proj},
        ))
    return jobs

  def _clean_figure_dirs(self):
    """Delete previously-rendered figures before a fresh render, so plots dropped from the registry
    (or renamed) don't linger as orphan PNGs and resurface in the report's 'More' catch-all.

    Only safe for a full render, where every current figure is re-rendered immediately after. A
    ``--plot-name`` render would otherwise delete the whole report to rebuild one figure of it.
    """
    import contextlib
    import glob

    from zairachem.base.vars import REPORT_SUBFOLDER

    report_dir = os.path.join(self.path, REPORT_SUBFOLDER)
    for sub, pattern in (("png", "*.png"), ("pdf", "*.pdf")):
      for f in glob.glob(os.path.join(report_dir, sub, pattern)):
        with contextlib.suppress(OSError):
          os.remove(f)

  def _render_plots(self, jobs):
    """Render all figures under a single compact progress bar, in parallel processes when worth it.

    Each worker process has its own pyplot state (spawned, Agg backend), so figures cannot interfere,
    and stylia's import no longer wipes matplotlib's font cache (see ``report/_stylia.py``); if the
    pool cannot run, figures are rendered here one at a time. A single bad figure is logged and
    skipped rather than sinking the whole report."""
    if not jobs:
      return
    if self.plot_name is None:
      self._clean_figure_dirs()
    workers = _render_workers(len(jobs))
    if workers > 1:
      try:
        self._render_parallel(jobs, workers)
        return
      except Exception as e:  # e.g. a job that cannot be pickled: render serially instead
        self.logger.debug(f"[report] parallel rendering unavailable ({e}); rendering serially")
        self._clean_figure_dirs()
    bar = LiveProgressBar(
      "Rendering plots",
      total=len(jobs),
      color=STEP_COLORS.get("report", "cyan"),
      discrete=True,
      show_bar=False,
    )
    n_na, n_fail = 0, 0
    # Record each rendered figure's declared grid footprint (rows, cols) — the source of truth the
    # HTML uses for the card's size badge (can't be reliably recovered from the tight-cropped PNG).
    cells_map = {}
    with bar.live():
      for label, cls, kwargs in jobs:
        bar.set_note(label)
        try:
          plot = cls(ax=None, path=self.path, **kwargs)
          if getattr(plot, "is_available", True):
            plot.save()
            cells_map[plot.name] = list(getattr(plot, "cells", (2, 2)))
          else:
            n_na += 1  # figure not applicable to this run (e.g. regression plots for a clf model)
        except Exception as e:
          n_fail += 1
          self.logger.warning(f"[report] plot {cls.__name__} failed: {e}")
        bar.advance()
      # Final note: anything that didn't produce a figure, so the persisted line tells the full story.
      extras = []
      if n_na:
        extras.append(f"{n_na} n/a")
      if n_fail:
        extras.append(f"{n_fail} failed")
      bar.set_note(" · ".join(extras))
    self._write_figure_cells(cells_map)

  def _render_parallel(self, jobs, workers):
    """Render ``jobs`` on ``workers`` spawned processes (see :meth:`_render_plots`)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    bar = LiveProgressBar(
      "Rendering plots",
      total=len(jobs),
      color=STEP_COLORS.get("report", "cyan"),
      discrete=True,
      show_bar=False,
    )
    n_na, n_fail, cells_map = 0, 0, {}
    ctx = multiprocessing.get_context("spawn")
    with bar.live(), ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
      futures = {
        ex.submit(_render_one, cls, self.path, kwargs): label for label, cls, kwargs in jobs
      }
      for fut in as_completed(futures):
        status, name, cells = fut.result()
        bar.set_note(futures[fut])
        if status == "ok":
          cells_map[name] = cells
        elif status == "na":
          n_na += 1
        else:
          n_fail += 1
          self.logger.warning(f"[report] plot {name}")
        bar.advance()
      extras = [f"{n} {what}" for n, what in ((n_na, "n/a"), (n_fail, "failed")) if n]
      bar.set_note(" · ".join(extras))
    self._write_figure_cells(cells_map)

  def _write_figure_cells(self, cells_map):
    """Persist ``{stem: [rows, cols]}`` for the HTML size badges (overwritten each render)."""
    import json

    from zairachem.base.vars import REPORT_SUBFOLDER

    report_dir = os.path.join(self.path, REPORT_SUBFOLDER)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "figure_cells.json"), "w") as f:
      json.dump(cells_map, f, indent=2)

  def _output_table(self):
    OutputTable(path=self.path).run()

  def _performance_table(self):
    PerformanceTable(path=self.path).run()

  def _html_report(self):
    from zairachem.report.html import write_html_report

    write_html_report(self.path)

  def run_all(self):
    """Write the tables and, unless disabled, the plots and the HTML report."""
    # Always: the prediction + performance tables (cheap; they ARE the results).
    self._output_table()
    self._performance_table()
    if not self.make_plots:
      self.logger.info("Skipping plots and HTML report (--no-report); wrote result tables only.")
      return
    self._render_plots(self._plot_jobs())
    self._html_report()

  def run(self):
    """Build the report, skipping it if already done."""
    step = PipelineStep("report", self.output_dir)
    if not step.is_done():
      self.run_all()
      step.update()
    else:
      self.logger.info("Report already done — skipping.")

    self.logger.info("Reporting successfully completed!")


def _render_workers(n_jobs):
  """Processes for rendering ``n_jobs`` figures: up to 4, about one per 8 figures, 1 = serial."""
  from zairachem.base.utils.concurrency import cpu_count

  return max(1, min(4, cpu_count() - 1, n_jobs // 8))


def _render_one(cls, path, kwargs):
  """Render one figure in a worker process: ``(status, name, cells)``, status ok / na / fail."""
  import matplotlib

  matplotlib.use("Agg")
  try:
    plot = cls(ax=None, path=path, **kwargs)
    if not getattr(plot, "is_available", True):
      plot.discard()  # it built a figure before it could decline; don't leave it open
      return "na", None, None
    plot.save()
    return "ok", plot.name, list(getattr(plot, "cells", (2, 2)))
  except Exception as e:
    return "fail", f"{cls.__name__} failed: {e}", None
