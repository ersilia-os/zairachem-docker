import os

from zairachem.report.table import OutputTable, PerformanceTable

from zairachem.report.plots import (
  ActivesInactivesPlot,
  ConfusionPlot,
  RocCurvePlot,
  OofScoreProbaPlot,
  OofScoreLogitPlot,
  OofScoreRankPlot,
  OofScoreLiftPlot,
  OofScoreRawPlot,
  OofScoreProbaPointsPlot,
  OofScoreLogitPointsPlot,
  OofScoreRankPointsPlot,
  OofScoreLiftPointsPlot,
  OofScoreRawPointsPlot,
  ProjectionMergedPlot,
  ProjectionClassPlot,
  ProjectionProbaPlot,
  PredictedScoreHistogramPlot,
  ScoreRankCurvePlot,
  DescriptorConsensusPlot,
  AdCoveragePlot,
  RankDistributionPlot,
  RegressionPlotRaw,
  RegressionPlotTransf,
  Transformation,
  IndividualEstimatorsR2Plot,
  CvAurocPlot,
  CvAuprPlot,
  CvMccPlot,
  CvF1Plot,
  CvBalancedAccuracyPlot,
  CvPrecisionPlot,
  CvRecallPlot,
  CvCutoffPlot,
  CvRocPlot,
  CvPrPlot,
  PrCurvePlot,
  EnrichmentCurvePlot,
  EnrichmentFactorCurvePlot,
  ThresholdSweepPlot,
  CalibrationCurvePlot,
  NormalizedConfusionPlot,
  ConfusionPrecisionPlot,
  ConfusionBreakdownPlot,
  DescriptorCorrelationPlot,
  TopKOverlapCurvePlot,
  PropertyMwPlot,
  PropertyLogpPlot,
  PredictedPropertyMwPlot,
  PredictedPropertyLogpPlot,
  ClassDonutPlot,
  ClassWafflePlot,
  StepTimingPlot,
  PhaseTimeDonutPlot,
  ResourceTimelinePlot,
  ProvenanceBarPlot,
  PerModelTimingPlot,
  HeldOutValidationPlot,
  HeldoutRocByStrategyPlot,
  HeldoutPrByStrategyPlot,
  HeldoutCalibrationByStrategyPlot,
  HeldoutEnrichmentFactorByStrategyPlot,
  HeldoutMetricBarsPlot,
  HeldoutConfusionByStrategyPlot,
)
from zairachem.report.fetcher import ResultsFetcher

from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.progress import LiveProgressBar, STEP_COLORS

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
  ("ad-coverage", AdCoveragePlot),
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
  ("regression-trans", RegressionPlotTransf),
  ("regression-raw", RegressionPlotRaw),
  ("transformation", Transformation),
  ("r2-individual", IndividualEstimatorsR2Plot),
  ("cv-auroc", CvAurocPlot),
  ("cv-aupr", CvAuprPlot),
  ("cv-mcc", CvMccPlot),
  ("cv-f1", CvF1Plot),
  ("cv-balacc", CvBalancedAccuracyPlot),
  ("cv-precision", CvPrecisionPlot),
  ("cv-recall", CvRecallPlot),
  ("cv-cutoff", CvCutoffPlot),
  ("cv-roc", CvRocPlot),
  ("cv-pr", CvPrPlot),
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
  ("heldout-roc", HeldoutRocByStrategyPlot),
  ("heldout-pr", HeldoutPrByStrategyPlot),
  ("heldout-calibration", HeldoutCalibrationByStrategyPlot),
  ("heldout-enrichment-factor", HeldoutEnrichmentFactorByStrategyPlot),
  ("heldout-metric-bars", HeldoutMetricBarsPlot),
  ("heldout-confusion", HeldoutConfusionByStrategyPlot),
  ("property-mw", PropertyMwPlot),
  ("property-logp", PropertyLogpPlot),
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
  "cv-aupr",
  "cv-mcc",
  "cv-f1",
  "cv-balacc",
  "cv-precision",
  "cv-recall",
  "cv-cutoff",
  "cv-roc",
  "cv-pr",
  "descriptor-correlation",
  "topk-overlap-curve",
  "heldout-validation",
  "heldout-roc",
  "heldout-pr",
  "heldout-calibration",
  "heldout-enrichment-factor",
  "heldout-metric-bars",
  "heldout-confusion",
}


class Reporter(ZairaBase):
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
    return jobs

  def _clean_figure_dirs(self):
    """Delete previously-rendered figures before a fresh render, so plots dropped from the registry
    (or renamed) don't linger as orphan PNGs and resurface in the report's 'More' catch-all. Every
    current figure is re-rendered right after, so this only removes stale ones."""
    import contextlib
    import glob
    from zairachem.base.vars import REPORT_SUBFOLDER

    report_dir = os.path.join(self.path, REPORT_SUBFOLDER)
    for sub, pattern in (("png", "*.png"), ("pdf", "*.pdf")):
      for f in glob.glob(os.path.join(report_dir, sub, pattern)):
        with contextlib.suppress(OSError):
          os.remove(f)

  def _render_plots(self, jobs):
    """Render all figures, in process and one at a time, under a single compact progress bar.
    (Parallel rendering across processes was tried but matplotlib's global pyplot state plus stylia's
    destructive import-time side effects made it fragile, for a step that is not the run's bottleneck —
    so it stays serial.) A single bad figure is logged and skipped rather than sinking the whole
    report."""
    if not jobs:
      return
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
    # Always: the prediction + performance tables (cheap; they ARE the results).
    self._output_table()
    self._performance_table()
    if not self.make_plots:
      self.logger.info("Skipping plots and HTML report (--no-report); wrote result tables only.")
      return
    self._render_plots(self._plot_jobs())
    self._html_report()

  def run(self):
    step = PipelineStep("report", self.output_dir)
    if not step.is_done():
      self.run_all()
      step.update()
    else:
      self.logger.info("Report already done — skipping.")

    self.logger.info("Reporting successfully completed!")
