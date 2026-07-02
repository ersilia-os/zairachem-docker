import os

from zairachem.report.table import OutputTable, PerformanceTable

from zairachem.report.plots import (
  ActivesInactivesPlot,
  ConfusionPlot,
  IndividualEstimatorsClassificationScorePlot,
  RocCurvePlot,
  ScoreViolinPlot,
  ScoreStripPlot,
  ProjectionMergedPlot,
  ProjectionClassPlot,
  RegressionPlotRaw,
  HistogramPlotRaw,
  RegressionPlotTransf,
  HistogramPlotTransf,
  Transformation,
  IndividualEstimatorsAurocPlot,
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
  EnrichmentFactorPlot,
  ThresholdSweepPlot,
  NormalizedConfusionPlot,
  PropertyMwPlot,
  PropertyLogpPlot,
  ClassDonutPlot,
  ClassWafflePlot,
  StepTimingPlot,
  PhaseTimeDonutPlot,
  ResourceTimelinePlot,
  ProvenanceBarPlot,
  PerModelTimingPlot,
  HeldOutValidationPlot,
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
  ("score-violin", ScoreViolinPlot),
  ("score-strip", ScoreStripPlot),
  ("regression-trans", RegressionPlotTransf),
  ("histogram-trans", HistogramPlotTransf),
  ("regression-raw", RegressionPlotRaw),
  ("histogram-raw", HistogramPlotRaw),
  ("transformation", Transformation),
  ("roc-individual", IndividualEstimatorsAurocPlot),
  ("raw-classification-scores", IndividualEstimatorsClassificationScorePlot),
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
  ("enrichment-factor", EnrichmentFactorPlot),
  ("threshold-sweep", ThresholdSweepPlot),
  ("confusion-normalized", NormalizedConfusionPlot),
  ("heldout-validation", HeldOutValidationPlot),
  ("property-mw", PropertyMwPlot),
  ("property-logp", PropertyLogpPlot),
  ("step-timing", StepTimingPlot),
  ("phase-time", PhaseTimeDonutPlot),
  ("resource-timeline", ResourceTimelinePlot),
  ("compute-provenance", ProvenanceBarPlot),
  ("model-timing", PerModelTimingPlot),
]


class Reporter(ZairaBase):
  def __init__(self, path, plot_name=None, make_plots=True):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
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
    for name, cls in _PLOT_SPECS:
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
    return jobs

  def _render_plots(self, jobs):
    """Render all figures, in process and one at a time, under a single compact progress bar.
    (Parallel rendering across processes was tried but matplotlib's global pyplot state plus stylia's
    destructive import-time side effects made it fragile, for a step that is not the run's bottleneck —
    so it stays serial.) A single bad figure is logged and skipped rather than sinking the whole
    report."""
    if not jobs:
      return
    bar = LiveProgressBar(
      "Rendering plots",
      total=len(jobs),
      color=STEP_COLORS.get("report", "cyan"),
      discrete=True,
      show_bar=False,
    )
    n_na, n_fail = 0, 0
    with bar.live():
      for label, cls, kwargs in jobs:
        bar.set_note(label)
        try:
          plot = cls(ax=None, path=self.path, **kwargs)
          if getattr(plot, "is_available", True):
            plot.save()
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
