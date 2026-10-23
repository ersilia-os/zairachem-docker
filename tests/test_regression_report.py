"""The regression report: its panels, its sections, and the words it puts on them.

A regression run renders a different set of figures from a classification one, under the same
section headings and from the same registry. These tests pin that the two agree -- every panel
registered, titled and owned by a section -- and that each panel renders, or declines, without
raising against a run folder built here rather than by the pipeline.
"""

import os

import numpy as np
import pytest


def _plot_specs():
  from zairachem.report.report import _PLOT_SPECS

  return dict(_PLOT_SPECS)


def test_inner_cv_reports_are_read_with_the_right_skill():
  """One cross-validation report, two vocabularies: AUROC or Spearman's rho, never a mix."""
  from zairachem.base.utils.summaries import _cv_headline
  from zairachem.report.cv import CHANCE, skill_keys, sort_best_first

  clf = [{"descriptor": "a", "oof_auc": 0.80}, {"descriptor": "b", "oof_auc": 0.90}]
  reg = [{"descriptor": "a", "oof_rho": 0.60}, {"descriptor": "b", "oof_rho": 0.80}]
  assert skill_keys(clf) == ("oof_auc", "train_auc", "AUROC")
  assert skill_keys(reg)[0] == "oof_rho"
  assert CHANCE["oof_auc"] == 0.5 and CHANCE["oof_rho"] == 0.0

  # Best first, on each task's own skill, and a missing skill never takes the top slot.
  assert [s["descriptor"] for s in sort_best_first(list(clf))] == ["b", "a"]
  assert [s["descriptor"] for s in sort_best_first(list(reg))] == ["b", "a"]
  assert [s["descriptor"] for s in sort_best_first(reg + [{"descriptor": "c"}])] == ["b", "a", "c"]

  # The terminal headline reads the same table, so it cannot name one metric and print another.
  label, mean, best, best_skill = _cv_headline(clf)
  assert (label, best, best_skill) == ("AUROC", "b", 0.90) and mean == pytest.approx(0.85)
  label, mean, best, _ = _cv_headline(reg)
  assert label.startswith("Spearman") and best == "b" and mean == pytest.approx(0.70)
  assert _cv_headline([{"descriptor": "a"}]) is None


def test_the_hitlist_ranks_a_regression_run_by_value(tmp_path):
  import csv

  from zairachem.report.html.tables import _hitlist_table_html

  rows = [
    {"input_smiles": "CCO", "value": "5.0", "lower": "4.0", "upper": "6.0", "similarity": "0.3"},
    {"input_smiles": "CCN", "value": "8.5", "lower": "7.5", "upper": "9.5", "similarity": "0.9"},
  ]
  with open(tmp_path / "output_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  out = _hitlist_table_html(str(tmp_path))
  assert "by predicted value" in out and "Interval" in out
  assert "Call" not in out  # a regression run has no call to make
  assert out.index("CCN") < out.index("CCO")  # ranked by value, best first


def _write_run(root, task, rows):
  """A minimal run folder: metadata/parameters.json plus inputs/data.csv."""
  import csv

  (root / "metadata").mkdir(parents=True, exist_ok=True)
  (root / "inputs").mkdir(parents=True, exist_ok=True)
  (root / "metadata" / "parameters.json").write_text(f'{{"task": "{task}"}}')
  with open(root / "inputs" / "data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  return str(root)


# --- Against a run folder built here, rather than by the pipeline -------------------------------


def _synthetic_run(root, n=60, descriptors=("eosA", "eosB"), truth=True, predict=False):
  """A run folder complete enough for the report's fetchers, with no pipeline involved."""
  import json

  import pandas as pd

  rng = np.random.default_rng(0)
  y = rng.normal(6.5, 1.2, n)
  smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "C1CCCCC1"] * (n // 6 + 1)
  smiles = smiles[:n]
  cid = [f"CID{i:03d}" for i in range(n)]

  for sub in ("inputs", "metadata", "model/pool", "report"):
    (root / sub).mkdir(parents=True, exist_ok=True)
  (root / "metadata" / "parameters.json").write_text(json.dumps({"task": "regression"}))
  (root / "session.json").write_text(
    json.dumps({
      "output_dir": str(root),
      "model_dir": str(root),
      "mode": "predict" if predict else "fit",
    })
  )

  data = {"compound_id": cid, "smiles": smiles}
  if truth:
    data["val"] = y
  pd.DataFrame(data).to_csv(root / "inputs" / "data.csv", index=False)

  pooled = y + rng.normal(0, 0.4, n)
  pd.DataFrame({
    "compound_id": cid,
    "smiles": smiles,
    "reg": pooled,
    "reg_lower": pooled - 1.0,
    "reg_upper": pooled + 1.0,
    "train_similarity": rng.random(n),
    "descriptor_std": rng.random(n) * 0.3,
  }).to_csv(root / "model" / "pool" / "results_unmapped.csv", index=False)

  (root / "descriptors").mkdir(exist_ok=True)
  (root / "descriptors" / "done_eos.json").write_text(json.dumps(list(descriptors)))
  for i, eos in enumerate(descriptors):
    d = root / "model" / "estimators" / "lq_estimators" / eos
    d.mkdir(parents=True, exist_ok=True)
    pred = y + rng.normal(0, 0.5 + 0.2 * i, n)
    pd.DataFrame({
      "compound_id": cid,
      "smiles": smiles,
      "reg": pred,
      "reg_rank": rng.random(n),
      "reg_ad": rng.random(n),
    }).to_csv(d / "results_unmapped.csv", index=False)
    (d / "cv_report.json").write_text(
      json.dumps({
        "oof_rho": 0.8 - 0.1 * i,
        "train_rho": 0.95,
        "overfit_gap": 0.15 + 0.1 * i,
        "portfolio": ["lr", "xgb"],
      })
    )
    pd.DataFrame({"oof_value": pred, "y": y}).to_csv(d / "oof.csv", index=False)

  # The raw input and its dedupe mapping, so the output table can map results back to input rows.
  pd.DataFrame({"smiles": smiles, "activity": y}).to_csv(
    root / "inputs" / "raw_input.csv", index=False
  )
  (root / "inputs" / "input_schema.json").write_text(
    json.dumps({"input_file": "none", "smiles_column": "smiles", "values_column": "activity"})
  )
  pd.DataFrame({"orig_idx": range(n), "uniq_idx": range(n), "compound_id": cid}).to_csv(
    root / "inputs" / "mapping.csv", index=False
  )

  proj = [{"name": "umap", "title": "UMAP projection", "x": "umap_x", "y": "umap_y"}]
  (root / "inputs" / "projections.json").write_text(json.dumps(proj))
  pd.DataFrame({"umap_x": rng.normal(size=n), "umap_y": rng.normal(size=n)}).to_csv(
    root / "inputs" / "projections.csv", index=False
  )
  return str(root)


REGRESSION_PANELS = [
  "IndividualEstimatorsR2Plot",
  "RegressionPlotRaw",
  "ResidualsPlot",
  "PotencyEnrichmentPlot",
  "IntervalCoveragePlot",
  "PredictedValueDistributionPlot",
  "TargetValueDistributionPlot",
  "ErrorVsSimilarityPlot",
  "ValueConsensusPlot",
  "PropertyValueMwPlot",
  "PropertyValueLogpPlot",
  "CvSpearmanPlot",
  "CvErrorCurvePlot",
]


@pytest.mark.parametrize("truth", [True, False])
def test_every_regression_panel_renders_or_declines(tmp_path, truth):
  """No panel may raise: the renderer logs and drops a failing plot, so a crash is a silent hole."""
  from zairachem.report import plots as plots_module

  path = _synthetic_run(
    tmp_path / ("labelled" if truth else "blind"), truth=truth, predict=not truth
  )
  import matplotlib.pyplot as plt

  plt.close("all")
  drawn = []
  for name in REGRESSION_PANELS:
    plot = getattr(plots_module, name)(ax=None, path=path)  # raises if the panel is broken
    if getattr(plot, "is_available", False):
      drawn.append(name)
    plot.save()  # what the renderer calls: it writes, or not, and closes the figure either way
  assert not plt.get_fignums(), "panels left their figures open"
  if truth:
    # With observed values in hand, the accuracy panels are the point of the report.
    assert {"RegressionPlotRaw", "ResidualsPlot", "PotencyEnrichmentPlot"} <= set(drawn)
    # The Dataset section's answer to the class-balance panel: it needs the observed values.
    assert "TargetValueDistributionPlot" in drawn
    assert "ValueConsensusPlot" not in drawn  # truth present: the error panels say more
  else:
    assert "ValueConsensusPlot" in drawn
    assert "RegressionPlotRaw" not in drawn  # nothing to compare the predictions against
    assert "TargetValueDistributionPlot" not in drawn


def test_the_value_projection_renders_from_a_synthetic_run(tmp_path):
  from zairachem.report.fetcher import ResultsFetcher
  from zairachem.report.plots import ProjectionValuePlot

  path = _synthetic_run(tmp_path / "run")
  projection = ResultsFetcher(path=path).get_projections()[0]
  plot = ProjectionValuePlot(ax=None, path=path, projection=projection)
  assert plot.is_available and plot.name == "projection-umap-value"


def test_a_degenerate_run_still_produces_readable_panels(tmp_path):
  """A constant target has no correlation and no spread; the panels must say so, not print "nan"."""
  import json

  import pandas as pd

  from zairachem.report import plots as plots_module

  path = _synthetic_run(tmp_path / "flat")
  n = len(pd.read_csv(os.path.join(path, "inputs", "data.csv")))
  flat = pd.DataFrame({
    "compound_id": [f"CID{i:03d}" for i in range(n)],
    "smiles": ["CCO"] * n,
    "val": [5.0] * n,
  })
  flat.to_csv(os.path.join(path, "inputs", "data.csv"), index=False)
  json.loads(open(os.path.join(path, "metadata", "parameters.json")).read())  # still regression

  for name in REGRESSION_PANELS:
    plot = getattr(plots_module, name)(ax=None, path=path)  # must not raise
    title = plot.ax.get_title() if getattr(plot, "is_available", False) else ""
    assert "nan" not in title.lower(), f"{name} shows a nan in its title: {title!r}"


def test_the_report_survives_a_model_that_lost_its_intervals(tmp_path):
  """Descriptor drift at predict drops the stacking weights and the interval model (``restrict``)."""
  import pandas as pd

  from zairachem.report import plots as plots_module
  from zairachem.report.html.tables import _hitlist_table_html

  path = _synthetic_run(tmp_path / "drifted")
  pooled = os.path.join(path, "model", "pool", "results_unmapped.csv")
  df = pd.read_csv(pooled).drop(columns=["reg_lower", "reg_upper"])
  df.to_csv(pooled, index=False)

  for name in REGRESSION_PANELS:
    plot = getattr(plots_module, name)(ax=None, path=path)  # must not raise
    if name == "IntervalCoveragePlot":
      assert not plot.is_available  # nothing to check coverage of
  # The output table and hitlist simply omit the interval rather than failing.
  report_dir = tmp_path / "drifted" / "report"
  report_dir.mkdir(exist_ok=True)
  pd.DataFrame({"input_smiles": ["CCO", "CCN"], "value": [5.0, 8.5]}).to_csv(
    report_dir / "output_table.csv", index=False
  )
  out = _hitlist_table_html(str(report_dir))
  assert "by predicted value" in out and "Interval" not in out


def test_one_missing_value_does_not_change_what_the_error_panel_plots(tmp_path):
  """Pre-filtering shortened the array, and the length check then fell through to interval width."""
  import pandas as pd

  from zairachem.report.plots import ErrorVsSimilarityPlot

  path = _synthetic_run(tmp_path / "gappy", predict=True)
  data_path = os.path.join(path, "inputs", "data.csv")
  data = pd.read_csv(data_path)
  data.loc[0, "val"] = None  # one compound was never measured
  data.to_csv(data_path, index=False)

  plot = ErrorVsSimilarityPlot(ax=None, path=path)
  assert plot.is_available
  assert "Absolute error" in plot.ax.get_title(), plot.ax.get_title()


def test_the_plot_registry_and_the_html_catalog_agree():
  """Covers both tasks: adding a plot without a section leaves it in the "More" catch-all.

  Five tables between them decide where a figure lands and what it is called, and nothing ties
  them to the registry. A regression panel reached the catch-all this way during development.
  """
  import collections

  from zairachem.report.html.catalog import (
    _CATEGORIES,
    _COMPUTE_GROUPS,
    _DATASET_GROUPS,
    _DIAG_GROUPS,
    _GROUPS,
    _TITLES,
  )
  from zairachem.report.report import _FIT_ONLY, _PLOT_SPECS

  names = [name for name, _cls in _PLOT_SPECS]
  assert not [n for n, c in collections.Counter(names).items() if c > 1], "registered twice"

  catalogued = {m for _a, _h, _d, members in _CATEGORIES for m in members}
  for table in (_GROUPS, _DIAG_GROUPS, _DATASET_GROUPS, _COMPUTE_GROUPS):
    catalogued |= {m for group in table for m in group["members"]}

  # The projection figures are registered per discovered projection, not in _PLOT_SPECS.
  fixed = [n for n in names if not n.startswith("projection-")]
  assert not [n for n in fixed if n not in catalogued], "registered but in no section"
  assert not [n for n in fixed if n not in _TITLES], "registered but untitled"
  assert not [m for m in catalogued if m not in names and not m.startswith("projection-")], (
    "a section names a figure nothing registers"
  )
  assert not [n for n in _FIT_ONLY if n not in names], "fit-only name is not a registered plot"


def test_the_enrichment_curve_reports_the_mean_of_the_top_slice(tmp_path):
  """Checked against a case whose answer is known by construction."""
  import pandas as pd

  from zairachem.report.plots import PotencyEnrichmentPlot

  path = _synthetic_run(tmp_path / "ranked", n=100)
  # A perfect ranker on a known target: the top 10% are exactly the ten highest values.
  values = np.linspace(1.0, 100.0, 100)
  cid = [f"CID{i:03d}" for i in range(100)]
  smiles = pd.read_csv(os.path.join(path, "inputs", "data.csv"))["smiles"].tolist()
  pd.DataFrame({"compound_id": cid, "smiles": smiles, "val": values}).to_csv(
    os.path.join(path, "inputs", "data.csv"), index=False
  )
  pooled = os.path.join(path, "model", "pool", "results_unmapped.csv")
  df = pd.read_csv(pooled)
  df["reg"] = values  # prediction order == truth order
  df.to_csv(pooled, index=False)

  plot = PotencyEnrichmentPlot(ax=None, path=path)
  assert plot.is_available
  # Top 10 of 100 are 91..100, mean 95.5; the whole set averages 50.5.
  assert "95.50" in plot.ax.get_title(), plot.ax.get_title()
  line = plot.ax.get_lines()[0]
  assert line.get_ydata()[-1] == pytest.approx(values.mean())  # the full set, at 100%
  assert np.all(np.diff(line.get_ydata()) <= 1e-9)  # a perfect ranker only ever gets worse


def test_the_coverage_panel_counts_the_compounds_inside_their_interval(tmp_path):
  import pandas as pd

  from zairachem.report.plots import IntervalCoveragePlot

  path = _synthetic_run(tmp_path / "coverage", n=100)
  truth = pd.read_csv(os.path.join(path, "inputs", "data.csv"))["val"].to_numpy(float)
  pooled_path = os.path.join(path, "model", "pool", "results_unmapped.csv")
  df = pd.read_csv(pooled_path)
  # Centre every interval on the truth, then push exactly ten compounds outside their own.
  df["reg_lower"], df["reg_upper"] = truth - 1.0, truth + 1.0
  df.loc[:9, "reg_lower"] = truth[:10] + 0.5
  df.to_csv(pooled_path, index=False)

  plot = IntervalCoveragePlot(ax=None, path=path)
  assert plot.is_available
  labels = [t.get_text() for t in plot.ax.get_xticklabels()]
  heights = [bar.get_height() for bar in plot.ax.patches]
  # Moving ten lower bounds also changed their width, so those ten are their own bin: it covers
  # none of them, the untouched bin covers all of its own, and the pooled bar is 90 of 100.
  assert labels[-1] == "All" and heights[-1] == pytest.approx(0.90)
  assert sorted(heights[:-1]) == [pytest.approx(0.0), pytest.approx(1.0)]


def test_rendering_one_plot_does_not_delete_the_others(tmp_path, monkeypatch):
  """``--plot-name`` rebuilds one figure; it used to clear the figure directory first."""
  from zairachem.report.report import Reporter

  path = _synthetic_run(tmp_path / "run")
  png = tmp_path / "run" / "report" / "png"
  png.mkdir(parents=True, exist_ok=True)
  for stem in ("regression-raw", "cv-spearman", "potency-enrichment"):
    (png / f"{stem}.png").write_bytes(b"not a real png")

  reporter = Reporter(path=path, plot_name="potency-enrichment")
  reporter._render_plots(reporter._plot_jobs())
  survivors = {p.stem for p in png.glob("*.png")}
  assert {"regression-raw", "cv-spearman"} <= survivors, survivors


def test_the_tables_are_written_even_with_no_report_folder(tmp_path):
  """``--skip-report`` renders no figures, and nothing else creates the folder they would live in."""
  import shutil

  from zairachem.report.table import PerformanceTable

  path = _synthetic_run(tmp_path / "tables")
  shutil.rmtree(os.path.join(path, "report"), ignore_errors=True)
  assert not os.path.exists(os.path.join(path, "report"))

  # The performance table alone: the output table additionally needs the input mapping, which this
  # fixture does not build. Both go through the same path helper.
  PerformanceTable(path=path).run()
  assert os.listdir(os.path.join(path, "report")) == ["performance_table.csv"]


@pytest.mark.parametrize(
  ("task", "truth", "expect_clf"),
  [
    ("classification", False, True),  # the case regression broke: a predict on unlabelled molecules
    ("classification", True, True),
    ("regression", False, False),
    ("regression", True, False),
  ],
)
def test_the_output_table_follows_the_run_s_task_not_its_truth(tmp_path, task, truth, expect_clf):
  """A predict on unlabelled molecules has no truth column, and still belongs to its own task."""
  import json

  import pandas as pd

  from zairachem.report.table import OutputTable, PerformanceTable

  root = tmp_path / f"{task}-{truth}"
  (root / "inputs").mkdir(parents=True)
  (root / "metadata").mkdir()
  (root / "metadata" / "parameters.json").write_text(json.dumps({"task": task}))
  (root / "session.json").write_text(
    json.dumps({"mode": "predict", "output_dir": str(root), "model_dir": str(root)})
  )
  data = {"compound_id": ["CID0"], "smiles": ["CCO"]}
  if truth:
    data["bin" if task == "classification" else "val"] = [1 if task == "classification" else 6.5]
  pd.DataFrame(data).to_csv(root / "inputs" / "data.csv", index=False)

  assert OutputTable(path=str(root)).is_clf is expect_clf
  table = PerformanceTable(path=str(root))
  assert table.is_clf is expect_clf

  # Nothing to score against: it writes no table and does not fail the run.
  table.run()
  assert not (root / "report" / "performance_table.csv").exists()


def test_the_colour_key_separates_the_families_that_share_a_colour():
  """Two swatches of the same colour under one heading read as one meaning; they are not."""
  from zairachem.report.colors import hexcol, legend, legend_groups

  for task, family in (("classification", "Classes"), ("regression", "Values")):
    groups = legend_groups(task)
    assert [g for g, _ in groups] == [family, "Descriptors"]
    # Nothing is lost or duplicated by the grouping.
    assert [e for _, entries in groups for e in entries] == legend(task)
    # The collision this grouping exists for: a colour that appears in both families.
    task_colors = {hexcol(k) for _, k in groups[0][1]}
    assert task_colors & {hexcol(k) for _, k in groups[1][1]}
