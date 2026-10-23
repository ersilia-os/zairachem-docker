"""The regression report: its panels, its sections, and the words it puts on them.

A regression run renders a different set of figures from a classification one, under the same
section headings and from the same registry. These tests pin that the two agree -- every panel
registered, titled and owned by a section -- and that each panel renders, or declines, without
raising against a run folder built here rather than by the pipeline.
"""

import os

import numpy as np
import pytest

# --- The report surface: every regression panel exists, and no panel belongs to both tasks -------


def _plot_specs():
  from zairachem.report.report import _PLOT_SPECS

  return dict(_PLOT_SPECS)


def test_every_regression_panel_is_registered_and_catalogued():
  from zairachem.report.html.catalog import (
    _CATEGORIES,
    _DATASET_GROUPS,
    _DIAG_GROUPS,
    _GROUPS,
    _TITLES,
  )

  specs = _plot_specs()
  panels = [
    "regression-raw",
    "regression-residuals",
    "regression-distribution",
    "regression-error-similarity",
    "potency-enrichment",
    "interval-coverage",
    "value-consensus",
    "cv-spearman",
    "property-mw-value",
    "property-logp-value",
    "r2-individual",
    "heldout-r2",
  ]
  missing = [p for p in panels if p not in specs]
  assert not missing, f"not registered in report.py: {missing}"
  # A figure with no catalogue title falls through to the report's "More" catch-all.
  untitled = [p for p in panels if p not in _TITLES]
  assert not untitled, f"no title in the HTML catalog: {untitled}"
  # A figure reaches the page through one of the section tables; which one depends on the section
  # that owns it (plot categories, the dataset and diagnostics blocks, or a carousel group).
  catalogued = {m for _a, _h, _d, members in _CATEGORIES for m in members}
  for table in (_GROUPS, _DIAG_GROUPS, _DATASET_GROUPS):
    catalogued |= {m for g in table for m in g["members"]}
  homeless = [p for p in panels if p not in catalogued]
  assert not homeless, f"in no HTML section: {homeless}"


def test_the_colour_legend_speaks_the_task_s_language():
  from zairachem.report.colors import hexcol, legend

  clf = dict((lbl, key) for lbl, key in legend("classification"))
  reg = dict((lbl, key) for lbl, key in legend("regression"))
  assert "Active" in clf and "Active" not in reg  # no classes in a regression report
  assert "High value" in reg and "Low value" in reg
  # Both tasks keep the provenance entries, and every key resolves to a real colour.
  for entries in (clf, reg):
    assert "From store" in entries and "Computed" in entries
    for key in entries.values():
      assert hexcol(key).startswith("#")


def test_every_semantic_colour_key_is_used():
  """A key nobody asks for is a claim about meaning that nothing keeps honest."""
  import pathlib
  import re

  from zairachem.report.colors import _SEMANTIC

  tree = pathlib.Path("zairachem")
  blob = "\n".join(p.read_text() for p in tree.rglob("*.py"))
  unused = [
    key
    for key in _SEMANTIC
    if len(re.findall(rf"""["']{re.escape(key)}["']""", blob)) <= 1  # only its own definition
  ]
  assert not unused, f"dead semantic colour keys: {unused}"


def test_inner_cv_reports_are_read_with_the_right_skill():
  from zairachem.report.cv import CHANCE, skill_keys, sort_best_first

  clf = [{"descriptor": "a", "oof_auc": 0.7}, {"descriptor": "b", "oof_auc": 0.9}]
  reg = [{"descriptor": "a", "oof_rho": 0.4}, {"descriptor": "b", "oof_rho": 0.8}]
  assert skill_keys(clf) == ("oof_auc", "train_auc", "AUROC")
  assert skill_keys(reg)[0] == "oof_rho"
  assert CHANCE["oof_auc"] == 0.5 and CHANCE["oof_rho"] == 0.0
  # Best first, on each task's own skill, and a missing skill never takes the top slot.
  assert [s["descriptor"] for s in sort_best_first(list(clf))] == ["b", "a"]
  assert [s["descriptor"] for s in sort_best_first(list(reg))] == ["b", "a"]
  with_gap = reg + [{"descriptor": "c"}]
  assert [s["descriptor"] for s in sort_best_first(with_gap)] == ["b", "a", "c"]


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


# --- The small task-aware helpers behind the report and the terminal ---------------------------


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


def test_a_run_knows_whether_it_has_observed_values(tmp_path):
  from zairachem.report.html.render import _has_truth

  labelled = _write_run(tmp_path / "a", "regression", [{"compound_id": "c1", "val": "5.0"}])
  unlabelled = _write_run(tmp_path / "b", "regression", [{"compound_id": "c1"}])
  assert _has_truth(labelled) and not _has_truth(unlabelled)
  # The same question for classification, and a folder that is not a run at all.
  assert _has_truth(
    _write_run(tmp_path / "c", "classification", [{"compound_id": "c", "bin": "1"}])
  )
  assert not _has_truth(str(tmp_path / "nowhere"))


def test_the_dataset_block_summarises_a_continuous_target(tmp_path):
  from zairachem.report.html.sections import _class_counts, _value_summary

  run = _write_run(
    tmp_path / "r",
    "regression",
    [{"compound_id": f"c{i}", "val": v} for i, v in enumerate("456789")],
  )
  summary = _value_summary(run)
  assert summary["total"] == 6 and summary["min"] == 4.0 and summary["max"] == 9.0
  assert summary["median"] == pytest.approx(6.5)
  # There is no class balance to report, and no value summary on a classification run.
  assert _class_counts(run) is None
  clf = _write_run(tmp_path / "c", "classification", [{"compound_id": "c", "bin": "1"}])
  assert _value_summary(clf) is None


def test_the_terminal_headline_names_the_task_s_skill():
  from zairachem.base.utils.summaries import _cv_headline

  clf = [{"descriptor": "a", "oof_auc": 0.80}, {"descriptor": "b", "oof_auc": 0.90}]
  reg = [{"descriptor": "a", "oof_rho": 0.60}, {"descriptor": "b", "oof_rho": 0.80}]
  label, mean, best, best_skill = _cv_headline(clf)
  assert (label, best, best_skill) == ("AUROC", "b", 0.90) and mean == pytest.approx(0.85)
  label, mean, best, best_skill = _cv_headline(reg)
  assert label.startswith("Spearman") and best == "b" and mean == pytest.approx(0.70)
  assert _cv_headline([{"descriptor": "a"}]) is None


def test_the_pooling_summary_does_not_claim_reliability_weighting_when_stacked():
  from zairachem.pool.reliability_pooler.pool import _tier

  assert (
    _tier(has_rank=True, has_ad=True, stacked=True) == "stacking weights + applicability domain"
  )
  assert _tier(has_rank=True, has_ad=False, stacked=True) == "stacking weights"
  # Classification is untouched by the new argument.
  assert "percentile reliability" in _tier(has_rank=True, has_ad=True)
  assert _tier(has_rank=False, has_ad=False) == "uniform (no applicability domain)"


# --- Every regression panel renders, or declines, without raising -------------------------------


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
  drawn = []
  for name in REGRESSION_PANELS:
    plot = getattr(plots_module, name)(ax=None, path=path)  # raises if the panel is broken
    if getattr(plot, "is_available", False):
      drawn.append(name)
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


def test_a_declined_plot_does_not_leave_its_figure_open(tmp_path):
  """Each plot belongs to one task, so most decline on any run -- after building a figure."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from zairachem.report.plots import RocCurvePlot
  from zairachem.report.report import _render_one

  path = _synthetic_run(tmp_path / "run")  # a regression run: the ROC plot has nothing to draw
  plt.close("all")
  for _ in range(25):  # matplotlib starts warning past twenty
    status, _name, _cells = _render_one(RocCurvePlot, path, {})
    assert status == "na"
  assert plt.get_fignums() == []


def test_the_coverage_panel_falls_back_when_the_widths_are_not_bins(tmp_path):
  """Mondrian gives three widths at most; a width per compound would be a bar per compound."""
  import pandas as pd

  from zairachem.report.plots import IntervalCoveragePlot

  path = _synthetic_run(tmp_path / "unbinned")
  pooled_path = os.path.join(path, "model", "pool", "results_unmapped.csv")
  df = pd.read_csv(pooled_path)
  half = np.linspace(0.5, 2.0, len(df))  # a distinct width for every compound
  df["reg_lower"], df["reg_upper"] = df["reg"] - half, df["reg"] + half
  df.to_csv(pooled_path, index=False)

  plot = IntervalCoveragePlot(ax=None, path=path)
  assert plot.is_available
  assert [t.get_text() for t in plot.ax.get_xticklabels()] == ["All"]


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


# --- The registry and the catalogue have to agree, for both tasks -------------------------------


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


def test_the_error_curve_is_the_cumulative_distribution_of_the_errors(tmp_path):
  """One line per descriptor, plus the error of predicting the training mean."""
  import json

  import pandas as pd

  from zairachem.report.plots import CvErrorCurvePlot

  path = _synthetic_run(tmp_path / "errors", n=100, descriptors=("eosA",))
  truth = np.linspace(1.0, 100.0, 100)
  # Errors 1.0, 2.0, ... 100.0 by construction, alternating sign so the mean stays put.
  errors = np.arange(1.0, 101.0) * np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
  d = os.path.join(path, "model", "estimators", "lq_estimators", "eosA")
  pd.DataFrame({"oof_value": truth + errors, "y": truth}).to_csv(
    os.path.join(d, "oof.csv"), index=False
  )
  json.dump({"oof_rho": 0.9, "train_rho": 0.95}, open(os.path.join(d, "cv_report.json"), "w"))

  plot = CvErrorCurvePlot(ax=None, path=path)
  assert plot.is_available
  descriptor_line = plot.ax.get_lines()[0]
  x, y = descriptor_line.get_xdata(), descriptor_line.get_ydata()
  assert x[0] == pytest.approx(1.0) and y[0] == pytest.approx(0.01)  # one of a hundred within 1.0
  assert x[49] == pytest.approx(50.0) and y[49] == pytest.approx(0.50)  # half within 50
  assert np.all(np.diff(y) > 0)  # a cumulative fraction only rises
  # The dashed baseline is the error of predicting the training mean for everything.
  baseline = plot.ax.get_lines()[-1]
  assert baseline.get_linestyle() == "--"
  assert max(baseline.get_xdata()) == pytest.approx(np.abs(truth - truth.mean()).max())


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


def test_a_target_that_goes_negative_renders_like_any_other(tmp_path):
  """LogD, binding free energy and pKa shifts are all routinely negative."""
  import pandas as pd

  from zairachem.report import plots as plots_module

  path = _synthetic_run(tmp_path / "negative", n=80)
  rng = np.random.default_rng(3)
  truth = rng.normal(-2.5, 1.5, 80)  # centred below zero, straddling it
  data_path = os.path.join(path, "inputs", "data.csv")
  data = pd.read_csv(data_path)
  data["val"] = truth
  data.to_csv(data_path, index=False)
  pooled_path = os.path.join(path, "model", "pool", "results_unmapped.csv")
  pooled = pd.read_csv(pooled_path)
  predicted = truth + rng.normal(0, 0.4, 80)
  pooled["reg"] = predicted
  pooled["reg_lower"], pooled["reg_upper"] = predicted - 1.0, predicted + 1.0
  pooled.to_csv(pooled_path, index=False)
  for eos in ("eosA", "eosB"):
    d = os.path.join(path, "model", "estimators", "lq_estimators", eos)
    per = pd.read_csv(os.path.join(d, "results_unmapped.csv"))
    per["reg"] = truth + rng.normal(0, 0.6, 80)
    per.to_csv(os.path.join(d, "results_unmapped.csv"), index=False)

  drawn = []
  for name in REGRESSION_PANELS:
    plot = getattr(plots_module, name)(ax=None, path=path)  # must not raise
    if getattr(plot, "is_available", False):
      drawn.append(name)
      assert "nan" not in plot.ax.get_title().lower(), name
  assert {"RegressionPlotRaw", "PotencyEnrichmentPlot", "IntervalCoveragePlot"} <= set(drawn)


def test_the_output_table_carries_the_regression_columns(tmp_path):
  """The deliverable: a value, its interval, and one column per descriptor -- and no call or rank."""
  import pandas as pd

  from zairachem.report.table import OutputTable

  path = _synthetic_run(tmp_path / "deliverable", n=40)
  OutputTable(path=path).run()
  table = pd.read_csv(os.path.join(path, "report", "output_table.csv"))
  assert len(table) == 40
  for column in ("input_smiles", "smiles", "label", "value", "lower", "upper"):
    assert column in table.columns, column
  assert {"value_eosA", "value_eosB"} <= set(table.columns)
  assert not {"proba", "call", "rank", "score"} & set(table.columns)
  assert (table["lower"] < table["value"]).all() and (table["value"] < table["upper"]).all()


def test_the_diagnostics_table_says_what_each_descriptor_cost():
  """One descriptor can take most of a run; nothing else in the report attributes the time."""
  from zairachem.report.html.tables import _cv_cell, _cv_table_html

  stats = [
    {"descriptor": "eosSlow", "oof_rho": 0.81, "train_rho": 0.99, "fit_seconds": 2501.2},
    {"descriptor": "eosFast", "oof_rho": 0.86, "train_rho": 0.99, "fit_seconds": 137.1},
  ]
  table = _cv_table_html(stats)
  assert "Train time" in table and "42m" in table and "2m" in table
  assert _cv_cell("fit_seconds", 45.0) == "45s"  # seconds below a minute and a half
  assert _cv_cell("fit_seconds", None) == "—"
  # A run whose reports predate the timing simply has no column.
  assert "Train time" not in _cv_table_html([{"descriptor": "eosA", "oof_rho": 0.8}])


def test_the_panels_hold_up_with_many_descriptors(tmp_path):
  """The default run keeps up to ten; the per-descriptor panels had only seen one to three."""
  from zairachem.report import plots as plots_module

  descriptors = tuple(f"eos{i:04d}" for i in range(10))
  path = _synthetic_run(tmp_path / "wide", n=80, descriptors=descriptors)

  for name in ("IndividualEstimatorsR2Plot", "CvSpearmanPlot", "CvErrorCurvePlot"):
    plot = getattr(plots_module, name)(ax=None, path=path)
    assert plot.is_available, name
    # Every descriptor gets a row (bars) or a line (curves), and a colour of its own.
    marks = len(plot.ax.patches) or len(plot.ax.get_lines())
    assert marks >= len(descriptors), (name, marks)

  from zairachem.report.colors import descriptor_colors_rgb

  colours = descriptor_colors_rgb(len(descriptors))
  assert len(colours) == len(descriptors)
