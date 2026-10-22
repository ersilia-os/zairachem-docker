"""Regression report plots: how close the predictions are, and where they stop being trustworthy.

At fit the truth is known for every compound, so the panels compare predicted with observed. At
predict there is usually no truth, and the useful questions become where the predictions sit
relative to what the model was trained on, and how wide their intervals are.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from zairachem.report import BasePlot
from zairachem.report.colors import descriptor_colors_rgb, hexcol, named_colors
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report.plots._common import (
  _draw_training_projection_background,
  _npg_cmap,
  _rdkit_properties,
  _subsample,
)


def _finite_pair(a, b):
  """The rows where both ``a`` and ``b`` are finite, as float arrays."""
  if a is None or b is None:
    return None, None
  a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
  if a.shape != b.shape:
    return None, None
  m = np.isfinite(a) & np.isfinite(b)
  return a[m], b[m]


class IndividualEstimatorsR2Plot(BasePlot):
  """Per-descriptor R² against the observed target (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.is_available = False
    if not self.has_reg_data():
      return
    self.name = "r2-individual"
    fetcher = ResultsFetcher(path=path)
    yt = fetcher.get_true_reg()
    df = fetcher._read_individual_estimator_results(fetcher.get_reg_tasks()[0])
    labels, scores = [], []
    for col in list(df.columns):
      t, p = _finite_pair(yt, list(df[col]))
      if t is None or len(t) < 2:
        continue
      labels.append(col.split("-")[-1])
      scores.append(r2_score(t, p))
    if not labels:
      return
    # Same shape as the cross-validation panels: one bar per descriptor, best on top, each in its
    # identity colour, so the reader can flip between panels without re-reading the axis.
    order = sorted(range(len(labels)), key=lambda i: scores[i])
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]
    palette = list(reversed(descriptor_colors_rgb(len(labels))))
    ax = self.ax
    y = list(range(len(labels)))
    ax.barh(y, scores, color=palette, alpha=0.9, height=0.6, zorder=2)
    lo = min(0.0, min(scores))
    hi = max(1.0, max(scores))
    for i, v in enumerate(scores):
      ax.text(lo + (hi - lo) * 0.01, i, f"{v:.2f}", va="center", ha="left", fontsize=7, zorder=4)
    ax.axvline(0.0, color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("R2")
    ax.set_title("Per-descriptor R2")
    self.is_available = True


class RegressionPlotRaw(BasePlot):
  """Predicted against observed, with R² and MAE (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.has_reg_data():
      return
    self.name = "regression-raw"
    fetcher = ResultsFetcher(path=path)
    yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
    if yt is None or len(yt) < 2:
      return
    ax = self.ax
    ax.scatter(yt, yp, color=named_colors.green, s=15, alpha=0.7)
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    # The identity line is the reference a scatter like this is read against.
    ax.plot([lo, hi], [lo, hi], color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"R2 = {r2_score(yt, yp):.3f} | MAE = {mean_absolute_error(yt, yp):.3f}")
    self.is_available = True


class ResidualsPlot(BasePlot):
  """Residual against predicted value: where the model is biased (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.has_reg_data():
      return
    self.name = "regression-residuals"
    fetcher = ResultsFetcher(path=path)
    yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
    if yt is None or len(yt) < 2:
      return
    residual = yp - yt
    ax = self.ax
    ax.scatter(yp, residual, color=named_colors.blue, s=15, alpha=0.7)
    ax.axhline(0.0, color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Predicted − observed")
    ax.set_title(f"Residuals · mean {residual.mean():+.3f}")
    self.is_available = True


class PredictedValueDistributionPlot(BasePlot):
  """Predicted values against the training distribution (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.is_regression():
      return
    self.name = "regression-distribution"
    fetcher = ResultsFetcher(path=path)
    yp = fetcher.get_pred_value_reg()
    if yp is None:
      return
    yp = np.asarray(yp, dtype=float)
    yp = yp[np.isfinite(yp)]
    if yp.size == 0:
      return
    ax = self.ax
    bins = min(30, max(8, yp.size // 5))
    try:
      # At predict, the training values say whether these predictions sit inside what the model
      # has seen; at fit the two are the same set.
      train = np.asarray(fetcher.get_true_reg_trained(), dtype=float)
      train = train[np.isfinite(train)]
    except Exception:
      train = np.array([])
    if train.size:
      ax.hist(
        train,
        bins=bins,
        color=named_colors.gray,
        alpha=0.45,
        edgecolor="white",
        lw=0.4,
        label="Training",
      )
    ax.hist(
      yp,
      bins=bins,
      color=named_colors.green,
      alpha=0.8,
      edgecolor="white",
      lw=0.4,
      label="Predicted",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Compounds")
    ax.set_title("Predicted values")
    if train.size:
      ax.legend(loc="upper right", fontsize=6)
    self.is_available = True


class ErrorVsSimilarityPlot(BasePlot):
  """Absolute error, or interval width, against similarity to the training set (regression only).

  The question a chemist asks of a prediction on a new scaffold: does accuracy fall off outside the
  chemistry the model was trained on? With no observed values, the interval width answers the same
  question from the model's own uncertainty.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.is_regression():
      return
    self.name = "regression-error-similarity"
    fetcher = ResultsFetcher(path=path)
    try:
      similarity = fetcher.get_train_similarity()
    except Exception:
      similarity = None
    if similarity is None:
      return
    try:
      yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
    except Exception:  # no truth column at predict
      yt, yp = None, None
    if yt is not None and len(yt) == len(similarity):
      quantity, label = np.abs(yp - yt), "Absolute error"
    else:
      lower, upper = fetcher.get_pred_interval_reg()
      if lower is None:
        return
      quantity, label = (
        np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float),
        ("Interval width"),
      )
    sim, quantity = _finite_pair(similarity, quantity)
    if sim is None or len(sim) < 2:
      return
    ax = self.ax
    ax.scatter(sim, quantity, color=named_colors.purple, s=15, alpha=0.7)
    ax.set_xlabel("Similarity to the training set")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs training similarity")
    self.is_available = True


class ProjectionValuePlot(BasePlot):
  """A 2-D projection with each compound coloured by its target value (regression only).

  The regression counterpart of the classification projection maps, which colour by class. At fit
  the colour is the observed value, at predict the predicted one, so the same panel answers "where
  does potency sit in this chemical space?" in both reports. Predict runs also get the trained
  model's molecules as a faint backdrop, which is the applicability domain drawn rather than scored.
  """

  def __init__(self, ax, path, projection):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = f"projection-{projection['name']}-value"
    self.is_available = False
    if not self.is_regression():
      return
    fetcher = ResultsFetcher(path=path)
    try:
      values, label = list(fetcher.get_true_reg()), "Observed value"
    except Exception:  # no truth at predict
      values, label = fetcher.get_pred_value_reg(), "Predicted value"
    if values is None:
      return
    v = np.asarray(values, dtype=float)
    xs = np.asarray(projection["xs"], dtype=float)
    ys = np.asarray(projection["ys"], dtype=float)
    if v.size != xs.size:  # alignment guard: values and coordinates must be the same dedup rows
      return
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(v)
    if not keep.any():
      return
    x, y, v = xs[keep], ys[keep], v[keep]
    ax = self.ax
    if self.is_predict():
      _draw_training_projection_background(ax, path, projection["name"])
    idx, alpha = _subsample(x.size)
    # Grey-to-crimson: the same "more is crimson" ramp the classification maps use for probability,
    # starting at grey rather than white so the weakest compounds stay visible on the page.
    sc = ax.scatter(
      x[idx],
      y[idx],
      c=v[idx],
      cmap=_npg_cmap(hexcol("active"), hexcol("neutral")),
      s=10,
      alpha=alpha,
      edgecolors="none",
    )
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    ax.set_title(f"{projection['title']} · by value")
    self.is_available = True


class _PropertyValuePlot(BasePlot):
  """Base for a physico-chemical property against the target value (regression only).

  The classic QSAR failure mode is a model that has only learned "bigger" or "greasier", so the
  panel carries Spearman rho between the property and the target: a high one says the ranking
  could largely be reproduced by that property alone. Subclasses set ``stem``, ``label`` and
  ``prop_index`` (0=MW, 1=LogP, 2=TPSA in ``_rdkit_properties``).
  """

  stem = "property-value"
  label = ""
  prop_index = 0

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = self.stem  # BasePlot.__init__ sets self.name="base"; use the subclass stem
    self.is_available = False
    if not self.is_regression():
      return
    from scipy.stats import spearmanr

    fetcher = ResultsFetcher(path=path)
    smiles = fetcher.get_smiles()
    try:
      values, axis = list(fetcher.get_true_reg()), "Observed value"
    except Exception:  # no truth at predict
      values, axis = fetcher.get_pred_value_reg(), "Predicted value"
    if not smiles or values is None:
      return
    prop, v = _finite_pair(_rdkit_properties(smiles)[self.prop_index], values)
    if prop is None or len(prop) < 3:
      return
    rho = spearmanr(prop, v).statistic
    ax = self.ax
    idx, alpha = _subsample(prop.size)
    ax.scatter(prop[idx], v[idx], color=named_colors.blue, s=12, alpha=alpha, edgecolors="none")
    # A handful of absurd RDKit LogP values would otherwise squash the bulk into a narrow band, so
    # the view clips to the central 98%. Every point still counts towards rho.
    lo, hi = np.percentile(prop, [1, 99])
    if hi > lo:
      pad = 0.05 * (hi - lo)
      ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(self.label)
    ax.set_ylabel(axis)
    ax.set_title(f"{self.label} vs value · rho = {rho:+.2f}")
    self.is_available = True


class PropertyValueMwPlot(_PropertyValuePlot):
  """Molecular weight against the target value (regression only)."""

  stem = "property-mw-value"
  label = "Molecular weight"
  prop_index = 0


class PropertyValueLogpPlot(_PropertyValuePlot):
  """LogP against the target value (regression only)."""

  stem = "property-logp-value"
  label = "LogP"
  prop_index = 1


class IntervalCoveragePlot(BasePlot):
  """Does the prediction interval keep its promise, and does it keep it everywhere?

  The intervals are Mondrian split-conformal: the training residuals are split into bins by
  applicability-domain score and each bin gets its own half-width, so every compound is handed the
  width of the bin it falls in. The guarantee is only worth anything if it holds *within* each bin,
  which is what this checks -- one bar per distinct interval width (i.e. per bin), against the
  nominal level. Needs the observed values, so it renders at fit and at predict with truth.
  """

  NOMINAL = 0.9

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.is_regression():
      return
    self.name = "interval-coverage"
    fetcher = ResultsFetcher(path=path)
    lower, upper = fetcher.get_pred_interval_reg()
    if lower is None:
      return
    try:
      truth = np.asarray(fetcher.get_true_reg(), dtype=float)
    except Exception:
      return  # no truth: nothing to check the interval against
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if truth.shape != lo.shape:
      return
    ok = np.isfinite(truth) & np.isfinite(lo) & np.isfinite(hi)
    if ok.sum() < 2:
      return
    truth, lo, hi = truth[ok], lo[ok], hi[ok]
    covered = (truth >= lo) & (truth <= hi)
    width = np.round(hi - lo, 6)
    # One bar per Mondrian bin (a bin is exactly one half-width), narrowest -- most typical
    # chemistry -- first, then an "All" bar for the pooled coverage.
    bins = sorted(set(width.tolist()))
    labels, values = [], []
    for w in bins:
      m = width == w
      if m.sum() == 0:
        continue
      labels.append(f"{w:.2f}")
      values.append(float(covered[m].mean()))
    labels.append("All")
    values.append(float(covered.mean()))
    ax = self.ax
    x = list(range(len(labels)))
    bar_colors = [named_colors.blue] * (len(labels) - 1) + [named_colors.green]
    ax.bar(x, values, color=bar_colors, alpha=0.9, width=0.6, zorder=2)
    ax.axhline(self.NOMINAL, color=named_colors.gray, lw=1, ls="--", zorder=3)
    for i, v in enumerate(values):
      ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Interval width")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Interval coverage by width · nominal {self.NOMINAL:.0%} (--)")
    self.is_available = True


class PotencyEnrichmentPlot(BasePlot):
  """If you took the top k% by prediction, how potent would they actually be?

  The screening question, and the one a correlation coefficient does not answer. The curve is the
  mean *observed* value of the top k% ranked by prediction, against the dataset mean (dashed): the
  gap at small k is what the model buys over picking compounds at random. Needs the observed
  values, so it renders at fit and at predict with truth.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.is_regression():
      return
    self.name = "potency-enrichment"
    fetcher = ResultsFetcher(path=path)
    try:
      yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
    except Exception:
      return
    if yt is None or len(yt) < 20:
      return
    order = np.argsort(-yp)  # best predicted first
    observed = yt[order]
    n = observed.size
    # Mean observed value of the top k%, for every k -- a cumulative mean over the ranked list.
    running = np.cumsum(observed) / np.arange(1, n + 1)
    pct = np.arange(1, n + 1) / n * 100.0
    keep = pct >= 1.0  # the first few compounds are too noisy a mean to plot
    ax = self.ax
    ax.plot(pct[keep], running[keep], color=named_colors.green, lw=1.8, zorder=3)
    ax.axhline(float(yt.mean()), color=named_colors.gray, lw=1, ls="--", zorder=1)
    top10 = float(running[max(0, int(0.1 * n) - 1)])
    ax.set_xlabel("Top % by predicted value")
    ax.set_ylabel("Mean observed value")
    ax.set_title(f"Selection quality · top 10% averages {top10:.2f}")
    self.is_available = True


class ValueConsensusPlot(BasePlot):
  """Pooled predicted value against how much the descriptors disagreed about it (regression only).

  The regression counterpart of the classification consensus panel. A prediction the descriptors
  agree on is worth more than one they do not, whatever its value, so the spread is the second
  number to read beside the hitlist. Needs no truth, which is the point: it is a triage signal for
  compounds nobody has measured.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.is_available = False
    if not self.is_regression():
      return
    self.name = "value-consensus"
    fetcher = ResultsFetcher(path=path)
    pooled = fetcher.get_pred_value_reg()
    if pooled is None:
      return
    try:
      per_descriptor = fetcher._read_individual_estimator_results(fetcher.get_reg_tasks()[0])
    except Exception:
      return
    if per_descriptor is None or per_descriptor.shape[1] < 2:
      return  # one descriptor cannot disagree with itself
    values = per_descriptor.values.astype(float)
    spread = np.nanstd(values, axis=1)
    pooled = np.asarray(pooled, dtype=float)
    if pooled.size != values.shape[0]:
      pooled = np.nanmean(values, axis=1)
    pooled, spread = _finite_pair(pooled, spread)
    if pooled is None or pooled.size == 0:
      return
    ax = self.ax
    idx, alpha = _subsample(pooled.size)
    sc = ax.scatter(
      pooled[idx],
      spread[idx],
      c=pooled[idx],
      cmap=_npg_cmap(hexcol("active"), hexcol("neutral")),
      s=12,
      alpha=alpha,
      edgecolors="none",
    )
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Predicted value")
    ax.set_xlabel("Pooled predicted value")
    ax.set_ylabel("Descriptor disagreement (std)")
    ax.set_title("Descriptor consensus")
    self.is_available = True
