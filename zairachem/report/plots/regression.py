"""Regression report plots, grouped by the question each one answers.

Three groups, in the order a reader works through them. **Accuracy** compares predicted with
observed and needs the truth, so it renders at fit and at predict against a labelled test set.
**Confidence** needs no truth at all -- where the predictions sit relative to the training set,
how much the descriptors disagreed, how wide the intervals are -- which is what makes it the core
of a prediction run on molecules nobody has measured. (Descriptor disagreement goes further and
renders *only* without truth, as its classification counterpart does: once the observed values are
in hand the accuracy panels say the same thing and more.) **Chemistry** puts both on the molecules
themselves: the projections and the physico-chemical properties, coloured by value.

Every panel here renders only for a regression run. Most classification panels are the mirror of
that, because they rest on a cutoff or a class; the few that rest on neither -- the descriptor
correlation heatmap and the top-x% overlap curve -- serve both tasks from where they already live.
The result is one report in two vocabularies rather than two reports.
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


# --- Accuracy: predicted against observed (needs the truth) ------------------------------------


class IndividualEstimatorsR2Plot(BasePlot):
  """Per-descriptor R² against the observed target (regression only).

  How much each descriptor achieves alone, so the pooled R² can be read against its best member:
  pooling that does not beat the best descriptor is pooling that is not earning its keep. These
  are the descriptors' own predictions on the run's rows, not their inner cross-validation, which
  ``cv-spearman`` reports separately.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "r2-individual"
    self.is_available = False
    if not self.has_reg_data():
      return
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
    self.name = "regression-raw"
    self.is_available = False
    if not self.has_reg_data():
      return
    fetcher = ResultsFetcher(path=path)
    yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
    if yt is None or len(yt) < 2:
      return
    ax = self.ax
    idx, alpha = _subsample(yt.size)
    ax.scatter(yt[idx], yp[idx], color=named_colors.green, s=12, alpha=alpha, edgecolors="none")
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
    self.name = "regression-residuals"
    self.is_available = False
    if not self.has_reg_data():
      return
    fetcher = ResultsFetcher(path=path)
    yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
    if yt is None or len(yt) < 2:
      return
    residual = yp - yt
    ax = self.ax
    idx, alpha = _subsample(yp.size)
    ax.scatter(
      yp[idx], residual[idx], color=named_colors.blue, s=12, alpha=alpha, edgecolors="none"
    )
    ax.axhline(0.0, color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Predicted − observed")
    ax.set_title(f"Residuals · mean {residual.mean():+.3f}")
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
    self.name = "potency-enrichment"
    self.is_available = False
    if not self.has_reg_data():  # a target column means a regression run that has its truth
      return
    fetcher = ResultsFetcher(path=path)
    yt, yp = _finite_pair(fetcher.get_true_reg(), fetcher.get_pred_value_reg())
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


class IntervalCoveragePlot(BasePlot):
  """Does the prediction interval keep its promise, and does it keep it everywhere?

  The intervals are Mondrian split-conformal: the training residuals are split into bins by
  applicability-domain score and each bin gets its own half-width, so every compound is handed the
  width of the bin it falls in. The guarantee is only worth anything if it holds *within* each bin,
  which is what this checks -- one bar per distinct interval width (i.e. per bin), against the
  nominal level, which is lazy-qsar's own default rather than a number repeated here. Needs the
  observed values, so it renders at fit and at predict with truth.
  """

  #: Most distinct interval widths that can still be read as applicability-domain bins.
  MAX_BINS = 8

  def __init__(self, ax, path):
    from lazyqsar.ensemble.intervals import DEFAULT_LEVEL

    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "interval-coverage"
    self.is_available = False
    if not self.has_reg_data():  # without the truth there is nothing to check the interval against
      return
    fetcher = ResultsFetcher(path=path)
    lower, upper = fetcher.get_pred_interval_reg()
    if lower is None:
      return
    truth = np.asarray(fetcher.get_true_reg(), dtype=float)
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
    # chemistry -- first, then an "All" bar for the pooled coverage. The design gives three bins at
    # most; anything else is not a bin structure, and a bar per compound would say nothing, so the
    # panel falls back to the pooled coverage alone.
    bins = sorted(set(width.tolist()))
    if len(bins) > self.MAX_BINS:
      bins = []
    labels, values = [], []
    for w in bins:
      m = width == w
      if m.sum() == 0:
        continue
      labels.append(f"{w:.2f}")
      values.append(float(covered[m].mean()))
    if len(labels) != 1:
      # With a single bin the pooled bar would just repeat it; with none it is all there is.
      labels.append("All")
      values.append(float(covered.mean()))
    ax = self.ax
    x = list(range(len(labels)))
    bar_colors = [named_colors.blue] * (len(labels) - 1) + [named_colors.green]
    ax.bar(x, values, color=bar_colors, alpha=0.9, width=0.6, zorder=2)
    ax.axhline(DEFAULT_LEVEL, color=named_colors.gray, lw=1, ls="--", zorder=3)
    for i, v in enumerate(values):
      ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Interval width")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Interval coverage by width · nominal {DEFAULT_LEVEL:.0%} (--)")
    self.is_available = True


# --- Confidence: how far the predictions are from what the model has seen (no truth needed) ----


class PredictedValueDistributionPlot(BasePlot):
  """Predicted values against the distribution the model was trained on (regression only).

  Two things to read here. At predict, whether these molecules fall inside the range the model has
  seen at all -- a prediction extrapolated past the training data is a guess however confident the
  interval looks. And on either run, how much narrower the predictions are than the training
  values: every regression shrinks toward the mean, and the size of that shrinkage is how much the
  model is hedging. Both histograms share one set of bin edges, so they can be read bar by bar.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "regression-distribution"
    self.is_available = False
    if not self.is_regression():
      return
    fetcher = ResultsFetcher(path=path)
    yp = fetcher.get_pred_value_reg()
    if yp is None:
      return
    yp = np.asarray(yp, dtype=float)
    yp = yp[np.isfinite(yp)]
    if yp.size == 0:
      return
    ax = self.ax
    try:
      # At predict, the training values say whether these predictions sit inside what the model
      # has seen; at fit the two are the same set.
      train = np.asarray(fetcher.get_true_reg_trained(), dtype=float)
      train = train[np.isfinite(train)]
    except Exception:
      train = np.array([])
    # One set of edges across both histograms. Passing a bin *count* would let matplotlib derive
    # different edges from each set's own range, and the two would not be comparable bar by bar --
    # which is the entire point of drawing them together.
    n_bins = min(30, max(8, yp.size // 5))
    both = np.concatenate([yp, train]) if train.size else yp
    lo, hi = float(both.min()), float(both.max())
    edges = np.linspace(lo, hi, n_bins + 1) if hi > lo else n_bins
    if train.size:
      ax.hist(
        train,
        bins=edges,
        color=named_colors.gray,
        alpha=0.45,
        edgecolor="white",
        lw=0.4,
        label="Training",
      )
    ax.hist(
      yp,
      bins=edges,
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
    self.name = "regression-error-similarity"
    self.is_available = False
    if not self.is_regression():
      return
    fetcher = ResultsFetcher(path=path)
    try:
      similarity = fetcher.get_train_similarity()
    except Exception:
      similarity = None
    if similarity is None:
      return
    # Error where the truth is known, interval width where it is not. Both are left unfiltered
    # here and aligned against the similarity in one step below: filtering first would shorten the
    # array, and a single missing value would then silently switch the panel to the other quantity.
    try:
      truth = np.asarray(fetcher.get_true_reg(), dtype=float)
    except Exception:  # no truth column at predict
      truth = None
    predicted = fetcher.get_pred_value_reg()
    if truth is not None and predicted is not None and truth.size == len(similarity):
      quantity = np.abs(np.asarray(predicted, dtype=float) - truth)
      label = "Absolute error"
    else:
      lower, upper = fetcher.get_pred_interval_reg()
      if lower is None:
        return
      quantity = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
      label = "Interval width"
    sim, quantity = _finite_pair(similarity, quantity)
    if sim is None or len(sim) < 2:
      return
    ax = self.ax
    idx, alpha = _subsample(sim.size)
    ax.scatter(
      sim[idx], quantity[idx], color=named_colors.blue, s=12, alpha=alpha, edgecolors="none"
    )
    ax.set_xlabel("Similarity to the training set")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs training similarity")
    self.is_available = True


class ValueConsensusPlot(BasePlot):
  """Pooled predicted value against how much the descriptors disagreed about it (regression only).

  The regression counterpart of the classification consensus panel, and gated the same way: it
  renders only where there is no truth. A prediction the descriptors agree on is worth more than
  one they do not, whatever its value, so the spread is the second number to read beside the
  hitlist -- but once the observed values are in hand the error panels say all this and more.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "value-consensus"
    self.is_available = False
    if not self.is_regression() or self.has_reg_data():
      return
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


# --- Chemistry: the molecules themselves, coloured by value ------------------------------------


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
    # A constant property (or a constant target) has no rank correlation to report; say so rather
    # than printing "nan" in the title.
    rho_text = "—" if rho is None or not np.isfinite(rho) else f"{rho:+.2f}"
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
    ax.set_title(f"{self.label} vs value · rho = {rho_text}")
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
