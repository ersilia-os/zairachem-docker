"""Regression report plots: how close the predictions are, and where they stop being trustworthy.

At fit the truth is known for every compound, so the panels compare predicted with observed. At
predict there is usually no truth, and the useful questions become where the predictions sit
relative to what the model was trained on, and how wide their intervals are.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from zairachem.report import BasePlot
from zairachem.report.colors import descriptor_colors_rgb, named_colors
from zairachem.report.fetcher import ResultsFetcher


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
