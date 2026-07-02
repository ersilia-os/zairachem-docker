import os
import numpy as np
from matplotlib.patches import Rectangle

from sklearn import metrics
from sklearn.metrics import (
  auc,
  roc_curve,
  r2_score,
  mean_absolute_error,
  precision_recall_curve,
  average_precision_score,
  precision_score,
  recall_score,
  f1_score,
  matthews_corrcoef,
  balanced_accuracy_score,
)

import matplotlib as plt
import seaborn as sns
import pandas as pd

from zairachem.report import BasePlot
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report import perf
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# Colors come from the single source of truth in ``report/colors.py`` (semantic palette anchored to
# stylia's NPG ArticleColors), shared with perf.py and the HTML dashboard so nothing drifts. The
# color-keyed ``named_colors`` (named_colors.red etc.) and ``category_palette`` keep the names the
# plot call sites already use.
from zairachem.report.colors import (  # noqa: E402
  category_palette,
  descriptor_colors_rgb,
  named_colors,
  phase_color_rgb as _phase_color,
  rgb as _color,
)
from zairachem.base.vars import REPORT_SUBFOLDER, VALIDATION_TABLE_FILENAME  # noqa: E402


class ActivesInactivesPlot(BasePlot):
  """Bar chart of the active vs. inactive compound counts (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 2))
    if self.has_clf_data():
      self.is_available = True
      self.name = "actives-inactives"
      ax = self.ax
      rf = ResultsFetcher(path=path)
      y = rf.clf_truth_proba()[0]  # labelled truth only (NaN dropped)
      actives = int(np.sum(y))
      inactives = len(y) - actives
      ax.bar(
        x=["Actives", "Inactives"],
        height=[actives, inactives],
        color=[_color("active"), _color("inactive")],
      )
      y_max = max(actives, inactives)
      ax.set_ylim(0, y_max * 1.12)
      p = actives / len(y) * 100
      q = 100 - p
      # Counts above each bar; the class percentage inside the bar (no title needed).
      for x, count, pct in ((0, actives, p), (1, inactives, q)):
        ax.text(
          x, count + y_max * 0.02, f"{count:,}", va="bottom", ha="center", color=named_colors.black
        )
        ax.text(
          x, count / 2, f"{pct:.1f}%", va="center", ha="center", color="white", fontweight="bold"
        )
      ax.set_ylabel("Number of compounds")
      ax.set_xlabel("")
    else:
      self.is_available = False


class ConfusionPlot(BasePlot):
  """Confusion matrix of the pooled binary classifier (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_clf_data():
      self.is_available = True
      self.name = "confusion-matrix"
      ax = self.ax
      bt, bp = ResultsFetcher(path=path).clf_truth_binary()
      if len(set(bp)) > 1:
        class_names = ["I (0)", "A (1)"]
        disp = metrics.ConfusionMatrixDisplay(
          metrics.confusion_matrix(bt, bp), display_labels=class_names
        )
        disp.plot(ax=ax, cmap=plt.cm.Greens, colorbar=False)
        # for labels in disp.text_.ravel():
        # labels.set_fontsize(22)
        ax.grid(False)
        ax.set_title("Confusion matrix")
      else:
        self.is_available = False
    else:
      self.is_available = False


class RocCurvePlot(BasePlot):
  """ROC curve and AUROC of the pooled classifier (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_clf_data():
      self.is_available = True
      self.name = "roc-curve"
      ax = self.ax
      bt, yp = ResultsFetcher(path=path).clf_truth_proba()
      fpr, tpr, _ = roc_curve(bt, yp)
      auroc = auc(fpr, tpr)
      color = named_colors.blue
      ax.plot(fpr, tpr, color=color, zorder=10000, lw=1.6)
      ax.fill_between(fpr, tpr, color=color, alpha=0.16, lw=0, zorder=1000)
      ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--")
      ax.set_title("AUROC = {0}".format(round(auroc, 2)))
      ax.set_xlabel("1-Specificity (FPR)")
      ax.set_ylabel("Sensitivity (TPR)")
      ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
      ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
      self.is_available = False


class ScoreViolinPlot(BasePlot):
  """Violin plot of the classifier score distribution split by true class (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 2))
    self.name = "score-violin"
    self.is_available = False
    if not self.has_clf_data():
      return
    bt, yp = ResultsFetcher(path=path).clf_truth_proba()
    # Same style as the OOF score lenses: white inner box + median line, no whiskers.
    if not _draw_score_violins(self.ax, yp, bt, "Classifier score (probability)"):
      return
    self.ax.set_title("Score distribution")
    self.is_available = True


class ScoreStripPlot(BasePlot):
  """Per-compound classifier scores as a jittered strip with quartile boxes, by true class."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 2))
    self.MAX_SAMPLES = 1000
    if self.has_clf_data():
      self.is_available = True
      self.name = "score-strip"
      ax = self.ax
      bt, yp = ResultsFetcher(path=path).clf_truth_proba()
      data = pd.DataFrame({"yp": yp, "bt": bt})
      data_a = data[data["bt"] == 1]
      data_i = data[data["bt"] == 0]
      if data_a.shape[0] > self.MAX_SAMPLES:
        data_a = data_a.sample(n=self.MAX_SAMPLES)
      if data_i.shape[0] > self.MAX_SAMPLES:
        data_i = data_i.sample(n=self.MAX_SAMPLES)
      y_i = data_i["yp"]
      y_a = data_a["yp"]
      n_i = np.random.uniform(-0.3, 0.3, len(y_i))
      n_a = np.random.uniform(-0.3, 0.3, len(y_a))
      ax.scatter(n_a + 1, y_a, color=named_colors.red, zorder=1, alpha=0.5, s=20)
      ax.scatter(n_i, y_i, color=named_colors.blue, zorder=1, alpha=0.5, s=20)
      p05 = np.percentile(data_a["yp"], 5)
      p25 = np.percentile(data_a["yp"], 25)
      p50 = np.percentile(data_a["yp"], 50)
      p75 = np.percentile(data_a["yp"], 75)
      p95 = np.percentile(data_a["yp"], 95)
      r = Rectangle(
        (0.85, p25),
        0.3,
        p75 - p25,
        color=named_colors.red,
        alpha=0.5,
        zorder=20000,
        lw=0,
        edgecolor=named_colors.red,
      )
      ax.plot([1, 1], [p75, p95], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([1, 1], [p05, p25], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([0.85, 1.15], [p25, p25], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([0.85, 1.15], [p50, p50], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([0.85, 1.15], [p75, p75], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([0.85, 0.85], [p25, p75], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([1.15, 1.15], [p25, p75], lw=1, color=named_colors.black, zorder=20000)
      ax.add_patch(r)
      p05 = np.percentile(data_i["yp"], 5)
      p25 = np.percentile(data_i["yp"], 25)
      p50 = np.percentile(data_i["yp"], 50)
      p75 = np.percentile(data_i["yp"], 75)
      p95 = np.percentile(data_i["yp"], 95)
      r = Rectangle(
        (-0.15, p25),
        0.3,
        p75 - p25,
        color=named_colors.blue,
        alpha=0.5,
        zorder=20000,
        lw=0,
        edgecolor=named_colors.blue,
      )
      ax.plot([0, 0], [p75, p95], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([0, 0], [p05, p25], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([-0.15, 0.15], [p25, p25], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([-0.15, 0.15], [p50, p50], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([-0.15, 0.15], [p75, p75], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([-0.15, -0.15], [p25, p75], lw=1, color=named_colors.black, zorder=20000)
      ax.plot([0.15, 0.15], [p25, p75], lw=1, color=named_colors.black, zorder=20000)
      ax.add_patch(r)
      ax.set_xticks([0, 1])
      ax.set_xticklabels(["Inactive", "Active"])
      ax.set_title("Score distribution")
      ax.set_xlabel("")
      ax.set_ylabel("Classifier score (probability)")
      ax.set_xlim(-0.5, 1.5)

    else:
      self.is_available = False


def _percentile_rank(p):
  """Empirical percentile rank in [0, 1] of each value (ties broken by first occurrence)."""
  p = np.asarray(p, dtype=float)
  order = np.argsort(p, kind="mergesort")
  ranks = np.empty(len(p), dtype=float)
  ranks[order] = np.arange(len(p))
  return ranks / max(1, len(p) - 1)


def _draw_score_violins(ax, values, y, ylabel, show_points=False, max_pts=1000):
  """Vertical violins of a per-sample score split by class (inactive vs active).

  Always overlays a white inner boxplot — IQR box + median line + whiskers to the 5th/95th
  percentiles (with small caps). When ``show_points`` is set, up to ``max_pts`` jittered points per
  class are drawn over a lightened violin. Returns False (unavailable) if neither class has finite
  values."""
  y = np.asarray(y, dtype=int)
  v = np.asarray(values, dtype=float)
  finite = np.isfinite(v)
  present = [
    (c, name, _color(key))
    for c, name, key in ((0, "Inactive", "inactive"), (1, "Active", "active"))
    if np.any((y == c) & finite)
  ]
  if not present:
    return False
  order = [name for _, name, _ in present]
  palette = [col for _, _, col in present]
  data = pd.DataFrame({
    "value": v[finite],
    "cls": np.where(y[finite] == 1, "Active", "Inactive"),
  })
  sns.violinplot(
    x="cls", y="value", data=data, ax=ax, order=order, palette=palette, cut=0, inner=None
  )
  if show_points:
    for coll in ax.collections:  # lighten the violin bodies so the points read on top
      coll.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (c, _, col) in enumerate(present):
      pts = v[(y == c) & finite]
      if len(pts) > max_pts:
        pts = rng.choice(pts, max_pts, replace=False)
      jitter = rng.uniform(-0.18, 0.18, size=len(pts))
      ax.scatter(
        i + jitter, pts, s=7, color=col, alpha=0.6, linewidths=0.3, edgecolors="white", zorder=5
      )
  # Inner white boxplot: whiskers (5th–95th pct, with caps) + IQR box + a median line.
  for i, (c, _, _) in enumerate(present):
    p5, q1, med, q3, p95 = np.percentile(v[(y == c) & finite], [5, 25, 50, 75, 95])
    w, cap = 0.11, 0.05
    ax.plot([i, i], [q3, p95], color="white", lw=1.2, zorder=10)
    ax.plot([i, i], [p5, q1], color="white", lw=1.2, zorder=10)
    ax.plot([i - cap, i + cap], [p95, p95], color="white", lw=1.2, zorder=10)
    ax.plot([i - cap, i + cap], [p5, p5], color="white", lw=1.2, zorder=10)
    ax.add_patch(
      Rectangle((i - w, q1), 2 * w, q3 - q1, fill=False, edgecolor="white", lw=1.4, zorder=11)
    )
    ax.plot([i - w, i + w], [med, med], color="white", lw=1.8, zorder=12)
  ax.set_xticks(range(len(present)))
  ax.set_xticklabels(order)
  ax.set_xlabel("")
  ax.set_ylabel(ylabel)
  return True


class _PooledOofScorePlot(BasePlot):
  """Base for a pooled out-of-fold score distribution (actives vs inactives), shown through one
  score-type lens. The pooled classifier score (``clf_truth_proba``) is, at fit time, the honest
  out-of-fold prediction (each descriptor contributes its OOF probability scattered to compound
  order); this base draws it as vertical class-split violins with an inner white boxplot, optionally
  with jittered points.

  Subclasses set ``stem``/``title``/``ylabel`` and implement ``transform(p, y)`` → per-sample values
  (same length as ``p``), or return ``None`` when the lens has no data yet (e.g. the raw score before
  the pipeline persists it). ``show_points`` toggles the jittered-points variant. Classification only.
  """

  stem = None
  title = ""
  ylabel = ""
  show_points = False

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 2))
    self.name = self.stem
    self.is_available = False
    if not self.has_clf_data():
      return
    ts = self._truth_score(path)
    if ts is None:  # this lens' source score isn't available (e.g. raw before a re-fit persists it)
      return
    y, p = ts
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(set(y.tolist())) < 2:
      return
    vals = self.transform(p, y)
    if vals is None:
      return
    if not _draw_score_violins(
      self.ax, np.asarray(vals, dtype=float), y, self.ylabel, show_points=self.show_points
    ):
      return
    self.ax.set_title(self.title)
    self.is_available = True

  def _truth_score(self, path):
    """``(y_true, score)`` the lens operates on, or ``None`` if unavailable. Default: the pooled
    calibrated OOF probability. Overridden by the raw lens to use the uncalibrated pooled score."""
    return ResultsFetcher(path=path).clf_truth_proba()

  def transform(self, p, y):
    raise NotImplementedError


class OofScoreProbaPlot(_PooledOofScorePlot):
  stem = "oof-score-proba"
  title = "Out-of-fold pooled score · probability"
  ylabel = "Probability"

  def transform(self, p, y):
    return np.clip(p, 0.0, 1.0)


class OofScoreLogitPlot(_PooledOofScorePlot):
  stem = "oof-score-logit"
  title = "Out-of-fold pooled score · log-odds"
  ylabel = "Log-odds (logit)"

  def transform(self, p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


class OofScoreRankPlot(_PooledOofScorePlot):
  stem = "oof-score-rank"
  title = "Out-of-fold pooled score · percentile rank"
  ylabel = "Percentile rank"

  def transform(self, p, y):
    return _percentile_rank(p)


class OofScoreLiftPlot(_PooledOofScorePlot):
  stem = "oof-score-lift"
  title = "Out-of-fold pooled score · lift"
  ylabel = "Lift (× base rate)"

  def transform(self, p, y):
    prior = float(np.mean(y))
    return p / prior if prior > 0 else None


class OofScoreRawPlot(_PooledOofScorePlot):
  """Uncalibrated raw-score lens: the pooled OOF score before per-head probability calibration
  (lazy-qsar ≥ 3.4.2 exposes ``oof_raw_``; estimate pools & scatters it to a ``clf_raw`` column).
  Unavailable (blank) for models fit before that, when no pooled raw OOF was persisted."""

  stem = "oof-score-raw"
  title = "Out-of-fold pooled score · raw (uncalibrated)"
  ylabel = "Raw score (uncalibrated)"

  def _truth_score(self, path):
    return ResultsFetcher(path=path).clf_truth_raw()  # None when clf_raw wasn't persisted

  def transform(self, p, y):
    return np.clip(p, 0.0, 1.0)


# Jittered-points variants (same transforms, points over a lightened violin).
class OofScoreProbaPointsPlot(OofScoreProbaPlot):
  stem = "oof-score-proba-pts"
  title = "Out-of-fold pooled score · probability (points)"
  show_points = True


class OofScoreLogitPointsPlot(OofScoreLogitPlot):
  stem = "oof-score-logit-pts"
  title = "Out-of-fold pooled score · log-odds (points)"
  show_points = True


class OofScoreRankPointsPlot(OofScoreRankPlot):
  stem = "oof-score-rank-pts"
  title = "Out-of-fold pooled score · percentile rank (points)"
  show_points = True


class OofScoreLiftPointsPlot(OofScoreLiftPlot):
  stem = "oof-score-lift-pts"
  title = "Out-of-fold pooled score · lift (points)"
  show_points = True


class IndividualEstimatorsAurocPlot(BasePlot):
  """Per-descriptor AUROC, one horizontal bar per descriptor model (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    if self.has_clf_data():
      self.name = "roc-individual"
      ax = self.ax
      self.fetcher = ResultsFetcher(path=path)
      tasks = self.fetcher.get_clf_tasks()
      task = tasks[0]
      mask = self.fetcher.clf_labeled_mask()  # labelled (NaN-free) rows only
      bt = np.asarray(self.fetcher.get_actives_inactives(), dtype=float)[mask].astype(int)
      df_ys = self.fetcher._read_individual_estimator_results(task)
      aucs = []
      labels = []
      for yp in sorted(df_ys.columns):
        fpr, tpr, _ = roc_curve(bt, np.asarray(df_ys[yp], dtype=float)[mask])
        aucs += [auc(fpr, tpr)]
        labels += [yp]
      y = list(range(len(labels)))
      x = aucs
      colors = category_palette.get(len(labels))  # one NPG categorical color per estimator

      def _format_label(l):
        l = l.replace("_", " ").replace("-", " ")
        return l.title()

      for i in y:
        ax.text(
          0.75,
          i,
          "{0} / {1}".format(_format_label(labels[i]), np.round(x[i], 3)),
          va="center",
          ha="center",
        )
      for i in y:
        r = Rectangle((0, i - 0.3), x[i], 0.6, color=colors[i], alpha=0.85)
        ax.add_patch(r)
      ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
      ax.set_ylabel("Estimators")
      ax.set_yticklabels("")
      ax.set_xlim(0.45, 1.05)
      ax.set_ylim(-0.6, len(labels) - 0.4)
      ax.set_yticks(y)
      ax.set_xlabel("AUROC")
      ax.set_title("Individual performances")
      self.is_available = True
    else:
      self.is_available = False


class IndividualEstimatorsClassificationScorePlot(BasePlot):
  """Per-descriptor classifier score distributions as boxplots (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    if self.has_clf_data():
      self.name = "raw-classification-scores"
      ax = self.ax
      self.fetcher = ResultsFetcher(path=path)
      tasks = self.fetcher.get_clf_tasks()
      task = tasks[0]
      df_ys = self.fetcher._read_individual_estimator_results(task)
      vals = []
      labels = []
      for yp in list(df_ys.columns):
        vals += list(df_ys[yp])
        labels += [yp] * len(df_ys[yp])
      data = pd.DataFrame({"label": labels, "values": vals})
      sns.boxplot(x="label", y="values", data=data, ax=ax)
      ax.set_ylabel("Classification score (probability)")
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      ax.set_xlabel("")
      self.is_available = True
    else:
      self.is_available = False


class IndividualEstimatorsR2Plot(BasePlot):
  """Per-descriptor R² against the transformed target (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    if self.has_reg_data():
      self.name = "r2-individual"
      ax = self.ax
      self.fetcher = ResultsFetcher(path=path)
      tasks = self.fetcher.get_reg_tasks()
      task = tasks[0]
      yt = ResultsFetcher(path=path).get_transformed()
      df_ys = self.fetcher._read_individual_estimator_results(task)
      scores = []
      labels = []
      for yp in list(df_ys.columns):
        scores += [r2_score(yt, list(df_ys[yp]))]
        labels += [yp]
      x = list(range(len(labels)))
      y = scores
      ax.scatter(x, y, color=named_colors.red)
      ax.set_xticks(x)
      ax.set_xticklabels(labels, rotation=90)
      ax.set_ylabel("R2")
      self.is_available = True
    else:
      self.is_available = False


def _subsample(n, cap=10000):
  """Point indices to keep for a class of size ``n`` (random subsample above ``cap``) and an adaptive
  alpha (lower for denser plots), so heavily populated projections stay readable."""
  if n <= cap:
    idx = np.arange(n)
  else:
    idx = np.random.default_rng(0).choice(n, cap, replace=False)
  alpha = float(np.clip(1500.0 / max(1, len(idx)), 0.12, 0.9))
  return idx, alpha


def _draw_class_density(ax, x, y, color):
  """One class's filled-KDE density surface + translucent points onto ``ax`` (no legend)."""
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if len(x) == 0:
    return
  idx, alpha = _subsample(len(x))
  xs, ys = x[idx], y[idx]
  try:
    if len(xs) >= 5:
      sns.kdeplot(
        x=xs, y=ys, ax=ax, fill=True, color=color, levels=8, thresh=0.05, alpha=0.35, bw_adjust=1.1
      )
  except Exception:
    pass
  ax.scatter(xs, ys, color=color, s=8, alpha=alpha, edgecolors="none", zorder=5)


def _projection_xy_by_class(path, projection):
  """``(x, y, bt)`` for a projection keeping only finite-coordinate, labelled rows (bt in {0,1})."""
  bt = np.asarray(ResultsFetcher(path=path).get_actives_inactives(), dtype=float)
  xs = np.asarray(projection["xs"], dtype=float)
  ys = np.asarray(projection["ys"], dtype=float)
  keep = np.isfinite(xs) & np.isfinite(ys) & ~np.isnan(bt)
  return xs[keep], ys[keep], bt[keep].astype(int)


class ProjectionMergedPlot(BasePlot):
  """A 2-D projection with both classes overlaid: per-class density surface + points (no legend).

  ``projection`` is a dict ``{name, title, x_label, y_label, xs, ys}`` from
  ``ResultsFetcher.get_projections``. Coloured by true class (crimson actives, cobalt inactives).
  """

  def __init__(self, ax, path, projection):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "projection-merged-" + projection["name"]
    self.is_available = False
    x, y, bt = _projection_xy_by_class(path, projection)
    if len(bt) == 0:
      return
    ax = self.ax
    _draw_class_density(ax, x[bt == 0], y[bt == 0], _color("inactive"))
    _draw_class_density(ax, x[bt == 1], y[bt == 1], _color("active"))
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    ax.set_title(projection["title"])
    self.is_available = True


class ProjectionClassPlot(BasePlot):
  """A 2-D projection for a single class (active or inactive): density surface + points (no legend)."""

  def __init__(self, ax, path, projection, cls):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    noun = "active" if cls == 1 else "inactive"
    self.name = f"projection-{projection['name']}-{noun}"
    self.is_available = False
    x, y, bt = _projection_xy_by_class(path, projection)
    mask = bt == cls
    if not mask.any():
      return
    ax = self.ax
    _draw_class_density(ax, x[mask], y[mask], _color(noun))
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    ax.set_title(f"{projection['title']} · {noun}s")
    self.is_available = True


class RegressionPlotTransf(BasePlot):
  """Predicted vs. observed scatter on the transformed activity, with R²/MAE (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_reg_data():
      self.is_available = True
      self.name = "regression-trans"
      ax = self.ax
      yt = ResultsFetcher(path=path).get_transformed()
      yp = ResultsFetcher(path=path).get_pred_reg_trans()
      ax.scatter(yt, yp, color=named_colors.purple, s=15, alpha=0.7)
      ax.set_xlabel("Observed Activity (Transformed)")
      ax.set_ylabel("Predicted Activity (Transformed)")
      ax.set_title(
        "R2 = {0} | MAE = {1}".format(
          round(r2_score(yt, yp), 3), round(mean_absolute_error(yt, yp), 3)
        )
      )
    else:
      self.is_available = False


class HistogramPlotTransf(BasePlot):
  """Histogram of the predicted (transformed) activity (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    if self.has_reg_data():
      self.is_available = True
      self.name = "histogram-trans"
      ax = self.ax
      yp = ResultsFetcher(path=path).get_pred_reg_trans()
      ax.hist(yp, color=named_colors.green)
      ax.set_xlabel("Predicted Activity")
      ax.set_ylabel("Frequency")
      ax.set_title("Predicted activity distribution")
    else:
      self.is_available = False


class RegressionPlotRaw(BasePlot):
  """Predicted vs. observed scatter on the raw activity, with R²/MAE (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_reg_data():
      self.is_available = True
      self.name = "regression-raw"
      ax = self.ax
      yt = ResultsFetcher(path=path).get_raw()
      yp = ResultsFetcher(path=path).get_pred_reg_raw()
      ax.scatter(yt, yp, color=named_colors.green, s=15, alpha=0.7)
      ax.set_xlabel("Observed Activity")
      ax.set_ylabel("Predicted Activity")
      ax.set_title(
        "R2 = {0} | MAE = {1}".format(
          round(r2_score(yt, yp), 3), round(mean_absolute_error(yt, yp), 3)
        )
      )
    else:
      self.is_available = False


class HistogramPlotRaw(BasePlot):
  """Histogram of the predicted (raw) activity (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    if self.has_reg_data():
      self.is_available = True
      self.name = "histogram-raw"
      ax = self.ax
      yp = ResultsFetcher(path=path).get_pred_reg_raw()
      ax.hist(yp, color=named_colors.green)
      ax.set_xlabel("Predicted Activity")
      ax.set_ylabel("Frequency")
      ax.set_title("Predicted activity distribution")
    else:
      self.is_available = False


class Transformation(BasePlot):
  """Scatter of the raw → transformed activity mapping applied before training (regression only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    if self.has_reg_data():
      self.is_available = True
      self.name = "transformation"
      ax = self.ax
      yt = ResultsFetcher(path=path).get_raw()
      ytrans = ResultsFetcher(path=path).get_transformed()
      ax.scatter(yt, ytrans, color=named_colors.green, s=15, alpha=0.7)
      ax.set_xlabel("Observed Activity (Raw)")
      ax.set_ylabel("Observed Activity (Transformed)")
      ax.set_title("Continuous data transformation")
    else:
      self.is_available = False


def _descriptor_color_map(stats):
  """``{descriptor: rgb}`` identity colours by best-first rank (``stats`` from ``get_cv_stats``)."""
  palette = descriptor_colors_rgb(len(stats))
  return {s["descriptor"]: palette[i] for i, s in enumerate(stats)}


class CvAurocPlot(BasePlot):
  """Per-descriptor cross-validation (OOF) AUROC bars, with the train AUROC marked for reference.

  Each bar carries the descriptor's identity colour (matching its ROC/PR line). The gap between the
  bar (OOF) and the │ marker (train) is the per-descriptor overfitting.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "cv-auroc"
    ax = self.ax
    stats_bf = [s for s in ResultsFetcher(path=path).get_cv_stats() if s.get("oof_auc") is not None]
    if not stats_bf:
      self.is_available = False
      return
    color_of = _descriptor_color_map(stats_bf)
    stats = list(reversed(stats_bf))  # ascending → best on top
    labels = [s["descriptor"] for s in stats]
    oof = [s["oof_auc"] for s in stats]
    y = list(range(len(labels)))
    ax.barh(
      y, oof, color=[color_of[s["descriptor"]] for s in stats], alpha=0.9, height=0.6, zorder=2
    )
    for i, s in enumerate(stats):
      tr = s.get("train_auc")
      if tr is not None:
        ax.plot([tr], [i], marker="|", markersize=14, color=named_colors.black, zorder=3)
      ax.text(0.41, i, f"{oof[i]:.2f}", va="center", ha="left", fontsize=7, zorder=4)
    ax.axvline(0.5, color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.4, 1.03)
    ax.set_xlabel("AUROC")
    ax.set_title("Cross-validation AUROC (bar) vs train (│)")
    self.is_available = True


class _CvBarPlot(BasePlot):
  """Base for a per-descriptor horizontal-bar metric figure over the lazy-qsar OOF predictions.

  Every bar plot shares one row order — descriptors best-first by inner-CV AUROC, best on top — and
  the per-descriptor identity colour, so flipping between metrics keeps each descriptor in the same
  place with the same hue. Subclasses set ``stem``/``title``/``xlabel``/``xlim``/``noskill`` and
  implement :meth:`compute` (returns the metric for one descriptor, or ``None`` to skip it).
  Threshold metrics receive ``preds`` already thresholded at the descriptor's ideal cutoff.
  """

  stem = None
  title = None
  xlabel = None
  xlim = (0, 1.03)
  noskill = None  # None, a fixed value, or "prior" (the base rate) → dashed reference line

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = self.stem
    ax = self.ax
    rf = ResultsFetcher(path=path)
    stats_bf = rf.get_cv_stats()
    color_of = _descriptor_color_map(stats_bf)
    entries, prior = [], None
    for s in stats_bf:
      oof = rf.get_cv_oof(s["descriptor"])
      if oof is None:
        continue
      proba = np.asarray(oof[0], dtype=float)
      yv = np.asarray(oof[1], dtype=int)
      if len(set(yv.tolist())) < 2:
        continue
      prior = float(yv.mean())
      cutoff = s.get("decision_cutoff_proba")
      preds = (proba >= cutoff).astype(int) if cutoff is not None else None
      val = self.compute(s, proba, yv, preds, prior)
      if val is None:
        continue
      entries.append((s["descriptor"], float(val)))
    if not entries:
      self.is_available = False
      return
    entries = list(reversed(entries))  # stats_bf is best-first → reverse puts the best bar on top
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    y = list(range(len(labels)))
    ax.barh(y, vals, color=[color_of[n] for n in labels], alpha=0.9, height=0.6, zorder=2)
    x0, x1 = self.xlim
    for i, v in enumerate(vals):
      ax.text(x0 + (x1 - x0) * 0.01, i, f"{v:.2f}", va="center", ha="left", fontsize=7, zorder=4)
    ns = prior if self.noskill == "prior" else self.noskill
    if ns is not None:
      ax.axvline(ns, color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(x0, x1)
    ax.set_xlabel(self.xlabel)
    ax.set_title(self.title)
    self.is_available = True

  def compute(self, s, proba, y, preds, prior):
    raise NotImplementedError


class CvAuprPlot(_CvBarPlot):
  """Per-descriptor OOF AUPR (average precision) bars, with the no-skill prior (base rate) marked."""

  stem = "cv-aupr"
  title = "Cross-validation AUPR (bar) vs no-skill prior (--)"
  xlabel = "AUPR (average precision)"
  noskill = "prior"

  def compute(self, s, proba, y, preds, prior):
    return average_precision_score(y, proba)


class CvMccPlot(_CvBarPlot):
  """Per-descriptor Matthews correlation coefficient at each descriptor's ideal cutoff."""

  stem = "cv-mcc"
  title = "MCC at the ideal cutoff"
  xlabel = "MCC (no skill = 0)"

  def compute(self, s, proba, y, preds, prior):
    return matthews_corrcoef(y, preds) if preds is not None else None


class CvF1Plot(_CvBarPlot):
  """Per-descriptor F1 score at each descriptor's ideal cutoff."""

  stem = "cv-f1"
  title = "F1 at the ideal cutoff"
  xlabel = "F1"

  def compute(self, s, proba, y, preds, prior):
    return f1_score(y, preds, zero_division=0) if preds is not None else None


class CvBalancedAccuracyPlot(_CvBarPlot):
  """Per-descriptor balanced accuracy at each descriptor's ideal cutoff (no skill = 0.5)."""

  stem = "cv-balacc"
  title = "Balanced accuracy at the ideal cutoff vs no skill (--)"
  xlabel = "Balanced accuracy"
  noskill = 0.5

  def compute(self, s, proba, y, preds, prior):
    return balanced_accuracy_score(y, preds) if preds is not None else None


class CvPrecisionPlot(_CvBarPlot):
  """Per-descriptor precision at each descriptor's ideal cutoff, with the no-skill prior marked."""

  stem = "cv-precision"
  title = "Precision at the ideal cutoff vs no-skill prior (--)"
  xlabel = "Precision"
  noskill = "prior"

  def compute(self, s, proba, y, preds, prior):
    return precision_score(y, preds, zero_division=0) if preds is not None else None


class CvRecallPlot(_CvBarPlot):
  """Per-descriptor recall (sensitivity) at each descriptor's ideal cutoff."""

  stem = "cv-recall"
  title = "Recall at the ideal cutoff"
  xlabel = "Recall (sensitivity)"

  def compute(self, s, proba, y, preds, prior):
    return recall_score(y, preds, zero_division=0) if preds is not None else None


class CvCutoffPlot(_CvBarPlot):
  """Per-descriptor ideal decision cutoff (the operating probability lazy-qsar selected)."""

  stem = "cv-cutoff"
  title = "Ideal decision cutoff (probability) vs 0.5 (--)"
  xlabel = "Decision cutoff (probability)"
  noskill = 0.5

  def compute(self, s, proba, y, preds, prior):
    return s.get("decision_cutoff_proba")


class CvRocPlot(BasePlot):
  """Overlaid out-of-fold ROC curves, one per descriptor (identity-coloured), with a bottom-right
  legend giving each descriptor's OOF AUROC."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "cv-roc"
    ax = self.ax
    rf = ResultsFetcher(path=path)
    stats_bf = rf.get_cv_stats()
    color_of = _descriptor_color_map(stats_bf)
    curves = []
    for s in stats_bf:
      oof = rf.get_cv_oof(s["descriptor"])
      if oof is None:
        continue
      proba, yv = oof
      if len(set(yv)) < 2:
        continue
      fpr, tpr, _ = roc_curve(yv, proba)
      curves.append((s["descriptor"], s.get("oof_auc"), fpr, tpr))
    if not curves:
      self.is_available = False
      return
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--", zorder=1)
    for name, a, fpr, tpr in curves:
      label = f"{name} ({a:.2f})" if a is not None else name
      ax.plot(fpr, tpr, color=color_of[name], lw=1.6, alpha=0.9, zorder=2, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1-Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Cross-validation ROC")
    ax.legend(loc="lower right", fontsize=6)
    self.is_available = True


class CvPrPlot(BasePlot):
  """Overlaid out-of-fold precision-recall curves, one per descriptor (identity-coloured), with a
  bottom-left legend giving each descriptor's AUPRC. The dashed line is the no-skill prior."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "cv-pr"
    ax = self.ax
    rf = ResultsFetcher(path=path)
    stats_bf = rf.get_cv_stats()
    color_of = _descriptor_color_map(stats_bf)
    curves, prior = [], None
    for s in stats_bf:
      oof = rf.get_cv_oof(s["descriptor"])
      if oof is None:
        continue
      proba, yv = oof
      if len(set(yv)) < 2:
        continue
      precision, recall, _ = precision_recall_curve(yv, proba)
      curves.append((s["descriptor"], average_precision_score(yv, proba), recall, precision))
      prior = float(np.mean(yv))
    if not curves:
      self.is_available = False
      return
    if prior is not None:
      ax.plot([0, 1], [prior, prior], color=named_colors.gray, lw=1, ls="--", zorder=1)
    for name, ap, recall, precision in curves:
      ax.plot(
        recall,
        precision,
        color=color_of[name],
        lw=1.6,
        alpha=0.9,
        zorder=2,
        label=f"{name} ({ap:.2f})",
      )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Cross-validation precision-recall")
    ax.legend(loc="lower left", fontsize=6)
    self.is_available = True


class CvCalibrationPlot(BasePlot):
  """Reliability curve on the out-of-fold predictions of the best descriptor."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "cv-calibration"
    ax = self.ax
    rf = ResultsFetcher(path=path)
    stats = rf.get_cv_stats()
    oof = rf.get_cv_oof(stats[0]["descriptor"]) if stats else None
    if oof is None:
      self.is_available = False
      return
    proba = np.asarray(oof[0], dtype=float)
    yv = np.asarray(oof[1], dtype=float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
      m = idx == b
      if m.sum() > 0:
        xs.append(float(proba[m].mean()))
        ys.append(float(yv[m].mean()))
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--", label="Perfect")
    ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=named_colors.blue, label=stats[0]["descriptor"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (OOF)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (cross-validated)")
    ax.legend(fontsize=7)
    self.is_available = True


class CvScoreDistributionPlot(BasePlot):
  """Out-of-fold score distribution (actives vs inactives) for the best descriptor."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 2))
    self.name = "cv-score-distribution"
    ax = self.ax
    rf = ResultsFetcher(path=path)
    stats = rf.get_cv_stats()
    oof = rf.get_cv_oof(stats[0]["descriptor"]) if stats else None
    if oof is None:
      self.is_available = False
      return
    data = pd.DataFrame({"yp": oof[0], "bt": oof[1]})
    if data["bt"].nunique() < 2:
      self.is_available = False
      return
    sns.violinplot(x="bt", y="yp", data=data, ax=ax, palette=[named_colors.blue, named_colors.red])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"])
    ax.set_xlabel("")
    ax.set_ylabel("Out-of-fold score (probability)")
    ax.set_title(f"OOF scores · {stats[0]['descriptor']}")
    self.is_available = True


# --- Additional classification plots (ranking, operating point, per-descriptor diagnostics) -------
#
# Each draws on ``self.ax`` (so it works standalone AND when handed an axis from the composed grid),
# guards on ``has_clf_data()`` and degenerate inputs (single class / empty CV), and sets
# ``self.is_available`` so the report skips it cleanly when there's nothing to show.


def _clf_truth_proba(path):
  """(y_true, y_proba) as float arrays for the pooled classifier, or (None, None).

  Drops unlabelled positions (NaN truth) so partial predict-time ground truth works; a no-op when
  truth is complete (e.g. at fit)."""
  rf = ResultsFetcher(path=path)
  yt = rf.get_actives_inactives()
  yp = rf.get_pred_proba_clf()
  if yt is None or yp is None:
    return None, None
  return rf._aligned_truth_pred(yt, yp)


class PrCurvePlot(BasePlot):
  """Precision-Recall curve (+ average precision); dashed baseline at class prevalence."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "pr-curve"
    self.is_available = False
    if not self.has_clf_data():
      return
    yt, yp = _clf_truth_proba(path)
    if yt is None or len(np.unique(yt)) < 2:
      return
    ax = self.ax
    precision, recall, _ = precision_recall_curve(yt, yp)
    ap = average_precision_score(yt, yp)
    prevalence = float(np.mean(yt))
    ax.plot(recall, precision, color=named_colors.blue, lw=1.6, zorder=1000)
    ax.fill_between(recall, precision, color=named_colors.blue, alpha=0.16, lw=0)
    ax.axhline(prevalence, color=named_colors.gray, lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"AUPR = {ap:.2f}")
    self.is_available = True


class EnrichmentCurvePlot(BasePlot):
  """Cumulative gain: fraction of actives recovered vs fraction of the library screened (by score)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "enrichment-curve"
    self.is_available = False
    if not self.has_clf_data():
      return
    yt, yp = _clf_truth_proba(path)
    if yt is None or yt.sum() < 1:
      return
    ax = self.ax
    order = np.argsort(-yp)
    yt_sorted = yt[order]
    n = len(yt_sorted)
    n_act = float(yt_sorted.sum())
    frac_screened = np.arange(1, n + 1) / n
    frac_found = np.cumsum(yt_sorted) / n_act
    ideal = np.minimum(frac_screened * n / n_act, 1.0)
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--", label="Random")
    ax.plot(frac_screened, ideal, color=named_colors.green, lw=1, ls=":", label="Ideal")
    ax.plot(frac_screened, frac_found, color=named_colors.blue, lw=1.6, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Fraction of library screened")
    ax.set_ylabel("Fraction of actives found")
    ax.set_title("Cumulative gain")
    ax.legend(fontsize=6, loc="lower right")
    self.is_available = True


class EnrichmentFactorPlot(BasePlot):
  """Enrichment factor at 1% / 5% / 10% of the ranked library (hit-rate-in-top / prevalence)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "enrichment-factor"
    self.is_available = False
    if not self.has_clf_data():
      return
    yt, yp = _clf_truth_proba(path)
    if yt is None or yt.sum() < 1:
      return
    ax = self.ax
    n = len(yt)
    prevalence = float(yt.mean())
    yt_sorted = yt[np.argsort(-yp)]
    fracs = [0.01, 0.05, 0.10]
    efs = []
    for fr in fracs:
      k = max(1, int(round(fr * n)))
      hit_rate = float(yt_sorted[:k].mean())
      efs.append(hit_rate / prevalence if prevalence > 0 else 0.0)
    labels = ["1%", "5%", "10%"]
    colors = category_palette.get(len(labels))
    ax.bar(labels, efs, color=colors, alpha=0.85)
    ax.axhline(1.0, color=named_colors.gray, lw=1, ls="--")
    for i, ef in enumerate(efs):
      ax.text(i, ef, f"{ef:.1f}×", va="bottom", ha="center", fontsize=7)
    ax.set_xlabel("Top fraction screened")
    ax.set_ylabel("Enrichment factor")
    ax.set_title("Early enrichment")
    self.is_available = True


class ThresholdSweepPlot(BasePlot):
  """Precision / recall / F1 / MCC as a function of the decision threshold."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "threshold-sweep"
    self.is_available = False
    if not self.has_clf_data():
      return
    yt, yp = _clf_truth_proba(path)
    if yt is None or len(np.unique(yt)) < 2:
      return
    ax = self.ax
    ts = np.linspace(0.02, 0.98, 49)
    prec, rec, f1s, mccs = [], [], [], []
    for t in ts:
      pred = (yp >= t).astype(int)
      prec.append(precision_score(yt, pred, zero_division=0))
      rec.append(recall_score(yt, pred, zero_division=0))
      f1s.append(f1_score(yt, pred, zero_division=0))
      mccs.append(matthews_corrcoef(yt, pred) if len(np.unique(pred)) > 1 else 0.0)
    ax.plot(ts, prec, color=named_colors.blue, lw=1.4, label="Precision")
    ax.plot(ts, rec, color=named_colors.red, lw=1.4, label="Recall")
    ax.plot(ts, f1s, color=named_colors.green, lw=1.4, label="F1")
    ax.plot(ts, mccs, color=named_colors.purple, lw=1.4, label="MCC")
    # Mark the operating threshold implied by the pooled binary calls, if recoverable. Use the
    # labelled-subset binary calls so it stays aligned with `yp` (which dropped unlabelled rows).
    _, bp = ResultsFetcher(path=path).clf_truth_binary()
    if bp is not None and len(bp) == len(yp):
      pos = yp[bp == 1]
      neg = yp[bp == 0]
      if len(pos) and len(neg):
        t_op = (float(pos.min()) + float(neg.max())) / 2
        ax.axvline(t_op, color=named_colors.gray, lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold sweep")
    ax.legend(fontsize=6, loc="lower center", ncol=2)
    self.is_available = True


class DescriptorMetricHeatmapPlot(BasePlot):
  """Per-descriptor CV metrics heatmap: OOF AUROC, train AUROC, overfit gap (column-normalized)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "descriptor-metric-heatmap"
    self.is_available = False
    stats = [s for s in ResultsFetcher(path=path).get_cv_stats() if s.get("oof_auc") is not None]
    if not stats:
      return
    ax = self.ax
    cols = ["OOF AUROC", "train AUROC", "overfit gap"]

    def _gap(s):
      g = s.get("overfit_gap")
      if g is None and s.get("train_auc") is not None:
        g = s["train_auc"] - s["oof_auc"]
      return g

    raw = np.array([[s.get("oof_auc"), s.get("train_auc"), _gap(s)] for s in stats], dtype=float)
    labels = [s["descriptor"] for s in stats]
    # Colour each column on its own min-max scale (metrics live on different ranges).
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
      col = raw[:, j]
      finite = col[np.isfinite(col)]
      lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
      norm[:, j] = 0.5 if hi == lo else (col - lo) / (hi - lo)
    ax.imshow(norm, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(raw.shape[0]):
      for j in range(raw.shape[1]):
        v = raw[i, j]
        if np.isfinite(v):
          ax.text(j, i, f"{v:.2f}", va="center", ha="center", fontsize=6, color="white")
    ax.set_title("Per-descriptor CV metrics")
    self.is_available = True


class OofOverfitScatterPlot(BasePlot):
  """Per-descriptor generalization map: out-of-fold AUROC vs overfit gap (train − OOF)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "oof-overfit-scatter"
    self.is_available = False
    stats = [s for s in ResultsFetcher(path=path).get_cv_stats() if s.get("oof_auc") is not None]
    if not stats:
      return
    ax = self.ax
    colors = category_palette.get(len(stats))
    for s, c in zip(stats, colors):
      gap = s.get("overfit_gap")
      if gap is None and s.get("train_auc") is not None:
        gap = s["train_auc"] - s["oof_auc"]
      if gap is None:
        continue
      ax.scatter(s["oof_auc"], gap, color=c, s=30, zorder=10)
      ax.annotate(
        s["descriptor"],
        (s["oof_auc"], gap),
        fontsize=5,
        xytext=(3, 3),
        textcoords="offset points",
      )
    ax.axhline(0, color=named_colors.gray, lw=1, ls="--")
    ax.set_xlabel("Out-of-fold AUROC")
    ax.set_ylabel("Overfit gap (train − OOF)")
    ax.set_title("Generalization by descriptor")
    self.is_available = True


class PooledVsBestAurocPlot(BasePlot):
  """Per-descriptor OOF AUROC bars with the pooled-model AUROC drawn as a reference line."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "pooled-vs-best-auroc"
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    stats = [s for s in rf.get_cv_stats() if s.get("oof_auc") is not None]
    if not stats:
      return
    ax = self.ax
    stats = sorted(stats, key=lambda s: s["oof_auc"])
    labels = [s["descriptor"] for s in stats]
    vals = [s["oof_auc"] for s in stats]
    y = list(range(len(labels)))
    ax.barh(y, vals, color=named_colors.gray, alpha=0.8, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    yt, yp = _clf_truth_proba(path)
    if yt is not None and len(np.unique(yt)) > 1:
      fpr, tpr, _ = roc_curve(yt, yp)
      pooled = auc(fpr, tpr)
      ax.axvline(pooled, color=named_colors.red, lw=1.4, label=f"Pooled {pooled:.2f}")
      ax.legend(fontsize=6, loc="lower right")
    ax.axvline(0.5, color=named_colors.gray, lw=1, ls="--", zorder=0)
    ax.set_xlim(0.4, 1.03)
    ax.set_xlabel("AUROC (OOF per descriptor)")
    ax.set_title("Pooled vs individual")
    self.is_available = True


class NormalizedConfusionPlot(BasePlot):
  """Row-normalized confusion matrix (per-class recall %), complementing the raw-count plot."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "confusion-normalized"
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    bt, bp = rf.clf_truth_binary()
    if bt is None or bp is None or len(bt) == 0 or len(set(bp.tolist())) < 2:
      return
    ax = self.ax
    cm = metrics.confusion_matrix(bt, bp).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    ax.imshow(norm, cmap=plt.cm.Greens, vmin=0, vmax=1)
    labels = ["I (0)", "A (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
      for j in range(2):
        ax.text(
          j,
          i,
          f"{norm[i, j] * 100:.0f}%",
          va="center",
          ha="center",
          fontsize=8,
          color="white" if norm[i, j] > 0.5 else "black",
        )
    ax.grid(False)
    ax.set_title("Confusion (row-normalized)")
    self.is_available = True


def _rdkit_properties(smiles):
  """(MW, LogP, TPSA) arrays for valid molecules; NaN where SMILES can't be parsed."""
  from rdkit import Chem
  from rdkit.Chem import Descriptors

  mw, logp, tpsa = [], [], []
  for smi in smiles:
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
      mw.append(np.nan)
      logp.append(np.nan)
      tpsa.append(np.nan)
    else:
      mw.append(Descriptors.MolWt(mol))
      logp.append(Descriptors.MolLogP(mol))
      tpsa.append(Descriptors.TPSA(mol))
  return np.array(mw), np.array(logp), np.array(tpsa)


def _draw_one_property(ax, values, bt, label):
  """Horizontal violin plot of one property split by class (actives vs inactives). Availability bool."""
  bt = np.asarray(bt, dtype=int)
  vals = np.asarray(values, dtype=float)
  finite = np.isfinite(vals)
  order = [
    n
    for n, present in (
      ("Inactive", np.any((bt == 0) & finite)),
      ("Active", np.any((bt == 1) & finite)),
    )
    if present
  ]
  if not order:
    return False
  data = pd.DataFrame({
    "value": vals[finite],
    "cls": np.where(bt[finite] == 1, "Active", "Inactive"),
  })
  palette = {"Inactive": _color("inactive"), "Active": _color("active")}
  sns.violinplot(
    x="value", y="cls", data=data, ax=ax, order=order, palette=[palette[n] for n in order], cut=0
  )
  # Overlay a white-outlined boxplot: whiskers (5th–95th pct, with caps) + IQR box + median line.
  for i, name in enumerate(order):
    cls = 1 if name == "Active" else 0
    p5, q1, med, q3, p95 = np.percentile(vals[(bt == cls) & finite], [5, 25, 50, 75, 95])
    h, cap = 0.13, 0.06
    ax.plot([q3, p95], [i, i], color="white", lw=1.2, zorder=10)
    ax.plot([p5, q1], [i, i], color="white", lw=1.2, zorder=10)
    ax.plot([p95, p95], [i - cap, i + cap], color="white", lw=1.2, zorder=10)
    ax.plot([p5, p5], [i - cap, i + cap], color="white", lw=1.2, zorder=10)
    ax.add_patch(
      Rectangle((q1, i - h), q3 - q1, 2 * h, fill=False, edgecolor="white", lw=1.4, zorder=11)
    )
    ax.plot([med, med], [i - h, i + h], color="white", lw=1.6, zorder=12)
  ax.set_xlabel(label)
  ax.set_ylabel("")
  return True


class _PropertyDistributionPlot(BasePlot):
  """Base for a single physico-chemical property histogram (actives vs inactives). Subclasses set
  ``stem``, ``label`` and ``prop_index`` (0=MW, 1=LogP, 2=TPSA in ``_rdkit_properties``)."""

  stem = "property"
  label = ""
  prop_index = 0

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = self.stem  # BasePlot.__init__ sets self.name="base"; use the subclass stem
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    smiles = rf.get_smiles()
    bt = rf.get_actives_inactives()
    if not smiles or bt is None:
      return
    keep = ~np.isnan(np.asarray(bt, dtype=float))
    smiles = [s for s, k in zip(smiles, keep) if k]
    bt = np.asarray(bt, dtype=float)[keep].astype(int)
    if len(bt) == 0:
      return
    values = _rdkit_properties(smiles)[self.prop_index]
    self.is_available = _draw_one_property(self.ax, values, bt, self.label)


class PropertyMwPlot(_PropertyDistributionPlot):
  """Molecular-weight distribution by class."""

  stem = "property-mw"
  label = "Molecular weight"
  prop_index = 0


class PropertyLogpPlot(_PropertyDistributionPlot):
  """LogP distribution by class."""

  stem = "property-logp"
  label = "LogP"
  prop_index = 1


def _draw_donut_labels(ax, wedges, labels, r=1.22, min_gap=0.34):
  """Outer wedge labels with leader lines, nudged apart vertically so they never overlap.

  Each label is anchored at its wedge's mid-angle on the ring, then labels sharing a horizontal side
  are spread to keep at least ``min_gap`` between them — the fix for donuts whose slices are so uneven
  that the built-in labels would collide."""
  info = []
  for w, lab in zip(wedges, labels):
    ang = np.deg2rad(0.5 * (w.theta1 + w.theta2))
    info.append({"lab": lab, "ang": ang, "x": float(np.cos(ang)), "y": float(np.sin(ang))})
  for side in (1, -1):
    grp = sorted((d for d in info if (1 if d["x"] >= 0 else -1) == side), key=lambda d: d["y"])
    for i in range(1, len(grp)):
      if grp[i]["y"] - grp[i - 1]["y"] < min_gap:
        grp[i]["y"] = grp[i - 1]["y"] + min_gap
  for d in info:
    right = d["x"] >= 0
    ax.annotate(
      d["lab"],
      xy=(np.cos(d["ang"]), np.sin(d["ang"])),
      xytext=(r if right else -r, d["y"]),
      ha="left" if right else "right",
      va="center",
      fontsize=8,
      arrowprops=dict(arrowstyle="-", color=named_colors.gray, lw=0.8, connectionstyle="arc3"),
    )


class ClassDonutPlot(BasePlot):
  """Donut of the active/inactive class balance (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "class-donut"
    self.is_available = False
    if not self.has_clf_data():
      return
    y = ResultsFetcher(path=path).clf_truth_proba()[0]
    actives = int(np.sum(y))
    inactives = len(y) - actives
    if actives + inactives == 0:
      return
    self.is_available = True
    ax = self.ax
    # No built-in outer labels (they collide on uneven splits); percentages sit inside each wedge and
    # the class names are placed by the de-colliding helper with leader lines.
    wedges, _texts, _auto = ax.pie(
      [actives, inactives],
      colors=[_color("active"), _color("inactive")],
      startangle=90,
      counterclock=False,
      wedgeprops=dict(width=0.42, edgecolor="white"),
      autopct=lambda p: f"{p:.0f}%",
      pctdistance=0.78,
      textprops={"fontsize": 7},
    )
    _draw_donut_labels(ax, wedges, ["Active", "Inactive"])
    ax.text(0, 0, f"{actives + inactives:,}", ha="center", va="center", fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.35, 1.35)


class ClassWafflePlot(BasePlot):
  """Waffle grid (100 cells) coloured by class in proportion (classification only)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "class-waffle"
    self.is_available = False
    if not self.has_clf_data():
      return
    from matplotlib.colors import ListedColormap

    y = ResultsFetcher(path=path).clf_truth_proba()[0]
    actives = int(np.sum(y))
    total = len(y)
    if total == 0:
      return
    self.is_available = True
    ax = self.ax
    rows, cols = 5, 20
    n_active_cells = int(round(100.0 * actives / total))
    flat = np.array([1] * n_active_cells + [0] * (rows * cols - n_active_cells))
    grid = flat.reshape(rows, cols)
    ax.imshow(grid, cmap=ListedColormap([_color("inactive"), _color("active")]), aspect="equal")
    # White gridlines between cells so the 100 squares read as a waffle rather than solid bands.
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_visible(False)
    ax.set_xlabel(f"Each square ≈ 1%  ·  {n_active_cells}% active")


# --- Computational performance figures (read session.json / provenance.json via perf) ------------


class StepTimingPlot(BasePlot):
  """Horizontal bars of wall-clock seconds per pipeline sub-step, coloured by phase."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "step-timing"
    self.is_available = False
    steps = [s for s in perf.step_telemetry(path)["steps"] if s["seconds"] is not None]
    if not steps:
      return
    self.is_available = True
    ax = self.ax
    y = list(range(len(steps)))
    ax.barh(
      y,
      [s["seconds"] for s in steps],
      color=[_phase_color(s["phase"]) for s in steps],
    )
    ax.set_yticks(y)
    ax.set_yticklabels([s["label"] for s in steps], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")


class PhaseTimeDonutPlot(BasePlot):
  """Donut of total wall-clock time aggregated by pipeline phase."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "phase-time"
    self.is_available = False
    agg = {}
    for s in perf.step_telemetry(path)["steps"]:
      if s["seconds"] is not None:
        agg[s["phase"]] = agg.get(s["phase"], 0.0) + s["seconds"]
    agg = {k: v for k, v in agg.items() if v > 0}
    if not agg:
      return
    self.is_available = True
    ax = self.ax
    phases = [p for p in perf.PHASE_ORDER if p in agg] + [
      p for p in agg if p not in perf.PHASE_ORDER
    ]
    vals = [agg[p] for p in phases]
    ax.pie(
      vals,
      labels=[p.capitalize() for p in phases],
      colors=[_phase_color(p) for p in phases],
      startangle=90,
      counterclock=False,
      wedgeprops=dict(width=0.42, edgecolor="white"),
      autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",
      pctdistance=0.78,
      textprops={"fontsize": 7},
    )
    ax.text(0, 0, perf.fmt_duration(sum(vals)), ha="center", va="center", fontsize=9)


class ResourceTimelinePlot(BasePlot):
  """CPU and RAM usage (%) across the pipeline steps."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "resource-timeline"
    self.is_available = False
    steps = [
      s
      for s in perf.step_telemetry(path)["steps"]
      if s["cpu"] is not None or s["ram_pct"] is not None
    ]
    if not steps:
      return
    self.is_available = True
    ax = self.ax
    x = list(range(len(steps)))
    cpu = [s["cpu"] if s["cpu"] is not None else np.nan for s in steps]
    ram = [s["ram_pct"] if s["ram_pct"] is not None else np.nan for s in steps]
    if not np.all(np.isnan(cpu)):
      ax.plot(x, cpu, marker="o", ms=3, lw=1.5, color=named_colors.red, label="CPU %")
    if not np.all(np.isnan(ram)):
      ax.plot(x, ram, marker="o", ms=3, lw=1.5, color=named_colors.blue, label="RAM %")
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in steps], rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Usage (%)")
    # Legend above the axes (frameless, horizontal) so it never overlaps the lines.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7)


class ProvenanceBarPlot(BasePlot):
  """Per-model molecule provenance: descriptors read from the store vs freshly computed."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "compute-provenance"
    self.is_available = False
    prov = perf.provenance(path)
    if not prov:
      return
    models = prov["models"]
    self.is_available = True
    ax = self.ax
    store = [m["from_store"] for m in models]
    y = list(range(len(models)))
    ax.barh(y, store, color=named_colors.blue, label="From store")
    ax.barh(
      y, [m["computed"] for m in models], left=store, color=named_colors.gray, label="Computed"
    )
    ax.set_yticks(y)
    ax.set_yticklabels([m["id"] for m in models], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Molecules")
    # Legend above the axes (frameless, horizontal) so it never overlaps the bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7)


class PerModelTimingPlot(BasePlot):
  """Wall-clock time spent computing each Ersilia model (featurizers + projections), if recorded."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = "model-timing"
    self.is_available = False
    prov = perf.provenance(path)
    if not prov:
      return
    models = [m for m in prov["models"] if m.get("seconds") is not None]
    if not models:
      return
    self.is_available = True
    ax = self.ax
    models = sorted(models, key=lambda m: m["seconds"], reverse=True)
    # A distinct NPG colour per model (the stylia-idiomatic palette for a many-bar chart). The model
    # ids on the y-axis identify each bar, so no legend is needed (and none can overlap).
    y = list(range(len(models)))
    ax.barh(y, [m["seconds"] for m in models], color=category_palette.get(len(models)))
    ax.set_yticks(y)
    ax.set_yticklabels([m["id"] for m in models], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")


class HeldOutValidationPlot(BasePlot):
  """Held-out AUROC per split schema, one point per fold (classification, ``--evaluate`` only).

  Random is the optimism anchor (dashed reference line); a drop under scaffold / Butina indicates
  weaker generalization to novel chemistry. Rendered only when ``--evaluate`` produced
  ``report/validation_table.csv``; otherwise the plot marks itself unavailable and is skipped.
  """

  # Display order and friendly x-axis labels for the schemas.
  _ORDER = [
    ("random", "Random"),
    ("scaffold", "Scaffold"),
    ("scaffold_det", "Scaffold\n(DeepChem)"),
    ("butina", "Butina"),
  ]

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(3, 3))
    self.name = "heldout-validation"
    self.is_available = False
    csv_path = os.path.join(path, REPORT_SUBFOLDER, VALIDATION_TABLE_FILENAME)
    if not os.path.exists(csv_path):
      return
    df = pd.read_csv(csv_path)
    if df.empty or "auroc" not in df.columns or "strategy" not in df.columns:
      return
    seen = set(df["strategy"])
    present = [(k, lbl) for k, lbl in self._ORDER if k in seen]
    present += [(k, k) for k in sorted(seen) if k not in {p[0] for p in self._ORDER}]
    if not present:
      return
    ax = self.ax
    colors = category_palette.get(len(present))
    random_mean = None
    xticks, xticklabels = [], []
    for i, (strat, lbl) in enumerate(present):
      vals = np.asarray(df[df["strategy"] == strat]["auroc"], dtype=float)
      vals = vals[~np.isnan(vals)]
      if len(vals) == 0:
        continue
      jitter = np.random.uniform(-0.12, 0.12, len(vals))
      ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[i], alpha=0.7, s=25, zorder=3)
      m = float(vals.mean())
      if strat == "random":
        random_mean = m
      ax.plot([i - 0.2, i + 0.2], [m, m], color=named_colors.black, lw=1.5, zorder=4)
      if len(vals) > 1:
        s = float(vals.std())
        ax.plot([i, i], [m - s, m + s], color=named_colors.black, lw=1, zorder=4)
      xticks.append(i)
      xticklabels.append(lbl)
    if not xticks:
      return
    if random_mean is not None:
      ax.axhline(random_mean, ls="--", lw=1, color=named_colors.gray, zorder=1)
    ax.axhline(0.5, ls=":", lw=1, color=named_colors.gray, zorder=1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(-0.5, len(present) - 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Held-out validation")
    self.is_available = True
