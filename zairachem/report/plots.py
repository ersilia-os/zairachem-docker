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
)

import matplotlib as plt
import seaborn as sns
import pandas as pd

from zairachem.report import BasePlot
from zairachem.report.fetcher import ResultsFetcher
from stylia import ArticleColors, CategoricalPalette, DivergingColormap
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# Publication palette: NPG (Nature Publishing Group) colors via stylia's ArticleColors, mapped to
# the semantic names the plots use. Not Ersilia-branded — suitable for papers.
_article_colors = ArticleColors()


class _Palette:
  blue = _article_colors.cobalt
  gray = _article_colors.silver
  black = _article_colors.black
  purple = _article_colors.periwinkle
  red = _article_colors.crimson
  green = _article_colors.lime


named_colors = _Palette()

# Cycling NPG palette for categorical series (a distinct color per model / estimator).
category_palette = CategoricalPalette("npg")


class ActivesInactivesPlot(BasePlot):
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
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
        color=[named_colors.red, named_colors.blue],
      )
      y_min = 0
      y_max = max(actives, inactives)
      y_range = y_max - y_min
      ax.set_ylim(0 - y_range * 0.02, y_max + y_range * 0.1)
      ax.text(
        0,
        actives + y_max * 0.02,
        actives,
        va="center",
        ha="center",
        color=named_colors.red,
      )
      ax.text(
        1,
        inactives + y_max * 0.02,
        inactives,
        va="center",
        ha="center",
        color=named_colors.blue,
      )
      ax.set_ylabel("Number of compounds")
      ax.set_xlabel("")
      p = np.round(actives / len(y) * 100, 1)
      q = np.round(100 - p, 1)
      ax.set_title("Actives = {0}%, Inactives = {1}%".format(p, q))
    else:
      self.is_available = False


class ConfusionPlot(BasePlot):
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
    if self.has_clf_data():
      self.is_available = True
      self.name = "score-violin"
      ax = self.ax
      bt, yp = ResultsFetcher(path=path).clf_truth_proba()
      data = pd.DataFrame({"yp": yp, "bt": bt})
      sns.violinplot(
        x="bt",
        y="yp",
        data=data,
        ax=ax,
        palette=[named_colors.blue, named_colors.red],
      )
      ax.set_xticks([0, 1])
      ax.set_xticklabels(["Inactive", "Active"])
      ax.set_title("Score distribution")
      ax.set_xlabel("")
      ax.set_ylabel("Classifier score (probability)")
    else:
      self.is_available = False


class ScoreStripPlot(BasePlot):
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
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


class IndividualEstimatorsAurocPlot(BasePlot):
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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


class ProjectionPlot(BasePlot):
  """Generic 2-D projection scatter.

  ``projection`` is a dict ``{name, title, x_label, y_label, xs, ys}`` (from
  ``ResultsFetcher.get_projections``). Points are coloured by the pooled prediction (classification
  probability, or regression value); known actives/inactives are outlined. In predict mode the
  training set is shown as a faint grey backdrop.
  """

  def __init__(self, ax, path, projection):
    BasePlot.__init__(self, ax=ax, path=path)
    self.name = "projection-" + projection["name"]
    ax = self.ax
    fetcher = ResultsFetcher(path=path)
    xs = np.array(projection["xs"], dtype=float)
    ys = np.array(projection["ys"], dtype=float)

    if self.is_predict():
      trained = fetcher.get_projection_trained(projection["name"])
      if trained is not None:
        tx, ty = trained
        ax.scatter(
          tx, ty, color=named_colors.gray, s=5, alpha=0.5, edgecolors="none", label="Training set"
        )

    if self.has_clf_data():
      bp = fetcher.get_actives_inactives()
      y_pred = fetcher.get_pred_proba_clf()
      cmap = DivergingColormap()
      cmap.fit(y_pred)
      ax.scatter(
        xs, ys, color=cmap.transform(y_pred), alpha=0.7, s=15, zorder=100000, edgecolors="none"
      )
      ina = [i for i, v in enumerate(bp) if v == 0]
      act = [i for i, v in enumerate(bp) if v == 1]
      ax.scatter(
        xs[ina],
        ys[ina],
        facecolor="none",
        edgecolors=named_colors.blue,
        s=15,
        lw=0.5,
        label="Known inactives",
        zorder=1000000,
      )
      ax.scatter(
        xs[act],
        ys[act],
        facecolor="none",
        edgecolors=named_colors.red,
        s=15,
        lw=0.5,
        label="Known actives",
        zorder=10000000,
      )
    else:
      y_pred = fetcher.get_pred_reg_raw() or fetcher.get_raw()
      if y_pred is not None:
        cmap = DivergingColormap()
        cmap.fit(y_pred)
        ax.scatter(xs, ys, color=cmap.transform(y_pred), alpha=0.7, s=15, edgecolors="none")
      else:
        ax.scatter(xs, ys, color=named_colors.blue, alpha=0.7, s=15, edgecolors="none")

    ax.set_title(projection["title"])
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    if ax.get_legend_handles_labels()[1]:
      ax.legend()
    self.is_available = True


class RegressionPlotTransf(BasePlot):
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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
  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
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


class CvAurocPlot(BasePlot):
  """Per-descriptor cross-validation (OOF) AUROC bars, with the train AUROC marked for reference.

  The gap between the bar (OOF) and the │ marker (train) is the per-descriptor overfitting.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
    self.name = "cv-auroc"
    ax = self.ax
    stats = [s for s in ResultsFetcher(path=path).get_cv_stats() if s.get("oof_auc") is not None]
    if not stats:
      self.is_available = False
      return
    stats = sorted(stats, key=lambda s: s["oof_auc"])  # ascending → best on top
    labels = [s["descriptor"] for s in stats]
    oof = [s["oof_auc"] for s in stats]
    y = list(range(len(labels)))
    colors = category_palette.get(len(labels))
    ax.barh(y, oof, color=colors, alpha=0.85, height=0.6, zorder=2)
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


class CvRocPlot(BasePlot):
  """Overlaid out-of-fold ROC curves, one per descriptor (legend shows each CV AUROC)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
    self.name = "cv-roc"
    ax = self.ax
    rf = ResultsFetcher(path=path)
    curves = []
    for s in rf.get_cv_stats():
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
    colors = category_palette.get(len(curves))
    for (name, a, fpr, tpr), c in zip(curves, colors):
      label = f"{name} ({a:.2f})" if a is not None else name
      ax.plot(fpr, tpr, color=c, lw=1.4, label=label)
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--")
    ax.set_xlabel("1-Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Cross-validation ROC")
    ax.legend(fontsize=6, loc="lower right")
    self.is_available = True


class CvCalibrationPlot(BasePlot):
  """Reliability curve on the out-of-fold predictions of the best descriptor."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(4.5, 3))
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(4.5, 3))
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
    BasePlot.__init__(self, ax=ax, path=path)
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
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
    BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
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
    BasePlot.__init__(self, ax=ax, path=path)
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


class ProjectionCorrectnessPlot(BasePlot):
  """The MW-vs-LogP projection coloured by prediction correctness (TP / TN / FP / FN)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
    self.name = "projection-correctness"
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    projections = rf.get_projections()
    bt = rf.get_actives_inactives()
    bp = rf.get_pred_binary_clf()
    if not projections or bt is None or bp is None:
      return
    proj = projections[0]
    xs = np.array(proj["xs"], dtype=float)
    ys = np.array(proj["ys"], dtype=float)
    bt = np.asarray(bt, dtype=float)
    bp = np.asarray(bp, dtype=float)
    # Only labelled compounds can be scored TP/TN/FP/FN; drop unlabelled (NaN truth) rows, keeping
    # the projection coordinates aligned.
    keep = ~np.isnan(bt)
    xs, ys, bt, bp = xs[keep], ys[keep], bt[keep].astype(int), bp[keep].astype(int)
    if len(bt) == 0:
      return
    ax = self.ax
    cats = [
      ("TN", (bt == 0) & (bp == 0), named_colors.blue),
      ("TP", (bt == 1) & (bp == 1), named_colors.green),
      ("FP", (bt == 0) & (bp == 1), named_colors.purple),
      ("FN", (bt == 1) & (bp == 0), named_colors.red),
    ]
    for label, mask, color in cats:
      if mask.any():
        ax.scatter(xs[mask], ys[mask], color=color, s=12, alpha=0.7, edgecolors="none", label=label)
    ax.set_xlabel(proj["x_label"])
    ax.set_ylabel(proj["y_label"])
    ax.set_title("Errors in chemical space")
    ax.legend(fontsize=6, markerscale=1.2)
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


def _draw_property_distributions(axes, smiles, bt):
  """Draw MW / LogP / TPSA histograms (actives vs inactives) onto the three given axes."""
  bt = np.asarray(bt, dtype=int)
  props = list(zip(("Molecular weight", "LogP", "TPSA"), _rdkit_properties(smiles)))
  for axx, (label, vals) in zip(axes, props):
    act = vals[(bt == 1) & np.isfinite(vals)]
    ina = vals[(bt == 0) & np.isfinite(vals)]
    if len(act) == 0 and len(ina) == 0:
      continue
    lo = np.nanmin(vals[np.isfinite(vals)])
    hi = np.nanmax(vals[np.isfinite(vals)])
    bins = np.linspace(lo, hi, 25) if hi > lo else 10
    axx.hist(ina, bins=bins, color=named_colors.blue, alpha=0.5, density=True, label="Inactive")
    axx.hist(act, bins=bins, color=named_colors.red, alpha=0.5, density=True, label="Active")
    axx.set_xlabel(label)
    axx.set_ylabel("Density")
  if len(axes):
    axes[0].legend(fontsize=6)


class PropertyDistributionsPlot(BasePlot):
  """Physico-chemical property distributions (MW / LogP / TPSA) by class, as a 1×3 panel."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path)
    self.name = "property-distributions"
    self.is_available = False
    if not self.has_clf_data():
      return
    import matplotlib.pyplot as plt

    rf = ResultsFetcher(path=path)
    smiles = rf.get_smiles()
    bt = rf.get_actives_inactives()
    if not smiles or bt is None:
      return
    # Keep only labelled compounds (drop NaN truth) so the actives/inactives split is well-defined.
    keep = ~np.isnan(np.asarray(bt, dtype=float))
    smiles = [s for s, k in zip(smiles, keep) if k]
    bt = np.asarray(bt, dtype=float)[keep].astype(int)
    if len(bt) == 0:
      return
    # Standalone: replace the single BasePlot axis with a 1×3 panel of its own.
    plt.close()
    import stylia

    _, axes = stylia.create_figure(1, 3, width=1.0, height=0.33)
    _draw_property_distributions([axes[k] for k in range(3)], smiles, bt)
    self.is_available = True
