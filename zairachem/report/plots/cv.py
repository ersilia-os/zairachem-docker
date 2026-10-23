import numpy as np
from sklearn.metrics import (
  average_precision_score,
  balanced_accuracy_score,
  f1_score,
  matthews_corrcoef,
  precision_recall_curve,
  precision_score,
  recall_score,
  roc_curve,
)

from zairachem.report import BasePlot
from zairachem.report.colors import (
  descriptor_colors_rgb,
  named_colors,
)
from zairachem.report.cv import CHANCE, skill_keys, task_of
from zairachem.report.fetcher import ResultsFetcher


def _descriptor_color_map(stats):
  """``{descriptor: rgb}`` identity colours by best-first rank (``stats`` from ``get_cv_stats``)."""
  palette = descriptor_colors_rgb(len(stats))
  return {s["descriptor"]: palette[i] for i, s in enumerate(stats)}


class _CvSkillBarPlot(BasePlot):
  """Per-descriptor cross-validation (OOF) skill bars, with the in-sample skill marked for reference.

  Each bar carries the descriptor's identity colour (matching its line in the other CV figures). The
  gap between the bar (OOF) and the │ marker (train) is the per-descriptor overfitting. Subclasses
  set ``stem``, ``task``, ``xlabel`` and ``xlim``; the skill is AUROC for classification and
  Spearman's rho for regression, and the plot renders only for the task whose reports it finds.
  """

  stem = None
  task = None
  xlabel = None
  xlim = (0.4, 1.03)

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = self.stem
    self.is_available = False
    all_stats = ResultsFetcher(path=path).get_cv_stats()
    if task_of(all_stats) != self.task:
      return  # the other task's reports; its own bar plot will draw them
    oof_key, train_key, _label = skill_keys(all_stats)
    stats_bf = [s for s in all_stats if s.get(oof_key) is not None]
    if not stats_bf:
      return
    ax = self.ax
    color_of = _descriptor_color_map(stats_bf)
    stats = list(reversed(stats_bf))  # ascending → best on top
    labels = [s["descriptor"] for s in stats]
    oof = [s[oof_key] for s in stats]
    lo, hi = self.xlim
    # The floor is where a useful descriptor starts, not a clip: one that scores below it is a
    # finding, and the axis grows to show how far below.
    lo = min(lo, min(oof) - 0.05)
    y = list(range(len(labels)))
    ax.barh(
      y, oof, color=[color_of[s["descriptor"]] for s in stats], alpha=0.9, height=0.6, zorder=2
    )
    for i, s in enumerate(stats):
      tr = s.get(train_key)
      if tr is not None:
        ax.plot([tr], [i], marker="|", markersize=14, color=named_colors.black, zorder=3)
      ax.text(
        lo + (hi - lo) * 0.01, i, f"{oof[i]:.2f}", va="center", ha="left", fontsize=7, zorder=4
      )
    ax.axvline(CHANCE[oof_key], color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(self.xlabel)
    ax.set_title(f"Cross-validation {self.xlabel} (bar) vs train (│)")
    self.is_available = True


class CvAurocPlot(_CvSkillBarPlot):
  """Per-descriptor OOF AUROC bars, with the train AUROC marked (classification only)."""

  stem = "cv-auroc"
  task = "classification"
  xlabel = "AUROC"
  xlim = (0.4, 1.03)


class CvSpearmanPlot(_CvSkillBarPlot):
  """Per-descriptor OOF Spearman rho bars, with the train rho marked (regression only)."""

  stem = "cv-spearman"
  task = "regression"
  xlabel = "Spearman \u03c1"
  # rho runs from 0 (chance) to 1; a negative one means the descriptor orders compounds backwards.
  xlim = (-0.05, 1.03)


class _CvBarPlot(BasePlot):
  """Base for a per-descriptor horizontal-bar metric figure over the lazy-qsar OOF predictions.

  Every bar plot shares one row order — descriptors best-first by inner-CV AUROC, best on top — and
  the per-descriptor identity colour, so flipping between metrics keeps each descriptor in the same
  place with the same hue. Subclasses set ``stem``/``title``/``xlabel``/``xlim``/``noskill`` and
  implement :meth:`compute` (returns the metric for one descriptor, or ``None`` to skip it).
  Threshold metrics receive ``preds`` already thresholded at the descriptor's balanced cutoff.
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
    """Return the average precision of the descriptor's OOF probabilities."""
    return average_precision_score(y, proba)


class CvMccPlot(_CvBarPlot):
  """Per-descriptor Matthews correlation coefficient at each descriptor's balanced cutoff."""

  stem = "cv-mcc"
  title = "MCC at the balanced cutoff"
  xlabel = "MCC (no skill = 0)"

  def compute(self, s, proba, y, preds, prior):
    """Return the MCC of the descriptor's OOF calls at its cutoff, or None without a cutoff."""
    return matthews_corrcoef(y, preds) if preds is not None else None


class CvF1Plot(_CvBarPlot):
  """Per-descriptor F1 score at each descriptor's balanced cutoff."""

  stem = "cv-f1"
  title = "F1 at the balanced cutoff"
  xlabel = "F1"

  def compute(self, s, proba, y, preds, prior):
    """Return the F1 of the descriptor's OOF calls at its cutoff, or None without a cutoff."""
    return f1_score(y, preds, zero_division=0) if preds is not None else None


class CvBalancedAccuracyPlot(_CvBarPlot):
  """Per-descriptor balanced accuracy at each descriptor's balanced cutoff (no skill = 0.5)."""

  stem = "cv-balacc"
  title = "Balanced accuracy at the balanced cutoff vs no skill (--)"
  xlabel = "Balanced accuracy"
  noskill = 0.5

  def compute(self, s, proba, y, preds, prior):
    """Return the balanced accuracy of the OOF calls at the cutoff, or None without a cutoff."""
    return balanced_accuracy_score(y, preds) if preds is not None else None


class CvPrecisionPlot(_CvBarPlot):
  """Per-descriptor precision at each descriptor's balanced cutoff, with the no-skill prior marked."""

  stem = "cv-precision"
  title = "Precision at the balanced cutoff vs no-skill prior (--)"
  xlabel = "Precision"
  noskill = "prior"

  def compute(self, s, proba, y, preds, prior):
    """Return the precision of the descriptor's OOF calls at its cutoff, or None without one."""
    return precision_score(y, preds, zero_division=0) if preds is not None else None


class CvRecallPlot(_CvBarPlot):
  """Per-descriptor recall (sensitivity) at each descriptor's balanced cutoff."""

  stem = "cv-recall"
  title = "Recall at the balanced cutoff"
  xlabel = "Recall (sensitivity)"

  def compute(self, s, proba, y, preds, prior):
    """Return the recall of the descriptor's OOF calls at its cutoff, or None without a cutoff."""
    return recall_score(y, preds, zero_division=0) if preds is not None else None


class CvCutoffPlot(_CvBarPlot):
  """Per-descriptor ideal decision cutoff (the operating probability lazy-qsar selected)."""

  stem = "cv-cutoff"
  title = "Balanced cutoff (probability) vs the default 0.5 (--)"
  xlabel = "Decision cutoff (probability)"
  noskill = 0.5

  def compute(self, s, proba, y, preds, prior):
    """Return the descriptor's ideal decision cutoff from its CV report, or None."""
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


class CvErrorCurvePlot(BasePlot):
  """Per-descriptor out-of-fold error curves: what fraction of compounds land within an error.

  The regression counterpart of the cross-validation ROC and PR curves, and read the same way --
  up and to the left is better, and the spread between the lines is how much the choice of
  descriptor matters. Built from each descriptor's own out-of-fold predictions, so a descriptor
  that memorised its training rows cannot flatter itself here. The dashed line is the error you
  would get by predicting the training mean for everything.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "cv-error-curve"
    self.is_available = False
    rf = ResultsFetcher(path=path)
    stats = rf.get_cv_stats()
    if task_of(stats) != "regression":
      return  # a classification run: its ROC and PR curves say this instead
    color_of = _descriptor_color_map(stats)
    curves, baseline = [], None
    for s in stats:
      oof = rf.get_cv_oof_values(s["descriptor"])
      if oof is None:
        continue
      pred = np.asarray(oof[0], dtype=float)
      truth = np.asarray(oof[1], dtype=float)
      keep = np.isfinite(pred) & np.isfinite(truth)
      if keep.sum() < 2:
        continue
      err = np.sort(np.abs(truth[keep] - pred[keep]))
      curves.append((s["descriptor"], err))
      if baseline is None:
        baseline = np.sort(np.abs(truth[keep] - truth[keep].mean()))
    if not curves:
      return
    ax = self.ax
    for name, err in curves:
      frac = np.arange(1, err.size + 1) / err.size
      ax.plot(err, frac, color=color_of[name], lw=1.6, alpha=0.9, label=name, zorder=3)
    if baseline is not None and baseline.size:
      ax.plot(
        baseline,
        np.arange(1, baseline.size + 1) / baseline.size,
        color=named_colors.gray,
        lw=1,
        ls="--",
        zorder=1,
        label="train mean",
      )
    # Clip to where the curves are informative; the long tail is a handful of bad predictions.
    ax.set_xlim(0, float(max(np.percentile(e, 98) for _n, e in curves)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Absolute out-of-fold error")
    ax.set_ylabel("Fraction of compounds within")
    ax.set_title("Cross-validation error curve")
    # A default run keeps up to ten descriptors; one column of eleven entries covers the curves.
    ax.legend(loc="lower right", fontsize=6, ncol=2 if len(curves) > 4 else 1)
    self.is_available = True
