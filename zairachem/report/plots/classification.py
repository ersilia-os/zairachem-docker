import numpy as np
from sklearn import metrics
from sklearn.metrics import (
  auc,
  average_precision_score,
  f1_score,
  matthews_corrcoef,
  precision_recall_curve,
  precision_score,
  recall_score,
  roc_curve,
)

from zairachem.pool.cutoffs import call_cutoff
from zairachem.report import BasePlot
from zairachem.report.colors import (
  named_colors,
  rgb as _color,
)
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report.plots._common import (
  _npg,
  _npg_cmap,
  _training_score_by_class,
)


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
        disp.plot(ax=ax, cmap=_npg_cmap("#6BBF59"), colorbar=False)
        # for labels in disp.text_.ravel():
        # labels.set_fontsize(22)
        ax.grid(False)
        ax.set_title(f"Confusion matrix · {call_cutoff(path)[0]} cutoff")
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
      color = _npg(1)[0]
      predict = self.is_predict()
      ax.plot(fpr, tpr, color=color, zorder=10000, lw=1.6, label="Test" if predict else None)
      ax.fill_between(fpr, tpr, color=color, alpha=0.16, lw=0, zorder=1000)
      title = "AUROC = {0}".format(round(auroc, 2))
      if predict:  # overlay the trained-OOF curve so the generalization gap is visible
        act_tr, ina_tr = _training_score_by_class(path)
        if act_tr.size and ina_tr.size:
          yt_tr = np.r_[np.ones(act_tr.size), np.zeros(ina_tr.size)]
          yp_tr = np.r_[act_tr, ina_tr]
          fpr_t, tpr_t, _ = roc_curve(yt_tr, yp_tr)
          ax.plot(
            fpr_t, tpr_t, color=named_colors.gray, lw=1.4, ls=":", zorder=9000, label="Training OOF"
          )
          title = f"AUROC · test {auroc:.2f} · train {auc(fpr_t, tpr_t):.2f}"
          ax.legend(loc="lower right", fontsize=6)
      ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--")
      ax.set_title(title)
      ax.set_xlabel("1-Specificity (FPR)")
      ax.set_ylabel("Sensitivity (TPR)")
      ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
      ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
      self.is_available = False


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
    c = _npg(1)[0]
    predict = self.is_predict()
    ax.plot(recall, precision, color=c, lw=1.6, zorder=1000, label="Test" if predict else None)
    ax.fill_between(recall, precision, color=c, alpha=0.16, lw=0)
    title = f"AUPR = {ap:.2f}"
    if predict:  # overlay the trained-OOF PR curve as a reference
      act_tr, ina_tr = _training_score_by_class(path)
      if act_tr.size and ina_tr.size:
        yt_tr = np.r_[np.ones(act_tr.size), np.zeros(ina_tr.size)]
        yp_tr = np.r_[act_tr, ina_tr]
        p_t, r_t, _ = precision_recall_curve(yt_tr, yp_tr)
        ax.plot(r_t, p_t, color=named_colors.gray, lw=1.4, ls=":", zorder=900, label="Training OOF")
        title = f"AUPR · test {ap:.2f} · train {average_precision_score(yt_tr, yp_tr):.2f}"
        ax.legend(loc="upper right", fontsize=6)
    ax.axhline(prevalence, color=named_colors.gray, lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
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
    ax.plot(frac_screened, ideal, color=_npg(3)[2], lw=1, ls=":", label="Ideal")
    ax.plot(frac_screened, frac_found, color=_npg(1)[0], lw=1.6, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Fraction of library screened")
    ax.set_ylabel("Fraction of actives found")
    ax.set_title("Cumulative gain")
    ax.legend(loc="lower right", fontsize=6)
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
    pal = _npg(4)
    ax.plot(ts, prec, color=pal[0], lw=1.4, label="Precision")
    ax.plot(ts, rec, color=pal[1], lw=1.4, label="Recall")
    ax.plot(ts, f1s, color=pal[2], lw=1.4, label="F1")
    ax.plot(ts, mccs, color=pal[3], lw=1.4, label="MCC")
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
    ax.legend(loc="lower center", ncol=2, fontsize=6)
    self.is_available = True


class CalibrationCurvePlot(BasePlot):
  """Reliability curve of the pooled out-of-fold probability: mean predicted probability vs the
  observed frequency of actives per decile bin, against the diagonal (perfect calibration). The title
  reports the Brier score (lower = better-calibrated)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "calibration-curve"
    self.is_available = False
    if not self.has_clf_data():
      return
    yt, yp = _clf_truth_proba(path)
    if yt is None or len(np.unique(yt)) < 2:
      return
    ax = self.ax
    p = np.asarray(yp, dtype=float)
    y = np.asarray(yt, dtype=float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
      m = idx == b
      if m.sum() > 0:
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
    brier = float(np.mean((p - y) ** 2))
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--", zorder=1)
    ax.plot(xs, ys, color=_npg(1)[0], lw=1.6, marker="o", ms=4, zorder=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of actives")
    ax.set_title(f"Calibration · Brier = {brier:.3f}")
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
    ax.imshow(norm, cmap=_npg_cmap("#6BBF59"), vmin=0, vmax=1)
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
    ax.set_title(f"Confusion at the {call_cutoff(path)[0]} cutoff (row-normalized · recall)")
    self.is_available = True


class ConfusionPrecisionPlot(BasePlot):
  """Column-normalized confusion matrix (per-predicted-class precision %) — the complement of the
  row-normalized recall view."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "confusion-precision"
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    bt, bp = rf.clf_truth_binary()
    if bt is None or bp is None or len(bt) == 0 or len(set(bp.tolist())) < 2:
      return
    ax = self.ax
    cm = metrics.confusion_matrix(bt, bp).astype(float)
    col_sums = cm.sum(axis=0, keepdims=True)
    norm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0)
    ax.imshow(norm, cmap=_npg_cmap("#6BBF59"), vmin=0, vmax=1)
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
    ax.set_title(f"Confusion at the {call_cutoff(path)[0]} cutoff (column-normalized · precision)")
    self.is_available = True


class ConfusionBreakdownPlot(BasePlot):
  """Outcome composition: each true class as a stacked bar split into correct vs error calls
  (Actives → TP / FN; Inactives → TN / FP), a more intuitive read than the matrix."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "confusion-breakdown"
    self.is_available = False
    if not self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    bt, bp = rf.clf_truth_binary()
    if bt is None or bp is None or len(bt) == 0 or len(set(bp.tolist())) < 2:
      return
    ax = self.ax
    cm = metrics.confusion_matrix(bt, bp)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    def seg(y, x0, w, color, txt):
      if w <= 0:
        return
      ax.barh(y, w, left=x0, color=color, edgecolor="white", height=0.62, zorder=2)
      ax.text(x0 + w / 2, y, txt, ha="center", va="center", color="white", fontweight="bold")

    # Active row (top): correct TP then error FN; Inactive row (bottom): correct TN then error FP.
    seg(1, 0, tp, _color("correct_positive"), f"TP {tp}")
    seg(1, tp, fn, _color("false_negative"), f"FN {fn}")
    seg(0, 0, tn, _color("correct_negative"), f"TN {tn}")
    seg(0, tn, fp, _color("false_positive"), f"FP {fp}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Inactive", "Active"])
    ax.set_xlabel("Number of compounds")
    ax.set_title(f"Outcome composition by true class · {call_cutoff(path)[0]} cutoff")
    self.is_available = True


class DescriptorCorrelationPlot(BasePlot):
  """Spearman correlation between the per-descriptor out-of-fold predictions — how redundant vs
  complementary the descriptor models are (high = agree, lower = adds a distinct signal).

  Correlating predictions with each other says nothing about what they predict, so this reads the
  same for a probability and for a value; both tasks get it.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "descriptor-correlation"
    self.is_available = False
    rf = ResultsFetcher(path=path)
    tasks = rf.get_clf_tasks() or rf.get_reg_tasks()
    if not tasks:
      return
    df = rf._read_individual_estimator_results(tasks[0])
    if df is None or df.shape[1] < 2:
      return
    # Order descriptors best-first (by the task's inner-CV skill) when available, for a consistent
    # reading across the report.
    ranked = [s["descriptor"] for s in rf.get_cv_stats()]
    cols = sorted(
      df.columns,
      key=lambda c: ranked.index(c.split("-")[-1]) if c.split("-")[-1] in ranked else 999,
    )
    corr = df[cols].corr(method="spearman")
    m = corr.values
    labels = [c.split("-")[-1] for c in cols]
    ax = self.ax
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("stylia_seq", ["#ffffff", _color("inactive")])
    vmin = float(np.nanmin(m[~np.eye(len(m), dtype=bool)])) if len(m) > 1 else 0.0
    ax.imshow(m, cmap=cmap, vmin=max(0.0, vmin - 0.02), vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(len(labels)):
      for j in range(len(labels)):
        ax.text(
          j,
          i,
          f"{m[i, j]:.2f}",
          va="center",
          ha="center",
          fontsize=6,
          color="white" if m[i, j] > 0.85 else "black",
        )
    ax.grid(False)
    ax.set_title("Descriptor prediction correlation (Spearman)")
    self.is_available = True


class TopKOverlapCurvePlot(BasePlot):
  """Do the descriptor models agree on their TOP compounds? For each top fraction x, the mean pairwise
  overlap (shared fraction) of the descriptors' top-x% ranked compounds, plotted vs the chance line
  ``y = x`` (the overlap two random top-x% sets share). Well above chance ⇒ the descriptors converge
  on the same hits; near chance ⇒ they disagree on what ranks highest. Complements the (global)
  Spearman correlation heatmap by focusing on the ranking head, which is what matters for
  screening/triage."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "topk-overlap-curve"
    self.is_available = False
    rf = ResultsFetcher(path=path)
    tasks = rf.get_clf_tasks() or rf.get_reg_tasks()
    if not tasks:
      return
    df = rf._read_individual_estimator_results(tasks[0])
    if df is None or df.shape[1] < 2:
      return
    scores = [np.nan_to_num(df[c].to_numpy(dtype=float), nan=-np.inf) for c in df.columns]
    orders = [np.argsort(-s) for s in scores]
    n = len(orders[0])
    if n < 5:
      return
    # ~40 log-spaced top-fractions, from ~0.5% of the set (floored to avoid single-compound noise)
    # up to the whole library.
    ks = np.unique(
      np.clip(np.geomspace(max(1, int(round(0.005 * n))), n, 40).round().astype(int), 1, n)
    )
    xs, means, los, his = [], [], [], []
    npd = len(orders)
    for k in ks:
      sets = [set(o[:k].tolist()) for o in orders]
      ov = [len(sets[i] & sets[j]) / k for i in range(npd) for j in range(i + 1, npd)]
      ov = np.asarray(ov, dtype=float)
      xs.append(k / n)
      means.append(float(ov.mean()))
      los.append(float(np.percentile(ov, 10)))
      his.append(float(np.percentile(ov, 90)))
    xs = np.asarray(xs)
    ax = self.ax
    c = _npg(1)[0]
    xx = np.geomspace(xs.min(), 1.0, 60)
    ax.plot(xx, xx, color=named_colors.gray, lw=1, ls="--", label="Chance")
    ax.fill_between(xs, los, his, color=c, alpha=0.16, lw=0)
    ax.plot(xs, means, color=c, lw=1.6, label="Mean pairwise")
    ax.set_xscale("log")
    ax.set_xlim(xs.min(), 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Top fraction (x)")
    ax.set_ylabel("Mean top-x% overlap")
    ax.set_title("Descriptor agreement on top-ranked compounds")
    ax.legend(loc="upper left", fontsize=6)
    self.is_available = True


class EnrichmentFactorCurvePlot(BasePlot):
  """Enrichment factor across ranking depth: EF(f) = hit-rate(top f) / prevalence, vs the fraction of
  the library screened. EF = 1 (dashed) is random; higher = better early enrichment. Log x-axis so the
  early, decision-relevant region is legible. A richer read than a fixed 1/5/10% barplot."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "enrichment-factor-curve"
    self.is_available = False
    if not self.has_clf_data():
      return
    yt, yp = _clf_truth_proba(path)
    if yt is None or yt.sum() < 1:
      return
    prevalence = float(yt.mean())
    if not (0.0 < prevalence < 1.0):
      return
    ax = self.ax
    n = len(yt)
    yt_sorted = yt[np.argsort(-yp)]
    ks = np.arange(1, n + 1)
    ef = (np.cumsum(yt_sorted) / ks) / prevalence
    frac = ks / n
    # Floor the plotted range at ~0.5% so the volatile single-compound head doesn't dominate the axis.
    kmin = max(1, int(round(0.005 * n)))
    sel = ks >= kmin
    c = _npg(1)[0]
    ax.plot(frac[sel], ef[sel], color=c, lw=1.6, label="Model")
    ax.axhline(1.0, color=named_colors.gray, lw=1, ls="--", label="Random")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fraction of library screened")
    ax.set_ylabel("Enrichment factor")
    ax.set_title(f"Enrichment factor · max {float(ef[sel].max()):.1f}×")
    ax.legend(loc="lower left", fontsize=6)
    self.is_available = True


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
