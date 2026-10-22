import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
  auc,
  average_precision_score,
  f1_score,
  matthews_corrcoef,
  precision_recall_curve,
  roc_curve,
)

from zairachem.base.vars import (
  DEFAULT_CUTOFF,
  REPORT_SUBFOLDER,
  VALIDATION_PREDICTIONS_FILENAME,
  VALIDATION_TABLE_FILENAME,
)
from zairachem.report import BasePlot
from zairachem.report.colors import (
  category_palette,
  named_colors,
  rgb as _color,
)

# Held-out split strategies: canonical order + display labels, shared by every held-out plot so a
# strategy is colored/labelled identically everywhere (the jitter plot AND the per-strategy curves).
_HELDOUT_SCHEMAS = [
  ("random", "Random"),
  ("scaffold", "Scaffold"),
  ("scaffold_det", "Deterministic"),
  ("butina", "Butina"),
]


def _heldout_color_map():
  """``{strategy_key: color}`` — a strategy keeps the same color across all held-out figures."""
  cols = category_palette.get(len(_HELDOUT_SCHEMAS))
  return {k: cols[i] for i, (k, _lbl) in enumerate(_HELDOUT_SCHEMAS)}


def _heldout_predictions(path, per_fold=False):
  """Held-out ``(y_true, y_score)`` from ``report/validation_predictions.csv``, in canonical order.

  Default: one entry per strategy with its folds POOLED (concatenated) → ``(key, label, y_true,
  y_score)``. ``per_fold=True`` → ``(key, label, fold, y_true, y_score)`` per fold. ``[]`` if the file
  is absent/empty (e.g. a run without ``--evaluate``, or before the predictions were persisted).
  """
  csv_path = os.path.join(path, REPORT_SUBFOLDER, VALIDATION_PREDICTIONS_FILENAME)
  if not os.path.exists(csv_path):
    return []
  df = pd.read_csv(csv_path)
  if df.empty or not {"strategy", "y_true", "y_score"}.issubset(df.columns):
    return []
  known = [k for k, _ in _HELDOUT_SCHEMAS]
  order = [k for k in known if k in set(df["strategy"])]
  order += sorted(set(df["strategy"]) - set(known))
  labels = dict(_HELDOUT_SCHEMAS)
  out = []
  for k in order:
    sub = df[df["strategy"] == k]
    if sub.empty:
      continue
    lbl = labels.get(k, k)
    if per_fold:
      for fold, g in sub.groupby("fold", sort=False):
        out.append((
          k,
          lbl,
          fold,
          g["y_true"].to_numpy(dtype=float),
          g["y_score"].to_numpy(dtype=float),
        ))
    else:
      out.append((
        k,
        lbl,
        sub["y_true"].to_numpy(dtype=float),
        sub["y_score"].to_numpy(dtype=float),
      ))
  return out


class HeldOutValidationPlot(BasePlot):
  """Held-out AUROC per split schema, one point per fold (classification, ``--evaluate`` only).

  Random is the optimism anchor (dashed reference line); a drop under scaffold / Butina indicates
  weaker generalization to novel chemistry. Rendered only when ``--evaluate`` produced
  ``report/validation_table.csv``; otherwise the plot marks itself unavailable and is skipped.
  """

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "heldout-validation"
    self.is_available = False
    csv_path = os.path.join(path, REPORT_SUBFOLDER, VALIDATION_TABLE_FILENAME)
    if not os.path.exists(csv_path):
      return
    df = pd.read_csv(csv_path)
    if df.empty or "auroc" not in df.columns or "strategy" not in df.columns:
      return
    seen = set(df["strategy"])
    present = [(k, lbl) for k, lbl in _HELDOUT_SCHEMAS if k in seen]
    present += [(k, k) for k in sorted(seen) if k not in {p[0] for p in _HELDOUT_SCHEMAS}]
    if not present:
      return
    ax = self.ax
    cmap = _heldout_color_map()
    random_mean = None
    xticks, xticklabels = [], []
    for i, (strat, lbl) in enumerate(present):
      vals = np.asarray(df[df["strategy"] == strat]["auroc"], dtype=float)
      vals = vals[~np.isnan(vals)]
      if len(vals) == 0:
        continue
      jitter = np.random.uniform(-0.12, 0.12, len(vals))
      ax.scatter(
        np.full(len(vals), i) + jitter, vals, color=cmap.get(strat), alpha=0.7, s=25, zorder=3
      )
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


# --- Held-out validation, per split strategy (pooled across each strategy's folds) ------------------
#
# All read report/validation_predictions.csv via _heldout_predictions(); one line/bar per split
# strategy, colored by _heldout_color_map() so colors match the jitter plot. Training-only (registered
# in report.py _FIT_ONLY). Unavailable (skipped) when --evaluate wasn't run.


class HeldoutRocByStrategyPlot(BasePlot):
  """Held-out ROC per split strategy (folds pooled); AUROC per strategy in the legend."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "heldout-roc"
    self.is_available = False
    curves = [
      (k, lbl, yt, ys) for (k, lbl, yt, ys) in _heldout_predictions(path) if len(np.unique(yt)) >= 2
    ]
    if not curves:
      return
    ax = self.ax
    cmap = _heldout_color_map()
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--", zorder=1)
    for k, lbl, yt, ys in curves:
      fpr, tpr, _ = roc_curve(yt, ys)
      ax.plot(
        fpr,
        tpr,
        color=cmap.get(k),
        lw=1.6,
        alpha=0.9,
        zorder=2,
        label=f"{lbl} ({auc(fpr, tpr):.2f})",
      )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1-Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Held-out ROC by split")
    ax.legend(loc="lower right", fontsize=6)
    self.is_available = True


class HeldoutPrByStrategyPlot(BasePlot):
  """Held-out precision-recall per split strategy (folds pooled); AUPR in the legend, prevalence line."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "heldout-pr"
    self.is_available = False
    curves = [
      (k, lbl, yt, ys) for (k, lbl, yt, ys) in _heldout_predictions(path) if len(np.unique(yt)) >= 2
    ]
    if not curves:
      return
    ax = self.ax
    cmap = _heldout_color_map()
    allyt = np.concatenate([yt for _, _, yt, _ in curves])
    ax.axhline(float(allyt.mean()), color=named_colors.gray, lw=1, ls="--", zorder=1)
    for k, lbl, yt, ys in curves:
      prec, rec, _ = precision_recall_curve(yt, ys)
      ax.plot(
        rec,
        prec,
        color=cmap.get(k),
        lw=1.6,
        alpha=0.9,
        zorder=2,
        label=f"{lbl} ({average_precision_score(yt, ys):.2f})",
      )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-out PR by split")
    ax.legend(loc="lower left", fontsize=6)
    self.is_available = True


class HeldoutCalibrationByStrategyPlot(BasePlot):
  """Held-out reliability curve per split strategy (folds pooled): mean predicted vs observed per decile."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "heldout-calibration"
    self.is_available = False
    curves = [
      (k, lbl, yt, ys) for (k, lbl, yt, ys) in _heldout_predictions(path) if len(np.unique(yt)) >= 2
    ]
    if not curves:
      return
    ax = self.ax
    cmap = _heldout_color_map()
    bins = np.linspace(0, 1, 11)
    ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1, ls="--", zorder=1)
    for k, lbl, yt, ys in curves:
      idx = np.clip(np.digitize(ys, bins) - 1, 0, 9)
      xs, obs = [], []
      for b in range(10):
        m = idx == b
        if m.sum() > 0:
          xs.append(float(ys[m].mean()))
          obs.append(float(yt[m].mean()))
      ax.plot(xs, obs, color=cmap.get(k), lw=1.4, marker="o", ms=3, alpha=0.9, zorder=2, label=lbl)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of actives")
    ax.set_title("Held-out calibration by split")
    ax.legend(loc="upper left", fontsize=6)
    self.is_available = True


class HeldoutEnrichmentFactorByStrategyPlot(BasePlot):
  """Held-out enrichment factor (hit-rate / prevalence) vs fraction screened, per split strategy (pooled)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "heldout-enrichment-factor"
    self.is_available = False
    curves = [
      (k, lbl, yt, ys)
      for (k, lbl, yt, ys) in _heldout_predictions(path)
      if yt.sum() >= 1 and 0.0 < yt.mean() < 1.0
    ]
    if not curves:
      return
    ax = self.ax
    cmap = _heldout_color_map()
    ax.axhline(1.0, color=named_colors.gray, lw=1, ls="--", zorder=1)
    for k, lbl, yt, ys in curves:
      n = len(yt)
      yts = yt[np.argsort(-ys)]
      ks = np.arange(1, n + 1)
      ef = (np.cumsum(yts) / ks) / float(yt.mean())
      frac = ks / n
      sel = ks >= max(1, int(round(0.02 * n)))
      ax.plot(frac[sel], ef[sel], color=cmap.get(k), lw=1.6, alpha=0.9, zorder=2, label=lbl)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fraction of library screened")
    ax.set_ylabel("Enrichment factor")
    ax.set_title("Held-out enrichment by split")
    ax.legend(loc="upper right", fontsize=6)
    self.is_available = True


class HeldoutMetricBarsPlot(BasePlot):
  """Held-out AUROC / AUPR / MCC / F1 per split strategy (mean ± std across folds).

  MCC and F1 use the default cutoff (0.5) on the calibrated held-out score."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "heldout-metric-bars"
    self.is_available = False
    per = _heldout_predictions(path, per_fold=True)
    if not per:
      return
    from collections import defaultdict

    agg = defaultdict(lambda: {"auroc": [], "aupr": [], "mcc": [], "f1": []})
    label_of = {}
    for k, lbl, _fold, yt, ys in per:
      label_of[k] = lbl
      if len(np.unique(yt)) < 2:
        continue
      pred = (ys >= DEFAULT_CUTOFF).astype(int)
      fpr, tpr, _ = roc_curve(yt, ys)
      agg[k]["auroc"].append(auc(fpr, tpr))
      agg[k]["aupr"].append(average_precision_score(yt, ys))
      agg[k]["mcc"].append(matthews_corrcoef(yt, pred) if len(np.unique(pred)) > 1 else 0.0)
      agg[k]["f1"].append(f1_score(yt, pred, zero_division=0))
    known = [k for k, _ in _HELDOUT_SCHEMAS]
    order = [k for k in known if k in agg] + [k for k in agg if k not in known]
    if not order:
      return
    ax = self.ax
    cmap = _heldout_color_map()
    metrics = ["auroc", "aupr", "mcc", "f1"]
    mlabels = ["AUROC", "AUPR", "MCC", "F1"]
    x = np.arange(len(metrics))
    nS = len(order)
    width = 0.8 / nS
    for si, k in enumerate(order):
      means = [float(np.mean(agg[k][m])) if agg[k][m] else 0.0 for m in metrics]
      stds = [float(np.std(agg[k][m])) if len(agg[k][m]) > 1 else 0.0 for m in metrics]
      ax.bar(
        x - 0.4 + width / 2 + si * width,
        means,
        width,
        yerr=stds,
        color=cmap.get(k),
        alpha=0.85,
        label=label_of.get(k, k),
        error_kw={"lw": 0.8},
      )
    ax.set_xticks(x)
    ax.set_xticklabels(mlabels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out metrics by split · default cutoff")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    self.is_available = True


class HeldoutConfusionByStrategyPlot(BasePlot):
  """Held-out outcome composition per split strategy: fraction TP / FN / TN / FP at the default cutoff (0.5)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 3))
    self.name = "heldout-confusion"
    self.is_available = False
    rows = [
      (k, lbl, yt, (ys >= DEFAULT_CUTOFF).astype(int))
      for (k, lbl, yt, ys) in _heldout_predictions(path)
      if len(yt)
    ]
    if not rows:
      return
    from matplotlib.patches import Patch

    ax = self.ax
    segs = [
      ("correct_positive", "TP", lambda yt, yp: int(np.sum((yt == 1) & (yp == 1)))),
      ("false_negative", "FN", lambda yt, yp: int(np.sum((yt == 1) & (yp == 0)))),
      ("correct_negative", "TN", lambda yt, yp: int(np.sum((yt == 0) & (yp == 0)))),
      ("false_positive", "FP", lambda yt, yp: int(np.sum((yt == 0) & (yp == 1)))),
    ]
    labels = []
    for yi, (_k, lbl, yt, yp) in enumerate(rows):
      n = len(yt)
      left = 0.0
      for ckey, _nm, fn in segs:
        frac = (fn(yt, yp) / n) if n else 0.0
        if frac > 0:
          ax.barh(yi, frac, left=left, color=_color(ckey), edgecolor="white", height=0.65, zorder=2)
          left += frac
      labels.append(lbl)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of held-out compounds")
    ax.set_title("Held-out outcome composition by split · default cutoff")
    handles = [Patch(color=_color(ck), label=nm) for ck, nm, _ in segs]
    ax.legend(handles=handles, loc="lower right", fontsize=6, ncol=2)
    self.is_available = True
