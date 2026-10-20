import numpy as np

from zairachem.base.vars import RANK_DECISION_CUTOFF
from zairachem.report import BasePlot
from zairachem.report.colors import (
  named_colors,
  rgb as _color,
)
from zairachem.report.fetcher import ResultsFetcher
from zairachem.report.plots._common import (
  _draw_one_property,
  _draw_training_projection_background,
  _npg,
  _npg_cmap,
  _rdkit_properties,
  _subsample,
  _training_score_by_class,
)

# --- Predict-only (truth-free) plots -------------------------------------------------------------
# These describe the PREDICTED set from the pooled probability alone — no ground truth needed — so
# they are the core of the predict report. Each renders only when there is NO truth column
# (``has_clf_data()`` is False). Since a fit run is always labelled, that condition holds *only* for a
# predict-without-ground-truth run, so these never fire at fit (fit report unchanged by construction).

_CUTOFF = 0.5  # pooled binarization threshold (pool.py: ``b_hat = y_hat > 0.5``)


def _predicted_proba(path):
  """Pooled predicted probabilities for the current run as a finite float array (or empty)."""
  yp = ResultsFetcher(path=path).get_pred_proba_clf()
  if yp is None:
    return np.array([])
  yp = np.asarray(yp, dtype=float)
  return yp[np.isfinite(yp)]


def _faded_density(ax, values, color, label=None, bins=24):
  """Draw a faint normalized density (filled step curve) of ``values`` over [0, 1] as a background
  reference — used to lay training-time score distributions behind predict-time histograms."""
  values = np.asarray(values, dtype=float)
  values = values[np.isfinite(values)]
  if values.size == 0:
    return
  h, edges = np.histogram(values, bins=bins, range=(0, 1), density=True)
  centers = (edges[:-1] + edges[1:]) / 2.0
  ax.fill_between(centers, h, color=color, alpha=0.12, lw=0, zorder=1)
  ax.plot(centers, h, color=color, alpha=0.4, lw=1.0, zorder=2, label=label)


def _predict_hist(ax, values, *, xlabel, title, xlim=None, cutoff=None, color=None, bins=(6, 20)):
  """Shared histogram styling for the predict-report distribution plots (score / AD / rank).

  ``bins`` is a ``(floor, cap)`` pair; the bin count scales with sample size between them."""
  floor, cap = bins
  kw = {"range": xlim} if xlim is not None else {}
  ax.hist(
    values,
    bins=min(cap, max(floor, values.size // 5)),
    color=color or _npg(1)[0],
    alpha=0.85,
    edgecolor="white",
    lw=0.4,
    **kw,
  )
  if cutoff is not None:
    ax.axvline(cutoff, color=named_colors.gray, lw=1, ls="--")
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("Compounds")
  if xlim is not None:
    ax.set_xlim(*xlim)


class PredictedScoreHistogramPlot(BasePlot):
  """Distribution of the pooled predicted probability across the scored set (predict report).

  Rendered for both predict scenarios (SMILES-only and labelled test set). The predicted scores are
  the foreground; the model's training-time score distributions (split by true class) are overlaid
  faded behind, as a reference for where actives vs inactives typically fall. When the predict set is
  itself labelled, the foreground is split by true class too."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "predicted-score-hist"
    self.is_available = False
    if not self.is_predict():
      return
    yp = _predicted_proba(path)
    if yp.size == 0:
      return
    ax = self.ax
    # Training reference (faded) on a twin axis so its density scale doesn't fight the count axis.
    bg = None
    act_tr, ina_tr = _training_score_by_class(path)
    if act_tr.size or ina_tr.size:
      bg = ax.twinx()
      bg.set_yticks([])
      bg.set_zorder(ax.get_zorder() - 1)
      ax.patch.set_visible(False)
      _faded_density(bg, ina_tr, _color("inactive"), label="Training inactive")
      _faded_density(bg, act_tr, _color("active"), label="Training active")
      bg.set_ylim(bottom=0)
    nbins = min(30, max(8, yp.size // 5))
    if self.has_clf_data():
      yt, yps = ResultsFetcher(path=path).clf_truth_proba()
      ax.hist(
        yps[yt == 0],
        bins=nbins,
        range=(0, 1),
        color=_color("inactive"),
        alpha=0.75,
        edgecolor="white",
        lw=0.4,
        label="Test inactive",
        zorder=10,
      )
      ax.hist(
        yps[yt == 1],
        bins=nbins,
        range=(0, 1),
        color=_color("active"),
        alpha=0.75,
        edgecolor="white",
        lw=0.4,
        label="Test active",
        zorder=11,
      )
      title = "Predicted scores by true class"
    else:
      ax.hist(
        yp,
        bins=nbins,
        range=(0, 1),
        color=_npg(1)[0],
        alpha=0.85,
        edgecolor="white",
        lw=0.4,
        label="Predicted",
        zorder=10,
      )
      n_pos = int((yp >= _CUTOFF).sum())
      title = f"Predicted scores · {n_pos}/{yp.size} ≥ {_CUTOFF:g}"
    ax.axvline(_CUTOFF, color=named_colors.gray, lw=1, ls="--", zorder=12)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Compounds")
    h, ll = ax.get_legend_handles_labels()
    if bg is not None:
      h2, l2 = bg.get_legend_handles_labels()
      h, ll = h + h2, ll + l2
    ax.legend(h, ll, loc="upper center", fontsize=6)
    self.is_available = True


class ScoreRankCurvePlot(BasePlot):
  """Compounds ranked best-first by predicted probability — the screening operating-point view."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "predicted-rank-curve"
    self.is_available = False
    if not self.is_predict():
      return
    yp = _predicted_proba(path)
    if yp.size == 0:
      return
    order = np.argsort(yp)[::-1]  # best-first
    ys = yp[order]
    xs = np.arange(1, ys.size + 1) / ys.size * 100.0  # % of library screened, best-first
    ax = self.ax
    ax.plot(xs, ys, color=_npg(1)[0], lw=1.8, zorder=10)
    ax.fill_between(xs, ys, color=_npg(1)[0], alpha=0.16, lw=0)
    if self.has_clf_data():  # labelled test set → colour each compound by its true class
      yt_all = np.asarray(ResultsFetcher(path=path).get_actives_inactives(), dtype=float)
      if yt_all.size == yp.size:
        yt_sorted = yt_all[order]
        ina = np.isfinite(yt_sorted) & (yt_sorted == 0)
        act = np.isfinite(yt_sorted) & (yt_sorted == 1)
        ax.scatter(
          xs[ina],
          ys[ina],
          color=_color("inactive"),
          s=10,
          alpha=0.6,
          edgecolors="none",
          zorder=11,
          label="Inactive",
        )
        ax.scatter(
          xs[act],
          ys[act],
          color=_color("active"),
          s=12,
          alpha=0.85,
          edgecolors="none",
          zorder=12,
          label="Active",
        )
        ax.legend(loc="upper right", fontsize=6)
    ax.axhline(_CUTOFF, color=named_colors.gray, lw=1, ls="--")
    frac = float((yp >= _CUTOFF).mean() * 100.0)
    ax.set_title(f"{frac:.0f}% of the set scores ≥ {_CUTOFF:g}")
    ax.set_xlabel("Top % of library (ranked by score)")
    ax.set_ylabel("Predicted probability")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    self.is_available = True


def _projection_xy_by_proba(path, projection):
  """``(x, y, proba)`` for a projection keeping finite-coordinate rows, aligned with pooled proba."""
  yp = np.asarray(ResultsFetcher(path=path).get_pred_proba_clf() or [], dtype=float)
  xs = np.asarray(projection["xs"], dtype=float)
  ys = np.asarray(projection["ys"], dtype=float)
  if yp.size != xs.size:  # alignment guard: proba and coords must be the same dedup rows
    return np.array([]), np.array([]), np.array([])
  keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(yp)
  return xs[keep], ys[keep], yp[keep]


class ProjectionProbaPlot(BasePlot):
  """A 2-D projection with each compound coloured by its predicted probability (no ground truth)."""

  def __init__(self, ax, path, projection):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = f"projection-{projection['name']}-proba"
    self.is_available = False
    if self.has_clf_data():
      return
    x, y, p = _projection_xy_by_proba(path, projection)
    if x.size == 0:
      return
    ax = self.ax
    # Backdrop of the trained model's chemical space (applicability domain).
    _draw_training_projection_background(ax, path, projection["name"])
    idx, alpha = _subsample(x.size)
    sc = ax.scatter(
      x[idx],
      y[idx],
      c=p[idx],
      cmap=_npg_cmap(_color("active")),
      vmin=0,
      vmax=1,
      s=10,
      alpha=alpha,
      edgecolors="none",
    )
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Predicted probability")
    ax.set_xlabel(projection["x_label"])
    ax.set_ylabel(projection["y_label"])
    ax.set_title(f"{projection['title']} · by score")
    self.is_available = True


class DescriptorConsensusPlot(BasePlot):
  """Per-descriptor agreement: pooled score vs the spread of the individual descriptor probabilities.

  A high pooled score with LOW descriptor spread is a confident, consensus call; high spread flags a
  prediction the descriptors disagree on. Truth-free (predict report)."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "descriptor-consensus"
    self.is_available = False
    if self.has_clf_data():
      return
    rf = ResultsFetcher(path=path)
    tasks = rf.get_clf_tasks()
    if not tasks:
      return
    try:
      D = rf._read_individual_estimator_results(tasks[0])
    except Exception:
      return
    if D is None or D.shape[1] < 2:
      return
    vals = D.values.astype(float)
    spread = np.nanstd(vals, axis=1)
    pooled = _predicted_proba(path)
    # Fall back to the per-descriptor mean if the pooled column length doesn't line up.
    if pooled.size != vals.shape[0]:
      pooled = np.nanmean(vals, axis=1)
    ax = self.ax
    ax.scatter(
      pooled,
      spread,
      c=pooled,
      cmap=_npg_cmap(_color("active")),
      vmin=0,
      vmax=1,
      s=12,
      alpha=0.6,
      edgecolors="none",
    )
    ax.set_xlabel("Pooled predicted probability")
    ax.set_ylabel("Descriptor disagreement (std)")
    ax.set_title(f"Consensus across {D.shape[1]} descriptors")
    ax.set_xlim(0, 1)
    self.is_available = True


class AdCoveragePlot(BasePlot):
  """Applicability-domain coverage: per-compound in-domain fraction across descriptors (predict).

  A spike at 1.0 means most queries are well inside the training domain; a left tail flags
  low-confidence, extrapolated predictions. Truth-free; needs the pooler's ``clf_ad`` column."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "ad-coverage"
    self.is_available = False
    if self.has_clf_data():
      return
    ad = ResultsFetcher(path=path).get_pred_ad_clf()
    if ad is None:
      return
    ad = np.asarray(ad, dtype=float)
    ad = ad[np.isfinite(ad)]
    if ad.size == 0:
      return
    n_out = int((ad <= 0.0).sum())
    _predict_hist(
      self.ax,
      ad,
      xlabel="In-domain fraction (across descriptors)",
      title=f"Applicability domain · {n_out}/{ad.size} fully out-of-domain",
      xlim=(0, 1),
    )
    self.is_available = True


class RankDistributionPlot(BasePlot):
  """Distribution of the pooled rank against the rank-reference library (predict, truth-free).

  lazy-qsar's ``rank`` scale: 0.50 beats 90% of drug-like space, 0.65 beats 99% (the binary call,
  dashed), 0.75 beats 99.9%."""

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 2))
    self.name = "rank-distribution"
    self.is_available = False
    if self.has_clf_data():
      return
    rk = ResultsFetcher(path=path).get_pred_rank_clf()
    if rk is None:
      return
    rk = np.asarray(rk, dtype=float)
    rk = rk[np.isfinite(rk)]
    if rk.size == 0:
      return
    _predict_hist(
      self.ax,
      rk,
      xlabel="Rank vs. drug-like reference library",
      title="Pooled rank",
      xlim=(0.0, 1.0),
      cutoff=RANK_DECISION_CUTOFF,
      color=_npg(2)[1],
    )
    self.is_available = True


class _PredictedPropertyDistributionPlot(BasePlot):
  """Base for a property distribution split by PREDICTED class (predict report, no ground truth).

  Same drawing as :class:`_PropertyDistributionPlot` but the split is the model's binary call
  (``clf_bin``) rather than the true label — a quick read on whether predicted actives differ
  physico-chemically from predicted inactives. Renders only without ground truth."""

  stem = "property"
  label = ""
  prop_index = 0

  def __init__(self, ax, path):
    BasePlot.__init__(self, ax=ax, path=path, cells=(2, 4))
    self.name = self.stem
    self.is_available = False
    if self.has_clf_data():
      return  # truth present → the class-split PropertyMw/LogpPlot cover this
    rf = ResultsFetcher(path=path)
    smiles = rf.get_smiles()
    bp = rf.get_pred_binary_clf()
    if not smiles or bp is None or len(smiles) != len(bp):
      return
    bp = np.asarray(bp, dtype=float)
    keep = np.isfinite(bp)
    smiles = [s for s, k in zip(smiles, keep) if k]
    bp = bp[keep].astype(int)
    if len(bp) == 0:
      return
    values = _rdkit_properties(smiles)[self.prop_index]
    self.is_available = _draw_one_property(self.ax, values, bp, self.label)


class PredictedPropertyMwPlot(_PredictedPropertyDistributionPlot):
  """Molecular-weight distribution split by predicted class (predict report)."""

  stem = "predicted-property-mw"
  label = "Molecular weight"
  prop_index = 0


class PredictedPropertyLogpPlot(_PredictedPropertyDistributionPlot):
  """LogP distribution split by predicted class (predict report)."""

  stem = "predicted-property-logp"
  label = "LogP"
  prop_index = 1
