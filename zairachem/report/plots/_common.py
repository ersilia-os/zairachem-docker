import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from zairachem.report.fetcher import ResultsFetcher

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# Colors come from the single source of truth in ``report/colors.py`` (semantic palette anchored to
# stylia's NPG ArticleColors), shared with perf.py and the HTML dashboard so nothing drifts. The
# color-keyed ``named_colors`` (named_colors.red etc.) and ``category_palette`` keep the names the
# plot call sites already use.
from zairachem.report.colors import (  # noqa: E402
  category_palette,
  named_colors,
  rgb as _color,
)


def _npg(n):
  """``n`` colours from stylia's NPG categorical palette (crimson, turquoise, amber, …)."""
  return category_palette.get(n)


def _npg_cmap(hex_hi, hex_lo="#ffffff"):
  """A white→NPG-colour sequential matplotlib colormap for heatmaps."""
  from matplotlib.colors import LinearSegmentedColormap

  return LinearSegmentedColormap.from_list("npg_seq", [hex_lo, hex_hi])


def _subsample(n, cap=10000):
  """Point indices to keep for a class of size ``n`` (random subsample above ``cap``) and an adaptive
  alpha (lower for denser plots), so heavily populated projections stay readable."""
  if n <= cap:
    idx = np.arange(n)
  else:
    idx = np.random.default_rng(0).choice(n, cap, replace=False)
  alpha = float(np.clip(1500.0 / max(1, len(idx)), 0.12, 0.9))
  return idx, alpha


def _training_score_by_class(path):
  """Training OOF pooled scores split by true class → ``(active_scores, inactive_scores)``.

  Used to overlay the model's training-time score behaviour behind predict-time distributions and
  curves (a reference for where actives vs inactives typically fall). NaN truth/score positions are
  dropped; returns two empty arrays when the trained reference is unavailable."""
  rf = ResultsFetcher(path=path)
  yt = rf.get_actives_inactives_trained()
  yp = rf.get_pred_proba_clf_trained()
  if yt is None or yp is None:
    return np.array([]), np.array([])
  yt = np.asarray(yt, dtype=float)
  yp = np.asarray(yp, dtype=float)
  keep = np.isfinite(yt) & np.isfinite(yp)
  yt, yp = yt[keep], yp[keep]
  return yp[yt == 1], yp[yt == 0]


def _draw_training_projection_background(ax, path, name):
  """Faint grey scatter of the trained model's molecules in the same projection space — an
  applicability-domain backdrop behind predict-time projections. No-op if the reference is missing."""
  coords = ResultsFetcher(path=path).get_projection_trained(name)
  if not coords:
    return
  xs = np.asarray(coords[0], dtype=float)
  ys = np.asarray(coords[1], dtype=float)
  keep = np.isfinite(xs) & np.isfinite(ys)
  xs, ys = xs[keep], ys[keep]
  if xs.size == 0:
    return
  idx, _ = _subsample(xs.size)
  ax.scatter(
    xs[idx], ys[idx], color=named_colors.gray, s=6, alpha=0.15, edgecolors="none", zorder=0
  )


_RDKIT_PROP_CACHE = {}


def _rdkit_properties(smiles):
  """(MW, LogP, TPSA) arrays for valid molecules; NaN where SMILES can't be parsed.

  Memoised per distinct SMILES sequence: the MW and LogP property plots — and their predicted-class
  variants — each request these, so without the cache every molecule would be re-parsed with RDKit
  several times per report run."""
  key = tuple(smiles)
  cached = _RDKIT_PROP_CACHE.get(key)
  if cached is not None:
    return cached
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
  result = (np.array(mw), np.array(logp), np.array(tpsa))
  _RDKIT_PROP_CACHE[key] = result
  return result


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
