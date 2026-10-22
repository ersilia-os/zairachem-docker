"""Report catalogue: section/group/title tables for the figures and small helpers on them."""

# Plots grouped into sections (anchor, heading, description, member stems). Unlisted → "More".
_CATEGORIES = [
  (
    "predictions",
    "Predictions",
    "What the model predicts for these molecules and where the decision cutoff falls.",
    ["predicted-score-hist", "rank-distribution", "predicted-rank-curve"],
  ),
  (
    "confidence",
    "Confidence",
    "How much to trust each prediction (descriptor agreement, similarity to the training set).",
    [
      "descriptor-consensus",
      "value-consensus",
      "tanimoto-similarity-to-train",
      "predicted-property-mw",
      "predicted-property-logp",
    ],
  ),
  (
    "performance",
    "Inner model performance",
    "Cross-validated (out-of-fold) performance of the pooled model against the labels.",
    [
      "oof-score-proba",
      "oof-score-proba-pts",
      "oof-score-logit",
      "oof-score-logit-pts",
      "oof-score-rank",
      "oof-score-rank-pts",
      "oof-score-lift",
      "oof-score-lift-pts",
      "oof-score-raw",
      "oof-score-raw-pts",
      "roc-curve",
      "pr-curve",
      "calibration-curve",
      "enrichment-curve",
      "threshold-sweep",
      "confusion-matrix",
      "confusion-normalized",
      "confusion-precision",
      "confusion-breakdown",
      "descriptor-correlation",
      "topk-overlap-curve",
      "r2-individual",
      "regression-raw",
      "regression-residuals",
      "regression-distribution",
      "regression-error-similarity",
      "potency-enrichment",
      "interval-coverage",
    ],
  ),
  (
    "validation",
    "Held-out validation",
    "Out-of-sample performance under random, scaffold, deterministic and Butina 80:20 splits.",
    [
      "heldout-roc",
      "heldout-pr",
      "heldout-calibration",
      "heldout-enrichment-factor",
      "heldout-validation",
      "heldout-r2",
      "heldout-metric-bars",
      "heldout-confusion",
    ],
  ),
]


# Predict-aware overrides. The static _CATEGORIES / _GROUPS read from the fit perspective; at predict
# the same figures describe the NEW molecules against the trained model, so a couple of headings shift
# meaning. Applied where the section heading and the carousel title are emitted (never mutating the
# static tables), keyed by section anchor / group key.
_PREDICT_SECTION = {
  "performance": (
    "Test-set performance",
    "How the trained model performs on your labelled test set (training reference overlaid).",
  ),
}


_PREDICT_GROUP_TITLE = {
  "oof-scores": "Predicted score by true class",
}


_TITLES = {
  "actives-inactives": "Class balance",
  "roc-curve": "ROC curve",
  "confusion-matrix": "Confusion matrix",
  "r2-individual": "R² by estimator",
  "regression-raw": "Predicted vs observed",
  "regression-residuals": "Residuals",
  "regression-distribution": "Predicted values",
  "regression-error-similarity": "Error vs training similarity",
  "interval-coverage": "Interval coverage",
  "potency-enrichment": "Selection quality",
  "cv-auroc": "Inner CV AUROC by descriptor",
  "cv-spearman": "Inner CV Spearman \u03c1 by descriptor",
  "cv-aupr": "Inner CV AUPR by descriptor",
  "cv-mcc": "MCC at ideal cutoff",
  "cv-f1": "F1 at ideal cutoff",
  "cv-balacc": "Balanced accuracy at ideal cutoff",
  "cv-precision": "Precision at ideal cutoff",
  "cv-recall": "Recall at ideal cutoff",
  "cv-cutoff": "Ideal decision cutoff",
  "cv-roc": "Cross-validation ROC",
  "cv-pr": "Cross-validation precision-recall",
  "projection-mwlogp": "Molecular weight vs LogP",
  "projection-umap": "UMAP projection",
  "projection-tsne": "t-SNE projection",
  "projection-pca": "PCA projection",
  "projection-tmap": "TMAP projection",
  "projection-correctness": "Prediction correctness in chemical space",
  "pr-curve": "Precision-recall curve",
  "enrichment-curve": "Enrichment curve (cumulative gain)",
  "enrichment-factor-curve": "Enrichment factor curve",
  "threshold-sweep": "Threshold sweep",
  "calibration-curve": "Calibration curve",
  "confusion-normalized": "Confusion matrix (recall)",
  "confusion-precision": "Confusion matrix (precision)",
  "confusion-breakdown": "Outcome composition",
  "descriptor-correlation": "Descriptor prediction correlation",
  "topk-overlap-curve": "Descriptor agreement (top-x% overlap)",
  "oof-score-proba": "Probability",
  "oof-score-logit": "Log-odds",
  "oof-score-rank": "Percentile rank",
  "oof-score-lift": "Lift",
  "oof-score-raw": "Raw",
  "oof-score-proba-pts": "Probability · points",
  "oof-score-logit-pts": "Log-odds · points",
  "oof-score-rank-pts": "Percentile rank · points",
  "oof-score-lift-pts": "Lift · points",
  "oof-score-raw-pts": "Raw · points",
  "heldout-validation": "Held-out AUROC by split",
  "heldout-r2": "Held-out R² by split",
  "heldout-roc": "Held-out ROC by split",
  "heldout-pr": "Held-out PR by split",
  "heldout-calibration": "Held-out calibration by split",
  "heldout-enrichment-factor": "Held-out enrichment by split",
  "heldout-metric-bars": "Held-out metrics by split",
  "heldout-confusion": "Held-out outcome composition by split",
  "class-donut": "Class balance (donut)",
  "class-waffle": "Class balance (waffle)",
  "property-mw": "Molecular weight",
  "property-logp": "LogP",
  "property-mw-value": "Molecular weight vs value",
  "property-logp-value": "LogP vs value",
  "predicted-score-hist": "Predicted score distribution",
  "predicted-rank-curve": "Ranked scores (operating point)",
  "value-consensus": "Descriptor consensus",
  "descriptor-consensus": "Descriptor consensus",
  "tanimoto-similarity-to-train": "Tanimoto similarity to the training set",
  "rank-distribution": "Pooled rank",
  "predicted-property-mw": "Molecular weight (by predicted class)",
  "predicted-property-logp": "LogP (by predicted class)",
  "projection-mwlogp-proba": "MW vs LogP · by score",
  "projection-pca-proba": "PCA · by score",
  "projection-umap-proba": "UMAP · by score",
  "projection-tsne-proba": "t-SNE · by score",
  "projection-tmap-proba": "TMAP · by score",
  "overview": "Report overview",
  "step-timing": "Step timing",
  "phase-time": "Time by phase",
  "model-timing": "Time per model",
  "resource-timeline": "Resource usage over time",
  "compute-provenance": "Compute provenance",
}


# Redundant / near-redundant figures collapsed into a single carousel card. A group renders once, at
# its ``home`` section; its members are suppressed everywhere else. Members appear in listed order and
# only if their PNG is present, so a group left with a single present member degrades to a plain card.
_GROUPS = [
  {
    "key": "oof-scores",
    "title": "Out-of-fold pooled score",
    "home": "performance",
    "members": [
      "oof-score-proba",
      "oof-score-proba-pts",
      "oof-score-logit",
      "oof-score-logit-pts",
      "oof-score-rank",
      "oof-score-rank-pts",
      "oof-score-lift",
      "oof-score-lift-pts",
      "oof-score-raw",
      "oof-score-raw-pts",
    ],
  },
  {
    "key": "perf-curves",
    "title": "Performance curves",
    "home": "performance",
    "members": [
      "roc-curve",
      "pr-curve",
      "calibration-curve",
      "enrichment-curve",
      "enrichment-factor-curve",
      "threshold-sweep",
    ],
  },
  {
    "key": "confusion",
    "title": "Confusion matrix",
    "home": "performance",
    "members": [
      "confusion-matrix",
      "confusion-normalized",
      "confusion-precision",
      "confusion-breakdown",
    ],
  },
  {
    "key": "regression",
    "title": "Predicted vs observed",
    "home": "performance",
    "members": ["regression-raw", "regression-residuals"],
  },
  {
    "key": "descriptor-agreement",
    "title": "Descriptor agreement",
    "home": "performance",
    "members": ["descriptor-correlation", "topk-overlap-curve"],
  },
  {
    "key": "heldout-curves",
    "title": "Held-out curves by split strategy",
    "home": "validation",
    "members": [
      "heldout-roc",
      "heldout-pr",
      "heldout-calibration",
      "heldout-enrichment-factor",
      "heldout-validation",
      "heldout-r2",
    ],
  },
  {
    "key": "heldout-summary",
    "title": "Held-out summary by split strategy",
    "home": "validation",
    "members": ["heldout-metric-bars", "heldout-confusion"],
  },
]


_STEM_TO_GROUP = {m: g for g in _GROUPS for m in g["members"]}


_ACRONYMS = {
  "roc": "ROC",
  "auroc": "AUROC",
  "umap": "UMAP",
  "tsne": "t-SNE",
  "pca": "PCA",
  "r2": "R²",
}


# Curated metric columns for the performance table (only those present are shown).


def _projection_label(stem):
  """Clean label for a dynamic projection stem, reusing the base projection titles in ``_TITLES``.

  ``projection-merged-umap`` → "UMAP projection"; ``projection-umap-active`` → "UMAP · actives".
  """
  body = stem[len("projection-") :]
  if body.startswith("merged-"):
    return _TITLES.get(f"projection-{body[len('merged-') :]}")
  for noun in ("active", "inactive"):
    if body.endswith(f"-{noun}"):
      base = _TITLES.get(f"projection-{body[: -(len(noun) + 1)]}")
      short = base.replace(" projection", "") if base else None
      return f"{short} · {noun}s" if short else None
  return None


def _humanize(stem):
  if stem in _TITLES:
    return _TITLES[stem]
  if stem.startswith("projection-"):
    label = _projection_label(stem)
    if label:
      return label
  words = stem.replace("_", " ").replace("-", " ").split()
  return " ".join(_ACRONYMS.get(w.lower(), w.capitalize()) for w in words)


# Held-out validation schemas: (csv strategy key, display label), in canonical (not ranked) order.
_VALIDATION_SCHEMAS = [
  ("random", "Random"),
  ("scaffold", "Scaffold"),
  ("scaffold_det", "Deterministic"),
  ("butina", "Butina"),
]


_BADGE_KEPT = (
  "<span style='background:#3fb95022;color:#2ea043;padding:1px 8px;border-radius:10px;"
  "font-weight:600;font-size:11.5px'>kept</span>"
)


_BADGE_DISCARDED = (
  "<span style='background:#f8514922;color:#cf222e;padding:1px 8px;border-radius:10px;"
  "font-weight:600;font-size:11.5px'>discarded</span>"
)


# Matplotlib compute figures, grouped into sliders. Each group renders as one carousel card (members
# pageable with arrows/dots); a group with a single present figure degrades to a plain card and
# upgrades to a slider automatically once more figures are added to it.
_COMPUTE_GROUPS = [
  {
    "key": "compute-timing",
    "title": "Timing & resources",
    "members": ["step-timing", "phase-time", "resource-timeline"],
  },
  {
    "key": "compute-models",
    "title": "Per-model",
    "members": ["compute-provenance", "model-timing"],
  },
]


_DATASET_GROUPS = [
  {
    "key": "class-balance",
    "title": "Class balance",
    "members": ["actives-inactives", "class-donut", "class-waffle"],
  },
  {
    "key": "properties",
    "title": "Property distributions",
    "members": ["property-mw", "property-logp"],
  },
  {
    "key": "properties-value",
    "title": "Property vs value",
    "members": ["property-mw-value", "property-logp-value"],
  },
]


# Projection display order (interpretable MW-vs-LogP first, then the learned embeddings).
_PROJECTION_ORDER = ["mwlogp", "umap", "tsne", "pca", "tmap"]


_DIAG_GROUPS = [
  {
    "key": "diag-metrics",
    "title": "Per-descriptor metrics",
    "members": [
      "cv-auroc",
      "cv-spearman",
      "cv-aupr",
      "cv-mcc",
      "cv-f1",
      "cv-balacc",
      "cv-precision",
      "cv-recall",
      "cv-cutoff",
    ],
  },
  {
    "key": "diag-curves",
    "title": "Cross-validation curves",
    "members": ["cv-roc", "cv-pr"],
  },
]
