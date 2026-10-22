"""Single source of truth for the report's semantic colors.

Everything that needs a color — the matplotlib plots (RGB), ``perf`` (hex), and the HTML/CSS
dashboard (hex) — pulls from here, so the palette can't drift across the three. Colors anchor to
stylia's ``ArticleColors`` (the non-branded NPG / Nature Publishing Group palette that the report
renders with, via ``stylia.set_style("article")``).

Convention: actives are crimson-red, inactives cobalt-blue, used consistently across every plot.
Depends only on ``stylia.colors`` (never on ``zairachem.report.__init__``) to avoid an import cycle.
"""

from stylia.colors import ArticleColors, CategoricalPalette

_AC = ArticleColors()

# Semantic key → stylia ArticleColors name. THE source of truth for what each color means.
_SEMANTIC = {
  "active": "crimson",
  "inactive": "cobalt",
  "neutral": "silver",  # the low end of a sequential value ramp
  "correct_positive": "lime",  # TP
  "correct_negative": "cobalt",  # TN
  "false_positive": "periwinkle",  # FP
  "false_negative": "crimson",  # FN
  "store": "cobalt",  # descriptor reused from the isaura store
  "computed": "silver",  # descriptor freshly computed this run
  "classification": "turquoise",
  "regression": "amber",
}


def rgb(key):
  """RGB tuple for a semantic key (for matplotlib)."""
  return getattr(_AC, _SEMANTIC[key])


def hexcol(key):
  """Hex string ``#RRGGBB`` for a semantic key (for CSS / HTML)."""
  return _AC.hex[_SEMANTIC[key]]


# Pipeline phase → stylia article (NPG) named colour. The matplotlib plots use the RGB view; perf.py
# and the HTML dashboard use the hex view — both from this one mapping so they always match.
_PHASE = {
  "setup": "cobalt",
  "describe": "turquoise",
  "projections": "amber",
  "treat": "periwinkle",
  "estimate": "crimson",
  "pool": "orchid",
  "report": "lime",
  "finish": "tangerine",
  "other": "silver",
}
PHASE_COLOR_RGB = {k: getattr(_AC, v) for k, v in _PHASE.items()}
PHASE_COLORS = {k: _AC.hex[v] for k, v in _PHASE.items()}


def phase_color_rgb(phase):
  """Return the RGB colour of a pipeline phase, falling back to the ``other`` phase colour."""
  return PHASE_COLOR_RGB.get(phase, PHASE_COLOR_RGB["other"])


# Backward-compatible color-keyed palette so existing plot call sites (named_colors.red, …) are
# unchanged. The names map to the same stylia colours the semantic keys above resolve to.
class _Palette:
  red = _AC.crimson
  blue = _AC.cobalt
  gray = _AC.silver
  green = _AC.lime
  purple = _AC.periwinkle
  black = _AC.black


named_colors = _Palette()

# Cycling NPG palette for categorical series (a distinct color per model / estimator).
category_palette = CategoricalPalette("npg")


def _rgb_to_hex(c):
  return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3])


def descriptor_colors_rgb(n):
  """``n`` per-descriptor identity colours (RGB), assigned by best-first rank. The same rank always
  maps to the same NPG hue, so a descriptor keeps one colour across every figure and the HTML bars."""
  return category_palette.get(n)


def descriptor_colors_hex(n):
  """``n`` per-descriptor identity colours as ``#RRGGBB`` (the CSS view of :func:`descriptor_colors_rgb`)."""
  return [_rgb_to_hex(c) for c in category_palette.get(n)]


# Ordered legend entries surfaced as the report's single transversal color key (label, semantic key),
# per task. "Active" and "Inactive" mean nothing on a regression report, where the same two hues carry
# the ends of the target's own range instead.
_LEGEND = {
  "classification": [
    ("Active", "active"),
    ("Inactive", "inactive"),
    ("Correct", "correct_positive"),
    ("Error", "false_negative"),
  ],
  "regression": [
    ("High value", "active"),
    ("Low value", "inactive"),
    ("Observed", "correct_positive"),
    ("Error", "false_negative"),
  ],
}
#: Provenance entries, appended to either task's legend (the descriptors are task-neutral).
_LEGEND_SHARED = [("From store", "store"), ("Computed", "computed")]


def legend(task="classification"):
  """Ordered ``(label, semantic key)`` legend entries for the report's transversal colour key."""
  return _LEGEND.get(task, _LEGEND["classification"]) + _LEGEND_SHARED
