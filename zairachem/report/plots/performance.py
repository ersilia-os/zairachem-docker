import numpy as np

from zairachem.report import BasePlot, perf
from zairachem.report.colors import (
  category_palette,
  named_colors,
  phase_color_rgb as _phase_color,
)

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
