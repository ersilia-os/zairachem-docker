"""Shared readers for the report's Computational performance section.

Pure data helpers (no matplotlib/stylia/HTML) so both the HTML renderer (:mod:`.html`) and the
matplotlib plot classes (:mod:`.plots`) read the same telemetry from one place: per-step timing and
resource snapshots from ``session.json``'s ``step_log``, the run's host machine specs, and the
molecule provenance (store vs freshly computed) from ``metadata/provenance.json``.
"""

import json
import os

from zairachem.base.vars import (
  METADATA_SUBFOLDER,
  PROVENANCE_FILENAME,
  SESSION_FILE,
)

# Sub-step name → (friendly label, phase). Phases group the session sub-steps under the top-level
# pipeline stages and drive bar/segment colour (shared by the CSS dashboard and the matplotlib plots).
SUBSTEP = {
  "initialize": ("Initialize", "setup"),
  "normalize_input": ("Normalize input", "setup"),
  "standardise_smiles": ("Standardize SMILES", "setup"),
  "tasks": ("Tasks", "setup"),
  "merge": ("Merge", "setup"),
  "setup_check": ("Setup check", "setup"),
  "clean": ("Clean", "setup"),
  "raw_descriptions": ("Descriptors", "describe"),
  "manifolds": ("Projections", "projections"),
  "treated_descriptions": ("Treat descriptors", "treat"),
  "lazy-qsar": ("Lazy-QSAR training", "estimate"),
  "simple_evaluation": ("Evaluation", "estimate"),
  "report": ("Report", "report"),
  "finish": ("Finish", "finish"),
}

# Stable phase order. Phase → colour now lives in the single source of truth
# (:data:`zairachem.report.colors.PHASE_COLORS` / ``PHASE_COLOR_RGB``), shared by the plots and the
# HTML dashboard; this module stays free of matplotlib/stylia/HTML, so it only owns the order.
PHASE_ORDER = ["setup", "describe", "projections", "treat", "estimate", "pool", "report", "finish"]


# Steps excluded from the per-step timing breakdown (post-compute housekeeping; see step_telemetry).
_TIMING_EXCLUDE = {"report", "finish"}


def _humanize(name):
  return " ".join(w.capitalize() for w in (name or "").replace("_", " ").replace("-", " ").split())


def fmt_duration(seconds):
  """Compact human duration: ``45s`` under a minute, else ``Xm Ys``."""
  s = int(round(seconds))
  return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


def host_info(output_dir):
  """The run's host machine specs from ``session.json``'s ``host`` block (``{}`` if absent)."""
  try:
    with open(os.path.join(output_dir, SESSION_FILE)) as f:
      return json.load(f).get("host") or {}
  except Exception:
    return {}


def step_telemetry(output_dir):
  """Per-step timing + resource snapshots from ``session.json``'s ``step_log``.

  Durations are gaps between consecutive completion timestamps (the first measured from the session
  ``time_stamp``). Returns ``{steps, total_seconds, peak_ram_gb, ram_total_gb, peak_cpu}``; ``steps``
  is empty when no ``step_log`` was recorded (older runs).
  """
  try:
    with open(os.path.join(output_dir, SESSION_FILE)) as f:
      data = json.load(f)
  except Exception:
    return {
      "steps": [],
      "total_seconds": None,
      "peak_ram_gb": None,
      "ram_total_gb": None,
      "peak_cpu": None,
    }
  total = data.get("elapsed_time")
  log = data.get("step_log") or []
  # Dedupe by last occurrence per name (tolerates re-runs) while keeping order. The report/finish
  # steps are excluded from the per-step breakdown: they are post-compute housekeeping, and the report
  # figures are rendered *before* those markers are written, so they can't time themselves — including
  # them would only add a spurious bar on a re-render. Total time (elapsed_time) still covers them.
  seen = {}
  for rec in log:
    if rec.get("name") in _TIMING_EXCLUDE:
      continue
    seen[rec.get("name")] = rec
  ordered = list(seen.values())
  prev = data.get("time_stamp")
  steps = []
  peak_ram = None
  peak_cpu = None
  ram_total = None
  for rec in ordered:
    name = rec.get("name")
    t = rec.get("t")
    secs = None
    if t is not None and prev is not None:
      secs = max(0.0, t - prev)
      prev = t
    label, phase = SUBSTEP.get(name, (_humanize(name), "other"))
    ram_used = rec.get("ram_used_gb")
    cpu = rec.get("cpu")
    if ram_used is not None:
      peak_ram = ram_used if peak_ram is None else max(peak_ram, ram_used)
    if cpu is not None:
      peak_cpu = cpu if peak_cpu is None else max(peak_cpu, cpu)
    if rec.get("ram_total_gb") is not None:
      ram_total = rec.get("ram_total_gb")
    steps.append({
      "name": name,
      "label": label,
      "phase": phase,
      "seconds": secs,
      "cpu": cpu,
      "ram_used_gb": ram_used,
      "ram_pct": rec.get("ram_pct"),
    })
  return {
    "steps": _collapse_setup(steps),
    "total_seconds": total,
    "peak_ram_gb": peak_ram,
    "ram_total_gb": ram_total,
    "peak_cpu": peak_cpu,
  }


def _collapse_setup(steps):
  """Collapse the many setup sub-steps into just ``Standardize SMILES`` + ``Other setup operations``.

  The setup phase has seven sub-steps; only SMILES standardization is interesting on its own, so the
  rest are summed into one bar (durations added, CPU/RAM taken as the peak across them). Non-setup
  phases are untouched. The two setup rows are emitted (SMILES first) where the setup block sits.
  """
  setup = [s for s in steps if s["phase"] == "setup"]
  if len(setup) <= 1:
    return steps
  smiles = next((s for s in setup if s["name"] == "standardise_smiles"), None)
  others = [s for s in setup if s["name"] != "standardise_smiles"]

  def _peak(items, key):
    vals = [s[key] for s in items if s[key] is not None]
    return max(vals) if vals else None

  other_secs = [s["seconds"] for s in others if s["seconds"] is not None]
  other = {
    "name": "other-setup",
    "label": "Other setup operations",
    "phase": "setup",
    "seconds": sum(other_secs) if other_secs else None,
    "cpu": _peak(others, "cpu"),
    "ram_used_gb": _peak(others, "ram_used_gb"),
    "ram_pct": _peak(others, "ram_pct"),
  }
  collapsed = ([smiles] if smiles else []) + ([other] if others else [])
  out = []
  emitted = False
  for s in steps:
    if s["phase"] != "setup":
      out.append(s)
    elif not emitted:
      out.extend(collapsed)
      emitted = True
  return out


def provenance(output_dir):
  """Per-model molecule provenance (store vs computed) from ``metadata/provenance.json``.

  Returns ``{models, migrated, totals}`` where ``models`` is a list of
  ``{id, role, total, from_store, computed}`` (featurizers then projections), or ``None`` if absent.
  """
  try:
    with open(os.path.join(output_dir, METADATA_SUBFOLDER, PROVENANCE_FILENAME)) as f:
      data = json.load(f)
  except Exception:
    return None
  models = []
  for role in ("featurizers", "projections"):
    for mid, rec in (data.get(role) or {}).items():
      models.append({
        "id": mid,
        "role": role[:-1],  # "featurizer" / "projection"
        "total": rec.get("n_total") or 0,
        "from_store": rec.get("n_from_project") or 0,
        "computed": rec.get("n_computed") or 0,
        "seconds": rec.get("seconds"),
      })
  if not models:
    return None
  sum_store = sum(m["from_store"] for m in models)
  sum_computed = sum(m["computed"] for m in models)
  denom = sum_store + sum_computed
  return {
    "models": models,
    "migrated": data.get("migrated") or {},
    "totals": {
      "from_store": sum_store,
      "computed": sum_computed,
      "reuse_pct": (100.0 * sum_store / denom) if denom else None,
    },
  }


def fmt_size(num_bytes):
  """Human-readable byte size: ``101 MB``, ``1.4 GB`` (binary units)."""
  size = float(num_bytes)
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if size < 1024 or unit == "TB":
      return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
    size /= 1024
  return f"{size:.1f} TB"


def _dir_size(path):
  total = 0
  for dirpath, _, filenames in os.walk(path):
    for f in filenames:
      try:
        total += os.path.getsize(os.path.join(dirpath, f))
      except OSError:
        continue
  return total


def store_size(params):
  """On-disk size of the isaura **project** bucket holding this run's cached descriptors.

  Returns ``{store_name, total_bytes, per_model}`` (per_model: bucket subdir → bytes), or ``None`` if
  the store is off or its local MinIO bucket can't be found. Only the per-project bucket is sized
  (``~/minio-data/<store>``) — fast and run-relevant — not the large shared public lake. The MinIO
  data dir defaults to ``~/minio-data`` and is overridable via ``MINIO_DATA_DIR``.
  """
  store = params.get("store") or params.get("contribute_store")
  if not store:
    return None
  root = os.environ.get("MINIO_DATA_DIR") or os.path.join(os.path.expanduser("~"), "minio-data")
  bucket = os.path.join(root, store)
  if not os.path.isdir(bucket):
    return None
  per_model = {}
  total = 0
  try:
    for entry in os.scandir(bucket):
      if entry.is_dir():
        sz = _dir_size(entry.path)
        per_model[entry.name] = sz
        total += sz
      elif entry.is_file():
        total += entry.stat().st_size
  except OSError:
    return None
  return {"store_name": store, "total_bytes": total, "per_model": per_model}
