"""Auto-sized worker pools, derived from the host at runtime — no baked-in hardware assumptions.

Every consumer should let an explicit value (a CLI flag) or the matching ``ZAIRACHEM_*_WORKERS`` env
var override these defaults; the helpers here only provide the *default* when nothing is set.

Two regimes:
- **CPU-bound** fan-out (e.g. rendering plots, RDKit standardization) scales with cores — more workers
  than cores just thrashes.
- **IO/network-bound** fan-out (e.g. registry/HTTP checks, downloads, hitting model-server containers)
  spends most of its time waiting, so a few more than cores is fine; we still cap it so we never
  hammer a registry or oversubscribe the shared Docker VM.
"""

import os

#: Ceiling for pure network fan-out (registry checks, downloads) — enough to hide latency, not a flood.
_IO_CAP = 8


def cpu_count():
  """Logical CPU count, with a safe fallback when undeterminable."""
  return os.cpu_count() or 4


def _env_int(env):
  if not env:
    return None
  raw = os.environ.get(env)
  if not raw:
    return None
  try:
    return max(1, int(raw))
  except ValueError:
    return None


def cpu_workers(n_items=None, cap=None, env=None):
  """Default worker count for CPU-bound fan-out: ~all cores, clamped to ``cap`` and ``n_items``.

  ``env`` (e.g. ``"ZAIRACHEM_ESTIMATE_WORKERS"``) takes precedence when set to a valid int.
  """
  w = _env_int(env)
  if w is None:
    w = cpu_count()
    if cap is not None:
      w = min(w, cap)
  if n_items is not None:
    w = min(w, max(1, n_items))
  return max(1, w)


def io_workers(n_items=None, cap=_IO_CAP, env=None):
  """Default worker count for IO/network-bound fan-out: capped at ``cap`` (default 8), clamped to items.

  ``env`` takes precedence when set to a valid int.
  """
  w = _env_int(env)
  if w is None:
    w = cap
  if n_items is not None:
    w = min(w, max(1, n_items))
  return max(1, w)
