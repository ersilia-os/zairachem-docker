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
import subprocess
from functools import lru_cache

#: Ceiling for pure network fan-out (registry checks, downloads) — enough to hide latency, not a flood.
_IO_CAP = 8


def cpu_count():
  """Logical CPU count, with a safe fallback when undeterminable."""
  return os.cpu_count() or 4


@lru_cache(maxsize=1)
def _docker_info():
  """``(cpus, memory_bytes)`` of the Docker engine from one ``docker info`` call, or None."""
  try:
    out = subprocess.run(
      ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"],
      capture_output=True,
      text=True,
      timeout=10,
    ).stdout.split()
    return int(out[0]), int(out[1])
  except (OSError, ValueError, IndexError, subprocess.SubprocessError):
    return None


def docker_cpu_count():
  """CPUs of the Docker engine (the VM on macOS/Windows), falling back to :func:`cpu_count`.

  Model servers run there, so their concurrency is bounded by it, not by the host's cores.
  """
  info = _docker_info()
  return max(1, info[0]) if info else cpu_count()


def docker_memory_gb():
  """Memory of the Docker engine (the VM on macOS/Windows) in GB, or None when it cannot be read."""
  info = _docker_info()
  return info[1] / 1024**3 if info else None


def host_memory_gb():
  """Physical memory of this machine in GB, or None when it cannot be read."""
  try:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
  except (ValueError, OSError, AttributeError):
    return None


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

  ``env`` (e.g. ``"ZAIRACHEM_DESCRIBE_WORKERS"``) takes precedence when set to a valid int.
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
