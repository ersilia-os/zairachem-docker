"""Per-model memory and speed profiles, measured on this machine.

A model server's memory grows with the request, roughly ``peak(n) = base + slope * n`` for ``n``
molecules (nearly flat for most models). Each profile keeps the (molecules, peak memory, seconds) of
past requests, one per request size, and fits that line, so the describe scheduler can reserve a
model's predicted peak and run models side by side only while those fit in the Docker VM; a model
not yet measured on this machine is reserved ``DEFAULT_DESCRIBE_MEMORY_GB``. Each profile also
records the CPU cores a request keeps busy and the response format the image supports (learned
once, from its first request).
Profiles live in ``BASE_DIR/model_profiles.json``, keyed by model and tied to the image digest.
"""

import json
import os
import threading

import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  BASE_DIR,
  DEFAULT_DESCRIBE_MEMORY_GB,
  MODEL_PROFILES_FILENAME,
)

#: Requests smaller than this are not used for the slope: their sampled peaks are the noisiest.
MIN_SLOPE_REQUEST = 10
#: Fraction of a model's memory reservation its predicted peak may take.
MEMORY_HEADROOM = 0.9
#: Request sizes kept per profile (the most recently used ones).
MAX_SAMPLES = 30
_lock = threading.Lock()


def _path():
  return os.path.join(BASE_DIR, MODEL_PROFILES_FILENAME)


def _load_all():
  try:
    with open(_path()) as f:
      data = json.load(f)
    return data if isinstance(data, dict) else {}
  except (OSError, ValueError):
    return {}


class ModelProfile:
  """Measured requests of one model's server, and what they predict.

  Parameters
  ----------
  model_id : str
      Ersilia model id.
  digest : str or None
      Image digest the samples were measured on.
  samples : list of [int, float, float], optional
      ``[molecules, peak_gb, seconds]`` per successful request.
  output_type : {"heavy", "simple"} or None, optional
      Response format this image supports, learned from its first request (binary "heavy" when
      the model can read the server's binary input, else JSON "simple"); None until known.
  cores : float, optional
      CPU cores a request keeps busy (median container CPU, highest seen); None until measured.
  """

  def __init__(self, model_id, digest=None, samples=None, output_type=None, cores=None):
    self.model_id = model_id
    self.digest = digest
    self.samples = [list(s) for s in (samples or [])]
    self.output_type = output_type
    self.cores = cores  # CPU cores a request keeps busy (highest median seen), or None

  def record(self, n, peak_gb, seconds):
    """Add one successful request (``peak_gb`` 0 when the memory could not be sampled).

    One sample is kept per request size, with the highest peak seen at that size (the cautious
    value) and the shortest time (``seconds`` None keeps the previous one): a request only gets
    slower when other models compete for the VM, so the fastest one is what the model needs. The
    ``MAX_SAMPLES`` most recently recorded sizes are kept.
    """
    n = int(n)
    old = next((s for s in self.samples if s[0] == n), None)
    peak = max(float(peak_gb), old[1] if old else 0.0) if peak_gb else (old[1] if old else 0.0)
    times = [t for t in (seconds, old[2] if old else None) if t is not None]
    t = float(min(times)) if times else None
    self.samples = [s for s in self.samples if s[0] != n] + [[n, peak, t]]
    self.samples = self.samples[-MAX_SAMPLES:]

  def _memory_points(self):
    return [(n, p) for n, p, _ in self.samples if p > 0]

  def memory_line(self):
    """``(base_gb, slope_gb_per_molecule)`` fitted on the requests, or None without two sizes.

    The slope comes from requests of at least ``MIN_SLOPE_REQUEST`` molecules and is kept
    non-negative; the line is then lifted so no measured peak lies above it. Noise in the sampled
    peaks can only make it more cautious.
    """
    pts = self._memory_points()
    big = [(n, p) for n, p in pts if n >= MIN_SLOPE_REQUEST]
    if len({n for n, _ in big}) < 2:
      return None
    slope = float(np.polyfit([n for n, _ in big], [p for _, p in big], 1)[0])
    slope = max(0.0, slope)
    base = max(p - slope * n for n, p in pts)
    return base, slope

  def predicted_peak(self, n):
    """Predicted peak memory (GB) of a request of ``n`` molecules, or None if not yet known.

    The memory line at ``n``, but never more than the highest peak measured on requests of ``n``
    molecules or more: memory does not shrink with the request, so those bound it (the line, lifted
    above every peak, would otherwise add its slope's noise on top of a size already measured).
    With a single measured size, memory is taken as proportional to the molecules above it.
    """
    pts = self._memory_points()
    if not pts:
      return None
    measured = [p for m, p in pts if m >= n]
    bound = max(measured) if measured else None
    line = self.memory_line()
    if line is not None:
      predicted = line[0] + line[1] * n
    else:
      m, p = max(pts)
      predicted = p * max(1.0, n / m)
    return min(predicted, bound) if bound is not None else predicted

  def measured_up_to(self, n):
    """Whether :meth:`predicted_peak` at ``n`` rests on measurements, not a proportional bound.

    True when the memory line is fitted, or a request of at least ``n`` molecules was measured.
    """
    pts = self._memory_points()
    return self.memory_line() is not None or any(m >= n for m, _ in pts)

  def seconds_per_request(self):
    """``(fixed_s, seconds_per_molecule)`` of one request, from the timed requests, or None.

    A line through them when they span two sizes and it rises with the request (a falling one is
    noise, e.g. a request slowed by other models); otherwise the largest timed request taken as all
    per-molecule time, an upper bound below that size and proportional above it.
    """
    pts = [(n, t) for n, _, t in self.samples if t is not None]
    if not pts:
      return None
    if len({n for n, _ in pts}) >= 2:
      slope, fixed = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
      if slope >= 0:
        return max(0.0, float(fixed)), float(slope)
    n, t = max(pts)
    return 0.0, t / n


def load(model_id, digest):
  """The stored profile of ``model_id`` if it was measured on image ``digest``, else a new one."""
  entry = _load_all().get(model_id) or {}
  if not entry or (digest and entry.get("digest") and entry["digest"] != digest):
    # Nothing measured on this image here: start empty (a new image may differ in memory, cores and
    # output format, so everything is learned again).
    return ModelProfile(model_id, digest)
  return ModelProfile(
    model_id,
    digest or entry.get("digest"),
    entry.get("samples"),
    entry.get("output_type"),
    entry.get("cores"),
  )


def save(profile):
  """Store ``profile`` (replacing the model's previous one).

  Read-modify-write under a thread lock and a file lock, so concurrent runs of this user never
  overwrite each other's models.
  """
  import fcntl

  os.makedirs(BASE_DIR, exist_ok=True)
  with _lock, open(_path() + ".lock", "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    data = _load_all()
    data[profile.model_id] = {
      "digest": profile.digest,
      "samples": profile.samples,
      "output_type": profile.output_type,
      "cores": profile.cores,
    }
    path = _path()
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
      with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
      os.replace(tmp, path)
    except OSError as e:
      logger.debug(f"[profiles] could not save {profile.model_id}: {e}")


def reservation_gb(profile, n, budget_gb, idle_gb=0.0):
  """Memory (GB) to reserve for a model sending requests of ``n`` molecules.

  What its predicted peak at ``n`` adds to the ``idle_gb`` its server already holds, with headroom
  and at most ``budget_gb``; ``DEFAULT_DESCRIBE_MEMORY_GB`` before any request was measured.
  """
  predicted = profile.predicted_peak(n) if profile.samples else None
  if predicted is None:
    reserve = DEFAULT_DESCRIBE_MEMORY_GB
  else:
    reserve = max(0.0, predicted - idle_gb) / MEMORY_HEADROOM
  return min(budget_gb, reserve) if budget_gb else reserve
