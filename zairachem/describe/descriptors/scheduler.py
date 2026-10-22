"""Scheduling of model servers by the Docker VM's memory and CPU.

Model servers share the Docker VM, and they differ a lot: some need GBs of memory, some keep every
core busy, most use about one. Running a fixed number side by side either wastes the VM
or overloads it, so models are admitted while their memory reservations and measured cores (from
their profiles, see :mod:`~zairachem.describe.descriptors.profiles`) fit; a model that fits nowhere
runs alone. :class:`PeakSampler` measures a container's memory and CPU while it works, from one
shared streaming ``docker stats`` (:class:`StatsMonitor`).
"""

import contextlib
import re
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from zairachem.base.vars import DESCRIBE_MEMORY_RESERVE_GB

_UNITS = {"GiB": 1.0, "MiB": 1 / 1024, "KiB": 1 / 1024**2, "B": 1 / 1024**3}
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _parse_gb(text):
  """``"1.2GiB"`` → 1.2 (GB); None when it cannot be parsed."""
  text = text.strip()
  for unit, factor in _UNITS.items():
    if text.endswith(unit):
      try:
        return float(text[: -len(unit)]) * factor
      except ValueError:
        return None
  return None


def memory_budget(vm_memory_gb):
  """Memory (GB) the model servers may share: the VM's minus a reserve, or None when unknown."""
  if vm_memory_gb is None:
    return None
  return max(0.0, vm_memory_gb - DESCRIBE_MEMORY_RESERVE_GB)


def containers_memory_gb():
  """Memory (GB) of each running container right now, ``{name: gb}`` ({} when unknown).

  Asked over the Docker socket when possible (a fraction of the CLI's ~3 s).
  """
  from zairachem.base.utils.docker_socket import containers_memory_gb as from_socket

  fast = from_socket()
  if fast is not None:
    return fast
  try:
    out = subprocess.run(
      ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
      capture_output=True,
      text=True,
      timeout=60,
    ).stdout
  except (OSError, subprocess.SubprocessError):
    return {}
  usage = {}
  for line in out.splitlines():
    name, _, mem = line.partition("\t")
    gb = _parse_gb(mem.split("/")[0])
    if gb is not None:
      usage[name] = gb
  return usage


def schedule(items, run, cost, budget_gb, max_running, cores=None, cpu_budget=None, width=None):
  """Call ``run(item)`` for every item on worker threads, admitting items while their costs fit.

  An item starts when its memory cost fits in what the running items leave of ``budget_gb``
  (``None``: no memory limit), its ``cores`` fit in what they leave of ``cpu_budget`` (``None``: no
  CPU limit), and fewer than ``max_running`` are running. When nothing is running, the first
  pending item starts whatever its cost, so an item larger than a budget runs alone instead of
  waiting forever. Items that fit are started ahead of a waiting larger one.

  Parameters
  ----------
  items : list
      Items in the order they should be tried.
  run : callable
      Work for one item; its exceptions are collected, not raised.
  cost : callable
      Memory cost (GB) of an item.
  budget_gb : float or None
      Memory the running items may use together.
  max_running : int
      Most items running at once.
  cores : callable, optional
      CPU cores an item keeps busy.
  cpu_budget : float, optional
      Cores the running items may use together.
  width : callable, optional
      ``width(item, free_gb, free_cores) -> k``: how many times its cost and cores an item takes,
      decided when it starts from what the running items leave free (``None``: unlimited). Default 1.

  Returns
  -------
  dict
      Item → the exception it raised, or None.
  """
  admit = _Admission(items, cost, budget_gb, max_running, cores, cpu_budget, width)
  running, outcome = {}, {}
  with ThreadPoolExecutor(max_workers=admit.max_running) as ex:
    while admit.pending or running:
      for item in admit.next_items():
        running[ex.submit(run, item)] = item
      done, _ = wait(running, return_when=FIRST_COMPLETED)
      for fut in done:
        item = running.pop(fut)
        admit.release(item)
        outcome[item] = fut.exception()
  return outcome


def simulate(items, seconds, cost, budget_gb, max_running, cores=None, cpu_budget=None, width=None):
  """Wall time (s) :func:`schedule` would take if each item ran for ``seconds(item, k)``.

  Same admission rules as :func:`schedule`, on a simulated clock; ``k`` is the item's width.
  """
  admit = _Admission(items, cost, budget_gb, max_running, cores, cpu_budget, width)
  clock, running = 0.0, []  # (end time, item)
  while admit.pending or running:
    running += [(clock + seconds(i, admit.widths[i]), i) for i in admit.next_items()]
    running.sort(key=lambda r: r[0])
    clock, item = running.pop(0)
    admit.release(item)
  return clock


class _Admission:
  """The admission rules shared by :func:`schedule` and :func:`simulate`."""

  def __init__(self, items, cost, budget_gb, max_running, cores, cpu_budget, width=None):
    self.pending = list(items)
    self.costs = {i: cost(i) for i in self.pending}
    self.cpus = {i: (cores(i) if cores else 0.0) for i in self.pending}
    self.budget_gb, self.cpu_budget = budget_gb, cpu_budget
    self.max_running = max(1, max_running)
    self.width = width
    self.widths = {}  # item -> how many times its cost and cores it took when it started
    self.n_running, self.used, self.busy = 0, 0.0, 0.0

  def _fits(self, i):
    if not self.n_running:
      return True
    mem_ok = self.budget_gb is None or self.used + self.costs[i] <= self.budget_gb
    cpu_ok = self.cpu_budget is None or self.busy + self.cpus[i] <= self.cpu_budget + 1e-9
    return mem_ok and cpu_ok

  def next_items(self):
    """Admit and return the pending items that may start now, in order."""
    started = []
    while self.n_running < self.max_running:
      item = next((i for i in self.pending if self._fits(i)), None)
      if item is None:
        break
      self.pending.remove(item)
      k = 1
      if self.width is not None:
        free_gb = None if self.budget_gb is None else self.budget_gb - self.used
        free_cores = None if self.cpu_budget is None else self.cpu_budget - self.busy
        k = max(1, int(self.width(item, free_gb, free_cores)))
      self.widths[item] = k
      self.n_running += 1
      self.used += k * self.costs[item]
      self.busy += k * self.cpus[item]
      started.append(item)
    return started

  def release(self, item):
    """Free what a finished item held."""
    k = self.widths.get(item, 1)
    self.n_running -= 1
    self.used -= k * self.costs[item]
    self.busy -= k * self.cpus[item]


class StatsMonitor:
  """One streaming ``docker stats`` for the whole process, shared by every :class:`PeakSampler`.

  Polling ``docker stats --no-stream`` per container costs ~2 s per call and keeps the Docker
  daemon busy; the streaming command reports all containers about once a second for the price of
  one process. Containers are matched by id prefix.
  """

  _instance = None
  _instance_lock = threading.Lock()

  def __init__(self):
    self._lock = threading.Lock()
    self._watched = {}  # short id -> {"peak": GB, "cpu": [cores, ...]}
    self._proc = None

  @classmethod
  def get(cls):
    """The process-wide monitor, started on first use."""
    with cls._instance_lock:
      if cls._instance is None:
        cls._instance = cls()
      cls._instance._ensure_started()
      return cls._instance

  def _ensure_started(self):
    if self._proc is not None and self._proc.poll() is None:
      return
    try:
      self._proc = subprocess.Popen(
        ["docker", "stats", "--format", "{{.ID}}\t{{.MemUsage}}\t{{.CPUPerc}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
      )
    except OSError:
      self._proc = None
      return
    threading.Thread(target=self._read, args=(self._proc,), daemon=True).start()

  def _read(self, proc):
    for raw in proc.stdout:
      parts = _ANSI.sub("", raw).strip().split("\t")
      if len(parts) != 3:
        continue
      short, mem, cpu = parts
      gb = _parse_gb(mem.split("/")[0])
      with self._lock:
        entry = self._watched.get(short)
        if entry is None:
          continue
        if gb is not None:
          entry["peak"] = max(entry["peak"], gb)
        with contextlib.suppress(ValueError):
          entry["cpu"].append(float(cpu.strip().rstrip("%")) / 100)

  def watch(self, container_id):
    """Start collecting for ``container_id``."""
    with self._lock:
      self._watched.setdefault(container_id[:12], {"peak": 0.0, "cpu": []})

  def take(self, container_id):
    """``(peak_gb, cores)`` for ``container_id`` since the last take; restarts collection."""
    with self._lock:
      entry = self._watched.get(container_id[:12])
      if entry is None:
        return 0.0, None
      peak, cpu = entry["peak"], sorted(entry["cpu"])
      entry["peak"], entry["cpu"] = 0.0, []
    return peak, (cpu[len(cpu) // 2] if cpu else None)

  def unwatch(self, container_id):
    """Stop collecting for ``container_id``."""
    with self._lock:
      self._watched.pop(container_id[:12], None)

  @classmethod
  def stop(cls):
    """Stop the streaming process (end of describe)."""
    with cls._instance_lock:
      inst = cls._instance
      if inst is not None and inst._proc is not None and inst._proc.poll() is None:
        inst._proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
          inst._proc.wait(timeout=5)
      cls._instance = None


class PeakSampler:
  """A container's peak memory and CPU while in use, read from the shared :class:`StatsMonitor`.

  :meth:`take` returns ``(peak_gb, cores)`` since the previous call: the highest memory seen and
  the median CPU use (1.0 = one core busy).
  """

  def __init__(self, container_id):
    self.container_id = container_id
    self._monitor = None

  def reset(self):
    """Return the peak since the last reset (GB) and start a new measurement."""
    return self.take()[0]

  def take(self):
    """``(peak_gb, cores)`` since the last call, then start a new measurement."""
    if self._monitor is None:
      return 0.0, None
    return self._monitor.take(self.container_id)

  def __enter__(self):
    if self.container_id:
      self._monitor = StatsMonitor.get()
      self._monitor.watch(self.container_id)
    return self

  def __exit__(self, *exc):
    if self._monitor is not None:
      self._monitor.unwatch(self.container_id)
    return False
