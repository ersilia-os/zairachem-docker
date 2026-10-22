"""The describe scheduler admits model servers by memory: big ones alone, small ones together."""

import threading
import time

from zairachem.describe.descriptors.scheduler import schedule

COSTS = {"a": 0.5, "big": 7.0, "b": 1.5, "c": 2.0, "d": 2.7}


def _trial(budget, max_running, fail=()):
  lock = threading.Lock()
  live, peak = {}, {"gb": 0.0, "n": 0, "big_alone": None}

  def run(m):
    with lock:
      live[m] = COSTS[m]
      peak["gb"] = max(peak["gb"], sum(live.values()))
      peak["n"] = max(peak["n"], len(live))
      if m == "big":
        peak["big_alone"] = len(live) == 1
    time.sleep(0.02)
    with lock:
      live.pop(m)
    if m in fail:
      raise RuntimeError("boom")

  return schedule(list(COSTS), run, COSTS.get, budget, max_running), peak


def test_big_model_runs_alone_and_small_ones_share():
  outcome, peak = _trial(budget=6.5, max_running=5)
  assert all(e is None for e in outcome.values())
  assert peak["big_alone"]
  assert peak["n"] > 1
  assert peak["gb"] <= 7.0


def test_cpu_cap_and_failures_are_kept():
  outcome, peak = _trial(budget=None, max_running=2, fail={"b"})
  assert peak["n"] <= 2
  assert isinstance(outcome["b"], RuntimeError)
  assert sum(e is None for e in outcome.values()) == len(COSTS) - 1


def test_profile_memory_line_is_cautious():
  from zairachem.describe.descriptors.profiles import ModelProfile

  # A noisy low reading at 2 molecules must not create a slope; the line stays above every peak.
  profile = ModelProfile("eosfake", samples=[[2, 2.5, 5.0], [10, 3.6, 9.0], [20, 3.7, 10.0]])
  base, slope = profile.memory_line()
  assert slope < 0.02
  assert all(base + slope * n >= p - 1e-9 for n, p, _ in profile.samples)


def _bare_client(batch_size=10000, splittable=True):
  """A ``BinaryStreamClient`` without a server, with a profile and a settable fake sampler."""
  from zairachem.base.utils.logging import logger
  from zairachem.describe.descriptors.api import BinaryStreamClient
  from zairachem.describe.descriptors.profiles import ModelProfile

  class _Sampler:
    last = 0.0

    def take(self):
      peak, self.last = self.last, 0.0
      return peak, 1.0

    def reset(self):
      return self.take()[0]

  c = BinaryStreamClient.__new__(BinaryStreamClient)
  c.logger = logger
  c.model_id, c.batch_size, c._initial_batch_size, c._largest_ok = (
    "eosfake",
    batch_size,
    batch_size,
    0,
  )
  c._show_progress, c._progress_cb, c.splittable = False, None, splittable
  c.server_workers = 1
  c.profile = ModelProfile("eosfake")
  c.sampler, c._report = _Sampler(), lambda *a: None
  return c, _Sampler


def _client_run(base, slope, hard_limit, n=380, batch_size=10000, splittable=True):
  """Run the describe request loop against a fake server whose memory is base + slope * n."""
  import numpy as np
  import requests

  c, _ = _bare_client(batch_size, splittable)
  sizes, failed = [], []

  def attempt(batch, output_type):
    peak = base + slope * len(batch)
    c.sampler.last = peak
    sizes.append(len(batch))
    if peak > hard_limit:
      failed.append(len(batch))
      raise requests.HTTPError("500 out of memory")
    return np.ones((len(batch), 3)), {}

  c._attempt = attempt
  _, n_ok = c._stream_batches([f"C{i}" for i in range(n)], list(range(n)), None, [None] * n)
  return n_ok, sizes, failed


def test_fixed_request_size_and_halving_net():
  # Everything in one request when it fits.
  n_ok, sizes, failed = _client_run(base=1.0, slope=0.001, hard_limit=6.0)
  assert n_ok == 380 and sizes == [380] and not failed
  # A request too big for memory is halved until it fits; nothing is lost.
  n_ok, sizes, failed = _client_run(base=1.0, slope=0.02, hard_limit=6.0)
  assert n_ok == 380 and failed and max(sizes[len(failed) :]) <= 250


def test_projector_requests_are_never_split():
  import pytest
  import requests

  n_ok, sizes, failed = _client_run(1.0, 0.001, hard_limit=6.0, splittable=False)
  assert n_ok == 380 and sizes == [380]
  with pytest.raises(requests.HTTPError):
    _client_run(1.0, 0.1, hard_limit=6.0, splittable=False)


def test_output_format_is_learned_once():
  import numpy as np
  import requests

  from zairachem.base.utils.logging import logger
  from zairachem.describe.descriptors.api import BinaryStreamClient
  from zairachem.describe.descriptors.profiles import ModelProfile

  c = BinaryStreamClient.__new__(BinaryStreamClient)
  c.logger, c.model_id, c.profile, c.output_type = (
    logger,
    "eosfake",
    ModelProfile("eosfake"),
    "heavy",
  )
  seen = []

  def attempt(batch, output_type):
    seen.append(output_type)
    if output_type == "heavy":
      raise requests.HTTPError("500 cannot read the binary input")
    return np.ones((len(batch), 2)), {}

  c._attempt = attempt
  c._try_request(["C"] * 5)
  c._try_request(["C"] * 5)
  assert c.profile.output_type == "simple"
  assert seen == ["heavy", "simple", "simple"]


def test_cpu_budget_limits_models_running_together():
  import threading
  import time

  live, peak, lock = set(), [0], threading.Lock()

  def run(m):
    with lock:
      live.add(m)
      peak[0] = max(peak[0], len(live))
    time.sleep(0.02)
    with lock:
      live.discard(m)

  items = ["a", "b", "c", "d", "e"]
  outcome = schedule(items, run, lambda m: 0.1, 10.0, 5, cores=lambda m: 2.0, cpu_budget=4.0)
  assert all(e is None for e in outcome.values())
  assert peak[0] == 2


def test_proportional_bound_is_not_a_measurement():
  from zairachem.describe.descriptors.profiles import ModelProfile

  profile = ModelProfile("eosfake", samples=[[380, 1.5, 40.0]])
  assert profile.measured_up_to(380) and not profile.measured_up_to(10000)
  assert profile.predicted_peak(10000) > 1.5  # still reserved cautiously


def test_simulated_wall_time_follows_admission():
  from zairachem.describe.descriptors.scheduler import simulate

  secs = {"a": 10.0, "b": 10.0, "big": 5.0}
  mem = {"a": 1.0, "b": 1.0, "big": 7.0}

  def seconds(i, k):
    return secs[i] / k

  # a and b share the budget; big fits with nothing else, so it runs alone afterwards.
  assert simulate(["a", "b", "big"], seconds, mem.get, 6.5, max_running=3) == 15.0
  # Cores: one core each on a one-core budget runs them one after another.
  one_core = {"cores": lambda i: 1.0, "cpu_budget": 1.0}
  assert simulate(["a", "b"], seconds, mem.get, None, 2, **one_core) == 20.0
  # Width: a lone item may take the whole budget (2 cores, so twice as fast).
  assert simulate(["a"], seconds, mem.get, None, 1, lambda i: 1.0, 2.0, lambda i, g, c: c) == 5.0


def test_server_workers_follow_free_memory_and_cores():
  from zairachem.describe.descriptors.raw import _server_workers

  plan = {"chunked": True, "reservation_gb": 2.0, "cores": 1.0}
  assert _server_workers(plan, free_gb=6.5, free_cores=5.0) == 3  # memory-bound
  assert _server_workers(plan, free_gb=20.0, free_cores=2.0) == 2  # core-bound
  assert _server_workers(plan, free_gb=1.0, free_cores=0.5) == 1  # never below one
  assert _server_workers(plan, None, None) == 4  # no limits known: the cap
  assert _server_workers({**plan, "chunked": False}, 20.0, 20.0) == 1  # below the chunking line


def test_chunked_requests_are_recorded_per_process():
  c, _ = _bare_client()
  c.sampler.last = 3.0
  c._record(10000, 0.0)
  # One worker: memory at its chunk size (4 chunks of 2,500), time at the request's size.
  (chunk, request) = sorted(c.profile.samples)
  assert chunk[:2] == [2500, 3.0] and chunk[2] is None
  assert request[0] == 10000 and request[1] == 0.0 and request[2] is not None
  assert c.profile.predicted_peak(2500) == 3.0  # the time sample carries no memory
  # Several workers: neither is recorded, cores are per copy.
  from zairachem.describe.descriptors.profiles import ModelProfile

  c.profile, c.server_workers, c.sampler.last = ModelProfile("eosfake"), 3, 3.3
  c._record(10000, 0.0)
  assert c.profile.samples == [] and c.profile.cores == 1.0 / 3


def test_request_time_is_robust_to_contention():
  from zairachem.describe.descriptors.profiles import ModelProfile

  # One timed size: its time, proportional above it.
  assert ModelProfile("x", samples=[[380, 1.0, 76.0]]).seconds_per_request() == (0.0, 0.2)
  # A small request slowed by other models makes the line fall: ignored for the largest request.
  slowed = ModelProfile("x", samples=[[95, 1.0, 140.0], [380, 1.0, 85.0]])
  assert slowed.seconds_per_request() == (0.0, 85.0 / 380)
  # The fastest time per size is kept.
  p = ModelProfile("x")
  p.record(380, 1.0, 120.0)
  p.record(380, 1.0, 85.0)
  p.record(380, 1.0, 150.0)
  assert p.samples == [[380, 1.0, 85.0]]


def test_measured_peak_bounds_the_memory_line():
  from zairachem.describe.descriptors.profiles import ModelProfile

  # Noisy small requests tilt the line; at a measured size the measurement is the bound.
  p = ModelProfile(
    "x", samples=[[10, 3.6, None], [30, 3.0, None], [50, 3.6, None], [380, 3.76, 1.0]]
  )
  assert p.predicted_peak(380) == 3.76
  assert p.predicted_peak(100) <= 3.76
  assert p.predicted_peak(1000) >= 3.76  # above every measurement, the line
