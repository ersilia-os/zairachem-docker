import json
import math
import os

import pandas as pd

from zairachem.base import ZairaBase
from zairachem.base.utils.isaura_report import quiet_isaura_reads
from zairachem.base.utils.progress import STEP_COLORS, LiveTableMonitor, _bar
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  DESCRIPTORS_SUBFOLDER,
  ERSILIA_DATA_FILENAME,
  RAW_DESC_FILENAME,
)
from zairachem.describe.descriptors.api import BinaryStreamClient, DescriptorFailedError
from zairachem.describe.descriptors.utils import Hdf5Data, get_model_url


class DescribeMonitor(LiveTableMonitor):
  """Live per-model table for the Describe step (and reused for Treat projections).

  Columns: Model | Status (queued → featurizing:<substep> → done/skipped) | Source | Time, where
  ``Source`` shows the store sourcing split ("N cached · M computed") reported by the client. One
  instance shows every model (and projector) describe schedules.
  """

  item_label = "Model"
  title = "Featurizing models"
  running_verb = "featurizing"

  #: Narrowest terminal that shows the Progress bar; below it the Status column's "batch k/n" says it.
  progress_min_width = 100

  def _show_progress_bar(self):
    from zairachem.base.utils.console import console

    return console.width >= self.progress_min_width

  def _columns(self, table):
    table.add_column("Source", width=26, no_wrap=True, overflow="ellipsis")
    if self._show_progress_bar():
      table.add_column("Progress", width=15, no_wrap=True)
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _row_cells(self, item_id, s):
    progress = [self._progress_cell(s)] if self._show_progress_bar() else []
    return [s["extra"].get("source", "[dim]—[/]"), *progress, self._fmt_time(s)]

  def _progress_cell(self, s):
    """Mini bar + percentage. Done → 100%; skipped/queued/no-batches-yet → em dash."""
    if s["status"] == "done":
      frac = 1.0
    elif s["status"] == "skipped":
      return "[dim]—[/]"
    else:
      frac = s["extra"].get("frac")
    if frac is None:
      return "[dim]—[/]"
    return f"{_bar(frac, 10)} {frac * 100:.0f}%"

  def apply_event(self, model_id, kind, *args):
    """Progress callback wired into ``BinaryStreamClient._progress_cb``.

    Handles ``("plan", n_cached, n_compute)`` (sourcing split) and ``("batch", done, total)``
    (drives both the ``batch k/n`` substep and the Progress bar/percentage).
    """
    if kind == "plan":
      n_cached, n_compute = args
      parts = []
      if n_cached:
        # Short when shared with a computed count, so both fit the column.
        parts.append(f"{n_cached:,} {'store' if n_compute else 'from store'}")
      if n_compute:
        parts.append(f"{n_compute:,} computed")
      self.update_fields(model_id, source=" [dim]·[/] ".join(parts) if parts else "[dim]—[/]")
      self.set_substep(model_id, "reading" if (n_cached and not n_compute) else "computing")
    elif kind == "batch":
      done, total = args
      self.update_fields(model_id, frac=(done / total if total else None))
      self.set_substep(model_id, f"batch {done}/{total}")


class RawDescriptors(ZairaBase):
  """Describe-step computation of each featurizer's raw descriptor matrix."""

  def __init__(self, path, batch_size=None, workers=None):
    ZairaBase.__init__(self)
    self.path = path
    self.params = self._load_params()
    self.input_csv = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
    self.input_csv_ersilia = os.path.join(self.path, DATA_SUBFOLDER, ERSILIA_DATA_FILENAME)
    self._process_ersilia_inputs()
    self.batch_size = batch_size
    self.workers = workers
    if self.is_predict():
      self.trained_path = self.get_trained_dir()

  def _process_ersilia_inputs(self):
    df = pd.read_csv(self.input_csv)
    df = df["smiles"]
    df.to_csv(self.input_csv_ersilia, index=False)

  def eos_ids(self):
    """Return the configured featurizer ids, deduplicated in config order."""
    # Preserve config order and dedup deterministically (set() order varies per process,
    # which would make the per-descriptor processing order non-reproducible).
    eos_ids = list(dict.fromkeys(self.params["featurizer_ids"]))
    return eos_ids

  def output_h5_filename(self, eos_id):
    """Return ``descriptors/<eos_id>/raw.h5``, creating the descriptor folder if needed."""
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, RAW_DESC_FILENAME)

  def _resolve_workers(self, n_models):
    """How many models to featurize concurrently. Resolution order: the explicit ``--workers`` flag →
    the ``ZAIRACHEM_DESCRIBE_WORKERS`` env var → an **auto** default of the Docker VM's CPUs (the
    model servers run there, not on the host); the scheduler then admits models by their measured
    cores and memory. Clamped to ``[1, n_models]``."""
    from zairachem.base.utils.concurrency import cpu_workers, docker_cpu_count

    if self.workers is not None:
      return max(1, min(self.workers, max(1, n_models)))
    return cpu_workers(n_models, cap=docker_cpu_count(), env="ZAIRACHEM_DESCRIBE_WORKERS")

  def _drop_or_raise(self, eos_id, error):
    """Leave a featurizer that failed on too many molecules out of a fit; raise at predict.

    At predict the trained model needs every one of its descriptors, so there is nothing to drop.
    """
    if self.is_predict():
      raise Exception(f"Raw descriptor calculations failed for model {eos_id}") from error
    from zairachem.base.utils.console import echo
    from zairachem.base.utils.matrices import remove_h5
    from zairachem.describe.descriptors.compose import service_logs
    from zairachem.describe.descriptors.utils import _service_name

    remove_h5(self.output_h5_filename(eos_id))
    echo(f"Dropping {eos_id}: {error}", kind="warning")
    self.logger.warning(f"[raw] {eos_id} dropped: {error}")
    self.logger.debug(f"[raw] {eos_id} server log (tail):\n{service_logs(_service_name(eos_id))}")

  def _run_eos(self, eos_id, show_progress=True, progress_cb=None, plan=None, sampler=None):
    # One fresh client per model (no shared mutable state) so models can be featurized concurrently.
    # Each has its own feature width and writes its own output H5.
    client = BinaryStreamClient(
      path=self.path,
      csv_path=self.input_csv_ersilia,
      model_id=eos_id,
      url=get_model_url(eos_id),
      project_name=os.path.basename(self.path),
    )
    if plan is not None:
      # Each request is measured into the model's profile (see _plan).
      client.profile = plan["profile"]
      client.sampler = sampler
      client.server_workers = plan.get("server_workers", client.server_workers)
    client._show_progress = show_progress
    if progress_cb is not None:
      # The shared live table reports progress; the client must not open its own bar/prints.
      client._progress_cb = progress_cb
      client._show_progress = False
    output_h5 = self.output_h5_filename(eos_id)
    res = client.run(output_h5=output_h5, isaura_batch_size=self.batch_size)
    if res.get("h5_file"):
      self.logger.info(f"[raw] {eos_id} streamed directly to {output_h5}")
    elif res.get("data") is not None:
      Hdf5Data(res).save(output_h5)
    else:
      raise Exception(f"No descriptor data returned for model {eos_id}")

  def _prewarm_versions(self, eos_ids):
    """Resolve every featurizer's version once and persist ``parameters.json`` a SINGLE time, before
    the (possibly parallel) describe. Each ``BinaryStreamClient`` otherwise resolves its version inside
    its worker thread and rewrites the shared ``parameters.json`` — concurrent, non-atomic writes that
    could race and corrupt the file. Pre-populating it means every client finds its version already
    cached (``ersilia_model_version`` is itself memoised) and never writes during the parallel phase."""
    from zairachem.base import _atomic_json, params_path
    from zairachem.base.utils.model_version import ersilia_model_version

    versions = self.params.get("latest_featurizer_version") or {}
    missing = [e for e in eos_ids if e not in versions]
    if missing:
      from concurrent.futures import ThreadPoolExecutor

      # One Docker Hub lookup each (~0.5 s), independent of each other.
      with ThreadPoolExecutor(max_workers=min(8, len(missing))) as ex:
        versions.update(zip(missing, ex.map(ersilia_model_version, missing)))
      self.params["latest_featurizer_version"] = versions
      _atomic_json(params_path(self.path), self.params)

  def _warn_if_versions_moved(self, eos_ids):
    """At predict, warn when a featurizer's image is not the version the model was fitted with.

    A predict run copies the trained run's ``parameters.json``, so the recorded version — and with
    it the transformer the raw matrix is scaled by — is the fit's. The descriptors themselves come
    from whichever image is on this machine now. If that image has crossed a major release since,
    new columns are scaled by a transformer built for the old ones and nothing downstream can
    tell: the matrix has the right shape, the run finishes, and the predictions are quietly wrong.

    A difference in either direction is reported: an older image is as wrong as a newer one, since
    the transformer belongs to neither. A version that cannot be resolved is skipped rather than
    guessed, which is what :func:`resolved_model_version` is for — ``ersilia_model_version`` falls
    back to ``v1``, and that fallback would read as a downgrade from every later version.
    """
    from concurrent.futures import ThreadPoolExecutor

    from zairachem.base.utils.model_version import resolved_model_version

    fitted = self.params.get("latest_featurizer_version") or {}
    known = [e for e in eos_ids if fitted.get(e)]
    if not known:
      return
    # One DockerHub lookup each and independent of one another, as _prewarm_versions resolves them.
    with ThreadPoolExecutor(max_workers=min(8, len(known))) as ex:
      current = dict(zip(known, ex.map(resolved_model_version, known)))
    moved = []
    for eos_id in known:
      was, now = fitted[eos_id], current[eos_id]
      if now and now != was:
        moved.append(f"{eos_id} {was}\u2192{now}")
    if moved:
      self.logger.warning(
        f"[raw] the local image is not the one this model was fitted with ({', '.join(moved)}); "
        "these molecules are featurized with it but scaled with the transformer of the fitted "
        "version, so the predictions may not mean what they say. Re-fit the model, or pull the "
        "image the fit used."
      )

  def run(self):
    """Featurize the run's molecules with each featurizer into ``descriptors/<eos_id>/raw.h5``.

    At fit every configured featurizer is computed; at predict only those the trained model uses.
    Models (and the projection models) run side by side as the Docker VM's memory and CPU allow,
    under one live table; the descriptors computed are listed in ``descriptors/done_eos.json``. A
    featurizer over its failure budget is dropped at fit; any other failure raises.
    """
    if self.is_predict():
      # Featurize only the descriptors the trained pooled model uses (the --max-descriptors
      # selection when it was screened, less those the pooling left out).
      from zairachem.base.utils.descriptors import pooled_descriptors

      eos_ids = pooled_descriptors(self.trained_path)
    else:
      eos_ids = self.eos_ids()
    # Projection models run alongside the descriptor models (their coordinates are cached for the
    # projections step), so their time overlaps describe's instead of adding to it.
    projector_ids = [p for p in (self.params.get("projection_ids") or []) if p not in eos_ids]
    self._prewarm_versions(eos_ids + projector_ids)
    if self.is_predict():
      self._warn_if_versions_moved(eos_ids)
    workers = self._resolve_workers(len(eos_ids) + len(projector_ids))
    # One live table: each model is a row (queued → featurizing → done). The client reports its
    # sourcing/batch progress into the row via the apply_event callback. The table is the run's
    # single live region, so isaura's own read-time bars are neutralized for the whole block (rich
    # permits only one live display).
    plans, budget = self._plan(eos_ids + projector_ids, projector_ids)
    monitor = DescribeMonitor(eos_ids + projector_ids, color=STEP_COLORS.get("describe", "green"))
    from zairachem.describe.descriptors.scheduler import StatsMonitor

    try:
      with monitor.live(), quiet_isaura_reads():
        done_eos = self._run_scheduled(eos_ids, workers, monitor, plans, budget, projector_ids)
    finally:
      StatsMonitor.stop()
    if not done_eos:
      raise Exception("Every featurizer failed; there are no descriptors to train on.")
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "w") as f:
      json.dump(done_eos, f, indent=4)

  def _plan(self, eos_ids, projector_ids=()):
    """Each model's memory reservation, cores and expected time, printed before describe.

    Built from the model's profile on this machine (image-digest matched), its Ersilia metadata
    (``Computational Performance``, a time prior for unprofiled models) and the Docker VM's memory.
    Warns about a model too big for the VM before any request is sent.
    """
    from zairachem.base.utils.concurrency import docker_cpu_count, docker_memory_gb
    from zairachem.base.utils.console import echo
    from zairachem.base.utils.model_version import _local_image_digest
    from zairachem.base.utils.utils import fetch_model_metadata, fetch_schema_from_github
    from zairachem.base.vars import (
      DEFAULT_DESCRIBE_CORES,
      DESCRIBE_BATCH_SIZE,
      DESCRIBE_MEMORY_RESERVE_GB,
      MAX_PROJECTION_MOLECULES,
      SERVER_CHUNKING_MOLECULES,
      SERVER_CHUNKS_PER_WORKER,
      SERVERS_PROJECT_NAME,
    )
    from zairachem.describe.descriptors import profiles, scheduler
    from zairachem.describe.descriptors.utils import _service_name

    vm = docker_memory_gb()
    budget = scheduler.memory_budget(vm)
    # Every running container holds memory now, this run's idle servers included (they are all up);
    # a model then needs only its peak above its own server's idle memory.
    usage = scheduler.containers_memory_gb()
    idle = {e: sum(gb for name, gb in usage.items() if _service_name(e) in name) for e in eos_ids}
    servers_gb = sum(gb for name, gb in usage.items() if SERVERS_PROJECT_NAME in name)
    vm_budget = None  # what the models could share with no warm server holding memory
    if budget is not None:
      vm_budget = max(0.0, budget - (sum(usage.values()) - servers_gb))
      budget = max(0.0, budget - sum(usage.values()))
    n_mol = len(self._input_smiles())
    n_request = max(1, min(n_mol, DESCRIBE_BATCH_SIZE))
    plans, tight = {}, []
    # Metadata (GitHub) and image digests are independent per model: fetch them all at once.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(eos_ids)))) as ex:
      digests = dict(zip(eos_ids, ex.map(_local_image_digest, eos_ids)))
      loaded = {e: profiles.load(e, digests[e]) for e in eos_ids}
      # The metadata is only a time prior, for models not yet timed on this machine.
      untimed = [e for e in eos_ids if loaded[e].seconds_per_request() is None]
      metas = dict(zip(untimed, ex.map(lambda e: fetch_model_metadata(e) or {}, untimed)))
      list(ex.map(fetch_schema_from_github, eos_ids))  # cached per process for the clients
    chunked = n_request >= SERVER_CHUNKING_MOLECULES
    # A chunked request's memory is that of one chunk's process, per server worker.
    n_memory = math.ceil(n_request / SERVER_CHUNKS_PER_WORKER) if chunked else n_request
    for eos_id in eos_ids:
      meta = metas.get(eos_id, {})
      profile = loaded[eos_id]
      if eos_id in projector_ids:
        # One request, never chunked, of at most MAX_PROJECTION_MOLECULES (see Manifolds).
        n_mol_e = n_req_e = n_mem_e = min(n_mol, MAX_PROJECTION_MOLECULES)
      else:
        n_mol_e, n_req_e, n_mem_e = n_mol, n_request, n_memory
      plans[eos_id] = {
        "profile": profile,
        "reservation_gb": max(
          MIN_RESERVATION_GB, profiles.reservation_gb(profile, n_mem_e, budget, idle[eos_id])
        ),
        "cores": min(float(docker_cpu_count()), profile.cores or DEFAULT_DESCRIBE_CORES),
        "seconds": _expected_seconds(profile, meta, n_mol_e, n_req_e),
        # Requests the server cuts into chunks, whose copies it can run side by side.
        "chunked": chunked and eos_id not in projector_ids,
      }
      # Warn only on a measured prediction; a proportional bound (e.g. a profile measured on a few
      # hundred molecules, for a 10,000-molecule request) only makes the reservation cautious.
      measured = profile.measured_up_to(n_mem_e)
      peak = profile.predicted_peak(n_mem_e) if measured else None
      if peak is not None and vm_budget is not None and peak > vm_budget:
        echo(
          f"{eos_id} is predicted to need ~{peak:.1f} GB for requests of {n_request:,} "
          f"molecules, more than the Docker VM can give it ({vm_budget:.1f} GB); its requests may "
          f"fail and be split. Give Docker ≥ {peak + DESCRIBE_MEMORY_RESERVE_GB:.0f} GB to avoid it.",
          kind="warning",
        )
      elif peak is not None and budget is not None and peak - idle[eos_id] > budget:
        tight.append(eos_id)
    if tight:
      echo(
        f"Idle model servers hold {servers_gb:.1f} GB of the Docker VM's memory, so "
        f"{', '.join(tight)} will run alone; `zairachem stop` frees it.",
        kind="info",
      )
    known = {e: p["seconds"] for e, p in plans.items() if p["seconds"] is not None}
    wall = None
    if known:
      # The scheduler's own admission (memory, cores, longest first) on a simulated clock.
      wall = scheduler.simulate(
        sorted(known, key=lambda e: -known[e]),
        lambda e, k: (
          known[e] * (SERVER_WORKERS_SERIAL_FRACTION + (1 - SERVER_WORKERS_SERIAL_FRACTION) / k)
        ),
        lambda e: plans[e]["reservation_gb"],
        budget,
        self._resolve_workers(len(eos_ids)),
        cores=lambda e: plans[e]["cores"],
        cpu_budget=_cpu_budget(docker_cpu_count()),
        width=lambda e, free_gb, free_cores: _server_workers(plans[e], free_gb, free_cores),
      )
    if wall is None or wall > DOCKER_SHARE_HINT_S:
      _note_docker_share(vm)  # more Docker resources only matter when describe takes a while
    if wall is not None:
      slowest = max(known, key=known.get)
      longest = f"; the longest, {slowest}, ≈ {_minutes(known[slowest])}" if len(known) > 1 else ""
      echo(f"Expected describe time ≈ {_minutes(wall)}{longest}.", kind="info")
    return plans, budget

  def _input_smiles(self):
    with open(self.input_csv_ersilia) as f:
      return [line for line in f.read().splitlines()[1:] if line]

  def _featurize_on_server(self, eos_id, service, plans, monitor):
    """Start ``eos_id``'s server if needed and featurize with it, measuring into its profile."""
    from zairachem.describe.descriptors import compose, profiles, scheduler

    if not compose.up([service]):
      raise RuntimeError(f"Could not start the {eos_id} model server.")
    sampler = scheduler.PeakSampler(compose.container_id(service))
    plan = plans.get(eos_id)
    try:
      with sampler:
        self._run_eos(
          eos_id,
          show_progress=False,
          progress_cb=monitor.apply_event,
          plan=plan,
          sampler=sampler,
        )
    finally:
      if plan is not None:
        profiles.save(plan["profile"])

  def _project_on_server(self, model_id, service, monitor):
    """Start a projection model's server if needed and cache its coordinates for this run."""
    from zairachem.describe.descriptors import compose, scheduler
    from zairachem.treat.imputers.manifolds import Manifolds

    if not compose.up([service]):
      raise RuntimeError(f"Could not start the {model_id} model server.")
    with scheduler.PeakSampler(compose.container_id(service)) as sampler:
      Manifolds(path=self.path).precompute(
        model_id, progress_cb=monitor.apply_event, sampler=sampler
      )

  def _run_scheduled(self, eos_ids, workers, monitor, plans, budget, projector_ids=()):
    """Featurize ``eos_ids`` side by side while their servers' memory fits in the Docker VM.

    Each model's server is started (if not already up) when the model is admitted and left running
    for later runs; a run holds the model's lock while it uses the server, so concurrent runs never
    share one at once. Its requests are measured into its profile (see
    :mod:`~zairachem.describe.descriptors.profiles`). Store writes are serialized inside
    ``BinaryStreamClient``. A model over its failure budget is dropped at fit; any other failure
    raises once every model has settled.
    """
    from zairachem.describe.descriptors import compose, scheduler
    from zairachem.describe.descriptors.utils import _service_name

    def _work(eos_id):
      monitor.start(eos_id)
      service = _service_name(eos_id)
      try:
        with compose.locked(eos_id):  # one run at a time on a shared server
          if eos_id in projector_ids:
            self._project_on_server(eos_id, service, monitor)
          else:
            self._featurize_on_server(eos_id, service, plans, monitor)
      except Exception:
        monitor.finish(eos_id, ok=False)
        raise
      monitor.finish(eos_id, ok=True)

    def _cost(eos_id):
      return plans[eos_id]["reservation_gb"]

    # Longest first, so the slowest models start early; done_eos keeps the config order.
    jobs = list(eos_ids) + list(projector_ids)
    order = sorted(jobs, key=lambda e: -(plans[e]["seconds"] or 0.0))
    from zairachem.base.utils.concurrency import docker_cpu_count

    def _cores(eos_id):
      return plans[eos_id]["cores"]

    def _width(eos_id, free_gb, free_cores):
      # Copies of the model its server may run for a chunked request; it takes that many times its
      # memory and cores from the scheduler (see _server_workers).
      plans[eos_id]["server_workers"] = _server_workers(plans[eos_id], free_gb, free_cores)
      return plans[eos_id]["server_workers"]

    outcome = scheduler.schedule(
      order,
      _work,
      _cost,
      budget,
      workers,
      cores=_cores,
      cpu_budget=_cpu_budget(docker_cpu_count()),
      width=_width,
    )
    for pid in projector_ids:
      if outcome.get(pid) is not None:  # the projections step computes it again
        self.logger.warning(f"[raw] projection {pid} failed during describe: {outcome[pid]}")
    done, errors = set(), {}
    for eos_id in eos_ids:
      e = outcome.get(eos_id)
      if e is None:
        done.add(eos_id)
      elif isinstance(e, DescriptorFailedError):
        self._drop_or_raise(eos_id, e)
      else:
        errors[eos_id] = e
        self.logger.error(f"[raw] {eos_id} failed: {e}")
    if errors:
      from zairachem.base.utils.console import plural

      first = next(iter(errors))
      raise Exception(
        f"Raw descriptor calculations failed for {plural(len(errors), 'model')}: {sorted(errors)}"
      ) from errors[first]
    # Preserve config order for deterministic downstream consumption.
    return [e for e in eos_ids if e in done]


#: Least memory (GB) reserved for a model, whatever its server already holds while idle.
MIN_RESERVATION_GB = 0.1

#: Share of a chunked request's time that more server workers do not shorten (model loading and
#: the server's own overhead): a request takes ``t * (f + (1 - f) / k)`` with ``k`` workers. Measured:
#: 10,000 molecules took 672 s with one worker and 344 s with three.
SERVER_WORKERS_SERIAL_FRACTION = 0.25


def _server_workers(plan, free_gb, free_cores):
  """Server workers for a model starting now: 1, or more for chunked requests, as far as the free
  memory and cores hold that many times the model's reservation and cores (at most
  ``MAX_DESCRIBE_SERVER_WORKERS``)."""
  from zairachem.base.vars import DESCRIBE_SERVER_WORKERS, MAX_DESCRIBE_SERVER_WORKERS

  if not plan.get("chunked"):
    return DESCRIBE_SERVER_WORKERS
  k = MAX_DESCRIBE_SERVER_WORKERS
  if free_gb is not None and plan["reservation_gb"] > 0:
    k = min(k, int(free_gb // plan["reservation_gb"]))
  if free_cores is not None and plan["cores"] > 0:
    k = min(k, int(free_cores // plan["cores"]))
  return max(DESCRIBE_SERVER_WORKERS, k)


#: Expected describe seconds above which the "give Docker more CPUs/memory" hint is shown.
DOCKER_SHARE_HINT_S = 60


def _float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _minutes(seconds):
  return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def _expected_seconds(profile, meta, n_mol, n):
  """Expected seconds to describe ``n_mol`` molecules with a model, or None if nothing is known.

  From the model's own measured requests when it has them; otherwise from its Ersilia metadata
  (``Computational Performance`` 1–5: seconds for 1 to 10,000 molecules, measured by ersilia test).
  """
  timing = profile.seconds_per_request()
  if timing is not None:
    fixed, per_mol = timing
  else:
    p1, p4, p5 = (_float(meta.get(f"Computational Performance {i}")) for i in (1, 4, 5))
    if p1 is None or p4 is None or p5 is None:
      return None
    fixed, per_mol = p1, max(0.0, (p5 - p4) / 9000)
  return math.ceil(n_mol / n) * fixed + n_mol * per_mol


def _note_docker_share(vm_memory_gb):
  """Say so when Docker runs with clearly less CPU or memory than the machine has.

  Model servers run inside Docker, so describe gets faster with what Docker Desktop is given
  (Settings → Resources); on Linux Docker uses the whole machine and this stays silent.
  """
  from zairachem.base.utils.concurrency import cpu_count, docker_cpu_count, host_memory_gb
  from zairachem.base.utils.console import echo

  host_cpus, docker_cpus, host_mem = cpu_count(), docker_cpu_count(), host_memory_gb()
  low_cpu = docker_cpus < 0.75 * host_cpus
  low_mem = bool(vm_memory_gb and host_mem and vm_memory_gb < 0.6 * host_mem)
  if low_cpu or low_mem:
    echo(
      f"Docker has {docker_cpus} of {host_cpus} CPUs and {vm_memory_gb or 0:.0f} of "
      f"{host_mem or 0:.0f} GB; more would speed up describe (Docker settings).",
      kind="info",
    )


def _cpu_budget(vm_cpus):
  """Cores the running models may keep busy together: the VM's CPUs times
  ``ZAIRACHEM_DESCRIBE_CPU_FACTOR`` (default 1.0; >1 lets model loading, partly disk-bound,
  overlap other models' computing)."""
  try:
    factor = float(os.environ.get("ZAIRACHEM_DESCRIBE_CPU_FACTOR", "1.0"))
  except ValueError:
    factor = 1.0
  return max(1.0, vm_cpus * factor)
