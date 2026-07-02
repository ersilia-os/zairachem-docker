import json, os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from zairachem.base import ZairaBase
from zairachem.base.utils.matrices import Hdf5
from zairachem.base.utils.isaura_report import quiet_isaura_reads
from zairachem.base.utils.progress import LiveTableMonitor, STEP_COLORS, _bar
from zairachem.describe.descriptors.api import BinaryStreamClient
from zairachem.describe.descriptors.utils import Hdf5Data, get_model_url
from zairachem.base.vars import (
  DATA_SUBFOLDER,
  DATA_FILENAME,
  DESCRIPTORS_SUBFOLDER,
  RAW_DESC_FILENAME,
  ERSILIA_DATA_FILENAME,
)


class DescribeMonitor(LiveTableMonitor):
  """Live per-model table for the Describe step (and reused for Treat projections).

  Columns: Model | Status (queued → featurizing:<substep> → done/skipped) | Source | Time, where
  ``Source`` shows the store sourcing split ("N cached · M computed") reported by the client. Both
  the serial and parallel describe paths drive one shared instance of this table.
  """

  item_label = "Model"
  title = "Featurizing models"
  running_verb = "featurizing"

  def _columns(self, table):
    table.add_column("Source", width=30, no_wrap=True, overflow="ellipsis")
    table.add_column("Progress", width=17, no_wrap=True)
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _row_cells(self, item_id, s):
    return [s["extra"].get("source", "[dim]—[/]"), self._progress_cell(s), self._fmt_time(s)]

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
        parts.append(f"{n_cached:,} from store")
      if n_compute:
        parts.append(f"{n_compute:,} computed")
      self.update_fields(model_id, source=" [dim]·[/] ".join(parts) if parts else "[dim]—[/]")
      self.set_substep(model_id, "reading" if (n_cached and not n_compute) else "computing")
    elif kind == "batch":
      done, total = args
      self.update_fields(model_id, frac=(done / total if total else None))
      self.set_substep(model_id, f"batch {done}/{total}")


class RawLoader(ZairaBase):
  def __init__(self):
    ZairaBase.__init__(self)
    self.path = self.get_output_dir()

  def open(self, eos_id):
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, RAW_DESC_FILENAME)
    return Hdf5(path)


class RawDescriptors(ZairaBase):
  def __init__(self, batch_size=None, workers=None):
    ZairaBase.__init__(self)
    self.path = self.get_output_dir()
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
    # Preserve config order and dedup deterministically (set() order varies per process,
    # which would make the per-descriptor processing order non-reproducible).
    eos_ids = list(dict.fromkeys(self.params["featurizer_ids"]))
    return eos_ids

  def done_eos_ids(self):
    with open(os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
      done_eos_ids = json.load(f)
    return done_eos_ids

  def output_h5_filename(self, eos_id):
    path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, RAW_DESC_FILENAME)

  def _resolve_workers(self, n_models):
    """How many models to featurize concurrently. Resolution order: the explicit ``--workers`` flag →
    the ``ZAIRACHEM_DESCRIBE_WORKERS`` env var → an **auto** default derived from the host's CPU count
    (about half the cores, leaving the shared Docker VM headroom for the model servers + redis/nginx).
    Clamped to ``[1, n_models]``."""
    from zairachem.base.utils.concurrency import cpu_count, cpu_workers

    if self.workers is not None:
      return max(1, min(self.workers, max(1, n_models)))
    return cpu_workers(n_models, cap=max(2, cpu_count() // 2), env="ZAIRACHEM_DESCRIBE_WORKERS")

  def _run_eos(self, eos_id, show_progress=True, progress_cb=None):
    # One fresh client per model (no shared mutable state) so models can be featurized concurrently.
    # Each starts in 'heavy' mode with its own feature width and writes its own output H5.
    client = BinaryStreamClient(
      csv_path=self.input_csv_ersilia,
      model_id=eos_id,
      url=get_model_url(eos_id),
      project_name=os.path.basename(self.path),
    )
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
    from zairachem.base import params_path
    from zairachem.base.utils.model_version import ersilia_model_version

    versions = self.params.get("latest_featurizer_version") or {}
    changed = False
    for eos_id in eos_ids:
      if eos_id not in versions:
        versions[eos_id] = ersilia_model_version(eos_id)
        changed = True
    if changed:
      self.params["latest_featurizer_version"] = versions
      with open(params_path(self.path), "w") as f:
        json.dump(self.params, f, indent=2)

  def run(self):
    eos_ids = self.done_eos_ids() if self.is_predict() else self.eos_ids()
    self._prewarm_versions(eos_ids)
    workers = self._resolve_workers(len(eos_ids))
    # One shared live table for both paths: each model is a row (queued → featurizing → done). The
    # client reports its sourcing/batch progress into the row via the apply_event callback. The
    # table is the run's single live region, so isaura's own read-time bars are neutralized for the
    # whole block (rich permits only one live display) — covering serial store reads too, not just
    # the concurrent path.
    monitor = DescribeMonitor(eos_ids, color=STEP_COLORS.get("describe", "green"))
    with monitor.live(), quiet_isaura_reads():
      if workers == 1:
        done_eos = []
        for eos_id in eos_ids:
          monitor.start(eos_id)
          try:
            self._run_eos(eos_id, show_progress=False, progress_cb=monitor.apply_event)
            done_eos.append(eos_id)
            monitor.finish(eos_id, ok=True)
          except Exception as e:
            monitor.finish(eos_id, ok=False)
            raise Exception(f"Raw descriptor calculations failed for model {eos_id}") from e
      else:
        done_eos = self._run_parallel(eos_ids, workers, monitor)
    with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "w") as f:
      json.dump(done_eos, f, indent=4)

  def _run_parallel(self, eos_ids, workers, monitor):
    # Workers each drive their own row of the shared live table. Writes (_contribute) are serialized
    # by a lock inside BinaryStreamClient.run, so only one model touches the store at a time; isaura
    # read bars are already neutralized by the caller (run). If any model fails, raise after all have
    # settled.
    done, errors = set(), {}

    def _work(eos_id):
      monitor.start(eos_id)
      self._run_eos(eos_id, show_progress=False, progress_cb=monitor.apply_event)

    with ThreadPoolExecutor(max_workers=workers) as ex:
      futures = {ex.submit(_work, e): e for e in eos_ids}
      for fut in as_completed(futures):
        eos_id = futures[fut]
        try:
          fut.result()
          done.add(eos_id)
          monitor.finish(eos_id, ok=True)
        except Exception as e:
          errors[eos_id] = e
          monitor.finish(eos_id, ok=False)
          self.logger.error(f"[raw] {eos_id} failed: {e}")
    if errors:
      first = next(iter(errors))
      raise Exception(
        f"Raw descriptor calculations failed for {len(errors)} model(s): {sorted(errors)}"
      ) from errors[first]
    # Preserve config order for deterministic downstream consumption.
    return [e for e in eos_ids if e in done]
