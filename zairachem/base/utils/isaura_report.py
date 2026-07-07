"""End-of-setup report: per-model coverage of the current input against the isaura store.

Informational only — never raises. If isaura is not installed or its engine (MinIO) is not
reachable, prints a short note and returns. Queries only the configured store (the bucket passed
in, i.e. ``params["read_store"]``).
"""

import contextlib
import json
import os
import re
import socket
from urllib.parse import urlparse

from zairachem.base.utils.console import active_color, console, echo
from zairachem.base.utils.logging import logger
from zairachem.base.utils.utils import get_bucket_records
from zairachem.base.utils.model_version import ersilia_model_version
from zairachem.base.utils.progress import LiveTableMonitor
from zairachem.base.vars import DEFAULT_ISAURA_BUCKET


class _NullProgress:
  """No-op stand-in for isaura's ``ReadProgress`` (matches its ``with ... as p`` + ``p.update()``
  interface). Only one rich live display may be active at a time, so when we drive our own themed
  table (describe/treat/migration) we neutralize isaura's read bars and render nothing for them."""

  def __init__(self, *a, **k):
    pass

  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False

  def update(self, *a, **k):
    pass


@contextlib.contextmanager
def quiet_isaura_reads():
  """Fully silence isaura's own terminal output for the duration, so it can't collide with our themed
  live tables (rich allows one live display at a time, and any stray print corrupts the region — e.g. a
  repeated table title).

  isaura writes through its **own** rich Console(s), which bypass our live region: ``ReadProgress`` and
  ``console.status`` spinners, direct ``console.print`` calls (progress bars, the "✓ N molecules
  written" line), and INFO logs via its loguru handler. We null the progress/status shims AND set
  rich's ``quiet`` flag on every isaura console so nothing leaks. Patched once around the work (not per
  thread) to avoid monkeypatch races; best-effort and restored on exit. Anything that actually matters
  (e.g. a failed contribute) is surfaced by zairachem's own console/logger, not isaura's."""
  try:
    import isaura.logging as _ilog
    import isaura.manage as _im
    from isaura.logging import logger as _il
  except Exception:
    yield
    return
  isaura_console = getattr(_il, "console", None)
  orig_rp = getattr(_im, "ReadProgress", None)
  # `status` is normally a class method (not an instance attribute); track whether the console already
  # had its own override so we can restore exactly — deleting our shim if there was none.
  had_status_attr = isaura_console is not None and "status" in vars(isaura_console)
  orig_status_attr = vars(isaura_console).get("status") if isaura_console is not None else None
  # Every distinct isaura console: the module-level one used for console.print (writer "✓" line,
  # progress) and the loguru handler's console. Mute both via rich's quiet flag; restore prior values.
  quieted = []
  seen_ids = set()
  for c in (getattr(_ilog, "console", None), isaura_console):
    if c is not None and hasattr(c, "quiet") and id(c) not in seen_ids:
      seen_ids.add(id(c))
      quieted.append((c, c.quiet))
  try:
    if orig_rp is not None:
      _im.ReadProgress = _NullProgress
    if isaura_console is not None:
      isaura_console.status = lambda *a, **k: contextlib.nullcontext()
    for c, _ in quieted:
      c.quiet = True
    yield
  finally:
    if orig_rp is not None:
      _im.ReadProgress = orig_rp
    if isaura_console is not None:
      if had_status_attr:
        isaura_console.status = orig_status_attr
      else:
        with contextlib.suppress(AttributeError):
          del isaura_console.status
    for c, prev in quieted:
      c.quiet = prev


def sanitize_project_name(name):
  """Turn a model-folder name into a valid isaura/S3 bucket name.

  S3/MinIO bucket names must be 3-63 chars, lowercase, and contain only letters, digits, hyphens
  and dots, starting and ending with an alphanumeric. Folder names often have uppercase or
  underscores (e.g. ``model_e2e2``), which MinIO rejects — so normalize deterministically.
  """
  s = re.sub(r"[^a-z0-9.-]", "-", str(name).lower())
  s = re.sub(r"-{2,}", "-", s).strip("-.")
  if len(s) < 3:
    s = f"{s}-zc" if s else "zc-project"
  return s[:63]


def _endpoint_running(endpoint, timeout=0.25):
  # Quick TCP probe of a MinIO endpoint so we never block on a slow/unreachable isaura call.
  parsed = urlparse(endpoint)
  host = parsed.hostname or "127.0.0.1"
  port = parsed.port or (443 if parsed.scheme == "https" else 9000)
  try:
    with socket.create_connection((host, port), timeout=timeout):
      return True
  except OSError:
    return False


def _engine_running():
  # Local MinIO engine (the lake / project stores live here).
  return _endpoint_running(os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000"))


# Sentinel for a store cell that could not be queried (engine down, or per-model error),
# as opposed to a successful query that simply found 0 compounds.
_UNAVAILABLE = None


def _found_local(bucket, model_id, version, smiles, store=None):
  """The subset of `smiles` cached in a *local* MinIO bucket (lake or project).

  A lightweight **inspect** (Bloom-filter membership via ``IsauraChecker.seen_many``) — it never
  fetches descriptor data. Pass a shared ``store`` to avoid reconstructing a MinIO connection on
  every call. Returns the set of found SMILES, or ``_UNAVAILABLE`` if the bucket couldn't be queried.
  Note: the Bloom filter can over-count slightly (false positives), never under-count.
  """
  try:
    from isaura.manage import IsauraChecker

    with IsauraChecker(bucket, model_id, version, store=store) as checker:
      seen = checker.seen_many(smiles)
    return {s for s, v in seen.items() if v[0]}
  except Exception as e:
    logger.debug("Store availability check failed for %s in %s (%s)", model_id, bucket, e)
    return _UNAVAILABLE


def report_store_availability(
  featurizer_ids, projection_ids, smiles, project=None, lake_bucket=DEFAULT_ISAURA_BUCKET
):
  """Show, per model, where the input compounds already live across the isaura stores.

  Purely informational — never raises and never fetches data (a lightweight Bloom-filter *inspect* of
  each store's membership index). Each model gets one **stacked coverage bar**: green = already in the
  project store, cyan = additionally in the local lake (beyond the project), dim = neither (must be
  computed). The right column is the compound count still to compute. The remote (cloud) store is not
  checked here.

  Parameters
  ----------
  featurizer_ids, projection_ids : list of str
      Descriptor and projector model IDs.
  smiles : list of str
      The run's (standardized) input SMILES to look up.
  project : str or None
      The run's project bucket name, if a read/write project store is configured.
  lake_bucket : str
      The local lake bucket (defaults to ``isaura-public``).
  """
  lake_bucket = lake_bucket or DEFAULT_ISAURA_BUCKET
  try:
    import isaura.manage  # noqa: F401  (presence check)
  except Exception:
    echo("Compound availability: unavailable (isaura not installed).", kind="warning")
    return
  if not _engine_running():
    echo(
      "Local isaura engine not running — skipping compound availability "
      "(start it with: isaura engine --start).",
      kind="warning",
    )
    return

  from rich import box
  from rich.table import Table

  total = len(smiles)
  width = 20
  color = active_color()
  smiles_list = list(smiles)
  # Bloom-filter inspects only — silence isaura's own read bars/spinners so nothing flashes; we render
  # one clean themed table at the end.
  with quiet_isaura_reads():
    try:
      from isaura.base import MinioStore

      local_store = MinioStore()
    except (SystemExit, Exception):
      # MinioStore.__init__ may sys.exit on an unhealthy endpoint; treat as "engine down".
      echo("Compound availability: unavailable (isaura engine unhealthy).", kind="warning")
      return

    table = Table(
      title="Compound availability across stores",
      caption=f"[green]█ project[/]  [cyan]█ lake[/]  [dim]█ to compute[/]   ·   {total:,} compounds",
      caption_justify="left",
      title_style=f"bold {color}",
      title_justify="left",
      box=box.SIMPLE_HEAD,
      border_style=color,
      header_style=f"bold {color}",
      pad_edge=False,
    )
    table.add_column("Kind", style="dim")
    table.add_column("Model", style="bold green")
    table.add_column("Coverage", no_wrap=True)
    table.add_column("To compute", justify="right")

    for kind, ids in (("descriptor", featurizer_ids), ("projector", projection_ids)):
      for m in ids:
        try:
          version = ersilia_model_version(m)
        except Exception:
          table.add_row(kind, m, "[dim]n/a[/]", "[dim]n/a[/]")
          continue
        proj_found = (
          _found_local(project, m, version, smiles_list, store=local_store) if project else set()
        )
        lake_found = _found_local(lake_bucket, m, version, smiles_list, store=local_store)
        if proj_found is _UNAVAILABLE or lake_found is _UNAVAILABLE:
          table.add_row(kind, m, "[dim]unavailable[/]", "[dim]—[/]")
          continue
        n_proj = len(proj_found)
        n_lake_extra = len(lake_found - proj_found)
        to_compute = total - len(proj_found | lake_found)
        pc = round(n_proj / total * width) if total else 0
        lc = round(n_lake_extra / total * width) if total else 0
        lc = min(lc, width - pc)  # never exceed the bar width
        dc = max(0, width - pc - lc)
        bar = f"[green]{'█' * pc}[/][cyan]{'█' * lc}[/][dim]{'█' * dc}[/]"
        pct = (100 * to_compute / total) if total else 0
        compute_cell = (
          f"[yellow]{to_compute:,}[/] [dim]({pct:.0f}%)[/]" if to_compute else "[dim]0[/]"
        )
        table.add_row(kind, m, bar, compute_cell)

  console.print(table)


def _list_project_names():
  """Existing isaura project/bucket names as a set, or None if it can't be determined."""
  try:
    import isaura  # noqa: F401  (presence check)
  except Exception:
    return None
  if not _engine_running():
    return None
  try:
    from isaura.base import MinioStore
    from isaura.const import MINIO_ENDPOINT, MINIO_LOCAL_AK, MINIO_LOCAL_SK

    store = MinioStore(endpoint=MINIO_ENDPOINT, access=MINIO_LOCAL_AK, secret=MINIO_LOCAL_SK)
    return {b.get("Name") for b in store.client.list_buckets().get("Buckets", [])}
  except (SystemExit, Exception):
    # MinioStore.__init__ may sys.exit on an unhealthy endpoint; treat any failure as "can't verify".
    return None


def _stored_versions(bucket):
  # Map model_id -> set of version strings present in the store.
  out = {}
  for r in get_bucket_records(bucket):
    name = r.get("model", "")
    if "/" in name:
      mid, ver = name.split("/", 1)
      out.setdefault(mid, set()).add(ver.split("/")[0])
  return out


def check_isaura_version_consistency(bucket, featurizer_ids, projection_ids):
  """Stop the run if the isaura store holds data for a model under a different major version.

  Compares each model's Model-Hub version (``ersilia_model_version``) against the version(s) the
  store already holds for it. A model whose store has data under a *different* version means those
  precalculations were built by another model build — using them would be wrong. Raises
  ``SystemExit(1)`` on any mismatch. Skips quietly if the store can't be read (isaura/engine
  unavailable) — we never block on inability to check.
  """
  bucket = bucket or DEFAULT_ISAURA_BUCKET
  try:
    import isaura.manage  # noqa: F401  (presence check)
  except Exception:
    return
  if not _engine_running():
    return
  try:
    with quiet_isaura_reads():
      stored = _stored_versions(bucket)
  except Exception as e:
    logger.debug(
      "Could not read stored versions from %s; skipping consistency check (%s)", bucket, e
    )
    return

  mismatches = []
  for m in list(featurizer_ids) + list(projection_ids):
    have = stored.get(m)
    if have and ersilia_model_version(m) not in have:
      mismatches.append((m, ersilia_model_version(m), sorted(have)))
  if not mismatches:
    return

  from rich import box
  from rich.table import Table

  table = Table(
    title=f"⚠ Isaura version mismatch · {bucket}",
    title_style="bold red",
    title_justify="left",
    box=box.ROUNDED,
    border_style="red",
    header_style="bold red",
  )
  table.add_column("Model", style="bold")
  table.add_column("Model Hub", style="green")
  table.add_column("Isaura", style="yellow")
  for m, mh, have in mismatches:
    table.add_row(m, mh, ", ".join(have))
  console.print(table)
  echo(
    f"{len(mismatches)} model(s) have isaura precalculations from a different version — ZairaChem cannot continue.",
    kind="error",
  )
  echo(
    "Use a matching model image, or clear/rebuild those store versions, then re-run.", kind="info"
  )
  raise SystemExit(1)


def project_exists(name):
  """True/False whether an isaura project (bucket) exists; None if it can't be determined."""
  names = _list_project_names()
  return None if names is None else (name in names)


def _write_smiles_csv(smiles):
  import tempfile

  import pandas as pd

  f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
  pd.DataFrame({"input": list(smiles)}).to_csv(f.name, index=False)
  f.close()
  return f.name


class MigrationMonitor(LiveTableMonitor):
  """Live per-model table for the lake → project migration (Setup), styled like the other steps.

  Columns: Model | Status (checking → reading → writing → done) | Migrated | Time.
  """

  item_label = "Model"
  running_verb = "migrating"

  def __init__(self, model_ids, project, color="cyan"):
    super().__init__(model_ids, color=color)
    self.title = f"Migrating isaura-public → {project}"

  def _columns(self, table):
    table.add_column("Migrated", justify="right", width=12, no_wrap=True)
    table.add_column("Time", justify="right", width=8, no_wrap=True)

  def _row_cells(self, item_id, s):
    return [s["extra"].get("migrated", "[dim]—[/]"), self._fmt_time(s)]


def create_and_migrate_project(project, featurizer_ids, projection_ids, smiles, output_dir=None):
  """Create the run's isaura project (public) and migrate matching precalcs from the lake into it.

  Runs at fit setup when write is enabled: creates the bucket named like the model folder, then per
  model copies the input molecules already present in isaura-public into the project. Skips with a
  warning if isaura/engine is unavailable; per-model failures are reported as '—'. Never raises.

  Only what the project is *missing* is migrated, so on a re-run (project already populated) this is a
  no-op — the project is then self-sufficient and the lake is not consulted again.
  """
  try:
    import isaura  # noqa: F401  (presence check)
  except Exception:
    echo("Isaura not installed — skipping project create/migrate.", kind="warning")
    return
  if not _engine_running():
    echo(
      "Isaura engine not running — cannot create the project (describe/treat will fail).",
      kind="warning",
    )
    return

  import json
  import os
  import tempfile

  from isaura.base import MinioStore
  from isaura.const import ACCESS_FILE, MINIO_ENDPOINT, MINIO_LOCAL_AK, MINIO_LOCAL_SK
  from isaura.manage import IsauraChecker, IsauraReader, IsauraWriter

  # Create the project bucket (public) and record its access level.
  try:
    store = MinioStore(endpoint=MINIO_ENDPOINT, access=MINIO_LOCAL_AK, secret=MINIO_LOCAL_SK)
    store.ensure_bucket(project)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
      json.dump({"access": "public"}, f)
      access_path = f.name
    store.upload_file(access_path, project, ACCESS_FILE)
    os.unlink(access_path)
  except (SystemExit, Exception):
    echo(f"Could not create isaura project '{project}'.", kind="error")
    return

  model_ids = list(featurizer_ids) + list(projection_ids)

  # Phase 1 — cheap Bloom inspects (isaura's bars silenced) to find, per model, what the lake holds
  # that the project is *missing*. No display; lets us skip the migration table entirely when there
  # is nothing to bring over (e.g. a re-run where the project is already fully populated).
  to_migrate = {}  # model -> (version, [smiles to migrate])  | (None, None) marks a per-model error
  with quiet_isaura_reads():
    for m in model_ids:
      try:
        version = ersilia_model_version(m)
        with IsauraChecker(DEFAULT_ISAURA_BUCKET, m, version) as checker:
          seen = checker.seen_many(smiles)
        present = [s for s in smiles if seen.get(s, [False, None])[0]]
        if present:
          try:
            with IsauraChecker(project, m, version) as proj_checker:
              proj_seen = proj_checker.seen_many(present)
            present = [s for s in present if not proj_seen.get(s, [False, None])[0]]
          except Exception:
            pass  # can't read the project — fall back to migrating all (the writer still dedups)
        to_migrate[m] = (version, present)
      except Exception as e:
        logger.debug("Migration pre-check failed for %s (%s)", m, e)
        to_migrate[m] = (None, None)

  if not any(present for _v, present in to_migrate.values()):
    if output_dir is not None:
      for m in model_ids:
        record_migration(output_dir, m, 0)
    echo(f"Project '{project}' already has all cached compounds — nothing to migrate.", kind="info")
    return

  # Phase 2 — read the missing compounds from the lake and write them into the project, per model,
  # under a themed live table (isaura's own read bars stay silenced so nothing flashes).
  monitor = MigrationMonitor(model_ids, project, color=active_color())
  with monitor.live(), quiet_isaura_reads():
    for m in model_ids:
      monitor.start(m)
      version, present = to_migrate[m]
      migrated = "—"
      try:
        if version is None:
          monitor.update_fields(m, migrated="[dim]—[/]")
          monitor.finish(m, ok=False)
          if output_dir is not None:
            record_migration(output_dir, m, 0)
          continue
        if not present:
          migrated = "0"
        else:
          monitor.set_substep(m, f"reading {len(present):,}")
          in_csv = _write_smiles_csv(present)
          try:
            df = IsauraReader(
              model_id=m,
              model_version=version,
              input_csv=in_csv,
              approximate=False,
              bucket=DEFAULT_ISAURA_BUCKET,
            ).read()
          finally:
            os.unlink(in_csv)
          n = len(df) if df is not None else 0
          if n:
            monitor.set_substep(m, f"writing {n:,}")
            IsauraWriter(input_csv=None, model_id=m, model_version=version, bucket=project).write(
              df=df
            )
          migrated = f"{n:,}"
        if output_dir is not None:
          record_migration(
            output_dir, m, 0 if migrated in ("—", "0") else int(migrated.replace(",", ""))
          )
        monitor.update_fields(m, migrated=("[dim]0[/]" if migrated == "0" else migrated))
        monitor.finish(m, ok=True)
      except Exception:
        monitor.update_fields(m, migrated="[dim]—[/]")
        monitor.finish(m, ok=False)


# --- Per-model data provenance: read-from-project vs computed-via-Ersilia (cache miss) ---


def _provenance_path(output_dir):
  from zairachem.base.vars import METADATA_SUBFOLDER, PROVENANCE_FILENAME

  return os.path.join(output_dir, METADATA_SUBFOLDER, PROVENANCE_FILENAME)


def _load_provenance(output_dir):
  path = _provenance_path(output_dir)
  if os.path.exists(path):
    try:
      with open(path) as f:
        return json.load(f)
    except Exception:
      pass
  return {"featurizers": {}, "projections": {}, "migrated": {}}


def _write_provenance(output_dir, data):
  path = _provenance_path(output_dir)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(data, f, indent=2)


def record_migration(output_dir, model_id, n):
  """Record how many molecules were migrated from the lake into the project for a model (B)."""
  data = _load_provenance(output_dir)
  data.setdefault("migrated", {})[model_id] = int(n)
  _write_provenance(output_dir, data)


def record_provenance(
  output_dir, kind, model_id, n_total, n_from_project, n_computed, elapsed_seconds=None
):
  """Record a model's describe/treat provenance. kind is 'featurizers' or 'projections'.

  ``elapsed_seconds`` (when given) is the wall-clock this model's run took, surfaced as the report's
  per-model timing.
  """
  data = _load_provenance(output_dir)
  entry = {
    "n_total": int(n_total),
    "n_from_project": int(n_from_project),
    "n_computed": int(n_computed),
  }
  if elapsed_seconds is not None:
    entry["seconds"] = round(float(elapsed_seconds), 2)
  data.setdefault(kind, {})[model_id] = entry
  _write_provenance(output_dir, data)


def report_data_provenance(output_dir):
  """Render per-model stacked bars of where each model's data came from this run.

  By the time Describe runs, any lake→project migration has already happened, so every row is either
  **read from the project store** or **computed via Ersilia** (a cache miss). The bar reflects that
  two-way split. (How the project itself was seeded from the lake is shown separately by the setup
  "Migrating isaura-public → project" table.) Reads provenance.json; skips quietly if absent.
  """
  data = _load_provenance(output_dir) if os.path.exists(_provenance_path(output_dir)) else None
  if not data or not (data.get("featurizers") or data.get("projections")):
    return

  from rich import box
  from rich.table import Table

  width = 20
  color = active_color()  # themed to the running step (green during Describe)
  table = Table(
    title="Data provenance per model",
    caption="[green]█ Project store[/]  [yellow]█ Ersilia Run[/]",
    caption_justify="left",
    title_style=f"bold {color}",
    title_justify="left",
    box=box.SIMPLE_HEAD,
    border_style=color,
    header_style=f"bold {color}",
    pad_edge=False,
  )
  table.add_column("Kind", style="dim")
  table.add_column("Model", style="bold green")
  table.add_column("Sources", no_wrap=True)
  table.add_column("Project store", justify="right", style="green")
  table.add_column("Ersilia Run", justify="right", style="yellow")
  table.add_column("Total", justify="right")

  groups = [("descriptor", "featurizers"), ("projector", "projections")]
  for kind, key in groups:
    models = data.get(key, {})
    if not models:
      continue
    for m, st in models.items():
      proj = int(st.get("n_from_project", 0))  # read from the project store
      ers = int(st.get("n_computed", 0))  # computed via Ersilia (cache misses)
      total = int(st.get("n_total", 0)) or (proj + ers) or 1
      pc = round(proj / total * width)
      ec = max(0, width - pc)
      bar = f"[green]{'█' * pc}[/][yellow]{'█' * ec}[/]"
      table.add_row(kind, m, bar, f"{proj:,}", f"{ers:,}", f"{st.get('n_total', proj + ers):,}")
  console.print(table)
