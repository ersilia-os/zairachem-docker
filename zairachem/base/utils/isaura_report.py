"""End-of-setup report: per-model coverage of the current input against the isaura store.

Informational only — never raises. If isaura is not installed or its engine (MinIO) is not
reachable, prints a short note and returns. Queries only the configured store (the bucket passed
in, i.e. ``params["read_store"]``).
"""

import json
import os
import re
import socket
from urllib.parse import urlparse

from zairachem.base.utils.console import active_color, console, echo
from zairachem.base.utils.utils import get_bucket_records
from zairachem.base.utils.model_version import ersilia_model_version
from zairachem.base.vars import DEFAULT_ISAURA_BUCKET


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


def _engine_running():
  # Quick TCP probe of the MinIO endpoint so we never block on a slow isaura call.
  endpoint = os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000")
  parsed = urlparse(endpoint)
  host = parsed.hostname or "127.0.0.1"
  port = parsed.port or 9000
  try:
    with socket.create_connection((host, port), timeout=0.25):
      return True
  except OSError:
    return False


def report_isaura_coverage(bucket, featurizer_ids, projection_ids, smiles):
  """Print a per-model table of how many input compounds are already cached in the store.

  Parameters
  ----------
  bucket : str or None
      The isaura store/bucket to query. When falsy (no store configured), the default public
      bucket (``isaura-public``) is used.
  featurizer_ids, projection_ids : list of str
      Descriptor and projector model IDs.
  smiles : list of str
      The run's (standardized) input SMILES to look up.
  """
  bucket = bucket or DEFAULT_ISAURA_BUCKET
  try:
    from isaura.manage import IsauraChecker
  except Exception:
    echo("Isaura store coverage: unavailable (isaura not installed).", kind="warning")
    return

  if not _engine_running():
    echo(
      "Isaura store coverage: unavailable (engine not running; start with: isaura engine --start).",
      kind="warning",
    )
    return

  from rich import box
  from rich.table import Table

  total = len(smiles)
  color = active_color()
  table = Table(
    title=f"💧 Isaura store coverage · {bucket}",
    title_style=f"bold {color}",
    title_justify="left",
    box=box.SIMPLE_HEAD,
    border_style=color,
    header_style=f"bold {color}",
    pad_edge=False,
  )
  table.add_column("Kind", style="dim")
  table.add_column("Model", style="bold green")
  table.add_column("Version", style="cyan")
  table.add_column("Cached", justify="right")
  table.add_column("Coverage", justify="right")

  groups = [("descriptor", featurizer_ids), ("projector", projection_ids)]
  for i, (kind, ids) in enumerate(groups):
    if i > 0:
      table.add_section()
    for m in ids:
      try:
        version = ersilia_model_version(m)
        with IsauraChecker(bucket, m, version) as checker:
          seen = checker.seen_many(smiles)
        cached = sum(1 for v in seen.values() if v[0])
        pct = (100 * cached / total) if total else 0
        color = "green" if cached else "dim"
        table.add_row(kind, m, version, f"{cached:,} / {total:,}", f"[{color}]{pct:.0f}%[/]")
      except Exception:
        table.add_row(kind, m, "—", "—", "[dim]n/a[/]")
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


def _project_is_empty(name):
  """True if the project holds no precalculation data (only root-level metadata, or nothing).

  Model precalculations live under ``<model_id>/<version>/...`` keys; root-level files such as
  ``access.json`` are metadata and don't count. Returns False if the bucket can't be read (the
  caller then treats it as a collision rather than wrongly reusing it).
  """
  try:
    from isaura.base import MinioStore
    from isaura.const import MINIO_ENDPOINT, MINIO_LOCAL_AK, MINIO_LOCAL_SK

    store = MinioStore(endpoint=MINIO_ENDPOINT, access=MINIO_LOCAL_AK, secret=MINIO_LOCAL_SK)
    resp = store.client.list_objects_v2(Bucket=name)
    return not any("/" in o.get("Key", "") for o in resp.get("Contents", []))
  except Exception:
    return False


def check_isaura_project_available(model_dir):
  """Stop if an isaura project already exists *with data* under the model folder's name.

  The run will write its precalculations to an isaura project named exactly like the model folder.
  A pre-existing project that already holds data would collide, so that raises ``SystemExit(1)``. An
  existing but *empty* project (e.g. left behind by an aborted run) is accepted and reused as-is.
  Skips quietly if isaura isn't installed or the engine isn't reachable (we can't verify).
  """
  names = _list_project_names()
  if names is None:
    return
  project = sanitize_project_name(os.path.basename(os.path.normpath(model_dir)))
  if project not in names:
    return
  if _project_is_empty(project):
    echo(f"Reusing existing empty isaura project '{project}'.", kind="info")
    return
  echo(
    f"An isaura project named '{project}' already exists with data — ZairaChem cannot continue.",
    kind="error",
  )
  echo(
    "Use a different model-directory name, or remove/rename that isaura project, then re-run.",
    kind="info",
  )
  raise SystemExit(1)


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
    stored = _stored_versions(bucket)
  except Exception:
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


def create_and_migrate_project(project, featurizer_ids, projection_ids, smiles, output_dir=None):
  """Create the run's isaura project (public) and migrate matching precalcs from the lake into it.

  Runs at fit setup when write is enabled: creates the bucket named like the model folder, then per
  model copies the input molecules already present in isaura-public into the project. Skips with a
  warning if isaura/engine is unavailable; per-model failures are reported as '—'. Never raises.
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

  # Migrate the input molecules already cached in the lake into the project, per model.
  from rich import box
  from rich.table import Table

  color = active_color()
  table = Table(
    title=f"💧 Migrating isaura-public → {project}",
    title_style=f"bold {color}",
    title_justify="left",
    box=box.SIMPLE_HEAD,
    border_style=color,
    header_style=f"bold {color}",
    pad_edge=False,
  )
  table.add_column("Kind", style="dim")
  table.add_column("Model", style="bold green")
  table.add_column("Migrated", justify="right")

  groups = [("descriptor", featurizer_ids), ("projector", projection_ids)]
  for i, (kind, ids) in enumerate(groups):
    if i > 0:
      table.add_section()
    for m in ids:
      migrated = "—"
      try:
        version = ersilia_model_version(m)
        with IsauraChecker(DEFAULT_ISAURA_BUCKET, m, version) as checker:
          seen = checker.seen_many(smiles)
        present = [s for s in smiles if seen.get(s, [False, None])[0]]
        if not present:
          migrated = "0"
        else:
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
            IsauraWriter(input_csv=None, model_id=m, model_version=version, bucket=project).write(
              df=df
            )
          migrated = f"{n:,}"
        if output_dir is not None:
          # 0 when nothing matched; an int otherwise (records B = molecules pulled from the lake).
          record_migration(
            output_dir, m, 0 if migrated in ("—", "0") else int(migrated.replace(",", ""))
          )
      except Exception:
        migrated = "—"
      table.add_row(kind, m, migrated)
  console.print(table)


# --- Per-model data provenance (A = Ersilia-computed, B = isaura-public, C = project native) ---


def _provenance_path(output_dir):
  from zairachem.base.vars import DATA_SUBFOLDER

  return os.path.join(output_dir, DATA_SUBFOLDER, "provenance.json")


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


def record_provenance(output_dir, kind, model_id, n_total, n_from_project, n_computed):
  """Record a model's describe/treat provenance. kind is 'featurizers' or 'projections'."""
  data = _load_provenance(output_dir)
  data.setdefault(kind, {})[model_id] = {
    "n_total": int(n_total),
    "n_from_project": int(n_from_project),
    "n_computed": int(n_computed),
  }
  _write_provenance(output_dir, data)


def report_data_provenance(output_dir=None):
  """Render per-model stacked bars of where each model's data came from this run.

  A = computed via Ersilia (cache misses), B = migrated from isaura-public this run, C = the
  project's own pre-existing data. Reads provenance.json; skips quietly if absent. When output_dir
  is None, it's resolved from the active session.
  """
  if output_dir is None:
    try:
      from zairachem.base.vars import BASE_DIR, SESSION_FILE

      with open(os.path.join(BASE_DIR, SESSION_FILE)) as f:
        output_dir = json.load(f)["output_dir"]
    except Exception:
      return
  data = _load_provenance(output_dir) if os.path.exists(_provenance_path(output_dir)) else None
  if not data or not (data.get("featurizers") or data.get("projections")):
    return

  from rich import box
  from rich.table import Table

  migrated = data.get("migrated", {})
  width = 20
  color = active_color()  # themed to the running step (green during Describe)
  table = Table(
    title="🧪 Data provenance per model",
    caption="[green]█ Project store[/]  [cyan]█ Lake store[/]  [yellow]█ Ersilia Run[/]",
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
  table.add_column("Lake store", justify="right", style="cyan")
  table.add_column("Ersilia Run", justify="right", style="yellow")
  table.add_column("Total", justify="right")

  groups = [("descriptor", "featurizers"), ("projector", "projections")]
  first = True
  for kind, key in groups:
    models = data.get(key, {})
    if not models:
      continue
    if not first:
      table.add_section()
    first = False
    for m, st in models.items():
      n_from_project = int(st.get("n_from_project", 0))
      a = int(st.get("n_computed", 0))  # Ersilia
      b = min(int(migrated.get(m, 0)), n_from_project)  # isaura-public (migrated)
      c = n_from_project - b  # project native
      total = int(st.get("n_total", 0)) or (a + b + c) or 1
      cc = round(c / total * width)
      bc = round(b / total * width)
      ac = max(0, width - cc - bc)
      bar = f"[green]{'█' * cc}[/][cyan]{'█' * bc}[/][yellow]{'█' * ac}[/]"
      table.add_row(kind, m, bar, f"{c:,}", f"{b:,}", f"{a:,}", f"{st.get('n_total', a + b + c):,}")
  console.print(table)
