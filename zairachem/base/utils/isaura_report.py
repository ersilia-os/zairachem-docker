"""End-of-setup report: per-model coverage of the current input against the isaura store.

Informational only — never raises. If isaura is not installed or its engine (MinIO) is not
reachable, prints a short note and returns. Queries only the configured store (the bucket passed
in, i.e. ``params["read_store"]``).
"""

import os
import socket
from urllib.parse import urlparse

from zairachem.base.utils.console import console, echo
from zairachem.base.utils.utils import get_bucket_records
from zairachem.base.utils.model_version import ersilia_model_version
from zairachem.base.vars import DEFAULT_ISAURA_BUCKET


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
  table = Table(
    title=f"💧 Isaura store coverage · {bucket}",
    title_style="bold cyan",
    title_justify="left",
    box=box.ROUNDED,
    border_style="cyan",
    header_style="bold cyan",
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


def check_isaura_project_available(model_dir):
  """Stop if an isaura project already exists with the model folder's name.

  The run will create an isaura project named exactly like the model folder to contribute its
  precalculations; a pre-existing project would collide. Raises ``SystemExit(1)`` if such a project
  exists. Skips quietly if isaura isn't installed or the engine isn't reachable (we can't verify).
  """
  names = _list_project_names()
  if names is None:
    return
  project = os.path.basename(os.path.normpath(model_dir))
  if project in names:
    echo(f"An isaura project named '{project}' already exists — ZairaChem cannot continue.", kind="error")
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
  table.add_column("isaura", style="yellow")
  for m, mh, have in mismatches:
    table.add_row(m, mh, ", ".join(have))
  console.print(table)
  echo(
    f"{len(mismatches)} model(s) have isaura precalculations from a different version — ZairaChem cannot continue.",
    kind="error",
  )
  echo("Use a matching model image, or clear/rebuild those store versions, then re-run.", kind="info")
  raise SystemExit(1)
