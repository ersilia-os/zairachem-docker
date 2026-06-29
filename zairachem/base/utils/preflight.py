"""Pre-flight checks for the setup step.

ZairaChem serves Ersilia Model Hub models as local Docker images (``ersiliaos/<eos_id>``).
Before a run we require (a) the Docker daemon to be running and (b) every required model image
to be present locally. Either failure stops the run with an informative message — we never
fetch anything automatically, we only show how.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor

from zairachem.base.utils.concurrency import io_workers
from zairachem.base.utils.console import active_color, console, echo
from zairachem.base.utils.model_version import ersilia_model_version, is_image_up_to_date
from zairachem.base.vars import ORG, REDIS_IMAGE, NGINX_IMAGE

#: Base/infrastructure images the model-serving stack needs, beyond the per-model images.
BASE_IMAGES = [REDIS_IMAGE, NGINX_IMAGE]


def _returncode(cmd, timeout=120):
  # Probe quietly: we deliberately expect non-zero exits (missing image / no daemon), so we
  # suppress output rather than route through run_command (which logs failures as errors). The
  # timeout is a safety ceiling, not a fixed wait: a healthy `docker info` returns immediately and a
  # slow-but-working daemon returns in its own time (no false negative), while a genuinely hung
  # daemon is cut off instead of freezing setup forever. A timed-out probe counts as a failed probe.
  try:
    return subprocess.run(
      cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout
    ).returncode
  except FileNotFoundError:
    return 127  # docker binary not installed
  except subprocess.TimeoutExpired:
    return 124  # daemon unresponsive within the timeout — treat as "not available"


def is_docker_running():
  """Return True if the Docker daemon responds (within the probe timeout) to ``docker info``."""
  return _returncode(["docker", "info"]) == 0


def _docker_image_present(image):
  return _returncode(["docker", "image", "inspect", image]) == 0


def require_docker_and_base():
  """Hard-gate the run on Docker + base images, before any work happens.

  Requires the Docker daemon to be running and the base/infrastructure images
  (``redis:latest``, ``nginx:alpine``) to be present locally. Raises ``SystemExit(1)`` with an
  informative message otherwise. Never fetches images. Model images are checked separately, at
  the end of setup, by :func:`report_model_images`.
  """
  # `docker info` / `docker image inspect` can be slow (esp. macOS Docker Desktop waking its VM), so
  # show a spinner — otherwise a slow-but-working daemon makes setup look frozen. Errors/prompts are
  # printed AFTER the spinner stops so they render cleanly.
  with console.status("[dim]Checking Docker daemon and base images…[/]", spinner="dots"):
    running = is_docker_running()
    missing_base = [img for img in BASE_IMAGES if not _docker_image_present(img)] if running else []

  if not running:
    echo(
      "Docker is not running or not responding. ZairaChem serves Ersilia models as Docker images "
      "and cannot continue.",
      kind="error",
    )
    echo(
      "Start (or restart) Docker — e.g. open Docker Desktop, or `docker info` should return "
      "promptly — then run the command again.",
      kind="info",
    )
    raise SystemExit(1)

  if missing_base:
    echo(
      "Required base Docker image(s) are not available locally. ZairaChem cannot continue.",
      kind="error",
    )
    echo("Fetch them with:", kind="info")
    for img in missing_base:
      console.print(f"      [cyan]docker pull {img}[/]")
    raise SystemExit(1)


_NUMERIC_DTYPES = {"int", "integer", "float", "float32", "float64", "numeric", "double"}


def _is_numeric_output(model_id):
  """(eligible, reason) — True if the model's output columns are all numeric. None if unverifiable."""
  from zairachem.base.utils.utils import fetch_schema_from_github

  schema = fetch_schema_from_github(model_id)
  if not schema:
    return None, "could not read its output schema (run_columns.csv)"
  dtypes = [str(t).strip().lower() for t in schema[1] if str(t).strip()]
  if not dtypes:
    return None, "could not read its output schema (run_columns.csv)"
  non_numeric = [t for t in dtypes if t not in _NUMERIC_DTYPES]
  if non_numeric:
    return False, f"output is not numerical (column types: {', '.join(sorted(set(dtypes)))})"
  return True, ""


def _is_projection(model_id):
  """(eligible, reason) — True if Subtask is 'projection' and the output column count is even.

  Returns None for the eligibility when metadata/schema can't be fetched (can't verify).
  """
  from zairachem.base.utils.utils import fetch_model_metadata, fetch_schema_from_github

  meta = fetch_model_metadata(model_id)
  if not meta:
    return None, "could not read its metadata (metadata.json/.yml)"
  subtask = str(meta.get("Subtask", "")).strip().lower()
  if subtask != "projection":
    return False, f"its Subtask is '{meta.get('Subtask', '?')}', not 'Projection'"
  schema = fetch_schema_from_github(model_id)
  n_cols = schema[2] if schema else meta.get("Output Dimension")
  try:
    n_cols = int(n_cols)
  except (TypeError, ValueError):
    return None, "could not determine its output column count"
  if n_cols < 2 or n_cols % 2 != 0:
    return False, f"output has {n_cols} columns; a projection needs an even count (2, 4, 6, …)"
  return True, ""


def validate_model_roles(featurizer_ids, projection_ids):
  """Stop the run if any model is being used in a role it doesn't support.

  Featurizers must have **numerical** output; projections must declare ``Subtask: Projection`` and
  expose an **even** number of output columns (consumed as (x, y) pairs). Models that can't be
  verified (metadata/schema unreachable) are skipped with a warning — we never block on an inability
  to check. Raises ``SystemExit(1)`` on a genuine role mismatch.
  """
  problems = []
  for m in featurizer_ids:
    eligible, reason = _is_numeric_output(m)
    if eligible is None:
      echo(f"Could not verify featurizer '{m}': {reason}. Proceeding.", kind="warning")
    elif not eligible:
      problems.append(("featurizer", m, reason))
  for m in projection_ids:
    eligible, reason = _is_projection(m)
    if eligible is None:
      echo(f"Could not verify projection '{m}': {reason}. Proceeding.", kind="warning")
    elif not eligible:
      problems.append(("projection", m, reason))

  if not problems:
    return
  echo(
    f"{len(problems)} model(s) are not valid for their requested role — ZairaChem cannot continue.",
    kind="error",
  )
  for role, m, reason in problems:
    console.print(f"      [red]✖[/] [bold]{m}[/] is not a valid [bold]{role}[/]: {reason}")
  echo(
    "Use a featurizer (numerical output) with -f, and a projection model "
    "(Subtask=Projection, even column count) with -p.",
    kind="info",
  )
  raise SystemExit(1)


def report_model_images(featurizer_ids, projection_ids):
  """Print a summary table of Ersilia model-image availability; fail if any are missing.

  Intended to run at the END of the setup step: shows every requested ``ersiliaos/<id>`` image
  with a ✓/✗ status, grouped by kind (descriptors vs projectors). If any are missing it prints
  the ``ersilia fetch`` commands and raises ``SystemExit(1)``. Never fetches images.

  Parameters
  ----------
  featurizer_ids : list of str
      Descriptor (featurizer) model IDs.
  projection_ids : list of str
      Projector (projection) model IDs.
  """
  from rich import box
  from rich.table import Table

  color = active_color()
  table = Table(
    title="Ersilia model images",
    title_style=f"bold {color}",
    title_justify="left",
    box=box.SIMPLE_HEAD,
    border_style=color,
    header_style=f"bold {color}",
    pad_edge=False,
  )
  table.add_column("Kind", style="dim")
  table.add_column("Model", style="bold green")
  table.add_column("Image", style="dim")
  table.add_column("Version", style="cyan")
  table.add_column("Status", justify="center")

  missing = []
  outdated = []
  groups = [("descriptor", featurizer_ids), ("projector", projection_ids)]
  items = [(kind, m) for kind, ids in groups for m in ids]

  # Each model costs a docker inspect + a registry digest check + a version lookup — all independent
  # and network/IO-bound, so probe them in parallel (results gathered in order) instead of serially.
  def _probe(item):
    kind, m = item
    image = f"{ORG}/{m.lower()}:latest"
    if not _docker_image_present(image):
      state = "missing"
    elif is_image_up_to_date(m) is False:
      state = "outdated"  # local :latest digest differs from the registry's current :latest
    else:
      state = "ok"
    return kind, m, image, ersilia_model_version(m), state

  with console.status("[dim]Checking Ersilia model images…[/]", spinner="dots"):
    with ThreadPoolExecutor(max_workers=io_workers(len(items))) as ex:
      probed = list(ex.map(_probe, items))

  status_markup = {
    "missing": "[red]✗ missing[/]",
    "outdated": "[yellow]⚠ behind latest[/]",
    "ok": "[green]✓ available[/]",
  }
  for kind, m, image, version, state in probed:
    if state == "missing":
      missing.append(m)
    elif state == "outdated":
      outdated.append(m)
    table.add_row(kind, m, image, version, status_markup[state])
  console.print(table)

  if missing:
    total = len(featurizer_ids) + len(projection_ids)
    echo(
      f"{len(missing)} of {total} model image(s) missing — ZairaChem cannot continue.",
      kind="error",
    )
    echo("Fetch them with the Ersilia CLI, then re-run:", kind="info")
    for m in missing:
      console.print(f"      [cyan]ersilia fetch {m} --from_dockerhub[/]")
    raise SystemExit(1)

  if outdated:
    echo(
      f"{len(outdated)} model image(s) are behind the registry :latest — ZairaChem cannot continue.",
      kind="error",
    )
    echo("Update them to the latest image, then re-run:", kind="info")
    for m in outdated:
      console.print(f"      [cyan]ersilia fetch {m} --from_dockerhub[/]")
    raise SystemExit(1)


def report_reference_transformers(featurizer_ids, reference_library):
  """Print a table of reference-library transformer availability per featurizer; fail if any missing.

  The treat step applies, for each featurizer, a transformer pre-fitted on ``reference_library`` and
  published to a public bucket. This runs at the END of setup (after the model images are confirmed)
  so a run aborts early — before any descriptors are computed — if a transformer is missing. The
  version checked is the same one the treat step will use (:func:`ersilia_model_version`). Never
  uploads or fits anything.
  """
  from rich import box
  from rich.table import Table

  from zairachem.treat.imputers.reference_transformer import transformer_exists, transformer_url

  color = active_color()
  table = Table(
    title=f"Reference-library transformers · {reference_library}",
    title_style=f"bold {color}",
    title_justify="left",
    box=box.SIMPLE_HEAD,
    border_style=color,
    header_style=f"bold {color}",
    pad_edge=False,
  )
  table.add_column("Featurizer", style="bold green")
  table.add_column("Version", style="cyan")
  table.add_column("Transformer", style="dim")
  table.add_column("Status", justify="center")

  # One HEAD request per featurizer — independent and network-bound, so check them in parallel
  # (results gathered in featurizer order).
  def _check(m):
    version = ersilia_model_version(m)
    url = transformer_url(m, version, reference_library=reference_library)
    ok = transformer_exists(m, version, reference_library=reference_library)
    return m, version, url, ok

  with console.status("[dim]Checking reference-library transformers…[/]", spinner="dots"):
    with ThreadPoolExecutor(max_workers=io_workers(len(featurizer_ids))) as ex:
      checked = list(ex.map(_check, featurizer_ids))

  missing = []
  for m, version, url, ok in checked:
    if ok:
      status = "[green]✓ available[/]"
    else:
      status = "[red]✗ missing[/]"
      missing.append((m, version, url))
    table.add_row(m, version, url.rsplit("/", 1)[-1], status)
  console.print(table)

  if missing:
    echo(
      f"{len(missing)} of {len(featurizer_ids)} featurizer(s) have no transformer in reference "
      f"library '{reference_library}' — ZairaChem cannot continue.",
      kind="error",
    )
    echo(
      "A transformer fitted on this reference library is required for each featurizer. "
      "Fit and upload the missing one(s), then re-run:",
      kind="info",
    )
    for m, version, url in missing:
      console.print(f"      [red]✖[/] [bold]{m}[/] ({version}) → [dim]{url}[/]")
    raise SystemExit(1)
