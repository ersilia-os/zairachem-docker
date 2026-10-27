"""Pre-flight checks for the setup step.

ZairaChem serves Ersilia Model Hub models as local Docker images (``ersiliaos/<eos_id>``).
Before a run we require (a) the Docker daemon to be running and (b) every required model image
to be present locally. Either failure stops the run with an informative message — model images are
never pulled automatically, we only show how. Setup does download the per-featurizer data files a run
needs: the reference-library transformers and the rank-reference matrices.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from zairachem.base.utils.concurrency import io_workers
from zairachem.base.utils.console import active_color, console, echo, plural
from zairachem.base.utils.logging import logger
from zairachem.base.utils.model_version import ersilia_model_version, is_image_up_to_date
from zairachem.base.utils.processes import other_runs
from zairachem.base.vars import ORG, REDIS_IMAGE

#: Base/infrastructure images the model-serving stack needs, beyond the per-model images.
BASE_IMAGES = [REDIS_IMAGE]


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
  """Return True if the Docker daemon responds (within the probe timeout)."""
  from zairachem.base.utils.docker_socket import daemon_running

  fast = daemon_running()  # milliseconds over the daemon's socket; the CLI is the fallback
  if fast is not None:
    return fast
  return _returncode(["docker", "version", "--format", "{{.Server.Version}}"]) == 0


def _docker_image_present(image):
  from zairachem.base.utils.docker_socket import image_present

  fast = image_present(image)  # milliseconds over the daemon's socket; the CLI is the fallback
  return fast if fast is not None else _returncode(["docker", "image", "inspect", image]) == 0


def require_docker_and_base():
  """Hard-gate the run on Docker + base images, before any work happens.

  Requires the Docker daemon to be running and the base/infrastructure images
  (``redis:7.4.2``) to be present locally. Raises ``SystemExit(1)`` with an
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
      "Required base Docker images are not available locally. ZairaChem cannot continue.",
      kind="error",
    )
    echo("Fetch them with:", kind="info")
    for img in missing_base:
      console.print(f"      [cyan]docker pull {img}[/]")
    raise SystemExit(1)


def reap_orphaned_servers():
  """Stop model servers left running by a ZairaChem run that was killed, and say so.

  A run stops the servers it started when it ends; one that was killed (SIGKILL, a closed terminal)
  could not, so the next run does it. Servers that a live run uses, or that no run started (for
  example one served with `ersilia serve`), are not touched.
  """
  from zairachem.describe.descriptors.compose import reap_orphans

  stopped = reap_orphans()
  if stopped:
    names = ", ".join(s.removesuffix("_api") for s in stopped)
    echo(
      f"Stopped {plural(len(stopped), 'model server')} left by a run that was killed: {names}.",
      kind="info",
    )


def warn_other_runs():
  """Warn when other ZairaChem runs of this user are active.

  Runs share the model servers and wait for each other's locks, so a run that is long, or was
  suspended with Ctrl-Z, holds up this one. Only warns: it never stops the run.
  """
  runs = other_runs()
  if not runs:
    return
  echo(
    f"{plural(len(runs), 'other ZairaChem run')} active; runs share the model servers and wait "
    "for each other:",
    kind="warning",
  )
  for run in runs:
    console.print(f"      {run['text']}")
  echo(
    "Resume a suspended run with `fg`, or stop it with `kill <pid>` (`kill -9` if it is suspended).",
    kind="info",
  )


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
    f"{plural(len(problems), 'model')} {'is' if len(problems) == 1 else 'are'} not valid for "
    "the requested role — ZairaChem cannot continue.",
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


def _model_images_table():
  """Return the empty "Ersilia model images" table with its columns."""
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
  return table


def _probe_model_image(item):
  """Return ``(kind, model, image, version, state)`` for one ``(kind, model)`` item.

  ``state`` is ``"missing"``, ``"outdated"``, ``"unverified"`` or ``"ok"``.
  """
  kind, m = item
  image = f"{ORG}/{m.lower()}:latest"
  if not _docker_image_present(image):
    state = "missing"
  else:
    current = is_image_up_to_date(m)
    if current is False:
      state = "outdated"  # local :latest digest differs from the registry's current :latest
    elif current is None:
      state = "unverified"  # the registry could not be reached: present, but not known current
    else:
      state = "ok"
  return kind, m, image, ersilia_model_version(m), state


_IMAGE_STATUS_MARKUP = {
  "missing": "[red]✗ missing[/]",
  "outdated": "[yellow]⚠ behind latest[/]",
  "unverified": "[dim]? not verified[/]",
  "ok": "[green]✓ available[/]",
}


def _stop_on_bad_images(missing, outdated, total):
  """Print fetch hints and raise ``SystemExit(1)`` if any model image is missing or outdated."""
  if missing:
    echo(
      f"{len(missing)} of {plural(total, 'model image')} missing — ZairaChem cannot continue.",
      kind="error",
    )
    echo("Fetch them with the Ersilia CLI, then re-run:", kind="info")
    for m in missing:
      console.print(f"      [cyan]ersilia fetch {m} --from_dockerhub[/]")
    raise SystemExit(1)

  if outdated:
    echo(
      f"{plural(len(outdated), 'model image')} {'is' if len(outdated) == 1 else 'are'} behind the "
      "registry's :latest — ZairaChem cannot continue.",
      kind="error",
    )
    echo("Update them to the latest image, then re-run:", kind="info")
    for m in outdated:
      console.print(f"      [cyan]ersilia fetch {m} --from_dockerhub[/]")
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
  table = _model_images_table()

  missing = []
  outdated = []
  groups = [("descriptor", featurizer_ids), ("projector", projection_ids)]
  items = [(kind, m) for kind, ids in groups for m in ids]

  # Each model costs a docker inspect + a registry digest check + a version lookup — all independent
  # and network/IO-bound, so probe them in parallel (results gathered in order) instead of serially.
  with console.status("[dim]Checking Ersilia model images…[/]", spinner="dots"):
    with ThreadPoolExecutor(max_workers=io_workers(len(items))) as ex:
      probed = list(ex.map(_probe_model_image, items))

  for kind, m, image, version, state in probed:
    if state == "missing":
      missing.append(m)
    elif state == "outdated":
      outdated.append(m)
    table.add_row(kind, m, image, version, _IMAGE_STATUS_MARKUP[state])
  console.print(table)
  if any(p[4] == "unverified" for p in probed):
    echo(
      "Could not compare these images with the registry (not pulled from it, or it is "
      "unreachable); if a model fails, update it with `ersilia fetch <id> --from_dockerhub`.",
      kind="warning",
    )

  _stop_on_bad_images(missing, outdated, len(featurizer_ids) + len(projection_ids))


def _transformers_table(reference_library):
  """Return the empty reference-library transformers table with its columns."""
  from rich import box
  from rich.table import Table

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
  return table


def _fetch_transformer(rt, m, reference_library, output_dir):
  """Download, check and save one featurizer's transformer.

  Parameters
  ----------
  rt : module
      :mod:`zairachem.treat.imputers.reference_transformer` (imported once by the caller).
  m : str
      Featurizer model ID.
  reference_library : str
      Reference library the transformer was fitted on.
  output_dir : str
      Model folder; the copy is saved under its transformers subfolder.

  Returns
  -------
  tuple
      ``(m, version, url, error)``; ``error`` is None when the copy was saved.
  """
  from zairachem.base.vars import TRANSFORMERS_SUBFOLDER

  version = ersilia_model_version(m)
  url = rt.transformer_url(m, version, reference_library=reference_library)
  try:
    transformer = rt.fetch_reference_transformer(m, version, reference_library=reference_library)
    rt.check_transformer(transformer, m, version)
    rt.save_local_transformer(
      transformer,
      os.path.join(output_dir, TRANSFORMERS_SUBFOLDER, rt.local_transformer_name(m, version)),
    )
    error = None
  except RuntimeError as e:
    error = str(e)
  return m, version, url, error


def _tabulate_transformers(table, checked):
  """Add one row per checked featurizer; return ``(kept, missing)``."""
  kept, missing = [], []
  for m, version, url, error in checked:
    if error is None:
      status = "[green]✓ downloaded[/]"
      kept.append(m)
    else:
      status = "[yellow]– skipped[/]"
      missing.append((m, version, url, error))
      logger.debug(f"[setup] {m} ({version}) skipped, {url}: {error}")
    table.add_row(m, version, url.rsplit("/", 1)[-1], status)
  return kept, missing


def _report_transformer_gaps(kept, missing, n_featurizers, reference_library):
  """Stop the run if no transformer is usable; otherwise warn about the skipped ones."""
  if not kept:
    echo(
      f"None of the {plural(n_featurizers, 'featurizer')} has a usable transformer in reference "
      f"library '{reference_library}' — ZairaChem cannot continue.",
      kind="error",
    )
    echo(
      "A transformer fitted on this reference library is required for each featurizer. "
      "Fit and upload the missing ones, then re-run:",
      kind="info",
    )
    for m, version, url, _ in missing:
      console.print(f"      [red]✖[/] [bold]{m}[/] ({version}) → [dim]{url}[/]")
    raise SystemExit(1)
  if missing:
    echo(
      f"Skipping {plural(len(missing), 'descriptor')} with no transformer in '{reference_library}': "
      f"{', '.join(m for m, *_ in missing)}.",
      kind="warning",
    )


def prepare_reference_transformers(featurizer_ids, reference_library, output_dir):
  """Download each featurizer's reference-library transformer into the model; return those usable.

  The treat step applies, for each featurizer, a transformer pre-fitted on ``reference_library`` and
  published to a public bucket. Setup runs this before anything else checks the descriptors, saving
  each one to ``<output_dir>/transformers/`` where treat and predict read it. A featurizer whose
  transformer is missing or malformed is skipped (reported, not fatal), so a descriptor whose
  transformer is not yet published simply sits out the run; only when none is left does the run
  stop. The version is the same one the treat step will use (:func:`ersilia_model_version`). Never
  uploads or fits anything.

  Returns
  -------
  list of str
      The featurizers with a usable transformer, in input order.
  """
  from zairachem.treat.imputers import reference_transformer as rt

  table = _transformers_table(reference_library)

  # One download per featurizer (0.5-2 MB each) — independent and network-bound, so fetch them in
  # parallel (results gathered in featurizer order). ``error`` is None when the copy was saved.
  def _check(m):
    return _fetch_transformer(rt, m, reference_library, output_dir)

  with console.status("[dim]Downloading reference-library transformers…[/]", spinner="dots"):
    with ThreadPoolExecutor(max_workers=io_workers(len(featurizer_ids))) as ex:
      checked = list(ex.map(_check, featurizer_ids))

  kept, missing = _tabulate_transformers(table, checked)
  console.print(table)

  _report_transformer_gaps(kept, missing, len(featurizer_ids), reference_library)
  return kept


_MB = 1 << 20


def prepare_rank_references(featurizer_ids):
  """Download each featurizer's published rank reference into the cache, one progress bar each.

  Cached ones are reused. The rest, if pre-screening keeps them for training, are computed on their
  model server during estimate and cached (the size given for those is the uncompressed upper
  bound: width x library size x float32). Fetches lazy-qsar's reference list first, since a matrix
  is only valid for its library. Never fatal: a failed download is computed instead.
  """
  from zairachem.base.utils.live import LiveProgressBar
  from zairachem.base.vars import RANK_REFERENCE_DIR
  from zairachem.describe.descriptors import rank_reference as rr

  if rr.reference_smiles() is None:
    echo(
      "lazy-qsar's rank reference list is unavailable; models will train without rank.",
      kind="warning",
    )
    return
  versions = {m: ersilia_model_version(m) for m in featurizer_ids}
  missing = [m for m in featurizer_ids if not rr.stored(m, versions[m])]
  computed = []
  for m in missing:
    size = rr.published_bytes(m, versions[m])
    if size is None:
      computed.append(m)
      continue
    total_mb = max(1, -(-size // _MB))
    bar = LiveProgressBar(
      f"Rank reference {m}", total=total_mb, color=active_color(), discrete=True
    )

    def _progress(done, _total, bar=bar):
      bar.done = min(bar.total, done // _MB)
      bar.set_note(f"{done / _MB:,.0f}/{size / _MB:,.0f} MB")

    with bar.live():
      ok = rr.download(m, versions[m], on_progress=_progress) is not None
      if ok:
        bar.done = bar.total
    if not ok:
      computed.append(m)
  if not computed:
    echo(f"Rank references ready in {RANK_REFERENCE_DIR} for all featurizers.")
    return
  echo(
    f"No published rank reference for {', '.join(computed)}: any of them that is trained is "
    f"computed and cached in {RANK_REFERENCE_DIR} (up to "
    f"{rr.format_bytes(rr.estimate_bytes(computed))}).",
    kind="warning",
  )
