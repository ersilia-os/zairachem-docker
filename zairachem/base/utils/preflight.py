"""Pre-flight checks for the setup step.

ZairaChem serves Ersilia Model Hub models as local Docker images (``ersiliaos/<eos_id>``).
Before a run we require (a) the Docker daemon to be running and (b) every required model image
to be present locally. Either failure stops the run with an informative message — we never
fetch anything automatically, we only show how.
"""

import subprocess

from zairachem.base.utils.console import active_color, console, echo
from zairachem.base.utils.model_version import ersilia_model_version, is_image_up_to_date
from zairachem.base.vars import ORG, REDIS_IMAGE, NGINX_IMAGE

#: Base/infrastructure images the model-serving stack needs, beyond the per-model images.
BASE_IMAGES = [REDIS_IMAGE, NGINX_IMAGE]


def _returncode(cmd):
  # Probe quietly: we deliberately expect non-zero exits (missing image / no daemon), so we
  # suppress output rather than route through run_command (which logs failures as errors).
  try:
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
  except FileNotFoundError:
    return 127  # docker binary not installed


def is_docker_running():
  """Return True if the Docker daemon responds to ``docker info``."""
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
  if not is_docker_running():
    echo(
      "Docker is not running. ZairaChem serves Ersilia models as Docker images and cannot continue.",
      kind="error",
    )
    echo("Start Docker (e.g. open Docker Desktop), then run the command again.", kind="info")
    raise SystemExit(1)

  missing_base = [img for img in BASE_IMAGES if not _docker_image_present(img)]
  if missing_base:
    echo(
      "Required base Docker image(s) are not available locally. ZairaChem cannot continue.",
      kind="error",
    )
    echo("Fetch them with:", kind="info")
    for img in missing_base:
      console.print(f"      [cyan]docker pull {img}[/]")
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
    title="🧩 Ersilia model images",
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
  for i, (kind, ids) in enumerate(groups):
    if i > 0:
      table.add_section()
    for m in ids:
      image = f"{ORG}/{m.lower()}:latest"
      if not _docker_image_present(image):
        status = "[red]✗ missing[/]"
        missing.append(m)
      elif is_image_up_to_date(m) is False:
        # local :latest digest differs from the registry's current :latest -> behind.
        status = "[yellow]⚠ behind latest[/]"
        outdated.append(m)
      else:
        status = "[green]✓ available[/]"
      table.add_row(kind, m, image, ersilia_model_version(m), status)
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
