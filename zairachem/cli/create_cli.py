import importlib.util

# Heavy pipeline classes (Describer, EstimatorPipeline, Reporter, run_fit, ...) pull in
# matplotlib, lazyqsar, xgboost and onnx. They are imported lazily inside the commands that
# use them so that `zairachem --help` and argument parsing stay fast.
# Silence matplotlib's "Matplotlib is building the font cache; this may take a moment." notice,
# which it logs on first import — set here (earliest entry point) before any matplotlib import.
import logging as _logging
import os
import socket
import subprocess
from urllib.parse import urlparse

import rich_click as click
import rich_click.rich_click as rc

from zairachem.base.utils.logging import logger
from zairachem.base.utils.rich_help import StatusGroupMixin
from zairachem.base.vars import REDIS_IMAGE

_logging.getLogger("matplotlib").setLevel(_logging.ERROR)


click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True

rc.USE_RICH_MARKUP = True
rc.SHOW_ARGUMENTS = True
rc.COLOR_SYSTEM = "truecolor"
rc.STYLE_OPTION = "bold magenta"
rc.STYLE_COMMAND = "bold green"
rc.STYLE_METAVAR = "italic yellow"
rc.STYLE_SWITCH = "underline cyan"
rc.STYLE_USAGE = "bold blue"
rc.STYLE_OPTION_DEFAULT = "dim italic"

# Main commands get a bold-green panel border to stand out; the advanced pipeline steps are
# dimmed (border + rows) so they clearly read as secondary.
rc.COMMAND_GROUPS = {
  "zairachem": [
    {
      "name": "Main commands",
      "commands": ["fit", "predict"],
      "panel_styles": {"border_style": "bold green"},
    },
    {
      "name": "Pipeline steps (advanced — run in this order)",
      "commands": ["setup", "describe", "treat", "estimate", "pool", "report", "finish"],
      "table_styles": {"row_styles": ["dim"]},
      "panel_styles": {"border_style": "dim"},
    },
  ]
}


def _isaura_installed():
  # Detect the optional `isaura` package with find_spec (no import, instant) so --help is fast.
  try:
    return importlib.util.find_spec("isaura") is not None
  except (ImportError, ValueError):
    return False


def _isaura_engine_running():
  # The isaura engine is a local MinIO instance; isaura reads its endpoint from MINIO_ENDPOINT
  # (default http://127.0.0.1:9000). A short TCP probe tells us whether it is up without
  # importing isaura or doing any S3/auth work. The default endpoint is loopback, so the probe
  # returns immediately; the timeout only bounds a misconfigured remote endpoint.
  endpoint = os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000")
  parsed = urlparse(endpoint)
  host = parsed.hostname or "127.0.0.1"
  port = parsed.port or 9000
  try:
    with socket.create_connection((host, port), timeout=0.25):
      return True
  except OSError:
    return False


def _isaura_status_line():
  # Report (a) whether isaura is installed and (b) whether its local engine is started, as a
  # single colored status line on the --help screen.
  if not _isaura_installed():
    return "[red]●[/] [bold]Isaura store[/]: [red]not installed[/]"
  if _isaura_engine_running():
    return "[green]●[/] [bold]Isaura store[/]: installed · [green]engine running[/]"
  return (
    "[yellow]●[/] [bold]Isaura store[/]: installed · [yellow]engine stopped[/] "
    "[dim](start with: isaura engine --start)[/]"
  )


def _docker_running():
  # Bounded with a timeout so a slow or hanging daemon never stalls --help.
  try:
    return (
      subprocess.run(
        ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3
      ).returncode
      == 0
    )
  except (OSError, subprocess.TimeoutExpired):
    return False


def _docker_image_present(image):
  # Bounded probe, only used on the --help screen (Docker is already confirmed up).
  try:
    return (
      subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3,
      ).returncode
      == 0
    )
  except (OSError, subprocess.TimeoutExpired):
    return False


def _docker_status_line():
  # Docker is required: ZairaChem serves the Ersilia models as Docker images. When the daemon
  # is up, also report whether the base images (redis) are pulled.
  if not _docker_running():
    return (
      "[red]●[/] [bold]Docker[/]: [red]not running[/] [dim](required — start Docker Desktop)[/]"
    )

  def mark(image):
    return "[green]✓[/]" if _docker_image_present(image) else "[red]✗[/]"

  return (
    "[green]●[/] [bold]Docker[/]: [green]running[/] "
    f"[dim]· base images:[/] redis {mark(REDIS_IMAGE)}"
  )


def _build_cli_help():
  return (
    "[bold green]ZairaChem CLI[/] — Automated QSAR modeling with the Ersilia Model Hub\n\n"
    + _docker_status_line()
    + "\n\n"
    + _isaura_status_line()
  )


class _StatusGroup(StatusGroupMixin, click.RichGroup):
  # StatusGroupMixin renders the Options and command panels with a shared first-column width so
  # their help text lines up at any terminal width (rich-click otherwise sizes the two panel
  # types differently). Here we additionally compute the title + live Docker/Isaura status lines
  # only when help is actually rendered, so the (potentially slow) `docker info` probe never runs
  # during normal command runs.
  def format_help(self, ctx, formatter):
    self.help = _build_cli_help()
    super().format_help(ctx, formatter)


@click.group(cls=_StatusGroup)
@click.option(
  "--verbose",
  "-v",
  is_flag=True,
  default=False,
  help="Show detailed logs in the console (quiet by default). Place before the command, e.g. `zairachem -v fit ...`.",
)
def cli(verbose):
  """Set the console log verbosity for every ZairaChem subcommand."""
  # Logging is quiet by default (WARNING/ERROR only); -v streams the full DEBUG log. The file
  # log always keeps everything. Stored on the logger so per-step configure() calls preserve it.
  logger.set_verbosity(verbose)


# Command modules, in registration (and --help) order.
COMMANDS = ("fit", "predict", "setup", "describe", "treat", "estimate", "pool", "report", "finish")


def create_cli():
  """Register every command on the ``cli`` group, in ``--help`` order, and return the group.

  Returns
  -------
  click.Group
    The ``zairachem`` command group with all commands attached.
  """
  # Each command module attaches itself with ``@cli.command`` on import, so the import order below
  # is the registration order (it drives the ordering on the --help screen).
  for name in COMMANDS:
    importlib.import_module(f"zairachem.cli.commands.{name}")

  return cli
