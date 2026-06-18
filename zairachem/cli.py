import os
import random
import socket
import subprocess
import importlib.util
from urllib.parse import urlparse
import numpy as np
import rich_click as click
import rich_click.rich_click as rc
from zairachem.base.utils.logging import logger
from zairachem.base.vars import RANDOM_SEED, REDIS_IMAGE, NGINX_IMAGE

# Heavy pipeline classes (Describer, EstimatorPipeline, Reporter, run_fit, ...) pull in
# matplotlib, lazyqsar, xgboost and onnx. They are imported lazily inside the commands that
# use them so that `zairachem --help` and argument parsing stay fast. Only the lightweight
# `zairachem.finish.finish` constant is needed at module load time.
from zairachem.finish.finish import CLEAN_TARGET_ALL


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
# Fix a proportional command-name/description column split so the description columns line up
# across BOTH panels (otherwise each panel auto-sizes to its own longest command name and the
# two panels misalign).
rc.STYLE_COMMANDS_TABLE_COLUMN_WIDTH_RATIO = (1, 9)

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


def process_group(clean, flush, anonymize, batch_size=None, clean_target=CLEAN_TARGET_ALL):
  # Each pipeline class is imported right before its step runs, so heavy dependencies load only
  # when that step executes (matplotlib for reporting, lazyqsar/xgboost/onnx for estimation) —
  # not all at once at the start. Several of these imports call loguru's logger.remove(), so
  # logger.configure() is re-asserted after each one to keep zairachem's log sinks alive.
  from zairachem.describe.descriptors.describe import Describer

  logger.configure()
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None, batch_size=batch_size).run()

  from zairachem.base.utils.isaura_report import report_data_provenance

  report_data_provenance()

  from zairachem.treat.imputers.impute import Imputer

  logger.configure()
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None, batch_size=batch_size).run()

  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  logger.configure()
  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None, batch_size=batch_size).run()

  from zairachem.pool.pipe import PoolerPipeline

  logger.configure()
  logger.debug("[#ff69b4]Running the pooling pipeline to aggregate the result using bagging[/]")
  PoolerPipeline(path=None, batch_size=batch_size).run()

  from zairachem.report.report import Reporter

  logger.configure()
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=None).run()

  from zairachem.finish.finish import Finisher

  logger.configure()
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(
    path=None, clean=clean, flush=flush, anonymize=anonymize, clean_target=clean_target
  ).run()


def common_options(
  require_input: bool = True,
  include_task: bool = False,
  include_eos: bool = False,
  include_clean_target: bool = False,
):
  def _decorator(func):
    options = [
      click.option(
        "--input-file", "-i", required=require_input, help="Path to the input CSV file."
      ),
      click.option(
        "--model-dir",
        "-m",
        required=False,
        default=None,
        help="Path to the model directory.",
      ),
      click.option(
        "--clean",
        is_flag=True,
        help="Delete descriptor files after the run to save space.",
      ),
      click.option(
        "--flush",
        is_flag=True,
        help="Delete model checkpoints after the run to save space.",
      ),
      click.option(
        "--anonymize", is_flag=True, help="Anonymize the input molecules in all outputs."
      ),
    ]
    if include_clean_target:
      options.append(
        click.option(
          "--clean-target",
          "-ct",
          type=click.Choice(["all", "model", "predict"], case_sensitive=False),
          default="all",
          help="Target for clean/flush/anonymize: 'all' (both model and predict dirs), 'model' (model dir only), or 'predict' (predict dir only, only valid during prediction).",
        )
      )
    if include_task:
      options.insert(
        1,
        click.option(
          "--classification/--regression",
          "-c/-r",
          default=True,
          help="Model type: classification (default) or regression.",
        ),
      )
    if include_eos:
      options.insert(
        2,
        click.option(
          "--eos-ids",
          "-e",
          required=False,
          default=None,
          help="Featurizer model IDs (e.g. 'eos8aa5' or 'eos8aa5,eos3l5f'), or a JSON file with featurizer_ids/projection_ids.",
        ),
      )

    for option in reversed(options):
      func = option(func)
    return func

  return _decorator


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
  # is up, also report whether the base images (redis, nginx) are pulled.
  if not _docker_running():
    return (
      "[red]●[/] [bold]Docker[/]: [red]not running[/] [dim](required — start Docker Desktop)[/]"
    )

  def mark(image):
    return "[green]✓[/]" if _docker_image_present(image) else "[red]✗[/]"

  return (
    "[green]●[/] [bold]Docker[/]: [green]running[/] "
    f"[dim]· base images:[/] redis {mark(REDIS_IMAGE)} nginx {mark(NGINX_IMAGE)}"
  )


def _build_cli_help():
  return (
    "[bold green]ZairaChem CLI[/] — Automated QSAR modeling with the Ersilia Model Hub\n\n"
    + _docker_status_line()
    + "\n\n"
    + _isaura_status_line()
  )


class _StatusGroup(click.RichGroup):
  # Compute the title + live Docker/Isaura status lines only when help is actually rendered,
  # so the (potentially slow) `docker info` probe never runs during normal command runs.
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
  # Logging is quiet by default (WARNING/ERROR only); -v streams the full DEBUG log. The file
  # log always keeps everything. Stored on the logger so per-step configure() calls preserve it.
  logger.set_verbosity(verbose)


@cli.command(name="fit", help="Train a QSAR model on labeled molecules.")
@common_options(require_input=True, include_task=True, include_eos=True, include_clean_target=True)
@click.option(
  "--store",
  "-s",
  type=click.Choice(["r", "w", "rw"], case_sensitive=False),
  default=None,
  help="Store access: r=read from isaura-public, w=write to the model's project, rw=both.",
)
@click.option(
  "--nearest-neighbors",
  "-nn",
  is_flag=True,
  default=False,
  help="Use nearest-neighbor search when fetching precalculations.",
)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def fit(
  input_file,
  classification,
  model_dir,
  eos_ids,
  clean,
  flush,
  anonymize,
  clean_target,
  store,
  nearest_neighbors,
  batch_size,
):
  from zairachem.setup.run_fit import run as run_fit

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  store_read = bool(store) and "r" in store.lower()
  store_write = bool(store) and "w" in store.lower()
  run_fit(
    input_file,
    task,
    store_read,
    nearest_neighbors,
    store_write,
    output_dir=model_dir,
    model_ids_file=eos_ids,
  )
  process_group(clean, flush, anonymize, batch_size=batch_size, clean_target=clean_target)


@cli.command(name="predict", help="Run predictions on a trained model.")
@common_options(
  require_input=True, include_task=False, include_eos=False, include_clean_target=True
)
@click.option(
  "--output-dir", "-o", required=False, help="Path to the output directory for predictions."
)
@click.option(
  "--store",
  "-s",
  type=click.Choice(["r", "w", "rw"], case_sensitive=False),
  default=None,
  help="Store access: r=read from isaura-public, w=write to the model's project, rw=both.",
)
@click.option(
  "--nearest-neighbors",
  "-nn",
  is_flag=True,
  default=False,
  help="Use nearest-neighbor search when fetching precalculations.",
)
@click.option(
  "--override-dir",
  is_flag=True,
  default=False,
  help="Overwrite the output directory if it already exists.",
)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def predict(
  input_file,
  model_dir,
  clean,
  flush,
  anonymize,
  clean_target,
  output_dir,
  store,
  nearest_neighbors,
  override_dir,
  batch_size,
):
  from zairachem.setup.run_predict import run as run_predict

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data for prediction[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  if clean_target:
    logger.info(f"[#ff69b4]Clean/flush/anonymize target: {clean_target}[/]")
  store_read = bool(store) and "r" in store.lower()
  store_write = bool(store) and "w" in store.lower()
  run_predict(
    input_file,
    model_dir,
    output_dir,
    override_dir,
    store_read=store_read,
    nn=nearest_neighbors,
    store_write=store_write,
  )
  process_group(clean, flush, anonymize, batch_size=batch_size, clean_target=clean_target)


@cli.command(name="setup", help="Preprocess input molecules into the working directory.")
@common_options(require_input=True, include_task=True, include_eos=True)
@click.option(
  "--store",
  "-s",
  type=click.Choice(["r", "w", "rw"], case_sensitive=False),
  default=None,
  help="Store access: r=read from isaura-public, w=write to the model's project, rw=both.",
)
def setup_cmd(input_file, classification, model_dir, eos_ids, clean, flush, anonymize, store):
  from zairachem.setup.run_fit import run as run_fit

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  store_read = bool(store) and "r" in store.lower()
  store_write = bool(store) and "w" in store.lower()
  run_fit(
    input_file,
    task,
    store_read=store_read,
    store_write=store_write,
    output_dir=model_dir,
    model_ids_file=eos_ids,
  )


@cli.command(name="describe", help="Compute molecular descriptors.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets.",
)
def describe_cmd(clean, flush, anonymize, batch_size):
  from zairachem.describe.descriptors.describe import Describer
  from zairachem.base.utils.isaura_report import report_data_provenance

  logger.configure()
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None, batch_size=batch_size).run()
  report_data_provenance()


@cli.command(name="treat", help="Impute and clean the computed descriptors.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets.",
)
def treat_cmd(clean, flush, anonymize, batch_size):
  from zairachem.treat.imputers.impute import Imputer

  logger.configure()
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None, batch_size=batch_size).run()


@cli.command(name="estimate", help="Train the base estimators.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets.",
)
def estimate_cmd(clean, flush, anonymize, batch_size):
  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  # lazyqsar (pulled in by the estimator) wipes loguru sinks at import; re-assert ours.
  logger.configure()

  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None, batch_size=batch_size).run()


@cli.command(name="pool", help="Aggregate base-estimator predictions by bagging.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets.",
)
def pool_cmd(clean, flush, anonymize, batch_size):
  from zairachem.pool.pipe import PoolerPipeline

  logger.configure()
  logger.debug("[#ff69b4]Running the pooling pipeline[/]")
  PoolerPipeline(path=None, batch_size=batch_size).run()


@cli.command(name="report", help="Generate the performance report and plots.")
@common_options(require_input=False)
@click.option("--plot-name", default=None, help="Optional name for the generated plot.")
def report_cmd(clean, flush, anonymize, plot_name):
  from zairachem.report.report import Reporter

  logger.configure()
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=plot_name).run()


@cli.command(name="finish", help="Finalize the run: clean, flush, and/or anonymize outputs.")
@common_options(require_input=False, include_clean_target=True)
def finish_cmd(clean, flush, anonymize, clean_target):
  from zairachem.finish.finish import Finisher

  logger.configure()
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(
    path=None, clean=clean, flush=flush, anonymize=anonymize, clean_target=clean_target
  ).run()


def main():
  # Baseline log sinks. Heavy pipeline modules are imported lazily inside each command/step to
  # keep startup fast and load each step's dependencies only when it runs; several of those
  # imports wipe loguru's handlers, so logger.configure() is re-asserted after each lazy import.
  logger.configure()
  # Fixed seed so the fit-time row shuffle (get_train_indices) and other RNG draws are
  # reproducible across runs. Note: lazyqsar's internal models expose no seed, so
  # predictions may still vary slightly; descriptor/setup ordering is now deterministic.
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  try:
    cli()
  except Exception:
    # Record the full traceback to the log file (console.log) before exiting, so a crash
    # leaves a diagnosable record instead of only a transient terminal message.
    logger.exception("ZairaChem terminated with an unhandled error")
    raise


if __name__ == "__main__":
  main()
