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
# use them so that `zairachem --help` and argument parsing stay fast.

# Silence matplotlib's "Matplotlib is building the font cache; this may take a moment." notice,
# which it logs on first import — set here (earliest entry point) before any matplotlib import.
import logging as _logging

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


def process_group(
  anonymize,
  batch_size=None,
  keep_intermediate_data=False,
  no_report=False,
  describe_workers=None,
):
  # Each pipeline class is imported right before its step runs, so heavy dependencies load only
  # when that step executes (matplotlib for reporting, lazyqsar/xgboost/onnx for estimation) —
  # not all at once at the start. Several of these imports call loguru's logger.remove(), so
  # logger.configure() is re-asserted after each one to keep zairachem's log sinks alive.
  # The shared `tracker` (begun in fit/predict) shows which step is running; start()/complete()
  # are no-ops if the tracker was not begun (e.g. standalone step commands).
  from zairachem.base.utils.progress import SUMMARIES, final_summary_panel, tracker

  from zairachem.describe.descriptors.describe import Describer

  logger.configure()
  tracker.start("describe")
  Describer(path=None, batch_size=batch_size, workers=describe_workers).run()
  tracker.complete("describe", SUMMARIES["describe"]())

  # Projections are an independent 2-D embedding shown as-is in the report (NOT a transformation of
  # the descriptors) — their own step, run while the model containers from Describe are still up.
  from zairachem.treat.imputers.manifolds import Manifolds
  from zairachem.base.utils.isaura_report import report_data_provenance

  logger.configure()
  tracker.start("projections")
  Manifolds(batch_size=batch_size).run()
  # Rendered here so it reflects BOTH featurizers (describe) and projectors (this step).
  report_data_provenance()
  tracker.complete("projections", SUMMARIES["projections"]())

  from zairachem.treat.imputers.impute import Imputer

  logger.configure()
  tracker.start("treat")
  Imputer(path=None, batch_size=batch_size).run()
  tracker.complete("treat", SUMMARIES["treat"]())

  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  logger.configure()
  tracker.start("estimate")
  EstimatorPipeline(path=None, batch_size=batch_size).run()
  tracker.complete("estimate", SUMMARIES["estimate"]())

  from zairachem.pool.pipe import PoolerPipeline

  logger.configure()
  tracker.start("pool")
  PoolerPipeline(path=None, batch_size=batch_size).run()
  tracker.complete("pool", SUMMARIES["pool"]())

  from zairachem.report.report import Reporter

  logger.configure()
  tracker.start("report")
  Reporter(path=None, plot_name=None, make_plots=not no_report).run()
  tracker.complete("report", SUMMARIES["report"]())

  from zairachem.finish.finish import Finisher

  logger.configure()
  tracker.start("finish")
  Finisher(
    path=None,
    anonymize=anonymize,
    keep_intermediate_data=keep_intermediate_data,
  ).run()
  tracker.complete("finish", SUMMARIES["finish"]())

  final_summary_panel()


# `--store` is an optional-value option: omitted -> None (no store); `--store` alone -> this
# sentinel (resolve to a default name); `--store X` -> the literal X.
_STORE_SENTINEL = "\x00default"


def _resolve_store(value, default):
  """Resolve the --store value to a concrete isaura project name, or None for 'no store'.

  ``value`` is None (omitted), the sentinel (bare ``--store`` → use ``default``), or an explicit
  name. The result is sanitized to a valid bucket name; ``isaura-public`` passes through unchanged.
  """
  if value is None:
    return None
  from zairachem.base.utils.isaura_report import sanitize_project_name

  name = default if value == _STORE_SENTINEL else value
  return sanitize_project_name(name) if name else None


def _trained_store(model_dir):
  """The store name persisted in a trained model's parameters.json (for predict reuse), or None."""
  if not model_dir:
    return None
  import json

  from zairachem.base.vars import METADATA_SUBFOLDER, PARAMETERS_FILE

  try:
    with open(os.path.join(model_dir, METADATA_SUBFOLDER, PARAMETERS_FILE)) as f:
      p = json.load(f)
    return p.get("store") or p.get("contribute_store") or p.get("read_store")
  except Exception:
    return None


_STORE_HELP = (
  "Cache descriptor precalculations in an isaura project to speed up re-runs. Omit for no store; "
  "bare --store uses the model-directory name; --store NAME uses NAME; --store isaura-public "
  "reads/writes the shared public lake directly."
)

_STORE_HELP_PREDICT = (
  "isaura store to read/write. Defaults to the store the model was trained with (omit or bare "
  "--store); --store NAME overrides; --store isaura-public uses the shared public lake."
)

_DESCRIBE_WORKERS_HELP = (
  "How many descriptor models to featurize in parallel (default: 1 = serial). They share host "
  "CPU/RAM/GPU through Docker, so raise this modestly."
)


def common_options(
  require_input: bool = True,
  include_task: bool = False,
  include_eos: bool = False,
  include_anonymize: bool = False,
  require_model: bool = False,
):
  def _decorator(func):
    options = [
      click.option(
        "--input-file",
        "-i",
        required=require_input,
        help="Input CSV of molecules (must contain a SMILES column).",
      ),
      click.option(
        "--model-dir",
        "-m",
        required=require_model,
        default=None,
        help="Model directory — created here when fitting; the trained model to use when predicting.",
      ),
    ]
    # Build in display order: required I/O (input, model) first, then task / featurizer-ids /
    # projection-ids, and anonymize last. (Commands that want anonymize grouped with their own
    # output-control flags declare it themselves instead of via include_anonymize.)
    if include_task:
      options.append(
        click.option(
          "--classification/--regression",
          "-c/-r",
          default=True,
          help="Model type: classification (default) or regression.",
        )
      )
    if include_eos:
      options.append(
        click.option(
          "--featurizer-ids",
          "-f",
          "featurizer_ids",
          required=False,
          default=None,
          help="Descriptor model IDs, comma-separated (e.g. eos8aa5,eos3l5f), or a JSON file with featurizer_ids/projection_ids keys. Default: a curated descriptor set.",
        )
      )
      options.append(
        click.option(
          "--projection-ids",
          "-p",
          "projection_ids",
          required=False,
          default=None,
          help="Projection model ID(s) for the report's 2-D embedding (default: eos1klk). Overrides any projection_ids in a --featurizer-ids JSON file.",
        )
      )
    if include_anonymize:
      options.append(
        click.option(
          "--anonymize",
          is_flag=True,
          help="Blank out molecule structures (SMILES / InChIKey) in all outputs.",
        )
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


@cli.command(name="fit", help="Train a QSAR model from a labelled CSV.")
# Required I/O (input, model from common_options) → task / featurizers → run config (store, override,
# batch, workers) → output controls (skip-report, keep-intermediate-data, anonymize). `--anonymize`
# is declared here (not via common_options) so it sits with the other output controls.
@common_options(require_input=True, include_task=True, include_eos=True, require_model=True)
@click.option(
  "--store", "-s", is_flag=False, flag_value=_STORE_SENTINEL, default=None, help=_STORE_HELP
)
@click.option(
  "--override",
  "override_dir",
  is_flag=True,
  default=False,
  help="Overwrite the model directory if it already exists (otherwise the run aborts).",
)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
@click.option("--workers", "describe_workers", default=None, type=int, help=_DESCRIBE_WORKERS_HELP)
@click.option(
  "--skip-report",
  "no_report",
  is_flag=True,
  default=False,
  help="Skip the plots and the HTML report (the slow, bulky part). The prediction and performance "
  "tables in results/ are still written.",
)
@click.option(
  "--keep-intermediate-data",
  is_flag=True,
  default=False,
  help="Keep intermediate data (descriptor matrices, 2-D projections, raw input copies) instead of "
  "cleaning it at the end. The trained model and results/report are always kept either way.",
)
@click.option(
  "--anonymize",
  is_flag=True,
  help="Blank out molecule structures (SMILES / InChIKey) in all outputs.",
)
def fit(
  input_file,
  classification,
  model_dir,
  featurizer_ids,
  projection_ids,
  anonymize,
  store,
  override_dir,
  batch_size,
  keep_intermediate_data,
  no_report,
  describe_workers,
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
  from zairachem.base.utils.progress import tracker

  store = _resolve_store(store, os.path.basename(os.path.normpath(model_dir)))
  tracker.begin(
    "ZairaChem · QSAR training",
    subtitle=f"{os.path.basename(input_file)} → {os.path.basename(os.path.normpath(model_dir))} · {task}",
  )
  proceed = run_fit(
    input_file,
    task,
    store,
    output_dir=model_dir,
    model_ids_file=featurizer_ids,
    projection_ids=projection_ids,
    override=override_dir,
  )
  if proceed:
    process_group(
      anonymize,
      batch_size=batch_size,
      keep_intermediate_data=keep_intermediate_data,
      no_report=no_report,
      describe_workers=describe_workers,
    )
  else:
    from zairachem.base.utils.progress import final_summary_panel

    final_summary_panel()


@cli.command(name="predict", help="Predict activities for new molecules with a trained model.")
# Options render in this order: required I/O (input, model from common_options; then output) →
# run config (store, override, batch, workers) → output controls (skip-report, keep-intermediate
# -data, anonymize). `--anonymize` is declared here (not via common_options) so it sits with the
# other output controls instead of between the required --model-dir and --output-dir.
@common_options(require_input=True, require_model=True)
@click.option(
  "--output-dir", "-o", required=True, help="Directory to write the predictions and report into."
)
@click.option(
  "--store", "-s", is_flag=False, flag_value=_STORE_SENTINEL, default=None, help=_STORE_HELP_PREDICT
)
@click.option(
  "--override",
  "override_dir",
  is_flag=True,
  default=False,
  help="Overwrite the output directory if it already exists (otherwise the run aborts).",
)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
@click.option("--workers", "describe_workers", default=None, type=int, help=_DESCRIBE_WORKERS_HELP)
@click.option(
  "--skip-report",
  "no_report",
  is_flag=True,
  default=False,
  help="Skip the plots and the HTML report. The prediction and performance tables in results/ are "
  "still written.",
)
@click.option(
  "--keep-intermediate-data",
  is_flag=True,
  default=False,
  help="Keep the prediction run's intermediate data (descriptor matrices, projections, input copies, "
  "and the whole pipeline/). By default a finished predict run keeps only the results and report.",
)
@click.option(
  "--anonymize",
  is_flag=True,
  help="Blank out molecule structures (SMILES / InChIKey) in all outputs.",
)
def predict(
  input_file,
  model_dir,
  anonymize,
  output_dir,
  store,
  override_dir,
  batch_size,
  describe_workers,
  keep_intermediate_data,
  no_report,
):
  from zairachem.setup.run_predict import run as run_predict

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data for prediction[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  # Default to the store the model was trained with (reuse its cache) — whether --store is omitted OR
  # given bare. An explicit --store NAME overrides; --store isaura-public uses the shared lake.
  trained = _trained_store(model_dir)
  if store is None or store == _STORE_SENTINEL:
    store = trained
  else:
    store = _resolve_store(store, trained)
  # run_predict validates the model folder and prints the run header only once it's confirmed ready.
  proceed = run_predict(
    input_file,
    model_dir,
    output_dir,
    override_dir,
    store=store,
  )
  if proceed:
    process_group(
      anonymize,
      batch_size=batch_size,
      keep_intermediate_data=keep_intermediate_data,
      no_report=no_report,
      describe_workers=describe_workers,
    )
  else:
    from zairachem.base.utils.progress import final_summary_panel

    final_summary_panel()


@cli.command(name="setup", help="Standardize and prepare the input molecules.")
@common_options(require_input=True, include_task=True, include_eos=True)
@click.option(
  "--store", "-s", is_flag=False, flag_value=_STORE_SENTINEL, default=None, help=_STORE_HELP
)
def setup_cmd(
  input_file,
  classification,
  model_dir,
  featurizer_ids,
  projection_ids,
  store,
):
  from zairachem.setup.run_fit import run as run_fit

  logger.configure()
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  _model = model_dir or os.path.splitext(os.path.basename(input_file))[0]
  store = _resolve_store(store, os.path.basename(os.path.normpath(_model)))
  run_fit(
    input_file,
    task,
    store,
    output_dir=model_dir,
    model_ids_file=featurizer_ids,
    projection_ids=projection_ids,
  )


@cli.command(name="describe", help="Compute molecular descriptors for each featurizer.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
@click.option("--workers", "describe_workers", default=None, type=int, help=_DESCRIBE_WORKERS_HELP)
def describe_cmd(batch_size, describe_workers):
  from zairachem.describe.descriptors.describe import Describer
  from zairachem.base.utils.isaura_report import report_data_provenance

  logger.configure()
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None, batch_size=batch_size, workers=describe_workers).run()
  report_data_provenance()


@cli.command(name="projections", help="Compute the 2-D projections shown in the report.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def projections_cmd(batch_size):
  from zairachem.treat.imputers.manifolds import Manifolds
  from zairachem.base.utils.isaura_report import report_data_provenance

  logger.configure()
  logger.debug("[#ff69b4]Computing 2-D projections[/]")
  Manifolds(batch_size=batch_size).run()
  report_data_provenance()


@cli.command(name="treat", help="Impute and scale the descriptor matrix.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def treat_cmd(batch_size):
  from zairachem.treat.imputers.impute import Imputer

  logger.configure()
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None, batch_size=batch_size).run()


@cli.command(name="estimate", help="Train the per-descriptor base estimators.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def estimate_cmd(batch_size):
  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  # lazyqsar (pulled in by the estimator) wipes loguru sinks at import; re-assert ours.
  logger.configure()

  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None, batch_size=batch_size).run()


@cli.command(name="pool", help="Combine the per-descriptor estimators into a consensus.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def pool_cmd(batch_size):
  from zairachem.pool.pipe import PoolerPipeline

  logger.configure()
  logger.debug("[#ff69b4]Running the pooling pipeline[/]")
  PoolerPipeline(path=None, batch_size=batch_size).run()


@cli.command(name="report", help="Render the plots, tables and HTML report.")
@common_options(require_input=False)
@click.option(
  "--plot-name", default=None, help="Render only the named plot instead of the full report."
)
@click.option(
  "--skip-report",
  "no_report",
  is_flag=True,
  default=False,
  help="Skip the plots and the HTML report; still write the prediction and performance tables.",
)
def report_cmd(plot_name, no_report):
  from zairachem.report.report import Reporter

  logger.configure()
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=plot_name, make_plots=not no_report).run()


@cli.command(name="finish", help="Assemble final outputs and clean up intermediate data.")
@common_options(require_input=False, include_anonymize=True)
@click.option(
  "--keep-intermediate-data",
  is_flag=True,
  default=False,
  help="Keep ALL intermediate data (matrices, projections, input copies, predict pipeline); by default they are "
  "cleaned. The fitted transformers are always kept.",
)
def finish_cmd(anonymize, keep_intermediate_data):
  from zairachem.finish.finish import Finisher

  logger.configure()
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(
    path=None,
    anonymize=anonymize,
    keep_intermediate_data=keep_intermediate_data,
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
