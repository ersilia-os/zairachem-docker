import sys
import rich_click as click
import rich_click.rich_click as rc
from zairachem.base.utils.logging import logger
from zairachem.setup.run_fit import run as run_fit
from zairachem.setup.run_predict import run as run_predict
from zairachem.describe.descriptors.describe import Describer
from zairachem.estimate.estimators.pipe import EstimatorPipeline
from zairachem.treat.imputers.impute import Imputer
from zairachem.pool.pipe import PoolerPipeline
from zairachem.report.report import Reporter
from zairachem.finish.finish import (
  Finisher,
  CLEAN_TARGET_ALL,
  CLEAN_TARGET_MODEL,
  CLEAN_TARGET_PREDICT,
)


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


def _preprocess_optional_arg(argv, flag, default_value):
  result = []
  i = 0
  while i < len(argv):
    result.append(argv[i])
    if argv[i] in (flag,):
      if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
        result.append(default_value)
    i += 1
  return result


def process_group(clean, flush, anonymize, batch_size=None, clean_target=CLEAN_TARGET_ALL):
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None, batch_size=batch_size).run()

  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None, batch_size=batch_size).run()

  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None, batch_size=batch_size).run()

  logger.debug("[#ff69b4]Running the pooling pipeline to aggregate the result using bagging[/]")
  PoolerPipeline(path=None, batch_size=batch_size).run()

  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=None).run()

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
      click.option("--input-file", "-i", required=require_input, help="Path to the input file."),
      click.option(
        "--model-dir",
        "-m",
        required=False,
        default=None,
        help="Directory where the model is stored.",
      ),
      click.option(
        "--clean",
        is_flag=True,
        help="Whether to clean the descriptors at the end of the run to save space.",
      ),
      click.option(
        "--flush",
        is_flag=True,
        help="Whether to flush the model checkpoints to save space (e.g., in train-test cross-validations).",
      ),
      click.option("--anonymize", is_flag=True, help="Whether to anonymize the inputs entirely."),
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
          help="Type of model, classification or regression.",
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
          help="Featurizer and Projection model ids from the Ersilia Model Hub",
        ),
      )

    for option in reversed(options):
      func = option(func)
    return func

  return _decorator


@click.group(help="[bold green]ZairaChem CLI[/] — run individual steps or full pipelines.")
def cli():
  pass


@cli.command(name="fit", help="Fit a model using ZairaChem")
@common_options(require_input=True, include_task=True, include_eos=True, include_clean_target=True)
@click.option(
  "--enable-store",
  "-es",
  default=None,
  help="Enables reading precalculations from isaura store.",
)
@click.option(
  "--nearest-neighbors",
  "-nn",
  is_flag=True,
  default=False,
  help="Enables nearest neighbor search for fetching calculations.",
)
@click.option(
  "--contribute-store",
  "-cs",
  default=None,
  help="Enables uploading precalculations to isaura store.",
)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Batch size for chunked processing (default: 10000). Controls memory usage for large datasets.",
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
  enable_store,
  nearest_neighbors,
  contribute_store,
  batch_size,
):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  run_fit(
    input_file,
    task,
    enable_store,
    nearest_neighbors,
    contribute_store,
    output_dir=model_dir,
    model_ids_file=eos_ids,
  )
  process_group(clean, flush, anonymize, batch_size=batch_size, clean_target=clean_target)


@cli.command(name="predict", help="Prepare artifacts for prediction (setup for inference).")
@common_options(
  require_input=True, include_task=False, include_eos=False, include_clean_target=True
)
@click.option("--output-dir", "-o", required=False, help="Path to the output model dir.")
@click.option(
  "--enable-store",
  "-es",
  default=None,
  help="Enables reading precalculations from isaura store. Reads from isaura-public by default, or specify a project name (e.g., -es my_project).",
)
@click.option(
  "--nearest-neighbors",
  "-nn",
  is_flag=True,
  default=False,
  help="Enables nearest neighbor search for fetching calculations!",
)
@click.option(
  "--contribute-store",
  "-cs",
  default=None,
  help="Enables uploading precalculations to isaura store. Without a project name, writes to zairatemp, copies to isaura-public, then cleans up. With a project name (e.g., -cs my_project), writes directly to that project.",
)
@click.option(
  "--override-dir",
  is_flag=True,
  default=False,
  help="Override the output dir if it already exists.",
)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Batch size for chunked processing (default: 10000). Controls memory usage for large datasets.",
)
def predict(
  input_file,
  model_dir,
  clean,
  flush,
  anonymize,
  clean_target,
  output_dir,
  enable_store,
  nearest_neighbors,
  contribute_store,
  override_dir,
  batch_size,
):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data for prediction[/]")
  if batch_size:
    logger.info(f"[#ff69b4]Using batch size: {batch_size}[/]")
  if clean_target:
    logger.info(f"[#ff69b4]Clean/flush/anonymize target: {clean_target}[/]")
  run_predict(
    input_file,
    model_dir,
    output_dir,
    override_dir,
    read_store=enable_store,
    nn=nearest_neighbors,
    contribute_store=contribute_store,
  )
  process_group(clean, flush, anonymize, batch_size=batch_size, clean_target=clean_target)


@cli.command(name="setup", help="Preprocess input data and create working artifacts.")
@common_options(require_input=True, include_task=True, include_eos=True)
def setup_cmd(input_file, classification, model_dir, eos_ids, clean, flush, anonymize):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  run_fit(input_file, task, output_dir=model_dir, model_ids_file=eos_ids)


@cli.command(name="describe", help="Compute molecular descriptors.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Batch size for chunked processing.",
)
def describe_cmd(clean, flush, anonymize, batch_size):
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None, batch_size=batch_size).run()


@cli.command(name="treat", help="Impute/clean treated features.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Batch size for chunked processing.",
)
def treat_cmd(clean, flush, anonymize, batch_size):
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None, batch_size=batch_size).run()


@cli.command(name="estimate", help="Train/estimate models.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Batch size for chunked processing.",
)
def estimate_cmd(clean, flush, anonymize, batch_size):
  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None, batch_size=batch_size).run()


@cli.command(name="pool", help="Aggregate results via bagging.")
@common_options(require_input=False)
@click.option(
  "--batch-size",
  "-bs",
  default=None,
  type=int,
  help="Batch size for chunked processing.",
)
def pool_cmd(clean, flush, anonymize, batch_size):
  logger.debug("[#ff69b4]Running the pooling pipeline[/]")
  PoolerPipeline(path=None, batch_size=batch_size).run()


@cli.command(name="report", help="Generate analysis report and plots.")
@common_options(require_input=False)
@click.option("--plot-name", default=None, help="Optional name for the generated plot.")
def report_cmd(clean, flush, anonymize, plot_name):
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=plot_name).run()


@cli.command(name="finish", help="Finalize run: clean, flush caches, and/or anonymize outputs.")
@common_options(require_input=False, include_clean_target=True)
def finish_cmd(clean, flush, anonymize, clean_target):
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(
    path=None, clean=clean, flush=flush, anonymize=anonymize, clean_target=clean_target
  ).run()


@cli.command(name="all", help="Run the entire pipeline end-to-end.")
@common_options(require_input=True, include_clean_target=True)
def run_all(input_file, model_dir, clean, flush, anonymize, clean_target):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_fit(input_file, model_dir)
  process_group(clean, flush, anonymize, clean_target=clean_target)


def main():
  sys.argv = _preprocess_optional_arg(sys.argv, "-es", "isaura-public")
  sys.argv = _preprocess_optional_arg(sys.argv, "--enable-store", "isaura-public")
  sys.argv = _preprocess_optional_arg(sys.argv, "-cs", "zairatemp")
  sys.argv = _preprocess_optional_arg(sys.argv, "--contribute-store", "zairatemp")
  cli()


if __name__ == "__main__":
  main()
