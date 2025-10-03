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
from zairachem.finish.finish import Finisher

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


def process_group(clean, flush, anonymize):
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None).run()

  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None).run()

  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None).run(time_budget_sec=None)

  logger.debug("[#ff69b4]Running the pooling pipeline to aggregate the result using bagging[/]")
  PoolerPipeline(path=None).run(time_budget_sec=None)

  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=None).run()

  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(path=None, clean=clean, flush=flush, anonymize=anonymize).run()


def common_options(require_input: bool = True):
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
        "--cutoff",
        "-c",
        required=False,
        default=None,
        help="Cutoff threshold (e.g. probability or value).",
      ),
      click.option(
        "--direction",
        "-d",
        required=False,
        default=None,
        help="Direction of processing (e.g. forward or backward).",
      ),
      click.option(
        "--parameters",
        "-p",
        required=False,
        default=None,
        help="Additional model parameters as a string.",
      ),
      click.option("--clean", is_flag=True, help="Whether to run in clean mode."),
      click.option("--flush", is_flag=True, help="Whether to flush caches or temporary files."),
      click.option("--anonymize", is_flag=True, help="Whether to anonymize outputs."),
    ]
    for option in reversed(options):
      func = option(func)
    return func

  return _decorator


@click.group(help="[bold green]ZairaChem CLI[/] — run individual steps or full pipelines.")
def cli():
  pass


@cli.command(name="setup", help="Preprocess input data and create working artifacts.")
@common_options(require_input=True)
def setup_cmd(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_fit(input_file, model_dir, cutoff, direction, parameters)


@cli.command(name="fit", help="Alias for 'setup'.")
@common_options(require_input=True)
def fit(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_fit(input_file, model_dir, cutoff, direction, parameters)
  process_group(clean, flush, anonymize)


@cli.command(name="predict", help="Prepare artifacts for prediction (setup for inference).")
@common_options(require_input=True)
@click.option("--output-dir", "-o", required=False, help="Path to the output model dir.")
@click.option(
  "--override-dir",
  is_flag=True,
  default=False,
  help="Override the output dir if it already exists.",
)
def predict(
  input_file,
  model_dir,
  cutoff,
  direction,
  parameters,
  clean,
  flush,
  anonymize,
  output_dir,
  override_dir,
):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data for prediction[/]")
  run_predict(input_file, model_dir, output_dir, override_dir)


@cli.command(name="describe", help="Compute molecular descriptors.")
@common_options(require_input=False)
def describe_cmd(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None).run()


@cli.command(name="treat", help="Impute/clean treated features.")
@common_options(require_input=False)
def treat_cmd(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None).run()


@cli.command(name="estimate", help="Train/estimate models.")
@common_options(require_input=False)
@click.option("--time-budget-sec", type=int, default=None, help="Optional time budget in seconds.")
def estimate_cmd(
  input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize, time_budget_sec
):
  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None).run(time_budget_sec=time_budget_sec)


@cli.command(name="pool", help="Aggregate results via bagging/ensembling.")
@common_options(require_input=False)
@click.option("--time-budget-sec", type=int, default=None, help="Optional time budget in seconds.")
def pool_cmd(
  input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize, time_budget_sec
):
  logger.debug("[#ff69b4]Running the pooling pipeline[/]")
  PoolerPipeline(path=None).run(time_budget_sec=time_budget_sec)


@cli.command(name="report", help="Generate analysis report and plots.")
@common_options(require_input=False)
@click.option("--plot-name", default=None, help="Optional name for the generated plot.")
def report_cmd(
  input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize, plot_name
):
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=plot_name).run()


@cli.command(name="finish", help="Finalize run: clean, flush caches, and/or anonymize outputs.")
@common_options(require_input=False)
def finish_cmd(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(path=None, clean=clean, flush=flush, anonymize=anonymize).run()


@cli.command(name="all", help="Run the entire pipeline end-to-end.")
@common_options(require_input=True)
def run_all(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_fit(input_file, model_dir, cutoff, direction, parameters)
  process_group(clean, flush, anonymize)


if __name__ == "__main__":
  cli()
