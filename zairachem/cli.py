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
  desc = Describer(path=None)
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  desc.run()

  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  impute = Imputer(path=None)
  impute.run()

  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  ep = EstimatorPipeline(path=None)
  ep.run(time_budget_sec=None)

  logger.debug("[#ff69b4]Running the pooling pipeline to aggregate the result using bagging[/]")
  ep = PoolerPipeline(path=None)
  ep.run(time_budget_sec=None)

  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  r = Reporter(path=None, plot_name=None)
  r.run()

  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  f = Finisher(path=None, clean=clean, flush=flush, anonymize=anonymize)
  f.run()


def common_options(func):
  options = [
    click.option("--input-file", "-i", required=True, help="Path to the input file."),
    click.option(
      "--model-dir", "-m", required=False, default=None, help="Directory where the model is stored."
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
    click.option("--flush", is_flag=True, help="Whether to flush any caches or temporary files."),
    click.option("--anonymize", is_flag=True, help="Whether to anonymize outputs."),
  ]
  for option in reversed(options):
    func = option(func)
  return func


@click.group()
def cli():
  pass


@cli.command()
@common_options
def fit(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_fit(input_file, model_dir, cutoff, direction, parameters)
  process_group(clean, flush, anonymize)


@cli.command()
@common_options
@click.option("--output-dir", "-o", required=False, help="Path to the output model dir.")
@click.option(
  "--override-dir",
  required=False,
  is_flag=True,
  default=False,
  help="Path to the output model dir.",
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
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_predict(input_file, model_dir, output_dir, override_dir)
  process_group(clean, flush, anonymize)


if __name__ == "__main__":
  cli()
