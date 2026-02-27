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
from zairachem.interpret.interpret import Interpreter

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
  EstimatorPipeline(path=None).run()

  logger.debug("[#ff69b4]Running the pooling pipeline to aggregate the result using bagging[/]")
  PoolerPipeline(path=None).run()

  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=None).run()

  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(path=None, clean=clean, flush=flush, anonymize=anonymize).run()

def common_options(
  require_input: bool = True, include_task: bool = False, include_eos: bool = False
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
        help="Whether to flush the model checkpoints to save space for example in train-test crossvalidations)",
      ),
      click.option("--anonymize", is_flag=True, help="Whether to anonymize the inputs entirely."),
    ]
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
@common_options(require_input=True, include_task=True, include_eos=True)
@click.option(
  "--enable-store",
  "-es",
  is_flag=True,
  default=False,
  help="Enables cache fetching and saving from isaura",
)
@click.option(
  "--access", "-a", default="public", help="Cache reading access level [either public or private]"
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
  is_flag=True,
  default=False,
  help="Enables to copy or contribute caches to the default buckets!",
)
@click.option(
  "--interpret-substructures",
  "-is",
  is_flag=True,
  default=False,
  help="Produce substructure feature contributions for each individual molecule.",
)

def fit(
  input_file,
  classification,
  model_dir,
  eos_ids,
  clean,
  flush,
  anonymize,
  enable_store,
  access,
  nearest_neighbors,
  contribute_store,
  interpret_substructures,
):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  run_fit(
    input_file,
    task,
    enable_store,
    access,
    nearest_neighbors,
    contribute_store,
    output_dir=model_dir,
    model_ids_file=eos_ids,
  )
  process_group(clean, flush, anonymize)
  logger.debug("[#ff69b4]Running the interpretability pipeline[/]")
  Interpreter(path=None, interpret_substructures=interpret_substructures).run()


@cli.command(name="predict", help="Prepare artifacts for prediction (setup for inference).")
@common_options(require_input=True, include_task=False, include_eos=False)
@click.option("--output-dir", "-o", required=False, help="Path to the output model dir.")
@click.option(
  "--override-dir",
  is_flag=True,
  default=False,
  help="Override the output dir if it already exists.",
)
@click.option(
  "--interpret-substructures",
  "-is",
  is_flag=True,
  default=False,
  help="Produce substructure feature contributions for each individual molecule.",
)

def predict(
  input_file,
  model_dir,
  clean,
  flush,
  anonymize,
  output_dir,
  override_dir,
  interpret_substructures,
):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data for prediction[/]")
  run_predict(input_file, model_dir, output_dir, override_dir)
  process_group(clean, flush, anonymize)
  logger.debug("[#ff69b4]Running the interpretability pipeline[/]")
  Interpreter(path=None, interpret_substructures=interpret_substructures).run()


@cli.command(name="setup", help="Preprocess input data and create working artifacts.")
@common_options(require_input=True, include_task=True, include_eos=True)
def setup_cmd(input_file, classification, model_dir, eos_ids):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  if classification:
    task = "classification"
  else:
    task = "regression"
  run_fit(input_file, task, output_dir=model_dir, model_ids_file=eos_ids)


@cli.command(name="describe", help="Compute molecular descriptors.")
@common_options(require_input=False)
def describe_cmd():
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  Describer(path=None).run()


@cli.command(name="treat", help="Impute/clean treated features.")
@common_options(require_input=False)
def treat_cmd():
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  Imputer(path=None).run()


@cli.command(name="estimate", help="Train/estimate models.")
@common_options(require_input=False)
def estimate_cmd():
  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  EstimatorPipeline(path=None).run()


@cli.command(name="pool", help="Aggregate results via bagging.")
@common_options(require_input=False)
def pool_cmd():
  logger.debug("[#ff69b4]Running the pooling pipeline[/]")
  PoolerPipeline(path=None).run()


@cli.command(name="report", help="Generate analysis report and plots.")
@common_options(require_input=False)
@click.option("--plot-name", default=None, help="Optional name for the generated plot.")
def report_cmd(plot_name):
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  Reporter(path=None, plot_name=plot_name).run()


@cli.command(name="finish", help="Finalize run: clean, flush caches, and/or anonymize outputs.")
@common_options(require_input=False)
def finish_cmd(clean, flush, anonymize):
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  Finisher(path=None, clean=clean, flush=flush, anonymize=anonymize).run()


@cli.command(name="interpret", help="Produce explanations for predictions")
@common_options(require_input=False)
@click.option(
  "--interpret-substructures",
  "-is",
  is_flag=True,
  default=False,
  help="Produce substructure feature contributions for each individual molecule.",
)
def interpret_cmd(interpret_substructures):
  logger.debug("[#ff69b4]Running the interpretability pipeline[/]")
  Interpreter(path=None, interpret_substructures=interpret_substructures).run()


@cli.command(name="all", help="Run the entire pipeline end-to-end.")
@common_options(require_input=True)
def run_all(input_file, model_dir, cutoff, direction, parameters, clean, flush, anonymize):
  logger.info("[#ff69b4]Running the setup pipeline to preprocess the input data[/]")
  run_fit(input_file, model_dir, cutoff, direction, parameters)
  process_group(clean, flush, anonymize)


if __name__ == "__main__":
  cli()
