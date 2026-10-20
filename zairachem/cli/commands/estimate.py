import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _MODEL_DIR_HELP
from zairachem.cli.steps import _activate_step


@cli.command(name="estimate", help="Train the per-descriptor base estimators.")
@click.option("--model-dir", "-m", required=True, help=_MODEL_DIR_HELP)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def estimate_cmd(model_dir, batch_size):
  """Fit the per-descriptor estimators."""
  from zairachem.estimate.estimators.pipe import EstimatorPipeline

  # lazyqsar (pulled in by the estimator) wipes loguru sinks at import; re-assert ours.
  logger.configure()

  logger.debug("[#ff69b4]Running the estimator pipeline[/]")
  output_dir = _activate_step(model_dir, "lazy-qsar", "simple_evaluation")
  EstimatorPipeline(path=output_dir, batch_size=batch_size).run()
