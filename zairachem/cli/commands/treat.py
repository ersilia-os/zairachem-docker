import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _MODEL_DIR_HELP
from zairachem.cli.steps import _activate_step


@cli.command(name="treat", help="Impute and scale the descriptor matrix.")
@click.option("--model-dir", "-m", required=True, help=_MODEL_DIR_HELP)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def treat_cmd(model_dir, batch_size):
  """Treat (scale and impute) the computed descriptors."""
  from zairachem.treat.imputers.impute import Imputer

  logger.configure()
  logger.debug("[#ff69b4]Running the treatment pipeline[/]")
  output_dir = _activate_step(model_dir, "treated_descriptions")
  Imputer(path=output_dir, batch_size=batch_size).run()
