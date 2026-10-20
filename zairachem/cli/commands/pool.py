import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _MODEL_DIR_HELP
from zairachem.cli.steps import _activate_step


@cli.command(name="pool", help="Combine the per-descriptor estimators into a consensus.")
@click.option("--model-dir", "-m", required=True, help=_MODEL_DIR_HELP)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
def pool_cmd(model_dir, batch_size):
  """Pool the per-descriptor predictions into the final prediction."""
  from zairachem.pool.pipe import PoolerPipeline

  logger.configure()
  logger.debug("[#ff69b4]Running the pooling pipeline[/]")
  output_dir = _activate_step(model_dir, "pool")
  PoolerPipeline(path=output_dir, batch_size=batch_size).run()
