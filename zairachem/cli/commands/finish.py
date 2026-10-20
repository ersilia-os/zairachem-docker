import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _MODEL_DIR_HELP
from zairachem.cli.steps import _activate_step


@cli.command(name="finish", help="Assemble final outputs and clean up intermediate data.")
@click.option("--model-dir", "-m", required=True, help=_MODEL_DIR_HELP)
@click.option(
  "--anonymize",
  is_flag=True,
  help="Blank out molecule structures (SMILES / InChIKey) in all outputs.",
)
@click.option(
  "--keep-intermediate-data",
  is_flag=True,
  default=False,
  help="Keep ALL intermediate data (matrices, projections, input copies, predict model/ tree); by default they are "
  "cleaned. The fitted transformers are always kept.",
)
def finish_cmd(model_dir, anonymize, keep_intermediate_data):
  """Finish the run: write the final tables and clean up intermediate data."""
  from zairachem.finish.finish import Finisher

  logger.configure()
  logger.debug("[#ff69b4]Running the finishing pipeline[/]")
  output_dir = _activate_step(model_dir, "finish")
  Finisher(
    path=output_dir,
    anonymize=anonymize,
    keep_intermediate_data=keep_intermediate_data,
  ).run()
