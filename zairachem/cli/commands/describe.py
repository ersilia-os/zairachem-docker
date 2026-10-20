import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _DESCRIBE_WORKERS_HELP, _MODEL_DIR_HELP
from zairachem.cli.steps import _activate_step


@cli.command(name="describe", help="Compute molecular descriptors for each featurizer.")
@click.option("--model-dir", "-m", required=True, help=_MODEL_DIR_HELP)
@click.option(
  "--batch-size",
  "-b",
  default=None,
  type=int,
  help="Rows per chunk when processing large datasets (default: 10000).",
)
@click.option("--workers", "describe_workers", default=None, type=int, help=_DESCRIBE_WORKERS_HELP)
def describe_cmd(model_dir, batch_size, describe_workers):
  """Compute the descriptors of an already prepared run folder."""
  from zairachem.base.utils.isaura_report import report_data_provenance
  from zairachem.describe.descriptors.describe import Describer

  logger.configure()
  logger.debug("[#ff69b4]Running the descriptor computation pipeline[/]")
  output_dir = _activate_step(model_dir, "raw_descriptions")
  Describer(path=output_dir, batch_size=batch_size, workers=describe_workers).run()
  report_data_provenance(output_dir)


# Projections (the report's 2-D embedding) run as part of the fit pipeline (process_group, between
# describe and treat) — they are not exposed as a standalone command.
