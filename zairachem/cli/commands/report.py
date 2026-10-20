import rich_click as click

from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli
from zairachem.cli.options import _MODEL_DIR_HELP
from zairachem.cli.steps import _activate_step


@cli.command(name="report", help="Render the plots, tables and HTML report.")
@click.option("--model-dir", "-m", required=True, help=_MODEL_DIR_HELP)
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
def report_cmd(model_dir, plot_name, no_report):
  """Render the report plots and tables."""
  from zairachem.report.report import Reporter

  logger.configure()
  logger.debug("[#ff69b4]Running the reporting pipeline[/]")
  output_dir = _activate_step(model_dir, "report")
  Reporter(path=output_dir, plot_name=plot_name, make_plots=not no_report).run()
