from zairachem.base.utils.logging import logger
from zairachem.cli.create_cli import cli


# Hidden from the command list; the --help status line points to it while servers are up.
@cli.command(
  name="stop", help="Stop the Ersilia model servers kept running between runs.", hidden=True
)
def stop_cmd():
  """Remove every model server container (and their redis cache) that runs keep up."""
  from zairachem.base.utils.console import echo
  from zairachem.describe.descriptors.compose import stop_model_servers

  logger.configure()
  stop_model_servers()
  echo("Model servers stopped.", kind="success")
