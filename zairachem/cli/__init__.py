# Importing create_cli runs the module-level setup (matplotlib log level, rich-click styling) and
# defines the group; create_cli() then registers the commands in their --help order.
from zairachem.cli.create_cli import create_cli

cli = create_cli()

from zairachem.cli.main import main  # noqa: E402

__all__ = ["cli", "main"]
