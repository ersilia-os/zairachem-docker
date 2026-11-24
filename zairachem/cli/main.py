import random
import sys

from zairachem.base.utils.lazyqsar_version import check_lazyqsar_version
from zairachem.base.utils.logging import logger
from zairachem.base.vars import RANDOM_SEED
from zairachem.cli.create_cli import COMMANDS, cli

_CHECKED_COMMANDS = frozenset(COMMANDS) - {"stop"}


def main():
  """Entry point of the ``zairachem`` command: seed the RNGs and run the CLI."""
  # Baseline log sinks. Heavy pipeline modules are imported lazily inside each command/step to
  # keep startup fast and load each step's dependencies only when it runs; several of those
  # imports wipe loguru's handlers, so logger.configure() is re-asserted after each lazy import.
  logger.configure()
  # Every pipeline command stops when the installed LazyQSAR is not the pinned release. Help screens,
  # `stop` (it only removes containers) and command lines click itself rejects (no command, a typo)
  # run on any LazyQSAR.
  if not _help_only(sys.argv[1:]) and _command(sys.argv[1:]) in _CHECKED_COMMANDS:
    check_lazyqsar_version()
  # Fixed seed so the fit-time row shuffle (get_train_indices) and other RNG draws are
  # reproducible across runs; each descriptor's training is seeded again just before it starts
  # (estimate), so a trained model does not depend on what ran before it.
  random.seed(RANDOM_SEED)
  if not _help_only(sys.argv[1:]):
    import numpy as np  # not needed (nor imported, ~60 ms) just to print a help screen

    np.random.seed(RANDOM_SEED)
  try:
    cli()
  except Exception:
    # Record the full traceback to the log file (console.log) before exiting, so a crash
    # leaves a diagnosable record instead of only a transient terminal message.
    logger.exception("ZairaChem terminated with an unhandled error")
    raise


def _help_only(args):
  """Whether the command line only asks for a help screen (nothing will be computed)."""
  return not args or any(a in ("-h", "--help") for a in args)


def _command(args):
  """The subcommand named on the command line (the group's only options are flags), or None."""
  return next((a for a in args if not a.startswith("-")), None)


if __name__ == "__main__":
  main()
