import random
import sys

from zairachem.base.utils.logging import logger
from zairachem.base.vars import RANDOM_SEED
from zairachem.cli.create_cli import cli


def main():
  """Entry point of the ``zairachem`` command: seed the RNGs and run the CLI."""
  # Baseline log sinks. Heavy pipeline modules are imported lazily inside each command/step to
  # keep startup fast and load each step's dependencies only when it runs; several of those
  # imports wipe loguru's handlers, so logger.configure() is re-asserted after each lazy import.
  logger.configure()
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


if __name__ == "__main__":
  main()
