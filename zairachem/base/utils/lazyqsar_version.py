"""Stop a run when the installed LazyQSAR is not the release this ZairaChem pins.

ZairaChem is installed with ``pip install -e .`` from a git clone, so a ``git pull`` changes the code
at once while LazyQSAR stays at the release installed before. The pin is mirrored in
``LAZYQSAR_VERSION`` (code, so a pull always brings it) and compared with the installed release's
metadata, without importing LazyQSAR, so the check costs nothing at startup. It reads the first
``lazyqsar`` distribution metadata on ``sys.path``; a source checkout put ahead of it (``PYTHONPATH``)
is not seen.
"""

import os
import sys
from importlib import metadata

from rich.markup import escape

from zairachem.base.utils.console import echo
from zairachem.base.utils.logging import logger
from zairachem.base.vars import LAZYQSAR_VERSION

#: Set to ``1`` to run with a different LazyQSAR anyway (a warning instead of a stop).
ALLOW_MISMATCH_ENV = "ZAIRACHEM_ALLOW_LAZYQSAR_MISMATCH"

_FIX = (
  "Re-run `pip install -e .` in the ZairaChem folder (or `bash install.sh`), then run the command "
  "again."
)


def check_lazyqsar_version():
  """Stop with exit code 1 unless the installed LazyQSAR is ``LAZYQSAR_VERSION``.

  With ``ZAIRACHEM_ALLOW_LAZYQSAR_MISMATCH=1`` a different release only warns. A missing LazyQSAR
  always stops: no fit or predict can run without it.

  Raises
  ------
  SystemExit
      When LazyQSAR is not installed, or its release differs and the override is not set.
  """
  try:
    installed = metadata.version("lazyqsar")
  except metadata.PackageNotFoundError:
    logger.info(f"LazyQSAR is not installed in {sys.prefix}; ZairaChem needs {LAZYQSAR_VERSION}")
    echo(
      f"LazyQSAR is not installed in this environment ({escape(sys.prefix)}). ZairaChem needs LazyQSAR "
      f"{LAZYQSAR_VERSION}.",
      kind="error",
    )
    echo(_FIX, kind="info")
    raise SystemExit(1) from None
  # The pin is ``==X.Y.Z``, which a local build (``X.Y.Z+cpu``) also meets.
  if installed.split("+", 1)[0] == LAZYQSAR_VERSION:
    return
  env = escape(sys.prefix)
  if os.environ.get(ALLOW_MISMATCH_ENV) == "1":
    logger.info(
      f"LazyQSAR {installed} installed, {LAZYQSAR_VERSION} pinned; going on ({ALLOW_MISMATCH_ENV}=1)"
    )
    echo(
      f"Running on LazyQSAR {escape(installed)}, not the {LAZYQSAR_VERSION} this ZairaChem pins "
      f"({ALLOW_MISMATCH_ENV}=1): models and scores may differ from a pinned install.",
      kind="warning",
    )
    return
  logger.info(f"LazyQSAR {installed} installed in {sys.prefix}, {LAZYQSAR_VERSION} pinned; stopped")
  echo(
    f"ZairaChem needs LazyQSAR {LAZYQSAR_VERSION}, but this environment ({env}) has "
    f"{escape(installed)}. This usually follows a `git pull` that moved the pin.",
    kind="error",
  )
  echo(_FIX, kind="info")
  echo(f"To run anyway (a deliberate cross-version test), set {ALLOW_MISMATCH_ENV}=1.", kind="info")
  raise SystemExit(1)
