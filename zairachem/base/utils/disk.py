"""Free-space checks, so a run says it will not fit before it spends hours finding out.

A descriptor matrix is rows x features x 4 bytes, and a run keeps two of them per descriptor --
the raw one and the scaled one. At a million compounds and 2,048 features that is 8 GB each, so
five descriptors want 80 GB. Running out of it surfaces as whatever error h5py raises on a short
write, hours in, with the run unfinishable and nothing saying why.
"""

import os
import shutil

from zairachem.base.utils.logging import logger


def free_bytes(path):
  """Bytes free on the filesystem holding ``path``, or its nearest existing ancestor."""
  while path and not os.path.exists(path):
    parent = os.path.dirname(path)
    if parent == path:
      break
    path = parent
  try:
    return shutil.disk_usage(path or ".").free
  except OSError as e:  # an unreadable mount point is not a reason to stop the run
    logger.debug(f"[disk] could not measure free space at {path!r}: {e}")
    return None


def path_bytes(path):
  """Bytes a file or directory occupies, following the tree; 0 when it is not there."""
  if os.path.isfile(path):
    return os.path.getsize(path)
  total = 0
  for root, _dirs, files in os.walk(path):
    for name in files:
      try:
        total += os.path.getsize(os.path.join(root, name))
      except OSError:
        continue
  return total


def _human(n):
  """``n`` bytes as a short size: ``"8.2 GB"``, ``"410 MB"``."""
  for unit, scale in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20)):
    if n >= scale:
      return f"{n / scale:.1f} {unit}"
  return f"{n / 2**10:.0f} KB"


def warn_if_short(path, needed, what):
  """Warn when ``needed`` bytes will not fit where the run is being written.

  A warning rather than a refusal: the figure is a projection, the filesystem may be shared, and
  a user who knows better should not be stopped by an estimate. What it buys is the chance to
  stop a run that cannot finish while stopping it is still cheap.

  Parameters
  ----------
  path : str
    Anywhere inside the run; the filesystem holding it is the one measured.
  needed : int
    Bytes the step is about to write.
  what : str
    What is about to be written, for the message.

  Returns
  -------
  bool
    True when the space is short (and a warning was issued).
  """
  free = free_bytes(path)
  if free is None or needed <= free:
    return False
  logger.warning(
    f"[disk] {what} needs about {_human(needed)} and there is {_human(free)} free. The run will "
    "stop when the disk does, part-written. Free some space, or re-run with fewer descriptors."
  )
  return True
