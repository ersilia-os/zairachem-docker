"""Import stylia without letting it delete matplotlib's font cache.

stylia's ``__init__`` runs ``shutil.rmtree(matplotlib.get_cachedir())`` on every import, so every
new process rebuilds matplotlib's font cache from scratch, ~9 s on macOS. Matplotlib's font manager
is loaded first (reading or writing the real cache), and stylia is then imported while
``get_cachedir`` points at a throwaway folder, which is what stylia deletes instead.
"""

import tempfile

import matplotlib
import matplotlib.font_manager  # noqa: F401  (loads or builds the real font cache first)


def _import_stylia():
  original = matplotlib.get_cachedir
  with tempfile.TemporaryDirectory(prefix="zairachem-stylia-") as decoy:
    matplotlib.get_cachedir = lambda: decoy
    try:
      import stylia
    finally:
      matplotlib.get_cachedir = original
  return stylia


stylia = _import_stylia()
