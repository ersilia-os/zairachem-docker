"""Resolve which descriptor models a run should actually train / pool / report on.

``done_eos.json`` (under ``descriptors/``) records every featurizer that was successfully computed.
When ``--max-descriptors`` pre-screening runs, it writes a chosen subset to
``metadata/selected_eos.json``; that subset then overrides ``done_eos.json`` everywhere downstream.
Keeping the two separate means ``done_eos.json`` stays an honest record of what was featurized, and a
held-out fold can hold its OWN ``selected_eos.json`` (its ``metadata/`` is a copy, not a symlink of the
parent's) without disturbing the shared, symlinked ``descriptors/``.
"""

import json
import os

from zairachem.base.vars import DESCRIPTORS_SUBFOLDER, METADATA_SUBFOLDER, SELECTED_EOS_FILENAME


def effective_descriptors(path):
  """The descriptor ids a run should use: ``metadata/selected_eos.json`` if present, else ``done_eos``.

  Parameters
  ----------
  path : str
    A run directory (the production model dir, a predict dir, or a fold workspace).

  Returns
  -------
  list of str
    Ordered descriptor (featurizer) ids.
  """
  selected = os.path.join(path, METADATA_SUBFOLDER, SELECTED_EOS_FILENAME)
  if os.path.exists(selected):
    with open(selected) as f:
      return list(json.load(f))
  with open(os.path.join(path, DESCRIPTORS_SUBFOLDER, "done_eos.json")) as f:
    return list(json.load(f))
