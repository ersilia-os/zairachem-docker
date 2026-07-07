import json, os, random, warnings
import numpy as np
import pandas as pd
from time import time
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  DATA_FILENAME,
  DATA_SUBFOLDER,
  METADATA_SUBFOLDER,
  PARAMETERS_FILE,
  SESSION_FILE,
)


def params_path(base):
  """Absolute path to a model/run folder's ``parameters.json`` (under ``metadata/``)."""
  return os.path.join(base, METADATA_SUBFOLDER, PARAMETERS_FILE)


warnings.filterwarnings("ignore")


def session_dir(path):
  """The run root for ``path``: the nearest ancestor (inclusive) that holds a ``session.json``.

  A run is fully described by its own folder — there is no global "active session" pointer. The
  session file lives at the run root (written by ``SessionFile.open_session``); a step whose
  ``self.path`` is a subfolder (e.g. ``inputs/``) or a held-out fold dir (``folds/<name>/``) resolves
  up to it. Raises if no session exists at or above ``path``."""
  d = os.path.abspath(path)
  while True:
    if os.path.exists(os.path.join(d, SESSION_FILE)):
      return d
    parent = os.path.dirname(d)
    if parent == d:
      raise FileNotFoundError(f"No {SESSION_FILE} found at or above {path!r}")
    d = parent


def read_session(path):
  """Load the run's ``session.json`` (``mode``, ``model_dir``, resume ``steps``, timing)."""
  with open(os.path.join(session_dir(path), SESSION_FILE), "r") as f:
    return json.load(f)


def write_session(path, data):
  with open(os.path.join(session_dir(path), SESSION_FILE), "w") as f:
    json.dump(data, f, indent=4)


class ZairaBase(object):
  def __init__(self):
    self.logger = logger

  def get_output_dir(self):
    # The run root (folder holding session.json) — resolved from self.path, no global lookup.
    return session_dir(self.path)

  def reset_time(self, path=None):
    path = path or self.path
    session = read_session(path)
    session["time_stamp"] = int(time())
    write_session(path, session)

  def update_elapsed_time(self, path=None):
    path = path or self.path
    session = read_session(path)
    session["elapsed_time"] = session["elapsed_time"] + (int(time()) - session["time_stamp"])
    write_session(path, session)

  def get_trained_dir(self):
    # The trained model dir: equals the run root at fit, differs at predict. Recorded per-run.
    return read_session(self.path)["model_dir"]

  def _load_params(self):
    """Load this run's ``parameters.json`` (under ``self.path/metadata/``)."""
    with open(params_path(self.path), "r") as f:
      return json.load(f)

  def is_predict(self):
    return read_session(self.path)["mode"] == "predict"

  def is_train(self):
    return not self.is_predict()

  def _dummy_indices(self, path):
    df = pd.read_csv(os.path.join(path, DATA_SUBFOLDER, DATA_FILENAME))
    idxs = np.array(list(range(df.shape[0])))
    random.shuffle(idxs)
    return idxs

  def get_train_indices(self, path):
    self.logger.debug("Training set is the full dataset. Interpret with caution!")
    idxs = self._dummy_indices(path)
    return idxs

  def get_validation_indices(self, path):
    self.logger.debug("Validation set is equivalent to the training set. Interpret with caution!")
    idxs = self._dummy_indices(path)
    return idxs
