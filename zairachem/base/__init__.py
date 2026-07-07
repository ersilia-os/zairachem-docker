import json, os, random, warnings
import numpy as np
import pandas as pd
from time import time
from zairachem.base.utils.logging import logger
from zairachem.base.vars import (
  BASE_DIR,
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


def resolve_output_dir(output_dir):
  if output_dir is None:
    system_session = os.path.join(BASE_DIR, SESSION_FILE)
    with open(system_session, "r") as f:
      session = json.load(f)
    return session["output_dir"]
  else:
    return os.path.abspath(output_dir)


def create_session_symlink(output_dir):
  if output_dir is None:
    output_dir = resolve_output_dir(output_dir)
  output_session = os.path.join(os.path.abspath(output_dir), SESSION_FILE)
  system_session = os.path.join(BASE_DIR, SESSION_FILE)
  if os.path.islink(system_session):
    os.unlink(system_session)
  os.symlink(output_session, system_session)


class ZairaBase(object):
  def __init__(self):
    self.logger = logger

  def get_output_dir(self):
    with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
      session = json.load(f)
    return session["output_dir"]

  def reset_time(self):
    with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
      session = json.load(f)
    session["time_stamp"] = int(time())
    output_dir = session["output_dir"]
    with open(os.path.join(output_dir, SESSION_FILE), "w") as f:
      json.dump(session, f, indent=4)

  def update_elapsed_time(self):
    with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
      session = json.load(f)
    delta_time = int(time()) - session["time_stamp"]
    session["elapsed_time"] = session["elapsed_time"] + delta_time
    output_dir = session["output_dir"]
    with open(os.path.join(output_dir, SESSION_FILE), "w") as f:
      json.dump(session, f, indent=4)

  def get_trained_dir(self):
    with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
      session = json.load(f)
    return session["model_dir"]

  def _load_params(self):
    """Load this run's ``parameters.json`` (from ``self.path``, else the active session dir)."""
    base = getattr(self, "path", None) or self.get_output_dir()
    with open(params_path(base), "r") as f:
      return json.load(f)

  def is_predict(self):
    with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
      session = json.load(f)
    if session["mode"] == "predict":
      return True
    else:
      return False

  def is_train(self):
    if self.is_predict():
      return False
    else:
      return True

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
