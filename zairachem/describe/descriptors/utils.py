import contextlib
import re
import subprocess
from typing import Optional

import h5py
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command
from zairachem.describe.descriptors import compose


class Hdf5Data:
  """Descriptor result of ``BinaryStreamClient.run``, held in memory to be saved as an H5 file."""

  def __init__(self, results):
    self.values = results["data"]
    self.dim = results["shape"][-1]
    self.dtype = results["dtype"]
    self._np_dtype = np.dtype(self.dtype) if self.dtype else None
    str_dt = h5py.string_dtype(encoding="utf-8")
    self.inputs = np.array(results["inputs"], dtype=str_dt)
    self.features = np.array(results["dims"], dtype=str_dt)

  def save(self, filename):
    """Write the matrix to ``filename`` as ``Values``, ``Inputs`` and ``Features`` datasets."""
    logger.info(f"[h5] saving {filename} shape={np.shape(self.values)}")
    with h5py.File(filename, "w") as f:
      f.create_dataset("Values", data=self.values)
      f.create_dataset("Inputs", data=self.inputs)
      f.create_dataset("Features", data=self.features)


def _service_name(model_id: str) -> str:
  from zairachem.base.generate_config import service_name

  return service_name(model_id)


def get_model_host_port(model_id: str) -> Optional[int]:
  """Host port of ``model_id``'s server in the shared project (``ports.json``), or None."""
  port = compose.load_ports().get(model_id)
  return int(port) if port else None


def get_model_url(model_id: str, host: str = "localhost") -> Optional[str]:
  """URL of ``model_id``'s server in the shared project, or None when it has no port."""
  port = get_model_host_port(model_id)
  return f"http://{host}:{port}/run" if port else None


def _ensure_network(name):
  try:
    out = subprocess.check_output(["docker", "network", "ls", "--format", "{{.Name}}"], text=True)
  except Exception:
    out = ""
  if not re.search(rf"(?m)^{re.escape(name)}$", out):
    with contextlib.suppress(Exception):
      run_command(["docker", "network", "create", name], quiet=True)
  out = subprocess.check_output(["docker", "network", "ls", "--format", "{{.Name}}"], text=True)
  if not re.search(rf"(?m)^{re.escape(name)}$", out):
    raise RuntimeError(f"docker network '{name}' not found")
