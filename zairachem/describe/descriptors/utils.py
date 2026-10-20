import contextlib
import os
import re
import subprocess
from typing import Optional

import h5py
import numpy as np

from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command
from zairachem.base.vars import BASE_DIR
from zairachem.describe.descriptors import compose

try:
  import yaml
except Exception:
  yaml = None


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


def load_yml(file):
  """Return the parsed YAML in ``file``, or None when the file does not exist."""
  if not os.path.exists(file):
    return None

  with open(file, "r") as f:
    data = yaml.safe_load(f)
  return data


def _service_name(model_id: str) -> str:
  s = re.sub(r"[^a-zA-Z0-9]+", "_", model_id.lower()).strip("_") or "svc"
  return f"{s}_api"


def write_service_file(model_ids):
  """Write ``model_ids``, one per line, to ``<BASE_DIR>/service.txt``."""
  with open(f"{BASE_DIR}/service.txt", "w") as f:
    for m in model_ids:
      f.write(f"{m}\n")


def get_services(compose_file):
  """Return the ``services`` mapping of a docker compose file, or None if missing or empty."""
  data = load_yml(compose_file)
  if not data:
    return None
  return data.get("services", {})


def service_exists(compose_file, model_ids):
  """Return whether every model has a service in ``compose_file`` (None if it has none)."""
  service_names = [_service_name(model_id) for model_id in model_ids]
  services = get_services(compose_file)
  if not services:
    return None
  return all(s in services for s in service_names)


def _parse_cli_port(out: str) -> Optional[int]:
  for line in out.splitlines():
    m = re.search(r":(\d+)\s*$", line.strip())
    if m:
      return int(m.group(1))
  return None


def _via_cli(service: str, path) -> Optional[int]:
  args = compose.compose_args(path)
  if args is None:
    return None
  cmd = [*args, "port", service, "80"]

  logger.debug(f"Resolving port for service '{service}' with: {' '.join(cmd)}")

  try:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    logger.debug(f"Command output: '{out}'")

    port = _parse_cli_port(out)
    if port is not None:
      logger.info(f"Resolved '{service}' → port {port}")
      return port

    logger.warning(f"No usable port found in output for '{service}': '{out}'")

  except subprocess.CalledProcessError as e:
    logger.error(
      f"Command failed: {' '.join(cmd)} | Return code: {e.returncode} | Output: {e.output.strip()}"
    )

  except Exception as e:
    logger.exception(f"Unexpected error running command: {' '.join(cmd)} | Error: {e}")

  logger.error(f"Failed to resolve port for service '{service}'.")
  return None


def _published_port_80(p) -> Optional[int]:
  """Host port that a compose ``ports`` entry maps to container port 80.

  Parameters
  ----------
  p : dict or str
      One entry of a service's ``ports`` list, in long (dict) or short (string) syntax.

  Returns
  -------
  int or None
      The published host port, or None when the entry does not map port 80.
  """
  if isinstance(p, dict):
    tgt = str(p.get("target", "")).split("/")[0]
    pub = str(p.get("published", "")).split("/")[0]
    if tgt.isdigit() and int(tgt) == 80 and pub.isdigit():
      return int(pub)
  elif isinstance(p, str):
    s = re.sub(r"\s+#.*$", "", p.strip())
    m = re.match(r"^(?:\[[^\]]+\]:)?(\d+):(\d+)(?:/(?:tcp|udp))?$", s)
    if m and int(m.group(2)) == 80:
      return int(m.group(1))
    parts = s.split(":")
    try:
      tgt = int(parts[-1].split("/")[0])
      if tgt == 80 and len(parts) >= 2:
        return int(parts[-2])
    except Exception:
      return None
  return None


def _via_yaml(service: str, path) -> Optional[int]:
  compose_file = compose.compose_file(path)
  if yaml is None or not compose_file.exists():
    return None
  with open(compose_file, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  svc = (data.get("services") or {}).get(service)
  if not svc:
    return None
  for p in svc.get("ports") or []:
    port = _published_port_80(p)
    if port is not None:
      return port
  return None


def get_model_host_port(model_id: str, path) -> Optional[int]:
  """Return the host port mapped to port 80 of the model's compose service, or None.

  Asks ``docker compose port`` first and falls back to parsing the compose YAML.
  """
  service = _service_name(model_id)
  return _via_cli(service, path) or _via_yaml(service, path)


def get_model_url(model_id: str, path, host: str = "localhost") -> Optional[str]:
  """URL of ``model_id``'s server in the run folder ``path``, or None when it is not up."""
  port = get_model_host_port(model_id, path)
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
