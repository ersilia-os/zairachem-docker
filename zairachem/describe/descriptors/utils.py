import h5py, os, re, subprocess
import numpy as np
from pathlib import Path
from typing import Optional
from zairachem.base.vars import ORG, BASE_DIR
try:
  import yaml
except Exception:
  yaml = None

compose_file = Path(__file__).parent.parent / "files" / "configs" / "docker-compose.yml"


class Hdf5Data:
  def __init__(self, results):
    try:
      self.values = results["data"]
      self.dim = results["shape"][-1]
      self.dtype = results["dtype"]
      if self.dtype is None:
        self._force_dtype = False
        self._np_dtype = None
      else:
        self._force_dtype = True
        self._np_dtype = np.dtype(self.dtype)

      str_dt = h5py.string_dtype(encoding="utf-8")
      self.inputs = np.array(results["inputs"], dtype=str_dt)
      self.features = np.array(results["dims"], dtype=str_dt)

    except Exception as e:
      raise RuntimeError(f"Failed to initialize Hdf5Data: {e}")

  def save(self, filename):
    try:
      with h5py.File(filename, "w") as f:
        f.create_dataset("Values", data=self.values)
        f.create_dataset("Inputs", data=self.inputs)
        f.create_dataset("Features", data=self.features)
    except OSError as e:
      raise IOError(f"Could not write to file '{filename}': {e}")


class Hdf5DataLoader(object):
  def __init__(self):
    self.values = None
    self.inputs = None
    self.features = None

  def load(self, h5_file):
    with h5py.File(h5_file, "r") as f:
      self.values = f["Values"][:]
      self.inputs = [x.decode("utf-8") for x in f["Inputs"][:]]
      self.features = [x.decode("utf-8") for x in f["Features"][:]]

def load_yml(file):
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    return data

def _service_name(model_id: str) -> str:
  s = re.sub(r"[^a-zA-Z0-9]+", "_", model_id.lower()).strip("_") or "svc"
  return f"{s}_api"

def get_model_docker_repo(model_ids):
  return [f"{ORG}/{model}" for model in model_ids]

def write_service_file(model_ids):
  with open(f"{BASE_DIR}/service.txt", "w") as f:
    for m in model_ids:
      f.write(f"{m}\n")

def get_services(compose_file):
    data = load_yml(compose_file)
    return data.get("services", {})

def service_exists(compose_file, model_ids):
    service_names = [_service_name(model_id) for model_id in model_ids]
    services = get_services(compose_file)
    return all(s in services for s in service_names)

def _parse_cli_port(out: str) -> Optional[int]:
  for line in out.splitlines():
    m = re.search(r":(\d+)\s*$", line.strip())
    if m:
      return int(m.group(1))
  return None


def _via_cli(service: str) -> Optional[int]:
  cmds = [
    ["docker", "compose", "-f", os.fspath(compose_file), "port", service, "80"],
    ["docker-compose", "-f", os.fspath(compose_file), "port", service, "80"],
  ]
  for cmd in cmds:
    try:
      out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
      port = _parse_cli_port(out)
      if port:
        return port
    except Exception:
      pass
  return None


def _via_yaml(service: str) -> Optional[int]:
  if yaml is None:
    return None
  with open(compose_file, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  svc = (data.get("services") or {}).get(service)
  if not svc:
    return None
  for p in svc.get("ports") or []:
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
        continue
  return None


def get_model_host_port(model_id: str) -> Optional[int]:
  service = _service_name(model_id)
  return _via_cli(service) or _via_yaml(service)


def get_model_url(model_id: str, host: str = "localhost") -> Optional[str]:
  port = get_model_host_port(model_id)
  return f"http://{host}:{port}/run" if port else None
