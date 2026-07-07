import json, os
from time import time

from zairachem.base import ZairaBase
from zairachem.base.vars import SESSION_FILE


class SessionFile(ZairaBase):
  def __init__(self, output_dir):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
    self.session_file = os.path.join(os.path.abspath(output_dir), SESSION_FILE)

  def _host_info(self):
    """Characteristics of the machine the run executes on (psutil + platform, fully guarded)."""
    info = {}
    try:
      import platform

      import psutil

      info["cpu_logical"] = psutil.cpu_count()
      info["cpu_physical"] = psutil.cpu_count(logical=False)
      info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
      info["arch"] = platform.machine()
      info["system"] = f"{platform.system()} {platform.release()}".strip()
      info["python"] = platform.python_version()
    except Exception:
      pass
    return info

  def open_session(self, mode, output_dir, model_dir=None):
    self.mode = mode
    self.output_dir = os.path.abspath(output_dir)
    if model_dir is None:
      self.model_dir = self.output_dir
    else:
      self.model_dir = os.path.abspath(model_dir)
    data = {
      "output_dir": self.output_dir,
      "model_dir": self.model_dir,
      "time_stamp": int(time()),
      "elapsed_time": 0,
      "mode": self.mode,
      "host": self._host_info(),
    }
    with open(self.session_file, "w") as f:
      json.dump(data, f, indent=4)


class PipelineStep(ZairaBase):
  def __init__(self, name, output_dir):
    ZairaBase.__init__(self)
    self.name = name
    sf = SessionFile(output_dir)
    self.session_file = sf.session_file

  def _read_session(self):
    if not os.path.exists(self.session_file):
      return None
    with open(self.session_file, "r") as f:
      data = json.load(f)
    if "steps" not in data:
      data["steps"] = []
    return data

  def _write_session(self, data):
    with open(self.session_file, "w") as f:
      json.dump(data, f, indent=4)

  def _telemetry(self):
    """A per-step telemetry snapshot: completion time plus host CPU/RAM (psutil-guarded)."""
    rec = {"name": self.name, "t": time()}
    try:
      import psutil

      rec["cpu"] = psutil.cpu_percent(interval=None)
      vm = psutil.virtual_memory()
      rec["ram_used_gb"] = round(vm.used / 1e9, 2)
      rec["ram_total_gb"] = round(vm.total / 1e9, 2)
      rec["ram_pct"] = vm.percent
    except Exception:
      pass
    return rec

  def update(self):
    data = self._read_session()
    data["steps"] += [self.name]
    data.setdefault("step_log", []).append(self._telemetry())
    self._write_session(data)

  def is_done(self):
    data = self._read_session()
    if data is None:
      return False
    if self.name in data["steps"]:
      return True
    else:
      return False

  def unmark(self):
    """Drop only this step's done-marker, leaving later steps intact.

    Used to force a single step to re-run (e.g. re-rendering the report) without disturbing the rest
    of the pipeline's resume state. A no-op if the step isn't marked done or the session file is
    absent.
    """
    data = self._read_session()
    if data is None:
      return
    data["steps"] = [s for s in data["steps"] if s != self.name]
    self._write_session(data)
