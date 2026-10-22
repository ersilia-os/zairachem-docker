"""The user's shared Docker compose project of Ersilia model servers.

Model servers are shared by every ZairaChem run of this user and stay up between runs (``zairachem
stop`` removes them): starting a container costs seconds per model per run, and a running server
holds little memory while idle. Each model's service is added the first time a run needs it, on a
port that then stays fixed (``ports.json``), so adding a model never recreates the others.
"""

import contextlib
import fcntl
import json
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command
from zairachem.base.vars import BASE_DIR, SERVERS_PROJECT_NAME

COMPOSE_FILENAME = "docker-compose.yml"
PORTS_FILENAME = "ports.json"
_thread_lock = threading.Lock()


def compose_dir():
  """Folder of the shared compose project (``BASE_DIR/servers``)."""
  return Path(BASE_DIR) / "servers"


def compose_file():
  """Path of the shared docker-compose file."""
  return compose_dir() / COMPOSE_FILENAME


def project_name():
  """Compose project name of the shared model servers."""
  return SERVERS_PROJECT_NAME


@contextlib.contextmanager
def locked(name):
  """Hold an exclusive lock on ``compose_dir()/<name>.lock`` (shared by all runs of this user)."""
  folder = compose_dir()
  folder.mkdir(parents=True, exist_ok=True)
  with open(folder / f"{name}.lock", "w") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(f, fcntl.LOCK_UN)


def load_ports():
  """Model id → host port of every model in the shared project."""
  try:
    return json.loads((compose_dir() / PORTS_FILENAME).read_text())
  except (OSError, ValueError):
    return {}


def save_ports(ports):
  """Store the model id → host port map of the shared project."""
  path = compose_dir() / PORTS_FILENAME
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(ports, indent=2, sort_keys=True))
  tmp.replace(path)


@lru_cache(maxsize=1)
def compose_cmd():
  """The compose invocation available here: ``docker compose`` (v2), else ``docker-compose``."""
  try:
    ok = (
      subprocess.run(
        ["docker", "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
      ).returncode
      == 0
    )
  except OSError:
    ok = False
  if ok:
    return ("docker", "compose")
  if shutil.which("docker-compose"):
    return ("docker-compose",)
  return None


def compose_args():
  """Compose command prefix bound to the run's project and file, or None without compose."""
  cmd = compose_cmd()
  if cmd is None:
    return None
  return [*cmd, "-p", project_name(), "-f", str(compose_file())]


def up(services=None):
  """Start the run's model servers (only ``services``, plus what they depend on, when given).

  Returns True when compose reports success.
  """
  args = compose_args()
  if args is None:
    logger.warning("[describe] neither `docker compose` nor `docker-compose` is available")
    return False
  # One compose command at a time on the shared project (threads of this run and other runs).
  with _thread_lock, locked("compose"):
    return run_command([*args, "up", "-d", *(services or [])], quiet=True).returncode == 0


def container_id(service):
  """Id of the running container of ``service``, or None."""
  args = compose_args()
  if args is None:
    return None
  try:
    out = subprocess.run([*args, "ps", "-q", service], capture_output=True, text=True, timeout=30)
  except (OSError, subprocess.SubprocessError):
    return None
  return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None


def service_logs(service, tail=30):
  """Last ``tail`` lines of a service's container log, or '' when unavailable."""
  args = compose_args()
  if args is None:
    return ""
  try:
    out = subprocess.run(
      [*args, "logs", "--no-color", "--tail", str(tail), service],
      capture_output=True,
      text=True,
      timeout=30,
    )
  except (OSError, subprocess.SubprocessError):
    return ""
  return out.stdout + out.stderr


def stop_model_servers():
  """Remove every shared model server and the redis cache (``zairachem stop``). Best effort.

  Their ports are forgotten, so the next run picks free ones. A no-op when no compose file was
  ever written.
  """
  if not compose_file().exists():
    return
  args = compose_args()
  if args is None:
    return
  with _thread_lock, locked("compose"):  # not in the middle of a run's own compose command
    result = run_command([*args, "down", "-v", "--remove-orphans"], quiet=True)
  if result.returncode == 0:
    # Ports are chosen again next time: another program may take a recorded one meanwhile.
    with locked("config"):
      (compose_dir() / PORTS_FILENAME).unlink(missing_ok=True)
    logger.info("[describe] model servers stopped")
  else:
    logger.warning(
      f"[describe] could not stop the model servers; run `docker compose -p "
      f"{project_name()} down -v` to remove them"
    )
