"""The user's shared Docker compose project of Ersilia model servers.

Model servers are shared by the ZairaChem runs of this user. A run stops the servers it started when
it ends, unless another run is still using them; ``state.json`` records, per service, whether a run
started it and which runs use it. A server that was already up (started by hand, or by an older
ZairaChem) is not ours and is left alone, and servers left behind by a run that was killed are
stopped by the next run (:func:`reap_orphans`). ``zairachem stop`` removes all of them. Each model's
service is added the first time a run needs it, on a port that then stays fixed (``ports.json``), so
adding a model never recreates the others.
"""

import atexit
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

from zairachem.base.utils.logging import logger
from zairachem.base.utils.processes import describe_process, identity, is_alive
from zairachem.base.utils.terminal import run_command
from zairachem.base.vars import BASE_DIR, SERVERS_PROJECT_NAME

COMPOSE_FILENAME = "docker-compose.yml"
PORTS_FILENAME = "ports.json"
STATE_FILENAME = "state.json"
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
def locked(name, on_wait=None):
  """Hold an exclusive lock on ``compose_dir()/<name>.lock`` (shared by all runs of this user).

  The holder writes its PID into the file. When the lock is already held, ``on_wait(pid)`` is called
  once with the holder's PID (None if it recorded none) before this waits for it, so a caller can say
  why it is not making progress; a waiting run is otherwise indistinguishable from a working one.
  """
  folder = compose_dir()
  folder.mkdir(parents=True, exist_ok=True)
  with open(folder / f"{name}.lock", "a+") as f:  # not "w": that would erase the holder's PID
    try:
      fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      pid = _holder_pid(f)
      logger.info(f"[compose] {name} is in use by {describe_process(pid)}; waiting for it")
      if on_wait is not None:
        on_wait(pid)
      fcntl.flock(f, fcntl.LOCK_EX)
    try:
      _write_pid(f, os.getpid())
      yield
    finally:
      _write_pid(f, None)
      fcntl.flock(f, fcntl.LOCK_UN)


def _holder_pid(f):
  # The PID the holder wrote, or None (not written yet, or unreadable).
  try:
    f.seek(0)
    return int(f.read().strip())
  except ValueError:
    return None


def _write_pid(f, pid):
  f.seek(0)
  f.truncate()
  if pid is not None:
    f.write(str(pid))
  f.flush()


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

  Records which of them this run started and that it uses them, so it stops them when it ends (see
  :func:`release_servers`). Returns True when compose reports success.
  """
  args = compose_args()
  if args is None:
    logger.warning("[describe] neither `docker compose` nor `docker-compose` is available")
    return False
  # One compose command at a time on the shared project (threads of this run and other runs).
  with _thread_lock, locked("compose"):
    before = running_services()
    ok = run_command([*args, "up", "-d", *(services or [])], quiet=True).returncode == 0
    if ok and services:
      _record_use(services, before)
  return ok


def running_services():
  """Names of the shared project's running services, or None when compose cannot say."""
  args = compose_args()
  if args is None:
    return None
  try:
    out = subprocess.run(
      [*args, "ps", "--services", "--status", "running"],
      capture_output=True,
      text=True,
      timeout=30,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  return set(out.stdout.split()) if out.returncode == 0 else None


def _load_state():
  try:
    return json.loads((compose_dir() / STATE_FILENAME).read_text())
  except (OSError, ValueError):
    return {}


def _save_state(state):
  path = compose_dir() / STATE_FILENAME
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2))
  tmp.replace(path)


_release_registered = False


def _record_use(services, before):
  # Under the compose lock. ``before`` is what was running before this run's ``up`` (None: unknown).
  global _release_registered
  state, me = _load_state(), identity()
  for service in services:
    entry = state.get(service)
    started_now = before is not None and service not in before
    if entry is None or started_now:
      # Ours only when it was not running before this run started it; unknown counts as not ours.
      entry = state[service] = {"started_by_run": started_now, "users": []}
    entry["users"] = [u for u in entry["users"] if is_alive(u) and u["pid"] != me["pid"]] + [me]
  _save_state(state)
  if not _release_registered:
    _release_registered = True
    atexit.register(release_servers)


def _stop_unused(release):
  # Stop the services a run started that no live run uses any more; returns their names.
  # ``release`` also drops this process from the users first (the end of a run).
  if not compose_file().exists():
    return []
  args = compose_args()
  if args is None:
    return []
  with _thread_lock, locked("compose"):
    state, me = _load_state(), identity()["pid"]
    for entry in state.values():
      entry["users"] = [
        u for u in entry["users"] if is_alive(u) and not (release and u["pid"] == me)
      ]
    idle = [s for s, e in state.items() if e.get("started_by_run") and not e["users"]]
    if idle:
      if run_command([*args, "rm", "-sf", *idle], quiet=True).returncode == 0:
        logger.info(f"[compose] stopped {', '.join(idle)}")
      else:
        logger.warning(f"[compose] could not stop {', '.join(idle)}; `zairachem stop` removes them")
        idle = []
    # Forget what was stopped and what nobody uses and we did not start (left as it is).
    _save_state({
      s: e for s, e in state.items() if e["users"] or (e.get("started_by_run") and s not in idle)
    })
  return idle


def release_servers():
  """Stop the servers this run started, unless another live run still uses them; best effort.

  Registered to run when the process exits. A server stays up for the last run that uses it, so
  the run that started it can leave first.
  """
  try:
    _stop_unused(release=True)
  except Exception as e:
    logger.warning(f"[compose] could not release the model servers ({e})")


def reap_orphans():
  """Stop servers left running by a run that was killed, and return their service names.

  A run killed before it could release its servers (SIGKILL, a closed terminal) leaves ones it
  started with no live user. Servers with a live user, or that no run started, are not touched.
  """
  try:
    return _stop_unused(release=False)
  except Exception as e:
    logger.warning(f"[compose] could not look for servers left by a killed run ({e})")
    return []


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
    (compose_dir() / STATE_FILENAME).unlink(missing_ok=True)
    logger.info("[describe] model servers stopped")
  else:
    logger.warning(
      f"[describe] could not stop the model servers; run `docker compose -p "
      f"{project_name()} down -v` to remove them"
    )
