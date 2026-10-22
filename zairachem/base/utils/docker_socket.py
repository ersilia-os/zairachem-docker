"""Quick questions to the Docker daemon over its local socket, without starting the Docker CLI.

``docker version`` / ``docker image inspect`` cost 50–250 ms each (the CLI process starts every
time); the same answers over the daemon's Unix socket take a few milliseconds. Each function returns
None when the socket cannot be used (no socket, a remote ``DOCKER_HOST``, Windows), so callers fall
back to the CLI.
"""

import http.client
import json
import os
import socket
from functools import lru_cache
from urllib.parse import quote

_TIMEOUT = 2.0


@lru_cache(maxsize=1)
def _socket_path():
  host = os.environ.get("DOCKER_HOST", "")
  if host:
    return host[len("unix://") :] if host.startswith("unix://") else None
  for path in (
    os.path.expanduser("~/.docker/run/docker.sock"),  # Docker Desktop (macOS)
    "/var/run/docker.sock",
  ):
    if os.path.exists(path):
      return path
  return None


class _UnixConnection(http.client.HTTPConnection):
  def __init__(self, path):
    super().__init__("localhost", timeout=_TIMEOUT)
    self._path = path

  def connect(self):
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.sock.settimeout(_TIMEOUT)
    self.sock.connect(self._path)


def _status(url):
  """HTTP status of a GET on the daemon's API; 0 when it refuses the connection (not running),
  None when the socket is unusable or too slow to say."""
  path = _socket_path()
  if path is None:
    return None
  conn = _UnixConnection(path)
  try:
    conn.request("GET", url)
    return conn.getresponse().status
  except (ConnectionRefusedError, FileNotFoundError):
    return 0
  except OSError:  # includes timeouts: a busy daemon is not a stopped one
    return None
  finally:
    conn.close()


def daemon_running():
  """True/False when the socket answers (or refuses), None when it cannot tell."""
  status = _status("/_ping")
  return None if status is None else status == 200


def image_present(image):
  """Whether ``image`` exists locally, or None when the socket cannot be used."""
  status = _status(f"/images/{quote(image, safe='')}/json")
  if not status:
    return None
  return status == 200


def running_containers(project):
  """Number of running containers of compose ``project``, or None when the socket is unusable."""
  path = _socket_path()
  if path is None:
    return None
  filters = quote(json.dumps({"label": [f"com.docker.compose.project={project}"]}))
  conn = _UnixConnection(path)
  try:
    conn.request("GET", f"/containers/json?filters={filters}")
    resp = conn.getresponse()
    if resp.status != 200:
      return None
    return len(json.loads(resp.read()))
  except (OSError, ValueError):
    return None
  finally:
    conn.close()


def _get_json(url):
  path = _socket_path()
  if path is None:
    return None
  conn = _UnixConnection(path)
  try:
    conn.request("GET", url)
    resp = conn.getresponse()
    return json.loads(resp.read()) if resp.status == 200 else None
  except (OSError, ValueError):
    return None
  finally:
    conn.close()


def containers_memory_gb():
  """Memory (GB) of each running container, ``{name: gb}``, or None when the socket is unusable.

  As ``docker stats`` reports it (usage minus inactive page cache), from one-shot readings taken in
  parallel. A container whose reading fails is left out.
  """
  from concurrent.futures import ThreadPoolExecutor

  listing = _get_json("/containers/json")
  if listing is None:
    return None
  names = {c["Id"]: (c.get("Names") or ["?"])[0].lstrip("/") for c in listing}

  def used(cid):
    stats = _get_json(f"/containers/{cid}/stats?stream=false&one-shot=true")
    mem = (stats or {}).get("memory_stats") or {}
    if "usage" not in mem:
      return None
    extra = mem.get("stats") or {}
    cache = extra.get("inactive_file", extra.get("total_inactive_file", 0))
    return max(0, mem["usage"] - cache) / 1024**3

  if not names:
    return {}
  with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
    readings = dict(zip(names, ex.map(used, names)))
  return {names[cid]: gb for cid, gb in readings.items() if gb is not None}
