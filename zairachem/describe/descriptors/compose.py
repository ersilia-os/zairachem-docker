import hashlib, re, shutil, subprocess
from functools import lru_cache
from pathlib import Path
from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command
from zairachem.base.vars import METADATA_SUBFOLDER

COMPOSE_FILENAME = "docker-compose.yml"


def compose_dir(path):
  """Folder holding the run's compose file, inside the run's own ``metadata/``."""
  return Path(path).resolve() / METADATA_SUBFOLDER / "compose"


def compose_file(path):
  """Path of the run's docker-compose file."""
  return compose_dir(path) / COMPOSE_FILENAME


def project_name(path):
  """Compose project name, stable for a given run folder and distinct between run folders."""
  root = Path(path).resolve()
  slug = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-") or "run"
  digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]
  return f"zairachem-{slug}-{digest}"


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


def compose_args(path):
  """Compose command prefix bound to the run's project and file, or None without compose."""
  cmd = compose_cmd()
  if cmd is None:
    return None
  return [*cmd, "-p", project_name(path), "-f", str(compose_file(path))]


def up(path):
  """Start the run's model servers; returns True when compose reports success."""
  args = compose_args(path)
  if args is None:
    logger.warning("[describe] neither `docker compose` nor `docker-compose` is available")
    return False
  return run_command([*args, "up", "-d"], quiet=True).returncode == 0


def stop_model_servers(path):
  """Remove the run's containers and redis cache. Idempotent and best effort.

  A no-op when the run never wrote a compose file, so it is safe to call from every exit path.
  """
  if not compose_file(path).exists():
    return
  args = compose_args(path)
  if args is None:
    return
  result = run_command([*args, "down", "-v", "--remove-orphans"], quiet=True)
  if result.returncode == 0:
    logger.info("[describe] model servers stopped")
  else:
    logger.warning(
      f"[describe] could not stop the model servers; run `docker compose -p "
      f"{project_name(path)} down -v` to remove them"
    )
