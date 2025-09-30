import shutil, subprocess, sys
from zairachem.base.utils.logging import logger
from zairachem.base.utils.terminal import run_command


def install_docker_compose(install_file):
  if shutil.which("docker-compose") is None:
    logger.warning("docker‑compose not found; running installer…")
    try:
      run_command(["bash", install_file])
      logger.info("Installation complete.")
    except subprocess.CalledProcessError as e:
      logger.error(f"Installation failed (exit code {e.returncode})", file=sys.stderr)
      sys.exit(e.returncode)
  else:
    logger.info("docker‑compose is already installed.")
