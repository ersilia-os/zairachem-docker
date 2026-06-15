import datetime, os, re, shlex, subprocess
from collections import namedtuple
from zairachem.base.vars import BASE_DIR

from zairachem.base.utils.logging import logger

# Full subprocess output (e.g. multi-GB docker pull progress) goes to its own file so it
# does not drown the application log. The main logger only gets a concise summary.
COMMANDS_LOG = "commands.log"


def _append_to_commands_log(text: str):
  os.makedirs(BASE_DIR, exist_ok=True)
  path = os.path.join(BASE_DIR, COMMANDS_LOG)
  with open(path, "a", encoding="utf-8") as f:
    f.write(text if text.endswith("\n") else text + "\n")


def run_command(cmd, quiet=None):
  shell = isinstance(cmd, str)
  if shell:
    run_cmd = cmd
    display_cmd = cmd
  else:
    run_cmd = [os.fspath(c) for c in cmd]  # <- coerce Path/PathLike to str
    display_cmd = " ".join(shlex.quote(x) for x in run_cmd)

  start = datetime.datetime.now()
  result = subprocess.run(
    run_cmd,
    shell=shell,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=os.environ,
  )
  end = datetime.datetime.now()

  CommandResult = namedtuple("CommandResult", ["returncode", "stdout", "stderr"])
  stdout_str = result.stdout.strip()
  stderr_str = result.stderr.strip()
  output = CommandResult(returncode=result.returncode, stdout=stdout_str, stderr=stderr_str)

  log_lines = [
    f"[{start.strftime('%Y-%m-%d %H:%M:%S')}] $ {display_cmd}",
  ]
  if stdout_str:
    log_lines += ["stdout:", stdout_str]
  if stderr_str:
    log_lines += ["stderr:", stderr_str]
  log_lines += [
    f"returncode: {result.returncode}",
    f"duration: {(end - start).total_seconds():.3f}s",
    "-" * 40,
  ]
  _append_to_commands_log("\n".join(log_lines))

  # Concise summary to the application log; full output lives in commands.log.
  duration = (end - start).total_seconds()
  logger.debug(f"$ {display_cmd} (rc={result.returncode}, {duration:.2f}s)")
  if result.returncode != 0:
    # A failing command is surfaced clearly (with a bounded stderr tail), regardless of quiet.
    tail = (stderr_str or stdout_str)[-1000:]
    logger.error(f"Command failed (rc={result.returncode}): {display_cmd}\n{tail}")

  return output


def is_quoted_list(s: str) -> bool:
  pattern = r"^(['\"])\[.*\]\1$"
  return bool(re.match(pattern, s))
