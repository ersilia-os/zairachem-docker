import datetime, os, re, shlex, subprocess, sys
from collections import namedtuple
from zairachem.base.vars import BASE_DIR, LOGGING_FILE

from zairachem.base.utils.logging import logger


def _append_to_log(text: str):
  os.makedirs(BASE_DIR, exist_ok=True)
  path = os.path.join(BASE_DIR, LOGGING_FILE)
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
  _append_to_log("\n".join(log_lines))

  if not quiet:
    if stdout_str:
      print(stdout_str)
    if stderr_str:
      print(stderr_str, file=sys.stderr)

  return output


def is_quoted_list(s: str) -> bool:
  pattern = r"^(['\"])\[.*\]\1$"
  return bool(re.match(pattern, s))
