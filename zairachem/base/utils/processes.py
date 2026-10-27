"""Other ZairaChem runs of this user, and what a process is doing.

Runs share the model servers and wait for each other's locks, so a run that is long, or was
suspended with Ctrl-Z, holds up the next one. These helpers name the holder so the wait can be
explained instead of looking like work.
"""

import os
import time

import psutil

_RUN_COMMANDS = ("fit", "predict")
_MAX_COMMAND = 60


def _run_command(cmdline):
  """The ``zairachem`` arguments after ``fit`` or ``predict`` joined, or None if not a run command."""
  for i, arg in enumerate(cmdline[:-1]):
    if os.path.basename(arg) == "zairachem":  # `zairachem ...` and `python -m zairachem ...`
      if cmdline[i + 1] not in _RUN_COMMANDS:
        return None
      command = " ".join(cmdline[i + 1 :])
      return command if len(command) <= _MAX_COMMAND else command[: _MAX_COMMAND - 1] + "…"
  return None


def _age(seconds):
  if seconds < 90:
    return f"{seconds:.0f} s"
  if seconds < 5400:
    return f"{seconds / 60:.0f} min"
  return f"{seconds / 3600:.1f} h"


def _describe(pid, command, created, status, brief):
  state = "suspended" if status == psutil.STATUS_STOPPED else "running"
  if brief:
    return f"pid {pid} · {state}"
  parts = [f"pid {pid}"]
  if command:
    parts.append(command)
  if created is not None:
    parts.append(_age(time.time() - created))
  parts.append(state)
  return " · ".join(parts)


def describe_process(pid, brief=False):
  """One line naming a process: ``pid 37500 · fit -i x.csv -m model · 11 min · suspended``.

  Parameters
  ----------
  pid : int or None
      Process ID; None (a holder that did not record one) reads as another run.
  brief : bool, optional
      Only the PID and whether it is running or suspended, short enough for a table cell.

  Returns
  -------
  str
      The description; "another run" or "pid N (gone)" when the process cannot be read.
  """
  if pid is None:
    return "another run"
  try:
    proc = psutil.Process(pid)
    return _describe(
      pid,
      _run_command(proc.cmdline()),
      proc.create_time(),
      proc.status(),
      brief,
    )
  except psutil.Error:
    return f"pid {pid} (gone)"


def other_runs():
  """The user's other ``zairachem fit`` and ``zairachem predict`` processes.

  This process, its parents and its children are left out: worker processes share the command
  line of the run that started them. Zombies are left out too.

  Returns
  -------
  list of dict
      One per run: ``pid``, ``text`` (see :func:`describe_process`) and ``suspended``.
  """
  me = psutil.Process()
  mine = {me.pid}
  mine.update(p.pid for p in me.parents())
  mine.update(p.pid for p in me.children(recursive=True))
  runs = []
  for proc in psutil.process_iter(["pid", "cmdline", "status", "create_time"]):
    info = proc.info
    status = info.get("status")
    if info["pid"] in mine or status == psutil.STATUS_ZOMBIE:
      continue
    command = _run_command(info.get("cmdline") or [])
    if command is None:
      continue
    text = _describe(info["pid"], command, info.get("create_time"), status, brief=False)
    runs.append({"pid": info["pid"], "text": text, "suspended": status == psutil.STATUS_STOPPED})
  return runs
