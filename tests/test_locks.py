"""A run waiting on another run says so, and names it."""

import os
import signal
import subprocess
import threading
import time
import types

import psutil

from zairachem.base.utils import processes
from zairachem.describe.descriptors import compose


def test_a_run_waiting_for_a_held_lock_is_told_who_holds_it(tmp_path, monkeypatch):
  monkeypatch.setattr(compose, "compose_dir", lambda: tmp_path)
  held, release = threading.Event(), threading.Event()

  def holder():
    with compose.locked("eosxxxx"):
      held.set()
      release.wait(10)

  thread = threading.Thread(target=holder)
  thread.start()
  assert held.wait(10)
  told = []

  def on_wait(pid):
    told.append(pid)
    release.set()  # the holder lets go once the waiter has been told

  with compose.locked("eosxxxx", on_wait=on_wait):
    pass
  thread.join(10)
  assert told == [os.getpid()]


def test_a_free_lock_does_not_tell_anyone_to_wait(tmp_path, monkeypatch):
  monkeypatch.setattr(compose, "compose_dir", lambda: tmp_path)
  told = []
  with compose.locked("eosxxxx", on_wait=told.append):
    pass
  with compose.locked("eosxxxx", on_wait=told.append):  # and it is free again after a release
    pass
  assert told == []


def _process(pid, cmdline, status=psutil.STATUS_SLEEPING):
  info = {"pid": pid, "cmdline": cmdline, "status": status, "create_time": 0.0}
  return types.SimpleNamespace(info=info)


def test_other_runs_finds_a_suspended_fit_and_nothing_else(monkeypatch):
  me = os.getpid()
  fake = [
    _process(me, ["python", "zairachem", "fit", "-i", "x.csv"]),
    _process(
      901,
      ["/env/bin/python3.11", "/env/bin/zairachem", "fit", "-i", "x.csv", "-m", "model"],
      psutil.STATUS_STOPPED,
    ),
    _process(902, ["/env/bin/zairachem", "predict", "-i", "y.csv"]),
    _process(903, ["/env/bin/zairachem", "report", "-m", "model"]),
    _process(904, ["python", "train.py", "fit"]),
    _process(905, ["/env/bin/zairachem", "fit"], psutil.STATUS_ZOMBIE),
  ]
  monkeypatch.setattr(processes.psutil, "process_iter", lambda attrs: fake)
  runs = {r["pid"]: r for r in processes.other_runs()}
  assert sorted(runs) == [901, 902]
  assert runs[901]["suspended"] and not runs[902]["suspended"]
  assert "suspended" in runs[901]["text"] and "fit -i x.csv -m model" in runs[901]["text"]


def test_a_stopped_process_is_described_as_suspended():
  assert processes.describe_process(None) == "another run"
  child = subprocess.Popen(["sleep", "30"])
  try:
    assert processes.describe_process(child.pid, brief=True) == f"pid {child.pid} · running"
    os.kill(child.pid, signal.SIGSTOP)  # what Ctrl-Z does
    for _ in range(100):
      if psutil.Process(child.pid).status() == psutil.STATUS_STOPPED:
        break
      time.sleep(0.05)
    assert processes.describe_process(child.pid, brief=True) == f"pid {child.pid} · suspended"
  finally:
    child.kill()
    child.wait()
