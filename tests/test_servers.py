"""A run stops the model servers it started, and only those."""

import types

import pytest

from zairachem.base.utils import processes
from zairachem.describe.descriptors import compose

PREFIX = ["docker", "compose", "-p", "zairachem-servers", "-f", "compose.yml"]


class Docker:
  """A stand-in for compose that records what it is told and keeps the set of running services."""

  def __init__(self):
    self.running = set()
    self.commands = []
    self.pid = 100  # the run that is acting
    self.alive = {100, 200}  # live runs; a PID outside it is a killed run

  def run(self, cmd, quiet=True):
    """Record a compose command and apply its effect on the running set."""
    self.commands.append(cmd)
    verb, *rest = cmd[len(PREFIX) :]
    if verb == "up":
      self.running |= set(rest[1:])
    elif verb == "rm":
      self.running -= set(rest[1:])
    return types.SimpleNamespace(returncode=0)

  def stopped(self):
    """The service sets passed to each ``rm`` so far."""
    return [set(c[len(PREFIX) + 2 :]) for c in self.commands if c[len(PREFIX)] == "rm"]


@pytest.fixture
def docker(tmp_path, monkeypatch):
  d = Docker()
  (tmp_path / "compose.yml").write_text("")
  monkeypatch.setattr(compose, "compose_dir", lambda: tmp_path)
  monkeypatch.setattr(compose, "compose_file", lambda: tmp_path / "compose.yml")
  monkeypatch.setattr(compose, "compose_args", lambda: list(PREFIX))
  monkeypatch.setattr(compose, "run_command", d.run)
  monkeypatch.setattr(compose, "running_services", lambda: set(d.running))
  monkeypatch.setattr(compose, "identity", lambda: {"pid": d.pid, "started": 1.0})
  monkeypatch.setattr(compose, "is_alive", lambda who: who["pid"] in d.alive)
  monkeypatch.setattr(compose, "_release_registered", True)  # no atexit handler in tests
  return d


def test_a_run_stops_the_servers_it_started(docker):
  assert compose.up(["redis", "eosa_api"])
  compose.release_servers()
  assert docker.stopped() == [{"redis", "eosa_api"}]
  assert docker.running == set()


def test_a_server_that_was_already_up_is_left_running(docker):
  docker.running = {"eosa_api"}  # started by hand, `ersilia serve`, or an older ZairaChem
  compose.up(["redis", "eosa_api"])
  compose.release_servers()
  assert docker.stopped() == [{"redis"}]
  assert docker.running == {"eosa_api"}


def test_only_the_last_run_using_a_server_stops_it(docker):
  compose.up(["eosa_api"])  # run 100 starts it
  docker.pid = 200
  compose.up(["eosa_api"])  # run 200 uses it while it is up
  docker.pid = 100
  compose.release_servers()  # the starter leaves first
  assert docker.stopped() == [] and docker.running == {"eosa_api"}
  docker.pid = 200
  compose.release_servers()
  assert docker.stopped() == [{"eosa_api"}]


def test_servers_left_by_a_killed_run_are_stopped_by_the_next_run(docker):
  compose.up(["redis", "eosa_api"])
  docker.alive.discard(100)  # the run was killed: it never released them
  docker.pid = 200
  assert sorted(compose.reap_orphans()) == ["eosa_api", "redis"]
  assert docker.running == set()


def test_reaping_leaves_servers_a_live_run_uses_or_no_run_started(docker):
  docker.running = {"eosb_api"}
  compose.up(["eosa_api", "eosb_api"])  # eosb_api was already up
  docker.pid = 200
  assert compose.reap_orphans() == []  # run 100 is still alive
  docker.alive.discard(100)
  assert compose.reap_orphans() == ["eosa_api"]
  assert docker.running == {"eosb_api"}


def test_a_reused_pid_does_not_keep_a_server_up():
  me = processes.identity()
  assert processes.is_alive(me)
  assert not processes.is_alive({"pid": me["pid"], "started": me["started"] - 3600})


def test_zairachem_only_ever_addresses_its_own_project_and_services(docker):
  docker.running = {"someone_elses_api"}
  compose.up(["eosa_api"])
  compose.release_servers()
  assert all(c[: len(PREFIX)] == PREFIX for c in docker.commands)  # always `-p zairachem-servers`
  assert docker.stopped() == [{"eosa_api"}]
  assert docker.running == {"someone_elses_api"}
