"""A pinned descriptor runs a released tag of its pinned major, whatever Docker Hub's :latest is."""

import io
import json
import subprocess
import types

import pytest
import requests

from zairachem.base.utils import model_version as mv, preflight
from zairachem.base.vars import DEFAULT_FEATURIZERS, MODEL_IMAGES_KEY

_real_dockerhub_tags = mv._dockerhub_tags

# Docker Hub's tags of eos3l5f: :latest has moved on to a new major, v3.
HUB = [
  ("v1.3.0", "2026-01-01T00:00:00Z", {"sha256:v130"}),
  ("v2.0.0", "2026-03-01T00:00:00Z", {"sha256:v200"}),
  ("v2.0.1", "2026-04-01T00:00:00Z", {"sha256:v201"}),
  ("v3.0.0", "2026-11-01T00:00:00Z", {"sha256:v300"}),
  ("latest", "2026-11-01T00:00:00Z", {"sha256:v300"}),
]


def _offline(repo):
  raise requests.ConnectionError("Docker Hub unreachable")


def _full(ref):
  return ref if ":" in ref.split("/")[-1] else f"{ref}:latest"


class _Docker:
  """The local images (``ref -> digest``, None for none), answering ``docker image inspect``,
  ``image ls`` and ``tag``; ``layers`` gives an image's layers (``ref -> list``)."""

  def __init__(self, images, layers=None, tag_fails=False):
    self.images = {_full(r): d for r, d in images.items()}
    self.layers = {_full(r): v for r, v in (layers or {}).items()}
    self.tag_fails = tag_fails
    self.tagged = []

  def run(self, cmd, **kwargs):
    def done(code, out="", err=""):
      return subprocess.CompletedProcess(cmd, code, stdout=out, stderr=err)

    if cmd[:3] == ["docker", "image", "inspect"]:
      ref = _full(cmd[3])
      if ref not in self.images:
        return done(1)
      if "RootFS" in cmd[-1]:
        return done(0, f"amd64 {json.dumps(self.layers[ref])}") if ref in self.layers else done(1)
      digest = self.images[ref]
      return done(0, f"{ref.split(':')[0]}@{digest}\n" if digest else "")
    if cmd[:3] == ["docker", "image", "ls"]:
      return done(0, "\n".join(r.split(":")[1] for r in self.images if r.split(":")[0] == cmd[3]))
    if cmd[:2] == ["docker", "tag"] and not self.tag_fails:
      self.tagged.append(cmd[2:])
      self.images[_full(cmd[3])] = self.images[_full(cmd[2])]
      return done(0)
    if cmd[:2] == ["docker", "tag"]:
      return done(1, err="no space left on device")
    raise AssertionError(f"unexpected command {cmd}")

  def present(self, image):
    return _full(image) in self.images


@pytest.fixture
def docker(monkeypatch):
  """Mocked Docker and Docker Hub; ``docker(images, hub)`` sets what each holds."""
  for name in ("_cache", "_digest_cache", "_tags_cache", "_release_cache"):
    monkeypatch.setattr(mv, name, {})
  monkeypatch.setattr(mv, "_unreachable", set())

  def setup(images, hub=HUB, **kwargs):
    local = _Docker(images, **kwargs)
    monkeypatch.setattr(mv.subprocess, "run", local.run)
    monkeypatch.setattr(mv, "_dockerhub_tags", hub if callable(hub) else lambda repo: hub)
    monkeypatch.setattr(preflight, "_docker_image_present", local.present)
    for cache in (mv._cache, mv._digest_cache, mv._tags_cache, mv._release_cache):
      cache.clear()
    return local

  return setup


@pytest.fixture
def said(monkeypatch):
  """Everything preflight prints, one string per line."""
  from rich.console import Console

  lines = []

  def show(*a, **k):
    out = Console(file=io.StringIO(), width=200, record=True)
    out.print(*a)
    lines.append(out.export_text())

  monkeypatch.setattr(preflight, "echo", lambda msg, **k: lines.append(str(msg)))
  monkeypatch.setattr(preflight.console, "print", show)
  return lines


def test_a_pinned_model_runs_when_latest_has_moved_on(docker, said):
  # The local :latest is an old v2 build; Docker Hub's :latest is now v3. Unpinned, that stops.
  docker({"ersiliaos/eos3l5f:latest": "sha256:v200", "ersiliaos/eos3l5f:v2.0.1": "sha256:v201"})
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])
  text = "\n".join(said)
  assert "ersiliaos/eos3l5f:v2.0.1" in text and "missing" not in text


def test_no_local_tag_of_the_pinned_major_stops_with_the_docker_pull_of_its_newest(docker, said):
  docker({"ersiliaos/eos3l5f:latest": "sha256:v300"})  # only :latest, which is v3
  mv.pin_models(["eos3l5f"])
  with pytest.raises(SystemExit):
    preflight.report_model_images(["eos3l5f"], [])
  text = "\n".join(said)
  assert "docker pull ersiliaos/eos3l5f:v2.0.1" in text  # Docker Hub's newest v2
  assert "ersilia fetch eos3l5f" not in text  # it would overwrite :latest, and not get v2
  # Docker Hub unreachable too: the stop says so, and names the major.
  said.clear()
  docker({"ersiliaos/eos3l5f:latest": "sha256:v300"}, hub=_offline)
  mv.pin_models(["eos3l5f"])
  with pytest.raises(SystemExit):
    preflight.report_model_images(["eos3l5f"], [])
  assert any("unreachable" in line and "v2" in line for line in said)


def test_a_newer_tag_of_the_major_on_docker_hub_is_a_line_not_a_stop(docker, said):
  hub = [(t, "2026-01-01T00:00:00Z", set()) for t in ("v1.1.1", "v1.2.0")]
  docker({"ersiliaos/eos9q2i:v1.1.1": "sha256:v111"}, hub=hub)
  mv.pin_models(["eos9q2i"])
  preflight.report_model_images(["eos9q2i"], [])  # no stop
  assert mv.image_tag("eos9q2i") == "v1.1.1"
  line = next(line for line in said if "is on Docker Hub" in line)
  assert "eos9q2i v1.2.0" in line and "local v1.1.1" in line
  assert "docker pull ersiliaos/eos9q2i:v1.2.0" in line


def test_an_unreachable_docker_hub_runs_the_local_tag(docker, said):
  docker({"ersiliaos/eos3l5f:v2.0.0": "sha256:v200"}, hub=_offline)
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])  # no stop
  assert "ersiliaos/eos3l5f:v2.0.0" in "\n".join(said)


def test_the_pin_is_the_version_of_the_scaler_the_store_and_the_rank_reference(docker):
  # The local image is v3, which is what the digest says; the pin still decides.
  docker({"ersiliaos/eos3l5f:latest": "sha256:v300", "ersiliaos/eos3l5f:v2.0.1": "sha256:v201"})
  assert mv.ersilia_model_version("eos3l5f") == "v3"  # unpinned: read from the image
  mv.pin_models(["eos3l5f"])
  assert mv.ersilia_model_version("eos3l5f") == "v2"
  assert mv.resolved_model_version("eos3l5f") == "v2"
  # The transformer setup downloads is the pinned major's.
  asked = []
  rt = types.SimpleNamespace(
    transformer_url=lambda m, v, reference_library: f"{m}_{v}",
    fetch_reference_transformer=lambda m, v, reference_library: asked.append(v) or {},
    check_transformer=lambda *a: None,
    save_local_transformer=lambda *a: None,
    local_transformer_name=lambda m, v: f"{m}_{v}",
  )
  assert preflight._fetch_transformer(rt, "eos3l5f", "lib", "/nowhere")[1] == "v2"
  assert asked == ["v2"]
  # A pinned version is known even when Docker Hub is not reachable.
  docker({"ersiliaos/eos3l5f:v2.0.1": "sha256:v201"}, hub=_offline)
  preflight.require_model_versions(["eos3l5f"])


def test_the_tag_is_the_newest_local_one_of_the_pinned_major(docker):
  tags = ("v2.2.0", "v2.10.0", "v1.9.0", "v3.0.0", "latest")
  hub = [(t, "2026-01-01T00:00:00Z", set()) for t in (*tags, "v2.11.0")]
  docker({f"ersiliaos/eos3l5f:{t}": "sha256:x" for t in tags}, hub=hub)
  mv.pin_models(["eos3l5f"])
  assert mv.image_tag("eos3l5f") == "v2.10.0"  # by version, not by text; v2.11.0 is not here
  # Nothing of the major here: no tag, so the check stops instead of running :latest.
  docker({"ersiliaos/eos3l5f:latest": "sha256:x"})
  mv.pin_models(["eos3l5f"])
  assert mv.image_tag("eos3l5f") is None
  assert preflight._probe_model_image(("descriptor", "eos3l5f"))[4] == "missing"


def test_each_tag_of_a_pinned_model_gets_its_own_compose_server(docker, monkeypatch, tmp_path):
  from zairachem.base.generate_config import generate_compose
  from zairachem.describe.descriptors import compose
  from zairachem.describe.descriptors.describe import Describer
  from zairachem.describe.descriptors.utils import _service_name, get_model_host_port

  monkeypatch.setattr(compose, "compose_dir", lambda: tmp_path)
  docker({"ersiliaos/eos3l5f:v2.0.1": "sha256:v201"})
  mv.pin_models(["eos3l5f", "eosxtra"])
  run = types.SimpleNamespace(models=["eos3l5f", "eosxtra"])
  Describer.create_config_files(run)
  assert _service_name("eos3l5f") == "eos3l5f_v2_0_1_api"
  assert _service_name("eosxtra") == "eosxtra_api"  # unpinned: today's name
  yml = compose.compose_file().read_text()
  assert "eos3l5f_v2_0_1_api:\n    image: ersiliaos/eos3l5f:v2.0.1\n" in yml
  assert "eosxtra_api:\n    image: ersiliaos/eosxtra\n" in yml
  first = get_model_host_port("eos3l5f")
  # A model fitted on v2.0.0 runs v2.0.0, on a server beside the v2.0.1 one.
  mv.pin_models(["eos3l5f"], {MODEL_IMAGES_KEY: {"eos3l5f": {"tag": "v2.0.0"}}})
  Describer.create_config_files(run)
  assert get_model_host_port("eos3l5f") not in (None, first)
  assert set(compose.load_ports()) == {"eos3l5f:v2.0.1", "eos3l5f:v2.0.0", "eosxtra"}
  assert "eos3l5f_v2_0_0_api" in generate_compose(compose.load_ports())


@pytest.fixture
def fit_setup(monkeypatch, tmp_path):
  """A fit's setup() with its data steps and network checks replaced, down to the image check."""
  from zairachem.setup.prep import training

  class Reached(Exception):
    pass

  noop = lambda *a, **k: None  # noqa: E731
  for gate in (
    "require_docker_and_base",
    "warn_other_runs",
    "reap_orphaned_servers",
    "validate_model_roles",
    "prepare_rank_references",
  ):
    monkeypatch.setattr(training, gate, noop)
  monkeypatch.setattr(training, "prepare_reference_transformers", lambda ids, *a: list(ids))

  def reached(*a, **k):
    raise Reached

  monkeypatch.setattr(training, "check_isaura_version_consistency", reached)
  for step in (
    "_initialize",
    "_normalize_input",
    "_standardise",
    "_tasks",
    "_merge",
    "_check",
    "_clean",
    "_evaluate_splits",
    "_print_summary",
  ):
    monkeypatch.setattr(training.TrainSetup, step, noop)
  (tmp_path / "metadata").mkdir()
  ts = training.TrainSetup.__new__(training.TrainSetup)
  ts.output_dir, ts.resume, ts.task = str(tmp_path), False, "classification"
  ts.featurizer_ids, ts.projection_ids = ["eos3l5f", "eosxtra"], ["eos1klk"]
  ts.params = {"featurizer_ids": ts.featurizer_ids, "projection_ids": ts.projection_ids}
  ts.reached = Reached
  return ts


def test_a_fit_records_the_tag_and_digest_it_ran(docker, fit_setup):
  docker({
    "ersiliaos/eos3l5f:latest": "sha256:v200",  # an old :latest, behind Docker Hub's
    "ersiliaos/eos3l5f:v2.0.1": "sha256:v201",
    "ersiliaos/eosxtra:latest": "sha256:xtra",
    "ersiliaos/eos1klk:latest": "sha256:klk",
  })
  with pytest.raises(fit_setup.reached):  # past the image check: the run goes on
    fit_setup.setup()
  with open(fit_setup.output_dir + "/metadata/parameters.json") as f:
    images = json.load(f)[MODEL_IMAGES_KEY]
  assert images == {
    "eos3l5f": {"tag": "v2.0.1", "digest": "sha256:v201"},
    "eosxtra": {"tag": "latest", "digest": "sha256:xtra"},
    "eos1klk": {"tag": "latest", "digest": "sha256:klk"},  # unpinned until its pin is set
  }


@pytest.fixture
def predict_setup(monkeypatch, tmp_path):
  """A predict's setup() against a fitted model, its data steps replaced, down to the image check."""
  from zairachem.setup.prep import prediction

  class Reached(Exception):
    pass

  noop = lambda *a, **k: None  # noqa: E731
  for gate in ("require_docker_and_base", "warn_other_runs", "reap_orphaned_servers"):
    monkeypatch.setattr(prediction, gate, noop)

  def reached(*a, **k):
    raise Reached

  monkeypatch.setattr(prediction, "check_isaura_version_consistency", reached)
  for step in ("_initialize", "_normalize_input", "_standardize", "_merge", "_check", "_clean"):
    monkeypatch.setattr(prediction.PredictSetup, step, noop)
  monkeypatch.setattr(prediction.PredictSetup, "_print_summary", noop)
  monkeypatch.setattr(prediction.PredictSetup, "has_tasks", False)
  monkeypatch.setattr(
    prediction.PredictSetup, "_model_id_groups", lambda self: (["eos3l5f", "eosxtra"], [])
  )

  def make(fitted):
    for d in ("model", "run"):
      (tmp_path / d / "metadata").mkdir(parents=True, exist_ok=True)
      (tmp_path / d / "metadata" / "parameters.json").write_text(json.dumps(fitted))
    ps = prediction.PredictSetup.__new__(prediction.PredictSetup)
    ps.model_dir, ps.output_dir = str(tmp_path / "model"), str(tmp_path / "run")
    ps.contribute_store = None
    ps.reached = Reached
    return ps

  return make


def test_predict_runs_the_tag_recorded_at_fit_and_stops_without_it(docker, predict_setup, said):
  fitted = {
    "latest_featurizer_version": {"eos3l5f": "v2", "eosxtra": "v1"},
    MODEL_IMAGES_KEY: {"eos3l5f": {"tag": "v2.0.0"}, "eosxtra": {"tag": "latest"}},
  }
  # v2.0.1 is newer, but the model was fitted with v2.0.0.
  images = {"ersiliaos/eos3l5f:v2.0.1": "sha256:v201", "ersiliaos/eosxtra:latest": "sha256:xtra"}
  # Its :latest is v2.0.1, newer, but a recorded tag runs exactly, and is never made from :latest.
  local = docker(
    {**images, "ersiliaos/eos3l5f:latest": "sha256:v201"},
    hub=[*HUB, ("latest", "2026-01-01T00:00:00Z", {"sha256:xtra"})],
  )
  ps = predict_setup(fitted)
  with pytest.raises(SystemExit):
    ps.setup()
  assert "docker pull ersiliaos/eos3l5f:v2.0.0" in "\n".join(said)
  assert local.tagged == []
  docker({**images, "ersiliaos/eos3l5f:v2.0.0": "sha256:v200"})
  ps = predict_setup(fitted)
  with pytest.raises(ps.reached):
    ps.setup()
  assert mv.image_tag("eos3l5f") == "v2.0.0"
  # The fit's tag runs: a newer tag on Docker Hub is not this run's business.
  assert not any("is on Docker Hub" in line for line in said)
  with open(ps.output_dir + "/metadata/parameters.json") as f:
    assert json.load(f)[MODEL_IMAGES_KEY]["eos3l5f"] == {"tag": "v2.0.0", "digest": "sha256:v200"}


def test_a_model_fitted_before_tags_were_recorded_runs_its_fitted_major(docker):
  from zairachem.describe.descriptors.raw import RawDescriptors

  # Fitted at v1 (no tag recorded); the table now pins v2, and :latest here is v3.
  fitted = {"latest_featurizer_version": {"eos3l5f": "v1"}}
  docker({"ersiliaos/eos3l5f:latest": "sha256:v300", "ersiliaos/eos3l5f:v1.3.0": "sha256:v130"})
  mv.pin_models(["eos3l5f"], fitted)
  assert mv.ersilia_model_version("eos3l5f") == "v1"
  assert mv.image_tag("eos3l5f") == "v1.3.0"
  assert preflight._probe_model_image(("descriptor", "eos3l5f"))[4] == "ok"
  # The image runs the fitted major, so predict has no "version moved" to warn about.
  warned = []
  run = types.SimpleNamespace(
    params=fitted, logger=types.SimpleNamespace(warning=lambda m, *a, **k: warned.append(m))
  )
  RawDescriptors._warn_if_versions_moved(run, ["eos3l5f"])
  assert warned == []


def test_a_model_without_a_pin_runs_latest_as_before(docker, monkeypatch):
  from zairachem.describe.descriptors.utils import _service_name

  monkeypatch.setattr(preflight, "is_image_up_to_date", lambda m: False)
  docker({"ersiliaos/eosxtra:latest": "sha256:xtra", "ersiliaos/eos1klk:latest": "sha256:klk"})
  mv.pin_models(["eosxtra", "eos1klk"], {MODEL_IMAGES_KEY: {"eosxtra": {"tag": "latest"}}})
  for m in ("eosxtra", "eos1klk"):  # an -f id, and the projection whose pin is not set yet
    assert mv.pinned_major(m) is None
    assert mv.image_ref(m) == f"ersiliaos/{m}"
    assert _service_name(m) == f"{m}_api"
    kind, _, image, _, state, _ = preflight._probe_model_image(("descriptor", m))
    assert (image, state) == (f"ersiliaos/{m}:latest", "outdated")  # behind :latest still stops
    assert "ersilia fetch" in preflight._fetch_hint(m)


@pytest.mark.parametrize("eos_id,major", sorted(DEFAULT_FEATURIZERS.items()))
def test_every_pinned_major_has_a_published_scaler_and_rank_reference(eos_id, major):
  """A pin moves only once its major's scaler and rank reference are published; this catches a
  pin that moved early."""
  from zairachem.describe.descriptors.rank_reference import published_url
  from zairachem.treat.imputers.reference_transformer import transformer_url

  for url in (transformer_url(eos_id, major), published_url(eos_id, major)):
    try:
      status = requests.head(url, timeout=10).status_code
    except requests.RequestException:
      pytest.skip("offline")
    assert status == 200, url


def test_a_step_command_runs_the_tags_its_setup_recorded(docker, tmp_path):
  from zairachem.cli.steps import _activate_step

  docker({"ersiliaos/eos3l5f:v2.0.1": "sha256:v201"})  # newer here, but setup chose v2.0.0
  (tmp_path / "metadata").mkdir()
  (tmp_path / "metadata" / "parameters.json").write_text(
    json.dumps({
      "featurizer_ids": ["eos3l5f"],
      "projection_ids": [],
      MODEL_IMAGES_KEY: {"eos3l5f": {"tag": "v2.0.0"}},
    })
  )
  _activate_step(str(tmp_path))
  assert mv.image_tag("eos3l5f") == "v2.0.0"


def test_install_pulls_docker_hubs_newest_tag_of_each_pinned_major(docker, capsys):
  docker({})
  mv.install_images(["eos3l5f", "eos1klk", "eosxtra"])
  assert capsys.readouterr().out.split("\n") == ["eos3l5f v2.0.1", "eos1klk -", "eosxtra -", ""]
  docker({}, hub=_offline)
  mv.install_images(["eos3l5f"])
  assert capsys.readouterr().out == "eos3l5f !\n"  # Docker Hub unreachable
  # ...unless a tag of the major is here: it is then reported as already present.
  docker({"ersiliaos/eos3l5f:v2.0.0": "sha256:v200"}, hub=_offline)
  mv.install_images(["eos3l5f"])
  assert capsys.readouterr().out == "eos3l5f v2.0.0\n"
  # No ids: the default models, in the table's order.
  mv.install_images([])
  assert [ln.split()[0] for ln in capsys.readouterr().out.splitlines()] == [
    *DEFAULT_FEATURIZERS,
    "eos1klk",
  ]


def test_an_upgrading_users_latest_is_tagged_as_the_release_it_is(docker, said):
  # Pulled before pins: :latest is v2.0.1 by digest, and no v2 tag is here.
  local = docker({"ersiliaos/eos3l5f:latest": "sha256:v201"})
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])  # no stop
  assert local.tagged == [["ersiliaos/eos3l5f:latest", "ersiliaos/eos3l5f:v2.0.1"]]
  assert mv.image_tag("eos3l5f") == "v2.0.1"
  assert local.images["ersiliaos/eos3l5f:latest"] == "sha256:v201"  # :latest left as it was
  assert any("release v2.0.1" in line and "nothing downloaded" in line for line in said)


def test_a_retagged_latest_without_a_digest_is_matched_by_its_layers(docker, monkeypatch, said):
  local = docker({"ersiliaos/eos3l5f:latest": None}, layers={"ersiliaos/eos3l5f:latest": ["b"]})
  registry = {"v2.0.1": ["c"], "v2.0.0": ["b"], "v1.3.0": ["a"], "v3.0.0": ["d"]}
  monkeypatch.setattr(mv, "_registry_layers", lambda m, arch, tag: registry[tag])
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])
  assert local.tagged == [["ersiliaos/eos3l5f:latest", "ersiliaos/eos3l5f:v2.0.0"]]


@pytest.mark.parametrize(
  "latest,hub,tag_fails",
  [
    ("sha256:v300", HUB, False),  # :latest is a v3 release: not the pinned major
    ("sha256:v201", _offline, False),  # Docker Hub cannot say which release it is
    ("sha256:v201", HUB, True),  # docker tag fails
  ],
)
def test_a_latest_that_cannot_be_tagged_as_the_pinned_major_stops(
  docker, said, monkeypatch, latest, hub, tag_fails
):
  local = docker({"ersiliaos/eos3l5f:latest": latest}, hub=hub, tag_fails=tag_fails)
  warned = []
  monkeypatch.setattr(mv.logger, "warning", lambda msg, *a, **k: warned.append(str(msg)))
  mv.pin_models(["eos3l5f"])
  with pytest.raises(SystemExit):
    preflight.report_model_images(["eos3l5f"], [])
  assert mv.image_tag("eos3l5f") is None
  assert local.images["ersiliaos/eos3l5f:latest"] == latest
  if tag_fails:  # the reason is shown, not only the stop
    assert local.tagged == []
    assert any("no space left on device" in w for w in warned)


def test_a_latest_of_another_major_is_not_compared_layer_by_layer(docker, monkeypatch, said):
  # :latest is v3 by its digest: no v2 layers can match, so the registry is not asked for them.
  docker({"ersiliaos/eos3l5f:latest": "sha256:v300"}, layers={"ersiliaos/eos3l5f:latest": ["d"]})
  asked = []
  monkeypatch.setattr(mv, "_registry_layers", lambda m, arch, tag: asked.append(tag) or ["x"])
  mv.pin_models(["eos3l5f"])
  with pytest.raises(SystemExit):
    preflight.report_model_images(["eos3l5f"], [])
  assert asked == []


def test_a_later_run_without_the_pin_drops_it(docker):
  docker({"ersiliaos/eosxtra:v1.0.0": "sha256:x1", "ersiliaos/eosxtra:latest": "sha256:xl"})
  mv._pins.clear()  # this test sets its own state, whatever the shared fixtures do
  mv.pin_models(["eosxtra"], {MODEL_IMAGES_KEY: {"eosxtra": {"tag": "v1.0.0"}}})
  assert mv.image_ref("eosxtra") == "ersiliaos/eosxtra:v1.0.0"
  assert mv._local_image_digest("eosxtra") == "sha256:x1"
  mv.pin_models(["eosxtra"])  # the next run in this process fits eosxtra with -f, unpinned
  assert mv.pinned_major("eosxtra") is None
  assert mv.image_ref("eosxtra") == "ersiliaos/eosxtra"
  assert mv._local_image_digest("eosxtra") == "sha256:xl"  # not the cached digest of v1.0.0


def test_pins_ignore_the_case_of_ids(docker):
  docker({"ersiliaos/eos3l5f:v2.0.1": "sha256:v201"})
  mv.pin_models(["EOS3L5F"])
  assert mv.pinned_major("eos3l5f") == mv.pinned_major("EOS3L5F") == "v2"
  assert mv.image_ref("EOS3L5F") == "ersiliaos/eos3l5f:v2.0.1"
  assert mv.server_key("EOS3L5F") == mv.server_key("eos3l5f") == "eos3l5f:v2.0.1"  # one server


def test_an_unreachable_docker_hub_is_asked_once_per_repo(docker, monkeypatch, said):
  # No v2 tag here, only a :latest: setup asks which release it is, then what to pull.
  docker({"ersiliaos/eos3l5f:latest": "sha256:v201"})
  monkeypatch.setattr(mv, "_dockerhub_tags", _real_dockerhub_tags)
  calls = []

  def get(url, **kwargs):
    calls.append(url)
    raise requests.ConnectionError("Docker Hub unreachable")

  monkeypatch.setattr(mv.requests, "get", get)
  mv.pin_models(["eos3l5f"])
  with pytest.raises(SystemExit):
    preflight.report_model_images(["eos3l5f"], [])
  assert len(calls) == 1
  assert any("unreachable" in line for line in said)


def test_a_step_command_without_a_local_tag_says_what_to_pull(docker):
  import click

  from zairachem.describe.descriptors.utils import _service_name

  docker({})
  mv.pin_models(["eos3l5f"])
  with pytest.raises(click.ClickException, match="docker pull ersiliaos/eos3l5f:v2.0.1"):
    _service_name("eos3l5f")


def test_a_newer_release_in_latest_beats_a_stale_local_tag(docker, said):
  # An old v2.0.0 tag is here, and :latest is v2.1.1 by its digest (eos9o72 on one machine).
  hub = [*HUB, ("v2.1.1", "2026-06-01T00:00:00Z", {"sha256:v211"})]
  local = docker(
    {"ersiliaos/eos3l5f:v2.0.0": "sha256:v200", "ersiliaos/eos3l5f:latest": "sha256:v211"}, hub=hub
  )
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])
  assert local.tagged == [["ersiliaos/eos3l5f:latest", "ersiliaos/eos3l5f:v2.1.1"]]
  assert mv.image_tag("eos3l5f") == "v2.1.1"
  assert mv.server_key("eos3l5f") == "eos3l5f:v2.1.1"
  # A local tag as new as :latest runs as it is, with nothing tagged.
  local = docker(
    {"ersiliaos/eos3l5f:v2.1.1": "sha256:v211", "ersiliaos/eos3l5f:latest": "sha256:v211"}, hub=hub
  )
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])
  assert local.tagged == [] and mv.image_tag("eos3l5f") == "v2.1.1"
  # Docker Hub unreachable: :latest cannot be identified, and the local v2.0.0 runs.
  local = docker(
    {"ersiliaos/eos3l5f:v2.0.0": "sha256:v200", "ersiliaos/eos3l5f:latest": "sha256:v211"},
    hub=_offline,
  )
  mv.pin_models(["eos3l5f"])
  preflight.report_model_images(["eos3l5f"], [])
  assert local.tagged == [] and mv.image_tag("eos3l5f") == "v2.0.0"
