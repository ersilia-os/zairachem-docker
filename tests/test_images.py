"""A model image that cannot be checked or cannot compute says so, and says why."""

import pytest
import requests

from zairachem.base.utils import preflight
from zairachem.describe.descriptors import profiles
from zairachem.describe.descriptors.api import BinaryStreamClient
from zairachem.describe.descriptors.profiles import ModelProfile


def _client(attempt):
  client = BinaryStreamClient.__new__(BinaryStreamClient)  # no server, no store
  client.model_id = "eosxxxx"
  client.logger = preflight.logger
  client.profile = ModelProfile("eosxxxx", "sha256:x")
  client._attempt = attempt
  return client


def test_a_failure_in_both_formats_reports_both():
  def attempt(batch, output_type):
    raise requests.HTTPError(f"500 in {output_type}")

  with pytest.raises(requests.HTTPError) as err:
    _client(attempt)._try_request(["CCO"])
  assert "500 in simple" in str(err.value)
  assert "500 in heavy" in str(err.value)


def test_a_remembered_format_that_stops_working_is_learned_again():
  def attempt(batch, output_type):
    if output_type == "heavy":
      raise requests.HTTPError("500 in heavy")
    return "rows", None

  client = _client(attempt)
  client.profile.output_type = "heavy"  # learned on an image that has since been replaced
  assert client._try_request(["CCO"]) == ("rows", None)
  assert client.profile.output_type == "simple"


def test_a_timeout_does_not_unlearn_the_remembered_format():
  def attempt(batch, output_type):
    raise requests.Timeout("slow")

  client = _client(attempt)
  client.profile.output_type = "heavy"
  with pytest.raises(requests.Timeout):
    client._try_request(["CCO"])
  assert client.profile.output_type == "heavy"


def test_what_belongs_to_the_image_is_not_kept_when_the_image_cannot_be_identified(monkeypatch):
  stored = {
    "eosxxxx": {
      "digest": "sha256:old",
      "samples": [[10, 1.0, 2.0]],
      "output_type": "heavy",
      "failed_at": 4096,
    }
  }
  monkeypatch.setattr(profiles, "_load_all", lambda: stored)
  unknown = profiles.load("eosxxxx", None)
  assert unknown.output_type is None and unknown.failed_at is None
  assert unknown.samples == [[10, 1.0, 2.0]]
  same = profiles.load("eosxxxx", "sha256:old")
  assert same.output_type == "heavy" and same.failed_at == 4096


def test_an_image_the_registry_cannot_vouch_for_is_not_shown_as_current(monkeypatch):
  monkeypatch.setattr(preflight, "_docker_image_present", lambda image: True)
  monkeypatch.setattr(preflight, "ersilia_model_version", lambda m: "v1")
  for current, state in ((True, "ok"), (False, "outdated"), (None, "unverified")):
    monkeypatch.setattr(preflight, "is_image_up_to_date", lambda m, c=current: c)
    assert preflight._probe_model_image(("projector", "eos1klk"))[4] == state


def _undigested_image(monkeypatch, local, registry):
  """A re-tagged image (no digest) with ``local`` layers, against a registry of ``{tag: layers}``."""
  from zairachem.base.utils import model_version

  monkeypatch.setattr(model_version, "_release_cache", {})
  monkeypatch.setattr(model_version, "_cache", {})
  monkeypatch.setattr(model_version, "_local_image_digest", lambda m: None)
  monkeypatch.setattr(model_version, "_local_image_layers_and_arch", lambda m: (local, "arm64"))
  monkeypatch.setattr(model_version, "_registry_layers", lambda m, arch, tag: registry[tag])
  # eos4ex3's real release history: the image was built before its v2.0.0 tag was pushed.
  tags = [
    ("v1.2.1", "2026-06-22T18:33:17Z", {"sha256:old"}),
    ("v2.0.0", "2026-08-31T11:05:32Z", {"sha256:new"}),
    ("latest", "2026-08-31T11:05:32Z", {"sha256:new"}),
  ]
  monkeypatch.setattr(model_version, "_dockerhub_tags", lambda repo: tags)
  return model_version


def test_an_image_without_a_registry_digest_is_judged_by_its_layers(monkeypatch):
  registry = {"latest": ["aaa"], "v2.0.0": ["aaa"], "v1.2.1": ["old"]}
  mv = _undigested_image(monkeypatch, ["aaa"], registry)
  assert mv.is_image_up_to_date("eos4ex3") is True
  mv = _undigested_image(monkeypatch, ["old"], registry)
  assert mv.is_image_up_to_date("eos4ex3") is False  # an older release: behind
  mv = _undigested_image(monkeypatch, ["built-here"], registry)
  assert mv.is_image_up_to_date("eos4ex3") is None  # matches nothing: unknown, not behind


def test_the_version_of_an_image_without_a_digest_does_not_depend_on_its_build_time(monkeypatch):
  registry = {"latest": ["aaa"], "v2.0.0": ["aaa"], "v1.2.1": ["old"]}
  mv = _undigested_image(monkeypatch, ["aaa"], registry)
  assert mv.resolved_model_version("eos4ex3") == "v2"  # built before v2.0.0 was pushed
  mv = _undigested_image(monkeypatch, ["old"], registry)
  assert mv.resolved_model_version("eos4ex3") == "v1"  # genuinely the older release
  mv = _undigested_image(monkeypatch, ["built-here"], registry)
  assert mv.resolved_model_version("eos4ex3") == "v2"  # unknown image: the registry's latest


def test_an_unreachable_registry_stops_a_run_instead_of_defaulting_the_version(monkeypatch):
  from zairachem.base.utils import model_version

  def reset(tags):
    monkeypatch.setattr(model_version, "_cache", {})
    monkeypatch.setattr(model_version, "_unreachable", set())
    monkeypatch.setattr(model_version, "_dockerhub_tags", tags)

  def down(repo):
    raise requests.ConnectionError("no route to hub.docker.com")

  reset(down)
  with pytest.raises(SystemExit):
    preflight.require_model_versions(["eosaaaa", "eosbbbb"])
  reset(down)  # a version the run already recorded is not looked up: predict works offline
  preflight.require_model_versions(["eosaaaa"], {"eosaaaa": "v2"})
  reset(lambda repo: [("latest", "2026-01-01T00:00:00Z", set())])
  preflight.require_model_versions(["eosaaaa"])  # no release at all is a real v1, not an outage


def test_reading_the_store_in_chunks_gives_each_chunk_its_own_rows(monkeypatch):
  import numpy as np
  import pandas as pd

  from zairachem.describe.descriptors import api

  class Reader:
    """Like isaura's: keeps the first inputs it is given and returns them on every later read."""

    def __init__(self, **kwargs):
      self.first = None

    def read(self, df):
      self.first = df if self.first is None else self.first
      return pd.DataFrame({"key": self.first["input"], "input": self.first["input"], "a": 1.0})

  monkeypatch.setattr(api, "_isaura_classes", lambda: (Reader, None))
  client = BinaryStreamClient.__new__(BinaryStreamClient)
  client.model_id, client.version, client.read_store = "eosxxxx", "v1", "bucket"
  values, cols = client._read_subset([f"C{i}" for i in range(25)], 10)
  assert cols == ["a"]
  assert not np.isnan(values).any()


def _server_answering(monkeypatch, heavy, simple):
  """A client whose server answers 200 in both formats, with the given arrays."""
  import types

  client = BinaryStreamClient.__new__(BinaryStreamClient)
  client.model_id = "eosxxxx"
  client.logger = preflight.logger
  client.profile = ModelProfile("eosxxxx", "sha256:x")
  client.url = "http://localhost:1/run"
  client.server_workers = 1
  client._ready = True
  client._feature_len = None
  response = types.SimpleNamespace(ok=True)
  monkeypatch.setattr(requests, "post", lambda *a, **k: response)
  client.decode_binary_stream = lambda r: (heavy, {})
  client._decode_json_records = lambda r: (simple, {})
  return client


def test_a_reply_with_no_values_is_not_a_success(monkeypatch):
  """An image that cannot read the binary request can still answer it with a 200 and empty rows.
  That used to pass as a result: the projections step reported done and wrote empty coordinates."""
  import numpy as np

  empty, values = np.full((3, 2), np.nan, dtype="float32"), np.ones((3, 2), dtype="float32")

  client = _server_answering(monkeypatch, empty, values)
  array, _ = client._try_request(["CCO", "CCN", "CCC"])
  assert np.array_equal(array, values) and client.profile.output_type == "simple"

  with pytest.raises(IOError) as err:
    _server_answering(monkeypatch, empty, empty)._try_request(["CCO", "CCN", "CCC"])
  assert "simple" in str(err.value) and "heavy" in str(err.value)

  # A model that cannot process one molecule leaves that row empty, which is not a failed reply.
  partly = values.copy()
  partly[1] = np.nan
  client = _server_answering(monkeypatch, partly, values)
  assert np.array_equal(client._try_request(["CCO", "CCN", "CCC"])[0], partly, equal_nan=True)
  assert client.profile.output_type == "heavy"


def test_a_projection_with_no_coordinates_is_not_written_or_reused(tmp_path, monkeypatch):
  import json
  import types

  import numpy as np

  from zairachem.describe.descriptors import api, profiles
  from zairachem.treat.imputers import manifolds as manifolds_module

  Manifolds = manifolds_module.Manifolds

  manifolds = Manifolds.__new__(Manifolds)
  manifolds.path, manifolds.inputs = str(tmp_path), ["CCO", "CCN"]
  cache = tmp_path / "descriptors" / "eosxxxx" / "projection.json"
  cache.parent.mkdir(parents=True)
  manifolds._cache_path = lambda model_id: str(cache)

  cache.write_text(json.dumps({"cols": ["a_x"], "rows": [{"a_x": None}, {"a_x": None}]}))
  assert manifolds._read_cache("eosxxxx") is None
  cache.write_text(json.dumps({"cols": ["a_x"], "rows": [{"a_x": 1.0}, {"a_x": None}]}))
  assert manifolds._read_cache("eosxxxx") is not None

  # And a projector that answers with nothing stops the step, rather than writing empty columns.
  made = []

  class Empty:
    def __init__(self, **kwargs):
      self.profile = None
      made.append(self)

    def run(self, **kwargs):
      return {"dims": ["a_x", "a_y"], "data": np.full((2, 2), np.nan, dtype="float32")}

  monkeypatch.setattr(api, "BinaryStreamClient", Empty)
  monkeypatch.setattr(manifolds_module, "get_model_url", lambda m: "http://localhost:1/run")
  monkeypatch.setattr(profiles, "load", lambda m, d: types.SimpleNamespace())
  monkeypatch.setattr(profiles, "save", lambda p: None)
  monkeypatch.setattr("zairachem.base.utils.model_version._local_image_digest", lambda m: None)
  cache.unlink()
  manifolds.input_file = str(tmp_path / "input.csv")
  manifolds._projection_sample, manifolds.batch_size = (lambda: None), 100
  with pytest.raises(RuntimeError, match="no coordinates"):
    manifolds._run_projection_model("eosxxxx")
  assert not cache.exists()
  assert made[0].fixed_output_type == "simple"  # projectors always ask for JSON


def test_a_client_with_a_fixed_format_neither_learns_nor_falls_back():
  """Projectors send one small request, where binary buys nothing; an image that cannot read it
  can answer with empty rows, so they use JSON and never try the other format."""
  seen = []

  def attempt(batch, output_type):
    seen.append(output_type)
    raise requests.HTTPError("500 in simple")

  client = _client(attempt)
  client.fixed_output_type = "simple"
  with pytest.raises(requests.HTTPError):
    client._try_request(["CCO"])
  assert seen == ["simple"] and client.profile.output_type is None

  client = _client(lambda batch, output_type: ("rows", None))
  client.fixed_output_type = "simple"
  assert client._try_request(["CCO"]) == ("rows", None)
  assert client.profile.output_type is None
