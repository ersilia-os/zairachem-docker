"""A model image that cannot be checked or cannot compute says so, and says why."""

import pytest
import requests

from zairachem.base.utils import preflight
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


def test_an_image_the_registry_cannot_vouch_for_is_not_shown_as_current(monkeypatch):
  monkeypatch.setattr(preflight, "_docker_image_present", lambda image: True)
  monkeypatch.setattr(preflight, "ersilia_model_version", lambda m: "v1")
  for current, state in ((True, "ok"), (False, "outdated"), (None, "unverified")):
    monkeypatch.setattr(preflight, "is_image_up_to_date", lambda m, c=current: c)
    assert preflight._probe_model_image(("projector", "eos1klk"))[4] == state


def test_an_image_without_a_registry_digest_is_current_only_if_its_config_matches(monkeypatch):
  from zairachem.base.utils import model_version

  monkeypatch.setattr(model_version, "_local_image_digest", lambda m: None)
  monkeypatch.setattr(model_version, "_local_image_id_and_arch", lambda m: ("sha256:aaa", "arm64"))
  monkeypatch.setattr(model_version, "_registry_latest_config", lambda m, arch: "sha256:aaa")
  assert model_version.is_image_up_to_date("eosxxxx") is True
  # Built or retagged locally, so it cannot be called behind: it is just unknown.
  monkeypatch.setattr(model_version, "_registry_latest_config", lambda m, arch: "sha256:bbb")
  assert model_version.is_image_up_to_date("eosxxxx") is None


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
