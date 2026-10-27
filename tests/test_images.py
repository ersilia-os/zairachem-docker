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
