"""A descriptor left out for want of a transformer is named, with its version and why."""

import pytest
from rich.text import Text

from zairachem.base.utils import preflight
from zairachem.base.utils.console import console

LIB = "ersilia_reference_library_v0"


def _missing(m, version, error_tail):
  url = f"https://example.org/{LIB}/{m}_{version}_transformer.json"
  return (m, version, url, f"Reference transformer for '{m}' {error_tail.format(url=url)}")


def _report(kept, missing, n):
  with console.capture() as out:
    preflight._report_transformer_gaps(kept, missing, n, LIB)
  # Plain text whatever the terminal (colour codes split the words); the console wraps long lines.
  return " ".join(Text.from_ansi(out.get()).plain.split())


def test_each_left_out_descriptor_is_named_with_its_version_and_reason():
  missing = [
    _missing("eos3l5f", "v2", "(HTTP 404 at {url})"),
    _missing("eos8aa5", "v3", "from {url}: ConnectionError(boom)"),
  ]
  text = _report(["eos8gop"], missing, 3)
  assert "1 of 3 descriptors have a usable transformer" in text
  assert "the other 2 are left out of this model" in text
  assert "eos3l5f (v2)" in text and "not published (HTTP 404)" in text
  assert "eos8aa5 (v3)" in text and "from the URL above: ConnectionError(boom)" in text


def test_a_server_error_is_not_called_unpublished():
  text = _report(["eos8gop"], [_missing("eos3l5f", "v2", "(HTTP 503 at {url})")], 2)
  assert "HTTP 503" in text
  assert "not published" not in text


def test_nothing_is_printed_when_every_transformer_is_there():
  assert _report(["eos8gop", "eos3l5f"], [], 2) == ""


def test_a_403_may_be_unreadable_rather_than_unpublished():
  text = _report(["eos8gop"], [_missing("eos3l5f", "v2", "(HTTP 403 at {url})")], 2)
  assert "not published or not readable (HTTP 403)" in text


def test_no_transformer_at_all_still_stops_the_run_and_says_why():
  with console.capture() as out, pytest.raises(SystemExit):
    preflight._report_transformer_gaps(
      [], [_missing("eos3l5f", "v2", "from {url}: ConnectionError(boom)")], 1, LIB
    )
  text = " ".join(Text.from_ansi(out.get()).plain.split())
  assert "eos3l5f (v2)" in text and "ConnectionError(boom)" in text
