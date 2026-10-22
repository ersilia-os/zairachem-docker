"""Shared test setup: in-process tests log to a temporary file, not the user's ``console.log``."""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _log_to_tmp(tmp_path_factory):
  from zairachem.base.utils import logging as zlogging

  log_file = os.path.join(str(tmp_path_factory.mktemp("logs")), "console.log")
  original = zlogging.Logger._log_to_file

  def _to_tmp(self):
    self._file = self.logger.add(log_file, level="DEBUG", format=zlogging._FILE_FORMAT)

  zlogging.Logger._log_to_file = _to_tmp
  zlogging.logger.configure()
  yield
  zlogging.Logger._log_to_file = original
  zlogging.logger.configure()
