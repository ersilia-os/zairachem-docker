import os
import pandas as pd

from zairachem.base.vars import (
  COMPOUNDS_FILENAME,
  STANDARD_COMPOUNDS_FILENAME,
  FOLDS_FILENAME,
  TASKS_FILENAME,
  VALUES_FILENAME,
  DATA_FILENAME,
)
from zairachem.base.utils.logging import logger


class SetupCleaner(object):
  def __init__(self, path):
    self.path = path
    self.data_file = pd.read_csv(os.path.join(path, DATA_FILENAME))

  def _individual_files(self):
    removed = 0
    for f in [
      COMPOUNDS_FILENAME,
      STANDARD_COMPOUNDS_FILENAME,
      FOLDS_FILENAME,
      TASKS_FILENAME,
      VALUES_FILENAME,
    ]:
      path = os.path.join(self.path, f)
      if os.path.exists(path):
        os.remove(path)
        removed += 1
    logger.info(f"[clean] Removed {removed} intermediate files")

  def _clean_data_file(self):
    keep_cols = [
      col
      for col in self.data_file.columns
      if not any(keyword in col for keyword in ["clf", "reg", "value"])
    ]
    df = self.data_file[keep_cols]
    df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
    logger.info(f"[clean] Cleaned data file: {len(df):,} rows, {len(keep_cols)} columns")

  def run(self):
    logger.info("[clean] Starting cleanup")
    self._individual_files()
    self._clean_data_file()
    logger.info("[clean] Cleanup complete")
