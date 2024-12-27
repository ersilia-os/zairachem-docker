import os
import pandas as pd

from zairabase.vars import (
    COMPOUNDS_FILENAME,
    STANDARD_COMPOUNDS_FILENAME,
    FOLDS_FILENAME,
    TASKS_FILENAME,
    VALUES_FILENAME,
    DATA_FILENAME
)

class SetupCleaner(object):
    def __init__(self, path):
        self.path = path
        self.data_file = pd.read_csv(os.path.join(path, DATA_FILENAME))

    def _individual_files(self):
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

    def _clean_data_file(self):
        keep_cols = [col for col in list(self.data_file.columns) if "clf" not in col or "reg" not in col or "value" not in col] #TODO amend once Reg is added
        keep_cols = [col for col in self.data_file.columns if not any(keyword in col for keyword in ["clf", "reg", "value"])]
        df = self.data_file[keep_cols]
        df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
        
    def run(self):
        self._individual_files()
        self._clean_data_file()