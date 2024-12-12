import shutil
import os

from . import (
    COMPOUNDS_FILENAME,
    STANDARD_COMPOUNDS_FILENAME,
    FOLDS_FILENAME,
    TASKS_FILENAME,
    VALUES_FILENAME,
)


class SetupCleaner(object):
    def __init__(self, path):
        self.path = path

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

    def run(self):
        self._individual_files()
