import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class MetaLogger:
    """
    Logs the way we like. Maybe should go in the plotting folder because they go together.
    """
    def __init__(self, logging_directory) -> None:
        super().__init__()
        self._logging_directory = logging_directory
        os.makedirs(logging_directory, exist_ok=True)
        self._logging_values = {}
        self._filenames = {}

    def add_field(self, field_name, filename):
        assert isinstance(field_name, str)
        assert field_name != ""
        for char in [" ", "/", "\\"]:
            assert char not in field_name

        folder_name = os.path.join(self._logging_directory, field_name)
        os.makedirs(folder_name, exist_ok=True)
        print(f"Successfully created the directory {folder_name}")

        full_path = os.path.join(folder_name, filename)
        self._filenames[field_name] = full_path

        assert field_name not in self._logging_values

        self._logging_values[field_name] = []

    def append_datapoint(self, field_name, datapoint, write=False):
        self._logging_values[field_name].append(datapoint)
        if write:
            self.write_field(field_name)

    def write_field(self, field_name):
        full_path = self._filenames[field_name]
        values = self._logging_values[field_name]
        with open(full_path, "wb+") as f:
            pickle.dump(values, f)

    def write_all_fields(self):
        for field_name in self._filenames.keys():
            self.write_field(field_name)