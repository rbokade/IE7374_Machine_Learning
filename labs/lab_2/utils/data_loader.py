import numpy as np
import pandas as pd


class DataLoader:
    _data = None

    def __init__(self, data_dir, seed=42):
        self.seed = seed
        self._data_dir = data_dir

    def load_data(self):
        self._data = pd.read_csv(self._data_dir)

    def split_dataset_into_train_val_test(
            self, train_prop=0.6, val_prop=0.2, test_prop=0.2
    ):
        assert self._data is not None, "No data loaded."
        indices_or_sections = [
            int(train_prop * len(self._data)),
            int((train_prop + val_prop) * len(self._data))
        ]
        train, val, test = np.split(
            self._data.sample(frac=1, random_state=42), indices_or_sections
        )
        return train, val, test

    def split_dataset_into_train_test(
            self, train_prop=0.7, test_prop=0.3
    ):
        assert self._data is not None, "No data loaded."
        indices_or_sections = [
            int(train_prop * len(self._data)),
        ]
        train, test = np.split(
            self._data.sample(frac=1, random_state=42), indices_or_sections
        )
        return train, test

    def normalize_data(self, data, means=None, stds=None):
        if means is None and stds is None:
            means = data.mean()
            stds = data.std()
        normalized_data = (data - means) / stds
        return normalized_data, means, stds

    @property
    def data(self):
        return self._data

