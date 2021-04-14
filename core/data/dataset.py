import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data, sequence_length, target="close", features=None):
        self.data = self._preprocess_data(data)
        self.features = features or list(self.data.select_dtypes("number").columns)
        self.target = target
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, index):
        index = (index + len(self)) % len(self)
        target = self.data[self.target][index : index + self.sequence_length].values
        center, scale = np.mean(target), np.std(target)
        normalized = (target - center) / (scale + 1e-8)
        encoded = normalized[np.newaxis, ...].astype(np.float32) # NN
        return {
            "x": encoded,
            "scales": (center, scale),
            "index": index,
        }

    def _preprocess_data(self, data):
        assert "time_idx" in data.columns
        return type(self).fill_missing_time_idx(data)

    @staticmethod
    def fill_missing_time_idx(data):
        time = data.time_idx
        start, end = time.min(), time.max()
        missing = start + np.nonzero(~np.isin(np.arange(start, end + 1), time))[0]
        missing_df = pd.DataFrame({"time_idx": missing})
        data = (
            data.merge(missing_df, how="outer", on="time_idx")
            .sort_values("time_idx")
            .fillna(method="ffill")
        )
        return data