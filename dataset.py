import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class SequenceDataset(Dataset):
    def __init__(self, X, y, meta):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.meta = meta.reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(df, feature_cols, sequence_length):
    X, y, meta = [], [], []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)

        values = g[feature_cols].values
        targets = g["target"].values

        for i in range(sequence_length, len(g)):
            X.append(values[i - sequence_length:i])
            y.append(targets[i])
            meta.append({
                "date": g.loc[i, "date"],
                "ticker": ticker,
                "target": targets[i]
            })

    return np.array(X), np.array(y), pd.DataFrame(meta)


def time_split(X, y, meta, train_ratio=0.70, valid_ratio=0.15):
    meta = meta.copy()
    meta["date"] = pd.to_datetime(meta["date"])
    order = meta.sort_values("date").index.values

    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)

    n = len(X)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    return (
        X[:train_end], y[:train_end], meta.iloc[:train_end],
        X[train_end:valid_end], y[train_end:valid_end], meta.iloc[train_end:valid_end],
        X[valid_end:], y[valid_end:], meta.iloc[valid_end:]
    )


def scale_by_train(X_train, X_valid, X_test):
    n_train, seq_len, n_features = X_train.shape

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def transform(X):
        n, s, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, s, f)

    return transform(X_train), transform(X_valid), transform(X_test), scaler
