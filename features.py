import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    all_frames = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        g["ret_1"] = g["Close"].pct_change()
        g["ret_5"] = g["Close"].pct_change(5)
        g["ret_10"] = g["Close"].pct_change(10)

        g["log_volume"] = np.log1p(g["Volume"])
        g["volume_chg"] = g["Volume"].pct_change()

        g["hl_spread"] = (g["High"] - g["Low"]) / g["Close"]
        g["oc_return"] = (g["Close"] - g["Open"]) / g["Open"]

        g["ma_5"] = g["Close"].rolling(5).mean() / g["Close"] - 1
        g["ma_10"] = g["Close"].rolling(10).mean() / g["Close"] - 1
        g["ma_20"] = g["Close"].rolling(20).mean() / g["Close"] - 1

        g["vol_5"] = g["ret_1"].rolling(5).std()
        g["vol_10"] = g["ret_1"].rolling(10).std()
        g["vol_20"] = g["ret_1"].rolling(20).std()

        # Target: next-day return.
        g["target"] = g["ret_1"].shift(-1)

        all_frames.append(g)

    out = pd.concat(all_frames, axis=0, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    return out


FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10",
    "log_volume", "volume_chg",
    "hl_spread", "oc_return",
    "ma_5", "ma_10", "ma_20",
    "vol_5", "vol_10", "vol_20"
]
