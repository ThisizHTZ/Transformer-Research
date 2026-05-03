import argparse
import os

import numpy as np
import pandas as pd
import yaml

from utils import save_json


def _normalize_weights(w: pd.Series) -> pd.Series:
    s = w.sum()
    if s <= 0:
        return pd.Series(np.repeat(1 / len(w), len(w)), index=w.index)
    return w / s


def equal_weight_weights(returns_window: pd.DataFrame) -> pd.Series:
    n = returns_window.shape[1]
    return pd.Series(np.repeat(1 / n, n), index=returns_window.columns)


def inverse_vol_weights(returns_window: pd.DataFrame, eps: float = 1e-8) -> pd.Series:
    vol = returns_window.std().replace(0, np.nan)
    inv = 1.0 / (vol + eps)
    inv = inv.fillna(0.0)
    return _normalize_weights(inv)


def mean_variance_weights(returns_window: pd.DataFrame, l2: float = 1e-4) -> pd.Series:
    mu = returns_window.mean().values
    cov = returns_window.cov().values
    cov = cov + np.eye(cov.shape[0]) * l2
    try:
        raw = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        return equal_weight_weights(returns_window)

    raw = np.clip(raw, 0, None)
    w = pd.Series(raw, index=returns_window.columns)
    return _normalize_weights(w)


def choose_weights(returns_window: pd.DataFrame, method: str) -> pd.Series:
    if method == "equal":
        return equal_weight_weights(returns_window)
    if method == "inv_vol":
        return inverse_vol_weights(returns_window)
    if method == "mean_var":
        return mean_variance_weights(returns_window)
    raise ValueError(f"Unknown method: {method}")


def run_fof_backtest(
    fund_returns: pd.DataFrame,
    rebalance_freq: str = "M",
    lookback_days: int = 126,
    weight_method: str = "inv_vol",
    transaction_cost: float = 0.0005,
):
    df = fund_returns.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("fund_returns index must be DatetimeIndex")

    rebal_dates = set(df.resample(rebalance_freq).last().index)
    current_w = pd.Series(np.repeat(1 / df.shape[1], df.shape[1]), index=df.columns)

    records = []
    weights_records = []

    for i, date in enumerate(df.index):
        if date in rebal_dates and i >= lookback_days:
            hist = df.iloc[i - lookback_days:i]
            target_w = choose_weights(hist, weight_method)
            turnover = float((target_w - current_w).abs().sum())
            cost = turnover * transaction_cost
            current_w = target_w
        else:
            turnover = 0.0
            cost = 0.0

        day_ret = float((current_w * df.loc[date]).sum() - cost)
        records.append(
            {
                "date": date,
                "fof_return": day_ret,
                "turnover": turnover,
                "cost": cost,
            }
        )

        w_row = {"date": date}
        w_row.update({f"w_{k}": float(v) for k, v in current_w.items()})
        weights_records.append(w_row)

    out = pd.DataFrame(records).sort_values("date")
    out["equity"] = (1 + out["fof_return"]).cumprod()

    wdf = pd.DataFrame(weights_records).sort_values("date")
    return out, wdf


def summarize(bt: pd.DataFrame, trading_days: int = 252) -> dict:
    daily = bt["fof_return"]
    vol = daily.std()
    sharpe = 0.0 if vol == 0 else float(np.sqrt(trading_days) * daily.mean() / vol)
    peak = np.maximum.accumulate(bt["equity"].values)
    mdd = float((bt["equity"].values / peak - 1).min())

    return {
        "total_return": float(bt["equity"].iloc[-1] - 1),
        "annualized_sharpe": sharpe,
        "max_drawdown": mdd,
        "daily_volatility": float(vol),
        "mean_daily_return": float(daily.mean()),
        "average_turnover": float(bt["turnover"].mean()),
    }


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    fof_cfg = cfg.get("fof", {})
    returns_path = fof_cfg.get("fund_returns_path", os.path.join(output_dir, "fund_returns.csv"))

    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Missing fund returns file: {returns_path}")

    fund_ret = pd.read_csv(returns_path)
    if "date" not in fund_ret.columns:
        raise ValueError("fund_returns.csv must include 'date' column")

    fund_ret["date"] = pd.to_datetime(fund_ret["date"])
    fund_ret = fund_ret.set_index("date").sort_index()

    bt, weights = run_fof_backtest(
        fund_returns=fund_ret,
        rebalance_freq=fof_cfg.get("rebalance_freq", "M"),
        lookback_days=int(fof_cfg.get("lookback_days", 126)),
        weight_method=fof_cfg.get("weight_method", "inv_vol"),
        transaction_cost=float(fof_cfg.get("transaction_cost", 0.0005)),
    )

    metrics = summarize(bt)

    bt.to_csv(os.path.join(output_dir, "fof_backtest_daily.csv"), index=False)
    weights.to_csv(os.path.join(output_dir, "fof_weights_daily.csv"), index=False)
    save_json(metrics, os.path.join(output_dir, "fof_metrics.json"))

    print(metrics)
    print("FOF backtest saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
