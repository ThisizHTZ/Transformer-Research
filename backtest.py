import argparse
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import save_json


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    return float(dd.min())


def annualized_sharpe(daily_returns, trading_days=252):
    std = daily_returns.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(trading_days) * daily_returns.mean() / std)


def run_backtest(pred_df, long_q=0.8, short_q=0.2, transaction_cost=0.0005):
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    prev_positions = {}

    for date, g in df.groupby("date"):
        g = g.copy()

        long_cut = g["prediction"].quantile(long_q)
        short_cut = g["prediction"].quantile(short_q)

        g["position"] = 0.0
        g.loc[g["prediction"] >= long_cut, "position"] = 1.0
        g.loc[g["prediction"] <= short_cut, "position"] = -1.0

        # Equal-weight gross exposure normalized to 1.
        gross = g["position"].abs().sum()
        if gross > 0:
            g["weight"] = g["position"] / gross
        else:
            g["weight"] = 0.0

        current_positions = dict(zip(g["ticker"], g["weight"]))

        # Simple turnover cost.
        tickers = set(prev_positions.keys()).union(current_positions.keys())
        turnover = sum(abs(current_positions.get(t, 0.0) - prev_positions.get(t, 0.0)) for t in tickers)
        cost = transaction_cost * turnover

        daily_ret = float((g["weight"] * g["target"]).sum() - cost)

        rows.append({
            "date": date,
            "strategy_return": daily_ret,
            "turnover": turnover,
            "cost": cost,
            "n_assets": len(g)
        })

        prev_positions = current_positions

    result = pd.DataFrame(rows).sort_values("date")
    result["equity"] = (1 + result["strategy_return"]).cumprod()

    return result


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    pred_path = os.path.join(output_dir, "predictions.csv")

    if not os.path.exists(pred_path):
        raise FileNotFoundError("Run evaluate.py first to create outputs/predictions.csv")

    pred_df = pd.read_csv(pred_path)

    bt = run_backtest(
        pred_df,
        long_q=cfg["backtest"]["long_quantile"],
        short_q=cfg["backtest"]["short_quantile"],
        transaction_cost=cfg["backtest"]["transaction_cost"]
    )

    metrics = {
        "total_return": float(bt["equity"].iloc[-1] - 1),
        "annualized_sharpe": annualized_sharpe(bt["strategy_return"]),
        "max_drawdown": max_drawdown(bt["equity"].values),
        "mean_daily_return": float(bt["strategy_return"].mean()),
        "daily_volatility": float(bt["strategy_return"].std()),
        "average_turnover": float(bt["turnover"].mean())
    }

    bt.to_csv(os.path.join(output_dir, "backtest_daily.csv"), index=False)
    save_json(metrics, os.path.join(output_dir, "backtest_metrics.json"))

    plt.figure(figsize=(9, 5))
    plt.plot(bt["date"], bt["equity"])
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Transformer Long/Short Backtest")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_equity_curve.png"), dpi=150)
    plt.close()

    print(metrics)
    print("Backtest saved to outputs/backtest_daily.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
