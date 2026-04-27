import argparse
import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_corr(x, y):
    if len(x) < 2:
        return np.nan
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return x.corr(y)


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_prediction_distribution(pred_df, fig_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["prediction"], bins=50, alpha=0.8)
    plt.xlabel("Predicted Next-Day Return")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "prediction_distribution.png"), dpi=150)
    plt.close()


def plot_target_distribution(pred_df, fig_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["target"], bins=50, alpha=0.8)
    plt.xlabel("Actual Next-Day Return")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actual Next-Day Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "target_distribution.png"), dpi=150)
    plt.close()


def plot_prediction_vs_target(pred_df, fig_dir):
    sample = pred_df.copy()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    plt.figure(figsize=(7, 6))
    plt.scatter(sample["prediction"], sample["target"], alpha=0.35, s=12)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Prediction")
    plt.ylabel("Actual Target")
    plt.title("Prediction vs Actual Return")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "prediction_vs_target_scatter.png"), dpi=150)
    plt.close()


def plot_daily_ic(pred_df, fig_dir):
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily_ic = (
        df.groupby("date")
        .apply(lambda x: safe_corr(x["prediction"], x["target"]))
        .reset_index(name="ic")
        .dropna()
    )

    daily_ic["rolling_ic_20d"] = daily_ic["ic"].rolling(20).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_ic["date"], daily_ic["ic"], alpha=0.45, label="Daily IC")
    plt.plot(daily_ic["date"], daily_ic["rolling_ic_20d"], linewidth=2, label="20D Rolling IC")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Information Coefficient")
    plt.title("Daily IC and 20-Day Rolling IC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "daily_ic_curve.png"), dpi=150)
    plt.close()

    daily_ic.to_csv(os.path.join(fig_dir, "daily_ic.csv"), index=False)
    return daily_ic


def plot_ic_distribution(daily_ic, fig_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(daily_ic["ic"].dropna(), bins=40, alpha=0.8)
    plt.axvline(daily_ic["ic"].mean(), linewidth=2, label=f"Mean IC = {daily_ic['ic'].mean():.4f}")
    plt.xlabel("Daily IC")
    plt.ylabel("Frequency")
    plt.title("IC Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ic_distribution.png"), dpi=150)
    plt.close()


def plot_cumulative_ic(daily_ic, fig_dir):
    tmp = daily_ic.dropna().copy()
    tmp["cumulative_ic"] = tmp["ic"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["date"], tmp["cumulative_ic"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Cumulative IC")
    plt.title("Cumulative Information Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cumulative_ic.png"), dpi=150)
    plt.close()


def plot_quantile_returns(pred_df, fig_dir, n_quantiles=5):
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def assign_quantile(x):
        try:
            return pd.qcut(x["prediction"], q=n_quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    df["quantile"] = df.groupby("date", group_keys=False).apply(assign_quantile)

    quantile_daily = (
        df.dropna(subset=["quantile"])
        .groupby(["date", "quantile"])["target"]
        .mean()
        .reset_index()
    )

    quantile_mean = (
        quantile_daily
        .groupby("quantile")["target"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(quantile_mean["quantile"].astype(str), quantile_mean["target"])
    plt.xlabel("Prediction Quantile")
    plt.ylabel("Average Next-Day Return")
    plt.title("Average Forward Return by Prediction Quantile")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "quantile_average_returns.png"), dpi=150)
    plt.close()

    pivot = quantile_daily.pivot(index="date", columns="quantile", values="target")
    cum = (1 + pivot.fillna(0)).cumprod()

    plt.figure(figsize=(10, 5))
    for col in cum.columns:
        plt.plot(cum.index, cum[col], label=f"Q{int(col)}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Return by Prediction Quantile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "quantile_cumulative_returns.png"), dpi=150)
    plt.close()

    quantile_daily.to_csv(os.path.join(fig_dir, "quantile_daily_returns.csv"), index=False)
    return quantile_daily


def plot_long_short_spread(pred_df, fig_dir, long_q=0.8, short_q=0.2):
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for date, g in df.groupby("date"):
        long_cut = g["prediction"].quantile(long_q)
        short_cut = g["prediction"].quantile(short_q)

        long_ret = g.loc[g["prediction"] >= long_cut, "target"].mean()
        short_ret = g.loc[g["prediction"] <= short_cut, "target"].mean()
        spread = long_ret - short_ret

        rows.append({
            "date": date,
            "long_return": long_ret,
            "short_return": short_ret,
            "long_short_spread": spread
        })

    ls = pd.DataFrame(rows).sort_values("date")
    ls["cumulative_long"] = (1 + ls["long_return"].fillna(0)).cumprod()
    ls["cumulative_short"] = (1 + ls["short_return"].fillna(0)).cumprod()
    ls["cumulative_spread"] = (1 + ls["long_short_spread"].fillna(0)).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(ls["date"], ls["cumulative_long"], label="Long Basket")
    plt.plot(ls["date"], ls["cumulative_short"], label="Short Basket")
    plt.plot(ls["date"], ls["cumulative_spread"], label="Long-Short Spread")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Long Basket vs Short Basket vs Long-Short Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "long_short_spread_curve.png"), dpi=150)
    plt.close()

    ls.to_csv(os.path.join(fig_dir, "long_short_spread.csv"), index=False)
    return ls


def plot_ticker_ic_heatmap(pred_df, fig_dir):
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    rows = []
    for (month, ticker), g in df.groupby(["month", "ticker"]):
        ic = safe_corr(g["prediction"], g["target"])
        rows.append({"month": month, "ticker": ticker, "ic": ic})

    ic_df = pd.DataFrame(rows).dropna()
    if ic_df.empty:
        return None

    pivot = ic_df.pivot(index="ticker", columns="month", values="ic")

    plt.figure(figsize=(12, max(4, 0.4 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Monthly IC")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title("Monthly IC Heatmap by Ticker")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ticker_monthly_ic_heatmap.png"), dpi=150)
    plt.close()

    ic_df.to_csv(os.path.join(fig_dir, "ticker_monthly_ic.csv"), index=False)
    return ic_df


def plot_backtest_curves(output_dir, fig_dir):
    path = os.path.join(output_dir, "backtest_daily.csv")
    if not os.path.exists(path):
        print("[WARN] outputs/backtest_daily.csv not found. Run backtest.py first.")
        return None

    bt = pd.read_csv(path)
    bt["date"] = pd.to_datetime(bt["date"])

    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1
    bt["rolling_sharpe_60d"] = (
        bt["strategy_return"].rolling(60).mean()
        / bt["strategy_return"].rolling(60).std()
        * np.sqrt(252)
    )

    plt.figure(figsize=(10, 5))
    plt.plot(bt["date"], bt["equity"])
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Strategy Equity Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "strategy_equity_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(bt["date"], bt["drawdown"])
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Strategy Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "strategy_drawdown.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(bt["date"], bt["rolling_sharpe_60d"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("60D Rolling Sharpe")
    plt.title("60-Day Rolling Sharpe")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rolling_sharpe_60d.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(bt["strategy_return"], bins=50, alpha=0.8)
    plt.xlabel("Daily Strategy Return")
    plt.ylabel("Frequency")
    plt.title("Distribution of Daily Strategy Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "strategy_return_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(bt["date"], bt["turnover"])
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.title("Daily Portfolio Turnover")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "daily_turnover.png"), dpi=150)
    plt.close()

    bt.to_csv(os.path.join(fig_dir, "backtest_daily_with_extra_metrics.csv"), index=False)
    return bt


def create_summary_report(output_dir, fig_dir, daily_ic):
    metrics = load_json(os.path.join(output_dir, "metrics.json"))
    bt_metrics = load_json(os.path.join(output_dir, "backtest_metrics.json"))

    report_path = os.path.join(fig_dir, "visual_report_summary.txt")

    mean_ic = np.nan
    ic_ir = np.nan
    if daily_ic is not None and len(daily_ic) > 0:
        mean_ic = daily_ic["ic"].mean()
        ic_ir = daily_ic["ic"].mean() / daily_ic["ic"].std() if daily_ic["ic"].std() != 0 else np.nan

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Quant Transformer Visual Report Summary\n")
        f.write("=" * 45 + "\n\n")

        f.write("[Prediction Metrics]\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Backtest Metrics]\n")
        for k, v in bt_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[IC Metrics]\n")
        f.write(f"mean_daily_ic: {mean_ic}\n")
        f.write(f"ic_information_ratio: {ic_ir}\n")

        f.write("\n[Generated Figures]\n")
        for name in sorted(os.listdir(fig_dir)):
            if name.endswith(".png"):
                f.write(f"- {name}\n")

    return report_path


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    pred_path = os.path.join(output_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError("outputs/predictions.csv not found. Run evaluate.py first.")

    pred_df = pd.read_csv(pred_path)

    print("[1/9] Plotting prediction distribution...")
    plot_prediction_distribution(pred_df, fig_dir)

    print("[2/9] Plotting target distribution...")
    plot_target_distribution(pred_df, fig_dir)

    print("[3/9] Plotting prediction vs target scatter...")
    plot_prediction_vs_target(pred_df, fig_dir)

    print("[4/9] Plotting daily IC...")
    daily_ic = plot_daily_ic(pred_df, fig_dir)

    print("[5/9] Plotting IC distribution and cumulative IC...")
    plot_ic_distribution(daily_ic, fig_dir)
    plot_cumulative_ic(daily_ic, fig_dir)

    print("[6/9] Plotting quantile returns...")
    plot_quantile_returns(pred_df, fig_dir, n_quantiles=5)

    print("[7/9] Plotting long-short spread...")
    plot_long_short_spread(
        pred_df,
        fig_dir,
        long_q=cfg["backtest"]["long_quantile"],
        short_q=cfg["backtest"]["short_quantile"]
    )

    print("[8/9] Plotting ticker monthly IC heatmap...")
    plot_ticker_ic_heatmap(pred_df, fig_dir)

    print("[9/9] Plotting backtest diagnostics...")
    plot_backtest_curves(output_dir, fig_dir)

    report_path = create_summary_report(output_dir, fig_dir, daily_ic)

    print(f"Done. Figures saved to: {fig_dir}")
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
