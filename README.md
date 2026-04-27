# Quant Transformer: Financial Time-Series Return Prediction

A complete runnable PyTorch project for predicting next-day stock returns using a Transformer encoder.

## What this project does

This project builds a full quantitative research pipeline:

1. Download daily OHLCV data with `yfinance`
2. Create technical/time-series features
3. Build rolling-window sequences
4. Train a Transformer encoder to predict next-day return
5. Evaluate regression and directional accuracy
6. Run a simple long/short backtest
7. Save metrics, plots, model weights, and predictions

## Project structure

```text
quant_transformer_project/
├── requirements.txt
├── config.yaml
├── README.md
├── src/
│   ├── data.py
│   ├── features.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── backtest.py
│   └── utils.py
└── outputs/
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/train.py --config config.yaml
python src/evaluate.py --config config.yaml
python src/backtest.py --config config.yaml
```

## Default task

The default config uses ETFs/stocks:

```text
SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA
```

The model predicts next-day return from the previous 30 trading days of features.

## Output

After running, results are saved in `outputs/`:

```text
outputs/
├── best_model.pt
├── metrics.json
├── predictions.csv
├── training_curve.png
├── backtest_equity_curve.png
└── backtest_metrics.json
```

## Notes

This is an educational research template, not trading advice. The default backtest is intentionally simple and does not include transaction costs, slippage, or portfolio constraints unless you add them.


## Extra visualizations

After training, evaluating, and backtesting, run:

```bash
python src/visualize.py --config config.yaml
```

This creates:

```text
outputs/figures/
├── prediction_distribution.png
├── target_distribution.png
├── prediction_vs_target_scatter.png
├── daily_ic_curve.png
├── ic_distribution.png
├── cumulative_ic.png
├── quantile_average_returns.png
├── quantile_cumulative_returns.png
├── long_short_spread_curve.png
├── ticker_monthly_ic_heatmap.png
├── strategy_equity_curve.png
├── strategy_drawdown.png
├── rolling_sharpe_60d.png
├── strategy_return_distribution.png
├── daily_turnover.png
└── visual_report_summary.txt
```
