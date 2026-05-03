from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelSchema:
    """Canonical column names used by the pipeline."""

    # Key columns
    date_col: str = "date"
    sid_col: str = "sid"

    # PIT visibility column
    available_at_col: str = "available_at"

    # Market data columns
    close_col: str = "close"
    volume_col: str = "volume"


SCHEMA = PanelSchema()


def build_trading_calendar(start: str, end: str, freq: str = "B") -> pd.DataFrame:
    """Create a simple business-day trading calendar.

    In production, replace with exchange calendars (e.g. NYSE/SSE) with holidays.
    """
    trade_dates = pd.date_range(start=start, end=end, freq=freq)
    cal = pd.DataFrame({"trade_date": trade_dates})
    cal["is_open"] = True
    return cal


def make_panel_skeleton(trading_calendar: pd.DataFrame, universe: Iterable[str]) -> pd.DataFrame:
    """Build date x stock panel skeleton."""
    trade_dates = pd.to_datetime(trading_calendar.loc[trading_calendar["is_open"], "trade_date"]).sort_values().unique()
    sids = sorted(set(universe))
    idx = pd.MultiIndex.from_product([trade_dates, sids], names=[SCHEMA.date_col, SCHEMA.sid_col])
    return pd.DataFrame(index=idx).reset_index()


def _prep_cutoff_ts(panel_df: pd.DataFrame, cutoff_time: str) -> pd.Series:
    date_str = pd.to_datetime(panel_df[SCHEMA.date_col]).dt.strftime("%Y-%m-%d")
    return pd.to_datetime(date_str + f" {cutoff_time}")


def pit_asof_join(
    panel_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    *,
    entity_col: str,
    value_col: str,
    name_col: str,
    available_at_col: str = SCHEMA.available_at_col,
    out_prefix: str,
    cutoff_time: str = "15:00:00",
) -> pd.DataFrame:
    """Point-in-time as-of join.

    For each (date, sid), only use records whose available_at <= date@cutoff_time.
    If multiple are visible, keep the latest available record.
    """
    required = {SCHEMA.date_col, SCHEMA.sid_col}
    if not required.issubset(panel_df.columns):
        raise ValueError(f"panel_df must include {required}")

    required_pit = {entity_col, name_col, value_col, available_at_col}
    if not required_pit.issubset(pit_df.columns):
        raise ValueError(f"pit_df must include {required_pit}")

    df = panel_df.copy()
    df["cutoff_ts"] = _prep_cutoff_ts(df, cutoff_time)

    result = df[[SCHEMA.date_col, SCHEMA.sid_col, "cutoff_ts"]].copy()

    for feat_name, g in pit_df.groupby(name_col, dropna=False):
        g = g.rename(columns={entity_col: SCHEMA.sid_col, value_col: "_value"}).copy()
        g[available_at_col] = pd.to_datetime(g[available_at_col])
        g = g.sort_values([SCHEMA.sid_col, available_at_col])

        left = df[[SCHEMA.sid_col, "cutoff_ts"]].sort_values([SCHEMA.sid_col, "cutoff_ts"])
        right = g[[SCHEMA.sid_col, available_at_col, "_value"]].sort_values([SCHEMA.sid_col, available_at_col])

        merged = pd.merge_asof(
            left,
            right,
            left_on="cutoff_ts",
            right_on=available_at_col,
            by=SCHEMA.sid_col,
            direction="backward",
            allow_exact_matches=True,
        )

        colname = f"{out_prefix}_{feat_name}"
        result[colname] = merged["_value"].values

    out = panel_df.merge(result.drop(columns=["cutoff_ts"]), on=[SCHEMA.date_col, SCHEMA.sid_col], how="left")
    return out


def add_forward_return_label(panel_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create forward return label without leakage."""
    if SCHEMA.close_col not in panel_df.columns:
        raise ValueError(f"panel_df must include {SCHEMA.close_col}")

    out = panel_df.sort_values([SCHEMA.sid_col, SCHEMA.date_col]).copy()
    out[f"ret_fwd_{horizon}d"] = out.groupby(SCHEMA.sid_col)[SCHEMA.close_col].shift(-horizon) / out[SCHEMA.close_col] - 1.0
    return out


def cross_sectional_winsorize_zscore(panel_df: pd.DataFrame, feature_cols: list[str], lower=0.01, upper=0.99) -> pd.DataFrame:
    """Daily cross-sectional winsorization + z-score normalization."""
    out = panel_df.copy()

    def _winsorize(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)

    for col in feature_cols:
        out[col] = out.groupby(SCHEMA.date_col)[col].transform(_winsorize)
        mu = out.groupby(SCHEMA.date_col)[col].transform("mean")
        sd = out.groupby(SCHEMA.date_col)[col].transform("std").replace(0, np.nan)
        out[col] = (out[col] - mu) / sd

    return out


if __name__ == "__main__":
    # Minimal runnable example
    cal = build_trading_calendar("2025-01-01", "2025-01-31")
    universe = ["AAPL", "MSFT"]
    panel = make_panel_skeleton(cal, universe)

    # mock market data
    rng = np.random.default_rng(7)
    px = panel[["date", "sid"]].copy()
    px["close"] = 100 + rng.normal(0, 1, len(px)).cumsum()
    px["volume"] = rng.integers(1_000_000, 2_000_000, len(px))
    panel = panel.merge(px, on=["date", "sid"], how="left")

    # mock fundamental PIT data
    fund = pd.DataFrame(
        {
            "sid": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "field_name": ["net_income", "roe", "net_income", "roe"],
            "field_value": [10.0, 0.20, 9.0, 0.18],
            "available_at": pd.to_datetime([
                "2025-01-10 20:00:00",
                "2025-01-10 20:00:00",
                "2025-01-12 20:00:00",
                "2025-01-12 20:00:00",
            ]),
        }
    )

    panel = pit_asof_join(
        panel,
        fund,
        entity_col="sid",
        name_col="field_name",
        value_col="field_value",
        out_prefix="fin",
        cutoff_time="15:00:00",
    )

    panel = add_forward_return_label(panel, horizon=5)

    feature_cols = [c for c in panel.columns if c.startswith("fin_")]
    panel = cross_sectional_winsorize_zscore(panel, feature_cols)

    panel = panel.set_index(["date", "sid"]).sort_index()
    print(panel.head(10))
