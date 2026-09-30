"""Technical indicators and next-day forecasts for aligned close prices.

Pure-ish functions over pandas objects (RandomForest training is deterministic
via ``random_state``).  Indicators and forecasts are computed on *raw* aligned
closes; normalization happens elsewhere and never enters the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RF_MIN_ROWS = 60


def moving_averages(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MA20": close.rolling(window=20, min_periods=1).mean(),
            "MA50": close.rolling(window=50, min_periods=1).mean(),
        },
        index=close.index,
    )


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Flat streak (zero loss) is fully overbought; zero gain is oversold.
    out = out.where(~(loss == 0) | (gain == 0), 100.0)
    out = out.where(~(gain == 0) | (loss != 0), 0.0)
    return out


def macd(close: pd.Series) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame(
        {"MACD": line, "MACD_signal": signal, "MACD_hist": line - signal},
        index=close.index,
    )


def indicators_for(close: pd.Series) -> pd.DataFrame:
    """MA/RSI/MACD table for one code's aligned raw close series."""
    close = close.dropna()
    if close.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"Close": close})
    df = pd.concat([df, moving_averages(close), macd(close)], axis=1)
    df["RSI"] = rsi(close)
    return df


def all_indicators(aligned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {code: indicators_for(aligned[code]) for code in aligned.columns}


def _drift_forecast(close: pd.Series) -> float:
    recent = close.dropna().tail(10)
    step = recent.diff().dropna().mean()
    if pd.isna(step):
        step = 0.0
    return float(close.dropna().iloc[-1] + step)


def _rf_forecast(close: pd.Series) -> float:
    from sklearn.ensemble import RandomForestRegressor

    s = close.dropna()
    df = pd.DataFrame(index=s.index)
    df["y"] = s
    for lag in (1, 2, 3, 5):
        df[f"lag{lag}"] = s.shift(lag)
    for win in (5, 10, 20):
        df[f"ma{win}"] = s.rolling(win, min_periods=1).mean()
    df["ret1"] = s.pct_change()
    df["rsi"] = rsi(s)
    m = macd(s)
    df["macd"] = m["MACD"]
    df["macd_hist"] = m["MACD_hist"]
    train = df.dropna()
    if len(train) < 40:
        return _drift_forecast(close)
    feature_cols = [c for c in df.columns if c != "y"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[feature_cols], train["y"])
    last = df[feature_cols].iloc[[-1]].ffill().bfill()
    return float(model.predict(last)[0])


def predict_next_close(close: pd.Series, *, use_rf: bool = True) -> float:
    """Next-trading-day close forecast in the code's raw price units."""
    s = close.dropna()
    if s.empty:
        return float("nan")
    if len(s) >= RF_MIN_ROWS and use_rf:
        return _rf_forecast(s)
    return _drift_forecast(s)


def predict_all(aligned: pd.DataFrame, *, use_rf: bool = True) -> dict[str, float]:
    return {
        code: predict_next_close(aligned[code], use_rf=use_rf)
        for code in aligned.columns
        if not aligned[code].dropna().empty
    }
