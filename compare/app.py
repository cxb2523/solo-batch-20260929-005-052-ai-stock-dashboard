"""Comparison app: fetch -> align -> indicators/prediction -> overlay window.

``compare_codes()`` returns per-code trend, MA/RSI/MACD and a next-day
prediction, and opens a window with the normalized overlay chart plus
per-row errors. The legacy single-stock ``create_advanced_chart(data,
symbol)`` signature stays available below.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .align import DEFAULT_MAX_GAP, gap_violations, normalize, union_align
from .data_source import QuoteDataSource


# ---------------------------------------------------------------- indicators
def moving_average(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
    signal_line = line.ewm(span=signal).mean()
    return line, signal_line, line - signal_line


def predict_next(close: pd.Series, window: int = 30) -> float:
    """Next-day close via least-squares trend over the recent window."""
    recent = close.dropna().tail(window)
    if len(recent) < 2:
        return float("nan")
    x = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent.to_numpy(dtype=float), 1)
    return float(intercept + slope * len(recent))


# --------------------------------------------------------------- comparison
def compare_codes(
    codes,
    period: str = "1y",
    source: QuoteDataSource | None = None,
    show: bool = True,
    save_path: str | None = None,
    max_gap: int = DEFAULT_MAX_GAP,
) -> dict:
    """Compare codes: returns trends, MA/RSI/MACD, predictions and errors.

    A code whose gaps exceed ``max_gap`` trading days, or whose fetch fails
    entirely, only gets an error on its own row; the window still opens.
    """
    source = source or QuoteDataSource()
    codes = [str(c).upper() for c in codes]

    data, errors = source.fetch_many(codes, period)
    close_map = {code: df["Close"] for code, df in data.items()}

    unfilled = union_align(close_map, fill=False)
    gaps = gap_violations(unfilled, max_gap) if not unfilled.empty else {}
    aligned = union_align(close_map, fill=True)
    normalized = normalize(aligned)

    per_code: dict[str, dict] = {}
    for code, df in data.items():
        close = df["Close"]
        macd_line, macd_signal, macd_hist = macd(close)
        per_code[code] = {
            "trend": normalized[code],
            "ma": moving_average(close),
            "rsi": rsi(close),
            "macd": {"line": macd_line, "signal": macd_signal, "hist": macd_hist},
            "prediction": predict_next(close),
        }

    for code, runs in gaps.items():
        detail = "; ".join(f"{pd.Timestamp(r[0]).date()}..{pd.Timestamp(r[1]).date()} ({r[2]}d)" for r in runs)
        message = f"gap > {max_gap} trading days: {detail}"
        errors[code] = f"{errors[code]}; {message}" if code in errors else message

    result = {
        "codes": per_code,
        "all_codes": codes,
        "aligned": aligned,
        "normalized": normalized,
        "errors": errors,
        "gaps": gaps,
    }
    if show or save_path:
        show_comparison_window(result, save_path=save_path, block=show)
    return result


# ------------------------------------------------------------------ window
def show_comparison_window(result: dict, save_path: str | None = None, block: bool = True):
    """Overlay normalized trends; list one status/error row per code."""
    import matplotlib

    if not block:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[3, 1.2])

    ax = fig.add_subplot(grid[0, 0])
    normalized = result["normalized"]
    for code in normalized.columns:
        ax.plot(normalized.index, normalized[code], label=code, linewidth=1.4)
    if normalized.columns.size:
        ax.legend(loc="best")
    ax.set_title("Normalized overlay (first aligned close = 1.0)")
    ax.set_ylabel("normalized close")
    ax.grid(alpha=0.3)

    panel = fig.add_subplot(grid[0, 1])
    panel.axis("off")
    panel.set_title("Per-code status")
    rows = []
    for code in result["all_codes"]:
        error = result["errors"].get(code)
        if error:
            rows.append((f"{code}: ERROR\n  {error}", "red"))
        else:
            prediction = result["codes"][code]["prediction"]
            rows.append((f"{code}: OK  next~{prediction:.2f}", "green"))
    y = 0.98
    for text, color in rows:
        panel.text(0.0, y, text, va="top", ha="left", fontsize=9,
                   family="monospace", color=color, transform=panel.transAxes)
        y -= 0.06 + 0.045 * text.count("\n")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
    if block:
        plt.show()
    return fig


# ------------------------------------------------- legacy single-stock API
def create_advanced_chart(data: pd.DataFrame, symbol: str):
    """Legacy single-stock chart (signature kept from stock_dashboard.py)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(f"{symbol} Price Action & Moving Averages", "Volume",
                        "MACD", "RSI & Stochastic"),
        row_heights=[0.5, 0.15, 0.2, 0.15],
    )
    fig.add_trace(
        go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                       low=data["Low"], close=data["Close"], name="Price"),
        row=1, col=1,
    )
    for col, name in [("SMA_20", "SMA 20"), ("SMA_50", "SMA 50"), ("SMA_200", "SMA 200")]:
        if col in data.columns and not data[col].isna().all():
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=name), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"), row=2, col=1)
    if all(c in data.columns for c in ("MACD", "MACD_signal")):
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal"), row=3, col=1)
    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"), row=4, col=1)
    fig.update_layout(title=f"{symbol} - Complete Technical Analysis Dashboard",
                      xaxis_rangeslider_visible=False, height=900, template="plotly_dark")
    return fig


# --------------------------------------------------------------------- CLI
def _demo_upstream(code: str, period: str = "1y") -> pd.DataFrame:
    """Deterministic fake quotes; BAD always fails, GAPPY has a long gap."""
    if code == "BAD":
        raise RuntimeError("simulated upstream outage")
    rng = np.random.default_rng(abs(hash(code)) % (2**32))
    days = pd.bdate_range("2026-01-05", periods=120)
    if code == "GAPPY":
        days = days.delete(range(40, 48))  # 8 missing trading days
    steps = rng.normal(0.001, 0.015, size=len(days)).cumsum()
    close = 100 * np.exp(steps)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": 1_000_000},
        index=days,
    )


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Compare stock codes in one window.")
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--period", default="1y")
    parser.add_argument("--demo", action="store_true", help="use fake quotes")
    parser.add_argument("--save", default=None, help="save the figure to a file")
    parser.add_argument("--no-show", action="store_true", help="do not open the window")
    args = parser.parse_args(argv)

    if args.demo:
        source = QuoteDataSource(upstream=_demo_upstream)
        codes = args.codes or ["AAA", "BBB", "GAPPY", "BAD"]
    else:
        source = QuoteDataSource()
        codes = args.codes or ["AAPL", "MSFT", "GOOGL"]

    return compare_codes(codes, period=args.period, source=source,
                         show=not args.no_show, save_path=args.save)


if __name__ == "__main__":
    main()

