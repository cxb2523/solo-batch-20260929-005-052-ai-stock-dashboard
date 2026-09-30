"""Compare window for multiple stock codes.

Run:

    python app.py                      # real yfinance fetch, default basket
    python app.py AAPL MSFT NVDA       # custom basket
    python app.py --demo               # injected fake quotes (offline smoke test)

The legacy single-stock plotting entry point is still importable from here:
``from app import create_advanced_chart`` keeps its old ``(data, symbol)``
signature.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from compare.data_source import QuoteFetcher
from compare.align import align_and_normalize
from compare.analytics import all_indicators, predict_all
from compare.charts import create_advanced_chart, create_compare_chart

DEFAULT_CODES = ["AAPL", "MSFT", "NVDA", "TSLA", "ZZBADTICKER"]
DEFAULT_PERIOD = "1y"

# Backwards-compatible re-export: old callers import this from app.py.
__all__ = ["compare_codes", "show_compare_window", "main", "create_advanced_chart"]


def compare_codes(
    codes,
    period: str = DEFAULT_PERIOD,
    *,
    fetcher: "QuoteFetcher | None" = None,
    use_rf: bool = True,
):
    """Fetch, align, normalize, score and forecast each code.

    Returns a dict with aligned closes, normalized trends, per-code indicator
    tables (MA/RSI/MACD), next-day forecasts and per-row errors.  A failed or
    gappy code only contributes an error row; the rest of the window is still
    produced.
    """
    requested = list(codes)
    own_fetcher = fetcher is None
    fetcher = fetcher or QuoteFetcher()

    try:
        results = fetcher.fetch_many(requested, period=period)
        series_map: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        cached_codes: dict[str, bool] = {}

        # Whole-code failure: one row, never aborts the other codes.
        for code in requested:
            result = results[code]
            cached_codes[code] = result.cached
            if result.ok:
                series_map[code] = result.data
            else:
                detail = result.error or "fetch failed"
                errors[code] = f"{code}: {detail}"

        # Union align -> gap detection -> normalize (order is intentional).
        union, aligned, normalized, gaps = align_and_normalize(series_map)

        # Gap is a row-level warning, not a drop: the code stays on the chart.
        for code, gap_list in gaps.items():
            for gap in gap_list:
                errors.setdefault(code, f"{code}: {gap.message()}")

        indicators = all_indicators(aligned)
        predictions = predict_all(aligned, use_rf=use_rf)

        return {
            "codes": requested,
            "period": period,
            "index": union,
            "aligned": aligned,
            "trends": normalized,
            "indicators": indicators,
            "predictions": predictions,
            "errors": dict(sorted(errors.items())),
            "cached": cached_codes,
        }
    finally:
        if own_fetcher:
            fetcher.close()


def show_compare_window(result, *, auto_open: bool = True) -> str:
    """Render the overlay + per-row error figure and open it in a browser."""
    fig = create_compare_chart(result["trends"], result["errors"])

    forecasts = result["predictions"]
    if forecasts:
        pred_lines = ["Next-trading-day forecasts (raw price units):"]
        for code in sorted(forecasts):
            pred_lines.append(f"  {code}: {forecasts[code]:.2f}")
        fig.add_annotation(
            text="<br>".join(pred_lines),
            xref="paper", yref="paper", x=0.01, y=0.99,
            showarrow=False, align="left",
            bgcolor="rgba(0,0,0,0.55)", bordercolor="#555",
            font=dict(size=11, color="#e8e8e8"),
        )

    if result["errors"]:
        print("Per-code errors:", file=sys.stderr)
        for msg in result["errors"].values():
            print(f"  {msg}", file=sys.stderr)

    if auto_open:
        fig.show()

    return fig


def _demo_quotes(code: str, period: str = "1y") -> pd.DataFrame:
    """Deterministic fake upstream used by ``python app.py --demo``."""
    import numpy as np

    if code == "ZZBADTICKER":
        raise RuntimeError("simulated whole-code failure")

    business_days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=180)
    seed = sum(ord(ch) for ch in code)
    rng = np.random.default_rng(seed)

    base = 50.0 + (seed % 200)
    drift = 0.0006 + (seed % 7) * 0.0003
    closes = base * (1 + drift) ** np.arange(len(business_days))
    closes = closes * (1 + rng.normal(0, 0.012, len(business_days))).cumprod()

    if code == "GAPPY":
        keep = list(range(0, 60)) + list(range(70, len(business_days)))
        business_days = business_days[keep]
        closes = closes[keep]

    opens = closes * (1 + rng.normal(0, 0.004, len(closes)))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.01, len(closes)))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.01, len(closes)))
    volume = rng.integers(1_000_000, 10_000_000, len(closes))

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volume,
        },
        index=business_days,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multi-code stock compare window")
    parser.add_argument("codes", nargs="*", help="stock codes (default: demo basket)")
    parser.add_argument("--period", default=DEFAULT_PERIOD,
                        help="yfinance period, e.g. 6mo/1y/2y (default 1y)")
    parser.add_argument("--demo", action="store_true",
                        help="use injected fake quotes (offline, includes a gap and a failure)")
    parser.add_argument("--no-open", action="store_true",
                        help="build figure without opening a browser window")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.demo:
        codes = args.codes or ["AAPL", "MSFT", "NVDA", "GAPPY", "ZZBADTICKER"]
        fetcher = QuoteFetcher(_demo_quotes, backoff_base=0.01)
    else:
        codes = args.codes or DEFAULT_CODES
        fetcher = QuoteFetcher()

    print(f"Fetching {len(codes)} code(s): {', '.join(codes)} (period={args.period})")
    result = compare_codes(codes, period=args.period, fetcher=fetcher)

    ok = [c for c in codes if c not in result["errors"]]
    print(f"Plotted {len(ok)}/{len(codes)} code(s); {len(result['errors'])} error row(s)")
    show_compare_window(result, auto_open=not args.no_open)
    fetcher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
