"""Multi-code compare package.

* :mod:`compare.data_source` -- concurrent, rate-limited, cached quote fetch.
* :mod:`compare.align` -- pure union alignment, first-close normalization, gaps.
* :mod:`compare.analytics` -- MA/RSI/MACD and next-day forecasts.
"""
from .data_source import (
    DEFAULT_TTL_SECONDS,
    DataSourceError,
    FetchResult,
    QuoteFetcher,
    yfinance_fetch,
)
from .align import (
    GAP_LIMIT_TRADING_DAYS,
    Gap,
    align_and_normalize,
    align_closes,
    find_gaps,
    normalize_first_close,
    union_trading_index,
)
from .analytics import all_indicators, indicators_for, predict_all, predict_next_close

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "DataSourceError",
    "FetchResult",
    "QuoteFetcher",
    "yfinance_fetch",
    "GAP_LIMIT_TRADING_DAYS",
    "Gap",
    "align_and_normalize",
    "align_closes",
    "find_gaps",
    "normalize_first_close",
    "union_trading_index",
    "all_indicators",
    "indicators_for",
    "predict_all",
    "predict_next_close",
]
