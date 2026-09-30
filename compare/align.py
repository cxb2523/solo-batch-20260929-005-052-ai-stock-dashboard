"""Pure alignment / normalization / gap functions.

Everything here is a pure function over pandas objects: no network, no cache,
no module state.  Normalization deliberately happens *after* alignment, so the
first-close base is read off the union-aligned series and never cached.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# A run of missing union trading days strictly longer than this, inside a
# code's own trading span, is reported on that code's row.
GAP_LIMIT_TRADING_DAYS = 5


@dataclass(frozen=True)
class Gap:
    """One internal hole in one code's series."""

    code: str
    start: pd.Timestamp
    end: pd.Timestamp
    length: int

    def message(self) -> str:
        return (
            f"{self.length} consecutive missing trading days "
            f"({self.start.date()} -> {self.end.date()}), > "
            f"{GAP_LIMIT_TRADING_DAYS}; forward-filled on the chart"
        )


def union_trading_index(series_map: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Sorted union of all trading-day indexes across codes."""
    stamps: set[pd.Timestamp] = set()
    for df in series_map.values():
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        stamps.update(pd.Timestamp(ts).normalize() for ts in idx)
    return pd.DatetimeIndex(sorted(stamps))


def align_closes(
    series_map: dict[str, pd.DataFrame],
    index: "pd.DatetimeIndex | None" = None,
) -> pd.DataFrame:
    """Reindex every Close column onto the union and forward-fill holes.

    Leading NaNs (code not yet listed on an earlier union date) are kept, not
    back-filled, so a younger series does not inherit a fabricated price.
    """
    union = index if index is not None else union_trading_index(series_map)
    closes = {}
    for code, df in series_map.items():
        s = df["Close"].copy()
        s.index = pd.DatetimeIndex(df.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]
        closes[code] = s.reindex(union)
    aligned = pd.DataFrame(closes, index=union)
    return aligned.ffill()


def find_gaps(
    series_map: dict[str, pd.DataFrame],
    index: "pd.DatetimeIndex | None" = None,
    *,
    limit: int = GAP_LIMIT_TRADING_DAYS,
) -> dict[str, list[Gap]]:
    """Find internal missing runs (holes > ``limit``) per code.

    The check uses raw reindexing *without* forward fill: only missing union
    dates between a code's first and last real trading day count.  Leading and
    trailing union dates are listing/delisting edges, not gaps.
    """
    union = index if index is not None else union_trading_index(series_map)
    found: dict[str, list[Gap]] = {}
    for code, df in series_map.items():
        s = df["Close"].copy()
        s.index = pd.DatetimeIndex(df.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]
        raw = s.reindex(union)
        valid = raw.dropna()
        if len(valid) < 2:
            continue
        span = raw.loc[valid.index[0] : valid.index[-1]]
        gaps: list[Gap] = []
        run_start = None
        run_end = None
        run_len = 0
        for stamp, value in span.items():
            if pd.isna(value):
                if run_start is None:
                    run_start = stamp
                run_end = stamp
                run_len += 1
            else:
                if run_len > limit:
                    gaps.append(Gap(code, run_start, run_end, run_len))
                run_start = None
                run_end = None
                run_len = 0
        if gaps:
            found[code] = gaps
    return found


def normalize_first_close(aligned: pd.DataFrame) -> pd.DataFrame:
    """Scale each column to 1.0 at its own first available (aligned) close."""
    normalized = aligned.copy()
    for code in normalized.columns:
        col = normalized[code].dropna()
        if col.empty:
            continue
        base = col.iloc[0]
        if base and not pd.isna(base):
            normalized[code] = normalized[code] / base
    return normalized


def align_and_normalize(
    series_map: dict[str, pd.DataFrame],
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, dict[str, list[Gap]]]:
    """Union align, detect gaps, then normalize.  Order matters (see module)."""
    union = union_trading_index(series_map)
    aligned = align_closes(series_map, union)
    gaps = find_gaps(series_map, union)
    normalized = normalize_first_close(aligned)
    return union, aligned, normalized, gaps
