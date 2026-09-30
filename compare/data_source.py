"""Concurrent market-quote fetching.

Responsibilities (the four interlocking rules):

* At most ``max_workers`` (4) concurrent upstream calls.
* Concurrent requests for the same ``(code, period)`` are coalesced into a
  single in-flight upstream call (a TTL cache alone cannot absorb repeats
  arriving in the same instant).
* Only successful series with >= ``min_rows`` (30) trading days are cached;
  failures are retried with exponential backoff and never cached.
* Cache entries live for ``ttl`` (300) seconds.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

FetchFunc = Callable[[str, str], "pd.DataFrame"]

DEFAULT_TTL_SECONDS = 300
DEFAULT_MAX_WORKERS = 4
DEFAULT_MIN_ROWS = 30
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5


class DataSourceError(RuntimeError):
    """Raised by the default upstream adapter on empty responses."""


@dataclass
class FetchResult:
    """Outcome of fetching one code."""

    code: str
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.data is not None


def yfinance_fetch(code: str, period: str = "1y") -> pd.DataFrame:
    """Default upstream adapter: yfinance history for one code."""
    import yfinance as yf

    df = yf.Ticker(code).history(period=period, auto_adjust=False)
    if df is None or len(df) == 0:
        raise DataSourceError(f"{code}: upstream returned no data")
    return df


class QuoteFetcher:
    """Fetch many codes concurrently with rate limiting, coalescing and TTL cache."""

    def __init__(
        self,
        fetch_func: Optional[FetchFunc] = None,
        *,
        ttl: float = DEFAULT_TTL_SECONDS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        min_rows: int = DEFAULT_MIN_ROWS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        sleep_func: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._fetch_func = fetch_func or yfinance_fetch
        self._ttl = ttl
        self._min_rows = min_rows
        self._max_attempts = max(1, max_attempts)
        self._backoff_base = backoff_base
        self._sleep = sleep_func or time.sleep
        self._clock = clock or time.monotonic
        # A shared 4-worker pool is the rate limiter: at most four upstream
        # calls run at once, even across concurrent fetch_many callers.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="quote-fetch"
        )
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()
        self._inflight: dict[tuple[str, str], Future] = {}
        self._inflight_lock = threading.Lock()

    def fetch_many(self, codes, period: str = "1y") -> dict[str, FetchResult]:
        """Fetch all codes concurrently; result preserves request order."""
        unique = list(dict.fromkeys(codes))
        futures = {code: self._submit(code, period) for code in unique}
        return {code: futures[code].result() for code in unique}

    def fetch_one(self, code: str, period: str = "1y") -> FetchResult:
        return self._submit(code, period).result()

    def _submit(self, code: str, period: str) -> Future:
        """Return an in-flight future for this key or start exactly one call."""
        key = (code, str(period))
        with self._inflight_lock:
            future = self._inflight.get(key)
            if future is not None:
                return future
            future = self._executor.submit(self._load, key)
            self._inflight[key] = future
        # Register outside the lock: add_done_callback runs the callback inline
        # on an already-finished future, and the callback needs this same lock.
        future.add_done_callback(lambda _f, k=key: self._drop_inflight(k))
        return future

    def _drop_inflight(self, key: tuple[str, str]) -> None:
        with self._inflight_lock:
            self._inflight.pop(key, None)

    def _read_cache(self, key: tuple[str, str]) -> Optional[pd.DataFrame]:
        now = self._clock()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry[0] <= self._ttl:
                return entry[1]
            self._cache.pop(key, None)
            return None

    def _load(self, key: tuple[str, str]) -> FetchResult:
        code, period = key
        cached = self._read_cache(key)
        if cached is not None:
            return FetchResult(code=code, data=cached, cached=True)
        data, error = self._fetch_with_retry(code, period)
        if data is not None:
            # Only successful, long-enough series are ever written to cache.
            with self._cache_lock:
                self._cache[key] = (self._clock(), data)
            return FetchResult(code=code, data=data)
        return FetchResult(code=code, error=error)

    def _fetch_with_retry(self, code: str, period: str):
        error = f"{code}: unknown upstream error"
        for attempt in range(1, self._max_attempts + 1):
            try:
                df = self._fetch_func(code, period)
            except Exception as exc:  # failure: back off and retry, never cache
                # Keep the message free of a code prefix: callers display one
                # row per code and may already know the code from elsewhere.
                error = f"upstream error: {exc}"
            else:
                if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) >= self._min_rows:
                    return df, None
                rows = 0 if df is None else len(df)
                error = (
                    f"{code}: only {rows} trading day(s) returned "
                    f"(need >= {self._min_rows})"
                )
            if attempt < self._max_attempts:
                self._sleep(self._backoff_base * (2 ** (attempt - 1)))
        return None, error

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "QuoteFetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
