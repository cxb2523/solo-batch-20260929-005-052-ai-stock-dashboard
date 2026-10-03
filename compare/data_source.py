"""Concurrent quote fetching with request coalescing, TTL cache and backoff.

Trade-offs implemented here (keep in sync with align.py / app.py):
- Concurrent requests for the same (code, period) are merged into ONE upstream
  call; otherwise the TTL cache cannot stop instantaneous duplicates.
- Only successful series with at least ``min_rows`` trading days are cached.
  Failures are never cached; instead an exponential backoff gate delays the
  next upstream retry for that code.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

DEFAULT_TTL = 300.0          # seconds
DEFAULT_MAX_WORKERS = 4      # upstream concurrency limit
DEFAULT_MIN_ROWS = 30        # minimum trading days required for caching
DEFAULT_BACKOFF_BASE = 0.5   # seconds; doubles per consecutive failure
DEFAULT_BACKOFF_CAP = 60.0


def _default_upstream(code: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV history via yfinance (imported lazily)."""
    import yfinance as yf

    history = yf.Ticker(code).history(period=period)
    if history is None or history.empty:
        raise RuntimeError(f"no data returned for {code}")
    return history


class _Inflight:
    """Shared slot for one in-flight upstream call."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: pd.DataFrame | None = None
        self.error: BaseException | None = None


class QuoteDataSource:
    """Fetch quotes for many codes concurrently, at most ``max_workers`` at a time."""

    def __init__(
        self,
        upstream=None,
        ttl: float = DEFAULT_TTL,
        max_workers: int = DEFAULT_MAX_WORKERS,
        min_rows: int = DEFAULT_MIN_ROWS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        clock=time.monotonic,
    ) -> None:
        self._upstream = upstream or _default_upstream
        self._ttl = ttl
        self._max_workers = max_workers
        self._min_rows = min_rows
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._clock = clock

        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers)
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._inflight: dict[tuple[str, str], _Inflight] = {}
        self._failures: dict[tuple[str, str], tuple[int, float]] = {}

    # ------------------------------------------------------------------ API
    def fetch(self, code: str, period: str = "1y") -> pd.DataFrame:
        """Fetch one code, merging with any identical in-flight request."""
        key = (code.upper(), period)
        now = self._clock()

        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] > now:
                return hit[1]

            failures, next_allowed = self._failures.get(key, (0, 0.0))
            if next_allowed > now:
                raise RuntimeError(
                    f"{key[0]}: upstream failing, retry after "
                    f"{next_allowed - now:.1f}s backoff (attempt {failures})"
                )

            slot = self._inflight.get(key)
            if slot is None:
                slot = _Inflight()
                self._inflight[key] = slot
                owner = True
            else:
                owner = False

        if not owner:
            slot.event.wait()
            if slot.error is not None:
                raise RuntimeError(f"{key[0]}: merged request failed: {slot.error}")
            return slot.result

        try:
            with self._semaphore:
                df = self._upstream(key[0], period)
            if df is None or len(df) == 0:
                raise RuntimeError(f"{key[0]}: empty series")
            self._on_success(key, df)
            slot.result = df
            return df
        except Exception as exc:
            self._on_failure(key)
            slot.error = exc
            raise
        finally:
            slot.event.set()
            with self._lock:
                self._inflight.pop(key, None)

    def fetch_many(
        self, codes, period: str = "1y"
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Fetch many codes concurrently (<= max_workers upstream calls).

        Returns ``(data, errors)``; a code that fails appears only in ``errors``
        and never blocks the others.
        """
        data: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self.fetch, code, period): code for code in codes}
            for future in as_completed(futures):
                code = futures[future].upper()
                try:
                    data[code] = future.result()
                except Exception as exc:  # noqa: BLE001 - surfaced per row
                    errors[code] = str(exc)
        return data, errors

    # -------------------------------------------------------------- internals
    def _on_success(self, key, df: pd.DataFrame) -> None:
        with self._lock:
            self._failures.pop(key, None)
            if len(df) >= self._min_rows:
                self._cache[key] = (self._clock() + self._ttl, df)

    def _on_failure(self, key) -> None:
        with self._lock:
            failures, _ = self._failures.get(key, (0, 0.0))
            failures += 1
            delay = min(self._backoff_cap, self._backoff_base * (2 ** (failures - 1)))
            self._failures[key] = (failures, self._clock() + delay)
