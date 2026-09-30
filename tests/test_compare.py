"""Tests for the compare package: coalescing, rate limit, gaps, failures, TTL."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from compare.data_source import QuoteFetcher
from compare.align import (
    GAP_LIMIT_TRADING_DAYS,
    align_and_normalize,
    align_closes,
    find_gaps,
    normalize_first_close,
    union_trading_index,
)


def make_series(rows: int = 60, start: str = "2024-01-01", price: float = 100.0,
                seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=rows)
    rng = np.random.default_rng(seed)
    close = price + np.arange(rows) + rng.normal(0, 0.01, rows)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": np.full(rows, 1_000_000),
        },
        index=idx,
    )


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_fetcher(fetch_func, *, ttl=300.0, max_workers=4, min_rows=30,
                 max_attempts=3, backoff_base=0.01, clock=None):
    sleeps = []
    clock = clock or FakeClock()
    return QuoteFetcher(
        fetch_func,
        ttl=ttl,
        max_workers=max_workers,
        min_rows=min_rows,
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        sleep_func=sleeps.append,
        clock=clock,
    ), clock, sleeps


def test_concurrent_same_code_coalesces_to_single_upstream_call():
    """Same-code requests fired together must share one upstream call."""
    calls = []
    entered = threading.Event()
    release = threading.Event()

    def fetch(code, period):
        calls.append(code)
        entered.set()
        release.wait(2.0)
        return make_series(seed=1)

    fetcher, _, _ = make_fetcher(fetch)
    results = {}

    def worker():
        results[threading.get_ident()] = fetcher.fetch_one("AAA", "1y")

    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(8)
    ]
    for t in threads:
        t.start()
    assert entered.wait(1.0)
    time.sleep(0.1)  # let the other threads park on the in-flight future
    release.set()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive()

    assert calls == ["AAA"]
    assert len(results) == 8
    assert all(r.ok and not r.cached for r in results.values())


def test_duplicate_codes_in_one_batch_hit_upstream_once_then_cache():
    call_count = threading.Semaphore(0)

    def fetch(code, period):
        call_count.release()
        return make_series(seed=2)

    fetcher, _, _ = make_fetcher(fetch)
    first = fetcher.fetch_many(["AAA", "AAA", "AAA"])
    assert call_count.acquire(timeout=1)
    assert not call_count.acquire(timeout=0.1)
    assert len(first) == 1 and first["AAA"].ok

    second = fetcher.fetch_many(["AAA"])
    assert second["AAA"].cached and not call_count.acquire(timeout=0.1)


def test_upstream_capped_at_four_concurrent():
    active = 0
    peak = 0
    lock = threading.Lock()

    def fetch(code, period):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return make_series(seed=sum(ord(c) for c in code))

    fetcher, _, _ = make_fetcher(fetch, max_workers=4)
    results = fetcher.fetch_many([f"C{i}" for i in range(12)])
    assert all(r.ok for r in results.values())
    assert peak == 4


def test_ttl_expiry_refetches():
    calls = []

    def fetch(code, period):
        calls.append(1)
        return make_series(seed=len(calls))

    clock = FakeClock()
    fetcher, _, _ = make_fetcher(fetch, ttl=300.0, clock=clock)

    assert fetcher.fetch_one("AAA").ok
    clock.advance(299.9)
    assert fetcher.fetch_one("AAA").cached
    clock.advance(0.2)  # now 300.1 > TTL
    fresh = fetcher.fetch_one("AAA")
    assert fresh.ok and not fresh.cached
    assert len(calls) == 2


def test_failure_is_retried_with_exponential_backoff_and_not_cached():
    attempts = []

    def flaky(code, period):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("boom")
        return make_series(seed=5)

    fetcher, _, sleeps = make_fetcher(
        flaky, max_attempts=3, backoff_base=0.01
    )
    result = fetcher.fetch_one("AAA")
    assert result.ok and not result.cached
    assert len(attempts) == 3
    assert sleeps == [pytest.approx(0.01), pytest.approx(0.02)]


def test_permanent_failure_never_caches_and_refetches():
    calls = 0

    def always_bad(code, period):
        nonlocal calls
        calls += 1
        raise RuntimeError("nope")

    fetcher, _, sleeps = make_fetcher(always_bad, max_attempts=3, backoff_base=0.01)
    first = fetcher.fetch_one("BAD")
    second = fetcher.fetch_one("BAD")
    assert not first.ok and not second.ok
    assert calls == 6  # failures do not populate the cache
    assert len(sleeps) == 4


def test_short_series_below_30_days_is_treated_as_failure():
    def fetch(code, period):
        if code == "SHORT":
            return make_series(rows=29)
        return make_series(rows=60)

    fetcher, _, _ = make_fetcher(fetch, max_attempts=2)
    out = fetcher.fetch_many(["SHORT", "OK"])
    assert not out["SHORT"].ok and "30" in out["SHORT"].error
    assert out["OK"].ok


def test_union_alignment_forward_fill_and_gap_detection():
    full = make_series(rows=40, seed=1)  # 2024-01-01 .. ~2024-02-23
    late = make_series(rows=30, start="2024-01-08", seed=2)

    # Punch a 6-trading-day internal hole in LATE (> 5 limit).
    hole_idx = late.index[10:16]
    gappy = late.drop(index=hole_idx)

    union = union_trading_index({"FULL": full, "LATE": gappy})
    aligned = align_closes({"FULL": full, "LATE": gappy}, union)
    gaps = find_gaps({"FULL": full, "LATE": gappy}, union)

    assert list(aligned.columns) == ["FULL", "LATE"]
    assert len(union) == 40  # LATE's range is a subset of FULL's
    # Leading dates are NaN, not back-filled; internal hole is forward-filled.
    assert pd.isna(aligned["LATE"].iloc[0])
    hole_pos = union.get_loc(hole_idx[0])
    assert aligned["LATE"].iloc[hole_pos] == gappy["Close"].iloc[9]
    assert aligned["FULL"].notna().all()

    assert "LATE" in gaps and len(gaps["LATE"]) == 1
    gap = gaps["LATE"][0]
    assert gap.length == 6 and gap.length > GAP_LIMIT_TRADING_DAYS
    assert gap.start == hole_idx[0] and gap.end == hole_idx[-1]
    assert "FULL" not in gaps


def test_five_day_gap_is_allowed_seven_days_errors():
    ref = make_series(rows=40, seed=9)
    # REF spans the full union calendar; dropping rows from GAP carves real
    # holes against that shared calendar.
    gap5 = ref.drop(index=ref.index[10:15])  # 5 missing -> at limit, OK
    gap7 = ref.drop(index=ref.index[10:17])  # 7 missing -> error
    assert find_gaps({"REF": ref, "GAP": gap5}) == {}
    found = find_gaps({"REF": ref, "GAP": gap7})
    assert found["GAP"][0].length == 7
    assert found["GAP"][0].start == ref.index[10]
    assert found["GAP"][0].end == ref.index[16]


def test_normalization_after_alignment_is_per_code_first_close():
    a = make_series(rows=20, price=100.0, seed=1)
    b = make_series(rows=10, start="2024-01-08", price=500.0, seed=2)
    _, aligned, normalized, _ = align_and_normalize({"AAA": a, "BBB": b})

    # AAA base must be its own first close regardless of the wider code set.
    first_a = aligned["AAA"].dropna().iloc[0]
    assert normalized["AAA"].dropna().iloc[0] == pytest.approx(1.0)
    assert normalized["AAA"].dropna().iloc[-1] == pytest.approx(
        aligned["AAA"].dropna().iloc[-1] / first_a
    )
    first_b = aligned["BBB"].dropna().iloc[0]
    assert normalized["BBB"].dropna().iloc[0] == pytest.approx(1.0)
    assert first_b == pytest.approx(500.0, rel=0.02)


def test_normalization_changes_with_code_set_so_it_must_not_be_cached():
    """Normalization is a post-alignment transform, never a cached series.

    The first-close base is intrinsic to each code, so it survives basket
    changes, while the forward-filled overlap region is defined by the union.
    Caching normalized values ahead of alignment would therefore be wrong: the
    cache can only hold raw upstream series.
    """
    a = make_series(rows=15, seed=1)
    b = make_series(rows=15, start="2024-01-22", seed=3)

    norm_ab = normalize_first_close(align_closes({"A": a, "B": b}))
    norm_a_only = normalize_first_close(align_closes({"A": a}))
    # A's own base is unchanged by the wider basket ...
    assert norm_ab["A"].dropna().iloc[0] == pytest.approx(1.0)
    assert norm_a_only["A"].dropna().iloc[0] == pytest.approx(1.0)
    # ... but the union also covers B's later dates, so the aligned frame is
    # wider than A alone and normalization must be (re)computed after
    # alignment, never read out of an upstream cache.
    assert len(norm_ab["A"]) == len(a) + len(b)
    assert len(norm_a_only["A"]) == len(a)
    overlap = norm_a_only.index.intersection(norm_ab.index)
    pd.testing.assert_series_equal(
        norm_ab["A"].loc[overlap], norm_a_only["A"].loc[overlap],
        check_names=False,
    )


def test_whole_code_failure_is_isolated_row():
    def fetch(code, period):
        if code == "DEAD":
            raise RuntimeError("404 not found")
        return make_series(rows=40, seed=sum(ord(c) for c in code))

    fetcher, _, _ = make_fetcher(fetch, max_attempts=1)
    out = fetcher.fetch_many(["OK1", "DEAD", "OK2"])
    assert out["OK1"].ok and out["OK2"].ok
    assert not out["DEAD"].ok and "404" in out["DEAD"].error
    assert out["DEAD"].error.startswith("upstream error:")


def test_compare_codes_reports_failure_row_but_plots_rest():
    from app import compare_codes

    def fetch(code, period):
        if code == "DEAD":
            raise RuntimeError("simulated outage")
        return make_series(rows=80, seed=sum(ord(c) for c in code))

    fetcher, _, _ = make_fetcher(fetch, max_attempts=1)
    result = compare_codes(["AAA", "DEAD", "BBB"], period="1y",
                           fetcher=fetcher, use_rf=False)

    assert set(result["aligned"].columns) == {"AAA", "BBB"}
    assert "DEAD" in result["errors"]
    assert result["errors"]["DEAD"].startswith("DEAD: upstream error:")
    assert "simulated outage" in result["errors"]["DEAD"]
    assert set(result["indicators"]) == {"AAA", "BBB"}
    for code, table in result["indicators"].items():
        for col in ("MA20", "MA50", "RSI", "MACD", "MACD_signal", "MACD_hist"):
            assert col in table.columns
    assert set(result["predictions"]) == {"AAA", "BBB"}
    assert all(np.isfinite(v) for v in result["predictions"].values())


def test_compare_codes_gap_becomes_row_warning_and_still_plots():
    from app import compare_codes

    def fetch(code, period):
        df = make_series(rows=80, seed=4)
        if code == "GAP":
            df = df.drop(index=df.index[40:47])  # 7-day internal hole
        return df

    fetcher, _, _ = make_fetcher(fetch)
    result = compare_codes(["AAA", "GAP"], period="1y", fetcher=fetcher,
                           use_rf=False)
    assert "GAP" in result["errors"]
    assert "7 consecutive missing" in result["errors"]["GAP"]
    assert result["errors"]["GAP"].startswith("GAP: 7")
    assert "GAP" in result["trends"].columns
    assert result["trends"]["GAP"].dropna().iloc[0] == pytest.approx(1.0)
