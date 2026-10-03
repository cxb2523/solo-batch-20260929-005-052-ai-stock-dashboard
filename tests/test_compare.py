"""Tests for the compare package, using fake quotes only."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from compare.align import gap_violations, normalize, union_align
from compare.app import compare_codes
from compare.data_source import QuoteDataSource


def make_history(days, start=100.0):
    idx = pd.bdate_range("2026-01-05", periods=days)
    close = start + np.arange(days, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1,
         "Close": close, "Volume": 1000},
        index=idx,
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# ------------------------------------------------------- concurrent merging
def test_concurrent_requests_merge_into_one_upstream_call():
    calls = []
    barrier = threading.Barrier(8)

    def upstream(code, period="1y"):
        calls.append(code)
        time.sleep(0.2)  # keep the in-flight window open
        return make_history(60)

    source = QuoteDataSource(upstream=upstream)

    def worker():
        barrier.wait()
        return source.fetch("AAA")

    with __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: worker(), range(8)))

    assert calls == ["AAA"]  # exactly one upstream call
    assert all(r is results[0] for r in results)


def test_fetch_many_limits_concurrency_to_four():
    active = 0
    peak = 0
    lock = threading.Lock()

    def upstream(code, period="1y"):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return make_history(60)

    source = QuoteDataSource(upstream=upstream)
    data, errors = source.fetch_many([f"C{i}" for i in range(10)])

    assert not errors
    assert len(data) == 10
    assert peak <= 4


# ------------------------------------------------------------------- gaps
def test_gap_violation_flags_only_the_gappy_code():
    good = make_history(60)["Close"]
    gappy_idx = pd.bdate_range("2026-01-05", periods=60).delete(range(20, 28))
    gappy = pd.Series(50 + np.arange(len(gappy_idx)), index=gappy_idx)

    unfilled = union_align({"GOOD": good, "GAPPY": gappy}, fill=False)
    violations = gap_violations(unfilled, max_gap=5)

    assert "GAPPY" in violations
    assert violations["GAPPY"][0][2] == 8
    assert "GOOD" not in violations


def test_alignment_forward_fills_and_normalizes_after_alignment():
    days = pd.bdate_range("2026-01-05", periods=10)
    a = pd.Series(np.arange(10.0) + 10, index=days)
    b = pd.Series(np.arange(8.0) + 100, index=days.delete([3, 4]))

    aligned = union_align({"A": a, "B": b})
    assert aligned["B"].isna().sum() == 0  # interior holes filled
    assert aligned["B"].iloc[4] == aligned["B"].iloc[2]  # forward-filled

    normed = normalize(aligned)
    assert normed["A"].iloc[0] == pytest.approx(1.0)
    assert normed["B"].iloc[0] == pytest.approx(1.0)
    # normalization is relative to each code's own first aligned close
    assert normed["B"].iloc[-1] == pytest.approx(aligned["B"].iloc[-1] / 100.0)


def test_compare_codes_reports_gap_error_per_row_without_aborting():
    def upstream(code, period="1y"):
        if code == "GAPPY":
            idx = pd.bdate_range("2026-01-05", periods=60).delete(range(20, 28))
            close = 50 + np.arange(len(idx), dtype=float)
            return pd.DataFrame({"Open": close, "High": close, "Low": close,
                                 "Close": close, "Volume": 1}, index=idx)
        return make_history(60)

    source = QuoteDataSource(upstream=upstream)
    result = compare_codes(["GOOD", "GAPPY"], source=source, show=False)

    assert "GOOD" not in result["errors"]
    assert "GAPPY" in result["errors"]
    assert "gap" in result["errors"]["GAPPY"]
    # both codes still have trends/indicators/predictions
    for code in ("GOOD", "GAPPY"):
        entry = result["codes"][code]
        assert not entry["trend"].dropna().empty
        assert not entry["ma"].dropna().empty
        assert not entry["rsi"].dropna().empty
        assert set(entry["macd"]) == {"line", "signal", "hist"}
        assert np.isfinite(entry["prediction"])


# --------------------------------------------------------- failure handling
def test_failed_code_errors_per_row_and_never_caches():
    calls = {"BAD": 0, "GOOD": 0}
    clock = FakeClock()

    def upstream(code, period="1y"):
        calls[code] += 1
        if code == "BAD":
            raise RuntimeError("boom")
        return make_history(60)

    source = QuoteDataSource(upstream=upstream, backoff_base=1.0, clock=clock)
    data, errors = source.fetch_many(["GOOD", "BAD"])

    assert "GOOD" in data
    assert "BAD" not in data
    assert "boom" in errors["BAD"]

    # failure is gated by backoff, not cached: immediate retry hits no upstream
    with pytest.raises(RuntimeError, match="backoff"):
        source.fetch("BAD")
    assert calls["BAD"] == 1

    # after the backoff window the upstream is retried (exponential: 1s, 2s)
    clock.advance(1.1)
    with pytest.raises(RuntimeError, match="boom"):
        source.fetch("BAD")
    assert calls["BAD"] == 2

    with pytest.raises(RuntimeError, match="backoff"):
        source.fetch("BAD")
    assert calls["BAD"] == 2  # second failure doubled the gate to 2s

    clock.advance(2.1)
    with pytest.raises(RuntimeError, match="boom"):
        source.fetch("BAD")
    assert calls["BAD"] == 3


def test_compare_codes_survives_a_fully_failing_code():
    def upstream(code, period="1y"):
        if code == "DEAD":
            raise RuntimeError("delisted")
        return make_history(60)

    source = QuoteDataSource(upstream=upstream)
    result = compare_codes(["ALIVE", "DEAD"], source=source, show=False)

    assert "ALIVE" in result["codes"]
    assert "DEAD" not in result["codes"]
    assert result["errors"]["DEAD"].find("delisted") >= 0
    assert "ALIVE" in result["normalized"].columns
    assert "DEAD" not in result["normalized"].columns


# ------------------------------------------------------------------- cache
def test_cache_hit_within_ttl_and_refetch_after_expiry():
    calls = []
    clock = FakeClock()

    def upstream(code, period="1y"):
        calls.append(code)
        return make_history(60)

    source = QuoteDataSource(upstream=upstream, ttl=300.0, clock=clock)
    source.fetch("AAA")
    source.fetch("AAA")
    assert calls == ["AAA"]  # served from cache

    clock.advance(299.0)
    source.fetch("AAA")
    assert calls == ["AAA"]

    clock.advance(2.0)  # now past the 300s TTL
    source.fetch("AAA")
    assert calls == ["AAA", "AAA"]


def test_short_series_is_not_cached():
    calls = []

    def upstream(code, period="1y"):
        calls.append(code)
        return make_history(10)  # fewer than 30 trading days

    source = QuoteDataSource(upstream=upstream)
    first = source.fetch("SMALL")
    second = source.fetch("SMALL")

    assert len(first) == len(second) == 10
    assert calls == ["SMALL", "SMALL"]  # never cached, refetched every time
