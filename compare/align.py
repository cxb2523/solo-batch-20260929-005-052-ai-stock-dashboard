"""Pure alignment helpers for multi-code comparison.

All functions here are side-effect free. Pipeline order matters:
union-align (on the raw closes) -> gap detection on the UNFILLED frame ->
forward-fill -> normalize. Normalization must happen after alignment and is
never cached, otherwise changing the code set would poison cached values.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_MAX_GAP = 5  # trading days


def union_align(close_map: dict[str, pd.Series], fill: bool = True) -> pd.DataFrame:
    """Align close series on the UNION of all trading days.

    Missing days become NaN and are forward-filled when ``fill`` is True
    (leading NaNs before a code's first trading day are kept).
    """
    if not close_map:
        return pd.DataFrame()
    union_index = None
    for series in close_map.values():
        union_index = series.index if union_index is None else union_index.union(series.index)
    frame = pd.DataFrame(
        {code: series.reindex(union_index) for code, series in close_map.items()}
    )
    if fill:
        frame = frame.ffill()
    return frame


def gap_runs(aligned_unfilled: pd.DataFrame) -> dict[str, list[tuple]]:
    """Find NaN runs per code in the aligned-but-unfilled frame.

    Only runs at or after a code's first real observation count as gaps
    (a later listing is not a gap). Returns ``{code: [(start, end, length)]}``.
    """
    gaps: dict[str, list[tuple]] = {}
    for code in aligned_unfilled.columns:
        series = aligned_unfilled[code]
        is_na = series.isna().to_numpy()
        valid = series.notna().to_numpy()
        if not valid.any():
            gaps[code] = [(series.index[0], series.index[-1], len(series))] if len(series) else []
            continue
        first_valid = int(valid.argmax())
        runs: list[tuple] = []
        start = None
        for pos in range(first_valid, len(series)):
            if is_na[pos] and start is None:
                start = pos
            elif not is_na[pos] and start is not None:
                runs.append((series.index[start], series.index[pos - 1], pos - start))
                start = None
        if start is not None:
            runs.append((series.index[start], series.index[-1], len(series) - start))
        gaps[code] = runs
    return gaps


def gap_violations(
    aligned_unfilled: pd.DataFrame, max_gap: int = DEFAULT_MAX_GAP
) -> dict[str, list[tuple]]:
    """Return only the runs longer than ``max_gap`` trading days, per code."""
    return {
        code: [run for run in runs if run[2] > max_gap]
        for code, runs in gap_runs(aligned_unfilled).items()
        if any(run[2] > max_gap for run in runs)
    }


def normalize(aligned: pd.DataFrame) -> pd.DataFrame:
    """Normalize each code by its first available close after alignment.

    Must be called on the aligned (forward-filled) frame, never cached.
    """
    out = aligned.copy()
    for code in out.columns:
        valid = out[code].dropna()
        if not valid.empty and valid.iloc[0] != 0:
            out[code] = out[code] / valid.iloc[0]
    return out
