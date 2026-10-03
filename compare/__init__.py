"""Multi-code comparison package split out of stock_dashboard.py."""

from .align import gap_runs, gap_violations, normalize, union_align
from .app import compare_codes, create_advanced_chart, main, show_comparison_window
from .data_source import QuoteDataSource

__all__ = [
    "QuoteDataSource",
    "union_align",
    "normalize",
    "gap_runs",
    "gap_violations",
    "compare_codes",
    "show_comparison_window",
    "create_advanced_chart",
    "main",
]
