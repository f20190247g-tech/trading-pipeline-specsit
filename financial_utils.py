"""
Shared Financial Utilities — standardized quarterly/annual data fetching,
growth computation, and TTM aggregation.

Used by: earnings_analysis.py, value_analysis.py, fundamental_veto.py
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


def get_quarterly_financials(ticker: str) -> dict:
    """Fetch quarterly income statement, balance sheet, and cash flow.

    Tries yfinance as primary source.
    Returns dict with DataFrames: {income, balance, cashflow}
    Columns are dates, rows are line items.
    """
    try:
        yticker = yf.Ticker(ticker)

        income = _safe_fetch(yticker, "quarterly_income_stmt")
        balance = _safe_fetch(yticker, "quarterly_balance_sheet")
        cashflow = _safe_fetch(yticker, "quarterly_cashflow")

        return {
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "ticker": ticker,
            "available": not income.empty,
        }
    except Exception as e:
        logger.warning("Failed to fetch quarterly financials for %s: %s", ticker, e)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(),
                "cashflow": pd.DataFrame(), "ticker": ticker, "available": False}


def get_annual_financials(ticker: str) -> dict:
    """Fetch annual income statement, balance sheet, and cash flow.

    Returns dict with DataFrames: {income, balance, cashflow}
    """
    try:
        yticker = yf.Ticker(ticker)

        income = _safe_fetch(yticker, "income_stmt")
        balance = _safe_fetch(yticker, "balance_sheet")
        cashflow = _safe_fetch(yticker, "cashflow")

        return {
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "ticker": ticker,
            "available": not income.empty,
        }
    except Exception as e:
        logger.warning("Failed to fetch annual financials for %s: %s", ticker, e)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(),
                "cashflow": pd.DataFrame(), "ticker": ticker, "available": False}


def _safe_fetch(yticker, attr: str) -> pd.DataFrame:
    """Safely fetch a yfinance financial attribute."""
    try:
        result = getattr(yticker, attr, None)
        if result is not None and not result.empty:
            return result
    except Exception:
        pass
    return pd.DataFrame()


def extract_row(df: pd.DataFrame, row_names: list[str]) -> pd.Series | None:
    """Extract a row from a financial DataFrame by trying multiple possible names.

    Args:
        df: Financial statement DataFrame (rows=items, cols=dates).
        row_names: List of possible row names (tries in order).

    Returns:
        pd.Series with dates as index, or None if not found.
    """
    if df.empty:
        return None
    for name in row_names:
        if name in df.index:
            return df.loc[name].dropna()
    return None


def compute_sequential_growth(values: pd.Series) -> pd.Series:
    """Compute period-over-period growth rate (%).

    Args:
        values: Series of values (e.g., quarterly EPS), newest first (yfinance default).

    Returns:
        Series of growth rates in %, same order as input.
    """
    if values is None or len(values) < 2:
        return pd.Series(dtype=float)
    # yfinance returns newest first; reverse for chronological
    chronological = values.iloc[::-1]
    growth = chronological.pct_change() * 100
    return growth.iloc[::-1]  # back to newest-first


def compute_yoy_growth(quarterly_values: pd.Series) -> pd.Series:
    """Compute year-over-year growth for quarterly data (%).

    Compares each quarter to the same quarter one year prior (4 quarters back).

    Args:
        quarterly_values: Series of quarterly values, newest first.

    Returns:
        Series of YoY growth rates in %.
    """
    if quarterly_values is None or len(quarterly_values) < 5:
        return pd.Series(dtype=float)
    chronological = quarterly_values.iloc[::-1]
    yoy = (chronological / chronological.shift(4) - 1) * 100
    return yoy.iloc[::-1]


def compute_ttm(quarterly_values: pd.Series, n: int = 4) -> float | None:
    """Compute trailing twelve months sum from quarterly data.

    Args:
        quarterly_values: Series of quarterly values, newest first.
        n: Number of quarters to sum (default 4 = TTM).

    Returns:
        Sum of last n quarters, or None if insufficient data.
    """
    if quarterly_values is None or len(quarterly_values) < n:
        return None
    return float(quarterly_values.head(n).sum())


def safe_divide(numerator, denominator, default=None):
    """Safe division handling zero/None."""
    if denominator is None or denominator == 0:
        return default
    if numerator is None:
        return default
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return default
