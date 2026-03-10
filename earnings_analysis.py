"""
Earnings Acceleration Engine — detects sequential EPS/revenue acceleration.
Minervini's #1 signal: accelerating quarterly earnings growth.
"""
import logging
from financial_utils import (
    get_quarterly_financials,
    extract_row,
    compute_yoy_growth,
    safe_divide,
)
from scoring_utils import normalize_score

logger = logging.getLogger(__name__)

# Row name variants across different yfinance versions / regions
EPS_ROW_NAMES = ["Basic EPS", "Diluted EPS", "BasicEPS", "DilutedEPS"]
REVENUE_ROW_NAMES = ["Total Revenue", "Operating Revenue", "TotalRevenue", "Revenue"]
NET_INCOME_ROW_NAMES = ["Net Income", "NetIncome", "Net Income Common Stockholders"]
OPERATING_INCOME_ROW_NAMES = ["Operating Income", "OperatingIncome", "EBIT"]


def compute_earnings_acceleration(ticker: str) -> dict:
    """Compute quarterly earnings acceleration for a stock.

    Fetches last 8 quarters, computes YoY growth for each quarter,
    then checks if growth is accelerating (2nd derivative positive).

    Returns:
        {
            "ticker": str,
            "data_available": bool,
            "eps_values": list[float],       # last 4-8 quarters (newest first)
            "eps_yoy_growth": list[float],    # YoY growth % per quarter
            "eps_acceleration": bool,         # is growth accelerating?
            "eps_accel_score": 0-100,
            "revenue_values": list[float],
            "revenue_yoy_growth": list[float],
            "revenue_acceleration": bool,
            "revenue_accel_score": 0-100,
            "combined_score": 0-100,
            "trend": "accelerating" | "decelerating" | "stable",
        }
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        fin = get_quarterly_financials(ticker)
        if not fin["available"]:
            return result

        income = fin["income"]

        # Extract EPS
        eps = extract_row(income, EPS_ROW_NAMES)
        eps_result = _analyze_acceleration(eps, "eps")

        # Extract Revenue
        revenue = extract_row(income, REVENUE_ROW_NAMES)
        rev_result = _analyze_acceleration(revenue, "revenue")

        # Extract Net Income (fallback for EPS)
        if eps_result is None:
            net_income = extract_row(income, NET_INCOME_ROW_NAMES)
            eps_result = _analyze_acceleration(net_income, "eps")

        if eps_result is None and rev_result is None:
            return result

        result["data_available"] = True

        if eps_result:
            result.update({
                "eps_values": eps_result["values"],
                "eps_yoy_growth": eps_result["yoy_growth"],
                "eps_acceleration": eps_result["is_accelerating"],
                "eps_accel_score": eps_result["score"],
            })

        if rev_result:
            result.update({
                "revenue_values": rev_result["values"],
                "revenue_yoy_growth": rev_result["yoy_growth"],
                "revenue_acceleration": rev_result["is_accelerating"],
                "revenue_accel_score": rev_result["score"],
            })

        # Combined score (EPS 60%, Revenue 40%)
        eps_score = result.get("eps_accel_score", 50)
        rev_score = result.get("revenue_accel_score", 50)
        combined = eps_score * 0.6 + rev_score * 0.4
        result["combined_score"] = round(combined, 1)

        # Overall trend
        eps_accel = result.get("eps_acceleration", False)
        rev_accel = result.get("revenue_acceleration", False)
        if eps_accel and rev_accel:
            result["trend"] = "accelerating"
        elif eps_accel or rev_accel:
            result["trend"] = "accelerating"
        elif combined < 40:
            result["trend"] = "decelerating"
        else:
            result["trend"] = "stable"

        return result

    except Exception as e:
        logger.warning("Earnings acceleration failed for %s: %s", ticker, e)
        return result


def _analyze_acceleration(values_series, label: str) -> dict | None:
    """Analyze acceleration for a quarterly data series.

    Returns dict with values, yoy_growth, is_accelerating, score.
    """
    if values_series is None or len(values_series) < 5:
        return None

    # Get last 8 quarters max (newest first, which is yfinance default)
    vals = values_series.head(8)
    values_list = [round(float(v), 2) for v in vals if not (isinstance(v, float) and v != v)]

    if len(values_list) < 5:
        return None

    # Compute YoY growth (need at least 5 quarters: current + 4 back)
    yoy_growth = []
    for i in range(len(values_list) - 4):
        current = values_list[i]
        year_ago = values_list[i + 4]
        growth = safe_divide(current - year_ago, abs(year_ago), None)
        if growth is not None:
            yoy_growth.append(round(growth * 100, 1))

    if len(yoy_growth) < 2:
        return None

    # Check acceleration: is the most recent growth > previous growth?
    # Use last 3 data points if available
    is_accelerating = False
    accel_magnitude = 0

    if len(yoy_growth) >= 2:
        # Most recent growth vs prior growth (newest first)
        is_accelerating = yoy_growth[0] > yoy_growth[1]
        accel_magnitude = yoy_growth[0] - yoy_growth[1]

    if len(yoy_growth) >= 3:
        # Confirm with 3-quarter trend
        is_accelerating = yoy_growth[0] > yoy_growth[1] > yoy_growth[2]
        if not is_accelerating:
            # Partial acceleration (at least improving)
            is_accelerating = yoy_growth[0] > yoy_growth[1]

    # Score: based on latest growth level + acceleration
    latest_growth = yoy_growth[0] if yoy_growth else 0

    # Base score from growth level
    if latest_growth >= 30:
        base_score = 80
    elif latest_growth >= 20:
        base_score = 70
    elif latest_growth >= 10:
        base_score = 55
    elif latest_growth >= 0:
        base_score = 40
    else:
        base_score = 20

    # Acceleration bonus/penalty
    if is_accelerating:
        score = min(100, base_score + min(accel_magnitude, 20))
    else:
        score = max(0, base_score - min(abs(accel_magnitude), 15))

    return {
        "values": values_list,
        "yoy_growth": yoy_growth,
        "is_accelerating": is_accelerating,
        "accel_magnitude": round(accel_magnitude, 1),
        "score": round(score, 1),
    }


def compute_margin_trends(ticker: str) -> dict:
    """Compute quarterly margin trends (gross, operating, net).

    Returns dict with margin values and expansion/compression status.
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        fin = get_quarterly_financials(ticker)
        if not fin["available"]:
            return result

        income = fin["income"]

        revenue = extract_row(income, REVENUE_ROW_NAMES)
        operating = extract_row(income, OPERATING_INCOME_ROW_NAMES)
        net_income = extract_row(income, NET_INCOME_ROW_NAMES)

        if revenue is None or len(revenue) < 4:
            return result

        result["data_available"] = True

        # Operating margin trend
        if operating is not None and len(operating) >= 4:
            op_margins = []
            for i in range(min(8, len(operating))):
                rev_val = revenue.iloc[i] if i < len(revenue) else None
                op_val = operating.iloc[i]
                margin = safe_divide(op_val, rev_val)
                if margin is not None:
                    op_margins.append(round(margin * 100, 1))

            if len(op_margins) >= 2:
                result["operating_margins"] = op_margins
                result["op_margin_expanding"] = op_margins[0] > op_margins[1]
                if len(op_margins) >= 4:
                    result["op_margin_yoy_change"] = round(op_margins[0] - op_margins[min(3, len(op_margins)-1)], 1)

        # Net margin trend
        if net_income is not None and len(net_income) >= 4:
            net_margins = []
            for i in range(min(8, len(net_income))):
                rev_val = revenue.iloc[i] if i < len(revenue) else None
                ni_val = net_income.iloc[i]
                margin = safe_divide(ni_val, rev_val)
                if margin is not None:
                    net_margins.append(round(margin * 100, 1))

            if len(net_margins) >= 2:
                result["net_margins"] = net_margins
                result["net_margin_expanding"] = net_margins[0] > net_margins[1]

        return result

    except Exception as e:
        logger.warning("Margin trend failed for %s: %s", ticker, e)
        return result
