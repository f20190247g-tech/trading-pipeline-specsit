"""
Layer 2: Sector Relative Strength Scanner
Identifies sectors with emerging relative strength using Mansfield RS.
"""
import pandas as pd
import numpy as np
from config import SECTOR_CONFIG


def compute_mansfield_rs(
    sector_close: pd.Series,
    index_close: pd.Series,
    ma_period: int = None,
) -> pd.Series:
    """
    Compute Mansfield Relative Strength.
    RS = (sector / index) / SMA(sector / index, ma_period) - 1
    Positive = outperforming, Negative = underperforming.
    """
    if ma_period is None:
        ma_period = SECTOR_CONFIG["rs_ma_period"]

    # Align dates
    combined = pd.DataFrame({
        "sector": sector_close,
        "index": index_close,
    }).dropna()

    ratio = combined["sector"] / combined["index"]
    # Smooth ratio with 5-day EMA to reduce daily noise
    ratio = ratio.ewm(span=5, min_periods=3).mean()
    ratio_ma = ratio.rolling(min(ma_period, len(ratio) - 1)).mean()
    mansfield_rs = (ratio / ratio_ma - 1) * 100  # as percentage
    return mansfield_rs


def compute_rs_momentum(
    sector_close: pd.Series,
    index_close: pd.Series,
    periods: list[int] = None,
) -> dict[str, float]:
    """
    Compute rate of change of relative strength over multiple periods.
    Returns dict: {"1m": x, "3m": y, "6m": z}
    """
    if periods is None:
        periods = SECTOR_CONFIG["momentum_periods"]

    ratio = sector_close / index_close
    ratio = ratio.dropna()
    # Smooth ratio with 5-day EMA to reduce daily noise
    ratio = ratio.ewm(span=5, min_periods=3).mean()

    labels = {5: "1w", 10: "2w", 21: "1m", 63: "3m", 126: "6m"}
    result = {}
    for p in periods:
        label = labels.get(p, f"{p}d")
        if len(ratio) > p:
            roc = (ratio.iloc[-1] / ratio.iloc[-p] - 1) * 100
            result[label] = round(roc, 2)
        else:
            result[label] = None
    return result


def analyze_rs_trend(mansfield_rs: pd.Series, lookback: int = 21) -> str:
    """
    Determine RS trend direction over the last `lookback` days.
    Returns: "rising", "falling", or "flat"
    """
    if len(mansfield_rs) < lookback:
        return "flat"
    recent = mansfield_rs.iloc[-lookback:]
    # Simple linear regression slope
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    if slope > 0.02:
        return "rising"
    elif slope < -0.02:
        return "falling"
    return "flat"


def classify_sector_stage(sector_close: pd.Series) -> dict:
    """Classify a sector index into Weinstein Stage 1-4.

    Uses 150-day MA as the primary decision line (≈30 weeks).

    Returns:
        dict with stage, detail, ma_slope, pct_above_ma.
    """
    if len(sector_close) < 200:
        return {"stage": 0, "detail": "Insufficient data"}

    ma150 = sector_close.rolling(150).mean()
    ma50 = sector_close.rolling(50).mean()

    latest = sector_close.iloc[-1]
    latest_ma150 = ma150.iloc[-1]
    latest_ma50 = ma50.iloc[-1]

    if pd.isna(latest_ma150):
        return {"stage": 0, "detail": "MA not available"}

    # MA slope (20-day)
    ma_slope = (ma150.iloc[-1] - ma150.iloc[-20]) / ma150.iloc[-20] * 100
    pct_above_ma = (latest / latest_ma150 - 1) * 100

    # Stage classification
    if latest > latest_ma150 and ma_slope > 0.1 and latest_ma50 > latest_ma150:
        stage = 2  # Advancing
        if pct_above_ma <= 5:
            substage = "Early S2"
        elif pct_above_ma <= 12:
            substage = "Mid S2"
        else:
            substage = "Late S2"
    elif latest > latest_ma150 and ma_slope < 0.1 and ma_slope > -0.1:
        stage = 3  # Topping
        substage = "Topping"
    elif latest < latest_ma150 and ma_slope < 0:
        stage = 4  # Declining
        substage = "Declining"
    else:
        stage = 1  # Basing
        substage = "Basing"

    return {
        "stage": stage,
        "substage": substage,
        "ma_slope": round(ma_slope, 2),
        "pct_above_ma": round(pct_above_ma, 1),
        "detail": f"Stage {stage} ({substage})",
    }


def scan_sectors(
    sector_data: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
) -> list[dict]:
    """
    Analyze all sectors and return ranked list.

    Returns list of dicts sorted by composite RS score (best first):
    [{
        "sector": name,
        "mansfield_rs": latest_value,
        "rs_trend": "rising"/"falling"/"flat",
        "momentum": {"1m": x, "3m": y, "6m": z},
        "composite_score": float,
    }, ...]
    """
    nifty_close = nifty_df["Close"]
    results = []

    for sector_name, sector_df in sector_data.items():
        sector_close = sector_df["Close"]

        # Mansfield RS
        mrs = compute_mansfield_rs(sector_close, nifty_close)
        if len(mrs.dropna()) < 20:
            continue

        sector_stage = classify_sector_stage(sector_close)

        # 5-day trailing mean for ranking stability
        latest_rs = mrs.iloc[-5:].mean() if len(mrs) >= 5 else mrs.iloc[-1]
        rs_trend = analyze_rs_trend(mrs)
        momentum = compute_rs_momentum(sector_close, nifty_close)

        # Composite score: weighted combination
        # Mansfield RS level (40%) + RS trend (20%) + momentum blend (40%)
        trend_score = {"rising": 1, "flat": 0, "falling": -1}[rs_trend]

        mom_values = [v for v in momentum.values() if v is not None]
        avg_momentum = np.mean(mom_values) if mom_values else 0

        composite = (
            0.4 * latest_rs +
            0.2 * trend_score * 2 +  # scale trend to be comparable
            0.4 * avg_momentum
        )

        results.append({
            "sector": sector_name,
            "mansfield_rs": round(latest_rs, 2),
            "rs_trend": rs_trend,
            "momentum": momentum,
            "composite_score": round(composite, 2),
            "sector_stage": sector_stage,
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def get_top_sectors(sector_rankings: list[dict], n: int = None) -> list[str]:
    """Return names of top N sectors by composite score."""
    if n is None:
        n = SECTOR_CONFIG["top_sectors_count"]
    # Only include sectors with positive RS trend or at least neutral
    top = [s for s in sector_rankings[:n] if s["rs_trend"] != "falling"]
    # If we filtered too many, take top N regardless
    if len(top) < 2:
        top = sector_rankings[:n]
    return [s["sector"] for s in top]


def print_sector_rankings(rankings: list[dict], top_n: int = None) -> None:
    """Pretty-print sector RS rankings."""
    if top_n is None:
        top_n = SECTOR_CONFIG["top_sectors_count"]

    print("\n" + "=" * 65)
    print("  LAYER 2: SECTOR RELATIVE STRENGTH")
    print("=" * 65)
    print(f"\n  {'Rank':<5} {'Sector':<22} {'RS':>7} {'Trend':<8} {'1m':>7} {'3m':>7} {'6m':>7} {'Score':>7}")
    print("  " + "-" * 73)

    for i, s in enumerate(rankings):
        mom = s["momentum"]
        marker = " <<" if i < top_n else ""
        trend_icon = {"rising": "^", "flat": "-", "falling": "v"}[s["rs_trend"]]
        print(
            f"  {i+1:<5} {s['sector']:<22} "
            f"{s['mansfield_rs']:>7.2f} {trend_icon:<8} "
            f"{mom.get('1m', 'N/A'):>7} {mom.get('3m', 'N/A'):>7} {mom.get('6m', 'N/A'):>7} "
            f"{s['composite_score']:>7.2f}{marker}"
        )
    print(f"\n  << = Top {top_n} sectors (hunting ground)")
    print()
