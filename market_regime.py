"""
Layer 1: Market Regime Detector
Determines overall market posture: Aggressive / Normal / Cautious / Defensive / Cash
"""
import pandas as pd
import numpy as np
from config import REGIME_CONFIG, REGIME_POSTURE, MACRO_LIQUIDITY_CONFIG
from scoring_utils import normalize_score, weighted_composite, grade_score


def compute_regime(
    nifty_df: pd.DataFrame,
    all_stock_data: dict[str, pd.DataFrame],
    macro_liquidity: dict | None = None,
) -> dict:
    """
    Analyze market regime based on:
    1. Nifty 50 vs 200 DMA
    2. 50 DMA vs 200 DMA (golden/death cross)
    3. Breadth: % of stocks above 50 DMA
    4. Breadth: % of stocks above 200 DMA
    5. Net new 52-week highs vs lows

    Returns dict with regime score, label, signals, and posture parameters.
    """
    close = nifty_df["Close"]
    cfg = REGIME_CONFIG

    signals = {}
    scores = []

    # ── Signal 1: Nifty 50 vs 200 DMA ──────────────────────────
    ma200 = close.rolling(200).mean()
    latest_close = close.iloc[-1]
    latest_ma200 = ma200.iloc[-1]
    pct_from_200 = ((latest_close - latest_ma200) / latest_ma200) * 100

    if pct_from_200 > cfg["index_near_200dma_pct"]:
        s1 = 1
        s1_label = f"Bullish (Nifty {pct_from_200:+.1f}% above 200 DMA)"
    elif pct_from_200 < -cfg["index_near_200dma_pct"]:
        s1 = -1
        s1_label = f"Bearish (Nifty {pct_from_200:+.1f}% below 200 DMA)"
    else:
        s1 = 0
        s1_label = f"Neutral (Nifty near 200 DMA, {pct_from_200:+.1f}%)"
    signals["index_vs_200dma"] = {"score": s1, "detail": s1_label}
    scores.append(s1)

    # ── Signal 2: 50 DMA vs 200 DMA crossover ──────────────────
    ma50 = close.rolling(50).mean()
    if ma50.iloc[-1] > ma200.iloc[-1]:
        if ma50.iloc[-5] <= ma200.iloc[-5]:  # recent golden cross
            s2 = 1
            s2_label = "Bullish (fresh golden cross)"
        else:
            s2 = 1
            s2_label = "Bullish (50 DMA > 200 DMA)"
    elif ma50.iloc[-1] < ma200.iloc[-1]:
        if ma50.iloc[-5] >= ma200.iloc[-5]:  # recent death cross
            s2 = -1
            s2_label = "Bearish (fresh death cross)"
        else:
            s2 = -1
            s2_label = "Bearish (50 DMA < 200 DMA)"
    else:
        s2 = 0
        s2_label = "Neutral (MAs converging)"
    signals["ma_crossover"] = {"score": s2, "detail": s2_label}
    scores.append(s2)

    # ── Signal 3: % of stocks above 50 DMA ─────────────────────
    above_50 = 0
    total = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 50:
            continue
        total += 1
        stock_ma50 = df["Close"].rolling(50).mean()
        if df["Close"].iloc[-1] > stock_ma50.iloc[-1]:
            above_50 += 1

    breadth_50 = (above_50 / total * 100) if total > 0 else 50
    if breadth_50 >= cfg["breadth_50dma_bullish"]:
        s3 = 1
        s3_label = f"Bullish ({breadth_50:.0f}% above 50 DMA)"
    elif breadth_50 <= cfg["breadth_50dma_bearish"]:
        s3 = -1
        s3_label = f"Bearish ({breadth_50:.0f}% above 50 DMA)"
    else:
        s3 = 0
        s3_label = f"Neutral ({breadth_50:.0f}% above 50 DMA)"
    signals["breadth_50dma"] = {"score": s3, "detail": s3_label, "value": breadth_50}
    scores.append(s3)

    # ── Signal 4: % of stocks above 200 DMA ────────────────────
    above_200 = 0
    total_200 = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 200:
            continue
        total_200 += 1
        stock_ma200 = df["Close"].rolling(200).mean()
        if df["Close"].iloc[-1] > stock_ma200.iloc[-1]:
            above_200 += 1

    breadth_200 = (above_200 / total_200 * 100) if total_200 > 0 else 50
    if breadth_200 >= cfg["breadth_200dma_bullish"]:
        s4 = 1
        s4_label = f"Bullish ({breadth_200:.0f}% above 200 DMA)"
    elif breadth_200 <= cfg["breadth_200dma_bearish"]:
        s4 = -1
        s4_label = f"Bearish ({breadth_200:.0f}% above 200 DMA)"
    else:
        s4 = 0
        s4_label = f"Neutral ({breadth_200:.0f}% above 200 DMA)"
    signals["breadth_200dma"] = {"score": s4, "detail": s4_label, "value": breadth_200}
    scores.append(s4)

    # ── Signal 5: Net new 52-week highs - lows ──────────────────
    new_highs = 0
    new_lows = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 252:
            continue
        high_52w = df["High"].rolling(252).max().iloc[-1]
        low_52w = df["Low"].rolling(252).min().iloc[-1]
        current = df["Close"].iloc[-1]
        # Within 2% of 52-week high = new high
        if current >= high_52w * 0.98:
            new_highs += 1
        # Within 2% of 52-week low = new low
        if current <= low_52w * 1.02:
            new_lows += 1

    net_new_highs = new_highs - new_lows
    if net_new_highs >= cfg["net_new_highs_bullish"]:
        s5 = 1
        s5_label = f"Bullish (Net new highs: {net_new_highs}, H:{new_highs} L:{new_lows})"
    elif net_new_highs <= cfg["net_new_highs_bearish"]:
        s5 = -1
        s5_label = f"Bearish (Net new highs: {net_new_highs}, H:{new_highs} L:{new_lows})"
    else:
        s5 = 0
        s5_label = f"Neutral (Net new highs: {net_new_highs}, H:{new_highs} L:{new_lows})"
    signals["net_new_highs"] = {"score": s5, "detail": s5_label, "highs": new_highs, "lows": new_lows}
    scores.append(s5)

    # ── Aggregate Regime Score ──────────────────────────────────
    raw_score = sum(scores)  # range: -5 to +5

    # ── Signal 6 (optional): Macro Liquidity Adjustment ───────
    if macro_liquidity:
        ml_adj = macro_liquidity.get("regime_adjustment", 0)
        raw_score += ml_adj
        signals["macro_liquidity"] = {
            "score": ml_adj,
            "detail": f"Macro liquidity: {macro_liquidity['label']} (score {macro_liquidity['score']:.0f}, adj {ml_adj:+d})",
        }

    # Map to -2..+2 scale
    if raw_score >= 4:
        regime_score = 2
    elif raw_score >= 2:
        regime_score = 1
    elif raw_score >= -1:
        regime_score = 0
    elif raw_score >= -3:
        regime_score = -1
    else:
        regime_score = -2

    posture = REGIME_POSTURE[regime_score]

    # ── Breadth trend (is breadth improving or deteriorating?) ──
    # Check if breadth_50 was higher or lower a week ago
    breadth_trend = "stable"
    above_50_prev = 0
    total_prev = 0
    for ticker, df in all_stock_data.items():
        if len(df) < 55:
            continue
        total_prev += 1
        stock_ma50 = df["Close"].rolling(50).mean()
        if len(stock_ma50) > 5 and df["Close"].iloc[-6] > stock_ma50.iloc[-6]:
            above_50_prev += 1
    if total_prev > 0:
        breadth_50_prev = above_50_prev / total_prev * 100
        if breadth_50 > breadth_50_prev + 3:
            breadth_trend = "improving"
        elif breadth_50 < breadth_50_prev - 3:
            breadth_trend = "deteriorating"

    return {
        "regime_score": regime_score,
        "raw_score": raw_score,
        "label": posture["label"],
        "posture": posture,
        "signals": signals,
        "breadth_trend": breadth_trend,
        "macro_liquidity": macro_liquidity,
        "summary": (
            f"Market Regime: {posture['label']} (score {regime_score}, raw {raw_score}/5)\n"
            f"  Breadth trend: {breadth_trend}\n"
            f"  Max capital: {posture['max_capital_pct']}% | "
            f"Risk/trade: {posture['risk_per_trade_pct']}% | "
            f"Max new positions: {posture['max_new_positions']}"
        ),
    }


def compute_macro_liquidity_score(
    macro_data: dict,
    fii_dii_data: dict | None = None,
) -> dict:
    """Compute macro liquidity regime score from FII flows, VIX, USD/INR, yield curve.

    Args:
        macro_data: Dict from fetch_macro_data() with keys like "India VIX", "USD/INR", etc.
        fii_dii_data: Optional dict with FII/DII cumulative flows by timeframe.

    Returns:
        Dict with score (0-100), components, label, and regime adjustment.
    """
    cfg = MACRO_LIQUIDITY_CONFIG
    components = {}

    # ── Component 1: FII Flow Trend (30%) ──────────────────────
    fii_score = 50.0  # neutral default
    if fii_dii_data:
        # Check multiple timeframes for flow direction
        flow_signals = []
        for key in ["1w", "2w", "1m"]:
            flow = fii_dii_data.get(key, {}).get("fii_net", 0)
            if flow > 0:
                flow_signals.append(1)
            elif flow < 0:
                flow_signals.append(-1)
            else:
                flow_signals.append(0)

        # Weight recent flows more
        if flow_signals:
            weighted_flow = (
                flow_signals[0] * 0.5 +  # 1w (most recent)
                (flow_signals[1] if len(flow_signals) > 1 else 0) * 0.3 +
                (flow_signals[2] if len(flow_signals) > 2 else 0) * 0.2
            )
            fii_score = normalize_score(weighted_flow, -1, 1)

    components["fii_flows"] = {"score": fii_score, "weight": cfg["fii_flow_weight"]}

    # ── Component 2: VIX Trend (25%) ──────────────────────────
    vix_score = 50.0
    vix_data = macro_data.get("India VIX", {})
    if vix_data:
        vix_price = vix_data.get("price", 18)
        # Lower VIX = better liquidity environment
        vix_score = normalize_score(vix_price, cfg["vix_overbought"], cfg["vix_oversold"])

        # Bonus/penalty for VIX direction
        vix_change = vix_data.get("change_pct", 0)
        if vix_change < -3:  # VIX dropping fast = bullish
            vix_score = min(100, vix_score + 10)
        elif vix_change > 5:  # VIX spiking = bearish
            vix_score = max(0, vix_score - 15)

    components["vix_trend"] = {"score": vix_score, "weight": cfg["vix_trend_weight"]}

    # ── Component 3: USD/INR Trend (20%) ─────────────────────
    usdinr_score = 50.0
    usdinr_data = macro_data.get("USD/INR", {})
    if usdinr_data:
        # Weakening rupee (rising USD/INR) = headwind for equity
        usdinr_change = usdinr_data.get("change_pct", 0)
        # Negative change_pct means rupee strengthening = positive
        usdinr_score = normalize_score(usdinr_change, cfg["usdinr_strong_threshold"], -cfg["usdinr_strong_threshold"])

    components["usdinr_trend"] = {"score": usdinr_score, "weight": cfg["usdinr_trend_weight"]}

    # ── Component 4: Yield Curve (25%) ─────────────────────────
    yc_score = 50.0
    spread_data = macro_data.get("10Y-5Y Spread", {})
    if spread_data:
        spread = spread_data.get("price", 0)
        # Positive spread = healthy, negative = inversion warning
        yc_score = normalize_score(spread, cfg["yield_curve_inversion_threshold"], 1.0)

    components["yield_curve"] = {"score": yc_score, "weight": cfg["yield_curve_weight"]}

    # ── Composite Score ────────────────────────────────────────
    composite = weighted_composite([
        (c["score"], c["weight"]) for c in components.values()
    ])

    # Map to label
    if composite >= 70:
        label = "Supportive"
        regime_adj = 1  # boost regime by +1
    elif composite >= 45:
        label = "Neutral"
        regime_adj = 0
    elif composite >= 25:
        label = "Headwind"
        regime_adj = -1
    else:
        label = "Hostile"
        regime_adj = -2

    return {
        "score": composite,
        "grade": grade_score(composite),
        "label": label,
        "regime_adjustment": regime_adj,
        "components": components,
    }


def print_regime(regime: dict) -> None:
    """Pretty-print regime analysis."""
    print("\n" + "=" * 65)
    print("  LAYER 1: MARKET REGIME ANALYSIS")
    print("=" * 65)
    print(f"\n  {regime['summary']}\n")
    print("  Individual Signals:")
    for name, sig in regime["signals"].items():
        icon = {1: "+", 0: "~", -1: "-"}.get(sig["score"], "?")
        print(f"    [{icon}] {name}: {sig['detail']}")
    print()
