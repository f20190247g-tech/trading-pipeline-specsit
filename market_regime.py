"""
Layer 1: Market Regime Detector
Determines overall market posture: Aggressive / Normal / Cautious / Defensive / Cash
"""
from __future__ import annotations
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

    # ── Signal 2: 50 DMA vs 200 DMA crossover (slope-aware) ─────
    ma50 = close.rolling(50).mean()
    ma50_slope = (ma50.iloc[-1] - ma50.iloc[-10]) / ma50.iloc[-10] * 100  # 10-day slope %
    ma_gap_pct = (ma50.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1] * 100

    if ma50.iloc[-1] > ma200.iloc[-1]:
        if ma50.iloc[-5] <= ma200.iloc[-5]:  # recent golden cross
            s2 = 1
            s2_label = "Bullish (fresh golden cross)"
        elif ma50_slope < -0.3:
            # 50 DMA above 200 DMA but declining — death cross approaching
            s2 = 0
            s2_label = f"Neutral (50 > 200 but declining, gap {ma_gap_pct:.1f}%)"
        else:
            s2 = 1
            s2_label = f"Bullish (50 DMA > 200 DMA, gap {ma_gap_pct:.1f}%)"
    elif ma50.iloc[-1] < ma200.iloc[-1]:
        if ma50.iloc[-5] >= ma200.iloc[-5]:  # recent death cross
            s2 = -1
            s2_label = "Bearish (fresh death cross)"
        elif ma50_slope > 0.3:
            # 50 DMA below 200 DMA but rising — golden cross approaching
            s2 = 0
            s2_label = f"Neutral (50 < 200 but rising, gap {ma_gap_pct:.1f}%)"
        else:
            s2 = -1
            s2_label = f"Bearish (50 DMA < 200 DMA, gap {ma_gap_pct:.1f}%)"
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


def compute_stockbee_breadth(
    all_stock_data: dict[str, pd.DataFrame],
    lookback_days: int = 30,
) -> dict:
    """Compute Stockbee-style breadth indicators from stock price data.

    Tracks the population of stocks making significant moves across timeframes.
    Inspired by Pradeep Bonde's Market Monitor methodology.

    Returns dict with:
        - daily: list of dicts (last `lookback_days` trading days)
        - latest: most recent day's readings
        - signals: interpreted on/off signals
        - insights: human-readable insights list
    """
    # Build a DataFrame of daily closes for all stocks
    all_closes = {}
    for ticker, df in all_stock_data.items():
        if len(df) < 70:  # need ~63 trading days for quarterly
            continue
        all_closes[ticker] = df["Close"]

    if not all_closes:
        return {"daily": [], "latest": {}, "signals": {}, "insights": []}

    closes_df = pd.DataFrame(all_closes)
    universe_size = len(closes_df.columns)

    # Daily returns
    daily_ret = closes_df.pct_change()

    # Period returns (from N days ago to today)
    ret_21d = closes_df / closes_df.shift(21) - 1   # ~1 month
    ret_34d = closes_df / closes_df.shift(34) - 1   # ~34 days
    ret_63d = closes_df / closes_df.shift(63) - 1   # ~1 quarter

    # Compute daily breadth metrics for last N days
    daily_records = []
    available_days = min(lookback_days, len(daily_ret) - 1)

    for offset in range(available_days):
        idx = -(offset + 1)  # -1, -2, ..., -N
        date = closes_df.index[idx]

        day_ret = daily_ret.iloc[idx]
        up_4 = int((day_ret > 0.04).sum())
        dn_4 = int((day_ret < -0.04).sum())

        # Monthly (21-day) metrics
        m_ret = ret_21d.iloc[idx] if len(ret_21d) > abs(idx) else pd.Series(dtype=float)
        up_25_month = int((m_ret > 0.25).sum()) if not m_ret.empty else 0
        dn_25_month = int((m_ret < -0.25).sum()) if not m_ret.empty else 0
        up_50_month = int((m_ret > 0.50).sum()) if not m_ret.empty else 0
        dn_50_month = int((m_ret < -0.50).sum()) if not m_ret.empty else 0

        # 34-day metrics
        d34_ret = ret_34d.iloc[idx] if len(ret_34d) > abs(idx) else pd.Series(dtype=float)
        up_13_34d = int((d34_ret > 0.13).sum()) if not d34_ret.empty else 0
        dn_13_34d = int((d34_ret < -0.13).sum()) if not d34_ret.empty else 0

        # Quarterly (63-day) metrics
        q_ret = ret_63d.iloc[idx] if len(ret_63d) > abs(idx) else pd.Series(dtype=float)
        up_25_qtr = int((q_ret > 0.25).sum()) if not q_ret.empty else 0
        dn_25_qtr = int((q_ret < -0.25).sum()) if not q_ret.empty else 0

        daily_records.append({
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "up_4_today": up_4,
            "dn_4_today": dn_4,
            "up_25_qtr": up_25_qtr,
            "dn_25_qtr": dn_25_qtr,
            "up_25_month": up_25_month,
            "dn_25_month": dn_25_month,
            "up_50_month": up_50_month,
            "dn_50_month": dn_50_month,
            "up_13_34d": up_13_34d,
            "dn_13_34d": dn_13_34d,
        })

    # Reverse so oldest first, newest last
    daily_records.reverse()

    # Compute rolling ratios on the daily records
    for i, rec in enumerate(daily_records):
        # 5-day ratio (sum of up_4 over last 5 days / sum of dn_4 over last 5 days)
        if i >= 4:
            window = daily_records[i-4:i+1]
            sum_up = sum(d["up_4_today"] for d in window)
            sum_dn = sum(d["dn_4_today"] for d in window)
            rec["ratio_5d"] = round(sum_up / sum_dn, 2) if sum_dn > 0 else (99.0 if sum_up > 0 else 1.0)
        else:
            rec["ratio_5d"] = None

        # 10-day ratio
        if i >= 9:
            window = daily_records[i-9:i+1]
            sum_up = sum(d["up_4_today"] for d in window)
            sum_dn = sum(d["dn_4_today"] for d in window)
            rec["ratio_10d"] = round(sum_up / sum_dn, 2) if sum_dn > 0 else (99.0 if sum_up > 0 else 1.0)
        else:
            rec["ratio_10d"] = None

    latest = daily_records[-1] if daily_records else {}

    # ── Interpret signals ─────────────────────────────────────
    signals = {}

    # Primary signal: Quarterly momentum pool (up 25% in quarter vs down 25%)
    qtr_up = latest.get("up_25_qtr", 0)
    qtr_dn = latest.get("dn_25_qtr", 0)
    if qtr_up > qtr_dn * 1.2:
        qtr_signal = 1
        qtr_label = f"ON — More stocks rising 25%+ ({qtr_up}) than falling ({qtr_dn})"
    elif qtr_dn > qtr_up * 1.2:
        qtr_signal = -1
        qtr_label = f"OFF — More stocks falling 25%+ ({qtr_dn}) than rising ({qtr_up})"
    else:
        qtr_signal = 0
        qtr_label = f"NEUTRAL — Roughly equal ({qtr_up} up vs {qtr_dn} down)"
    signals["quarterly_momentum"] = {"score": qtr_signal, "detail": qtr_label,
                                     "up": qtr_up, "down": qtr_dn}

    # Daily momentum ratio (5-day)
    ratio_5d = latest.get("ratio_5d")
    if ratio_5d is not None:
        if ratio_5d >= 1.5:
            r5_signal = 1
            r5_label = f"Bullish ({ratio_5d:.2f}x — buyers dominating)"
        elif ratio_5d <= 0.7:
            r5_signal = -1
            r5_label = f"Bearish ({ratio_5d:.2f}x — sellers dominating)"
        else:
            r5_signal = 0
            r5_label = f"Mixed ({ratio_5d:.2f}x)"
    else:
        r5_signal = 0
        r5_label = "Insufficient data"
    signals["daily_momentum_5d"] = {"score": r5_signal, "detail": r5_label,
                                    "value": ratio_5d}

    # 10-day ratio (the key signal per Stockbee)
    ratio_10d = latest.get("ratio_10d")
    if ratio_10d is not None:
        if ratio_10d >= 1.3:
            r10_signal = 1
            r10_label = f"Bullish ({ratio_10d:.2f}x — sustained buying)"
        elif ratio_10d <= 0.8:
            r10_signal = -1
            r10_label = f"Bearish ({ratio_10d:.2f}x — sustained selling)"
        else:
            r10_signal = 0
            r10_label = f"Neutral ({ratio_10d:.2f}x)"
    else:
        r10_signal = 0
        r10_label = "Insufficient data"
    signals["daily_momentum_10d"] = {"score": r10_signal, "detail": r10_label,
                                     "value": ratio_10d}

    # Monthly thrust (up 25%+ in a month)
    m_up = latest.get("up_25_month", 0)
    m_dn = latest.get("dn_25_month", 0)
    if m_up > m_dn and m_up >= 10:
        mt_signal = 1
        mt_label = f"Thrust detected — {m_up} stocks up 25%+ in a month"
    elif m_dn > m_up and m_dn >= 10:
        mt_signal = -1
        mt_label = f"Damage detected — {m_dn} stocks down 25%+ in a month"
    else:
        mt_signal = 0
        mt_label = f"No thrust ({m_up} up, {m_dn} down)"
    signals["monthly_thrust"] = {"score": mt_signal, "detail": mt_label,
                                 "up": m_up, "down": m_dn}

    # ── Generate insights ─────────────────────────────────────
    insights = []

    # Trend in quarterly pool
    if len(daily_records) >= 10:
        qtr_up_10ago = daily_records[-10].get("up_25_qtr", 0)
        qtr_dn_10ago = daily_records[-10].get("dn_25_qtr", 0)
        qtr_up_now = latest.get("up_25_qtr", 0)
        qtr_dn_now = latest.get("dn_25_qtr", 0)

        if qtr_up_now < qtr_up_10ago and qtr_dn_now > qtr_dn_10ago:
            insights.append({
                "type": "warning",
                "text": f"Momentum pool shrinking: stocks up 25%/qtr fell from {qtr_up_10ago} to {qtr_up_now}, "
                        f"while stocks down 25%/qtr grew from {qtr_dn_10ago} to {qtr_dn_now} over 10 days.",
            })
        elif qtr_up_now > qtr_up_10ago and qtr_dn_now < qtr_dn_10ago:
            insights.append({
                "type": "positive",
                "text": f"Momentum pool expanding: stocks up 25%/qtr grew from {qtr_up_10ago} to {qtr_up_now}, "
                        f"while stocks down 25%/qtr fell from {qtr_dn_10ago} to {qtr_dn_now} over 10 days.",
            })

    # 5-day ratio trend
    if len(daily_records) >= 10:
        r5_5ago = daily_records[-6].get("ratio_5d")
        r5_now = latest.get("ratio_5d")
        if r5_5ago is not None and r5_now is not None:
            if r5_now > 1.3 and r5_5ago < 1.0:
                insights.append({
                    "type": "positive",
                    "text": f"5-day ratio flipped bullish: {r5_5ago:.2f} → {r5_now:.2f}. Short-term buyers returning.",
                })
            elif r5_now < 0.7 and r5_5ago > 1.0:
                insights.append({
                    "type": "warning",
                    "text": f"5-day ratio flipped bearish: {r5_5ago:.2f} → {r5_now:.2f}. Short-term sellers taking over.",
                })

    # Extreme readings
    up_4_today = latest.get("up_4_today", 0)
    dn_4_today = latest.get("dn_4_today", 0)
    up_4_pct = up_4_today / universe_size * 100 if universe_size > 0 else 0
    dn_4_pct = dn_4_today / universe_size * 100 if universe_size > 0 else 0

    if up_4_pct > 15:
        insights.append({
            "type": "positive",
            "text": f"Broad thrust day: {up_4_today} stocks ({up_4_pct:.0f}%) up 4%+ today. "
                    "Historically precedes sustained rallies when from oversold conditions.",
        })
    elif dn_4_pct > 15:
        insights.append({
            "type": "warning",
            "text": f"Broad damage day: {dn_4_today} stocks ({dn_4_pct:.0f}%) down 4%+ today. "
                    "Panic selling — watch for capitulation reversal or continued damage.",
        })

    # 34-day medium-term health
    up_13 = latest.get("up_13_34d", 0)
    dn_13 = latest.get("dn_13_34d", 0)
    if dn_13 > up_13 * 2 and dn_13 > 50:
        insights.append({
            "type": "warning",
            "text": f"Medium-term damage spreading: {dn_13} stocks down 13%+ in 34 days vs only {up_13} up. "
                    "Broad-based weakness, not just index.",
        })
    elif up_13 > dn_13 * 2 and up_13 > 50:
        insights.append({
            "type": "positive",
            "text": f"Medium-term breadth healthy: {up_13} stocks up 13%+ in 34 days vs only {dn_13} down. "
                    "Broad participation in rally.",
        })

    # Overall assessment
    bull_signals = sum(1 for s in signals.values() if s["score"] > 0)
    bear_signals = sum(1 for s in signals.values() if s["score"] < 0)
    if bear_signals >= 3:
        assessment = "Market breadth is decisively negative — reduce exposure and size down."
        assessment_type = "danger"
    elif bear_signals >= 2:
        assessment = "Breadth deteriorating — be selective, tighten stops, avoid new positions."
        assessment_type = "warning"
    elif bull_signals >= 3:
        assessment = "Breadth strongly positive — environment supports new positions with full sizing."
        assessment_type = "positive"
    elif bull_signals >= 2:
        assessment = "Breadth improving — cautiously add positions in leaders."
        assessment_type = "positive"
    else:
        assessment = "Breadth mixed — wait for clearer signal before committing capital."
        assessment_type = "neutral"

    insights.insert(0, {"type": assessment_type, "text": assessment})

    return {
        "daily": daily_records,
        "latest": latest,
        "signals": signals,
        "insights": insights,
        "universe_size": universe_size,
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
