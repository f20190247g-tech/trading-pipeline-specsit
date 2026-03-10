"""
Layer 4: Weinstein Stage Analysis + Entry Filter
Classifies stocks into Stages 1-4, detects base breakouts, counts bases.
"""
import pandas as pd
import numpy as np
from config import STAGE_CONFIG, STOP_CONFIG, WEEKLY_STAGE_CONFIG
from data_fetcher import compute_atr


def classify_stage(df: pd.DataFrame) -> dict:
    """
    Classify a stock into Weinstein Stage 1-4.

    Stage 1 (Basing): Price sideways around flat 150 MA
    Stage 2 (Advancing): Price > rising 150 MA, MAs aligned bullishly
    Stage 3 (Topping): Price churning around flattening 150 MA
    Stage 4 (Declining): Price < falling 150 MA

    Returns dict with stage number, confidence, and details.
    """
    cfg = STAGE_CONFIG

    if len(df) < cfg["ma_long"] + 30:
        return {"stage": 0, "confidence": 0, "detail": "Insufficient data"}

    close = df["Close"]
    ma50 = close.rolling(cfg["ma_short"]).mean()
    ma150 = close.rolling(cfg["ma_mid"]).mean()
    ma200 = close.rolling(cfg["ma_long"]).mean()

    latest = close.iloc[-1]
    latest_ma50 = ma50.iloc[-1]
    latest_ma150 = ma150.iloc[-1]
    latest_ma200 = ma200.iloc[-1]

    # MA slopes (over last 20 days)
    ma150_slope = (ma150.iloc[-1] - ma150.iloc[-20]) / ma150.iloc[-20] * 100
    ma200_slope = (ma200.iloc[-1] - ma200.iloc[-20]) / ma200.iloc[-20] * 100

    # 52-week high/low
    high_52w = df["High"].tail(252).max() if len(df) >= 252 else df["High"].max()
    low_52w = df["Low"].tail(252).min() if len(df) >= 252 else df["Low"].min()
    pct_above_52w_low = (latest / low_52w - 1) * 100
    pct_below_52w_high = (1 - latest / high_52w) * 100

    # ── Stage 2 Criteria (Weinstein + Minervini) ────────────────
    s2_checks = {
        "price_above_ma150": latest > latest_ma150,
        "price_above_ma200": latest > latest_ma200,
        "ma50_above_ma150": latest_ma50 > latest_ma150,
        "ma150_above_ma200": latest_ma150 > latest_ma200,
        "ma200_rising": ma200_slope > 0,
        "above_52w_low_30pct": pct_above_52w_low >= cfg["price_above_52w_low_pct"],
        "within_52w_high_25pct": pct_below_52w_high <= cfg["price_within_52w_high_pct"],
    }
    s2_score = sum(s2_checks.values())

    # ── Stage Classification ────────────────────────────────────
    if s2_score >= 6:
        stage = 2
        confidence = s2_score / 7
    elif s2_score >= 4 and ma150_slope > -0.5:
        # Transitioning — could be late Stage 1 entering Stage 2
        # or early Stage 3
        if latest > latest_ma150 and ma150_slope > 0:
            stage = 2
            confidence = s2_score / 7
        else:
            stage = 1  # basing
            confidence = 0.6
    elif latest < latest_ma150 and latest < latest_ma200 and ma150_slope < 0:
        stage = 4  # declining
        confidence = 0.8 if ma200_slope < 0 else 0.6
    elif latest > latest_ma200 and ma150_slope < 0.3 and ma150_slope > -0.3:
        stage = 3  # topping
        confidence = 0.6
    else:
        # Default to stage 1 (basing) if unclear
        stage = 1
        confidence = 0.4

    return {
        "stage": stage,
        "confidence": round(confidence, 2),
        "s2_checks": s2_checks,
        "s2_score": s2_score,
        "ma_slopes": {"ma150": round(ma150_slope, 2), "ma200": round(ma200_slope, 2)},
        "detail": f"Stage {stage} (confidence {confidence:.0%})",
    }


def detect_bases(df: pd.DataFrame) -> list[dict]:
    """
    Detect consolidation bases (flat price ranges with declining volume).

    A base is identified as a period where:
    - Price range (high-low) is within X% (the base depth)
    - Duration is between min and max days
    - Volume tends to contract

    Returns list of bases with their properties.
    """
    cfg = STAGE_CONFIG
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    bases = []
    i = len(df) - 1  # start from most recent

    while i > cfg["base_min_days"]:
        # Look backwards to find a consolidation range
        base_high = high.iloc[i]
        base_low = low.iloc[i]

        j = i - 1
        while j >= max(0, i - cfg["base_max_days"]):
            # Expand the range
            base_high = max(base_high, high.iloc[j])
            base_low = min(base_low, low.iloc[j])

            depth = (base_high - base_low) / base_high * 100
            if depth > cfg["base_max_depth_pct"]:
                break

            j -= 1

        base_length = i - j
        depth = (base_high - base_low) / base_high * 100

        if (
            cfg["base_min_days"] <= base_length <= cfg["base_max_days"]
            and depth <= cfg["base_max_depth_pct"]
        ):
            # Check for volume contraction within base
            base_vol = volume.iloc[j:i+1]
            first_half_vol = base_vol.iloc[:len(base_vol)//2].mean()
            second_half_vol = base_vol.iloc[len(base_vol)//2:].mean()
            vol_contracting = second_half_vol < first_half_vol

            bases.append({
                "start_idx": j,
                "end_idx": i,
                "start_date": str(df.index[j].date()) if hasattr(df.index[j], 'date') else str(df.index[j]),
                "end_date": str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i]),
                "length_days": base_length,
                "base_high": round(float(base_high), 2),
                "base_low": round(float(base_low), 2),
                "depth_pct": round(depth, 1),
                "volume_contracting": vol_contracting,
            })

            # Jump past this base to look for earlier ones
            i = j - 5
        else:
            i -= 10  # step back and try again

    bases.reverse()  # chronological order
    return bases


def count_bases_in_stage2(df: pd.DataFrame, bases: list[dict]) -> int:
    """Count how many base breakouts have occurred in the current Stage 2 advance."""
    if not bases:
        return 0

    ma150 = df["Close"].rolling(150).mean()
    count = 0
    for base in bases:
        idx = base["end_idx"]
        if idx < len(ma150) and df["Close"].iloc[idx] > ma150.iloc[idx]:
            count += 1
    return count


def detect_breakout(df: pd.DataFrame, bases: list[dict]) -> dict | None:
    """
    Check if the most recent price action is a breakout from the latest base.

    Breakout = price closes above base_high on volume >= 1.5x average.
    """
    cfg = STAGE_CONFIG
    if not bases:
        return None

    latest_base = bases[-1]
    base_high = latest_base["base_high"]

    # Check last 5 trading days for a breakout
    recent = df.tail(5)
    avg_vol = df["Volume"].tail(50).mean()

    for idx in range(len(recent)):
        row = recent.iloc[idx]
        if (
            row["Close"] > base_high
            and row["Volume"] >= avg_vol * cfg["volume_surge_multiple"]
        ):
            return {
                "breakout": True,
                "breakout_date": str(recent.index[idx].date()) if hasattr(recent.index[idx], 'date') else str(recent.index[idx]),
                "breakout_price": round(float(row["Close"]), 2),
                "base_high": base_high,
                "volume_ratio": round(float(row["Volume"] / avg_vol), 1),
                "base_depth_pct": latest_base["depth_pct"],
                "base_length_days": latest_base["length_days"],
            }

    return None


def detect_vcp(df: pd.DataFrame, base: dict) -> dict:
    """
    Detect Volatility Contraction Pattern within a base.
    VCP = successive tightenings of the price range.
    """
    cfg = STAGE_CONFIG
    start = base["start_idx"]
    end = base["end_idx"]

    if end - start < 20:
        return {"is_vcp": False, "contractions": 0}

    # Divide the base into segments and measure range of each
    segment_len = max(5, (end - start) // 4)
    ranges = []
    for k in range(start, end, segment_len):
        segment = df.iloc[k:k + segment_len]
        if len(segment) > 2:
            r = (segment["High"].max() - segment["Low"].min()) / segment["High"].max() * 100
            ranges.append(r)

    if len(ranges) < 2:
        return {"is_vcp": False, "contractions": 0}

    # Count contractions (each range smaller than the previous)
    contractions = 0
    for k in range(1, len(ranges)):
        if ranges[k] < ranges[k - 1] * cfg["vcp_contraction_ratio"]:
            contractions += 1

    return {
        "is_vcp": contractions >= cfg["vcp_contractions_min"],
        "contractions": contractions,
        "ranges": [round(r, 1) for r in ranges],
    }


def compute_entry_and_stop(
    df: pd.DataFrame,
    breakout: dict,
    base: dict,
) -> dict:
    """
    Compute entry price, initial stop loss, and risk per share.
    """
    entry_price = breakout["breakout_price"]

    # Initial stop: just below base low
    buffer_pct = STOP_CONFIG["initial_stop_buffer_pct"]
    stop_loss = round(base["base_low"] * (1 - buffer_pct / 100), 2)

    # ATR-based stop alternative
    atr = compute_atr(df, STOP_CONFIG["atr_period"])
    atr_val = float(atr.iloc[-1]) if len(atr.dropna()) > 0 else 0
    atr_stop = round(entry_price - STOP_CONFIG["atr_multiple"] * atr_val, 2)

    # Use the tighter of the two (higher stop = less risk)
    effective_stop = max(stop_loss, atr_stop)
    risk_per_share = round(entry_price - effective_stop, 2)
    risk_pct = round(risk_per_share / entry_price * 100, 2)

    return {
        "entry_price": entry_price,
        "base_stop": stop_loss,
        "atr_stop": atr_stop,
        "effective_stop": effective_stop,
        "risk_per_share": risk_per_share,
        "risk_pct": risk_pct,
        "atr": round(atr_val, 2),
    }


def confirm_weekly_stage(daily_df: pd.DataFrame, daily_stage_result: dict) -> dict:
    """Confirm stage classification using weekly data (Weinstein's preferred timeframe).

    Resamples daily to weekly, checks 30-week MA alignment.
    Returns confirmation dict merged into stage result.
    """
    cfg_w = WEEKLY_STAGE_CONFIG

    # Resample daily to weekly OHLCV
    weekly = daily_df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    if len(weekly) < cfg_w["weekly_ma_period"] + 5:
        return {"weekly_confirmed": None, "weekly_detail": "Insufficient weekly data"}

    close_w = weekly["Close"]
    ma30w = close_w.rolling(cfg_w["weekly_ma_period"]).mean()

    latest_close = close_w.iloc[-1]
    latest_ma = ma30w.iloc[-1]

    # Check if 30-week MA is rising for N weeks
    ma_rising_count = 0
    for i in range(1, min(len(ma30w), 20)):
        if pd.notna(ma30w.iloc[-i]) and pd.notna(ma30w.iloc[-i-1]):
            if ma30w.iloc[-i] > ma30w.iloc[-i-1]:
                ma_rising_count += 1
            else:
                break
        else:
            break

    price_above_ma = latest_close > latest_ma if pd.notna(latest_ma) else False
    ma_rising = ma_rising_count >= cfg_w["weekly_ma_rising_weeks"]

    # Weekly MA slope (%)
    if pd.notna(ma30w.iloc[-1]) and pd.notna(ma30w.iloc[-5]) and ma30w.iloc[-5] != 0:
        weekly_slope = (ma30w.iloc[-1] - ma30w.iloc[-5]) / ma30w.iloc[-5] * 100
    else:
        weekly_slope = 0

    # Confirm Stage 2 on weekly
    weekly_s2 = price_above_ma and ma_rising

    # Distance from 30-week MA (for early/mid/late Stage 2)
    if pd.notna(latest_ma) and latest_ma > 0:
        pct_above_ma = (latest_close / latest_ma - 1) * 100
    else:
        pct_above_ma = 0

    if pct_above_ma <= 5:
        s2_substage = "Early"
    elif pct_above_ma <= 15:
        s2_substage = "Mid"
    else:
        s2_substage = "Late"

    return {
        "weekly_confirmed": weekly_s2,
        "weekly_price_above_30wma": price_above_ma,
        "weekly_ma_rising": ma_rising,
        "weekly_ma_rising_weeks": ma_rising_count,
        "weekly_ma_slope": round(weekly_slope, 2),
        "weekly_pct_above_ma": round(pct_above_ma, 1),
        "weekly_s2_substage": s2_substage if daily_stage_result.get("stage") == 2 else None,
        "weekly_detail": f"{'Confirmed' if weekly_s2 else 'Not confirmed'} on weekly ({s2_substage} S2, {pct_above_ma:.1f}% above 30W MA)",
    }


def compute_consolidation_quality(df: pd.DataFrame, base: dict) -> dict:
    """Grade consolidation tightness (Minervini's SEPA precision layer).

    Measures:
    1. Range compression: how tight is the price range relative to base high?
    2. Volume dry-up: is volume declining in the consolidation?
    3. Daily range progression: are daily ranges contracting?

    Returns quality grade and score.
    """
    start = base["start_idx"]
    end = base["end_idx"]
    base_data = df.iloc[start:end+1]

    if len(base_data) < 10:
        return {"quality_grade": "C", "quality_score": 30, "detail": "Base too short"}

    # 1. Range compression (base depth as % of base high)
    depth_pct = base.get("depth_pct", 35)

    # 2. Volume dry-up: compare last third volume to first third
    n = len(base_data)
    third = max(3, n // 3)
    first_third_vol = base_data["Volume"].iloc[:third].mean()
    last_third_vol = base_data["Volume"].iloc[-third:].mean()

    if first_third_vol > 0:
        vol_dryup_pct = (1 - last_third_vol / first_third_vol) * 100
    else:
        vol_dryup_pct = 0

    # 3. Daily range progression (are ranges getting tighter?)
    daily_ranges = ((base_data["High"] - base_data["Low"]) / base_data["Close"] * 100)
    first_half_range = daily_ranges.iloc[:n//2].mean()
    second_half_range = daily_ranges.iloc[n//2:].mean()

    if first_half_range > 0:
        range_compression_pct = (1 - second_half_range / first_half_range) * 100
    else:
        range_compression_pct = 0

    final_range_pct = daily_ranges.iloc[-5:].mean() if len(daily_ranges) >= 5 else daily_ranges.mean()

    # 4. Compute quality grade
    # Textbook: tight range + volume dry-up + range compression
    tight_checks = 0
    if depth_pct <= 15:
        tight_checks += 2  # very tight base
    elif depth_pct <= 25:
        tight_checks += 1  # normal tightness

    if vol_dryup_pct >= 40:
        tight_checks += 2  # strong volume dry-up
    elif vol_dryup_pct >= 20:
        tight_checks += 1

    if range_compression_pct >= 40:
        tight_checks += 2  # daily ranges compressing
    elif range_compression_pct >= 20:
        tight_checks += 1

    if final_range_pct <= 1.5:
        tight_checks += 1  # very tight final range

    if tight_checks >= 6:
        grade = "A+"
        score = 95
    elif tight_checks >= 5:
        grade = "A"
        score = 85
    elif tight_checks >= 3:
        grade = "B"
        score = 65
    elif tight_checks >= 2:
        grade = "C"
        score = 45
    else:
        grade = "D"
        score = 25

    return {
        "quality_grade": grade,
        "quality_score": score,
        "depth_pct": round(depth_pct, 1),
        "vol_dryup_pct": round(vol_dryup_pct, 1),
        "range_compression_pct": round(range_compression_pct, 1),
        "final_range_pct": round(final_range_pct, 2),
        "tight_checks": tight_checks,
        "detail": f"Grade {grade} (depth {depth_pct:.0f}%, vol dryup {vol_dryup_pct:.0f}%, compression {range_compression_pct:.0f}%)",
    }


def number_bases_in_stage2(df: pd.DataFrame, bases: list[dict]) -> list[dict]:
    """Number each base within the current Stage 2 advance.

    Walks bases forward from the start of Stage 2 (when price first crossed
    above rising 150 MA). Each base gets a 'base_number' field (1, 2, 3...).

    Minervini insight: 1st-2nd base = best R:R, 3rd = ok, 4th+ = late/risky.
    """
    if not bases or len(df) < 150:
        return bases

    ma150 = df["Close"].rolling(150).mean()

    # Find where the current Stage 2 began
    # Walk backwards from most recent — find last time price crossed above rising 150 MA
    s2_start_idx = 0
    for i in range(len(df) - 1, 150, -1):
        if pd.notna(ma150.iloc[i]) and df["Close"].iloc[i] < ma150.iloc[i]:
            s2_start_idx = i + 1
            break

    # Number bases that occurred AFTER the Stage 2 start
    base_num = 0
    for base in bases:
        if base["end_idx"] >= s2_start_idx:
            base_num += 1
            base["base_number"] = base_num
            base["base_risk"] = "low" if base_num <= 2 else ("medium" if base_num == 3 else "high")
        else:
            base["base_number"] = 0  # pre-Stage 2
            base["base_risk"] = "n/a"

    return bases


def detect_stage_transitions(df: pd.DataFrame, lookback: int = 60) -> list[dict]:
    """Detect stage transitions in recent price action.

    Checks rolling windows within the lookback period to find where
    the stock's stage classification changed.

    Key transitions:
    - 1->2: Best entry (basing to advancing) — BULLISH
    - 2->3: Warning (advancing to topping) — CAUTION
    - 3->4: Exit signal (topping to declining) — BEARISH
    - 4->1: Early accumulation — WATCH
    """
    if len(df) < 250:
        return []

    transitions = []
    check_points = [5, 10, 20, 40, 60]  # days ago to check
    check_points = [p for p in check_points if p <= lookback and p < len(df) - 200]

    if not check_points:
        return []

    current_stage = classify_stage(df)["stage"]

    for days_ago in check_points:
        past_df = df.iloc[:len(df) - days_ago]
        if len(past_df) < 230:
            continue
        past_stage = classify_stage(past_df)["stage"]

        if past_stage != current_stage:
            transition_type = f"{past_stage}\u2192{current_stage}"
            signal_map = {
                "1\u21922": "BULLISH", "4\u21921": "WATCH", "4\u21922": "BULLISH",
                "2\u21923": "CAUTION", "3\u21924": "BEARISH", "2\u21921": "CAUTION",
                "1\u21924": "BEARISH", "3\u21922": "BULLISH", "3\u21921": "WATCH",
            }
            signal = signal_map.get(transition_type, "NEUTRAL")

            transitions.append({
                "from_stage": past_stage,
                "to_stage": current_stage,
                "transition": transition_type,
                "signal": signal,
                "days_ago": days_ago,
                "approx_date": str(df.index[-days_ago].date()) if hasattr(df.index[-days_ago], 'date') else str(df.index[-days_ago]),
            })
            break  # only report the most recent transition

    return transitions


def analyze_stock_stage(df: pd.DataFrame, ticker: str) -> dict:
    """
    Full stage analysis for a single stock.
    Returns comprehensive analysis dict.
    """
    stage = classify_stage(df)
    weekly = confirm_weekly_stage(df, stage)
    bases = detect_bases(df)
    bases = number_bases_in_stage2(df, bases)
    transitions = detect_stage_transitions(df)
    base_count = count_bases_in_stage2(df, bases)
    breakout = detect_breakout(df, bases) if bases else None

    result = {
        "ticker": ticker,
        "stage": stage,
        "bases_found": len(bases),
        "base_count_in_stage2": base_count,
        "breakout": breakout,
        "entry_setup": None,
        "vcp": None,
        "weekly": weekly,
        "consolidation": None,
        "transitions": transitions,
    }

    # Only generate entry if Stage 2 with a valid breakout
    if stage["stage"] == 2 and breakout and base_count <= STAGE_CONFIG["max_base_count"]:
        latest_base = bases[-1]
        vcp = detect_vcp(df, latest_base)
        consolidation = compute_consolidation_quality(df, latest_base)
        entry_stop = compute_entry_and_stop(df, breakout, latest_base)

        result["entry_setup"] = entry_stop
        result["vcp"] = vcp
        result["consolidation"] = consolidation

    return result


def filter_stage2_candidates(
    stock_data: dict[str, pd.DataFrame],
    screened_stocks: list[dict],
) -> list[dict]:
    """
    Apply stage analysis to all screened stocks.
    Returns only Stage 2 stocks with valid breakout setups.
    """
    candidates = []

    for stock_info in screened_stocks:
        ticker = stock_info["ticker"]
        if ticker not in stock_data:
            continue

        df = stock_data[ticker]
        analysis = analyze_stock_stage(df, ticker)

        if (
            analysis["stage"]["stage"] == 2
            and analysis["stage"]["confidence"] >= 0.7
            and analysis["base_count_in_stage2"] <= STAGE_CONFIG["max_base_count"]
        ):
            # Merge screening data with stage analysis
            combined = {**stock_info, **analysis}
            candidates.append(combined)

    # Sort by: has breakout first, then by leadership score
    candidates.sort(
        key=lambda x: (
            1 if x["breakout"] else 0,
            x.get("leadership_score", 0),
        ),
        reverse=True,
    )
    return candidates


def scan_all_stages(
    stock_data: dict[str, pd.DataFrame],
    min_s2_score: int = 4,
) -> list[dict]:
    """Run stage classification on ALL stocks in the universe.

    Unlike filter_stage2_candidates() which only looks at screened stocks,
    this scans every stock and returns anything that scores >= min_s2_score
    on the 7-point Stage 2 checklist.

    Returns list of dicts sorted by s2_score desc, then breakout status.
    Each dict has: ticker, stage, bases_found, base_count_in_stage2,
    breakout, entry_setup, vcp.
    """
    from data_fetcher import get_sector_for_stock

    results = []
    for ticker, df in stock_data.items():
        if len(df) < 230:  # need 200 MA + buffer
            continue

        stage_info = classify_stage(df)
        s2_score = stage_info.get("s2_score", 0)

        if s2_score < min_s2_score:
            continue

        bases = detect_bases(df)
        bases = number_bases_in_stage2(df, bases)
        base_count = count_bases_in_stage2(df, bases)
        breakout = detect_breakout(df, bases) if bases else None
        weekly = confirm_weekly_stage(df, stage_info)
        vcp = None
        entry_setup = None
        consolidation = None

        if stage_info["stage"] == 2 and breakout and base_count <= STAGE_CONFIG["max_base_count"]:
            if bases:
                vcp = detect_vcp(df, bases[-1])
                entry_setup = compute_entry_and_stop(df, breakout, bases[-1])
                consolidation = compute_consolidation_quality(df, bases[-1])

        sector = get_sector_for_stock(ticker)

        results.append({
            "ticker": ticker,
            "sector": sector,
            "stage": stage_info,
            "bases_found": len(bases),
            "base_count_in_stage2": base_count,
            "breakout": breakout,
            "entry_setup": entry_setup,
            "vcp": vcp,
            "weekly": weekly,
            "consolidation": consolidation,
            "close": round(float(df["Close"].iloc[-1]), 2),
        })

    results.sort(
        key=lambda x: (
            x["stage"].get("s2_score", 0),
            1 if x["breakout"] else 0,
        ),
        reverse=True,
    )
    return results


def print_stage_results(candidates: list[dict], max_show: int = 20) -> None:
    """Pretty-print Stage 2 candidates."""
    print("\n" + "=" * 100)
    print("  LAYER 4: STAGE 2 CANDIDATES")
    print("=" * 100)

    breakout_candidates = [c for c in candidates if c.get("breakout")]
    watchlist_candidates = [c for c in candidates if not c.get("breakout")]

    if breakout_candidates:
        print(f"\n  ACTIVE BREAKOUTS ({len(breakout_candidates)}):")
        print(
            f"  {'Ticker':<14} {'Stage':>6} {'Bases':>6} {'Entry':>8} {'Stop':>8} "
            f"{'Risk%':>6} {'VolRatio':>9} {'VCP':>5} {'Score':>7}"
        )
        print("  " + "-" * 87)

        for c in breakout_candidates[:max_show]:
            bo = c["breakout"]
            es = c.get("entry_setup", {})
            vcp = c.get("vcp", {})
            print(
                f"  {c['ticker']:<14} "
                f"{'S2':>6} "
                f"{c['base_count_in_stage2']:>6} "
                f"{es.get('entry_price', 'N/A'):>8} "
                f"{es.get('effective_stop', 'N/A'):>8} "
                f"{es.get('risk_pct', 'N/A'):>5}% "
                f"{bo.get('volume_ratio', 'N/A'):>8}x "
                f"{'Y' if vcp and vcp.get('is_vcp') else 'N':>5} "
                f"{c.get('leadership_score', 0):>7.2f}"
            )

    if watchlist_candidates:
        print(f"\n  WATCHLIST — Stage 2, no breakout yet ({len(watchlist_candidates)}):")
        print(f"  {'Ticker':<14} {'Conf':>6} {'Bases':>6} {'%frHigh':>8} {'RS/Nifty':>9} {'Score':>7}")
        print("  " + "-" * 58)

        for c in watchlist_candidates[:max_show]:
            print(
                f"  {c['ticker']:<14} "
                f"{c['stage']['confidence']:>5.0%} "
                f"{c['base_count_in_stage2']:>6} "
                f"{c.get('dist_from_high_pct', 0):>7.1f}% "
                f"{c.get('rs_vs_nifty', 0):>9.2f} "
                f"{c.get('leadership_score', 0):>7.2f}"
            )

    print(f"\n  Total Stage 2 candidates: {len(candidates)}")
    print()
