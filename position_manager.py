"""
Position Manager — persistent position tracking with trailing stops,
8-week hold rule, climax detection, and trade history.

Stores data in positions.json and trade_history.json in project root.
"""
import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import STOP_CONFIG, PROFIT_CONFIG, POSITION_CONFIG, ALLOCATION_CONFIG, REGIME_POSTURE
from data_fetcher import compute_atr

logger = logging.getLogger(__name__)

POSITIONS_FILE = Path(__file__).parent / "positions.json"
TRADE_HISTORY_FILE = Path(__file__).parent / "trade_history.json"


# ── Load / Save ────────────────────────────────────────────────

def load_positions() -> list[dict]:
    """Load active positions from disk."""
    if not POSITIONS_FILE.exists():
        return []
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load positions: %s", e)
        return []


def save_positions(positions: list[dict]):
    """Save active positions to disk."""
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to save positions: %s", e)


def load_trade_history() -> list[dict]:
    """Load closed trade history from disk."""
    if not TRADE_HISTORY_FILE.exists():
        return []
    try:
        with open(TRADE_HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load trade history: %s", e)
        return []


def save_trade_history(history: list[dict]):
    """Save trade history to disk."""
    try:
        with open(TRADE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to save trade history: %s", e)


# ── Position CRUD ──────────────────────────────────────────────

def add_position(
    ticker: str,
    entry_date: str,
    entry_price: float,
    shares: int,
    initial_stop: float,
    notes: str = "",
    conviction: str = "",
    target_shares: int = 0,
) -> dict:
    """Add a new position. Returns the created position dict."""
    positions = load_positions()

    tranches = []
    if conviction and target_shares > 0:
        tranches = [{"date": entry_date, "price": entry_price, "shares": shares, "label": "Initial"}]

    position = {
        "id": str(uuid.uuid4())[:8],
        "ticker": ticker.upper(),
        "entry_date": entry_date,
        "entry_price": entry_price,
        "shares": shares,
        "initial_stop": initial_stop,
        "trailing_stop": initial_stop,
        "highest_close": entry_price,
        "notes": notes,
        "hold_until": None,
        "conviction": conviction,
        "target_shares": target_shares,
        "tranches": tranches,
        "created_at": datetime.now().isoformat(),
    }

    positions.append(position)
    save_positions(positions)
    return position


def close_position(
    position_id: str,
    exit_date: str,
    exit_price: float,
    reason: str = "",
) -> dict | None:
    """Close a position, move to trade history, compute P&L."""
    positions = load_positions()
    history = load_trade_history()

    target = None
    remaining = []
    for p in positions:
        if p["id"] == position_id:
            target = p
        else:
            remaining.append(p)

    if target is None:
        return None

    # Compute P&L
    entry_price = target["entry_price"]
    shares = target["shares"]
    pnl = (exit_price - entry_price) * shares
    pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0

    # Days held
    try:
        entry_dt = datetime.strptime(target["entry_date"], "%Y-%m-%d")
        exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
        days_held = (exit_dt - entry_dt).days
    except Exception:
        days_held = 0

    trade = {
        **target,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_reason": reason,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "days_held": days_held,
        "closed_at": datetime.now().isoformat(),
    }

    history.append(trade)
    save_positions(remaining)
    save_trade_history(history)
    return trade


# ── Position Analysis ──────────────────────────────────────────

def _compute_trailing_stop(highest_close: float, atr_value: float) -> float:
    """ATR-based trailing stop: highest_close - (multiplier x ATR)."""
    multiplier = STOP_CONFIG["atr_multiple"]
    return round(highest_close - (multiplier * atr_value), 2)


def _check_8_week_hold(position: dict, current_price: float) -> dict | None:
    """Check if 8-week hold rule applies (20%+ gain in <3 weeks)."""
    cfg = PROFIT_CONFIG
    entry_price = position["entry_price"]
    gain_pct = ((current_price / entry_price) - 1) * 100

    try:
        entry_dt = datetime.strptime(position["entry_date"], "%Y-%m-%d")
    except Exception:
        return None

    weeks_held = (datetime.now() - entry_dt).days / 7

    if gain_pct >= cfg["first_target_gain_pct"] and weeks_held < cfg["fast_gain_threshold_weeks"]:
        hold_until = entry_dt + timedelta(weeks=cfg["hold_min_weeks_if_fast"])
        return {
            "active": True,
            "hold_until": hold_until.strftime("%Y-%m-%d"),
            "reason": f"{gain_pct:.0f}% gain in {weeks_held:.1f} weeks — hold 8 weeks",
        }

    if position.get("hold_until"):
        try:
            hold_dt = datetime.strptime(position["hold_until"], "%Y-%m-%d")
            if datetime.now() < hold_dt:
                return {
                    "active": True,
                    "hold_until": position["hold_until"],
                    "reason": f"8-week hold active until {position['hold_until']}",
                }
        except Exception:
            pass

    return None


def _detect_climax(df: pd.DataFrame, entry_date: str) -> bool:
    """Detect climax top: volume >3x avg on the largest up-day since entry."""
    cfg = PROFIT_CONFIG
    try:
        entry_dt = pd.to_datetime(entry_date)
        since_entry = df[df.index >= entry_dt].copy()
        if len(since_entry) < 5:
            return False

        since_entry["daily_return"] = since_entry["Close"].pct_change()
        since_entry["avg_vol"] = since_entry["Volume"].rolling(50, min_periods=10).mean()

        up_days = since_entry[since_entry["daily_return"] > 0]
        if up_days.empty:
            return False

        biggest_up_idx = up_days["daily_return"].idxmax()
        biggest_up = since_entry.loc[biggest_up_idx]

        avg_vol = biggest_up.get("avg_vol", 0)
        if avg_vol and avg_vol > 0:
            vol_ratio = biggest_up["Volume"] / avg_vol
            return vol_ratio >= cfg["climax_volume_multiple"]

    except Exception:
        pass
    return False


def _check_pullback_add(df: pd.DataFrame) -> bool:
    """Check if price pulled back to within 2% of 10 DMA and bounced."""
    try:
        if len(df) < 15:
            return False

        recent = df.iloc[-5:]
        ma10 = df["Close"].rolling(10).mean()

        for i in range(-3, 0):
            close = df["Close"].iloc[i]
            ma_val = ma10.iloc[i]
            if ma_val and ma_val > 0:
                distance_pct = abs((close / ma_val) - 1) * 100
                if distance_pct <= 2:
                    # Check if it bounced (next day higher)
                    if i + 1 < 0 and df["Close"].iloc[i + 1] > close:
                        return True
                    elif i + 1 == 0 and df["Close"].iloc[-1] > close:
                        return True
    except Exception:
        pass
    return False


def compute_pyramid_triggers(position: dict, df: pd.DataFrame) -> list[dict]:
    """Compute pyramid add triggers for an existing position.

    Pyramid rules (Minervini/O'Neil):
    1. Only add when position is profitable (proving the trade is working)
    2. Add on pullbacks to rising 10/21 EMA
    3. Add on breakout from tight consolidation within the trend
    4. Never add more than ALLOCATION_CONFIG max_pyramid_adds

    Returns list of trigger dicts with conditions and suggested action.
    """
    from config import ALLOCATION_CONFIG

    entry_price = position["entry_price"]
    current_price = float(df["Close"].iloc[-1])
    gain_pct = ((current_price / entry_price) - 1) * 100

    tranches = position.get("tranches", [])
    num_adds = len(tranches) - 1  # first tranche is the initial buy
    max_adds = ALLOCATION_CONFIG["max_pyramid_adds"]
    min_gain = ALLOCATION_CONFIG["pyramid_min_gain_pct"]

    triggers = []

    if num_adds >= max_adds:
        return [{"trigger": "MAX_ADDS_REACHED", "active": False,
                 "detail": f"Already at {num_adds} adds (max {max_adds})"}]

    if gain_pct < min_gain:
        return [{"trigger": "NOT_PROFITABLE_ENOUGH", "active": False,
                 "detail": f"Gain {gain_pct:.1f}% < min {min_gain}% for pyramid"}]

    # Trigger 1: Pullback to 10 EMA
    if len(df) >= 15:
        ema10 = df["Close"].ewm(span=10).mean()
        ema_val = float(ema10.iloc[-1])
        distance_to_ema = abs((current_price / ema_val) - 1) * 100

        if distance_to_ema <= 2.0 and current_price > entry_price:
            triggers.append({
                "trigger": "PULLBACK_TO_10EMA",
                "active": True,
                "price_level": round(ema_val, 2),
                "detail": f"Price within {distance_to_ema:.1f}% of 10 EMA ({ema_val:.1f})",
            })

    # Trigger 2: Pullback to 21 EMA
    if len(df) >= 25:
        ema21 = df["Close"].ewm(span=21).mean()
        ema21_val = float(ema21.iloc[-1])
        distance_to_ema21 = abs((current_price / ema21_val) - 1) * 100

        if distance_to_ema21 <= 2.0 and current_price > entry_price:
            triggers.append({
                "trigger": "PULLBACK_TO_21EMA",
                "active": True,
                "price_level": round(ema21_val, 2),
                "detail": f"Price within {distance_to_ema21:.1f}% of 21 EMA ({ema21_val:.1f})",
            })

    # Trigger 3: New base breakout while in trend
    # Check if recent price formed a tight range and is breaking out
    if len(df) >= 30:
        recent_20 = df.tail(20)
        high_20 = recent_20["High"].max()
        low_20 = recent_20["Low"].min()
        range_pct = (high_20 - low_20) / high_20 * 100

        if range_pct <= 10 and current_price >= high_20 * 0.99:
            triggers.append({
                "trigger": "TIGHT_RANGE_BREAKOUT",
                "active": True,
                "price_level": round(float(high_20), 2),
                "detail": f"Breaking out of {range_pct:.1f}% range (last 20 days)",
            })

    if not triggers:
        triggers.append({
            "trigger": "NO_TRIGGER",
            "active": False,
            "detail": f"Position +{gain_pct:.1f}% but no pyramid trigger active",
        })

    return triggers


def get_positions_summary(stock_data: dict) -> list[dict]:
    """Enrich each active position with current price, updated trailing stop,
    and suggested action.

    Args:
        stock_data: Dict of ticker -> DataFrame with OHLCV.

    Returns:
        List of position dicts with added analysis fields.
    """
    positions = load_positions()
    summaries = []

    for pos in positions:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        shares = pos["shares"]

        # Find price data
        df = stock_data.get(ticker)
        if df is None:
            df = stock_data.get(f"{ticker}.NS")
        if df is None or df.empty:
            summaries.append({
                **pos,
                "current_price": None,
                "pnl": 0,
                "pnl_pct": 0,
                "days_held": 0,
                "suggested_action": "NO DATA",
                "action_reason": "Price data unavailable",
            })
            continue

        current_price = df["Close"].iloc[-1]

        # Only consider closes SINCE entry date for highest close
        try:
            entry_dt = pd.to_datetime(pos["entry_date"])
            since_entry = df[df.index >= entry_dt]
            max_close_since_entry = since_entry["Close"].max() if not since_entry.empty else entry_price
        except Exception:
            max_close_since_entry = current_price
        highest_close = max(pos.get("highest_close", entry_price), max_close_since_entry)

        # Update trailing stop
        atr = compute_atr(df)
        atr_val = atr.iloc[-1] if not atr.empty and pd.notna(atr.iloc[-1]) else 0
        new_trailing = _compute_trailing_stop(highest_close, atr_val) if atr_val > 0 else pos["trailing_stop"]
        # Trailing stop only moves up
        trailing_stop = max(new_trailing, pos.get("trailing_stop", pos["initial_stop"]))

        # P&L
        pnl = (current_price - entry_price) * shares
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        # Days held
        try:
            entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_dt).days
        except Exception:
            days_held = 0

        # Determine suggested action
        action = "HOLD"
        reason = "Position within parameters"

        # Check stop loss
        if current_price <= trailing_stop:
            action = "SELL"
            reason = f"Below trailing stop ({trailing_stop:.1f})"
        # Check 8-week hold rule
        elif hold_info := _check_8_week_hold(pos, current_price):
            if hold_info["active"]:
                action = "HOLD"
                reason = hold_info["reason"]
                # Update hold_until in position
                pos["hold_until"] = hold_info["hold_until"]
        # Check climax
        elif _detect_climax(df, pos["entry_date"]):
            action = "PARTIAL SELL"
            reason = f"Climax volume detected — sell {PROFIT_CONFIG['partial_sell_pct']}%"
        # Check pullback add
        elif _check_pullback_add(df):
            action = "ADD"
            reason = "Pullback to 10 DMA with bounce"

        # Update position on disk
        pos["highest_close"] = highest_close
        pos["trailing_stop"] = trailing_stop

        # Pyramid triggers
        pyramid_triggers = compute_pyramid_triggers(pos, df) if df is not None and not df.empty else []

        summaries.append({
            **pos,
            "current_price": round(current_price, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "days_held": days_held,
            "atr": round(atr_val, 2),
            "suggested_action": action,
            "action_reason": reason,
            "pyramid_triggers": pyramid_triggers,
        })

    # Save updated positions (trailing stops, highest close)
    save_positions(positions)
    return summaries


def add_tranche(position_id: str, date: str, price: float, shares: int, label: str = "") -> dict | None:
    """Add a pyramid tranche to an existing position."""
    positions = load_positions()
    for pos in positions:
        if pos["id"] == position_id:
            tranches = pos.get("tranches", [])
            tranches.append({"date": date, "price": price, "shares": shares, "label": label or f"Add #{len(tranches)}"})
            pos["tranches"] = tranches
            pos["shares"] = pos["shares"] + shares
            # Recalculate weighted avg entry price
            total_cost = sum(t["price"] * t["shares"] for t in tranches)
            total_shares = sum(t["shares"] for t in tranches)
            pos["entry_price"] = round(total_cost / total_shares, 2) if total_shares > 0 else pos["entry_price"]
            save_positions(positions)
            return pos
    return None


def calculate_position_size(
    entry_price: float,
    stop_price: float,
    capital: float,
    risk_pct: float,
    max_position_pct: float = 0,
    conviction: str = "",
) -> dict:
    """Calculate position size based on risk and conviction.

    Returns dict with shares, position_value, risk_amount, etc.
    """
    if max_position_pct <= 0:
        max_position_pct = POSITION_CONFIG["max_single_position_pct"]

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return {"shares": 0, "position_value": 0, "risk_amount": 0,
                "risk_pct_of_capital": 0, "position_pct_of_capital": 0,
                "target_shares": 0, "initial_shares": 0, "error": "Stop must be below entry"}

    risk_amount = capital * (risk_pct / 100)
    shares = int(risk_amount / risk_per_share)

    # Cap by max position size
    max_value = capital * (max_position_pct / 100)
    max_shares_by_value = int(max_value / entry_price)
    shares = min(shares, max_shares_by_value)

    # Conviction-based target sizing
    tiers = ALLOCATION_CONFIG["conviction_tiers"]
    pyramid_sizes = ALLOCATION_CONFIG["pyramid_sizes"]
    target_pct = tiers.get(conviction, {}).get("target_pct", max_position_pct)
    target_value = capital * (target_pct / 100)
    target_shares = int(target_value / entry_price)

    # Initial tranche is first pyramid slice of the target
    initial_fraction = pyramid_sizes[0] if pyramid_sizes else 0.5
    initial_shares = min(shares, int(target_shares * initial_fraction))
    if initial_shares < 1:
        initial_shares = 1

    position_value = initial_shares * entry_price
    actual_risk = initial_shares * risk_per_share

    return {
        "shares": initial_shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(actual_risk, 2),
        "risk_pct_of_capital": round(actual_risk / capital * 100, 2) if capital > 0 else 0,
        "position_pct_of_capital": round(position_value / capital * 100, 2) if capital > 0 else 0,
        "target_shares": target_shares,
        "initial_shares": initial_shares,
        "target_pct": target_pct,
        "risk_per_share": round(risk_per_share, 2),
    }


def get_portfolio_heat(summaries: list[dict], capital: float, regime_score: int = 0) -> dict:
    """Calculate total open risk across all positions.

    Returns dict with total risk, risk % of capital, regime limit, utilization.
    """
    posture = REGIME_POSTURE.get(regime_score, REGIME_POSTURE[0])
    max_portfolio_risk_pct = POSITION_CONFIG["max_portfolio_risk_pct"]

    risk_per_position = []
    total_risk = 0.0
    for s in summaries:
        entry = s.get("entry_price", 0)
        stop = s.get("trailing_stop", s.get("initial_stop", 0))
        shares = s.get("shares", 0)
        current = s.get("current_price", entry)
        if current and stop and shares:
            # Risk is from current price to stop, not entry to stop
            risk = max(0, (current - stop)) * shares
            risk_per_position.append({
                "ticker": s.get("ticker", ""),
                "risk": round(risk, 2),
                "risk_pct": round(risk / capital * 100, 2) if capital > 0 else 0,
                "shares": shares,
                "stop": stop,
                "current": current,
            })
            total_risk += risk

    total_risk_pct = round(total_risk / capital * 100, 2) if capital > 0 else 0
    utilization_pct = round(total_risk_pct / max_portfolio_risk_pct * 100, 1) if max_portfolio_risk_pct > 0 else 0

    return {
        "total_risk": round(total_risk, 2),
        "total_risk_pct": total_risk_pct,
        "regime_limit_pct": max_portfolio_risk_pct,
        "regime_label": posture["label"],
        "utilization_pct": min(utilization_pct, 100),
        "risk_per_position": risk_per_position,
    }


def get_trade_stats() -> dict:
    """Compute summary statistics from trade history."""
    history = load_trade_history()
    if not history:
        return {"total_trades": 0, "win_rate": 0, "avg_gain": 0, "avg_loss": 0,
                "total_pnl": 0, "avg_days_held": 0}

    total = len(history)
    winners = [t for t in history if t.get("pnl", 0) > 0]
    losers = [t for t in history if t.get("pnl", 0) <= 0]

    win_rate = len(winners) / total * 100 if total > 0 else 0
    avg_gain = sum(t["pnl_pct"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t["pnl_pct"] for t in losers) / len(losers) if losers else 0
    total_pnl = sum(t.get("pnl", 0) for t in history)
    avg_days = sum(t.get("days_held", 0) for t in history) / total if total > 0 else 0

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "avg_gain": round(avg_gain, 1),
        "avg_loss": round(avg_loss, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_days_held": round(avg_days, 0),
    }
