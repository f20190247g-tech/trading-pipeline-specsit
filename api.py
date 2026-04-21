"""
SDWB Live Data Bridge — FastAPI server
Reads pipeline cache files and serves JSON to the HTML dashboard.

Install:
    pip install fastapi uvicorn

Run (from repo root):
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Then open SDWB Dashboard.html — it will show "LIVE" in the topbar.
"""
from __future__ import annotations
import datetime as dt
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(title="SDWB Data Bridge", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache file locations (mirrors app.py) ─────────────────────
CACHE_DIR     = Path(__file__).parent / "scan_cache"
WEEKLY_CACHE  = CACHE_DIR / "last_weekly_scan.pkl"
DAILY_CACHE   = CACHE_DIR / "last_daily_check.pkl"
MONTHLY_CACHE = CACHE_DIR / "last_monthly_data.pkl"
LEGACY_CACHE  = CACHE_DIR / "last_scan.pkl"

# ── Serialisation helpers ──────────────────────────────────────
def _safe(v):
    """Recursively convert numpy / pandas types to JSON-safe Python."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if (math.isnan(float(v)) or math.isinf(float(v))) else float(v)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    if isinstance(v, pd.Series):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, pd.DataFrame):
        return v.reset_index().applymap(lambda x: None if (isinstance(x, float) and math.isnan(x)) else x).to_dict(orient="records")
    if isinstance(v, dt.datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    return v


def _flt(v, default=0.0) -> float:
    try:
        f = float(v or default)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _int(v, default=0) -> int:
    try:
        return int(v or default)
    except Exception:
        return default


def _load_pickle(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _load_all() -> dict:
    """Merge all cache layers (weekly → daily → monthly overlays)."""
    data: dict = {}
    for path in [LEGACY_CACHE, WEEKLY_CACHE]:
        data.update(_load_pickle(path))
    data.update(_load_pickle(DAILY_CACHE))
    data.update(_load_pickle(MONTHLY_CACHE))
    return data


# ── Regime transformer ─────────────────────────────────────────
def _build_regime(regime_raw: dict) -> dict:
    signals_raw = regime_raw.get("signals", {})
    sig_list = []
    for name, sig in signals_raw.items():
        if isinstance(sig, dict):
            sig_list.append({
                "name": name.replace("_", " ").title(),
                "score": _int(sig.get("score", 0)),
                "detail": str(sig.get("detail", "")),
            })

    bullish = sum(1 for s in sig_list if s["score"] > 0)
    label   = regime_raw.get("label", "NORMAL")
    score   = _int(regime_raw.get("regime_score", regime_raw.get("raw_score", 0)))

    # Map momentum from breadth_trend string
    bt = str(regime_raw.get("breadth_trend", "stable")).lower()
    mom_map = {"improving": "Improving", "deteriorating": "Fading",
               "stable": "Mixed", "rising": "Improving", "falling": "Fading"}
    momentum = mom_map.get(bt, "Mixed")

    return {
        "label":         label,
        "score":         score,
        "color":         "#4895ef",
        "bullish":       bullish,
        "total":         len(sig_list),
        "momentum":      momentum,
        "breadth_trend": bt.capitalize(),
        "signals":       sig_list[:5],
        "derivatives":   [],   # populated separately if available
    }


# ── Nifty transformer ──────────────────────────────────────────
def _build_nifty(nifty_df) -> dict:
    if nifty_df is None or len(nifty_df) < 210:
        return {}
    try:
        close = nifty_df["Close"].dropna()
        return {
            "current": round(_flt(close.iloc[-1]), 2),
            "prev":    round(_flt(close.iloc[-2]), 2),
            "ma50":    round(_flt(close.rolling(50).mean().iloc[-1]), 2),
            "ma200":   round(_flt(close.rolling(200).mean().iloc[-1]), 2),
            "high52":  round(_flt(close.tail(252).max()), 2),
            "low52":   round(_flt(close.tail(252).min()), 2),
        }
    except Exception:
        return {}


# ── Breadth transformer ────────────────────────────────────────
def _build_breadth(brs: dict, signals_raw: dict) -> dict:
    def _breadth_val(key):
        sig = signals_raw.get(key, {})
        if isinstance(sig, dict):
            return round(_flt(sig.get("value", sig.get("val", 50))), 1)
        return 50.0

    sp = brs.get("stage_pcts", {}) if brs else {}
    sc = brs.get("stage_counts", {}) if brs else {}
    return {
        "above50":  _breadth_val("breadth_50dma"),
        "above200": _breadth_val("breadth_200dma"),
        "stage1":   round(_flt(sp.get(1, sp.get("1", 25))), 1),
        "stage2":   round(_flt(sp.get(2, sp.get("2", 35))), 1),
        "stage3":   round(_flt(sp.get(3, sp.get("3", 20))), 1),
        "stage4":   round(_flt(sp.get(4, sp.get("4", 20))), 1),
        "s2_count": _int(sc.get(2, sc.get("2", 0))),
        "s2_bo":    _int(brs.get("s2_with_breakouts", 0)) if brs else 0,
        "score":    _int(brs.get("breadth_score", 50)) if brs else 50,
    }


# ── Sector transformer ─────────────────────────────────────────
def _sector_signal(rs: float, trend: str) -> tuple[str, str]:
    t = trend.lower()
    if rs > 0.5 and t in ("rising", "uptrend", "↑"):
        return "Bullish Thrust", "#00c87e"
    if rs > 0 and t in ("flat", "→"):
        return "Pullback Slowing", "#f0a500"
    if rs <= 0 and t in ("rising", "uptrend", "↑"):
        return "Bullish Inflection", "#f0a500"
    if rs < -0.5 and t in ("falling", "downtrend", "↓"):
        return "Bearish Breakdown", "#f03e50"
    if rs > 0 and t in ("falling", "downtrend", "↓"):
        return "Bearish Inflection", "#f03e50"
    return "Momentum Mixed", "#5a7090"


def _build_sectors(sector_rankings: list, top_sectors: list) -> list:
    out = []
    for s in sector_rankings:
        mom  = s.get("momentum", {}) or {}
        rs   = _flt(s.get("mansfield_rs", 0))
        trend_raw = str(s.get("rs_trend", "Flat"))
        arrow = "↑" if trend_raw.lower() in ("rising", "uptrend") else \
                "↓" if trend_raw.lower() in ("falling", "downtrend") else "→"
        signal, sig_color = _sector_signal(rs, trend_raw)
        score = _flt(s.get("composite_score", 50))
        name  = s.get("sector", s.get("name", ""))

        sector_stage = s.get("sector_stage", {})
        if isinstance(sector_stage, dict):
            stg = sector_stage.get("substage") or \
                  (f"S{sector_stage.get('stage','')}" if sector_stage.get("stage") else "—")
        else:
            stg = f"S{sector_stage}" if sector_stage else "—"

        out.append({
            "name":     name,
            "rs":       round(rs, 2),
            "trend":    arrow,
            "w1":       round(_flt(mom.get("1w", 0)), 1),
            "w2":       round(_flt(mom.get("2w", 0)), 1),
            "m1":       round(_flt(mom.get("1m", 0)), 1),
            "m3":       round(_flt(mom.get("3m", 0)), 1),
            "m6":       round(_flt(mom.get("6m", 0)), 1),
            "signal":   signal,
            "sigColor": sig_color,
            "score":    round(score),
            "top":      name in top_sectors,
            "stage":    stg,
        })
    return out


# ── Stock transformer ──────────────────────────────────────────
_SIG_ORDER = {
    "BUY — Breakout": 0, "BUY — Setup Ready": 1,
    "WATCH — VCP Forming": 2, "WATCH — In Base": 3, "WATCH — Improving": 4,
    "WAIT — Far From High": 5, "MONITOR": 6,
}

def _build_stocks(screened: list, stage2_candidates: list, watchlist: list) -> list:
    s2_map = {c["ticker"]: c for c in stage2_candidates if c.get("ticker")}
    wl_map = {w.get("ticker", ""): w for w in watchlist}
    out = []

    for s in screened:
        ticker = s.get("ticker", "")
        rs1m  = _flt(s.get("rs_1m", s.get("rs_vs_nifty", 0)))
        rs3m  = _flt(s.get("rs_3m", 0))
        rs6m  = _flt(s.get("rs_vs_nifty", 0))
        dist  = _flt(s.get("dist_from_high_pct", 0))
        accum = _flt(s.get("accumulation_ratio", 1.0), 1.0)
        close = _flt(s.get("close", 0))

        s2d   = s2_map.get(ticker, {})
        s2_sc = _int((s2d.get("stage") or {}).get("s2_score", 0))
        bo    = (s2d.get("breakout") or {})
        vcp   = (s2d.get("vcp") or {})
        wld   = wl_map.get(ticker, {})

        # Signal
        if wld.get("action") == "BUY":
            signal = "BUY — Breakout" if bo.get("breakout") else "BUY — Setup Ready"
        elif bo.get("breakout"):
            signal = "BUY — Breakout"
        elif s2_sc >= 6 and vcp.get("is_vcp"):
            signal = "WATCH — VCP Forming"
        elif s2_sc >= 5:
            signal = "WATCH — In Base"
        elif dist > 15:
            signal = "WAIT — Far From High"
        elif rs1m > 0 and accum > 1.1:
            signal = "WATCH — Improving"
        else:
            signal = "MONITOR"

        # Trend label
        if rs1m > rs3m > 0:
            trend = "Accelerating"
        elif rs1m > 0 >= rs3m:
            trend = "Turning Up"
        elif rs1m > 0 and rs3m > rs1m:
            trend = "Strong, Slowing"
        elif rs1m <= 0 < rs3m:
            trend = "Turning Down"
        elif rs1m <= 0 and rs3m <= 0:
            trend = "Weak"
        else:
            trend = "Mixed"

        out.append({
            "ticker": ticker.replace(".NS", ""),
            "sector": s.get("sector", ""),
            "rs1m":   round(rs1m, 1),
            "rs3m":   round(rs3m, 1),
            "rs6m":   round(rs6m, 1),
            "dist":   round(dist, 1),
            "accum":  round(accum, 2),
            "signal": signal,
            "close":  round(close, 2),
            "s2":     s2_sc,
            "trend":  trend,
        })

    out.sort(key=lambda x: _SIG_ORDER.get(x["signal"], 7))
    return out


# ── Watchlist transformer ──────────────────────────────────────
def _build_watchlist(watchlist_raw: list) -> list:
    out = []
    for w in watchlist_raw:
        ep  = w.get("entry_setup", {}) or {}
        pos = w.get("position", {}) or {}
        tgt = w.get("targets", {}) or {}
        pil = w.get("conviction_pillars", {}) or {}
        ea  = w.get("earnings_analysis", {}) or {}
        out.append({
            "ticker":  w.get("ticker", "").replace(".NS", ""),
            "sector":  w.get("sector", ""),
            "conv":    round(_flt(w.get("conviction_score", 0))),
            "t":       round(_flt(pil.get("technical", 0))),
            "v":       round(_flt(pil.get("value", 0))),
            "m":       round(_flt(pil.get("macro", 0))),
            "entry":   round(_flt(ep.get("entry_price", 0)), 2),
            "stop":    round(_flt(ep.get("effective_stop", 0)), 2),
            "risk":    round(_flt(ep.get("risk_pct", 0)), 1),
            "rr":      round(_flt(tgt.get("reward_risk_ratio", 0)), 1),
            "shares":  _int(pos.get("shares", 0)),
            "value":   round(_flt(pos.get("position_value", 0))),
            "earnings": ea.get("trend", "—") if ea.get("data_available") else "—",
            "fund":    w.get("fundamental_flag", "CLEAN"),
            "action":  w.get("action", "WATCH"),
        })
    return out


# ── Positions transformer ──────────────────────────────────────
def _build_positions(stock_data: dict) -> tuple[list, list]:
    try:
        from position_manager import get_positions_summary, load_trade_history
        summaries = get_positions_summary(stock_data)
        pos_out = []
        for p in summaries:
            pos_out.append({
                "id":         p.get("id", 0),
                "ticker":     p.get("ticker", "").replace(".NS", ""),
                "entry_date": p.get("entry_date", ""),
                "entry":      round(_flt(p.get("entry_price", 0)), 2),
                "current":    round(_flt(p.get("current_price", 0)), 2),
                "shares":     _int(p.get("shares", 0)),
                "stop_init":  round(_flt(p.get("initial_stop", 0)), 2),
                "stop_trail": round(_flt(p.get("trailing_stop", 0)), 2),
                "atr":        round(_flt(p.get("atr", 0)), 1),
                "high_since": round(_flt(p.get("highest_close", 0)), 2),
                "days":       _int(p.get("days_held", 0)),
                "conviction": p.get("conviction", ""),
                "action":     p.get("suggested_action", "HOLD"),
                "pnl":        round(_flt(p.get("pnl", 0))),
                "pnl_pct":    round(_flt(p.get("pnl_pct", 0)), 2),
                "reason":     p.get("action_reason", ""),
            })

        hist = load_trade_history()
        hist_out = []
        for t in hist:
            hist_out.append({
                "ticker":  t.get("ticker", "").replace(".NS", ""),
                "entry":   round(_flt(t.get("entry_price", 0)), 2),
                "exit":    round(_flt(t.get("exit_price", 0)), 2),
                "shares":  _int(t.get("shares", 0)),
                "days":    _int(t.get("days_held", 0)),
                "pnl":     round(_flt(t.get("pnl", 0))),
                "pnl_pct": round(_flt(t.get("pnl_pct", 0)), 1),
                "reason":  t.get("exit_reason", ""),
            })
        return pos_out, hist_out
    except Exception:
        return [], []


# ── FII/DII transformer ────────────────────────────────────────
def _build_fii(fii_today: dict, fii_flows: dict, fii_hist) -> dict:
    fii_net = _int(fii_today.get("fii_net", 0))
    dii_net = _int(fii_today.get("dii_net", 0))

    # Cumulative
    cum = []
    for tf in ["1w", "1m", "3m", "6m", "1y"]:
        fl = (fii_flows or {}).get(tf, {})
        if fl.get("days_available", 0) > 0:
            cum.append({
                "tf":  tf.upper(),
                "fii": _int(fl.get("fii_net", 0)),
                "dii": _int(fl.get("dii_net", 0)),
            })

    # Daily history (last 15 rows)
    daily = []
    if fii_hist is not None and not fii_hist.empty:
        import pandas as _pd
        df = fii_hist.copy()
        df["date"] = _pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False).head(15)
        for _, row in df.iterrows():
            daily.append({
                "d":   row["date"].strftime("%d %b"),
                "fii": _int(row.get("fii_net", 0)),
                "dii": _int(row.get("dii_net", 0)),
            })

        # Streak
        streak_dir = "sell" if daily[0]["fii"] < 0 else "buy"
        streak_days, streak_total = 0, 0
        for d in daily:
            if streak_dir == "sell" and d["fii"] < 0:
                streak_days  += 1
                streak_total += d["fii"]
            elif streak_dir == "buy" and d["fii"] >= 0:
                streak_days  += 1
                streak_total += d["fii"]
            else:
                break

        vel5d = sum(d["fii"] for d in daily[:5])

        # Monthly aggregation
        df2 = fii_hist.copy()
        df2["date"] = _pd.to_datetime(df2["date"])
        df2["month"] = df2["date"].dt.to_period("M")
        mg = df2.groupby("month").agg(fii=("fii_net","sum"), dii=("dii_net","sum")).reset_index()
        mg = mg.sort_values("month", ascending=False).head(7)
        monthly = [{"m": str(r["month"]), "fii": _int(r["fii"]), "dii": _int(r["dii"])} for _, r in mg.iterrows()]
    else:
        streak_dir, streak_days, streak_total, vel5d, monthly = "sell", 1, fii_net, fii_net * 5, []

    date_str = fii_today.get("date", dt.datetime.now().strftime("%a, %d %b %Y"))
    if not isinstance(date_str, str):
        date_str = dt.datetime.now().strftime("%a, %d %b %Y")

    return {
        "date":       date_str,
        "fii_net":    fii_net,
        "dii_net":    dii_net,
        "streak":     {"dir": streak_dir, "days": streak_days, "total": streak_total},
        "vel5d":      vel5d,
        "cumulative": cum,
        "daily":      daily,
        "monthly":    monthly,
    }


# ── Earnings transformer ───────────────────────────────────────
def _build_earnings(es: dict) -> list:
    if not es:
        return []
    q_label = es.get("quarter_label", "Q4FY25")
    out = []
    for key, reported_flag in [("upcoming", False), ("reported", True)]:
        for e in es.get(key, []):
            if not isinstance(e, dict) or not e.get("ticker"):
                continue
            out.append({
                "ticker":   str(e.get("ticker", "")).replace(".NS", ""),
                "sector":   str(e.get("sector", "")),
                "q":        q_label,
                "eps_g":    round(_flt(e.get("eps_growth", e.get("eps_yoy", 0))), 1),
                "rev_g":    round(_flt(e.get("revenue_growth", e.get("rev_yoy", 0))), 1),
                "trend":    str(e.get("trend", "—")),
                "date":     str(e.get("result_date", e.get("date", "—"))),
                "reported": reported_flag,
            })
    return out


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "caches": {
            "weekly":  WEEKLY_CACHE.exists(),
            "daily":   DAILY_CACHE.exists(),
            "monthly": MONTHLY_CACHE.exists(),
            "legacy":  LEGACY_CACHE.exists(),
        },
        "timestamp": dt.datetime.now().isoformat(),
    }


@app.get("/api/snapshot")
def snapshot():
    """Single endpoint — returns everything the HTML dashboard needs."""
    data = _load_all()
    if not data:
        return {
            "is_live":  False,
            "error":    "No scan cache found. Run a scan first (python pipeline.py or Streamlit Run Scan).",
            "scan_date": "",
        }

    regime_raw   = data.get("regime", {}) or {}
    signals_raw  = regime_raw.get("signals", {}) or {}
    brs          = data.get("breadth_by_stage", {}) or {}
    fii_today    = data.get("daily_fii_dii") or {}
    fii_flows    = data.get("daily_fii_dii_flows") or {}
    fii_hist     = data.get("fii_history")  # may not exist — that's fine

    # Merge daily stock data over weekly for fresher prices
    stock_data = dict(data.get("stock_data", {}) or {})
    daily_sd   = data.get("daily_stock_data") or {}
    stock_data.update(daily_sd)

    positions, trade_history = _build_positions(stock_data)

    return {
        "is_live":      True,
        "scan_date":    data.get("scan_date", data.get("last_weekly_scan_date", "")),
        "capital":      _flt(data.get("capital", 5_000_000), 5_000_000),
        "regime":       _build_regime(regime_raw),
        "nifty":        _build_nifty(data.get("nifty_df")),
        "breadth":      _build_breadth(brs, signals_raw),
        "sectors":      _build_sectors(data.get("sector_rankings", []), data.get("top_sectors", [])),
        "stocks":       _build_stocks(
                            data.get("screened_stocks", []),
                            data.get("stage2_candidates", []),
                            data.get("final_watchlist", []),
                        ),
        "watchlist":    _build_watchlist(data.get("final_watchlist", [])),
        "positions":    positions,
        "trade_history": trade_history,
        "fii_dii":      _build_fii(fii_today, fii_flows, fii_hist),
        "earnings":     _build_earnings(data.get("earnings_season") or {}),
    }


# ── Run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n  SDWB Data Bridge starting on http://localhost:8001")
    print("  Open SDWB Dashboard.html — it will connect automatically.\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
