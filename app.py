"""
Trading Pipeline Dashboard — Main Entry Point
Run with: streamlit run app.py
"""
import logging
import sys
import io
import os
import pickle
import datetime as dt
from zoneinfo import ZoneInfo
from pathlib import Path

IST = ZoneInfo("Asia/Kolkata")

import streamlit as st
import plotly.graph_objects as go

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── Page config (must be first st call) ─────────────────────────
st.set_page_config(
    page_title="Trading Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base typography — finance terminal monospace */
    html, body, [class*="css"] {
        font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
    }

    /* Section headers — terminal style */
    h4 {
        color: #6a6a8a !important;
        font-size: 0.75em !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        border-bottom: 1px solid #1e1e2e !important;
        padding-bottom: 8px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Remove default Streamlit padding bloat */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.8em !important;
        color: #555 !important;
    }

    /* Data tables */
    .stDataFrame {
        font-size: 0.85em;
    }

    /* Metric cards in st.metric */
    [data-testid="stMetric"] {
        background: #12121e;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Section divider */
    .section-divider {
        border-top: 1px solid #1a1a2a;
        margin: 2rem 0 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

from config import (POSITION_CONFIG, SECTOR_CONFIG, REGIME_CONFIG, SMART_MONEY_CONFIG,
                     MACRO_GROUPS, RISK_GAUGE_THRESHOLDS, MACRO_DERIVATIVE_LABELS,
                     NIFTY_PE_BANDS, INDIA_RISK_CONTEXT, CAP_TIER_INDICES)
from data_fetcher import (
    fetch_index_data, fetch_all_stock_data, fetch_sector_data,
    fetch_price_data, fetch_macro_data, get_sector_map,
    fetch_index_history, compute_breadth_timeseries,
    compute_cap_breadth_timeseries, compute_sector_breadth_timeseries,
)
from market_regime import compute_regime, compute_stockbee_breadth
from sector_rs import scan_sectors, get_top_sectors
from stock_screener import screen_stocks
from stage_filter import filter_stage2_candidates, scan_all_stages
from fundamental_veto import generate_final_watchlist
from conviction_scorer import rank_candidates_by_conviction, get_top_conviction_ideas, get_top_ideas_by_sector
from position_manager import get_positions_summary, load_positions
from nse_data_fetcher import get_nse_fetcher, compute_fii_dii_flows
from dashboard_helpers import (
    regime_color, build_nifty_sparkline, build_macro_pulse_html,
    build_macro_card_html, build_mini_heatmap, compute_quality_radar,
    build_risk_gauge_card_html, build_grouped_macro_pulse_html,
    build_yield_curve_indicator_html, build_macro_trend_chart,
    build_macro_trend_lw_html, compute_macro_derivatives,
    build_derivative_lw_html, compute_all_sector_rs_timeseries,
    compute_derivatives, detect_inflection_points,
    build_earnings_season_card_html,
    cadence_badge, build_daily_actions_html,
    compute_rsi, compute_fear_greed, build_fear_greed_gauge_html,
    build_cap_tier_card_html, build_sector_heatmap_html,
    build_risk_matrix_html, build_valuation_card_html,
    compute_advance_decline, signal_color,
)

# ── Scan Cache (disk persistence) ──────────────────────────────
CACHE_DIR = Path(__file__).parent / "scan_cache"
WEEKLY_CACHE_FILE = CACHE_DIR / "last_weekly_scan.pkl"
DAILY_CACHE_FILE = CACHE_DIR / "last_daily_check.pkl"
MONTHLY_CACHE_FILE = CACHE_DIR / "last_monthly_data.pkl"
CACHE_FILE = WEEKLY_CACHE_FILE  # backward compat

WEEKLY_CACHE_KEYS = [
    "scan_date", "capital", "nifty_df", "all_stock_data", "sector_data",
    "regime", "sector_rankings", "top_sectors", "stock_data",
    "screened_stocks", "stage2_candidates", "all_stage2_stocks", "final_watchlist",
    "macro_data", "quality_radar", "universe_count",
    "ai_summary", "ai_summary_source",
    "earnings_season",
    "macro_liquidity", "fii_gate", "breadth_by_stage", "stockbee_breadth",
    "last_weekly_scan_date",
]

DAILY_CACHE_KEYS = [
    "daily_stock_data", "daily_macro_data", "daily_fii_dii", "daily_fii_dii_flows",
    "daily_breakout_alerts", "daily_position_summaries",
    "last_daily_check_date",
]

MONTHLY_CACHE_KEYS = [
    "monthly_earnings_cache", "monthly_value_cache",
    "last_monthly_refresh_date",
]

# Backward compat alias
CACHE_KEYS = WEEKLY_CACHE_KEYS


def _save_cache(filepath, keys):
    CACHE_DIR.mkdir(exist_ok=True)
    data = {k: st.session_state[k] for k in keys if k in st.session_state}
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def _load_cache(filepath):
    if not filepath.exists():
        return False
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except Exception:
        return False


def save_scan_to_disk():
    _save_cache(WEEKLY_CACHE_FILE, WEEKLY_CACHE_KEYS)

def save_daily_to_disk():
    _save_cache(DAILY_CACHE_FILE, DAILY_CACHE_KEYS)

def save_monthly_to_disk():
    _save_cache(MONTHLY_CACHE_FILE, MONTHLY_CACHE_KEYS)

def load_scan_from_disk():
    # Try new weekly cache first, fall back to legacy
    legacy = CACHE_DIR / "last_scan.pkl"
    loaded = _load_cache(WEEKLY_CACHE_FILE)
    if not loaded and legacy.exists():
        loaded = _load_cache(legacy)
    # Also load daily + monthly overlays
    _load_cache(DAILY_CACHE_FILE)
    _load_cache(MONTHLY_CACHE_FILE)
    return loaded


def _check_staleness(date_key: str, ttl_hours: int) -> tuple[bool, str]:
    """Check staleness of a cadence. Returns (is_stale, age_description)."""
    date_str = st.session_state.get(date_key)
    if not date_str:
        return True, "never"
    try:
        scan_dt = dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        age = dt.datetime.now(IST) - scan_dt
        hours = age.total_seconds() / 3600
        if hours < 1:
            age_str = f"{int(age.total_seconds() / 60)}m ago"
        elif hours < 24:
            age_str = f"{int(hours)}h ago"
        else:
            age_str = f"{int(hours / 24)}d ago"
        return hours > ttl_hours, age_str
    except Exception:
        return True, "unknown"


def is_weekly_stale() -> tuple[bool, str]:
    return _check_staleness("last_weekly_scan_date", 7 * 24)

def is_daily_stale() -> tuple[bool, str]:
    return _check_staleness("last_daily_check_date", 18)  # stale after 18h

def is_monthly_stale() -> tuple[bool, str]:
    return _check_staleness("last_monthly_refresh_date", 30 * 24)

def is_cache_stale(max_age_hours: int = 24) -> bool:
    stale, _ = _check_staleness("last_weekly_scan_date", max_age_hours)
    if st.session_state.get("last_weekly_scan_date") is None:
        # Fall back to legacy scan_date
        date_str = st.session_state.get("scan_date")
        if not date_str:
            return True
        try:
            scan_dt = dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            return (dt.datetime.now(IST) - scan_dt).total_seconds() > max_age_hours * 3600
        except Exception:
            return True
    return stale


# Auto-load cached scan on first visit
if "regime" not in st.session_state:
    if load_scan_from_disk():
        if is_cache_stale():
            st.toast(f"Cached scan from {st.session_state.get('scan_date', '?')} is stale — click Run Scan to refresh")
        else:
            st.toast(f"Loaded scan from {st.session_state.get('scan_date', 'disk')}")


# ── Sidebar ─────────────────────────────────────────────────────
# Use config defaults — no sidebar knobs needed
capital = POSITION_CONFIG["total_capital"]
top_n = SECTOR_CONFIG["top_sectors_count"]

with st.sidebar:
    st.title("Trading Pipeline")
    st.caption("Weinstein + O'Neil / Minervini")

    st.divider()

    scan_mode = st.radio(
        "Scan Mode",
        ["Weekend Scan", "Daily Check", "Monthly Refresh"],
        index=0,
        help="**Weekend**: Full pipeline (regime, sectors, stages, watchlist). "
             "**Daily**: Fresh prices, breakouts, position updates. "
             "**Monthly**: Earnings + value analysis enrichment.",
    )

    has_weekly = "regime" in st.session_state
    _daily_disabled = not has_weekly and scan_mode == "Daily Check"
    _monthly_disabled = not has_weekly and scan_mode == "Monthly Refresh"

    _btn_labels = {
        "Weekend Scan": "Run Weekend Scan",
        "Daily Check": "Run Daily Check",
        "Monthly Refresh": "Run Monthly Refresh",
    }
    run_scan_btn = st.button(
        _btn_labels[scan_mode],
        type="primary",
        use_container_width=True,
        disabled=_daily_disabled or _monthly_disabled,
    )
    if _daily_disabled or _monthly_disabled:
        st.caption("Run Weekend Scan first.")

    # Status indicators
    st.markdown("**Status**")

    def _status_dot(stale, age_str):
        color = "#ef5350" if stale else "#26a69a"
        return f'<span style="color:{color};">●</span> {age_str}'

    w_stale, w_age = is_weekly_stale()
    d_stale, d_age = is_daily_stale()
    m_stale, m_age = is_monthly_stale()

    st.markdown(
        f'<div style="font-size:0.8em;line-height:2;font-family:monospace;">'
        f'Weekly: {_status_dot(w_stale, w_age)}<br>'
        f'Daily: {_status_dot(d_stale, d_age)}<br>'
        f'Monthly: {_status_dot(m_stale, m_age)}<br>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "universe_count" in st.session_state:
        st.caption(f"Universe: {st.session_state.universe_count} stocks")

    # Breakout alerts badge
    _daily_alerts = st.session_state.get("daily_breakout_alerts", [])
    if _daily_alerts:
        st.success(f"{len(_daily_alerts)} breakout alert{'s' if len(_daily_alerts) != 1 else ''}!", icon="🔔")

    st.divider()

    run_earnings = st.button("Earnings Scan", use_container_width=True)
    if "earnings_season" in st.session_state and st.session_state.earnings_season:
        es = st.session_state.earnings_season
        st.caption(f"Earnings: {es.get('quarter_label', '?')} | {es.get('reported_count', 0)}/{es.get('total_universe', 0)}")

    st.divider()
    st.caption("Built with Streamlit + Plotly")


# ── Scan Orchestration ──────────────────────────────────────────
def run_weekend_scan():
    """Run the full pipeline — regime, sectors, stages, watchlist. Weekly cadence."""
    # Capture stdout to suppress print output from pipeline modules
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        with st.status("Running pipeline scan...", expanded=True) as status:
            progress = st.progress(0)

            # Step 0: Fetch macro data
            st.write("Fetching macro data...")
            progress.progress(2)
            macro_data = fetch_macro_data()
            st.session_state.macro_data = macro_data

            # Step 1: Fetch Nifty data
            st.write("Fetching Nifty 50 index data...")
            progress.progress(5)
            nifty_df = fetch_index_data()
            st.session_state.nifty_df = nifty_df

            # Step 2: Fetch all stock data
            from data_fetcher import get_all_stock_tickers
            all_tickers = get_all_stock_tickers()
            st.session_state.universe_count = len(all_tickers)
            st.write(f"Fetching stock universe data ({len(all_tickers)} stocks — this may take a few minutes)...")
            progress.progress(10)
            all_stock_data = fetch_all_stock_data()
            st.session_state.all_stock_data = all_stock_data
            st.write(f"  Loaded {len(all_stock_data)} stocks")
            progress.progress(30)

            # Step 3: Market regime + macro liquidity
            st.write("Computing market regime...")
            from market_regime import compute_macro_liquidity_score
            macro_liquidity = None
            fii_dii_data_scan = None
            try:
                _nse_scan = get_nse_fetcher()
                fii_dii_data_scan = _nse_scan.compute_fii_dii_flows()
            except Exception:
                pass
            macro_liquidity = compute_macro_liquidity_score(macro_data, fii_dii_data_scan)
            st.session_state.macro_liquidity = macro_liquidity
            regime = compute_regime(nifty_df, all_stock_data, macro_liquidity=macro_liquidity)
            st.session_state.regime = regime
            st.write(f"  Regime: {regime['label']} (score {regime['regime_score']:+d})")

            st.write("Computing Stockbee breadth indicators...")
            stockbee = compute_stockbee_breadth(all_stock_data, lookback_days=30)
            st.session_state.stockbee_breadth = stockbee
            progress.progress(40)

            # Step 4: Sector RS
            st.write("Fetching sector data & computing RS...")
            sector_data = fetch_sector_data()
            st.session_state.sector_data = sector_data
            progress.progress(55)

            sector_rankings = scan_sectors(sector_data, nifty_df)
            st.session_state.sector_rankings = sector_rankings

            top_sectors = get_top_sectors(sector_rankings, n=top_n)
            st.session_state.top_sectors = top_sectors
            st.write(f"  Top sectors: {', '.join(top_sectors)}")
            progress.progress(60)

            # Step 5: Stock screening
            st.write("Screening stocks in top sectors...")
            stock_data = dict(all_stock_data)
            sector_map = get_sector_map()
            needed = []
            for sector in top_sectors:
                for t in sector_map.get(sector, []):
                    if t not in stock_data:
                        needed.append(t)
            if needed:
                extra = fetch_price_data(needed)
                stock_data.update(extra)
            st.session_state.stock_data = stock_data

            screened = screen_stocks(stock_data, nifty_df, sector_data, top_sectors)
            st.session_state.screened_stocks = screened
            st.write(f"  {len(screened)} stocks passed screening")
            progress.progress(75)

            # Step 6: Stage filter (pipeline candidates from screened stocks)
            st.write("Running stage analysis...")
            stage2 = filter_stage2_candidates(stock_data, screened) if screened else []
            st.session_state.stage2_candidates = stage2
            st.write(f"  {len(stage2)} pipeline Stage 2 candidates (from top sectors)")

            # Step 6b: Broad stage scan (ALL stocks in universe, s2_score >= 4)
            st.write("Scanning full universe for Stage 2 stocks...")
            all_stage2 = scan_all_stages(stock_data, min_s2_score=4)
            st.session_state.all_stage2_stocks = all_stage2
            st.write(f"  {len(all_stage2)} Stage 2 stocks across all sectors")
            progress.progress(85)

            # Step 6c: FII gating
            fii_gate = None
            try:
                from fii_gating import check_fii_gate
                fii_gate = check_fii_gate(fii_dii_data_scan)
                st.session_state.fii_gate = fii_gate
                if fii_gate and fii_gate.get("gated"):
                    st.write(f"  FII Gate: {fii_gate['gate_level'].upper()}")
            except Exception:
                pass

            # Step 6d: Breadth by stage
            try:
                from breadth_analysis import compute_breadth_by_stage
                from stage_filter import classify_stage
                stage_results = []
                for t, sdf in all_stock_data.items():
                    if len(sdf) >= 230:
                        stage_results.append({"ticker": t, "stage": classify_stage(sdf)})
                breadth = compute_breadth_by_stage(stage_results)
                st.session_state.breadth_by_stage = breadth
                st.write(f"  Breadth: {breadth['breadth_label']} (S2: {breadth['stage_pcts'].get(2, 0):.0f}%)")
            except Exception:
                pass

            # Step 6e: Enrich candidates with earnings + value analysis
            if stage2:
                st.write("Enriching candidates (earnings + value)...")
                try:
                    from earnings_analysis import compute_earnings_acceleration
                    for c in stage2:
                        try:
                            c["earnings_analysis"] = compute_earnings_acceleration(c["ticker"])
                        except Exception:
                            c["earnings_analysis"] = {"data_available": False}
                except ImportError:
                    pass
                try:
                    from value_analysis import compute_value_score
                    for c in stage2:
                        try:
                            c["value_analysis"] = compute_value_score(c["ticker"])
                        except Exception:
                            c["value_analysis"] = {"data_available": False}
                except ImportError:
                    pass
                progress.progress(88)

            # Step 7: Fundamental veto + watchlist
            st.write("Applying fundamental veto & sizing positions...")
            watchlist = generate_final_watchlist(stage2, regime, capital) if stage2 else []
            # Apply R:R scan
            if watchlist:
                try:
                    from rr_scanner import scan_asymmetric_setups
                    watchlist = scan_asymmetric_setups(watchlist)
                except ImportError:
                    pass
            st.session_state.final_watchlist = watchlist
            progress.progress(92)

            # Step 8: Quality Radar
            st.write("Computing Quality Radar...")
            quality_radar = compute_quality_radar(watchlist)
            st.session_state.quality_radar = quality_radar
            progress.progress(95)

            buy_count = sum(1 for w in watchlist if w.get("action") == "BUY")
            watch_count = sum(1 for w in watchlist if w.get("action") in ("WATCH", "WATCHLIST"))
            st.write(f"  {buy_count} BUY signals, {watch_count} watchlist")

            # Step 9: Market Summary
            st.write("Generating market summary...")
            from ai_summary import generate_market_summary
            _fii_dii_scan = None
            _fii_dii_flows_scan = {}
            try:
                _nse = get_nse_fetcher()
                _fii_dii_scan = _nse.fetch_fii_dii_data()
                _hist = _nse.fetch_fii_dii_historical()
                if _hist is not None and not _hist.empty:
                    _fii_dii_flows_scan = compute_fii_dii_flows(_hist)
            except Exception:
                pass
            summary, source = generate_market_summary(
                macro_data, regime, _fii_dii_scan, _fii_dii_flows_scan, sector_rankings,
            )
            st.session_state.ai_summary = summary
            st.session_state.ai_summary_source = source
            st.write(f"  Summary: {source}")

            now_str = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            st.session_state.scan_date = now_str
            st.session_state.last_weekly_scan_date = now_str
            st.session_state.capital = capital

            # Attach monthly enrichment if available
            _me = st.session_state.get("monthly_earnings_cache", {})
            _mv = st.session_state.get("monthly_value_cache", {})
            if _me or _mv:
                for c in stage2:
                    t = c.get("ticker", "")
                    if t in _me and "earnings_analysis" not in c:
                        c["earnings_analysis"] = _me[t]
                    if t in _mv and "value_analysis" not in c:
                        c["value_analysis"] = _mv[t]

            # Save to disk for persistence across restarts
            st.write("Saving scan results to disk...")
            save_scan_to_disk()
            progress.progress(100)

            status.update(label="Weekend scan complete!", state="complete")

    finally:
        sys.stdout = old_stdout


def run_daily_check():
    """Lightweight daily check — fresh prices, breakouts, position updates."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        with st.status("Running daily check...", expanded=True) as status:
            progress = st.progress(0)

            # D1: Collect tickers to refresh
            st.write("Identifying tickers to refresh...")
            daily_tickers = set()
            for w in st.session_state.get("final_watchlist", []):
                daily_tickers.add(w.get("ticker", ""))
            for p in load_positions():
                daily_tickers.add(p.get("ticker", ""))
            daily_tickers.discard("")
            st.write(f"  {len(daily_tickers)} tickers")
            progress.progress(10)

            # D2: Fetch fresh price data
            st.write("Fetching fresh prices...")
            if daily_tickers:
                daily_data = fetch_price_data(list(daily_tickers))
                st.session_state.daily_stock_data = daily_data
                # Merge into main stock_data so other pages see fresh prices
                main_sd = st.session_state.get("stock_data", {})
                main_sd.update(daily_data)
                st.session_state.stock_data = main_sd
            progress.progress(40)

            # D3: Fresh macro data + FII/DII
            st.write("Fetching macro data...")
            try:
                daily_macro = fetch_macro_data()
                st.session_state.macro_data = daily_macro
                st.session_state.daily_macro_data = daily_macro
            except Exception:
                pass
            try:
                _nse = get_nse_fetcher()
                _fii = _nse.fetch_fii_dii_data()
                st.session_state.daily_fii_dii = _fii
                _hist = _nse.fetch_fii_dii_historical()
                if _hist is not None and not _hist.empty:
                    _flows = compute_fii_dii_flows(_hist)
                    st.session_state.daily_fii_dii_flows = _flows
            except Exception:
                pass
            progress.progress(60)

            # D4: Position management updates
            st.write("Updating positions...")
            pos_data = st.session_state.get("stock_data", {})
            pos_summaries = get_positions_summary(pos_data)
            st.session_state.daily_position_summaries = pos_summaries
            progress.progress(75)

            # D5: Breakout detection on watchlist
            st.write("Checking for breakouts...")
            from stage_filter import detect_bases, detect_breakout
            alerts = []
            for w in st.session_state.get("final_watchlist", []):
                t = w.get("ticker", "")
                df = pos_data.get(t)
                if df is None or df.empty or len(df) < 50:
                    continue
                try:
                    bases = detect_bases(df)
                    if bases:
                        bo = detect_breakout(df, bases)
                        if bo and bo.get("breakout"):
                            alerts.append({
                                "ticker": t,
                                "breakout_price": bo.get("breakout_price", 0),
                                "volume_ratio": bo.get("volume_ratio", 0),
                                "base_high": bo.get("base_high", 0),
                            })
                except Exception:
                    pass
            st.session_state.daily_breakout_alerts = alerts
            if alerts:
                st.write(f"  {len(alerts)} breakout alert(s)!")
            progress.progress(90)

            # D6: Save + timestamp
            now_str = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            st.session_state.last_daily_check_date = now_str
            save_daily_to_disk()
            save_scan_to_disk()  # also persist updated stock_data
            progress.progress(100)

            status.update(label=f"Daily check complete — {len(alerts)} alerts", state="complete")

    finally:
        sys.stdout = old_stdout


def run_monthly_refresh():
    """Monthly enrichment — earnings acceleration + value analysis for all candidates."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        with st.status("Running monthly refresh...", expanded=True) as status:
            progress = st.progress(0)

            # M1: Collect all candidate tickers
            candidates = st.session_state.get("stage2_candidates", [])
            all_s2 = st.session_state.get("all_stage2_stocks", [])
            ticker_set = set()
            for c in candidates:
                ticker_set.add(c.get("ticker", ""))
            for s in all_s2[:100]:  # cap at top 100 to keep it reasonable
                ticker_set.add(s.get("ticker", ""))
            ticker_set.discard("")
            st.write(f"Enriching {len(ticker_set)} stocks with earnings + value analysis...")
            progress.progress(5)

            # M2: Earnings acceleration
            st.write("Computing earnings acceleration...")
            earnings_cache = {}
            try:
                from earnings_analysis import compute_earnings_acceleration
                for i, t in enumerate(ticker_set):
                    try:
                        earnings_cache[t] = compute_earnings_acceleration(t)
                    except Exception:
                        earnings_cache[t] = {"data_available": False}
                    if (i + 1) % 20 == 0:
                        progress.progress(5 + int(40 * (i + 1) / len(ticker_set)))
            except ImportError:
                st.write("  earnings_analysis module not available")
            st.session_state.monthly_earnings_cache = earnings_cache
            progress.progress(45)

            # M3: Value analysis
            st.write("Computing value scores...")
            value_cache = {}
            try:
                from value_analysis import compute_value_score
                for i, t in enumerate(ticker_set):
                    try:
                        value_cache[t] = compute_value_score(t)
                    except Exception:
                        value_cache[t] = {"data_available": False}
                    if (i + 1) % 20 == 0:
                        progress.progress(45 + int(40 * (i + 1) / len(ticker_set)))
            except ImportError:
                st.write("  value_analysis module not available")
            st.session_state.monthly_value_cache = value_cache
            progress.progress(85)

            # M4: Merge back into candidates
            st.write("Merging into candidates...")
            for c in candidates:
                t = c.get("ticker", "")
                if t in earnings_cache:
                    c["earnings_analysis"] = earnings_cache[t]
                if t in value_cache:
                    c["value_analysis"] = value_cache[t]
            progress.progress(90)

            # M5: Recompute conviction with fresh data
            st.write("Recomputing conviction scores...")
            try:
                sector_rankings = st.session_state.get("sector_rankings", [])
                _ml = st.session_state.get("macro_liquidity")
                _fg = st.session_state.get("fii_gate")
                ranked = rank_candidates_by_conviction(
                    candidates=candidates,
                    sector_rankings=sector_rankings,
                    macro_liquidity=_ml,
                    fii_gate=_fg,
                )
            except Exception:
                pass

            # M6: Save + timestamp
            now_str = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            st.session_state.last_monthly_refresh_date = now_str
            save_monthly_to_disk()
            save_scan_to_disk()  # re-save weekly with enriched data
            progress.progress(100)

            status.update(label=f"Monthly refresh complete — {len(ticker_set)} stocks enriched", state="complete")

    finally:
        sys.stdout = old_stdout


# ── Scan Button Handler ──────────────────────────────────────
if run_scan_btn:
    if scan_mode == "Weekend Scan":
        run_weekend_scan()
    elif scan_mode == "Daily Check":
        run_daily_check()
    elif scan_mode == "Monthly Refresh":
        run_monthly_refresh()


# ── Earnings Scan Orchestration ───────────────────────────────
if run_earnings:
    from earnings_season import run_earnings_scan, load_earnings_cache
    from data_fetcher import load_universe as _load_universe

    with st.status("Running earnings scan...", expanded=True) as status:
        progress = st.progress(0)
        st.write("Loading universe...")
        universe_df = _load_universe()

        def _earnings_progress(current, total, symbol):
            if total > 0:
                progress.progress(min(current / total, 0.99))
            clean = symbol.replace(".NS", "") if isinstance(symbol, str) else symbol
            if current % 50 == 0:
                st.write(f"  Processing {current}/{total} — {clean}")

        st.write(f"Fetching quarterly results for {len(universe_df)} stocks...")
        result = run_earnings_scan(universe_df, progress_callback=_earnings_progress)
        st.session_state.earnings_season = result
        progress.progress(1.0)

        # Save main cache too (includes earnings_season key)
        save_scan_to_disk()
        status.update(
            label=f"Earnings scan complete — {result.get('reported_count', 0)}/{result.get('total_universe', 0)} reported",
            state="complete",
        )


# ── Home Page — India Market Dashboard ────────────────────────
scan_date_str = st.session_state.get("scan_date", "")
st.markdown(
    f'<div style="margin-bottom:1.5rem;">'
    f'<h1 style="margin-bottom:2px;font-size:1.6em;font-weight:700;color:#e0e0e0;">India Market Dashboard</h1>'
    f'<div style="color:#555;font-size:0.82em;font-family:monospace;">'
    f'{dt.datetime.now(IST).strftime("%A, %d %B %Y")}'
    f'{" &middot; Last scan: " + scan_date_str if scan_date_str else ""}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

if "regime" not in st.session_state:
    st.info("Click **Run Scan** in the sidebar to start the pipeline.")
    st.stop()

regime = st.session_state.regime
watchlist = st.session_state.get("final_watchlist", [])
top_sectors = st.session_state.get("top_sectors", [])
macro_data = st.session_state.get("macro_data", {})
nifty_df = st.session_state.get("nifty_df")
all_stock_data = st.session_state.get("all_stock_data", {})
sector_rankings = st.session_state.get("sector_rankings", [])
sector_data = st.session_state.get("sector_data", {})

# Fetch FII/DII data
nse_fetcher = get_nse_fetcher()
fii_dii = None
fii_dii_history = None
fii_dii_flows = {}
try:
    fii_dii = nse_fetcher.fetch_fii_dii_data()
except Exception:
    pass
try:
    fii_dii_history = nse_fetcher.fetch_fii_dii_historical()
    if fii_dii_history is not None and not fii_dii_history.empty:
        fii_dii_flows = compute_fii_dii_flows(fii_dii_history)
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════
# SECTION 0: India Pulse Strip (4 key numbers)
# ══════════════════════════════════════════════════════════════════
if macro_data:
    _pulse_items = []
    # Nifty 50
    _n = macro_data.get("Nifty 50", {})
    if _n:
        _nc = "#26a69a" if _n.get("change_pct", 0) >= 0 else "#ef5350"
        _pulse_items.append(("NIFTY 50", f'{_n.get("price", 0):,.1f}', f'{_n.get("change_pct", 0):+.2f}%', _nc))
    # India VIX
    _v = macro_data.get("India VIX", {})
    if _v:
        vp = _v.get("price", 18)
        _vl = "CALM" if vp < 14 else "FEAR" if vp > 22 else "CAUTION"
        _vc2 = "#26a69a" if vp < 14 else "#ef5350" if vp > 22 else "#FF9800"
        _pulse_items.append(("INDIA VIX", f'{vp:.1f}', _vl, _vc2))
    # USD/INR
    _u = macro_data.get("USD/INR", {})
    if _u:
        up = _u.get("price", 83)
        _uc = "#26a69a" if up < 82 else "#ef5350" if up > 86 else "#888"
        _pulse_items.append(("USD/INR", f'{up:.2f}', f'{_u.get("change_pct", 0):+.2f}%', _uc))
    # Liquidity Score
    _ml = st.session_state.get("macro_liquidity", {})
    if _ml:
        _mls = _ml.get("score", 50)
        _mll = _ml.get("label", "")
        _mlc = "#26a69a" if _mls >= 60 else "#ef5350" if _mls < 40 else "#FF9800"
        _pulse_items.append(("LIQUIDITY", f'{_mls:.0f}/100', _mll, _mlc))

    if _pulse_items:
        _pcols = st.columns(len(_pulse_items))
        for _pc, (_plbl, _pval, _psub, _pclr) in zip(_pcols, _pulse_items):
            with _pc:
                st.markdown(
                    f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:6px;padding:12px 16px;text-align:center;">'
                    f'<div style="font-size:0.65em;color:#555;text-transform:uppercase;letter-spacing:0.1em;">{_plbl}</div>'
                    f'<div style="font-size:1.4em;font-weight:700;color:#e8e8e8;font-family:monospace;">{_pval}</div>'
                    f'<div style="font-size:0.8em;color:{_pclr};font-weight:600;">{_psub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════
# SECTION 1: Market Regime + Fear & Greed
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Market Regime & Sentiment", unsafe_allow_html=True)

label = regime["label"]
color = regime_color(label)
signals = regime.get("signals", {})
bullish_count = sum(1 for s in signals.values() if isinstance(s, dict) and s.get("score", 0) > 0)
total_signals = len(signals)
breadth_trend = regime.get("breadth_trend", "stable")
trend_label = {"improving": "Breadth Improving", "deteriorating": "Breadth Weakening"}.get(breadth_trend, "Breadth Stable")
trend_color_val = {"improving": "#26a69a", "deteriorating": "#ef5350"}.get(breadth_trend, "#888")

# FII/DII inline
fii_net_str = ""
if fii_dii:
    _fn = fii_dii.get("fii_net", 0)
    _dn = fii_dii.get("dii_net", 0)
    fii_net_str = (
        f'<span style="margin-left:15px;font-size:0.9em;">'
        f'FII: <span style="color:{"#26a69a" if _fn >= 0 else "#ef5350"};font-weight:600;">{_fn:+,.0f} Cr</span>'
        f' &nbsp; DII: <span style="color:{"#26a69a" if _dn >= 0 else "#ef5350"};font-weight:600;">{_dn:+,.0f} Cr</span>'
        f'</span>'
    )

_fg_col, _regime_col = st.columns([1, 2])

with _fg_col:
    # Fear & Greed Gauge
    if nifty_df is not None:
        fg = compute_fear_greed(regime, macro_data, fii_dii_flows, nifty_df)
        st.markdown(build_fear_greed_gauge_html(fg["score"], fg["label"]), unsafe_allow_html=True)
        with st.expander("Components", expanded=False):
            for k, c in fg["components"].items():
                _cclr = "#26a69a" if c["score"] > 60 else "#ef5350" if c["score"] < 40 else "#888"
                st.markdown(f'<span style="font-size:0.8em;color:{_cclr};">{c["label"]} ({c["score"]:.0f})</span>', unsafe_allow_html=True)

with _regime_col:
    # Regime banner
    st.markdown(
        f'<div style="background:{color}0d;border-left:3px solid {color};'
        f'padding:10px 18px;border-radius:0 6px 6px 0;margin-bottom:10px;">'
        f'<span style="font-size:1.4em;font-weight:800;color:{color};">{label.upper()}</span>'
        f'<span style="font-size:0.85em;margin-left:12px;color:#999;">'
        f'{bullish_count}/{total_signals} bullish &middot; '
        f'<span style="color:{trend_color_val};">{trend_label}</span></span>'
        f'{fii_net_str}</div>',
        unsafe_allow_html=True,
    )

    # Signal chips
    _sig_names = {"index_vs_200dma": "Nifty vs 200DMA", "ma_crossover": "50/200 Cross",
                  "breadth_50dma": "> 50DMA", "breadth_200dma": "> 200DMA", "net_new_highs": "Hi/Lo"}
    _chips = []
    for key, dname in _sig_names.items():
        sig = signals.get(key, {})
        sc = sig.get("score", 0)
        _cc = "#26a69a" if sc > 0 else "#ef5350" if sc < 0 else "#888"
        _ci = "+" if sc > 0 else ("-" if sc < 0 else "~")
        _chips.append(f'<span style="display:inline-block;background:{_cc}15;border:1px solid {_cc}33;border-radius:6px;padding:3px 8px;margin:2px;font-size:0.8em;"><span style="color:{_cc};font-weight:700;">{_ci}</span> <span style="color:#ccc;">{dname}</span></span>')
    st.markdown(f'<div>{"".join(_chips)}</div>', unsafe_allow_html=True)

    # Stage breadth
    _brs = st.session_state.get("breadth_by_stage")
    if _brs:
        _sp = _brs.get("stage_pcts", {})
        _bl = _brs.get("breadth_label", "")
        _bc = "#26a69a" if _brs.get("breadth_score", 50) >= 55 else "#ef5350" if _brs.get("breadth_score", 50) < 45 else "#FF9800"
        st.markdown(
            f'<div style="display:flex;gap:8px;align-items:center;margin:6px 0;">'
            f'<span style="font-size:0.7em;color:#666;">STAGES:</span>'
            f'<span style="font-size:0.7em;background:#2196F322;color:#2196F3;padding:2px 6px;border-radius:3px;">S1:{_sp.get(1,0):.0f}%</span>'
            f'<span style="font-size:0.7em;background:#26a69a22;color:#26a69a;padding:2px 6px;border-radius:3px;">S2:{_sp.get(2,0):.0f}%</span>'
            f'<span style="font-size:0.7em;background:#FF980022;color:#FF9800;padding:2px 6px;border-radius:3px;">S3:{_sp.get(3,0):.0f}%</span>'
            f'<span style="font-size:0.7em;background:#ef535022;color:#ef5350;padding:2px 6px;border-radius:3px;">S4:{_sp.get(4,0):.0f}%</span>'
            f'<span style="font-size:0.7em;font-weight:600;color:{_bc};">{_bl}</span></div>',
            unsafe_allow_html=True,
        )

    # FII gate
    _fgate = st.session_state.get("fii_gate")
    if _fgate and _fgate.get("gated"):
        _gc = "#ef5350" if _fgate["gate_level"] == "severe" else "#FF9800"
        st.markdown(f'<div style="background:{_gc}12;border-left:2px solid {_gc};border-radius:0 4px 4px 0;padding:5px 10px;font-size:0.78em;"><span style="color:{_gc};font-weight:600;">FII GATE: {_fgate["gate_level"].upper()}</span> <span style="color:#888;">{_fgate["reason"]}</span></div>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: Market Cap Tiers (Large / Mid / Small)
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Market Cap Tiers", unsafe_allow_html=True)

from sector_rs import classify_sector_stage

_tier_data = [
    ("Nifty 50", nifty_df, macro_data.get("Nifty 50", {})),
    ("Nifty Midcap 150", sector_data.get("Nifty Midcap 150"), None),
    ("Nifty Smallcap 250", sector_data.get("Nifty Smallcap 250"), None),
]
_tier_cols = st.columns(len(_tier_data))

for _tc, (_tname, _tdf, _tmacro) in zip(_tier_cols, _tier_data):
    with _tc:
        if _tdf is not None and len(_tdf) > 200:
            _close = _tdf["Close"]
            _price = float(_close.iloc[-1])
            _chg = float((_close.iloc[-1] / _close.iloc[-2] - 1) * 100) if len(_close) > 1 else 0
            if _tmacro:
                _chg = _tmacro.get("change_pct", _chg)
            _ma200 = float(_close.rolling(200).mean().iloc[-1])
            _dist = (_price - _ma200) / _ma200 * 100
            _rsi = float(compute_rsi(_close, 14).iloc[-1])
            try:
                _stage_info = classify_sector_stage(_tdf)
                _stg = f"S{_stage_info.get('stage', '?')}"
            except Exception:
                _stg = "-"
            st.markdown(build_cap_tier_card_html(_tname, _price, _chg, _dist, _rsi, _stg), unsafe_allow_html=True)
        else:
            st.caption(f"{_tname}: data unavailable")

# ── Index Price Charts with 50/200 DMA ──
import plotly.graph_objects as go

_idx_chart_period = st.selectbox("Chart Period", ["1y", "3y", "5y"], index=1, key="idx_chart_period")
_idx_tickers = {"Nifty 50": "^NSEI", "Nifty Midcap 150": "NIFTYMIDCAP150.NS", "Nifty Smallcap 250": "NIFTY_SMLCAP_250.NS"}
_idx_tabs = st.tabs(list(_idx_tickers.keys()))

for _itab, (_iname, _itick) in zip(_idx_tabs, _idx_tickers.items()):
    with _itab:
        _idf = fetch_index_history(_itick, period=_idx_chart_period)
        if _idf is not None and len(_idf) > 50:
            _ic = _idf["Close"]
            _ma50 = _ic.rolling(50).mean()
            _ma200 = _ic.rolling(200).mean()
            _fig_idx = go.Figure()
            _fig_idx.add_trace(go.Scatter(x=_ic.index, y=_ic, name="Price", line=dict(color="#e0e0e0", width=1.5)))
            _fig_idx.add_trace(go.Scatter(x=_ma50.index, y=_ma50, name="50 DMA", line=dict(color="#2196F3", width=1, dash="dot")))
            _fig_idx.add_trace(go.Scatter(x=_ma200.index, y=_ma200, name="200 DMA", line=dict(color="#FF9800", width=1, dash="dash")))
            _fig_idx.update_layout(
                height=350, template="plotly_dark",
                margin=dict(l=50, r=20, t=30, b=30),
                yaxis_title=_iname, xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(_fig_idx, use_container_width=True)
        else:
            st.caption(f"{_iname}: insufficient data")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Where Is The Money? (FII/DII + Sector Heatmap)
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Where Is The Money?", unsafe_allow_html=True)

# FII/DII flows table
if fii_dii_flows:
    st.markdown("**FII / DII Cumulative Net Flows (Cr)**")

    def _fmt_flow(val):
        if val is None:
            return '<span style="color:#555;">—</span>'
        _fc = "#26a69a" if val >= 0 else "#ef5350"
        _fd = f"{val/1000:+,.1f}K" if abs(val) >= 10000 else f"{val:+,.0f}"
        return f'<span style="color:{_fc};font-weight:600;">{_fd}</span>'

    _tf_order = ["1w", "2w", "1m", "3m", "6m", "1y", "2y", "5y"]
    _hdr = "".join(f'<th style="padding:6px 10px;color:#666;font-size:0.8em;text-align:center;text-transform:uppercase;">{tf}</th>' for tf in _tf_order)
    _fii_c, _dii_c = "", ""
    for tf in _tf_order:
        fl = fii_dii_flows.get(tf, {})
        _d = fl.get("days_available", 0)
        _fii_c += f'<td style="padding:6px 10px;text-align:center;">{_fmt_flow(fl.get("fii_net")) if _d > 0 else _fmt_flow(None)}</td>'
        _dii_c += f'<td style="padding:6px 10px;text-align:center;">{_fmt_flow(fl.get("dii_net")) if _d > 0 else _fmt_flow(None)}</td>'

    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;background:#0f0f1a;border-radius:6px;overflow:hidden;font-family:monospace;">'
        f'<thead><tr style="border-bottom:1px solid #1e1e2e;"><th style="padding:6px 10px;color:#666;font-size:0.8em;text-align:left;width:80px;"></th>{_hdr}</tr></thead>'
        f'<tbody><tr style="border-bottom:1px solid #1e1e2e;"><td style="padding:6px 10px;font-weight:600;color:#999;">FII</td>{_fii_c}</tr>'
        f'<tr><td style="padding:6px 10px;font-weight:600;color:#999;">DII</td>{_dii_c}</tr></tbody></table>',
        unsafe_allow_html=True,
    )

    if fii_dii_history is not None and not fii_dii_history.empty:
        import pandas as pd
        _earliest = pd.to_datetime(fii_dii_history["date"]).min().strftime("%d %b %Y")
        _latest = pd.to_datetime(fii_dii_history["date"]).max().strftime("%d %b %Y")
        st.caption(f"History: {_earliest} to {_latest} ({len(fii_dii_history)} days). Builds with each scan.")
elif not fii_dii:
    st.caption("FII/DII data unavailable — NSE API may be down.")

# Sector Heatmap (table + RS bar chart)
if sector_rankings:
    st.markdown("**Sector Rotation Heatmap**")
    st.markdown(build_sector_heatmap_html(sector_rankings, top_sectors), unsafe_allow_html=True)

    # Sector RS Bar Chart
    import plotly.graph_objects as go
    _sr_names = [s.get("sector", s.get("name", "")) for s in sector_rankings]
    _sr_rs = [s.get("mansfield_rs", 0) for s in sector_rankings]
    _sr_colors = ["#26a69a" if v > 0 else "#ef5350" for v in _sr_rs]
    _sr_stages = []
    for s in sector_rankings:
        si = s.get("sector_stage", {})
        _sr_stages.append(f"S{si.get('stage','?')}" if isinstance(si, dict) else str(si))

    _fig_rs = go.Figure(go.Bar(
        x=_sr_rs, y=_sr_names, orientation="h",
        marker_color=_sr_colors,
        text=[f"{r:+.1f} ({st})" for r, st in zip(_sr_rs, _sr_stages)],
        textposition="outside", textfont=dict(size=10),
    ))
    _fig_rs.update_layout(
        height=max(350, len(_sr_names) * 28), template="plotly_dark",
        margin=dict(l=120, r=60, t=20, b=30),
        xaxis_title="Mansfield RS", yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="#444", zerolinewidth=1),
    )
    st.plotly_chart(_fig_rs, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 3B: Market Breadth Charts
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Market Breadth", unsafe_allow_html=True)

if all_stock_data:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Overall breadth
    _breadth_df = compute_breadth_timeseries(all_stock_data, lookback=756)
    if not _breadth_df.empty:
        _fig_b = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                               subplot_titles=["% Stocks Above Moving Average", "Advance / Decline Line"],
                               row_heights=[0.55, 0.45])
        _fig_b.add_trace(go.Scatter(x=_breadth_df.index, y=_breadth_df["pct_above_50dma"],
                                     name="% > 50 DMA", line=dict(color="#2196F3", width=1.5)), row=1, col=1)
        _fig_b.add_trace(go.Scatter(x=_breadth_df.index, y=_breadth_df["pct_above_200dma"],
                                     name="% > 200 DMA", line=dict(color="#FF9800", width=1.5)), row=1, col=1)
        _fig_b.add_hline(y=50, line_dash="dash", line_color="#444", row=1, col=1)
        _fig_b.add_hline(y=70, line_dash="dot", line_color="#26a69a33", row=1, col=1, annotation_text="Overbought")
        _fig_b.add_hline(y=30, line_dash="dot", line_color="#ef535033", row=1, col=1, annotation_text="Oversold")

        _fig_b.add_trace(go.Scatter(x=_breadth_df.index, y=_breadth_df["ad_line"],
                                     name="A/D Line", line=dict(color="#AB47BC", width=1.5)), row=2, col=1)
        _fig_b.update_layout(height=500, template="plotly_dark", margin=dict(l=50, r=20, t=40, b=30),
                             legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
                             hovermode="x unified")
        _fig_b.update_yaxes(title_text="%", row=1, col=1)
        _fig_b.update_yaxes(title_text="Cumulative", row=2, col=1)
        st.plotly_chart(_fig_b, use_container_width=True)

    # Cap-tier breadth (Large / Mid / Small)
    st.markdown("**Breadth by Market Cap**")
    _cap_ma = st.radio("MA Period", [50, 200], horizontal=True, key="cap_breadth_ma")

    # Get tier symbol lists from sector map
    _smap = get_sector_map()
    _n50_csv = Path("scan_cache/nifty50_symbols.txt")
    _mid_csv = Path("scan_cache/midcap_symbols.txt")
    _sm_csv = Path("scan_cache/smallcap_symbols.txt")

    # Build tier lists from stock data based on approximate market cap ranges
    # Use all stocks and segment by which index they belong to
    _all_tickers = list(all_stock_data.keys())
    # Simple heuristic: split by data availability and stock count
    _n50_syms = [t for t in _all_tickers[:50]]  # first 50 tend to be large cap (from Nifty TM ordering)
    _mid_syms = [t for t in _all_tickers[50:200]]
    _sm_syms = [t for t in _all_tickers[200:]]

    _cap_b = compute_cap_breadth_timeseries(
        all_stock_data, nifty50_symbols=_n50_syms,
        midcap_symbols=_mid_syms, smallcap_symbols=_sm_syms,
        lookback=756, ma_period=_cap_ma,
    )
    if _cap_b:
        _fig_cap = go.Figure()
        _cap_colors = {"Large Cap": "#2196F3", "Mid Cap": "#FF9800", "Small Cap": "#AB47BC"}
        for _cn, _cs in _cap_b.items():
            _fig_cap.add_trace(go.Scatter(x=_cs.index, y=_cs, name=_cn,
                                           line=dict(color=_cap_colors.get(_cn, "#888"), width=1.5)))
        _fig_cap.add_hline(y=50, line_dash="dash", line_color="#444")
        _fig_cap.update_layout(
            height=350, template="plotly_dark", margin=dict(l=50, r=20, t=30, b=30),
            yaxis_title=f"% Above {_cap_ma} DMA", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(_fig_cap, use_container_width=True)

    # Sector breadth filter
    st.markdown("**Breadth by Sector**")
    _sec_ma = st.radio("MA Period", [50, 200], horizontal=True, key="sec_breadth_ma")
    _all_sector_names = sorted(set(_smap.values())) if _smap else []
    _sel_sectors = st.multiselect("Filter Sectors", _all_sector_names,
                                   default=_all_sector_names[:5] if _all_sector_names else [],
                                   key="breadth_sector_filter")
    if _smap and _sel_sectors:
        _filtered_map = {t: s for t, s in _smap.items() if s in _sel_sectors}
        _sec_b = compute_sector_breadth_timeseries(all_stock_data, _filtered_map, lookback=756, ma_period=_sec_ma)
        if _sec_b:
            _fig_sec = go.Figure()
            _sec_palette = ["#2196F3", "#FF9800", "#26a69a", "#AB47BC", "#ef5350", "#8BC34A", "#FFD700", "#00BCD4", "#FF5722", "#9C27B0"]
            for _i, (_sn, _ss) in enumerate(_sec_b.items()):
                _fig_sec.add_trace(go.Scatter(x=_ss.index, y=_ss, name=_sn,
                                               line=dict(color=_sec_palette[_i % len(_sec_palette)], width=1.5)))
            _fig_sec.add_hline(y=50, line_dash="dash", line_color="#444")
            _fig_sec.update_layout(
                height=350, template="plotly_dark", margin=dict(l=50, r=20, t=30, b=30),
                yaxis_title=f"% Above {_sec_ma} DMA", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_fig_sec, use_container_width=True)
else:
    st.caption("Run a scan to see breadth data.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: India Risk Dashboard
# ══════════════════════════════════════════════════════════════════
if macro_data:
    st.markdown("#### India Risk Dashboard", unsafe_allow_html=True)

    # Risk matrix sentence
    st.markdown(build_risk_matrix_html(macro_data, fii_dii), unsafe_allow_html=True)

    # Risk gauge cards
    _india_risk_labels = ["India VIX", "USD/INR", "Crude Oil", "Gold", "US 10Y"]
    _irp = [l for l in _india_risk_labels if l in macro_data]
    _ir_cols = st.columns(len(_irp) if _irp else 1)
    for _irc, _irl in zip(_ir_cols, _irp):
        with _irc:
            th = RISK_GAUGE_THRESHOLDS.get(_irl)
            ctx = INDIA_RISK_CONTEXT.get(_irl, {})
            st.markdown(build_risk_gauge_card_html(_irl, macro_data[_irl], th), unsafe_allow_html=True)
            if ctx.get("india_note"):
                st.markdown(f'<div style="font-size:0.65em;color:#555;text-align:center;margin-top:-8px;">{ctx["india_note"]}</div>', unsafe_allow_html=True)

    # Yield curve
    _spread = macro_data.get("10Y-5Y Spread")
    if _spread:
        st.markdown(build_yield_curve_indicator_html(_spread["price"]), unsafe_allow_html=True)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: Index Valuation + PE Chart with Bands
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Index Valuation", unsafe_allow_html=True)
try:
    from data_fetcher import fetch_index_valuation
    import plotly.graph_objects as go

    _val = fetch_index_valuation()
    if _val.get("available"):
        _nifty_price = macro_data.get("Nifty 50", {}).get("price", 0)
        _us10y = macro_data.get("US 10Y", {}).get("price")
        st.markdown(build_valuation_card_html(
            _nifty_price, _val["pe"], NIFTY_PE_BANDS,
            _val["earnings_yield"], _us10y,
        ), unsafe_allow_html=True)

        # Nifty PE Chart with Historical Bands
        # Use Nifty price history to derive implied PE using current EPS
        _pe_chart_period = st.selectbox("PE Chart Period", ["3y", "5y", "10y"], index=0, key="pe_chart_period")
        _nifty_hist = fetch_index_history("^NSEI", period=_pe_chart_period)
        if _nifty_hist is not None and len(_nifty_hist) > 50:
            # Derive implied PE: current PE / current price * historical price
            _curr_pe = _val["pe"]
            _curr_price = float(_nifty_hist["Close"].iloc[-1])
            if _curr_price > 0 and _curr_pe > 0:
                _implied_eps = _curr_price / _curr_pe
                _hist_pe = _nifty_hist["Close"] / _implied_eps

                _fig_pe = go.Figure()
                _fig_pe.add_trace(go.Scatter(
                    x=_hist_pe.index, y=_hist_pe, name="Nifty PE (implied)",
                    line=dict(color="#2196F3", width=2),
                ))

                # PE Bands as shaded regions
                _band_pairs = [
                    (NIFTY_PE_BANDS["extreme_low"], NIFTY_PE_BANDS["cheap"], "Extreme Value", "#26a69a", 0.08),
                    (NIFTY_PE_BANDS["cheap"], NIFTY_PE_BANDS["fair_low"], "Cheap", "#26a69a", 0.05),
                    (NIFTY_PE_BANDS["fair_low"], NIFTY_PE_BANDS["fair_high"], "Fair Value", "#888888", 0.05),
                    (NIFTY_PE_BANDS["fair_high"], NIFTY_PE_BANDS["expensive"], "Expensive", "#FF9800", 0.05),
                    (NIFTY_PE_BANDS["expensive"], NIFTY_PE_BANDS["bubble"], "Bubble", "#ef5350", 0.08),
                ]
                for _lo, _hi, _lbl, _clr, _opa in _band_pairs:
                    _fig_pe.add_hrect(y0=_lo, y1=_hi, fillcolor=_clr, opacity=_opa,
                                       line_width=0, annotation_text=_lbl,
                                       annotation_position="right")

                _fig_pe.add_hline(y=NIFTY_PE_BANDS["long_term_avg"], line_dash="dash", line_color="#FFD700",
                                  annotation_text=f'LT Avg ({NIFTY_PE_BANDS["long_term_avg"]}x)')
                _fig_pe.update_layout(
                    height=400, template="plotly_dark",
                    margin=dict(l=50, r=80, t=30, b=30),
                    yaxis_title="PE Ratio", xaxis_title="",
                    yaxis=dict(range=[max(10, _hist_pe.min() * 0.9), min(35, _hist_pe.max() * 1.1)]),
                    hovermode="x unified",
                )
                st.plotly_chart(_fig_pe, use_container_width=True)

                # Implied EPS Trend
                st.markdown("**Implied Index EPS Trend**")
                _fig_eps = go.Figure()
                _trailing_eps = _nifty_hist["Close"] / _hist_pe  # = _implied_eps (constant)
                # More useful: show Nifty price / fixed PE multiples to show EPS-equivalent
                # Actually derive a rolling earnings proxy from price / PE
                _eps_12m = _nifty_hist["Close"].rolling(252).mean() / NIFTY_PE_BANDS["long_term_avg"]
                _fig_eps.add_trace(go.Scatter(
                    x=_eps_12m.index, y=_eps_12m, name="Rolling Earnings Proxy",
                    line=dict(color="#8BC34A", width=2),
                ))
                _fig_eps.update_layout(
                    height=250, template="plotly_dark",
                    margin=dict(l=50, r=20, t=30, b=30),
                    yaxis_title="Earnings (proxy)", hovermode="x unified",
                )
                st.plotly_chart(_fig_eps, use_container_width=True)

        # PE snapshots trend (accumulated over scans)
        _snaps = _val.get("snapshots", [])
        if len(_snaps) > 5:
            with st.expander("PE Snapshots (from scan history)", expanded=False):
                _fig_snap = go.Figure()
                _fig_snap.add_trace(go.Scatter(
                    x=[s["date"] for s in _snaps], y=[s["pe"] for s in _snaps],
                    name="Actual PE (NIFTYBEES proxy)", line=dict(color="#FF9800", width=2),
                ))
                _fig_snap.add_hline(y=NIFTY_PE_BANDS["long_term_avg"], line_dash="dash", line_color="#666",
                                    annotation_text=f'LT Avg ({NIFTY_PE_BANDS["long_term_avg"]}x)')
                _fig_snap.update_layout(height=250, template="plotly_dark", margin=dict(l=50, r=20, t=30, b=30),
                                        yaxis_title="PE Ratio")
                st.plotly_chart(_fig_snap, use_container_width=True)
    else:
        st.caption("Valuation data unavailable — yfinance may not have returned PE data.")
except Exception as _ve:
    st.caption(f"Valuation: {_ve}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 6: Macro Momentum (India-focused derivatives)
# ══════════════════════════════════════════════════════════════════
if macro_data:
    st.markdown("#### Macro Momentum", unsafe_allow_html=True)

    macro_derivs = compute_macro_derivatives(macro_data, MACRO_DERIVATIVE_LABELS)
    if macro_derivs:
        _md_cols = st.columns(len(macro_derivs))
        for _mdc, (_mdl, _mdd) in zip(_md_cols, macro_derivs.items()):
            with _mdc:
                _inf = _mdd["inflection"]
                _roc_v = _mdd["roc"].iloc[-1] if not _mdd["roc"].empty else 0
                _acc_v = _mdd["accel"].iloc[-1] if not _mdd["accel"].empty else 0
                st.markdown(
                    f'<div style="background:#0f0f1a;border-left:2px solid {_inf["color"]};border-radius:4px;padding:12px 14px;">'
                    f'<div style="font-size:0.68em;color:#666;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">{_mdl}</div>'
                    f'<div style="font-size:0.85em;font-weight:600;color:{_inf["color"]};">{_inf["icon"]} {_inf["label"]}</div>'
                    f'<div style="font-size:0.72em;color:#888;margin-top:4px;font-family:monospace;">ROC: {_roc_v:+.2f} &middot; Accel: {_acc_v:+.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Derivative Charts", expanded=False):
            for _mdl, _mdd in macro_derivs.items():
                _h = build_derivative_lw_html(_mdd["series"], _mdd["roc"], _mdd["accel"], _mdl)
                if _h:
                    st.components.v1.html(_h, height=380, scrolling=False)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: Earnings Season
# ══════════════════════════════════════════════════════════════════
earnings_data = st.session_state.get("earnings_season")
if not earnings_data:
    from earnings_season import load_earnings_cache
    earnings_data = load_earnings_cache()
    if earnings_data:
        st.session_state.earnings_season = earnings_data

if earnings_data:
    st.markdown("#### Earnings Season", unsafe_allow_html=True)
    st.markdown(build_earnings_season_card_html(earnings_data), unsafe_allow_html=True)
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: AI Market Summary
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Market Summary", unsafe_allow_html=True)

ai_summary = st.session_state.get("ai_summary", "")
ai_source = st.session_state.get("ai_summary_source", "")

if ai_summary:
    _src_color = "#2196F3" if "AI" in ai_source else "#FF9800"
    st.markdown(
        f'<div style="background:#0f0f1a;border-top:2px solid #5C9DFF;border-radius:6px;padding:18px 22px;margin-bottom:12px;">'
        f'<div style="color:#ccc;font-size:0.9em;line-height:1.7;">{ai_summary}</div>'
        f'<div style="color:#555;font-size:0.72em;margin-top:10px;font-family:monospace;">Source: <span style="color:{_src_color};">{ai_source}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Regenerate Summary", key="regen_summary"):
        from ai_summary import generate_market_summary
        summary, source = generate_market_summary(macro_data, regime, fii_dii, fii_dii_flows, sector_rankings)
        st.session_state.ai_summary = summary
        st.session_state.ai_summary_source = source
        save_scan_to_disk()
        st.rerun()
else:
    st.caption("Run a scan to generate market summary.")


# ══════════════════════════════════════════════════════════════════
# SECTION 9: Top Conviction Ideas
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Top Conviction Ideas", unsafe_allow_html=True)

if watchlist and sector_rankings:
    _macro_liq = st.session_state.get("macro_liquidity")
    _fii_gate = st.session_state.get("fii_gate")
    ranked = rank_candidates_by_conviction(candidates=list(watchlist), sector_rankings=sector_rankings, macro_liquidity=_macro_liq, fii_gate=_fii_gate)
    sector_ideas = get_top_ideas_by_sector(ranked, top_sectors, per_sector=3)

    if sector_ideas:
        _stabs = st.tabs([f"{s} ({len(ideas)})" for s, ideas in sector_ideas.items()])
        for _stab, (_sname, ideas) in zip(_stabs, sector_ideas.items()):
            with _stab:
                _srp = next((i+1 for i, r in enumerate(sector_rankings) if (r.get("sector") or r.get("name", "")) == _sname), None)
                st.caption(f"{'#' + str(_srp) + ' Sector' if _srp else ''} — {len(ideas)} pick{'s' if len(ideas) != 1 else ''}")
                _icols = st.columns(len(ideas))
                for _ic, idea in zip(_icols, ideas):
                    with _ic:
                        _cs = idea.get("conviction_score", 0)
                        _tn = idea.get("ticker", "").replace(".NS", "")
                        _es = idea.get("entry_setup", {}) or {}
                        _pos = idea.get("position", {})
                        _tgt = idea.get("targets", {})
                        _pill = idea.get("conviction_pillars", {})
                        _cc = "#26a69a" if _cs >= 60 else "#FF9800" if _cs >= 40 else "#ef5350"
                        _rr = idea.get("rr_ratio", 0)
                        _rrs = f'<span style="font-size:0.7em;color:#FFD700;margin-left:6px;">{_rr:.1f}R</span>' if _rr >= 3 else ''
                        st.markdown(
                            f'<div style="background:#0f0f1a;border:1px solid {_cc}44;border-radius:8px;padding:14px;text-align:center;">'
                            f'<div style="font-size:1.1em;font-weight:600;color:#e0e0e0;">{_tn}{_rrs}</div>'
                            f'<div style="font-size:1.5em;font-weight:700;color:{_cc};margin:6px 0;font-family:monospace;">{_cs:.0f}</div>'
                            f'<div style="font-size:0.65em;color:#555;text-transform:uppercase;">CONVICTION</div>'
                            f'<div style="display:flex;justify-content:center;gap:6px;margin-top:8px;">'
                            f'<span style="font-size:0.62em;color:#2196F3;">T:{_pill.get("technical",0):.0f}</span>'
                            f'<span style="font-size:0.62em;color:#8BC34A;">V:{_pill.get("value",0):.0f}</span>'
                            f'<span style="font-size:0.62em;color:#FF9800;">M:{_pill.get("macro",0):.0f}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                        if _es:
                            st.markdown(f"Entry **{_es.get('entry_price',0):.1f}** | Stop **{_es.get('effective_stop',0):.1f}** | Risk **{_es.get('risk_pct',0):.1f}%**")
                        if _pos.get("shares"):
                            st.caption(f"Shares: {_pos['shares']} | R:R {_tgt.get('reward_risk_ratio',0):.1f}")
        st.caption(f"{sum(len(v) for v in sector_ideas.values())} ideas across {len(sector_ideas)} sectors")
    else:
        st.markdown('<div style="background:#0f0f1a;border-radius:6px;padding:20px;text-align:center;color:#555;font-style:italic;border:1px solid #1e1e2e;">No high-conviction setups today — patience is alpha</div>', unsafe_allow_html=True)
else:
    st.caption("Run a scan to generate conviction rankings.")


# ══════════════════════════════════════════════════════════════════
# SECTION 10: Active Positions + Bulk Deals
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Active Positions", unsafe_allow_html=True)

positions = load_positions()
if positions:
    _pos_sd = st.session_state.get("stock_data", {})
    _pos_dd = st.session_state.get("daily_stock_data", {})
    if _pos_dd:
        _pos_sd = {**_pos_sd, **_pos_dd}
    pos_summaries = get_positions_summary(_pos_sd)
    if pos_summaries:
        _tpnl = sum(s.get("pnl", 0) for s in pos_summaries)
        _pclr = "#26a69a" if _tpnl >= 0 else "#ef5350"
        _p1, _p2 = st.columns([1, 4])
        with _p1:
            st.metric("Open Positions", len(pos_summaries))
            st.markdown(f'<div style="font-size:1.2em;font-weight:600;color:{_pclr};">P&L: {_tpnl:+,.0f}</div>', unsafe_allow_html=True)
        with _p2:
            import pandas as pd
            _pr = [{"Ticker": s["ticker"].replace(".NS",""), "Entry": f"{s['entry_price']:.1f}",
                     "Current": f"{s.get('current_price',0):.1f}" if s.get("current_price") else "N/A",
                     "P&L %": f"{s.get('pnl_pct',0):+.1f}%", "Days": s.get("days_held",0),
                     "Action": ({'SELL':'🔴','PARTIAL SELL':'🟠','ADD':'🟢','HOLD':'⚪'}.get(s.get('suggested_action','HOLD'),'') + ' ' + s.get('suggested_action','HOLD'))}
                    for s in pos_summaries[:5]]
            st.dataframe(pd.DataFrame(_pr), use_container_width=True, hide_index=True)
        if len(pos_summaries) > 5:
            st.caption(f"+{len(pos_summaries)-5} more — see Positions page")
else:
    st.caption("No active positions — add from Positions page.")

# Bulk Deals
st.markdown("**Recent Bulk Deals (Watchlist)**")
_wt = {w.get("ticker","").replace(".NS","").replace(".BO","").upper() for w in watchlist if w.get("ticker")}
_deals = []
try:
    from datetime import timedelta
    _from = (dt.datetime.now(IST) - timedelta(days=7)).strftime("%d-%m-%Y")
    _to = dt.datetime.now(IST).strftime("%d-%m-%Y")
    _all_bulk = nse_fetcher.fetch_bulk_deals(_from, _to)
    _deals = [d for d in _all_bulk if d.get("symbol","").upper() in _wt]
except Exception:
    pass
if _deals:
    import pandas as pd
    st.dataframe(pd.DataFrame([{"Date": d.get("date",""), "Symbol": d.get("symbol",""), "Client": d.get("client_name","")[:30], "Action": d.get("deal_type",""), "Qty": f"{d.get('quantity',0):,.0f}"} for d in _deals[:10]]), use_container_width=True, hide_index=True)
else:
    st.caption("No recent bulk deals in watchlist stocks.")


# ══════════════════════════════════════════════════════════════════
# SECTION 11: Global Context (collapsed)
# ══════════════════════════════════════════════════════════════════
if macro_data:
    with st.expander("Global Context (Overnight)", expanded=False):
        _gl = MACRO_GROUPS["Global Indices"]
        _gp = [l for l in _gl if l in macro_data]
        for _rs in range(0, len(_gp), 4):
            _rl = _gp[_rs:_rs+4]
            _gc = st.columns(4)
            for _c, _l in zip(_gc, _rl):
                with _c:
                    st.markdown(build_macro_card_html(_l, macro_data[_l]), unsafe_allow_html=True)
        # Currencies
        _curr = ["USD/INR", "EUR/USD", "USD/JPY"]
        _cp = [l for l in _curr if l in macro_data]
        _comm = ["Crude Oil", "Brent Crude", "Gold", "Silver", "Copper"]
        _cmp = [l for l in _comm if l in macro_data]
        _allcc = _cp + _cmp
        for _rs in range(0, len(_allcc), 4):
            _rl = _allcc[_rs:_rs+4]
            _gc = st.columns(4)
            for _c, _l in zip(_gc, _rl):
                with _c:
                    st.markdown(build_macro_card_html(_l, macro_data[_l]), unsafe_allow_html=True)

st.divider()
st.markdown("**Drill deeper:** Market Regime | Sector Rotation | Stock Opportunities | Positions | Stock Deep Dive | Watchlist")
