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

from config import POSITION_CONFIG, SECTOR_CONFIG, REGIME_CONFIG, SMART_MONEY_CONFIG, MACRO_GROUPS, RISK_GAUGE_THRESHOLDS, MACRO_DERIVATIVE_LABELS
from data_fetcher import (
    fetch_index_data, fetch_all_stock_data, fetch_sector_data,
    fetch_price_data, fetch_macro_data, get_sector_map,
)
from market_regime import compute_regime
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
)

# ── Scan Cache (disk persistence) ──────────────────────────────
CACHE_DIR = Path(__file__).parent / "scan_cache"
CACHE_FILE = CACHE_DIR / "last_scan.pkl"

CACHE_KEYS = [
    "scan_date", "capital", "nifty_df", "all_stock_data", "sector_data",
    "regime", "sector_rankings", "top_sectors", "stock_data",
    "screened_stocks", "stage2_candidates", "all_stage2_stocks", "final_watchlist",
    "macro_data", "quality_radar", "universe_count",
    "ai_summary", "ai_summary_source",
    "earnings_season",
    "macro_liquidity", "fii_gate", "breadth_by_stage",
]


def save_scan_to_disk():
    """Persist current scan results to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    data = {k: st.session_state[k] for k in CACHE_KEYS if k in st.session_state}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


def load_scan_from_disk():
    """Load previous scan results from disk into session state."""
    if not CACHE_FILE.exists():
        return False
    try:
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except Exception:
        return False


def is_cache_stale(max_age_hours: int = 24) -> bool:
    """Check if the cached scan data is older than max_age_hours."""
    scan_date_str = st.session_state.get("scan_date")
    if not scan_date_str:
        return True
    try:
        scan_dt = dt.datetime.strptime(scan_date_str, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        age = dt.datetime.now(IST) - scan_dt
        return age.total_seconds() > max_age_hours * 3600
    except Exception:
        return True


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

    run_scan = st.button("Run Scan", type="primary", use_container_width=True)

    if "scan_date" in st.session_state:
        st.caption(f"Last scan: {st.session_state.scan_date}")
        if "universe_count" in st.session_state:
            st.caption(f"Universe: {st.session_state.universe_count} stocks")
        if is_cache_stale():
            st.warning("Data is stale (>24h old)", icon="⚠️")

    st.divider()

    run_earnings = st.button("Earnings Scan", use_container_width=True)
    if "earnings_season" in st.session_state and st.session_state.earnings_season:
        es = st.session_state.earnings_season
        st.caption(f"Earnings: {es.get('quarter_label', '?')} | {es.get('reported_count', 0)}/{es.get('total_universe', 0)}")

    st.divider()
    st.caption("Built with Streamlit + Plotly")


# ── Scan Orchestration ──────────────────────────────────────────
def run_pipeline_scan():
    """Run the full 5-layer pipeline and store results in session state."""
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

            st.session_state.scan_date = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            st.session_state.capital = capital

            # Save to disk for persistence across restarts
            st.write("Saving scan results to disk...")
            save_scan_to_disk()
            progress.progress(100)

            status.update(label="Scan complete!", state="complete")

    finally:
        sys.stdout = old_stdout


if run_scan:
    run_pipeline_scan()


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


# ── Home Page — Morning Briefing ──────────────────────────────
scan_date_str = st.session_state.get("scan_date", "")
st.markdown(
    f'<div style="margin-bottom:1.5rem;">'
    f'<h1 style="margin-bottom:2px;font-size:1.6em;font-weight:700;color:#e0e0e0;">Morning Briefing</h1>'
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

# Fetch FII/DII data (current day + historical)
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
# Nifty 50 Hero Card
# ══════════════════════════════════════════════════════════════════
if macro_data:
    nifty_data = macro_data.get("Nifty 50")
    if nifty_data:
        _nifty_price = nifty_data.get("price", 0)
        _nifty_chg = nifty_data.get("change_pct", 0)
        _nifty_color = "#26a69a" if _nifty_chg >= 0 else "#ef5350"
        _nifty_arrow = "&#9650;" if _nifty_chg >= 0 else "&#9660;"
        st.markdown(
            f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:6px;padding:16px 24px;margin-bottom:1.2rem;display:flex;align-items:center;justify-content:space-between;">'
            f'<div>'
            f'<span style="font-size:0.7em;color:#666;text-transform:uppercase;letter-spacing:0.1em;">NIFTY 50</span>'
            f'<div style="font-size:1.8em;font-weight:700;color:#e8e8e8;font-family:monospace;">{_nifty_price:,.1f}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:1.1em;color:{_nifty_color};font-family:monospace;">{_nifty_arrow} {_nifty_chg:+.2f}%</div>'
            f'<div style="font-size:0.72em;color:#555;font-family:monospace;">{nifty_data.get("change", 0):+,.1f} pts</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 1: Global Markets Overnight
# ══════════════════════════════════════════════════════════════════
if macro_data:
    st.markdown("#### Global Markets Overnight")
    _global_labels = MACRO_GROUPS["Global Indices"]
    _global_present = [l for l in _global_labels if l in macro_data]
    for row_start in range(0, len(_global_present), 4):
        row_labels = _global_present[row_start:row_start + 4]
        cols = st.columns(4)
        for col, lbl in zip(cols, row_labels):
            with col:
                st.markdown(build_macro_card_html(lbl, macro_data[lbl]), unsafe_allow_html=True)

    with st.expander("1-Year Trend", expanded=True):
        trend_html = build_macro_trend_lw_html(macro_data, _global_present, title="Global Indices — 1Y % Change")
        if trend_html:
            st.components.v1.html(trend_html, height=370, scrolling=False)
        else:
            st.caption("Trend data not available — run a fresh scan.")

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: Risk Gauges
# ══════════════════════════════════════════════════════════════════
if macro_data:
    st.markdown("#### Risk Gauges")
    _risk_labels = ["VIX", "India VIX", "Dollar Index", "US 10Y", "Crude Oil", "Gold"]
    _risk_present = [l for l in _risk_labels if l in macro_data]
    cols = st.columns(len(_risk_present) if _risk_present else 1)
    for col, lbl in zip(cols, _risk_present):
        with col:
            th = RISK_GAUGE_THRESHOLDS.get(lbl)
            st.markdown(build_risk_gauge_card_html(lbl, macro_data[lbl], th), unsafe_allow_html=True)

    # Yield curve spread — with spacer above
    spread_data = macro_data.get("10Y-5Y Spread")
    if spread_data:
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.markdown(build_yield_curve_indicator_html(spread_data["price"]), unsafe_allow_html=True)

    with st.expander("1-Year Trend", expanded=True):
        trend_html = build_macro_trend_lw_html(macro_data, _risk_present, title="Risk Gauges — 1Y % Change")
        if trend_html:
            st.components.v1.html(trend_html, height=370, scrolling=False)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 2b: Macro Momentum (Derivatives)
# ══════════════════════════════════════════════════════════════════
if macro_data:
    st.markdown("#### Macro Momentum")

    macro_derivs = compute_macro_derivatives(macro_data, MACRO_DERIVATIVE_LABELS)
    if macro_derivs:
        cols = st.columns(len(macro_derivs))
        for col, (label, d) in zip(cols, macro_derivs.items()):
            with col:
                inf = d["inflection"]
                roc_val = d["roc"].iloc[-1] if not d["roc"].empty else 0
                accel_val = d["accel"].iloc[-1] if not d["accel"].empty else 0
                st.markdown(
                    f'<div style="background:#0f0f1a;border-left:2px solid {inf["color"]};'
                    f'border-radius:4px;padding:12px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:0.68em;color:#666;text-transform:uppercase;'
                    f'letter-spacing:0.06em;margin-bottom:6px;">{label}</div>'
                    f'<div style="font-size:0.85em;font-weight:600;color:{inf["color"]};">'
                    f'{inf["icon"]} {inf["label"]}</div>'
                    f'<div style="font-size:0.72em;color:#888;margin-top:4px;font-family:monospace;">'
                    f'ROC: {roc_val:+.2f} &middot; Accel: {accel_val:+.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Macro Derivative Charts", expanded=True):
            for label, d in macro_derivs.items():
                html = build_derivative_lw_html(d["series"], d["roc"], d["accel"], label)
                if html:
                    st.components.v1.html(html, height=380, scrolling=False)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Currencies & Commodities
# ══════════════════════════════════════════════════════════════════
if macro_data:
    st.markdown("#### Currencies & Commodities")
    _curr_labels = MACRO_GROUPS["Currencies"]
    _curr_present = [l for l in _curr_labels if l in macro_data]
    _comm_labels = MACRO_GROUPS["Commodities"]
    _comm_present = [l for l in _comm_labels if l in macro_data]

    # All 8 items in rows of 4
    _all_cc = _curr_present + _comm_present
    for row_start in range(0, len(_all_cc), 4):
        row_labels = _all_cc[row_start:row_start + 4]
        cols = st.columns(4)
        for col, lbl in zip(cols, row_labels):
            with col:
                st.markdown(build_macro_card_html(lbl, macro_data[lbl]), unsafe_allow_html=True)

    with st.expander("1-Year Trends", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            trend_html = build_macro_trend_lw_html(macro_data, _curr_present, title="Currencies — 1Y % Change", height=300)
            if trend_html:
                st.components.v1.html(trend_html, height=320, scrolling=False)
        with c2:
            trend_html = build_macro_trend_lw_html(macro_data, _comm_present, title="Commodities — 1Y % Change", height=300)
            if trend_html:
                st.components.v1.html(trend_html, height=320, scrolling=False)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: India Market Health
# ══════════════════════════════════════════════════════════════════
st.markdown("#### India Market Health")

label = regime["label"]
color = regime_color(label)
score = regime["regime_score"]

# Regime badge
signals = regime.get("signals", {})
bullish_count = sum(1 for s in signals.values() if isinstance(s, dict) and s.get("score", 0) > 0)
bearish_count = sum(1 for s in signals.values() if isinstance(s, dict) and s.get("score", 0) < 0)
total_signals = bullish_count + bearish_count + sum(
    1 for s in signals.values() if isinstance(s, dict) and s.get("score", 0) == 0
)
breadth_trend = regime.get("breadth_trend", "stable")
trend_label = {"improving": "Breadth Improving", "deteriorating": "Breadth Weakening"}.get(
    breadth_trend, "Breadth Stable"
)
trend_icon_color = {"improving": "#26a69a", "deteriorating": "#ef5350"}.get(breadth_trend, "#888")

# FII/DII inline
fii_net_str = ""
if fii_dii:
    fii_net = fii_dii.get("fii_net", 0)
    dii_net = fii_dii.get("dii_net", 0)
    fii_color = "#26a69a" if fii_net >= 0 else "#ef5350"
    dii_color = "#26a69a" if dii_net >= 0 else "#ef5350"
    fii_net_str = (
        f'<span style="margin-left:15px; font-size:0.9em;">'
        f'FII: <span style="color:{fii_color}; font-weight:600;">{fii_net:+,.0f} Cr</span>'
        f' &nbsp; DII: <span style="color:{dii_color}; font-weight:600;">{dii_net:+,.0f} Cr</span>'
        f'</span>'
    )

st.markdown(
    f'<div style="background:{color}0d;border-left:3px solid {color};'
    f'padding:10px 18px;border-radius:0 6px 6px 0;margin-bottom:10px;">'
    f'<span style="font-size:1.4em;font-weight:800;color:{color};letter-spacing:0.02em;">{label.upper()}</span>'
    f'<span style="font-size:0.85em;margin-left:12px;color:#999;">'
    f'{bullish_count}/{total_signals} bullish &middot; '
    f'<span style="color:{trend_icon_color};">{trend_label}</span>'
    f'</span>'
    f'{fii_net_str}'
    f'</div>',
    unsafe_allow_html=True,
)

# Signal breakdown chips
signal_names = {
    "index_vs_200dma": "Nifty vs 200 DMA",
    "ma_crossover": "50/200 DMA Cross",
    "breadth_50dma": "Stocks > 50 DMA",
    "breadth_200dma": "Stocks > 200 DMA",
    "net_new_highs": "New Highs vs Lows",
}
signal_chips = []
for key, display_name in signal_names.items():
    sig = signals.get(key, {})
    sc = sig.get("score", 0)
    if sc > 0:
        chip_color, chip_bg = "#26a69a", "#26a69a22"
        icon = "+"
    elif sc < 0:
        chip_color, chip_bg = "#ef5350", "#ef535022"
        icon = "-"
    else:
        chip_color, chip_bg = "#888", "#88888822"
        icon = "~"
    detail = sig.get("detail", "")
    paren_start = detail.find("(")
    short_detail = detail[paren_start:] if paren_start >= 0 else ""
    signal_chips.append(
        f'<span style="display:inline-block; background:{chip_bg}; border:1px solid {chip_color}33;'
        f' border-radius:6px; padding:4px 10px; margin:2px 4px; font-size:0.85em;">'
        f'<span style="color:{chip_color}; font-weight:700;">{icon}</span> '
        f'<span style="color:#ccc;">{display_name}</span> '
        f'<span style="color:#888; font-size:0.85em;">{short_detail}</span>'
        f'</span>'
    )

st.markdown(
    f'<div style="margin-bottom:8px;">{"".join(signal_chips)}</div>',
    unsafe_allow_html=True,
)

# Breadth by stage distribution
breadth = st.session_state.get("breadth_by_stage")
if breadth:
    s_pcts = breadth.get("stage_pcts", {})
    b_score = breadth.get("breadth_score", 50)
    b_label = breadth.get("breadth_label", "")
    b_color = "#26a69a" if b_score >= 55 else "#ef5350" if b_score < 45 else "#FF9800"
    st.markdown(
        f'<div style="display:flex;gap:12px;align-items:center;margin:6px 0 10px 0;">'
        f'<span style="font-size:0.75em;color:#666;">STAGE BREADTH:</span>'
        f'<span style="font-size:0.75em;background:#2196F322;color:#2196F3;padding:2px 8px;border-radius:4px;">S1: {s_pcts.get(1, 0):.0f}%</span>'
        f'<span style="font-size:0.75em;background:#26a69a22;color:#26a69a;padding:2px 8px;border-radius:4px;">S2: {s_pcts.get(2, 0):.0f}%</span>'
        f'<span style="font-size:0.75em;background:#FF980022;color:#FF9800;padding:2px 8px;border-radius:4px;">S3: {s_pcts.get(3, 0):.0f}%</span>'
        f'<span style="font-size:0.75em;background:#ef535022;color:#ef5350;padding:2px 8px;border-radius:4px;">S4: {s_pcts.get(4, 0):.0f}%</span>'
        f'<span style="font-size:0.75em;font-weight:600;color:{b_color};">{b_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# FII gate status
fii_gate_data = st.session_state.get("fii_gate")
if fii_gate_data and fii_gate_data.get("gated"):
    gate_color = "#ef5350" if fii_gate_data["gate_level"] == "severe" else "#FF9800"
    st.markdown(
        f'<div style="background:{gate_color}12;border-left:2px solid {gate_color};'
        f'border-radius:0 4px 4px 0;padding:6px 12px;margin-bottom:8px;font-size:0.8em;">'
        f'<span style="color:{gate_color};font-weight:600;">FII GATE: {fii_gate_data["gate_level"].upper()}</span>'
        f' <span style="color:#888;">{fii_gate_data["reason"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Multi-timeframe cumulative flows table
if fii_dii_flows:
    st.markdown("**Cumulative Net Flows (Cr)**")

    def _fmt_flow(val):
        """Format flow value with color."""
        if val is None:
            return '<span style="color:#555;">—</span>'
        color = "#26a69a" if val >= 0 else "#ef5350"
        if abs(val) >= 10000:
            display = f"{val / 1000:+,.1f}K"
        else:
            display = f"{val:+,.0f}"
        return f'<span style="color:{color}; font-weight:600;">{display}</span>'

    timeframe_order = ["1w", "2w", "1m", "3m", "6m", "1y", "2y", "5y"]
    header_cells = "".join(
        f'<th style="padding:6px 10px; color:#666; font-size:0.8em; text-align:center; text-transform:uppercase;">{tf}</th>'
        for tf in timeframe_order
    )

    fii_cells = ""
    dii_cells = ""
    for tf in timeframe_order:
        flow = fii_dii_flows.get(tf, {})
        fii_val = flow.get("fii_net")
        dii_val = flow.get("dii_net")
        days = flow.get("days_available", 0)
        fii_cells += f'<td style="padding:6px 10px; text-align:center;">{_fmt_flow(fii_val) if days > 0 else _fmt_flow(None)}</td>'
        dii_cells += f'<td style="padding:6px 10px; text-align:center;">{_fmt_flow(dii_val) if days > 0 else _fmt_flow(None)}</td>'

    st.markdown(
        f"""<table style="width:100%; border-collapse:collapse; background:#0f0f1a; border-radius:6px; overflow:hidden; font-family:monospace;">
            <thead>
                <tr style="border-bottom:1px solid #1e1e2e;">
                    <th style="padding:6px 10px; color:#666; font-size:0.8em; text-align:left; width:80px;"></th>
                    {header_cells}
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #1e1e2e;">
                    <td style="padding:6px 10px; font-weight:600; color:#999;">FII</td>
                    {fii_cells}
                </tr>
                <tr>
                    <td style="padding:6px 10px; font-weight:600; color:#999;">DII</td>
                    {dii_cells}
                </tr>
            </tbody>
        </table>""",
        unsafe_allow_html=True,
    )

    if fii_dii_history is not None and not fii_dii_history.empty:
        import pandas as pd
        earliest = pd.to_datetime(fii_dii_history["date"]).min().strftime("%d %b %Y")
        latest = pd.to_datetime(fii_dii_history["date"]).max().strftime("%d %b %Y")
        total_days = len(fii_dii_history)
        st.caption(f"History: {earliest} to {latest} ({total_days} trading days). Data builds up with each scan.")

elif not fii_dii:
    st.caption("FII/DII data unavailable — NSE API may be down. Data will accumulate with each scan.")


# ══════════════════════════════════════════════════════════════════
# SECTION 4a: Earnings Season
# ══════════════════════════════════════════════════════════════════
earnings_data = st.session_state.get("earnings_season")
if not earnings_data:
    from earnings_season import load_earnings_cache
    earnings_data = load_earnings_cache()
    if earnings_data:
        st.session_state.earnings_season = earnings_data

if earnings_data:
    st.markdown("#### Earnings Season")
    st.markdown(build_earnings_season_card_html(earnings_data), unsafe_allow_html=True)
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 4b: Sector Momentum Shifts
# ══════════════════════════════════════════════════════════════════
sector_data = st.session_state.get("sector_data", {})
nifty_df_for_rs = st.session_state.get("nifty_df")
if sector_data and nifty_df_for_rs is not None:
    rs_df = compute_all_sector_rs_timeseries(sector_data, nifty_df_for_rs)
    if not rs_df.empty:
        emerging = []
        fading = []
        for sector in rs_df.columns:
            rs_series = rs_df[sector].dropna()
            if len(rs_series) < 60:
                continue
            d = compute_derivatives(rs_series)
            rs_level = rs_series.iloc[-1]
            inf = detect_inflection_points(d["roc"], d["accel"], level=rs_level)
            is_top = sector in top_sectors
            if inf["signal"] in ("bullish_inflection", "bullish_thrust", "pullback_slowing") and not is_top:
                emerging.append((sector, inf))
            elif inf["signal"] in ("bearish_inflection", "rolling_over", "recovery_fading", "bearish_breakdown") and is_top:
                fading.append((sector, inf))

        if emerging or fading:
            st.markdown("#### Sector Momentum Shifts")
            c1, c2 = st.columns(2)
            with c1:
                if emerging:
                    st.markdown("**Emerging** (non-top sectors gaining momentum)")
                    for sector, inf in emerging[:4]:
                        st.markdown(
                            f'<div style="background:{inf["color"]}12;border-left:2px solid {inf["color"]};'
                            f'border-radius:0 4px 4px 0;padding:8px 12px;margin-bottom:6px;">'
                            f'<span style="font-weight:700;color:#ccc;">{sector}</span>'
                            f'<span style="color:{inf["color"]};margin-left:8px;font-size:0.85em;">'
                            f'{inf["icon"]} {inf["label"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No emerging sectors detected.")
            with c2:
                if fading:
                    st.markdown("**Fading** (top sectors losing momentum)")
                    for sector, inf in fading[:4]:
                        st.markdown(
                            f'<div style="background:{inf["color"]}12;border-left:2px solid {inf["color"]};'
                            f'border-radius:0 4px 4px 0;padding:8px 12px;margin-bottom:6px;">'
                            f'<span style="font-weight:700;color:#ccc;">{sector}</span>'
                            f'<span style="color:{inf["color"]};margin-left:8px;font-size:0.85em;">'
                            f'{inf["icon"]} {inf["label"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("All top sectors holding strong.")
            st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: AI Market Summary
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Market Summary")

ai_summary = st.session_state.get("ai_summary", "")
ai_source = st.session_state.get("ai_summary_source", "")

if ai_summary:
    source_color = "#2196F3" if "AI" in ai_source else "#FF9800"
    st.markdown(
        f'<div style="background:#0f0f1a;border-top:2px solid #5C9DFF;'
        f'border-radius:6px;padding:18px 22px;margin-bottom:12px;">'
        f'<div style="color:#ccc;font-size:0.9em;line-height:1.7;">{ai_summary}</div>'
        f'<div style="color:#555;font-size:0.72em;margin-top:10px;font-family:monospace;">'
        f'Source: <span style="color:{source_color};">{ai_source}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Regenerate Summary", key="regen_summary"):
        from ai_summary import generate_market_summary
        sector_rankings = st.session_state.get("sector_rankings", [])
        summary, source = generate_market_summary(
            macro_data, regime, fii_dii, fii_dii_flows, sector_rankings,
        )
        st.session_state.ai_summary = summary
        st.session_state.ai_summary_source = source
        save_scan_to_disk()
        st.rerun()
else:
    st.caption("Run a scan to generate market summary.")


# ══════════════════════════════════════════════════════════════════
# SECTION 6: Conviction Ideas + Positions + Bulk Deals
# ══════════════════════════════════════════════════════════════════
st.markdown("#### Top Conviction Ideas")

with st.expander("How Conviction Scores Work", expanded=False):
    st.markdown("""
**Tri-Factor Conviction Score** (0-100) combines three pillars:

| Pillar | Weight | Components |
|--------|--------|------------|
| **Technical** | 50% | Sector rank, Stage 2 score, base count, RS percentile, accumulation, consolidation quality, weekly confirmation |
| **Value** | 25% | ROIC history, FCF yield, fortress balance sheet, DCF margin of safety, competitive moat |
| **Macro** | 25% | Macro liquidity regime, earnings acceleration, FII/DII flow gating |

**Bonus** (up to +10): VCP pattern, bulk deals, volume surge, delivery %, asymmetric R:R.

**60+** = high conviction (green). **40-60** = moderate (orange). **<40** = lower (red). **T/V/M** = Technical/Value/Macro pillar scores.
""")

sector_rankings = st.session_state.get("sector_rankings", [])
stage2_candidates = st.session_state.get("stage2_candidates", [])

if watchlist and sector_rankings:
    _macro_liq = st.session_state.get("macro_liquidity")
    _fii_gate = st.session_state.get("fii_gate")
    ranked = rank_candidates_by_conviction(
        candidates=list(watchlist),
        sector_rankings=sector_rankings,
        macro_liquidity=_macro_liq,
        fii_gate=_fii_gate,
    )
    sector_ideas = get_top_ideas_by_sector(ranked, top_sectors, per_sector=3)

    if sector_ideas:
        sector_tabs = st.tabs([f"{s} ({len(ideas)})" for s, ideas in sector_ideas.items()])
        for tab, (sector_name, ideas) in zip(sector_tabs, sector_ideas.items()):
            with tab:
                sector_rank_pos = next(
                    (i + 1 for i, r in enumerate(sector_rankings)
                     if (r.get("sector") or r.get("name", "")) == sector_name),
                    None,
                )
                rank_label = f"#{sector_rank_pos} Sector" if sector_rank_pos else ""
                st.caption(f"{rank_label} — {len(ideas)} pick{'s' if len(ideas) != 1 else ''}")

                idea_cols = st.columns(len(ideas))
                for idx, (col, idea) in enumerate(zip(idea_cols, ideas)):
                    with col:
                        conv_score = idea.get("conviction_score", 0)
                        ticker_name = idea.get("ticker", "").replace(".NS", "")
                        es = idea.get("entry_setup", {}) or {}
                        pos = idea.get("position", {})
                        targets = idea.get("targets", {})
                        vcp = idea.get("vcp")

                        rationale = []
                        s2_score = idea.get("stage", {}).get("s2_score", 0)
                        if s2_score == 7:
                            rationale.append("Perfect S2")
                        elif s2_score >= 5:
                            rationale.append(f"S2: {s2_score}/7")
                        if vcp and vcp.get("is_vcp"):
                            rationale.append("VCP")
                        breakout = idea.get("breakout", {})
                        if breakout and breakout.get("base_number", 99) <= 2:
                            rationale.append(f"Base #{breakout['base_number']}")
                        accum = idea.get("accumulation_ratio", 0)
                        if accum and accum > 1.3:
                            rationale.append(f"Accum {accum:.1f}x")

                        conv_color = "#26a69a" if conv_score >= 60 else "#FF9800" if conv_score >= 40 else "#ef5350"
                        pillars = idea.get("conviction_pillars", {})
                        tech_s = pillars.get("technical", 0)
                        val_s = pillars.get("value", 0)
                        macro_s = pillars.get("macro", 0)
                        ea = idea.get("earnings_analysis", {})
                        ea_trend = ea.get("trend", "")
                        ea_color = {"accelerating": "#26a69a", "decelerating": "#ef5350"}.get(ea_trend, "#888")
                        ea_badge = f'<span style="font-size:0.6em;background:{ea_color}22;color:{ea_color};padding:2px 5px;border-radius:3px;margin-left:4px;">{ea_trend}</span>' if ea_trend else ''
                        rr = idea.get("rr_ratio", 0)
                        rr_str = f'<span style="font-size:0.7em;color:#FFD700;margin-left:6px;">{rr:.1f}R</span>' if rr >= 3 else ''

                        st.markdown(
                            f'<div style="background:#0f0f1a;border:1px solid {conv_color}44;border-radius:8px;padding:14px;text-align:center;">'
                            f'<div style="font-size:1.1em;font-weight:600;color:#e0e0e0;">{ticker_name}{rr_str}</div>'
                            f'<div style="font-size:1.5em;font-weight:700;color:{conv_color};margin:6px 0;font-family:monospace;">{conv_score:.0f}</div>'
                            f'<div style="font-size:0.65em;color:#555;text-transform:uppercase;letter-spacing:0.1em;">CONVICTION{ea_badge}</div>'
                            f'<div style="display:flex;justify-content:center;gap:6px;margin-top:8px;">'
                            f'<span style="font-size:0.62em;color:#2196F3;">T:{tech_s:.0f}</span>'
                            f'<span style="font-size:0.62em;color:#8BC34A;">V:{val_s:.0f}</span>'
                            f'<span style="font-size:0.62em;color:#FF9800;">M:{macro_s:.0f}</span>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if es:
                            st.markdown(
                                f"Entry **{es.get('entry_price', 0):.1f}** | "
                                f"Stop **{es.get('effective_stop', 0):.1f}** | "
                                f"Risk **{es.get('risk_pct', 0):.1f}%**"
                            )
                        if pos.get("shares"):
                            st.caption(f"Shares: {pos['shares']} | R:R {targets.get('reward_risk_ratio', 0):.1f}")
                        if rationale:
                            st.caption(" | ".join(rationale))

                        why_parts = []
                        if s2_score >= 6:
                            why_parts.append(f"Stage 2 ({s2_score}/7)")
                        if vcp and vcp.get("is_vcp"):
                            why_parts.append("VCP breakout")
                        risk_pct = es.get("risk_pct", 0) if es else 0
                        if risk_pct and 0 < risk_pct < 5:
                            why_parts.append(f"low risk ({risk_pct:.1f}%)")
                        if accum and accum > 1.3:
                            why_parts.append(f"accumulation ({accum:.1f}x)")
                        if why_parts:
                            st.caption(f"Why: {', '.join(why_parts)}.")

        total = sum(len(v) for v in sector_ideas.values())
        st.caption(f"{total} ideas across {len(sector_ideas)} sectors")
    else:
        st.markdown(
            '<div style="background:#0f0f1a; border-radius:6px; padding:20px; text-align:center;'
            ' color:#555; font-style:italic; margin:10px 0; border:1px solid #1e1e2e;">'
            'No high-conviction setups today — patience is alpha</div>',
            unsafe_allow_html=True,
        )
else:
    st.caption("Run a scan to generate conviction rankings.")


# ── Active Positions ──────────────────────────────────────────
st.markdown("#### Active Positions")

positions = load_positions()
if positions:
    stock_data = st.session_state.get("stock_data", {})
    pos_summaries = get_positions_summary(stock_data)

    if pos_summaries:
        total_open_pnl = sum(s.get("pnl", 0) for s in pos_summaries)
        pnl_color = "#26a69a" if total_open_pnl >= 0 else "#ef5350"

        pos_col1, pos_col2 = st.columns([1, 4])
        with pos_col1:
            st.metric("Open Positions", len(pos_summaries))
            st.markdown(
                f'<div style="font-size:1.2em; font-weight:600; color:{pnl_color};">'
                f'Open P&L: {total_open_pnl:+,.0f}</div>',
                unsafe_allow_html=True,
            )
        with pos_col2:
            import pandas as pd
            pos_rows = []
            for s in pos_summaries[:5]:
                action = s.get("suggested_action", "HOLD")
                action_icons = {"SELL": "🔴", "PARTIAL SELL": "🟠", "ADD": "🟢", "HOLD": "⚪"}
                pos_rows.append({
                    "Ticker": s["ticker"].replace(".NS", ""),
                    "Entry": f"{s['entry_price']:.1f}",
                    "Current": f"{s.get('current_price', 0):.1f}" if s.get("current_price") else "N/A",
                    "P&L %": f"{s.get('pnl_pct', 0):+.1f}%",
                    "Days": s.get("days_held", 0),
                    "Action": f"{action_icons.get(action, '')} {action}",
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        if len(pos_summaries) > 5:
            st.caption(f"+{len(pos_summaries) - 5} more — see Positions page")
else:
    st.caption("No active positions — add positions from the Positions page.")


# ── Bulk Deals ────────────────────────────────────────────────
st.markdown("**Recent Bulk Deals (Watchlist Stocks)**")
watchlist_tickers = set()
for w in watchlist:
    t = w.get("ticker", "").replace(".NS", "").replace(".BO", "").upper()
    if t:
        watchlist_tickers.add(t)

recent_deals = []
try:
    from datetime import timedelta
    from_7d = (dt.datetime.now(IST) - timedelta(days=7)).strftime("%d-%m-%Y")
    to_7d = dt.datetime.now(IST).strftime("%d-%m-%Y")
    all_bulk = nse_fetcher.fetch_bulk_deals(from_7d, to_7d)
    recent_deals = [d for d in all_bulk if d.get("symbol", "").upper() in watchlist_tickers]
except Exception:
    pass

if recent_deals:
    import pandas as pd
    deal_rows = []
    for d in recent_deals[:10]:
        deal_rows.append({
            "Date": d.get("date", ""),
            "Symbol": d.get("symbol", ""),
            "Client": d.get("client_name", "")[:30],
            "Action": d.get("deal_type", ""),
            "Qty": f"{d.get('quantity', 0):,.0f}",
        })
    st.dataframe(pd.DataFrame(deal_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No recent bulk deals in watchlist stocks.")


# ── Quick Navigation ──────────────────────────────────────────
st.divider()
st.markdown("""
**Drill deeper via sidebar pages:**
Market Regime | Sector Rotation | Stock Opportunities | Positions | Stock Deep Dive | Watchlist
""")
