"""Page 1: Market Regime Analysis"""
import streamlit as st
import pandas as pd

from dashboard_helpers import (
    build_candlestick_chart,
    build_regime_gauge,
    build_breadth_chart,
    compute_breadth_timeseries,
    compute_net_new_highs_timeseries,
    compute_all_derivatives,
    build_derivative_chart,
    detect_inflection_points,
    regime_color,
    signal_color,
    resample_ohlcv,
    build_lw_candlestick_html,
    build_lw_area_chart_html,
)
from config import REGIME_CONFIG
import plotly.graph_objects as go

st.set_page_config(page_title="Market Regime", page_icon="📊", layout="wide")
st.title("Market Regime Analysis")

if "regime" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

regime = st.session_state.regime
nifty_df = st.session_state.nifty_df
all_stock_data = st.session_state.all_stock_data

# ── Compute derivatives upfront (needed for integrated view) ─────
with st.spinner("Computing momentum derivatives..."):
    deriv_results = compute_all_derivatives(nifty_df, all_stock_data)

# Count derivative signals
deriv_bullish = 0
deriv_bearish = 0
for name, d in (deriv_results or {}).items():
    sig = d.get("inflection", {}).get("signal", "")
    if "bullish" in sig:
        deriv_bullish += 1
    elif "bearish" in sig:
        deriv_bearish += 1

# ── Regime Banner (integrated levels + momentum) ─────────────────
label = regime["label"]
color = regime_color(label)
raw = regime["raw_score"]
score = regime["regime_score"]
signals = regime["signals"]

bullish_count = sum(1 for s in signals.values() if isinstance(s, dict) and s.get("score", 0) > 0)
total_signals = len(signals)

# Momentum verdict from derivatives
if deriv_bearish >= 3:
    momentum_label = "Momentum Breaking Down"
    momentum_color = "#ef5350"
elif deriv_bearish >= 2:
    momentum_label = "Momentum Fading"
    momentum_color = "#FF9800"
elif deriv_bullish >= 3:
    momentum_label = "Momentum Strengthening"
    momentum_color = "#26a69a"
elif deriv_bullish >= 2:
    momentum_label = "Momentum Improving"
    momentum_color = "#8BC34A"
else:
    momentum_label = "Momentum Mixed"
    momentum_color = "#888"

st.markdown(
    f"""
    <div style="background: {color}22; border-left: 5px solid {color};
                padding: 20px 25px; border-radius: 0 8px 8px 0; margin-bottom: 15px;">
        <div style="font-size: 2em; font-weight: 700; color: {color};">{label.upper()}</div>
        <div style="font-size: 1.1em; color: #ccc; margin-top: 5px;">
            Levels: {bullish_count} of {total_signals} bullish
            &nbsp;|&nbsp;
            <span style="color:{momentum_color};">{momentum_label}</span>
            &nbsp;|&nbsp;
            Breadth {regime.get('breadth_trend', 'stable')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Understanding Market Regime", expanded=True):
    st.markdown("""
**Two lenses on the same market:**

**Level Signals** answer: *"Where is the market right now?"*
- Is Nifty above/below key moving averages?
- What % of stocks are in uptrends?
- Are more stocks making new highs or new lows?

**Momentum Derivatives** answer: *"Which direction is it heading?"*
- Is the rate of change positive or negative?
- Is the rate of change itself accelerating or decelerating?

**How to read them together:**
| Levels | Momentum | Meaning |
|--------|----------|---------|
| Bullish | Strengthening | Full risk-on — best environment for new positions |
| Bullish | Fading | Still OK but tighten stops — momentum is rolling over |
| Bearish | Improving | Early recovery — watch for confirmation before adding |
| Bearish | Breaking Down | Full risk-off — avoid new longs, protect capital |

When levels and momentum **disagree**, momentum usually leads by 2-4 weeks. A bullish market with fading momentum often precedes a correction.
""")

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Integrated Signal + Momentum View
# ══════════════════════════════════════════════════════════════════
st.subheader("Market Health — Levels & Momentum")

st.markdown("**Level Signals** — where is the market now?")
level_cols = st.columns(len(signals))
for col, (sig_name, sig_data) in zip(level_cols, signals.items()):
    sc = sig_data["score"]
    sc_color = signal_color(sc)
    icon = "+" if sc > 0 else ("-" if sc < 0 else "~")
    with col:
        st.markdown(
            f"""<div style="background: {sc_color}15; border: 1px solid {sc_color}44;
                        border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 1.3em; color: {sc_color}; font-weight: 700;">{icon}</div>
                <div style="font-size: 0.8em; color: #ccc; margin-top: 4px;">
                    {sig_name.replace('_', ' ').title()}
                </div>
                <div style="font-size: 0.75em; color: #999; margin-top: 4px;">
                    {sig_data['detail']}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

if deriv_results:
    st.markdown("**Momentum Derivatives** — which direction is it heading?")
    deriv_cols = st.columns(len(deriv_results))
    for col, (name, d) in zip(deriv_cols, deriv_results.items()):
        inflection = d.get("inflection", {})
        sig_label = inflection.get("label", "N/A")
        sig_clr = inflection.get("color", "#888")
        sig_icon = inflection.get("icon", "--")
        sig_detail = inflection.get("detail", "")
        with col:
            st.markdown(
                f"""<div style="background:{sig_clr}15; border:1px solid {sig_clr}44;
                    border-radius:8px; padding:10px; text-align:center;">
                    <div style="font-size:1.3em; color:{sig_clr}; font-weight:700;">{sig_icon}</div>
                    <div style="font-size:0.8em; color:#ccc; margin-top:4px;">{name}</div>
                    <div style="font-size:0.8em; color:{sig_clr}; font-weight:600; margin-top:4px;">{sig_label}</div>
                    <div style="font-size:0.7em; color:#999; margin-top:2px;">{sig_detail}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ── Breadth by Weinstein Stage ─────────────────────────────────
breadth = st.session_state.get("breadth_by_stage")
if breadth:
    st.subheader("Breadth by Weinstein Stage")
    s_pcts = breadth.get("stage_pcts", {})
    s_counts = breadth.get("stage_counts", {})
    b_score = breadth.get("breadth_score", 50)
    b_label = breadth.get("breadth_label", "")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stage 1 (Basing)", f"{s_pcts.get(1, 0):.0f}%", help=f"{s_counts.get(1, 0)} stocks")
    c2.metric("Stage 2 (Advancing)", f"{s_pcts.get(2, 0):.0f}%", help=f"{s_counts.get(2, 0)} stocks")
    c3.metric("Stage 3 (Topping)", f"{s_pcts.get(3, 0):.0f}%", help=f"{s_counts.get(3, 0)} stocks")
    c4.metric("Stage 4 (Declining)", f"{s_pcts.get(4, 0):.0f}%", help=f"{s_counts.get(4, 0)} stocks")
    c5.metric("Breadth Score", f"{b_score:.0f}", delta=b_label)
    s2_bo = breadth.get("s2_with_breakouts", 0)
    if s2_bo:
        st.caption(f"{s2_bo} Stage 2 stocks with active breakouts ({breadth.get('s2_breakout_pct', 0):.1f}% of universe)")

# ── Macro Liquidity Score ─────────────────────────────────────
macro_liq = st.session_state.get("macro_liquidity")
if macro_liq:
    st.subheader("Macro Liquidity")
    ml_score = macro_liq.get("score", 50)
    ml_label = macro_liq.get("label", "")
    ml1, ml2, ml3 = st.columns(3)
    ml1.metric("Liquidity Score", f"{ml_score:.0f}/100")
    ml2.metric("Regime", ml_label)
    adj = macro_liq.get("regime_adjustment", 0)
    ml3.metric("Regime Adj", f"{adj:+d}" if adj else "0")

# ── Nifty Candlestick Chart ────────────────────────────────────
st.subheader("Nifty 50 Index")
_nifty_tf_label = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="regime_nifty_tf")
_tf_map = {"Weekly": "W", "Daily": "D", "Monthly": "ME"}
_tf = _tf_map[_nifty_tf_label]
nifty_chart_df = resample_ohlcv(nifty_df, _tf)
nifty_chart_html = build_lw_candlestick_html(nifty_chart_df, "Nifty 50", mas=[50, 200], height=500)
st.components.v1.html(nifty_chart_html, height=510)

# ── Breadth Gauges ──────────────────────────────────────────────
st.subheader("Market Breadth")
g1, g2 = st.columns(2)

breadth_50_val = signals.get("breadth_50dma", {}).get("value", 50)
breadth_200_val = signals.get("breadth_200dma", {}).get("value", 50)

with g1:
    fig = build_regime_gauge(
        breadth_50_val, "% Above 50 DMA",
        (REGIME_CONFIG["breadth_50dma_bearish"], REGIME_CONFIG["breadth_50dma_bullish"]),
    )
    st.plotly_chart(fig, use_container_width=True)

with g2:
    fig = build_regime_gauge(
        breadth_200_val, "% Above 200 DMA",
        (REGIME_CONFIG["breadth_200dma_bearish"], REGIME_CONFIG["breadth_200dma_bullish"]),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Breadth Timeseries ─────────────────────────────────────────
st.subheader("Breadth Over Time (Last 90 Days)")
with st.spinner("Computing breadth timeseries..."):
    breadth_50_ts = compute_breadth_timeseries(all_stock_data, ma_period=50, lookback=90)
    breadth_200_ts = compute_breadth_timeseries(all_stock_data, ma_period=200, lookback=90)
    area_series = []
    if not breadth_50_ts.empty:
        area_series.append({
            "name": "% > 50 DMA",
            "times": breadth_50_ts.index.strftime("%Y-%m-%d").tolist(),
            "values": breadth_50_ts.values.tolist(),
            "color": "#2196F3",
            "topColor": "rgba(33,150,243,0.3)",
            "bottomColor": "rgba(33,150,243,0.0)",
        })
    if not breadth_200_ts.empty:
        area_series.append({
            "name": "% > 200 DMA",
            "times": breadth_200_ts.index.strftime("%Y-%m-%d").tolist(),
            "values": breadth_200_ts.values.tolist(),
            "color": "#FF9800",
            "topColor": "rgba(255,152,0,0.3)",
            "bottomColor": "rgba(255,152,0,0.0)",
        })
    if area_series:
        breadth_html = build_lw_area_chart_html(area_series, title="Market Breadth Over Time", height=400)
        st.components.v1.html(breadth_html, height=410)
    else:
        st.caption("Insufficient data for breadth chart.")

# ── Net New Highs / Lows ───────────────────────────────────────
st.subheader("Net New Highs / Lows (Last 90 Days)")
with st.spinner("Computing new highs/lows..."):
    nh_df = compute_net_new_highs_timeseries(all_stock_data, lookback=90)
    if not nh_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=nh_df.index, y=nh_df["New Highs"], name="New Highs",
            marker_color="#26a69a",
        ))
        fig.add_trace(go.Bar(
            x=nh_df.index, y=nh_df["New Lows"], name="New Lows",
            marker_color="#ef5350",
        ))
        fig.add_trace(go.Scatter(
            x=nh_df.index, y=nh_df["Net"], name="Net",
            line=dict(color="#2196F3", width=2),
        ))
        fig.update_layout(
            barmode="relative",
            height=400,
            template="plotly_dark",
            yaxis_title="Count",
            margin=dict(l=50, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Insufficient data for new highs/lows chart.")

# ── Derivative Charts (expandable) ─────────────────────────────
if deriv_results:
    st.subheader("Derivative Charts")
    st.caption("Expand any series to see the ROC and acceleration plots.")
    for name, d in deriv_results.items():
        if d["roc"].empty:
            continue
        with st.expander(f"{name} — Derivative Chart", expanded=True):
            fig = build_derivative_chart(d["series"], d["roc"], d["accel"], name, lookback=90)
            st.plotly_chart(fig, use_container_width=True)
