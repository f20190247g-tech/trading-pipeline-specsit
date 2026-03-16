"""Page 8: FII & DII Institutional Flow Dashboard

Inspired by MrChartist/fii-dii-data — rebuilt natively in Streamlit
using our existing NSE data fetcher infrastructure.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from nse_data_fetcher import get_nse_fetcher, compute_fii_dii_flows

st.set_page_config(page_title="FII & DII Flows", page_icon="💰", layout="wide")

# ── Styling ─────────────────────────────────────────────────────────
st.markdown("""<style>
.fii-hero { background: linear-gradient(135deg, #1a1a2e 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.flow-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; }
.num-big { font-size: 32px; font-weight: 800; font-family: monospace; letter-spacing: -1px; }
.num-med { font-size: 22px; font-weight: 700; font-family: monospace; }
.t-green { color: #00D09E; } .t-red { color: #FB3640; } .t-muted { color: #8b949e; }
.streak-pill { display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 700; }
.streak-sell { background: rgba(251,54,64,0.15); color: #FB3640; border: 1px solid rgba(251,54,64,0.3); }
.streak-buy { background: rgba(0,208,158,0.15); color: #00D09E; border: 1px solid rgba(0,208,158,0.3); }
.heatcell { display: inline-block; width: 16px; height: 16px; border-radius: 3px; margin: 1px; }
</style>""", unsafe_allow_html=True)

st.title("FII & DII Institutional Flows")
st.caption("Institutional money flow tracker for Indian equity markets")


# ── Data Loading ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching FII/DII history from NSE...")
def load_fii_dii_data():
    """Load historical FII/DII data using our NSE fetcher."""
    nse = get_nse_fetcher()
    today_data = nse.fetch_fii_dii_data()
    history = nse.fetch_fii_dii_historical(days=1825)
    flows = {}
    if history is not None and not history.empty:
        flows = compute_fii_dii_flows(history)
    return today_data, history, flows


try:
    today_data, history_df, flow_data = load_fii_dii_data()
except Exception as e:
    st.error(f"Failed to fetch FII/DII data: {e}")
    st.info("NSE may be unreachable. Try refreshing, or check your network connection.")
    st.stop()

if history_df is None or history_df.empty:
    st.warning("No FII/DII historical data available. NSE may not have returned data — try again later.")
    st.stop()

# Prepare history
df = history_df.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date", ascending=False).reset_index(drop=True)

# Ensure numeric columns
for col in ["fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── Helper Functions ────────────────────────────────────────────────
def fmt_cr(v):
    """Format value in Crores with sign and color."""
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.0f} Cr"


def fmt_cr_html(v, bold=False):
    """Format with HTML color."""
    color = "#00D09E" if v > 0 else "#FB3640" if v < 0 else "#8b949e"
    weight = "800" if bold else "600"
    sign = "+" if v > 0 else ""
    return f'<span style="color:{color}; font-weight:{weight}; font-family:monospace;">{sign}{v:,.0f} Cr</span>'


def compute_streak(df):
    """Compute FII consecutive buy/sell streak."""
    if df.empty:
        return "neutral", 0, 0
    direction = "sell" if df.iloc[0]["fii_net"] < 0 else "buy"
    count = 0
    total = 0
    for _, row in df.iterrows():
        if direction == "sell" and row["fii_net"] < 0:
            count += 1
            total += row["fii_net"]
        elif direction == "buy" and row["fii_net"] > 0:
            count += 1
            total += row["fii_net"]
        else:
            break
    return direction, count, total


def heatmap_color(val, max_abs):
    """Return a hex color for the heatmap cell."""
    if max_abs == 0:
        return "#21262d"
    ratio = min(abs(val) / max_abs, 1.0)
    if val < 0:
        # Red gradient
        if ratio > 0.8:
            return "#FF3B30"
        elif ratio > 0.6:
            return "#D72828"
        elif ratio > 0.3:
            return "#A81B1B"
        else:
            return "#681A1A"
    else:
        # Green gradient
        if ratio > 0.8:
            return "#32D74B"
        elif ratio > 0.6:
            return "#28B03D"
        elif ratio > 0.3:
            return "#1B892B"
        else:
            return "#165620"


# ── Latest Session Data ─────────────────────────────────────────────
latest = df.iloc[0] if not df.empty else None

if latest is not None:
    fii_net = latest["fii_net"]
    dii_net = latest["dii_net"]
    combined = fii_net + dii_net
    latest_date = latest["date"].strftime("%A, %d %B %Y")

    # Flow strength meter
    fii_abs = abs(fii_net)
    dii_abs = abs(dii_net)
    total_flow = fii_abs + dii_abs
    fii_pct = round(fii_abs / total_flow * 100) if total_flow > 0 else 50
    dii_pct = 100 - fii_pct

    # Streak
    streak_dir, streak_count, streak_vol = compute_streak(df)

    # ── HERO SECTION ────────────────────────────────────────────────
    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Momentum badge
        if fii_net < -5000:
            badge = "Aggressive Selling"
            badge_cls = "streak-sell"
        elif fii_net < -2000:
            badge = "Moderate Selling"
            badge_cls = "streak-sell"
        elif fii_net > 5000:
            badge = "Aggressive Buying"
            badge_cls = "streak-buy"
        elif fii_net > 2000:
            badge = "Moderate Buying"
            badge_cls = "streak-buy"
        else:
            badge = "Neutral"
            badge_cls = "streak-sell" if fii_net < 0 else "streak-buy"

        fii_label = "FII Selling" if fii_net < 0 else "FII Buying"
        dii_label = "DII Support" if dii_net > 0 else "DII Selling"
        fii_color = "#FB3640" if fii_net < 0 else "#00D09E"
        dii_color = "#00D09E" if dii_net > 0 else "#FB3640"
        net_color = "#00D09E" if combined > 0 else "#FB3640"

        hero_html = (
            f'<div class="fii-hero">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">'
            f'<div><div style="color:#0AD1A6; font-size:16px; font-weight:700; text-transform:uppercase;">Latest Session</div>'
            f'<div style="color:#8b949e; font-size:13px; margin-top:4px;">{latest_date}</div></div>'
            f'<span class="streak-pill {badge_cls}">{badge}</span></div>'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:24px;">'
            f'<div><div style="color:#8b949e; font-size:12px; font-weight:700; letter-spacing:0.5px;">FII / FPI NET</div>'
            f'<div class="num-big" style="color:{fii_color};">{fmt_cr(fii_net)}</div></div>'
            f'<div style="text-align:right;"><div style="color:#8b949e; font-size:12px; font-weight:700; letter-spacing:0.5px;">DII NET</div>'
            f'<div class="num-med" style="color:{dii_color};">{fmt_cr(dii_net)}</div></div></div>'
            f'<div style="margin-top:20px; padding:14px; background:#0d1117; border:1px solid #21262d; border-radius:10px;">'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">'
            f'<span style="color:{fii_color};">{fii_label}: {fii_pct}%</span>'
            f'<span style="color:{dii_color};">{dii_label}: {dii_pct}%</span></div>'
            f'<div style="height:6px; width:100%; display:flex; border-radius:3px; overflow:hidden; background:#21262d;">'
            f'<div style="background:#FB3640; width:{fii_pct}%;"></div>'
            f'<div style="background:#00D09E; width:{dii_pct}%;"></div></div></div>'
            f'<div style="margin-top:16px; background:#0d1117; border-radius:10px; padding:14px 18px; display:flex; justify-content:space-between; align-items:center; border:1px solid #21262d;">'
            f'<span style="font-size:13px; font-weight:700; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Combined Liquidity</span>'
            f'<span class="num-med" style="color:{net_color};">{fmt_cr(combined)}</span></div></div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    with col_side:
        # Streak widget
        s_color = "#FB3640" if streak_dir == "sell" else "#00D09E"
        s_label = "Selling" if streak_dir == "sell" else "Buying"
        streak_html = (
            f'<div class="flow-card" style="margin-bottom:12px;">'
            f'<div style="color:#8b949e; font-size:12px; font-weight:600; margin-bottom:8px;">Current FII Streak</div>'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div class="num-med" style="color:{s_color};">{streak_count} Day{"s" if streak_count != 1 else ""} {s_label}</div>'
            f'<div style="text-align:right;"><div style="color:#8b949e; font-size:11px;">Velocity</div>'
            f'<div style="color:{s_color}; font-weight:700; font-family:monospace;">{fmt_cr(streak_vol)}</div></div></div></div>'
        )
        st.markdown(streak_html, unsafe_allow_html=True)

        # 5-day velocity
        last5 = df.head(5)
        fii_5d = last5["fii_net"].sum() if len(last5) >= 5 else last5["fii_net"].sum()
        avg_daily = fii_5d / min(len(last5), 5)
        vel_color = "#FB3640" if fii_5d < 0 else "#00D09E"
        vel_html = (
            f'<div class="flow-card" style="margin-bottom:12px;">'
            f'<div style="color:#8b949e; font-size:12px; font-weight:600; margin-bottom:8px;">5-Day FII Net Velocity</div>'
            f'<div class="num-med" style="color:{vel_color};">{fmt_cr(fii_5d)}</div>'
            f'<div style="color:#8b949e; font-size:12px; margin-top:6px;">Avg daily: <span style="color:{vel_color}; font-weight:700;">{fmt_cr(avg_daily)}/day</span></div></div>'
        )
        st.markdown(vel_html, unsafe_allow_html=True)

        # Cumulative flow cards
        if flow_data:
            yr1 = flow_data.get("1y", {})
            fii_1y = yr1.get("fii_net", 0) or 0
            dii_1y = yr1.get("dii_net", 0) or 0
            cum_html = (
                f'<div class="flow-card">'
                f'<div style="display:flex; justify-content:space-between;">'
                f'<div><div style="color:#8b949e; font-size:11px; font-weight:600;">FII 1Y Cumulative</div>'
                f'<div style="color:{"#FB3640" if fii_1y < 0 else "#00D09E"}; font-weight:800; font-family:monospace; font-size:16px;">{fmt_cr(fii_1y)}</div></div>'
                f'<div style="text-align:right;"><div style="color:#8b949e; font-size:11px; font-weight:600;">DII 1Y Cumulative</div>'
                f'<div style="color:{"#00D09E" if dii_1y > 0 else "#FB3640"}; font-weight:800; font-family:monospace; font-size:16px;">{fmt_cr(dii_1y)}</div></div></div></div>'
            )
            st.markdown(cum_html, unsafe_allow_html=True)

    # ── CUMULATIVE FLOW SUMMARY CARDS ───────────────────────────────
    if flow_data:
        st.markdown("---")
        st.subheader("Cumulative Flow Summary")
        timeframes = ["1w", "2w", "1m", "3m", "6m", "1y", "2y", "5y"]
        cols = st.columns(len(timeframes))
        for i, tf in enumerate(timeframes):
            fl = flow_data.get(tf, {})
            fn = fl.get("fii_net") or 0
            dn = fl.get("dii_net") or 0
            days = fl.get("days_available", 0)
            with cols[i]:
                st.markdown(
                    f'<div class="flow-card" style="text-align:center; padding:12px;">'
                    f'<div style="color:#0AD1A6; font-weight:800; font-size:14px; margin-bottom:8px;">{tf.upper()}</div>'
                    f'<div style="font-size:11px; color:#8b949e; margin-bottom:4px;">FII</div>'
                    f'{fmt_cr_html(fn, bold=True)}'
                    f'<div style="font-size:11px; color:#8b949e; margin:4px 0;">DII</div>'
                    f'{fmt_cr_html(dn, bold=True)}'
                    f'<div style="font-size:10px; color:#484f58; margin-top:6px;">{days} days</div></div>',
                    unsafe_allow_html=True,
                )

# ── TABS ────────────────────────────────────────────────────────────
st.markdown("---")
tab_daily, tab_heatmap, tab_charts = st.tabs(
    ["Daily Flow Matrix", "Visual Heatmaps", "Historical Charts"]
)

# ── TAB 1: DAILY DATA TABLE ────────────────────────────────────────
with tab_daily:
    # Sub-period selector
    period = st.radio(
        "Period",
        ["Last 15 Days", "Last 30 Days", "Last 90 Days", "All History"],
        horizontal=True,
    )
    period_map = {"Last 15 Days": 15, "Last 30 Days": 30, "Last 90 Days": 90, "All History": len(df)}
    n = period_map[period]
    view_df = df.head(n).copy()

    # Smart filters
    filter_mode = st.radio(
        "Filter",
        ["All Data", "FII Bloodbath (< -5k Cr)", "DII Absorption (> +5k Cr)", "Extreme Divergence"],
        horizontal=True,
    )
    if filter_mode == "FII Bloodbath (< -5k Cr)":
        view_df = view_df[view_df["fii_net"] < -5000]
    elif filter_mode == "DII Absorption (> +5k Cr)":
        view_df = view_df[view_df["dii_net"] > 5000]
    elif filter_mode == "Extreme Divergence":
        view_df = view_df[(view_df["fii_net"] < -8000) & (view_df["dii_net"] > 8000)]

    if view_df.empty:
        st.info("No sessions match this filter.")
    else:
        # Build display table
        display = pd.DataFrame()
        display["Date"] = view_df["date"].dt.strftime("%d %b %Y")

        has_buy_sell = "fii_buy" in view_df.columns and view_df["fii_buy"].abs().sum() > 0
        if has_buy_sell:
            display["FII Buy"] = view_df["fii_buy"].apply(lambda x: f"{x:,.0f}")
            display["FII Sell"] = view_df["fii_sell"].apply(lambda x: f"{x:,.0f}")
        display["FII Net"] = view_df["fii_net"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
        if has_buy_sell:
            display["DII Buy"] = view_df["dii_buy"].apply(lambda x: f"{x:,.0f}")
            display["DII Sell"] = view_df["dii_sell"].apply(lambda x: f"{x:,.0f}")
        display["DII Net"] = view_df["dii_net"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
        display["Combined"] = (view_df["fii_net"] + view_df["dii_net"]).apply(
            lambda x: f"{'+' if x > 0 else ''}{x:,.0f}"
        )

        # Color the FII/DII net columns
        def color_net(val):
            try:
                v = float(val.replace(",", "").replace("+", ""))
                if v > 0:
                    return "color: #00D09E"
                elif v < 0:
                    return "color: #FB3640"
            except Exception:
                pass
            return ""

        net_cols = ["FII Net", "DII Net", "Combined"]
        styled = display.style.applymap(color_net, subset=[c for c in net_cols if c in display.columns])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(len(display) * 35 + 38, 600))

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg FII Net/Day", f"{view_df['fii_net'].mean():,.0f} Cr")
        c2.metric("Avg DII Net/Day", f"{view_df['dii_net'].mean():,.0f} Cr")
        c3.metric("FII Sell Days", f"{(view_df['fii_net'] < 0).sum()}/{len(view_df)}")
        c4.metric("DII Buy Days", f"{(view_df['dii_net'] > 0).sum()}/{len(view_df)}")


# ── TAB 2: HEATMAPS ────────────────────────────────────────────────
with tab_heatmap:
    heat_days = st.slider("Days to show", 15, 90, 45)
    heat_df = df.head(heat_days)

    if not heat_df.empty:
        max_abs_fii = max(heat_df["fii_net"].abs().max(), 1)
        max_abs_dii = max(heat_df["dii_net"].abs().max(), 1)

        col_fii, col_dii = st.columns(2)

        with col_fii:
            st.markdown("**FII Flow Heatmap**")
            st.caption("Red = selling, Green = buying. Intensity = magnitude")
            cells_html = ""
            for _, row in heat_df.iterrows():
                color = heatmap_color(row["fii_net"], max_abs_fii)
                tip = f'{row["date"].strftime("%d %b")}: {fmt_cr(row["fii_net"])}'
                cells_html += f'<div class="heatcell" style="background:{color};" title="{tip}"></div>'
            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; gap:2px;">{cells_html}</div>',
                unsafe_allow_html=True,
            )
            # Legend
            st.markdown(
                '<div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:#8b949e;">'
                '<span>Heavy Selling</span><span>Heavy Buying</span></div>',
                unsafe_allow_html=True,
            )

        with col_dii:
            st.markdown("**DII Flow Heatmap**")
            st.caption("Red = selling, Green = buying. Intensity = magnitude")
            cells_html = ""
            for _, row in heat_df.iterrows():
                color = heatmap_color(row["dii_net"], max_abs_dii)
                tip = f'{row["date"].strftime("%d %b")}: {fmt_cr(row["dii_net"])}'
                cells_html += f'<div class="heatcell" style="background:{color};" title="{tip}"></div>'
            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; gap:2px;">{cells_html}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:#8b949e;">'
                '<span>Light Activity</span><span>Extreme Accumulation</span></div>',
                unsafe_allow_html=True,
            )

        # Divergence heatmap
        st.markdown("---")
        st.markdown("**FII vs DII Divergence**")
        st.caption("Shows days where FII and DII move in opposite directions")
        div_html = ""
        for _, row in heat_df.iterrows():
            fn, dn = row["fii_net"], row["dii_net"]
            # Divergence = FII selling + DII buying (or vice versa)
            if fn < 0 and dn > 0:
                intensity = min((abs(fn) + dn) / (max_abs_fii + max_abs_dii) * 2, 1.0)
                r = int(255 * (1 - intensity * 0.3))
                g = int(180 + 75 * intensity)
                color = f"#{r:02x}{g:02x}40"
            elif fn > 0 and dn < 0:
                intensity = min((fn + abs(dn)) / (max_abs_fii + max_abs_dii) * 2, 1.0)
                color = f"#FB{int(54 + 100 * (1 - intensity)):02x}{int(64 + 100 * (1 - intensity)):02x}"
            else:
                color = "#21262d"
            tip = f'{row["date"].strftime("%d %b")}: FII {fmt_cr(fn)}, DII {fmt_cr(dn)}'
            div_html += f'<div class="heatcell" style="background:{color};" title="{tip}"></div>'
        st.markdown(
            f'<div style="display:flex; flex-wrap:wrap; gap:2px;">{div_html}</div>',
            unsafe_allow_html=True,
        )


# ── TAB 3: CHARTS ──────────────────────────────────────────────────
with tab_charts:
    chart_period = st.radio(
        "Chart period", ["3 Months", "6 Months", "1 Year", "All"], horizontal=True, key="chart_period"
    )
    chart_map = {"3 Months": 63, "6 Months": 126, "1 Year": 252, "All": len(df)}
    chart_n = chart_map[chart_period]
    chart_df = df.head(chart_n).sort_values("date")

    # Monthly aggregation
    monthly = chart_df.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly_agg = monthly.groupby("month").agg(
        fii_net=("fii_net", "sum"), dii_net=("dii_net", "sum")
    ).reset_index()
    monthly_agg["month_str"] = monthly_agg["month"].astype(str)

    # Bar chart: Monthly FII vs DII
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_agg["month_str"],
        y=monthly_agg["fii_net"],
        name="FII Net",
        marker_color=["#FB3640" if v < 0 else "#00D09E" for v in monthly_agg["fii_net"]],
    ))
    fig_monthly.add_trace(go.Bar(
        x=monthly_agg["month_str"],
        y=monthly_agg["dii_net"],
        name="DII Net",
        marker_color=["#00D09E" if v > 0 else "#FB3640" for v in monthly_agg["dii_net"]],
    ))
    fig_monthly.update_layout(
        title="Monthly Net Flows (Cr)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Cumulative divergence chart
    chart_df_sorted = chart_df.sort_values("date")
    cum_fii = chart_df_sorted["fii_net"].cumsum()
    cum_dii = chart_df_sorted["dii_net"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=chart_df_sorted["date"],
        y=cum_fii,
        name="FII Cumulative",
        line=dict(color="#FB3640", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(251,54,64,0.1)",
    ))
    fig_cum.add_trace(go.Scatter(
        x=chart_df_sorted["date"],
        y=cum_dii,
        name="DII Cumulative",
        line=dict(color="#00D09E", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,208,158,0.1)",
    ))
    fig_cum.update_layout(
        title="Cumulative Flow Divergence (Cr)",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Daily bar chart
    with st.expander("Daily Net Flows", expanded=False):
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=chart_df_sorted["date"],
            y=chart_df_sorted["fii_net"],
            name="FII Net",
            marker_color=["#FB3640" if v < 0 else "#00D09E" for v in chart_df_sorted["fii_net"]],
            opacity=0.7,
        ))
        fig_daily.add_trace(go.Bar(
            x=chart_df_sorted["date"],
            y=chart_df_sorted["dii_net"],
            name="DII Net",
            marker_color=["#00D09E" if v > 0 else "#FB3640" for v in chart_df_sorted["dii_net"]],
            opacity=0.7,
        ))
        fig_daily.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=350,
            margin=dict(l=60, r=20, t=20, b=40),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

# ── WEEKLY & MONTHLY AGGREGATION TABLES ─────────────────────────────
st.markdown("---")
col_w, col_m = st.columns(2)

with col_w:
    st.subheader("Weekly Flows")
    weekly = df.copy()
    weekly["week"] = weekly["date"].dt.to_period("W")
    weekly_agg = weekly.groupby("week").agg(
        fii_net=("fii_net", "sum"), dii_net=("dii_net", "sum")
    ).reset_index().sort_values("week", ascending=False).head(12)
    weekly_agg["combined"] = weekly_agg["fii_net"] + weekly_agg["dii_net"]
    weekly_agg["Week"] = weekly_agg["week"].astype(str)
    weekly_agg["FII Net"] = weekly_agg["fii_net"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
    weekly_agg["DII Net"] = weekly_agg["dii_net"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
    weekly_agg["Combined"] = weekly_agg["combined"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
    weekly_agg["Signal"] = weekly_agg["combined"].apply(
        lambda x: "Accumulation" if x > 0 else "Distribution"
    )
    display_w = weekly_agg[["Week", "FII Net", "DII Net", "Combined", "Signal"]]
    st.dataframe(
        display_w.style.applymap(color_net, subset=["FII Net", "DII Net", "Combined"]),
        use_container_width=True,
        hide_index=True,
    )

with col_m:
    st.subheader("Monthly Flows")
    monthly_view = df.copy()
    monthly_view["month"] = monthly_view["date"].dt.to_period("M")
    monthly_agg_v = monthly_view.groupby("month").agg(
        fii_net=("fii_net", "sum"), dii_net=("dii_net", "sum")
    ).reset_index().sort_values("month", ascending=False).head(24)
    monthly_agg_v["combined"] = monthly_agg_v["fii_net"] + monthly_agg_v["dii_net"]
    monthly_agg_v["Month"] = monthly_agg_v["month"].astype(str)
    monthly_agg_v["FII Net"] = monthly_agg_v["fii_net"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
    monthly_agg_v["DII Net"] = monthly_agg_v["dii_net"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
    monthly_agg_v["Combined"] = monthly_agg_v["combined"].apply(lambda x: f"{'+' if x > 0 else ''}{x:,.0f}")
    # DII multiplier (how much DII absorbs FII selling)
    monthly_agg_v["DII/FII"] = monthly_agg_v.apply(
        lambda r: f"{r['dii_net']/abs(r['fii_net']):.1f}x" if r["fii_net"] < 0 and r["dii_net"] > 0 else "N/A",
        axis=1,
    )
    display_m = monthly_agg_v[["Month", "FII Net", "DII Net", "Combined", "DII/FII"]]
    st.dataframe(
        display_m.style.applymap(color_net, subset=["FII Net", "DII Net", "Combined"]),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ──────────────────────────────────────────────────────────
if history_df is not None and not history_df.empty:
    earliest = pd.to_datetime(df["date"]).min().strftime("%d %b %Y")
    latest_d = pd.to_datetime(df["date"]).max().strftime("%d %b %Y")
    st.caption(
        f"Data: {earliest} to {latest_d} ({len(df)} trading days). "
        f"Source: NSE India. Inspired by [MrChartist/fii-dii-data](https://github.com/MrChartist/fii-dii-data)."
    )
