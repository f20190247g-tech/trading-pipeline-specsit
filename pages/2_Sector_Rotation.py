"""Page 2: Sector Rotation & Relative Strength"""
import streamlit as st
import pandas as pd

from dashboard_helpers import (
    build_momentum_heatmap,
    compute_all_sector_rs_timeseries,
    build_lw_line_chart_html,
    compute_derivatives,
    detect_inflection_points,
)

st.set_page_config(page_title="Sector Rotation", page_icon="📊", layout="wide")
st.title("Sector Rotation")

if "sector_rankings" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

sector_rankings = st.session_state.sector_rankings
top_sectors = st.session_state.top_sectors
sector_data = st.session_state.sector_data
nifty_df = st.session_state.nifty_df

# ── Compute RS timeseries upfront (used by chart + derivatives) ──
with st.spinner("Computing sector RS timeseries..."):
    rs_df = compute_all_sector_rs_timeseries(sector_data, nifty_df)

# ── Compute derivatives for ALL sectors ──────────────────────────
all_sector_derivs = {}
if not rs_df.empty:
    for sector in rs_df.columns:
        rs_series = rs_df[sector].dropna()
        if len(rs_series) < 60:
            continue
        d = compute_derivatives(rs_series)
        rs_level = rs_series.iloc[-1]
        inflection = detect_inflection_points(d["roc"], d["accel"], level=rs_level)
        latest_roc = d["roc"].iloc[-1] if not d["roc"].empty else 0
        latest_accel = d["accel"].iloc[-1] if not d["accel"].empty else 0
        all_sector_derivs[sector] = {
            "inflection": inflection,
            "roc": latest_roc,
            "accel": latest_accel,
        }

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Combined Sector Rankings (table + heatmap merged)
# ══════════════════════════════════════════════════════════════════
st.subheader("Sector Rankings")

with st.expander("How Sector Ranking Works", expanded=True):
    st.markdown("""
**Mansfield Relative Strength** measures each sector index's performance vs Nifty 50 over ~52 weeks. Positive RS = outperforming, negative = underperforming.

**Columns explained:**
- **RS:** Current Mansfield RS value — higher = stronger vs market.
- **Trend:** Whether the RS line is rising, flat, or falling over last 21 days.
- **1w to 6m:** Rolling rate-of-change of sector/Nifty ratio from today. All periods are rolling trading days from today (1w=5d, 2w=10d, 1m=21d, 3m=63d, 6m=126d).
- **Signal:** RS derivative inflection — context-aware labels based on ROC, acceleration, and RS level:
  - **Bullish Thrust** — rising and accelerating (strong trend)
  - **Pullback Slowing** — strong sector (RS > 0) dipping but decline losing steam (watch for bounce)
  - **Bullish Inflection** — weak sector (RS < 0) where deterioration is slowing (early reversal)
  - **Bearish Inflection** — strong sector still rising but momentum fading (caution)
  - **Recovery Fading** — weak sector was recovering but recovery losing steam
  - **Rolling Over** — strong sector declining and decline accelerating (danger)
  - **Bearish Breakdown** — weak sector declining and accelerating down
- **Score:** Composite of RS level (40%) + RS trend (20%) + momentum blend (40%).

**Why sectors matter:** Institutional money rotates between sectors. Stocks in top-ranked sectors have tailwinds — the pipeline only hunts in the top sectors.
""")

# Build combined table: rank + RS + trend + momentum periods + derivative signal
rows = []
for i, s in enumerate(sector_rankings):
    mom = s["momentum"]
    sector_name = s["sector"]
    deriv = all_sector_derivs.get(sector_name, {})
    inflection = deriv.get("inflection", {})
    sector_stage_data = s.get("sector_stage", {})
    if isinstance(sector_stage_data, dict):
        stage_num = sector_stage_data.get("stage", "")
        stage_sub = sector_stage_data.get("substage", "")
        stage_label = stage_sub if stage_sub else (f"S{stage_num}" if stage_num else "—")
    else:
        stage_label = f"S{sector_stage_data}" if sector_stage_data else "—"
    rows.append({
        "Rank": i + 1,
        "Sector": sector_name,
        "Stage": stage_label,
        "RS": s["mansfield_rs"],
        "Trend": s["rs_trend"],
        "1w": mom.get("1w", None),
        "2w": mom.get("2w", None),
        "1m": mom.get("1m", None),
        "3m": mom.get("3m", None),
        "6m": mom.get("6m", None),
        "Signal": inflection.get("label", "—"),
        "Score": s["composite_score"],
        "Top": ">>>" if sector_name in top_sectors else "",
    })

df_table = pd.DataFrame(rows)


def _style_sector_row(row):
    """Highlight top sectors and color momentum cells."""
    styles = ["" for _ in row]
    if row["Top"] == ">>>":
        styles = ["background-color: #26a69a18" for _ in row]

    # Color momentum columns
    for col_idx, col_name in enumerate(row.index):
        if col_name in ("1w", "2w", "1m", "3m", "6m"):
            val = row[col_name]
            if pd.notna(val) and isinstance(val, (int, float)):
                if val > 1:
                    styles[col_idx] = "background-color: #26a69a33; color: #26a69a"
                elif val < -1:
                    styles[col_idx] = "background-color: #ef535033; color: #ef5350"

    # Color signal column
    signal_idx = list(row.index).index("Signal") if "Signal" in row.index else -1
    if signal_idx >= 0:
        sig = row["Signal"]
        if "Bullish Thrust" in str(sig):
            styles[signal_idx] = "color: #4CAF50; font-weight: 600"
        elif "Bullish Inflection" in str(sig):
            styles[signal_idx] = "color: #FFD700; font-weight: 600"
        elif "Bearish Inflection" in str(sig):
            styles[signal_idx] = "color: #FF9800; font-weight: 600"
        elif "Bearish Breakdown" in str(sig):
            styles[signal_idx] = "color: #F44336; font-weight: 600"

    return styles


st.dataframe(
    df_table.style.apply(_style_sector_row, axis=1),
    use_container_width=True,
    hide_index=True,
    height=min(600, len(rows) * 38 + 40),
)

# ══════════════════════════════════════════════════════════════════
# SECTION 2: Emerging & Fading Sectors (derivative-based alerts)
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Emerging & Fading Sectors")
st.caption("RS derivative analysis on ALL sectors — spots rotation before it shows in rankings.")

# Separate into emerging (bullish inflection in non-top sectors) and fading
emerging = []
fading = []
for sector_name, deriv in all_sector_derivs.items():
    inflection = deriv.get("inflection", {})
    signal = inflection.get("signal", "")
    roc = deriv.get("roc", 0)
    accel = deriv.get("accel", 0)
    is_top = sector_name in top_sectors

    # Emerging: non-top sectors with improving momentum
    if signal in ("bullish_inflection", "bullish_thrust", "pullback_slowing") and not is_top:
        emerging.append({
            "Sector": sector_name,
            "Signal": inflection.get("label", ""),
            "Detail": inflection.get("detail", ""),
            "ROC": f"{roc:.2f}" if pd.notna(roc) else "—",
            "Accel": f"{accel:.2f}" if pd.notna(accel) else "—",
        })
    # Fading: top sectors losing momentum
    elif signal in ("bearish_inflection", "rolling_over", "recovery_fading", "bearish_breakdown") and is_top:
        fading.append({
            "Sector": sector_name,
            "Signal": inflection.get("label", ""),
            "Detail": inflection.get("detail", ""),
            "ROC": f"{roc:.2f}" if pd.notna(roc) else "—",
            "Accel": f"{accel:.2f}" if pd.notna(accel) else "—",
        })

em_col, fa_col = st.columns(2)

with em_col:
    st.markdown("**Emerging** — non-top sectors with bullish RS inflection")
    if emerging:
        for e in emerging:
            sig_color = "#4CAF50" if "Thrust" in e["Signal"] else "#FFD700"
            st.markdown(
                f"""<div style="background:{sig_color}12; border-left:3px solid {sig_color};
                    padding:8px 12px; border-radius:0 6px 6px 0; margin-bottom:6px;">
                    <span style="font-weight:700;">{e['Sector']}</span>
                    <span style="color:{sig_color}; margin-left:8px; font-size:0.85em;">{e['Signal']}</span>
                    <div style="font-size:0.8em; color:#999;">{e['Detail']}</div>
                    <div style="font-size:0.75em; color:#666;">ROC: {e['ROC']} | Accel: {e['Accel']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No emerging sectors detected right now.")

with fa_col:
    st.markdown("**Fading** — top sectors losing momentum")
    if fading:
        for f in fading:
            sig_color = "#F44336" if "Breakdown" in f["Signal"] else "#FF9800"
            st.markdown(
                f"""<div style="background:{sig_color}12; border-left:3px solid {sig_color};
                    padding:8px 12px; border-radius:0 6px 6px 0; margin-bottom:6px;">
                    <span style="font-weight:700;">{f['Sector']}</span>
                    <span style="color:{sig_color}; margin-left:8px; font-size:0.85em;">{f['Signal']}</span>
                    <div style="font-size:0.8em; color:#999;">{f['Detail']}</div>
                    <div style="font-size:0.75em; color:#666;">ROC: {f['ROC']} | Accel: {f['Accel']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("All top sectors holding strong.")

# ══════════════════════════════════════════════════════════════════
# SECTION 3: RS Line Chart
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Mansfield RS Over Time")
if not rs_df.empty:
    plot_rs_df = rs_df.iloc[-180:] if len(rs_df) > 180 else rs_df
    lw_series = []
    bright_colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#FFEB3B", "#FF5722"]
    color_idx = 0
    # Also highlight emerging sectors
    emerging_names = {e["Sector"] for e in emerging}
    for col in plot_rs_df.columns:
        is_top = col in top_sectors
        is_emerging = col in emerging_names
        if is_top:
            color = bright_colors[color_idx % len(bright_colors)]
            color_idx += 1
            width = 3
        elif is_emerging:
            color = "#FFD700"
            width = 2
        else:
            color = "#555"
            width = 1
        lw_series.append({
            "name": col,
            "times": plot_rs_df.index.strftime("%Y-%m-%d").tolist(),
            "values": plot_rs_df[col].tolist(),
            "color": color,
            "lineWidth": width,
        })
    rs_html = build_lw_line_chart_html(lw_series, title="Sector Mansfield RS (vs Nifty 50)", height=500, zero_line=True)
    st.components.v1.html(rs_html, height=510)
else:
    st.caption("Insufficient sector data for RS chart.")

# ══════════════════════════════════════════════════════════════════
# SECTION 4: Full Derivative Table (all sectors)
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("All Sectors — Derivative Signals")
st.caption("RS rate-of-change and acceleration for every sector. Sorted by acceleration (most improving first).")

if all_sector_derivs:
    deriv_rows = []
    for sector_name, deriv in all_sector_derivs.items():
        inflection = deriv.get("inflection", {})
        roc = deriv.get("roc", 0)
        accel = deriv.get("accel", 0)
        deriv_rows.append({
            "Sector": sector_name,
            "Status": "TOP" if sector_name in top_sectors else "",
            "Signal": inflection.get("label", "—"),
            "ROC": round(roc, 2) if pd.notna(roc) else None,
            "Acceleration": round(accel, 2) if pd.notna(accel) else None,
            "Detail": inflection.get("detail", ""),
        })
    # Sort by acceleration descending (most improving first)
    deriv_rows.sort(key=lambda x: x.get("Acceleration") or -999, reverse=True)
    st.dataframe(pd.DataFrame(deriv_rows), use_container_width=True, hide_index=True)
else:
    st.caption("Insufficient data for derivative analysis.")
