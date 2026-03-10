"""Page 3: Stock Opportunities — Merged Stock Scanner + Stage Analysis"""
import streamlit as st
import pandas as pd
import plotly.express as px

from stock_screener import compute_stock_rs
from dashboard_helpers import resample_ohlcv, build_lw_candlestick_html
from stage_filter import detect_bases

st.set_page_config(page_title="Stock Opportunities", page_icon="📊", layout="wide")
st.title("Stock Opportunities")

if "screened_stocks" not in st.session_state and "all_stage2_stocks" not in st.session_state:
    st.info("Run a scan first from the home page.")
    st.stop()

screened = st.session_state.get("screened_stocks", [])
top_sectors = st.session_state.get("top_sectors", [])
nifty_df = st.session_state.get("nifty_df")
stock_data = st.session_state.get("stock_data", {})
all_stage2 = st.session_state.get("all_stage2_stocks", [])
pipeline_candidates = st.session_state.get("stage2_candidates", [])

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Sector Pipeline", "Full Universe Stage 2", "How It Works"])


# ══════════════════════════════════════════════════════════════════
# TAB 1: Sector Pipeline (former Stock Scanner)
# ══════════════════════════════════════════════════════════════════
with tab1:
    if not screened:
        st.warning("No stocks passed the screening criteria.")
    else:
        # Backfill rs_1m / rs_3m if missing (old cached scan)
        if screened[0].get("rs_1m") is None and nifty_df is not None:
            nifty_close = nifty_df["Close"]
            for s in screened:
                ticker = s["ticker"]
                df_stock = stock_data.get(ticker)
                if df_stock is not None and not df_stock.empty:
                    close = df_stock["Close"]
                    s["rs_1m"] = compute_stock_rs(close, nifty_close, period=21)
                    s["rs_3m"] = compute_stock_rs(close, nifty_close, period=63)
                else:
                    s["rs_1m"] = 0
                    s["rs_3m"] = 0

        # Cross-reference with stage2 candidates and watchlist for action signals
        stage2_map = {}
        for c in st.session_state.get("stage2_candidates", []):
            stage2_map[c["ticker"]] = c
        watchlist_map = {}
        for w in st.session_state.get("final_watchlist", []):
            watchlist_map[w.get("ticker", "")] = w

        def _derive_action_signal(s, stage2_info, watchlist_info):
            rs_1m = s.get("rs_1m", 0)
            rs_3m = s.get("rs_3m", 0)
            dist = s.get("dist_from_high_pct", 0)
            accum = s.get("accumulation_ratio", 1.0)
            if watchlist_info and watchlist_info.get("action") == "BUY":
                es = watchlist_info.get("entry_setup", {})
                if es:
                    return "BUY — Setup Ready", "#4CAF50"
            if stage2_info:
                breakout = stage2_info.get("breakout", {})
                if breakout and breakout.get("breakout"):
                    return "BUY — Breakout", "#4CAF50"
                stage = stage2_info.get("stage", {})
                if stage.get("stage") == 2:
                    vcp = stage2_info.get("vcp", {})
                    if vcp and vcp.get("is_vcp"):
                        return "WATCH — VCP Forming", "#FFD700"
                    return "WATCH — In Base", "#2196F3"
            if rs_1m > rs_3m and rs_1m > 5 and accum > 1.3:
                return "WATCH — Accumulating", "#2196F3"
            if rs_1m < rs_3m * 0.5 and rs_3m > 10:
                return "WAIT — Momentum Fading", "#FF9800"
            if dist > 15:
                return "WAIT — Far From High", "#FF9800"
            if rs_1m > 0 and accum > 1.1:
                return "WATCH — Improving", "#2196F3"
            return "MONITOR", "#888"

        def _rs_trend_label(rs_1m, rs_3m, rs_6m):
            if rs_1m > rs_3m > 0:
                return "Accelerating"
            if rs_1m > 0 and rs_3m > 0 and rs_1m < rs_3m:
                return "Strong, Slowing"
            if rs_1m > 0 and rs_3m <= 0:
                return "Turning Up"
            if rs_1m <= 0 and rs_3m > 0:
                return "Turning Down"
            if rs_1m <= 0 and rs_3m <= 0:
                return "Weak"
            return "Mixed"

        # Filters
        _sf1, _sf2 = st.columns([3, 1])
        with _sf1:
            sectors_in_results = sorted(set(s["sector"] for s in screened))
            selected_sectors = st.multiselect(
                "Filter by sector", sectors_in_results, default=sectors_in_results, key="scanner_sectors"
            )
        with _sf2:
            ticker_search = st.text_input("Search ticker", key="scanner_ticker_search").strip().upper()
        filtered = [
            s for s in screened
            if s["sector"] in selected_sectors
            and (not ticker_search or ticker_search in s["ticker"].upper())
        ]
        st.metric("Stocks Shown", len(filtered))

        # Results Table
        rows = []
        for s in filtered:
            rs_1m = s.get("rs_1m", 0)
            rs_3m = s.get("rs_3m", 0)
            rs_6m = s.get("rs_vs_nifty", 0)
            stage2_info = stage2_map.get(s["ticker"])
            wl_info = watchlist_map.get(s["ticker"])
            signal, signal_color = _derive_action_signal(s, stage2_info, wl_info)
            trend = _rs_trend_label(rs_1m, rs_3m, rs_6m)
            rows.append({
                "Ticker": s["ticker"],
                "Sector": s["sector"],
                "RS 1m": float(f"{rs_1m:.1f}"),
                "RS 3m": float(f"{rs_3m:.1f}"),
                "RS 6m": float(f"{rs_6m:.1f}"),
                "RS Trend": trend,
                "Dist %": float(f"{s.get('dist_from_high_pct', 0):.1f}"),
                "Accum": float(f"{s.get('accumulation_ratio', 0):.2f}"),
                "Signal": signal,
                "Close": float(f"{s.get('close', 0):.2f}"),
            })

        df = pd.DataFrame(rows)

        def _style_scanner_row(row):
            styles = ["" for _ in row]
            for col_idx, col_name in enumerate(row.index):
                if col_name == "Signal":
                    sig = str(row[col_name])
                    if sig.startswith("BUY"):
                        styles[col_idx] = "color: #4CAF50; font-weight: 700"
                    elif sig.startswith("WATCH"):
                        styles[col_idx] = "color: #2196F3; font-weight: 600"
                    elif sig.startswith("WAIT"):
                        styles[col_idx] = "color: #FF9800"
                    else:
                        styles[col_idx] = "color: #888"
                elif col_name == "RS Trend":
                    trend = str(row[col_name])
                    if trend == "Accelerating":
                        styles[col_idx] = "color: #4CAF50; font-weight: 600"
                    elif trend.startswith("Strong"):
                        styles[col_idx] = "color: #8BC34A"
                    elif trend == "Turning Up":
                        styles[col_idx] = "color: #FFD700"
                    elif trend in ("Turning Down", "Weak"):
                        styles[col_idx] = "color: #FF9800"
                elif col_name in ("RS 1m", "RS 3m", "RS 6m"):
                    val = row[col_name]
                    if isinstance(val, (int, float)):
                        if val > 10:
                            styles[col_idx] = "color: #26a69a"
                        elif val < 0:
                            styles[col_idx] = "color: #ef5350"
            return styles

        if not df.empty:
            st.dataframe(
                df.style.apply(_style_scanner_row, axis=1),
                use_container_width=True,
                hide_index=True,
                height=min(700, len(rows) * 38 + 40),
            )

        # Action Summary
        buy_count = sum(1 for r in rows if r["Signal"].startswith("BUY"))
        watch_count = sum(1 for r in rows if r["Signal"].startswith("WATCH"))
        wait_count = sum(1 for r in rows if r["Signal"].startswith("WAIT"))
        s1, s2, s3 = st.columns(3)
        s1.metric("BUY Signals", buy_count)
        s2.metric("WATCH (Waiting for Trigger)", watch_count)
        s3.metric("WAIT (Not Ready)", wait_count)

        # Scatter Plot
        st.subheader("RS Momentum vs Proximity to High")
        st.caption("Best candidates: top-right (accelerating RS + near highs). Bubble size = volume, color = accumulation.")
        if len(rows) > 0:
            fig = px.scatter(
                df, x="Dist %", y="RS 1m",
                size="Accum", color="RS Trend",
                hover_name="Ticker",
                color_discrete_map={
                    "Accelerating": "#4CAF50",
                    "Strong, Slowing": "#8BC34A",
                    "Turning Up": "#FFD700",
                    "Turning Down": "#FF9800",
                    "Weak": "#ef5350",
                    "Mixed": "#888",
                },
                size_max=25,
                template="plotly_dark",
            )
            fig.update_layout(
                height=500,
                xaxis_title="Distance from 52-week High (%)",
                yaxis_title="1-Month RS vs Nifty (%) — Recent Momentum",
                margin=dict(l=50, r=20, t=30, b=30),
            )
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: Full Universe Stage 2 (former Stage Analysis)
# ══════════════════════════════════════════════════════════════════
with tab2:
    if not all_stage2 and not pipeline_candidates:
        st.info("Run a scan first from the home page.")
    else:
        # Build set of pipeline candidate tickers for highlighting
        pipeline_tickers = {c["ticker"] for c in pipeline_candidates}

        # Summary Metrics
        perfect_7 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 7]
        score_6 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 6]
        score_5 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 5]
        score_4 = [s for s in all_stage2 if s.get("stage", {}).get("s2_score", 0) == 4]
        breakout_count = sum(1 for s in all_stage2 if s.get("breakout") and s["breakout"].get("breakout"))
        vcp_count = sum(1 for s in all_stage2 if s.get("vcp") and s["vcp"].get("is_vcp"))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("7/7 Perfect", len(perfect_7))
        c2.metric("6/7 Strong", len(score_6))
        c3.metric("5/7 Solid", len(score_5))
        c4.metric("4/7 Emerging", len(score_4))
        c5.metric("Active Breakouts", breakout_count)

        # Filters
        col_f1, col_f2, col_f3, col_f4 = st.columns([3, 1, 1, 1])
        with col_f1:
            sectors_available = sorted(set(s.get("sector", "Unknown") for s in all_stage2))
            selected_sectors_s2 = st.multiselect(
                "Filter by sector", sectors_available, default=sectors_available, key="stage2_sectors"
            )
        with col_f2:
            min_score = st.selectbox("Min S2 Score", [4, 5, 6, 7], index=0, key="stage2_min_score")
        with col_f3:
            show_breakouts_only = st.checkbox("Breakouts only", value=False, key="stage2_bo_only")
        with col_f4:
            s2_ticker_search = st.text_input("Search ticker", key="stage2_ticker_search").strip().upper()

        # Apply filters
        filtered_s2 = [
            s for s in all_stage2
            if s.get("sector", "Unknown") in selected_sectors_s2
            and s.get("stage", {}).get("s2_score", 0) >= min_score
            and (not show_breakouts_only or (s.get("breakout") and s["breakout"].get("breakout")))
            and (not s2_ticker_search or s2_ticker_search in s["ticker"].upper())
        ]

        st.metric("Stocks Shown", len(filtered_s2))

        # Results Table
        st.subheader("Stage 2 Candidates — Full Universe")
        s2_rows = []
        for s in filtered_s2:
            stage = s.get("stage", {})
            breakout = s.get("breakout", {}) or {}
            entry_setup = s.get("entry_setup", {}) or {}
            vcp = s.get("vcp", {}) or {}
            in_pipeline = s["ticker"] in pipeline_tickers
            in_top_sector = s.get("sector", "") in top_sectors

            consol = s.get("consolidation", {}) or {}
            weekly = s.get("weekly", {}) or {}
            s2_rows.append({
                "Ticker": s["ticker"],
                "Sector": s.get("sector", ""),
                "S2 Score": f"{stage.get('s2_score', 0)}/7",
                "Stage": stage.get("stage", "?"),
                "Base #": s.get("base_count_in_stage2", 0),
                "Grade": consol.get("quality_grade", "—"),
                "Weekly": "✓" if weekly.get("weekly_confirmed") else "✗",
                "Breakout": "YES" if breakout.get("breakout") else "",
                "VCP": "YES" if vcp.get("is_vcp") else "",
                "Entry": round(entry_setup.get("entry_price", 0), 1) if entry_setup.get("entry_price") else "",
                "Stop": round(entry_setup.get("effective_stop", 0), 1) if entry_setup.get("effective_stop") else "",
                "Risk %": f"{entry_setup.get('risk_pct', 0):.1f}%" if entry_setup.get("risk_pct") else "",
                "Close": s.get("close", 0),
                "Pipeline": "YES" if in_pipeline else "",
                "Top Sector": "YES" if in_top_sector else "",
            })

        df_s2 = pd.DataFrame(s2_rows)

        def _style_stage_row(row):
            styles = ["" for _ in row]
            for col_idx, col_name in enumerate(row.index):
                if col_name == "S2 Score":
                    score_str = str(row[col_name])
                    if score_str.startswith("7"):
                        styles[col_idx] = "color: #4CAF50; font-weight: 700"
                    elif score_str.startswith("6"):
                        styles[col_idx] = "color: #8BC34A; font-weight: 600"
                    elif score_str.startswith("5"):
                        styles[col_idx] = "color: #FFD700"
                    elif score_str.startswith("4"):
                        styles[col_idx] = "color: #FF9800"
                elif col_name == "Breakout" and row[col_name] == "YES":
                    styles[col_idx] = "color: #4CAF50; font-weight: 700"
                elif col_name == "VCP" and row[col_name] == "YES":
                    styles[col_idx] = "color: #FFD700; font-weight: 600"
                elif col_name == "Pipeline" and row[col_name] == "YES":
                    styles[col_idx] = "color: #2196F3; font-weight: 700"
                elif col_name == "Top Sector" and row[col_name] == "YES":
                    styles[col_idx] = "color: #26a69a"
            return styles

        if not df_s2.empty:
            st.dataframe(
                df_s2.style.apply(_style_stage_row, axis=1),
                use_container_width=True,
                hide_index=True,
                height=min(700, len(s2_rows) * 38 + 40),
            )
        else:
            st.warning("No stocks match current filters.")

        # Sector Distribution
        st.subheader("Stage 2 by Sector")
        if all_stage2:
            sector_counts = {}
            for s in all_stage2:
                sec = s.get("sector", "Unknown")
                score = s.get("stage", {}).get("s2_score", 0)
                if sec not in sector_counts:
                    sector_counts[sec] = {"7/7": 0, "6/7": 0, "5/7": 0, "4/7": 0, "total": 0}
                sector_counts[sec]["total"] += 1
                if score == 7:
                    sector_counts[sec]["7/7"] += 1
                elif score == 6:
                    sector_counts[sec]["6/7"] += 1
                elif score == 5:
                    sector_counts[sec]["5/7"] += 1
                elif score == 4:
                    sector_counts[sec]["4/7"] += 1

            dist_rows = []
            for sec, counts in sorted(sector_counts.items(), key=lambda x: x[1]["total"], reverse=True):
                is_top = sec in top_sectors
                dist_rows.append({
                    "Sector": sec,
                    "Top?": "YES" if is_top else "",
                    "Total": counts["total"],
                    "7/7": counts["7/7"] or "",
                    "6/7": counts["6/7"] or "",
                    "5/7": counts["5/7"] or "",
                    "4/7": counts["4/7"] or "",
                })
            st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

        # Candlestick Charts
        st.subheader("Charts")
        tickers = [s["ticker"] for s in filtered_s2[:50]]
        selected = st.multiselect("Select stocks to chart", tickers, default=tickers[:3], key="stage2_charts")
        _stage_tf_label = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="stage_chart_tf")
        _tf_map = {"Weekly": "W", "Daily": "D", "Monthly": "ME"}
        _tf = _tf_map[_stage_tf_label]

        for ticker in selected:
            cand = next((s for s in filtered_s2 if s["ticker"] == ticker), None)
            if not cand:
                continue
            df_stock = stock_data.get(ticker)
            if df_stock is None or df_stock.empty:
                st.caption(f"No price data for {ticker}")
                continue

            bases = detect_bases(df_stock)
            breakout = cand.get("breakout")
            entry_setup = cand.get("entry_setup")

            lw_markers = []
            if breakout and breakout.get("breakout"):
                try:
                    bo_date = pd.to_datetime(breakout["breakout_date"]).strftime("%Y-%m-%d")
                    lw_markers.append({
                        "time": bo_date, "position": "belowBar",
                        "color": "#4CAF50", "shape": "arrowUp", "text": "Breakout",
                    })
                except Exception:
                    pass

            if bases:
                for base in bases:
                    try:
                        bs = pd.to_datetime(base["start_date"]).strftime("%Y-%m-%d")
                        be = pd.to_datetime(base["end_date"]).strftime("%Y-%m-%d")
                        lw_markers.append({"time": bs, "position": "aboveBar", "color": "#FFD700", "shape": "square", "text": "Base Start"})
                        lw_markers.append({"time": be, "position": "aboveBar", "color": "#FFD700", "shape": "square", "text": "Base End"})
                    except Exception:
                        pass

            lw_price_lines = []
            if entry_setup:
                ep = entry_setup.get("entry_price")
                es = entry_setup.get("effective_stop")
                if ep:
                    lw_price_lines.append({"price": ep, "color": "#2196F3", "lineStyle": 2, "title": f"Entry {ep:.1f}"})
                if es:
                    lw_price_lines.append({"price": es, "color": "#F44336", "lineStyle": 2, "title": f"Stop {es:.1f}"})
            if bases:
                last_base = bases[-1]
                bh = last_base.get("base_high")
                bl = last_base.get("base_low")
                if bh:
                    lw_price_lines.append({"price": bh, "color": "#FFD700", "lineStyle": 3, "title": f"Base High {bh:.1f}"})
                if bl:
                    lw_price_lines.append({"price": bl, "color": "#FFD700", "lineStyle": 3, "title": f"Base Low {bl:.1f}"})

            chart_df = resample_ohlcv(df_stock, _tf)
            chart_html = build_lw_candlestick_html(
                chart_df, ticker, mas=[50, 150, 200],
                height=550, markers=lw_markers or None, price_lines=lw_price_lines or None,
            )
            st.components.v1.html(chart_html, height=560)


# ══════════════════════════════════════════════════════════════════
# TAB 3: How It Works
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
### Sector Pipeline (Tab 1)

**This table answers one question: which stocks should I act on, and when?**

Every stock here has already passed the pipeline's filters (in a top sector, outperforming Nifty, near highs, institutional accumulation). The columns tell you the *timing* story:

| Column | What It Tells You | Action Relevance |
|--------|------------------|------------------|
| **RS 1m / 3m / 6m** | Multi-timeframe relative strength vs Nifty. All rolling trading days from today. | **Key pattern:** 1m > 3m = accelerating (best time to enter). 1m < 3m = decelerating (wait or avoid). |
| **RS Trend** | Plain-language summary of the RS trajectory | "Accelerating" = highest priority. "Strong, Slowing" = already worked, tighten stops. "Turning Up" = early — watch for breakout. |
| **Dist from High** | How close to 52-week high | <5% = near breakout zone. >15% = still building base, needs more time. |
| **Accum** | Up-day volume / down-day volume (50d) | >1.5 = institutions buying aggressively. <1.0 = selling. |
| **Signal** | Synthesized action call from all the above + stage analysis | **BUY** = entry conditions met. **WATCH** = strong but needs trigger. **WAIT** = not ready yet. |

---

### Full Universe Stage 2 (Tab 2)

**This scans the ENTIRE stock universe** (not just top sectors) for Stage 2 setups.

**S2 Score (out of 7)** checks these criteria:
1. Price > 150 MA
2. Price > 200 MA
3. 50 MA > 150 MA (MA alignment)
4. 150 MA > 200 MA (MA alignment)
5. 200 MA rising (uptrend confirmed)
6. Price 30%+ above 52-week low (not in a hole)
7. Price within 25% of 52-week high (near highs)

**How to use the scores:**
- **7/7** = textbook Stage 2 — strongest candidates for immediate entry on breakout
- **6/7** = one criteria slightly off — still very strong, watch for the missing piece to click
- **5/7** = solid but developing — often the 200 MA hasn't turned up yet or price is still building a base
- **4/7** = early transition from Stage 1 to Stage 2 — earliest opportunities, higher risk

**Pipeline column** shows which stocks also passed the sector + RS + accumulation filters. These have the full pipeline's backing.
""")
