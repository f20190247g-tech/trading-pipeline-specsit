"""Page 5: Single Stock Deep Dive — Comprehensive Research Tool"""
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from dashboard_helpers import (
    build_candlestick_chart,
    build_rs_line_chart,
    build_analyst_ratings_chart,
    format_large_number,
    regime_color,
    resample_ohlcv,
    _quarter_labels,
    build_lw_candlestick_html,
    build_lw_line_chart_html,
    safe_pct_change,
)
from data_fetcher import fetch_price_data, get_all_stock_tickers
from stage_filter import analyze_stock_stage, detect_bases
from fundamental_veto import fetch_fundamentals, apply_fundamental_veto
from nse_data_fetcher import get_nse_fetcher
from config import SMART_MONEY_CONFIG

st.set_page_config(page_title="Stock Deep Dive", page_icon="📊", layout="wide")

if "nifty_df" not in st.session_state:
    st.info("Run a scan first from the home page (needed for Nifty benchmark data).")
    st.stop()

nifty_df = st.session_state.nifty_df
all_stock_data = st.session_state.get("stock_data", {})

# ── Ticker Selection ────────────────────────────────────────────
col_input, col_select = st.columns([1, 1])
with col_input:
    ticker_input = st.text_input("Enter ticker (e.g. RELIANCE.NS)", value="")
with col_select:
    all_tickers = sorted(get_all_stock_tickers())
    ticker_select = st.selectbox("Or select from universe", [""] + all_tickers)

ticker = ticker_input.strip().upper() if ticker_input.strip() else ticker_select
if not ticker:
    st.caption("Select or enter a ticker to begin.")
    st.stop()

# ── Fetch Price Data ────────────────────────────────────────────
with st.spinner(f"Loading data for {ticker}..."):
    if ticker in all_stock_data and not all_stock_data[ticker].empty:
        df = all_stock_data[ticker]
    else:
        fetched = fetch_price_data([ticker])
        df = fetched.get(ticker)

    if df is None or df.empty:
        st.error(f"No price data found for {ticker}. Check if the ticker is valid.")
        st.stop()

# ── Fetch yfinance detailed info ────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf_info(t):
    """Fetch full yfinance info + quarterly/annual financials + analyst data."""
    yticker = yf.Ticker(t)
    info = yticker.info or {}

    try:
        qtr_income = yticker.quarterly_income_stmt
    except Exception:
        qtr_income = pd.DataFrame()

    try:
        ann_income = yticker.income_stmt
    except Exception:
        ann_income = pd.DataFrame()

    try:
        qtr_balance = yticker.quarterly_balance_sheet
    except Exception:
        qtr_balance = pd.DataFrame()

    try:
        ann_balance = yticker.balance_sheet
    except Exception:
        ann_balance = pd.DataFrame()

    try:
        qtr_cashflow = yticker.quarterly_cashflow
    except Exception:
        qtr_cashflow = pd.DataFrame()

    try:
        ann_cashflow = yticker.cashflow
    except Exception:
        ann_cashflow = pd.DataFrame()

    # Analyst data
    try:
        recs = yticker.get_recommendations()
    except Exception:
        recs = pd.DataFrame()

    try:
        upgrades = yticker.upgrades_downgrades
    except Exception:
        upgrades = pd.DataFrame()

    try:
        targets = yticker.analyst_price_targets
    except Exception:
        targets = None

    return {
        "info": info,
        "qtr_income": qtr_income,
        "ann_income": ann_income,
        "qtr_balance": qtr_balance,
        "ann_balance": ann_balance,
        "qtr_cashflow": qtr_cashflow,
        "ann_cashflow": ann_cashflow,
        "recommendations": recs,
        "upgrades": upgrades,
        "targets": targets,
    }

with st.spinner("Fetching detailed info..."):
    yf_data = fetch_yf_info(ticker)

info = yf_data["info"]
qtr_income = yf_data["qtr_income"]
ann_income = yf_data["ann_income"]
qtr_balance = yf_data["qtr_balance"]
ann_balance = yf_data["ann_balance"]
qtr_cashflow = yf_data["qtr_cashflow"]
ann_cashflow = yf_data["ann_cashflow"]
recs_df = yf_data["recommendations"]
upgrades_df = yf_data["upgrades"]
targets_data = yf_data["targets"]

company_name = info.get("longName") or info.get("shortName") or ticker
current_price = df["Close"].iloc[-1]
prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
change = current_price - prev_close if prev_close else 0
change_pct = (change / prev_close * 100) if prev_close else 0

# ── Header ──────────────────────────────────────────────────────
chg_color = "#26a69a" if change >= 0 else "#ef5350"
st.markdown(
    f"""
    <div style="margin-bottom: 10px;">
        <span style="font-size: 1.8em; font-weight: 700;">{company_name}</span>
        <span style="font-size: 1.1em; color: #999; margin-left: 10px;">{ticker}</span>
    </div>
    <div>
        <span style="font-size: 2.2em; font-weight: 700;">{current_price:,.2f}</span>
        <span style="font-size: 1.2em; color: {chg_color}; margin-left: 10px;">
            {change:+.2f} ({change_pct:+.2f}%)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Performance Returns ─────────────────────────────────────────
def compute_return(df, days):
    if len(df) < days:
        return None
    return (df["Close"].iloc[-1] / df["Close"].iloc[-days] - 1) * 100

periods = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
returns = {label: compute_return(df, d) for label, d in periods.items()}

cols = st.columns(len(returns))
for col, (label, ret) in zip(cols, returns.items()):
    if ret is not None:
        color = "#26a69a" if ret >= 0 else "#ef5350"
        col.markdown(
            f"""<div style="text-align:center; background:#1e1e1e; border-radius:8px;
                           padding:8px 4px; border: 1px solid #333;">
                <div style="font-size:0.8em; color:#999;">{label}</div>
                <div style="font-size:1.1em; font-weight:600; color:{color};">{ret:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        col.markdown(
            f"""<div style="text-align:center; background:#1e1e1e; border-radius:8px;
                           padding:8px 4px; border: 1px solid #333;">
                <div style="font-size:0.8em; color:#999;">{label}</div>
                <div style="font-size:1.1em; color:#666;">N/A</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Key Stats + Fundamentals side by side ───────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    # Stage analysis (still needed for metrics below)
    analysis = analyze_stock_stage(df, ticker)
    stage = analysis.get("stage", {})
    breakout = analysis.get("breakout")
    entry_setup = analysis.get("entry_setup")
    vcp = analysis.get("vcp")
    bases = detect_bases(df)

    # Build markers from breakout data
    lw_markers = []
    if breakout and breakout.get("breakout"):
        try:
            bo_date = pd.to_datetime(breakout["breakout_date"]).strftime("%Y-%m-%d")
            lw_markers.append({
                "time": bo_date, "position": "belowBar",
                "color": "#4CAF50", "shape": "arrowUp",
                "text": "Breakout",
            })
        except Exception:
            pass

    # Build price lines from entry setup
    lw_price_lines = []
    if entry_setup:
        ep = entry_setup.get("entry_price")
        es = entry_setup.get("effective_stop")
        if ep:
            lw_price_lines.append({"price": ep, "color": "#2196F3", "lineStyle": 2, "title": f"Entry {ep:.1f}"})
        if es:
            lw_price_lines.append({"price": es, "color": "#F44336", "lineStyle": 2, "title": f"Stop {es:.1f}"})

    _dd_tf_label = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="deep_dive_chart_tf")
    _tf_map_dd = {"Weekly": "W", "Daily": "D", "Monthly": "ME"}
    _tf_dd = _tf_map_dd[_dd_tf_label]
    chart_df = resample_ohlcv(df, _tf_dd)
    chart_html = build_lw_candlestick_html(
        chart_df, ticker, mas=[50, 150, 200],
        height=550, markers=lw_markers or None, price_lines=lw_price_lines or None,
    )
    st.components.v1.html(chart_html, height=560)

with right_col:
    st.markdown("##### Key Fundamentals")

    def _fmt(val, fmt=",.2f", suffix="", pct=False, cr=False):
        if val is None:
            return "N/A"
        if cr:
            return f"{val / 1e7:,.2f} Cr"
        if pct:
            return f"{val * 100:.2f}%"
        return f"{val:{fmt}}{suffix}"

    fundamentals_data = {
        "Market Cap": _fmt(info.get("marketCap"), cr=True),
        "EPS (TTM)": _fmt(info.get("trailingEps")),
        "P/E Ratio": _fmt(info.get("trailingPE")),
        "Forward P/E": _fmt(info.get("forwardPE")),
        "P/B Ratio": _fmt(info.get("priceToBook")),
        "Book Value": _fmt(info.get("bookValue")),
        "EBITDA": _fmt(info.get("ebitda"), cr=True),
        "Dividend Yield": _fmt(info.get("dividendYield"), pct=True),
        "ROE": _fmt(info.get("returnOnEquity"), pct=True),
        "ROA": _fmt(info.get("returnOnAssets"), pct=True),
        "Debt/Equity": _fmt(info.get("debtToEquity")),
        "Current Ratio": _fmt(info.get("currentRatio")),
        "Profit Margin": _fmt(info.get("profitMargins"), pct=True),
        "Revenue Growth": _fmt(info.get("revenueGrowth"), pct=True),
        "Earnings Growth": _fmt(info.get("earningsGrowth"), pct=True),
    }

    fund_html = ""
    for label, val in fundamentals_data.items():
        fund_html += f"""
        <div style="display:flex; justify-content:space-between; padding:5px 0;
                    border-bottom:1px solid #333;">
            <span style="color:#999;">{label}</span>
            <span style="font-weight:600;">{val}</span>
        </div>"""
    st.markdown(fund_html, unsafe_allow_html=True)

    # Industry / Sector
    st.markdown(
        f"""<div style="margin-top:10px; font-size:0.9em;">
            <b>Industry:</b> {info.get('industry', 'N/A')} &nbsp;|&nbsp;
            <b>Sector:</b> {info.get('sector', 'N/A')}
        </div>""",
        unsafe_allow_html=True,
    )

# ── Price Stats Row ─────────────────────────────────────────────
st.divider()
ps1, ps2, ps3, ps4 = st.columns(4)
day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
day_low = info.get("dayLow") or info.get("regularMarketDayLow")
w52_high = info.get("fiftyTwoWeekHigh")
w52_low = info.get("fiftyTwoWeekLow")
vol = info.get("volume") or info.get("regularMarketVolume")
avg_vol = info.get("averageVolume")

ps1.metric("Day Range", f"{day_low:,.1f} - {day_high:,.1f}" if day_low and day_high else "N/A")
ps2.metric("52W Range", f"{w52_low:,.1f} - {w52_high:,.1f}" if w52_low and w52_high else "N/A")
ps3.metric("Volume", format_large_number(vol))
ps4.metric("Avg Volume", format_large_number(avg_vol))

# ══════════════════════════════════════════════════════════════════
# ANALYST RATINGS — Expanded
# ══════════════════════════════════════════════════════════════════
st.divider()
analyst_count = info.get("numberOfAnalystOpinions")
rec_key = info.get("recommendationKey", "").replace("_", " ").title()
target_mean = info.get("targetMeanPrice")
target_high = info.get("targetHighPrice")
target_low = info.get("targetLowPrice")

if analyst_count and analyst_count > 0:
    st.subheader("Analyst Ratings & Targets")
    ar1, ar2 = st.columns([1, 2])

    with ar1:
        # Recommendation badge
        rec_colors = {
            "Strong Buy": "#4CAF50", "Buy": "#8BC34A",
            "Hold": "#FF9800", "Sell": "#F44336", "Strong Sell": "#B71C1C",
        }
        rc = rec_colors.get(rec_key, "#2196F3")
        st.markdown(
            f"""<div style="text-align:center; padding:20px;">
                <div style="display:inline-block; border:4px solid {rc}; border-radius:50%;
                            width:100px; height:100px; line-height:100px; text-align:center;">
                    <span style="color:{rc}; font-weight:700; font-size:1.1em;">{rec_key.upper()}</span>
                </div>
                <div style="color:#999; margin-top:8px;">from {analyst_count} analysts</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with ar2:
        # Price targets
        if target_mean:
            upside = (target_mean / current_price - 1) * 100
            upside_color = "#26a69a" if upside >= 0 else "#ef5350"
            st.markdown(
                f"""<div style="padding:10px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <div><span style="color:#999;">Target Low</span><br>
                             <span style="font-size:1.2em;">{target_low:,.1f}</span></div>
                        <div style="text-align:center;">
                             <span style="color:#999;">Mean Target</span><br>
                             <span style="font-size:1.4em; font-weight:700;">{target_mean:,.1f}</span><br>
                             <span style="color:{upside_color}; font-size:0.9em;">
                                ({upside:+.1f}% upside)
                             </span></div>
                        <div style="text-align:right;"><span style="color:#999;">Target High</span><br>
                             <span style="font-size:1.2em;">{target_high:,.1f}</span></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Target range bar
            fig_target = go.Figure()
            fig_target.add_shape(
                type="line", x0=target_low, x1=target_high, y0=0, y1=0,
                line=dict(color="#555", width=6),
            )
            fig_target.add_trace(go.Scatter(
                x=[current_price], y=[0], mode="markers",
                marker=dict(size=16, color="#2196F3", symbol="diamond"),
                name="Current",
            ))
            fig_target.add_trace(go.Scatter(
                x=[target_mean], y=[0], mode="markers",
                marker=dict(size=14, color="#FF9800", symbol="circle"),
                name="Mean Target",
            ))
            fig_target.update_layout(
                height=80, margin=dict(l=0, r=0, t=0, b=0),
                template="plotly_dark", showlegend=True,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 0.5]),
                legend=dict(orientation="h", yanchor="top", y=1.5),
            )
            st.plotly_chart(fig_target, use_container_width=True)

    # Ratings stacked bar chart
    if recs_df is not None and not recs_df.empty:
        ratings_fig = build_analyst_ratings_chart(recs_df)
        if ratings_fig:
            st.plotly_chart(ratings_fig, use_container_width=True)

    # Recent upgrades/downgrades table
    if upgrades_df is not None and not upgrades_df.empty:
        st.markdown("**Recent Upgrades / Downgrades**")
        recent_upgrades = upgrades_df.head(10).copy()
        display_cols = []
        for c in ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action"]:
            if c in recent_upgrades.columns:
                display_cols.append(c)
        if not display_cols and len(recent_upgrades.columns) > 0:
            display_cols = list(recent_upgrades.columns[:5])
        if display_cols:
            st.dataframe(recent_upgrades[display_cols], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# FINANCIALS — Extended (Quarterly + Annual tabs)
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Financials")

# Fetch extended data from NSE
nse = get_nse_fetcher()
nse_quarterly = None
nse_annual = None
try:
    nse_quarterly = nse.fetch_quarterly_consolidated(ticker, num_quarters=20)
except Exception:
    pass
try:
    nse_annual = nse.fetch_annual_results(ticker, num_years=10)
except Exception:
    pass

# Helper to extract yfinance financials as a fallback DataFrame
def _yf_quarterly_to_df(qtr_inc, qtr_bal, qtr_cf):
    """Convert yfinance quarterly statements into our standard format."""
    if qtr_inc is None or qtr_inc.empty:
        return None

    def _get(stmt, name):
        if stmt is not None and not stmt.empty and name in stmt.index:
            return stmt.loc[name].sort_index()
        return None

    revenue = _get(qtr_inc, "Total Revenue")
    net_income = _get(qtr_inc, "Net Income")
    ebitda = _get(qtr_inc, "EBITDA")
    operating = _get(qtr_inc, "Operating Income")
    eps = _get(qtr_inc, "Diluted EPS")
    total_debt = _get(qtr_bal, "Total Debt")
    cash = _get(qtr_bal, "Cash And Cash Equivalents")
    ocf = _get(qtr_cf, "Operating Cash Flow")
    capex = _get(qtr_cf, "Capital Expenditure")

    if revenue is None:
        return None

    dates = revenue.index
    rows = []
    for d in dates:
        rev_val = revenue.get(d)
        ni_val = net_income.get(d) if net_income is not None else None
        eb_val = ebitda.get(d) if ebitda is not None else None
        op_val = operating.get(d) if operating is not None else None
        eps_val = eps.get(d) if eps is not None else None
        opm = (op_val / rev_val * 100) if rev_val and op_val and pd.notna(rev_val) and pd.notna(op_val) and rev_val != 0 else None
        npm = (ni_val / rev_val * 100) if rev_val and ni_val and pd.notna(rev_val) and pd.notna(ni_val) and rev_val != 0 else None
        eb_m = (eb_val / rev_val * 100) if rev_val and eb_val and pd.notna(rev_val) and pd.notna(eb_val) and rev_val != 0 else None

        row = {
            "date": d,
            "revenue": rev_val if pd.notna(rev_val) else None,
            "ebitda": eb_val if eb_val is not None and pd.notna(eb_val) else None,
            "operating_income": op_val if op_val is not None and pd.notna(op_val) else None,
            "net_income": ni_val if ni_val is not None and pd.notna(ni_val) else None,
            "diluted_eps": eps_val if eps_val is not None and pd.notna(eps_val) else None,
            "opm_pct": opm,
            "ebitda_margin_pct": eb_m,
            "npm_pct": npm,
        }

        if total_debt is not None:
            row["total_debt"] = total_debt.get(d) if d in total_debt.index else None
        if cash is not None:
            row["cash"] = cash.get(d) if d in cash.index else None
        if ocf is not None:
            row["operating_cashflow"] = ocf.get(d) if d in ocf.index else None
        if capex is not None:
            row["capex"] = capex.get(d) if d in capex.index else None

        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _yf_annual_to_df(ann_inc, ann_bal, ann_cf):
    """Convert yfinance annual statements into our standard format."""
    return _yf_quarterly_to_df(ann_inc, ann_bal, ann_cf)


# Decide which quarterly data to use
if nse_quarterly is not None and not nse_quarterly.empty and len(nse_quarterly) > 4:
    qtr_data = nse_quarterly
    qtr_source = "NSE"
else:
    qtr_data = _yf_quarterly_to_df(qtr_income, qtr_balance, qtr_cashflow)
    qtr_source = "yfinance"

if nse_annual is not None and not nse_annual.empty and len(nse_annual) > 3:
    ann_data = nse_annual
    ann_source = "NSE"
else:
    ann_data = _yf_annual_to_df(ann_income, ann_balance, ann_cashflow)
    ann_source = "yfinance"

def _fmt_cr(v):
    """Format a number as Cr or plain, returning string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if abs(v) >= 1e5:
        return f"{v / 1e7:,.0f} Cr"
    return f"{v:,.2f}"

def _fmt_pct(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.1f}%"

def _build_financials_table(data, is_annual=False):
    """Build a formatted financials DataFrame for display."""
    tbl = data.copy()
    if is_annual:
        tbl["Period"] = tbl["date"].dt.strftime("FY %Y")
    else:
        tbl["Period"] = _quarter_labels(tbl["date"])
    cols = {"Period": tbl["Period"]}
    if "revenue" in tbl.columns:
        cols["Revenue"] = tbl["revenue"].apply(_fmt_cr)
    if "ebitda" in tbl.columns:
        cols["EBITDA"] = tbl["ebitda"].apply(_fmt_cr)
    if "operating_income" in tbl.columns:
        cols["Op. Income"] = tbl["operating_income"].apply(_fmt_cr)
    if "net_income" in tbl.columns:
        cols["Net Income"] = tbl["net_income"].apply(_fmt_cr)
    if "diluted_eps" in tbl.columns:
        cols["EPS"] = tbl["diluted_eps"].apply(lambda v: f"{v:.2f}" if pd.notna(v) and v is not None else "—")
    if "opm_pct" in tbl.columns:
        cols["OPM %"] = tbl["opm_pct"].apply(_fmt_pct)
    if "npm_pct" in tbl.columns:
        cols["NPM %"] = tbl["npm_pct"].apply(_fmt_pct)
    return pd.DataFrame(cols)

def _build_margins_table(data, is_annual=False):
    """Build a margins-focused table."""
    tbl = data.copy()
    if is_annual:
        tbl["Period"] = tbl["date"].dt.strftime("FY %Y")
    else:
        tbl["Period"] = _quarter_labels(tbl["date"])
    cols = {"Period": tbl["Period"]}
    if "opm_pct" in tbl.columns:
        cols["OPM %"] = tbl["opm_pct"].apply(_fmt_pct)
    if "ebitda_margin_pct" in tbl.columns:
        cols["EBITDA Margin %"] = tbl["ebitda_margin_pct"].apply(_fmt_pct)
    if "npm_pct" in tbl.columns:
        cols["NPM %"] = tbl["npm_pct"].apply(_fmt_pct)
    return pd.DataFrame(cols)

def _build_growth_table(data, is_annual=False):
    """Build a YoY growth table."""
    tbl = data.copy().reset_index(drop=True)
    if is_annual:
        tbl["Period"] = tbl["date"].dt.strftime("FY %Y")
        shift = 1  # compare year-over-year
    else:
        tbl["Period"] = _quarter_labels(tbl["date"])
        shift = 4  # compare same quarter last year
    cols = {"Period": tbl["Period"]}
    if "revenue" in tbl.columns:
        rev_growth = safe_pct_change(tbl["revenue"], periods=shift)
        cols["Revenue Growth YoY"] = rev_growth.apply(_fmt_pct)
    if "net_income" in tbl.columns:
        ni_growth = safe_pct_change(tbl["net_income"], periods=shift)
        cols["Net Income Growth YoY"] = ni_growth.apply(_fmt_pct)
    if "diluted_eps" in tbl.columns:
        eps_growth = safe_pct_change(tbl["diluted_eps"], periods=shift)
        cols["EPS Growth YoY"] = eps_growth.apply(_fmt_pct)
    return pd.DataFrame(cols)


fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs(["Quarterly", "Annual", "Margins", "Growth"])

with fin_tab1:
    if qtr_data is not None and not qtr_data.empty:
        st.caption(f"Source: {qtr_source} ({len(qtr_data)} quarters)")
        display_df = _build_financials_table(qtr_data, is_annual=False)
        # Show newest first
        st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("Quarterly financials unavailable.")

with fin_tab2:
    if ann_data is not None and not ann_data.empty:
        st.caption(f"Source: {ann_source} ({len(ann_data)} years)")
        display_df = _build_financials_table(ann_data, is_annual=True)
        st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)

        # Return ratios from yfinance info
        roe_val = info.get("returnOnEquity")
        roa_val = info.get("returnOnAssets")
        if roe_val or roa_val:
            st.markdown("**Current Return Ratios**")
            rr1, rr2, rr3 = st.columns(3)
            rr1.metric("ROE", f"{roe_val*100:.1f}%" if roe_val else "N/A")
            rr2.metric("ROA", f"{roa_val*100:.1f}%" if roa_val else "N/A")
            if len(ann_data) >= 4 and "revenue" in ann_data.columns:
                first_rev = ann_data["revenue"].iloc[0]
                last_rev = ann_data["revenue"].iloc[-1]
                n_years = len(ann_data) - 1
                if first_rev and last_rev and first_rev > 0 and last_rev > 0 and n_years > 0:
                    cagr = ((last_rev / first_rev) ** (1 / n_years) - 1) * 100
                    rr3.metric(f"Revenue CAGR ({n_years}Y)", f"{cagr:.1f}%")
                else:
                    rr3.metric("Revenue CAGR", "N/A")
            else:
                rr3.metric("Revenue CAGR", "N/A")
    else:
        st.caption("Annual financials unavailable.")

with fin_tab3:
    margin_src = qtr_data if qtr_data is not None and not qtr_data.empty else ann_data
    if margin_src is not None and not margin_src.empty:
        is_ann = margin_src is ann_data
        label = "Annual" if is_ann else "Quarterly"
        st.caption(f"{label} Margins")
        display_df = _build_margins_table(margin_src, is_annual=is_ann)
        st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)
        # Also show annual margins if quarterly was shown
        if not is_ann and ann_data is not None and not ann_data.empty:
            st.caption("Annual Margins")
            display_df2 = _build_margins_table(ann_data, is_annual=True)
            st.dataframe(display_df2.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("Margin data unavailable.")

with fin_tab4:
    growth_src = qtr_data if qtr_data is not None and not qtr_data.empty else ann_data
    if growth_src is not None and not growth_src.empty:
        is_ann = growth_src is ann_data
        label = "Annual" if is_ann else "Quarterly (YoY)"
        st.caption(f"{label} Growth")
        display_df = _build_growth_table(growth_src, is_annual=is_ann)
        st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)
        if not is_ann and ann_data is not None and not ann_data.empty:
            st.caption("Annual Growth")
            display_df2 = _build_growth_table(ann_data, is_annual=True)
            st.dataframe(display_df2.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("Growth data unavailable.")

# ══════════════════════════════════════════════════════════════════
# SHAREHOLDING PATTERN
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Shareholding Pattern")

shareholding_data = None
try:
    shareholding_data = nse.fetch_shareholding_pattern(ticker, num_quarters=20)
except Exception:
    pass

if shareholding_data:
    st.caption(f"Source: NSE ({len(shareholding_data)} quarters)")
    sh_rows = []
    for d in shareholding_data:
        row = {
            "Quarter": d["date"].strftime("%b %Y") if hasattr(d["date"], "strftime") else str(d["date"]),
            "Promoter %": f"{d.get('promoter_pct', 0) or 0:.1f}",
        }
        if d.get("fpi_pct") is not None:
            row["FPI/FII %"] = f"{d['fpi_pct']:.1f}"
        if d.get("dii_pct") is not None:
            row["DII %"] = f"{d['dii_pct']:.1f}"
        row["Public %"] = f"{d.get('public_pct', 0) or 0:.1f}"
        if d.get("pledge_pct") is not None:
            row["Pledge %"] = f"{d['pledge_pct']:.1f}"
        sh_rows.append(row)
    st.dataframe(pd.DataFrame(sh_rows).iloc[::-1], use_container_width=True, hide_index=True)
else:
    # Fallback: yfinance snapshot
    try:
        yticker = yf.Ticker(ticker)
        mh = yticker.major_holders
        if mh is not None and not mh.empty:
            st.caption("Current snapshot only (extended history unavailable)")
            # yfinance major_holders has values in col 0 and descriptions in col 1
            if len(mh.columns) >= 2:
                fallback_df = pd.DataFrame({
                    "Metric": mh.iloc[:, 1].values,
                    "Value": mh.iloc[:, 0].apply(
                        lambda v: f"{v*100:.1f}%" if isinstance(v, float) and v < 1 else str(v)
                    ).values,
                })
            else:
                # Single column — add standard labels
                labels = [
                    "% Held by Insiders", "% Held by Institutions",
                    "% Float Held by Institutions", "Number of Institutions",
                ]
                vals = mh.iloc[:, 0].tolist()
                fallback_df = pd.DataFrame({
                    "Metric": labels[:len(vals)],
                    "Value": [
                        f"{v*100:.1f}%" if isinstance(v, float) and v < 1 else str(v)
                        for v in vals
                    ],
                })
            st.dataframe(fallback_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Shareholding data unavailable.")
    except Exception:
        st.caption("Shareholding data unavailable.")

# Pre-fetch announcements (used by Smart Money insider filter and Announcements section)
announcements = None
try:
    announcements = nse.fetch_announcements(ticker, months=3)
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════
# SMART MONEY — Delivery %, Bulk/Block Deals, Insider Activity
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Smart Money")

nse_sm = get_nse_fetcher()

# Delivery %
delivery = None
try:
    delivery = nse_sm.fetch_delivery_data(ticker)
except Exception:
    pass

if delivery and delivery.get("delivery_pct", 0) > 0:
    d_pct = delivery["delivery_pct"]
    if d_pct > SMART_MONEY_CONFIG["delivery_threshold_high"]:
        d_color = "#26a69a"
        d_label = "HIGH"
    elif d_pct > SMART_MONEY_CONFIG["delivery_threshold_low"]:
        d_color = "#FF9800"
        d_label = "MODERATE"
    else:
        d_color = "#ef5350"
        d_label = "LOW"

    st.markdown(
        f"""<div style="display:inline-block; background:#1e1e1e; border:2px solid {d_color};
            border-radius:10px; padding:12px 24px; margin-bottom:12px;">
            <span style="color:#999; font-size:0.85em;">Delivery %</span><br>
            <span style="font-size:1.6em; font-weight:700; color:{d_color};">{d_pct:.1f}%</span>
            <span style="color:{d_color}; margin-left:8px; font-size:0.9em;">{d_label}</span>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.caption("Delivery data unavailable.")

# Bulk & Block Deals (90 days)
st.markdown("**Bulk & Block Deals (90 Days)**")
ticker_clean = ticker.replace(".NS", "").replace(".BO", "").upper()

bulk_deals = []
block_deals = []
try:
    bulk_deals = nse_sm.fetch_bulk_deals()
    block_deals = nse_sm.fetch_block_deals()
except Exception:
    pass

# Filter for this ticker
ticker_deals = []
for d in bulk_deals:
    if d.get("symbol", "").upper() == ticker_clean:
        ticker_deals.append({**d, "type": "Bulk"})
for d in block_deals:
    if d.get("symbol", "").upper() == ticker_clean:
        ticker_deals.append({**d, "type": "Block"})

if ticker_deals:
    deal_rows = []
    for d in ticker_deals:
        deal_rows.append({
            "Date": d.get("date", ""),
            "Type": d.get("type", ""),
            "Client": d.get("client_name", ""),
            "Action": d.get("deal_type", ""),
            "Qty": f"{d.get('quantity', 0):,.0f}",
            "Price": f"{d.get('price', 0):,.2f}",
        })
    st.dataframe(pd.DataFrame(deal_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No bulk or block deals found for this stock in the last 90 days.")

# Insider Trading — filter announcements for insider-related keywords
if announcements:
    insider_keywords = ["insider", "sast", "acquisition", "pledge", "encumbrance"]
    insider_ann = [
        a for a in announcements
        if any(kw in (a.get("subject", "") + " " + a.get("category", "")).lower() for kw in insider_keywords)
    ]
    if insider_ann:
        st.markdown("**Insider Activity (from Announcements)**")
        ins_rows = []
        for a in insider_ann:
            ins_rows.append({
                "Date": a["date"].strftime("%Y-%m-%d") if hasattr(a["date"], "strftime") else str(a["date"]),
                "Subject": a.get("subject", "")[:150],
                "Category": a.get("category", ""),
            })
        st.dataframe(pd.DataFrame(ins_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# NSE ANNOUNCEMENTS
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Corporate Announcements (3 Months)")

if announcements:
    # Category filter
    categories = sorted(set(a.get("category", "General") for a in announcements))
    if len(categories) > 1:
        selected_cats = st.multiselect(
            "Filter by category", categories, default=categories, key="ann_cat_filter"
        )
        filtered = [a for a in announcements if a.get("category", "General") in selected_cats]
    else:
        filtered = announcements

    if filtered:
        ann_rows = []
        for a in filtered:
            ann_rows.append({
                "Date": a["date"].strftime("%Y-%m-%d") if hasattr(a["date"], "strftime") else str(a["date"]),
                "Subject": a.get("subject", ""),
                "Category": a.get("category", "General"),
            })
        st.dataframe(
            pd.DataFrame(ann_rows), use_container_width=True, hide_index=True,
            height=min(400, len(ann_rows) * 38 + 40),
        )
    else:
        st.caption("No announcements match the selected filters.")
else:
    st.caption("Announcements unavailable (NSE data not accessible).")

# ── RS Line Chart ──────────────────────────────────────────────
st.divider()
st.subheader("Relative Strength vs Nifty")

# Compute RS line and RS 50 MA for Lightweight Charts
_rs_combined = pd.DataFrame({"stock": df["Close"], "bench": nifty_df["Close"]}).dropna()
if not _rs_combined.empty:
    _rs_line = _rs_combined["stock"] / _rs_combined["bench"]
    _rs_line = _rs_line.iloc[-180:]
    _rs_ma = _rs_line.rolling(min(50, len(_rs_line) - 1)).mean()
    _rs_times = _rs_line.index.strftime("%Y-%m-%d").tolist()
    rs_chart_html = build_lw_line_chart_html(
        series_list=[
            {"name": "RS Line", "times": _rs_times, "values": _rs_line.tolist(), "color": "#2196F3", "lineWidth": 2},
            {"name": "RS 50 MA", "times": _rs_times, "values": _rs_ma.tolist(), "color": "#FF9800", "lineWidth": 1},
        ],
        title=f"{ticker} Relative Strength vs Nifty",
        height=350,
        zero_line=False,
    )
    st.components.v1.html(rs_chart_html, height=360)
else:
    st.caption("Insufficient data for RS chart.")

# ── Stage 2 Checklist ──────────────────────────────────────────
st.divider()
st.subheader("Stage Analysis")
sa1, sa2, sa3, sa4 = st.columns(4)
sa1.metric("Stage", stage.get("stage", "?"))
sa2.metric("S2 Score", f"{stage.get('s2_score', 0)}/7")
sa3.metric("Confidence", f"{stage.get('confidence', 0):.0%}")
sa4.metric("VCP", "Yes" if vcp and vcp.get("is_vcp") else "No")

if breakout and breakout.get("breakout"):
    bo1, bo2, bo3 = st.columns(3)
    bo1.metric("Breakout Price", f"{breakout['breakout_price']:.1f}")
    bo2.metric("Volume Ratio", f"{breakout.get('volume_ratio', 0):.1f}x")
    bo3.metric("Base Depth", f"{breakout.get('base_depth_pct', 0):.1f}%")

if entry_setup:
    en1, en2, en3 = st.columns(3)
    en1.metric("Entry", f"{entry_setup.get('entry_price', 0):.1f}")
    en2.metric("Stop", f"{entry_setup.get('effective_stop', 0):.1f}")
    en3.metric("Risk", f"{entry_setup.get('risk_pct', 0):.1f}%")

    # R:R Targets sell plan
    try:
        from rr_scanner import compute_multi_level_targets
        rr_targets = compute_multi_level_targets(entry_setup["entry_price"], entry_setup["effective_stop"])
        if rr_targets:
            st.markdown("**Sell Plan (R-Multiple Targets):**")
            rr_rows = [{"R": f"{t['r_multiple']}R", "Price": f"{t['price']:,.2f}", "Gain": f"+{t['gain_pct']}%", "Sell": f"{t['sell_pct']}%", "Action": t["action"]} for t in rr_targets]
            st.dataframe(pd.DataFrame(rr_rows), use_container_width=True, hide_index=True)
    except Exception:
        pass

# Weekly confirmation + consolidation + transitions
_weekly = analysis.get("weekly", {})
if _weekly:
    _w_conf = _weekly.get("weekly_confirmed")
    _w_sub = _weekly.get("weekly_s2_substage", "")
    st.caption(f"Weekly: {'✓ Confirmed' if _w_conf else '✗ Not confirmed'}{f' ({_w_sub} S2)' if _w_sub else ''} | {_weekly.get('weekly_pct_above_ma', 0):.1f}% above 30W MA")

_consol = analysis.get("consolidation", {})
if _consol and _consol.get("quality_grade"):
    st.caption(f"Consolidation: Grade {_consol['quality_grade']} | Depth: {_consol.get('depth_pct', 0):.0f}% | Vol dry-up: {_consol.get('vol_dryup_pct', 0):.0f}%")

_transitions = analysis.get("transitions", [])
if _transitions:
    _tr = _transitions[0]
    st.caption(f"Stage Transition: {_tr['transition']} ({_tr['signal']}) ~{_tr['days_ago']} days ago")

st.markdown("**Stage 2 Checklist**")
s2_checks = stage.get("s2_checks", {})
check_cols = st.columns(min(len(s2_checks), 4)) if s2_checks else []
for i, (check_name, passed) in enumerate(s2_checks.items()):
    icon = "✅" if passed else "❌"
    check_cols[i % len(check_cols)].markdown(f"{icon} {check_name.replace('_', ' ').title()}")

# ── Fundamental Veto ───────────────────────────────────────────
st.divider()
st.subheader("Fundamental Veto Check")
with st.spinner("Running fundamental veto..."):
    fundamentals = fetch_fundamentals(ticker)
    veto_result = apply_fundamental_veto(fundamentals)

if veto_result["passes"]:
    st.success(f"PASS (confidence: {veto_result['confidence']})")
else:
    st.error(f"VETOED (confidence: {veto_result['confidence']})")
    for reason in veto_result.get("reasons", []):
        st.markdown(f"- {reason}")

# ── Earnings Acceleration ─────────────────────────────────────
st.divider()
st.subheader("Earnings Acceleration")
try:
    from earnings_analysis import compute_earnings_acceleration
    _ea = compute_earnings_acceleration(ticker)
    if _ea.get("data_available"):
        _ea_trend = _ea.get("trend", "stable")
        _ea_score = _ea.get("combined_score", 0)
        ea1, ea2, ea3, ea4 = st.columns(4)
        ea1.metric("Trend", _ea_trend.title())
        ea2.metric("Score", f"{_ea_score:.0f}/100")
        ea3.metric("EPS Accel", "Yes" if _ea.get("eps_acceleration") else "No")
        ea4.metric("Rev Accel", "Yes" if _ea.get("revenue_acceleration") else "No")
        _eps_yoy = _ea.get("eps_yoy_growth", [])
        _rev_yoy = _ea.get("revenue_yoy_growth", [])
        if _eps_yoy or _rev_yoy:
            with st.expander("Quarterly YoY Growth Detail", expanded=False):
                if _eps_yoy:
                    st.markdown("**EPS YoY (newest first):** " + " → ".join(f"{v:+.1f}%" for v in _eps_yoy[:4]))
                if _rev_yoy:
                    st.markdown("**Revenue YoY (newest first):** " + " → ".join(f"{v:+.1f}%" for v in _rev_yoy[:4]))
    else:
        st.caption("Earnings data not available for this ticker.")
except Exception as _e:
    st.caption(f"Earnings analysis: {_e}")

# ── Value Analysis ────────────────────────────────────────────
st.divider()
st.subheader("Value Analysis")
try:
    from value_analysis import compute_value_score
    _va = compute_value_score(ticker)
    if _va.get("data_available"):
        _roic = _va.get("roic", {})
        _fcf = _va.get("fcf", {})
        _fort = _va.get("fortress", {})
        _moat = _va.get("moat", {})
        _dcf = _va.get("dcf", {})

        va1, va2, va3, va4, va5 = st.columns(5)
        va1.metric("Value Score", f"{_va.get('value_score', 0):.0f}/100")
        va2.metric("ROIC", f"{_roic['avg_roic_3y']:.1f}%" if _roic.get('avg_roic_3y') else "N/A")
        va3.metric("FCF Yield", f"{_fcf['fcf_yield_pct']:.1f}%" if _fcf.get('fcf_yield_pct') else "N/A")
        va4.metric("Fortress", f"{_fort['fortress_score']:.0f}" if _fort.get('fortress_score') else "N/A")
        va5.metric("Moat", _moat.get('moat_grade', 'N/A') if _moat.get('data_available') else "N/A")
        with st.expander("Value Breakdown", expanded=False):
            _roic_hist = _roic.get("roic_history", [])
            if _roic_hist:
                st.markdown("**ROIC History:** " + " → ".join(f"{r['roic_pct']:.1f}%" for r in _roic_hist[:5]))
            _dcf_upside = _dcf.get("upside_pct")
            if _dcf_upside is not None:
                _dcf_c = "#26a69a" if _dcf_upside > 0 else "#ef5350"
                st.markdown(f"**DCF Upside:** <span style='color:{_dcf_c}'>{_dcf_upside:+.1f}%</span>", unsafe_allow_html=True)
            _cash_conv = _fcf.get("cash_conversion")
            if _cash_conv is not None:
                st.markdown(f"**Cash Conversion:** {_cash_conv:.2f}x")
    else:
        st.caption("Value analysis data not available for this ticker.")
except Exception as _e:
    st.caption(f"Value analysis: {_e}")

# ── Company Description ────────────────────────────────────────
desc = info.get("longBusinessSummary")
if desc:
    st.divider()
    with st.expander("About the Company", expanded=True):
        st.write(desc)
