"""
Dashboard helper functions: chart builders, timeseries computations, cache wrappers.
"""
import json

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sector_rs import compute_mansfield_rs


# ── Badge & Action Panel Helpers ─────────────────────────────────


def cadence_badge(cadence: str) -> str:
    """Return small HTML badge indicating data cadence (W=Weekly, D=Daily, M=Monthly)."""
    colors = {"W": "#5C9DFF", "D": "#26a69a", "M": "#FF9800"}
    labels = {"W": "WEEKLY", "D": "DAILY", "M": "MONTHLY"}
    c = colors.get(cadence, "#888")
    l = labels.get(cadence, cadence)
    return (f'<span style="font-size:0.55em;background:{c}22;color:{c};'
            f'padding:1px 6px;border-radius:3px;margin-left:8px;'
            f'letter-spacing:0.08em;vertical-align:middle;">{l}</span>')


def build_daily_actions_html(breakout_alerts: list, position_actions: list, macro_changes: list) -> str:
    """Build the Daily Action Items panel HTML."""
    sections = []

    if breakout_alerts:
        items = "".join(
            f'<div style="padding:4px 0;"><span style="color:#4CAF50;font-weight:600;">{a["ticker"].replace(".NS","")}</span>'
            f' — broke above {a.get("breakout_price",0):.1f} on {a.get("volume_ratio",0):.1f}x volume</div>'
            for a in breakout_alerts
        )
        sections.append(
            f'<div style="margin-bottom:12px;">'
            f'<div style="font-size:0.75em;color:#4CAF50;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Breakout Alerts ({len(breakout_alerts)})</div>'
            f'{items}</div>'
        )

    if position_actions:
        action_colors = {"SELL": "#ef5350", "ADD": "#26a69a", "PARTIAL SELL": "#FF9800"}
        items = "".join(
            f'<div style="padding:4px 0;"><span style="font-weight:600;color:#ccc;">{a["ticker"].replace(".NS","")}</span>'
            f' — <span style="color:{action_colors.get(a.get("action", ""), "#888")};">{a.get("action", "HOLD")}</span>'
            f' <span style="color:#888;font-size:0.85em;">{a.get("reason", "")}</span></div>'
            for a in position_actions
        )
        sections.append(
            f'<div style="margin-bottom:12px;">'
            f'<div style="font-size:0.75em;color:#2196F3;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Position Actions</div>'
            f'{items}</div>'
        )

    if macro_changes:
        items = "".join(f'<div style="padding:2px 0;color:#ccc;font-size:0.85em;">{m}</div>' for m in macro_changes)
        sections.append(
            f'<div>'
            f'<div style="font-size:0.75em;color:#FF9800;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Macro Changes</div>'
            f'{items}</div>'
        )

    if not sections:
        sections.append('<div style="color:#555;font-style:italic;">No actionable items. Run Daily Check for latest data.</div>')

    content = "".join(sections)
    return (
        f'<div style="background:#0f0f1a;border:1px solid #26a69a33;border-radius:8px;padding:18px 22px;margin-bottom:16px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        f'<span style="font-size:1.0em;font-weight:700;color:#e0e0e0;">Today\'s Action Items</span>'
        f'<span style="font-size:0.55em;background:#26a69a22;color:#26a69a;padding:2px 8px;border-radius:3px;letter-spacing:0.08em;">DAILY</span>'
        f'</div>'
        f'{content}'
        f'</div>'
    )


# ── Safe Growth Computation ──────────────────────────────────────


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Compute percentage change that handles negative base values correctly.

    Standard pct_change divides by the base value: (new - old) / old.
    When the base is negative (e.g. EPS going from -10 to -2), this inverts
    the sign: (-2 - -10) / -10 = -80%, which wrongly suggests a decline.

    This function uses abs(old) as the denominator so the direction is always
    correct: (-2 - -10) / |-10| = +80%, correctly showing improvement.
    Returns NaN where the base is zero (division undefined).
    """
    old = series.shift(periods)
    change = series - old
    base = old.abs().replace(0, np.nan)
    return (change / base) * 100


# ── Timeseries Computations ─────────────────────────────────────


def compute_breadth_timeseries(
    all_stock_data: dict[str, pd.DataFrame],
    ma_period: int = 50,
    lookback: int = 90,
) -> pd.Series:
    """Compute daily % of stocks above their N-day MA over the lookback window."""
    above_ma = {}
    for ticker, df in all_stock_data.items():
        if len(df) < ma_period + lookback:
            continue
        ma = df["Close"].rolling(ma_period).mean()
        above = (df["Close"] > ma).astype(int)
        above_ma[ticker] = above

    if not above_ma:
        return pd.Series(dtype=float)

    combined = pd.DataFrame(above_ma)
    breadth = combined.mean(axis=1) * 100  # percentage
    return breadth.iloc[-lookback:]


def compute_net_new_highs_timeseries(
    all_stock_data: dict[str, pd.DataFrame],
    lookback: int = 90,
    high_low_period: int = 52 * 5,
) -> pd.DataFrame:
    """Compute daily new highs, new lows, and net for bar chart."""
    highs_count = {}
    lows_count = {}

    for ticker, df in all_stock_data.items():
        if len(df) < high_low_period:
            continue
        rolling_high = df["Close"].rolling(high_low_period).max()
        rolling_low = df["Close"].rolling(high_low_period).min()
        is_high = (df["Close"] >= rolling_high).astype(int)
        is_low = (df["Close"] <= rolling_low).astype(int)
        highs_count[ticker] = is_high
        lows_count[ticker] = is_low

    if not highs_count:
        return pd.DataFrame()

    new_highs = pd.DataFrame(highs_count).sum(axis=1)
    new_lows = pd.DataFrame(lows_count).sum(axis=1)
    result = pd.DataFrame({
        "New Highs": new_highs,
        "New Lows": -new_lows,
        "Net": new_highs - new_lows,
    })
    return result.iloc[-lookback:]


def compute_all_sector_rs_timeseries(
    sector_data: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Mansfield RS timeseries for all sectors. Returns DataFrame: date x sector."""
    rs_dict = {}
    nifty_close = nifty_df["Close"]
    for sector_name, sector_df in sector_data.items():
        try:
            rs = compute_mansfield_rs(sector_df["Close"], nifty_close)
            rs_dict[sector_name] = rs
        except Exception:
            continue
    if not rs_dict:
        return pd.DataFrame()
    return pd.DataFrame(rs_dict).dropna(how="all")


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "W") -> pd.DataFrame:
    """Resample OHLCV data to weekly or monthly candles.

    Args:
        df: DataFrame with OHLC + Volume columns and a DatetimeIndex.
        timeframe: 'D' (passthrough), 'W' (weekly), 'ME' (monthly).

    Returns:
        Resampled DataFrame.
    """
    if timeframe == "D":
        return df

    # Use W-FRI so weekly candles land on Fridays (trading days),
    # avoiding weekend dates that rangebreaks would hide.
    rule = "W-FRI" if timeframe == "W" else "BME"
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    resampled = df.resample(rule).agg(agg).dropna(subset=["Open"])
    return resampled


# ── Chart Builders ──────────────────────────────────────────────


def build_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    mas: list[int] | None = None,
    bases: list[dict] | None = None,
    breakout: dict | None = None,
    entry_setup: dict | None = None,
    height: int = 600,
) -> go.Figure:
    """Build a candlestick chart with optional MA overlays, base shading, breakout markers."""
    if mas is None:
        mas = [50, 150, 200]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Moving averages
    ma_colors = {50: "#2196F3", 150: "#FF9800", 200: "#E91E63"}
    for period in mas:
        if len(df) >= period:
            ma_vals = df["Close"].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma_vals, name=f"{period} MA",
                    line=dict(width=1.5, color=ma_colors.get(period, "#888")),
                ),
                row=1, col=1,
            )

    # Base shading
    if bases:
        for base in bases:
            try:
                start = pd.to_datetime(base["start_date"])
                end = pd.to_datetime(base["end_date"])
                fig.add_shape(
                    type="rect", x0=start, x1=end,
                    y0=base["base_low"], y1=base["base_high"],
                    fillcolor="rgba(255, 193, 7, 0.15)",
                    line=dict(color="rgba(255, 193, 7, 0.5)", width=1),
                    row=1, col=1,
                )
            except Exception:
                pass

    # Breakout marker
    if breakout and breakout.get("breakout"):
        try:
            bo_date = pd.to_datetime(breakout["breakout_date"])
            fig.add_trace(
                go.Scatter(
                    x=[bo_date], y=[breakout["breakout_price"]],
                    mode="markers", name="Breakout",
                    marker=dict(size=14, color="#4CAF50", symbol="triangle-up"),
                ),
                row=1, col=1,
            )
        except Exception:
            pass

    # Entry and stop lines
    if entry_setup:
        entry_price = entry_setup.get("entry_price")
        stop = entry_setup.get("effective_stop")
        if entry_price:
            fig.add_hline(
                y=entry_price, line_dash="dash", line_color="#2196F3",
                annotation_text=f"Entry: {entry_price:.1f}",
                row=1, col=1,
            )
        if stop:
            fig.add_hline(
                y=stop, line_dash="dash", line_color="#F44336",
                annotation_text=f"Stop: {stop:.1f}",
                row=1, col=1,
            )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.5),
        row=2, col=1,
    )

    # Volume average line
    if len(df) >= 50:
        vol_avg = df["Volume"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=vol_avg, name="50d Avg Vol",
                line=dict(width=1, color="#FF9800"),
            ),
            row=2, col=1,
        )

    fig.update_layout(
        height=height,
        title=f"{ticker}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        tickformat="%b %d", nticks=20, row=1, col=1,
    )
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        tickformat="%b %d", nticks=20, row=2, col=1,
    )

    return fig


def build_regime_gauge(value: float, title: str, thresholds: tuple) -> go.Figure:
    """Build a gauge indicator for breadth values."""
    bearish, bullish = thresholds
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2196F3"},
            "steps": [
                {"range": [0, bearish], "color": "#FFCDD2"},
                {"range": [bearish, bullish], "color": "#FFF9C4"},
                {"range": [bullish, 100], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20), template="plotly_dark")
    return fig


def build_sector_rs_chart(
    rs_df: pd.DataFrame,
    top_sectors: list[str],
    lookback: int = 180,
) -> go.Figure:
    """Multi-line RS chart, top sectors bold."""
    fig = go.Figure()
    plot_df = rs_df.iloc[-lookback:] if len(rs_df) > lookback else rs_df

    for col in plot_df.columns:
        is_top = col in top_sectors
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[col], name=col,
            line=dict(width=3 if is_top else 1, dash=None if is_top else "dot"),
            opacity=1.0 if is_top else 0.4,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Sector Mansfield RS (vs Nifty 50)",
        yaxis_title="RS %",
        height=500,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_momentum_heatmap(sector_rankings: list[dict]) -> go.Figure:
    """Sector momentum heatmap: sectors x 1w/2w/1m/3m/6m."""
    sectors = [s["sector"] for s in sector_rankings]
    periods = ["1w", "2w", "1m", "3m", "6m"]
    z = []
    for s in sector_rankings:
        row = [s["momentum"].get(p, 0) or 0 for p in periods]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=periods, y=sectors,
        colorscale=[[0, "#ef5350"], [0.5, "#ffffff"], [1, "#26a69a"]],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig.update_layout(
        title="Sector Momentum (RS Rate of Change)",
        height=max(300, len(sectors) * 35 + 100),
        template="plotly_dark",
        margin=dict(l=150, r=20, t=60, b=30),
    )
    return fig


def build_breadth_chart(
    breadth_50_ts: pd.Series,
    breadth_200_ts: pd.Series,
    thresholds_50: tuple = (40, 60),
    thresholds_200: tuple = (45, 65),
) -> go.Figure:
    """Breadth area chart with threshold bands."""
    fig = go.Figure()

    if not breadth_50_ts.empty:
        fig.add_trace(go.Scatter(
            x=breadth_50_ts.index, y=breadth_50_ts.values,
            name="% > 50 DMA", fill="tozeroy",
            line=dict(color="#2196F3"),
            fillcolor="rgba(33, 150, 243, 0.2)",
        ))
        fig.add_hline(y=thresholds_50[0], line_dash="dot", line_color="#ef5350", opacity=0.5,
                      annotation_text=f"50 DMA Bearish ({thresholds_50[0]}%)")
        fig.add_hline(y=thresholds_50[1], line_dash="dot", line_color="#26a69a", opacity=0.5,
                      annotation_text=f"50 DMA Bullish ({thresholds_50[1]}%)")

    if not breadth_200_ts.empty:
        fig.add_trace(go.Scatter(
            x=breadth_200_ts.index, y=breadth_200_ts.values,
            name="% > 200 DMA", fill="tozeroy",
            line=dict(color="#FF9800"),
            fillcolor="rgba(255, 152, 0, 0.2)",
        ))

    fig.update_layout(
        title="Market Breadth Over Time",
        yaxis_title="% of Stocks",
        yaxis_range=[0, 100],
        height=400,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_portfolio_pie(watchlist: list[dict], total_capital: float) -> go.Figure:
    """Capital allocation pie chart for BUY signals."""
    buys = [w for w in watchlist if w.get("action") == "BUY" and w.get("position")]
    if not buys:
        fig = go.Figure()
        fig.add_annotation(text="No BUY signals", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=350, template="plotly_dark")
        return fig

    labels = [b["ticker"] for b in buys]
    values = [b["position"].get("position_value", 0) for b in buys]
    allocated = sum(values)
    labels.append("Cash")
    values.append(max(0, total_capital - allocated))

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        textinfo="label+percent",
        marker=dict(line=dict(color="#1e1e1e", width=2)),
    ))
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_rs_line_chart(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    ticker: str,
    lookback: int = 180,
) -> go.Figure:
    """RS line chart: stock vs benchmark."""
    combined = pd.DataFrame({"stock": stock_close, "bench": benchmark_close}).dropna()
    if combined.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300, template="plotly_dark")
        return fig

    rs = combined["stock"] / combined["bench"]
    rs = rs.iloc[-lookback:]
    rs_ma = rs.rolling(min(50, len(rs) - 1)).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name="RS Line", line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=rs_ma.index, y=rs_ma.values, name="RS 50 MA", line=dict(color="#FF9800", width=1, dash="dot")))
    fig.update_layout(
        title=f"{ticker} Relative Strength vs Nifty",
        yaxis_title="Relative Strength",
        height=350,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


# ── Lightweight Charts (TradingView open-source) ──────────────


def _lw_chart_template(chart_js: str, height: int, title: str = "") -> str:
    """Common boilerplate for Lightweight Charts HTML embed."""
    title_html = f'<div style="color:#999;font-size:0.85em;padding:4px 8px;">{title}</div>' if title else ""
    return f"""
    <div id="lw-wrapper" style="width:100%;height:{height}px;background:#1e1e1e;border-radius:8px;overflow:hidden;">
        {title_html}
        <div id="lw-container" style="width:100%;height:{height - (28 if title else 0)}px;"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        var container = document.getElementById('lw-container');
        var chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: container.clientHeight,
            layout: {{
                background: {{ type: 'solid', color: '#1e1e1e' }},
                textColor: '#999',
            }},
            grid: {{
                vertLines: {{ color: '#2a2a2a' }},
                horzLines: {{ color: '#2a2a2a' }},
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{ borderColor: '#2a2a2a' }},
            timeScale: {{ borderColor: '#2a2a2a', timeVisible: false }},
        }});
        {chart_js}
        chart.timeScale().fitContent();
        new ResizeObserver(function() {{
            chart.applyOptions({{ width: container.clientWidth, height: container.clientHeight }});
        }}).observe(container);
    }})();
    </script>
    """


def build_lw_candlestick_html(
    df: pd.DataFrame,
    ticker: str,
    mas: list[int] | None = None,
    height: int = 550,
    markers: list[dict] | None = None,
    price_lines: list[dict] | None = None,
) -> str:
    """Build Lightweight Charts candlestick + volume + MAs as embeddable HTML."""
    if mas is None:
        mas = [50, 150, 200]

    times = df.index.strftime("%Y-%m-%d").tolist()
    candle_data = [
        {"time": t, "open": round(float(o), 2), "high": round(float(h), 2),
         "low": round(float(l), 2), "close": round(float(c), 2)}
        for t, o, h, l, c in zip(times, df["Open"], df["High"], df["Low"], df["Close"])
    ]

    vol_data = []
    if "Volume" in df.columns:
        vol_data = [
            {"time": t, "value": int(v),
             "color": "rgba(38,166,154,0.4)" if c >= o else "rgba(239,83,80,0.4)"}
            for t, v, c, o in zip(times, df["Volume"], df["Close"], df["Open"])
        ]

    ma_colors = {50: "#2196F3", 150: "#FF9800", 200: "#E91E63"}
    ma_lines_js = ""
    for period in mas:
        if len(df) >= period:
            ma_vals = df["Close"].rolling(period).mean()
            ma_data = [
                {"time": t, "value": round(float(v), 2)}
                for t, v in zip(times, ma_vals) if pd.notna(v)
            ]
            color = ma_colors.get(period, "#888")
            ma_lines_js += f"""
            var ma{period} = chart.addLineSeries({{
                color: '{color}', lineWidth: 1,
                title: '{period} MA', priceLineVisible: false,
                lastValueVisible: false,
            }});
            ma{period}.setData({json.dumps(ma_data)});
            """

    markers_js = ""
    if markers:
        sorted_markers = sorted(markers, key=lambda m: m["time"])
        markers_js = f"candleSeries.setMarkers({json.dumps(sorted_markers)});"

    price_lines_js = ""
    if price_lines:
        for pl in price_lines:
            price_lines_js += f"""
            candleSeries.createPriceLine({{
                price: {pl['price']},
                color: '{pl.get("color", "#2196F3")}',
                lineWidth: 1,
                lineStyle: {pl.get("lineStyle", 2)},
                axisLabelVisible: true,
                title: '{pl.get("title", "")}',
            }});
            """

    chart_js = f"""
    var candleSeries = chart.addCandlestickSeries({{
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    }});
    candleSeries.setData({json.dumps(candle_data)});
    {markers_js}
    {price_lines_js}
    {ma_lines_js}
    """

    if vol_data:
        chart_js += f"""
        var volSeries = chart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: 'volume',
        }});
        chart.priceScale('volume').applyOptions({{
            scaleMargins: {{ top: 0.8, bottom: 0 }},
        }});
        volSeries.setData({json.dumps(vol_data)});
        """

    return _lw_chart_template(chart_js, height, ticker)


def build_lw_line_chart_html(
    series_list: list[dict],
    title: str = "",
    height: int = 400,
    zero_line: bool = False,
) -> str:
    """Build Lightweight Charts multi-line chart.

    Each series: {name, times: list[str], values: list[float], color, lineWidth}
    """
    lines_js = ""
    for i, s in enumerate(series_list):
        data = [
            {"time": t, "value": round(float(v), 4)}
            for t, v in zip(s["times"], s["values"]) if pd.notna(v)
        ]
        color = s.get("color", "#2196F3")
        width = s.get("lineWidth", 2)
        lines_js += f"""
        var line{i} = chart.addLineSeries({{
            color: '{color}', lineWidth: {width},
            title: '{s.get("name", "")}', priceLineVisible: false,
            lastValueVisible: true,
        }});
        line{i}.setData({json.dumps(data)});
        """

    if zero_line:
        lines_js += """
        var zeroLine = chart.addLineSeries({
            color: '#555', lineWidth: 1, lineStyle: 2,
            priceLineVisible: false, lastValueVisible: false,
            title: '',
        });
        // Build zero line from first series time range
        """
        if series_list:
            first = series_list[0]
            valid_times = [t for t, v in zip(first["times"], first["values"]) if pd.notna(v)]
            if len(valid_times) >= 2:
                zero_data = [{"time": valid_times[0], "value": 0}, {"time": valid_times[-1], "value": 0}]
                lines_js += f"zeroLine.setData({json.dumps(zero_data)});"

    return _lw_chart_template(lines_js, height, title)


def build_lw_area_chart_html(
    series_list: list[dict],
    title: str = "",
    height: int = 400,
) -> str:
    """Build Lightweight Charts area chart.

    Each series: {name, times: list[str], values: list[float], color, topColor, bottomColor}
    """
    areas_js = ""
    for i, s in enumerate(series_list):
        data = [
            {"time": t, "value": round(float(v), 2)}
            for t, v in zip(s["times"], s["values"]) if pd.notna(v)
        ]
        color = s.get("color", "#2196F3")
        top_color = s.get("topColor", color.replace(")", ",0.4)").replace("rgb", "rgba") if "rgb" in color else color + "66")
        bottom_color = s.get("bottomColor", color.replace(")", ",0.0)").replace("rgb", "rgba") if "rgb" in color else color + "00")
        areas_js += f"""
        var area{i} = chart.addAreaSeries({{
            lineColor: '{color}', lineWidth: 2,
            topColor: '{top_color}',
            bottomColor: '{bottom_color}',
            title: '{s.get("name", "")}',
            priceLineVisible: false,
            lastValueVisible: true,
        }});
        area{i}.setData({json.dumps(data)});
        """

    return _lw_chart_template(areas_js, height, title)


def build_nifty_sparkline(nifty_df: pd.DataFrame, days: int = 90) -> go.Figure:
    """Small Nifty line chart for home page."""
    recent = nifty_df.iloc[-days:]
    color = "#26a69a" if recent["Close"].iloc[-1] >= recent["Close"].iloc[0] else "#ef5350"
    fig = go.Figure(go.Scatter(
        x=recent.index, y=recent["Close"],
        fill="tozeroy", line=dict(color=color, width=2),
        fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color else f"rgba(38,166,154,0.1)" if color == "#26a69a" else "rgba(239,83,80,0.1)",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
    )
    return fig


# ── Druckenmiller Derivatives ──────────────────────────────────


def compute_derivatives(
    series: pd.Series,
    roc_period: int = 20,
    pre_smooth: int = 5,
) -> dict:
    """
    Compute first and second derivatives using Druckenmiller-style weekly smoothing.

    Process:
    1. Pre-smooth raw daily data with 5-day rolling mean (weekly equivalent)
    2. Compute 4-week (20-day) ROC on smoothed data → first derivative
    3. Compute 4-week change of the ROC → second derivative (acceleration)

    This gives "rolling weekly as of today" — updates daily but reads at weekly grain.
    """
    if len(series) < roc_period * 3:
        return {"series": series, "roc": pd.Series(dtype=float), "accel": pd.Series(dtype=float)}

    # Step 1: Pre-smooth to weekly equivalent (removes daily noise)
    smoothed = series.rolling(pre_smooth, min_periods=1).mean()

    # Step 2: First derivative — 4-week rate of change
    # Use diff() instead of pct_change() because Mansfield RS can be negative,
    # and pct_change divides by the base value — when the base is negative,
    # the sign inverts, giving wrong directional signals.
    roc = smoothed.diff(roc_period)

    # Step 3: Second derivative — 1-week change in the ROC (acceleration)
    # Use 5-day (1 trading week) diff, not roc_period (20 days).
    # Using roc_period here made accel span 40 days total — far too laggy
    # to detect inflections like a bounce from a selloff.
    accel = roc.diff(pre_smooth)

    return {
        "series": smoothed,
        "roc": roc,
        "accel": accel,
    }


def detect_inflection_points(
    roc: pd.Series,
    accel: pd.Series,
    level=None,
) -> dict:
    """
    Detect Druckenmiller-style inflection points.

    When `level` is provided (e.g. current Mansfield RS value), labels become
    context-aware — a sector at RS +10 that's declining gets a different label
    than a sector at RS -10 that's declining, even if ROC/accel signs are the same.

    Without `level` (for price-based series like VIX, Nifty), uses generic labels.

    Signal matrix (ROC x Accel x Level):
    ┌──────────┬──────────────────────────────────────┬──────────────────────────────────────┐
    │          │        Accel > 0                     │        Accel < 0                     │
    ├──────────┼──────────────────────────────────────┼──────────────────────────────────────┤
    │ ROC > 0  │ Bullish Thrust (rising+accel)        │ Level>0: Bearish Inflection          │
    │          │                                      │ Level<0: Recovery Fading              │
    ├──────────┼──────────────────────────────────────┼──────────────────────────────────────┤
    │ ROC < 0  │ Level>0: Pullback Slowing            │ Level>0: Rolling Over                │
    │          │ Level<0: Bullish Inflection           │ Level<0: Bearish Breakdown           │
    └──────────┴──────────────────────────────────────┴──────────────────────────────────────┘
    """
    if roc.empty or accel.empty:
        return {"signal": "no_data", "label": "Insufficient data", "color": "#666", "icon": "--"}

    latest_roc = roc.iloc[-1]
    latest_accel = accel.iloc[-1]

    # Whether the underlying level is positive (strong) or negative (weak)
    # When level is not provided, treat as positive (price-based series)
    is_positive = level is None or level > 0

    if latest_roc < 0 and latest_accel > 0:
        if is_positive:
            return {
                "signal": "pullback_slowing",
                "label": "Pullback Slowing",
                "detail": "Was strong, now declining but the decline is losing steam — watch for bounce",
                "color": "#FFD700",
                "icon": "~>",
            }
        else:
            return {
                "signal": "bullish_inflection",
                "label": "Bullish Inflection",
                "detail": "Weak and declining but deterioration slowing — early reversal signal",
                "color": "#FFD700",
                "icon": "~>",
            }
    elif latest_roc > 0 and latest_accel > 0:
        return {
            "signal": "bullish_thrust",
            "label": "Bullish Thrust",
            "detail": "Rising and accelerating — strong trend",
            "color": "#4CAF50",
            "icon": ">>",
        }
    elif latest_roc > 0 and latest_accel < 0:
        if is_positive:
            return {
                "signal": "bearish_inflection",
                "label": "Bearish Inflection",
                "detail": "Strong and still rising but momentum fading — caution",
                "color": "#FF9800",
                "icon": "<~",
            }
        else:
            return {
                "signal": "recovery_fading",
                "label": "Recovery Fading",
                "detail": "Was recovering from weakness but recovery is losing steam",
                "color": "#FF9800",
                "icon": "<~",
            }
    elif latest_roc < 0 and latest_accel < 0:
        if is_positive:
            return {
                "signal": "rolling_over",
                "label": "Rolling Over",
                "detail": "Was strong but now declining and accelerating down — reduce exposure",
                "color": "#F44336",
                "icon": "<<",
            }
        else:
            return {
                "signal": "bearish_breakdown",
                "label": "Bearish Breakdown",
                "detail": "Weak and declining further — avoid",
                "color": "#F44336",
                "icon": "<<",
            }
    else:
        return {
            "signal": "neutral",
            "label": "Neutral",
            "detail": "No clear directional momentum",
            "color": "#888",
            "icon": "--",
        }


def build_derivative_chart(
    series: pd.Series,
    roc: pd.Series,
    accel: pd.Series,
    title: str,
    lookback: int = 90,
    height: int = 350,
) -> go.Figure:
    """
    Three-panel chart: Price/Level, First Derivative (ROC), Second Derivative (Acceleration).
    Highlights zero-line crossings and inflection zones.
    """
    s = series.iloc[-lookback:]
    r = roc.iloc[-lookback:]
    a = accel.iloc[-lookback:]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=[title, "Rate of Change (1st Deriv)", "Acceleration (2nd Deriv)"],
    )

    # Panel 1: Price/Level
    color = "#2196F3"
    fig.add_trace(
        go.Scatter(x=s.index, y=s.values, name=title, line=dict(color=color, width=2)),
        row=1, col=1,
    )

    # Panel 2: First derivative (ROC) — bar chart colored by sign
    r_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in r.values]
    fig.add_trace(
        go.Bar(x=r.index, y=r.values, name="ROC %", marker_color=r_colors, opacity=0.7),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=2, col=1)

    # Panel 3: Second derivative (Acceleration) — area chart
    a_pos = a.clip(lower=0)
    a_neg = a.clip(upper=0)
    fig.add_trace(
        go.Scatter(
            x=a.index, y=a_pos.values, name="Accel +", fill="tozeroy",
            line=dict(width=0), fillcolor="rgba(76,175,80,0.4)",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=a.index, y=a_neg.values, name="Accel -", fill="tozeroy",
            line=dict(width=0), fillcolor="rgba(244,67,54,0.4)",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=3, col=1)

    # Mark inflection zones in Panel 3 — where accel crosses zero while ROC is still negative
    # (the Druckenmiller "turn" signal)
    for i in range(1, len(a)):
        if pd.isna(a.iloc[i]) or pd.isna(a.iloc[i-1]) or pd.isna(r.iloc[i]):
            continue
        # Bullish inflection: accel crosses above zero while ROC still negative
        if a.iloc[i] > 0 and a.iloc[i-1] <= 0 and r.iloc[i] < 0:
            fig.add_vline(
                x=a.index[i], line_dash="dot", line_color="#FFD700",
                opacity=0.6, row=3, col=1,
            )

    fig.update_layout(
        height=height,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=20),
    )
    # Use date axis with clean formatting instead of category (which shows raw timestamps)
    fig.update_xaxes(tickformat="%b %d", nticks=10, tickangle=-45)

    return fig


def compute_macro_derivatives(macro_data: dict, labels: list[str]) -> dict:
    """Compute derivatives for macro indicators using their close_series.
    Returns dict: {label: {series, roc, accel, inflection}}
    """
    results = {}
    for label in labels:
        d = macro_data.get(label, {})
        series_list = d.get("close_series", [])
        dates = d.get("dates", [])
        if len(series_list) < 60:
            continue
        series = pd.Series(series_list, index=pd.to_datetime(dates))
        deriv = compute_derivatives(series)
        deriv["inflection"] = detect_inflection_points(deriv["roc"], deriv["accel"])
        results[label] = deriv
    return results


def build_macro_trend_lw_html(
    macro_data: dict,
    labels: list[str],
    title: str = "",
    height: int = 350,
    normalize: bool = True,
) -> str:
    """Multi-line trend chart using Lightweight Charts instead of Plotly.
    Reuses build_lw_line_chart_html() with normalized % change series.
    """
    colors = ["#5C9DFF", "#FF9F43", "#26d9a0", "#FF6B6B", "#AB7AFF",
              "#FFD93D", "#00BCD4", "#FF85A2"]
    series_list = []
    for i, label in enumerate(labels):
        d = macro_data.get(label, {})
        close = d.get("close_series", [])
        dates = d.get("dates", [])
        if len(close) < 2:
            continue
        if normalize:
            base = close[0] if close[0] != 0 else 1
            values = [(v / base - 1) * 100 for v in close]
        else:
            values = close
        series_list.append({
            "name": label,
            "times": dates,
            "values": values,
            "color": colors[i % len(colors)],
            "lineWidth": 2,
        })
    if not series_list:
        return ""
    return build_lw_line_chart_html(series_list, title=title, height=height, zero_line=normalize)


def build_derivative_lw_html(
    series: pd.Series,
    roc: pd.Series,
    accel: pd.Series,
    title: str,
    lookback: int = 90,
    height: int = 360,
) -> str:
    """Three-panel derivative chart using Lightweight Charts.
    Panel 1: Price line. Panel 2: ROC area (green/red). Panel 3: Acceleration area.
    Stacked vertically in a single HTML container.
    """
    s = series.iloc[-lookback:]
    r = roc.iloc[-lookback:].dropna()
    a = accel.iloc[-lookback:].dropna()

    if s.empty or r.empty or a.empty:
        return ""

    # Generate unique IDs for each sub-chart
    import hashlib
    uid = hashlib.md5(title.encode()).hexdigest()[:8]

    def _series_json(sr):
        return json.dumps([
            {"time": t.strftime("%Y-%m-%d") if hasattr(t, "strftime") else str(t),
             "value": round(float(v), 4)}
            for t, v in zip(sr.index, sr.values) if pd.notna(v)
        ])

    # Split ROC into positive/negative for area coloring
    r_pos = r.clip(lower=0)
    r_neg = r.clip(upper=0)
    a_pos = a.clip(lower=0)
    a_neg = a.clip(upper=0)

    panel_height = (height - 40) // 3

    html = f"""
    <div style="width:100%;background:#1e1e1e;border-radius:8px;overflow:hidden;padding:4px 0;">
        <div style="color:#999;font-size:0.85em;padding:4px 8px;">{title}</div>
        <div id="price_{uid}" style="width:100%;height:{panel_height}px;"></div>
        <div style="color:#666;font-size:0.7em;padding:2px 8px;">ROC %</div>
        <div id="roc_{uid}" style="width:100%;height:{panel_height}px;"></div>
        <div style="color:#666;font-size:0.7em;padding:2px 8px;">Acceleration</div>
        <div id="accel_{uid}" style="width:100%;height:{panel_height}px;"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        var chartOpts = {{
            layout: {{ background: {{ type: 'solid', color: '#1e1e1e' }}, textColor: '#999' }},
            grid: {{ vertLines: {{ color: '#2a2a2a' }}, horzLines: {{ color: '#2a2a2a' }} }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{ borderColor: '#2a2a2a' }},
            timeScale: {{ borderColor: '#2a2a2a', timeVisible: false }},
        }};

        // Panel 1: Price
        var c1 = document.getElementById('price_{uid}');
        var chart1 = LightweightCharts.createChart(c1, Object.assign({{}}, chartOpts, {{
            width: c1.clientWidth, height: c1.clientHeight
        }}));
        var priceLine = chart1.addLineSeries({{ color: '#2196F3', lineWidth: 2, priceLineVisible: false }});
        priceLine.setData({_series_json(s)});
        chart1.timeScale().fitContent();

        // Panel 2: ROC (area)
        var c2 = document.getElementById('roc_{uid}');
        var chart2 = LightweightCharts.createChart(c2, Object.assign({{}}, chartOpts, {{
            width: c2.clientWidth, height: c2.clientHeight
        }}));
        var rocPos = chart2.addAreaSeries({{
            lineColor: '#26a69a', lineWidth: 1,
            topColor: 'rgba(38,166,154,0.4)', bottomColor: 'rgba(38,166,154,0.0)',
            priceLineVisible: false, lastValueVisible: false,
        }});
        rocPos.setData({_series_json(r_pos)});
        var rocNeg = chart2.addAreaSeries({{
            lineColor: '#ef5350', lineWidth: 1,
            topColor: 'rgba(239,83,80,0.0)', bottomColor: 'rgba(239,83,80,0.4)',
            priceLineVisible: false, lastValueVisible: false,
        }});
        rocNeg.setData({_series_json(r_neg)});
        chart2.timeScale().fitContent();

        // Panel 3: Acceleration (area)
        var c3 = document.getElementById('accel_{uid}');
        var chart3 = LightweightCharts.createChart(c3, Object.assign({{}}, chartOpts, {{
            width: c3.clientWidth, height: c3.clientHeight
        }}));
        var accelPos = chart3.addAreaSeries({{
            lineColor: '#4CAF50', lineWidth: 1,
            topColor: 'rgba(76,175,80,0.4)', bottomColor: 'rgba(76,175,80,0.0)',
            priceLineVisible: false, lastValueVisible: false,
        }});
        accelPos.setData({_series_json(a_pos)});
        var accelNeg = chart3.addAreaSeries({{
            lineColor: '#F44336', lineWidth: 1,
            topColor: 'rgba(244,67,54,0.0)', bottomColor: 'rgba(244,67,54,0.4)',
            priceLineVisible: false, lastValueVisible: false,
        }});
        accelNeg.setData({_series_json(a_neg)});
        chart3.timeScale().fitContent();

        // Sync resize
        new ResizeObserver(function() {{
            chart1.applyOptions({{ width: c1.clientWidth }});
            chart2.applyOptions({{ width: c2.clientWidth }});
            chart3.applyOptions({{ width: c3.clientWidth }});
        }}).observe(c1);
    }})();
    </script>
    """
    return html


def compute_all_derivatives(
    nifty_df: pd.DataFrame,
    all_stock_data: dict[str, pd.DataFrame],
    macro_data: dict | None = None,
) -> dict:
    """
    Compute derivatives for all key series.
    Returns dict of {name: {series, roc, accel, inflection}}.
    """
    results = {}

    # 1. Nifty 50 Price
    nifty_close = nifty_df["Close"]
    d = compute_derivatives(nifty_close)
    d["inflection"] = detect_inflection_points(d["roc"], d["accel"])
    results["Nifty 50"] = d

    # 2. Breadth (% > 50 DMA)
    breadth = compute_breadth_timeseries(all_stock_data, ma_period=50, lookback=252)
    if not breadth.empty:
        d = compute_derivatives(breadth)
        d["inflection"] = detect_inflection_points(d["roc"], d["accel"])
        results["Breadth (% > 50 DMA)"] = d

    # 3. Net New Highs
    nnh_df = compute_net_new_highs_timeseries(all_stock_data, lookback=252)
    if not nnh_df.empty and "Net" in nnh_df.columns:
        net_series = nnh_df["Net"]
        d = compute_derivatives(net_series)
        d["inflection"] = detect_inflection_points(d["roc"], d["accel"])
        results["Net New Highs"] = d

    # 4. VIX proxy — realized volatility from Nifty returns
    if len(nifty_close) > 42:
        realized_vol = nifty_close.pct_change().rolling(21).std() * np.sqrt(252) * 100
        d = compute_derivatives(realized_vol.dropna())
        d["inflection"] = detect_inflection_points(d["roc"], d["accel"])
        results["Realized Volatility"] = d

    return results


# ── Formatting Helpers ──────────────────────────────────────────


def regime_color(label: str) -> str:
    """Return color for regime label."""
    return {
        "Aggressive": "#4CAF50",
        "Normal": "#8BC34A",
        "Cautious": "#FF9800",
        "Defensive": "#F44336",
        "Cash": "#9E9E9E",
    }.get(label, "#9E9E9E")


def signal_color(score: int) -> str:
    """Return color for individual signal score."""
    if score > 0:
        return "#4CAF50"
    elif score < 0:
        return "#F44336"
    return "#FF9800"


def build_macro_card_html(label: str, data: dict) -> str:
    """Build HTML for a single macro metric card."""
    price = data.get("price", 0)
    change = data.get("change", 0)
    change_pct = data.get("change_pct", 0)
    week_prices = data.get("week_prices", [])

    # Format price
    if price >= 10000:
        price_str = f"{price:,.0f}"
    elif price >= 100:
        price_str = f"{price:,.1f}"
    else:
        price_str = f"{price:,.2f}"

    # Color
    if change > 0:
        color = "#26a69a"
        arrow = "&#9650;"  # ▲
    elif change < 0:
        color = "#ef5350"
        arrow = "&#9660;"  # ▼
    else:
        color = "#888"
        arrow = "&#8212;"  # —

    # Mini sparkline SVG
    sparkline_svg = ""
    if len(week_prices) >= 2:
        mn = min(week_prices)
        mx = max(week_prices)
        rng = mx - mn if mx != mn else 1
        w, h = 70, 24
        pts_str = " ".join(
            f"{i / (len(week_prices) - 1) * w:.1f},{h - ((v - mn) / rng) * h:.1f}"
            for i, v in enumerate(week_prices)
        )
        sparkline_svg = f'<svg width="{w}" height="{h}" style="vertical-align:middle;margin-left:8px;"><polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'

    # Subtle left border by direction
    border_color = "#26a69a" if change > 0 else "#ef5350" if change < 0 else "#333"

    return (
        f'<div style="background:#0f0f1a;border-left:2px solid {border_color};'
        f'border-radius:4px;padding:12px 14px;min-height:72px;margin-bottom:10px;">'
        f'<div style="font-size:0.68em;color:#666;text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:8px;">{label}</div>'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;">'
        f'<span style="font-size:1.2em;font-weight:600;color:#e8e8e8;'
        f'font-family:monospace;">{price_str}</span>'
        f'<span style="font-size:0.78em;color:{color};font-family:monospace;">'
        f'{arrow}{change_pct:+.2f}%</span>'
        f'</div>'
        f'{sparkline_svg}'
        f'</div>'
    )


def build_macro_pulse_html(macro_data: dict) -> str:
    """Build the full Macro Pulse row HTML."""
    cards = [build_macro_card_html(label, data) for label, data in macro_data.items()]
    cards_html = "".join(cards)
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:space-between;margin-bottom:20px;">{cards_html}</div>'


def build_risk_gauge_card_html(label: str, data: dict, thresholds: dict | None = None) -> str:
    """Build HTML for a risk gauge card with contextual status label."""
    price = data.get("price", 0)
    change_pct = data.get("change_pct", 0)
    week_prices = data.get("week_prices", [])

    # Format price
    if price >= 10000:
        price_str = f"{price:,.0f}"
    elif price >= 100:
        price_str = f"{price:,.1f}"
    else:
        price_str = f"{price:,.2f}"

    # Change arrow/color
    if data.get("change", 0) > 0:
        chg_color = "#26a69a"
        arrow = "&#9650;"
    elif data.get("change", 0) < 0:
        chg_color = "#ef5350"
        arrow = "&#9660;"
    else:
        chg_color = "#888"
        arrow = "&#8212;"

    # Status label from thresholds
    status_label = ""
    status_color = "#888"
    border_color = "#333"
    if thresholds:
        low = thresholds["low"]
        high = thresholds["high"]
        labels = thresholds["labels"]
        if price <= low:
            status_label = labels[0]
            status_color = "#26a69a"
            border_color = "#26a69a"
        elif price >= high:
            status_label = labels[2]
            status_color = "#ef5350"
            border_color = "#ef5350"
        else:
            status_label = labels[1]
            status_color = "#FF9800"
            border_color = "#FF9800"

    # Status pill badge
    status_html = ""
    if status_label:
        status_html = (
            f'<div style="margin-top:8px;">'
            f'<span style="font-size:0.65em;font-weight:700;color:{status_color};'
            f'background:{status_color}18;padding:2px 8px;border-radius:3px;'
            f'letter-spacing:0.08em;">{status_label}</span></div>'
        )

    # Mini sparkline SVG
    sparkline_svg = ""
    if len(week_prices) >= 2:
        mn, mx = min(week_prices), max(week_prices)
        rng = mx - mn if mx != mn else 1
        w, h = 60, 20
        points = " ".join(f"{i / (len(week_prices) - 1) * w:.1f},{h - ((v - mn) / rng) * h:.1f}" for i, v in enumerate(week_prices))
        sparkline_svg = f'<svg width="{w}" height="{h}" style="vertical-align:middle;margin-left:6px;opacity:0.7;"><polyline points="{points}" fill="none" stroke="{chg_color}" stroke-width="1.5"/></svg>'

    return (
        f'<div style="background:#0f0f1a;border-left:2px solid {border_color};'
        f'border-radius:4px;padding:12px 14px;min-height:72px;margin-bottom:10px;">'
        f'<div style="font-size:0.68em;color:#666;text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:8px;">{label}</div>'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;">'
        f'<span style="font-size:1.2em;font-weight:600;color:#e8e8e8;'
        f'font-family:monospace;">{price_str}</span>'
        f'<span style="font-size:0.78em;color:{chg_color};font-family:monospace;">'
        f'{arrow}{change_pct:+.2f}%{sparkline_svg}</span>'
        f'</div>'
        f'{status_html}'
        f'</div>'
    )


def build_grouped_macro_pulse_html(macro_data: dict, group_labels: list[str], title: str = "") -> str:
    """Filter macro_data to a specific group and render as a flex row of cards."""
    filtered = {label: macro_data[label] for label in group_labels if label in macro_data}
    if not filtered:
        return ""
    cards_html = "".join(build_macro_card_html(label, data) for label, data in filtered.items())
    title_html = f'<div style="font-size:0.85em;color:#888;margin-bottom:6px;font-weight:600;">{title}</div>' if title else ""
    return f'{title_html}<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">{cards_html}</div>'


def build_yield_curve_indicator_html(spread: float) -> str:
    """Small HTML showing 10Y-5Y spread with status."""
    if spread > 0.25:
        status = "NORMAL"
        color = "#26a69a"
    elif spread > 0:
        status = "FLAT"
        color = "#FF9800"
    else:
        status = "INVERTED"
        color = "#ef5350"

    return (
        f'<div style="display:inline-block;background:#0f0f1a;border-radius:4px;padding:8px 16px;border-left:2px solid {color};">'
        f'<span style="color:#666;font-size:0.8em;">10Y-5Y Spread: </span>'
        f'<span style="color:#e8e8e8;font-weight:700;font-family:monospace;">{spread:+.2f}%</span>'
        f'<span style="color:{color};font-weight:700;margin-left:8px;font-size:0.85em;">{status}</span>'
        f'</div>'
    )


def build_macro_trend_chart(
    macro_data: dict,
    labels: list[str],
    title: str = "",
    height: int = 320,
    normalize: bool = True,
) -> go.Figure | None:
    """Build a multi-line 3-month trend chart from macro close_series data.

    Always normalizes to % change from start so different scales are comparable.
    """
    fig = go.Figure()
    has_data = False

    colors = ["#5C9DFF", "#FF9F43", "#26d9a0", "#FF6B6B", "#AB7AFF",
              "#FFD93D", "#00BCD4", "#FF85A2"]

    for i, label in enumerate(labels):
        d = macro_data.get(label, {})
        series = d.get("close_series", [])
        dates = d.get("dates", [])
        if not series or not dates or len(series) < 2:
            continue

        if normalize:
            base = series[0] if series[0] != 0 else 1
            values = [(v / base - 1) * 100 for v in series]
        else:
            values = series

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=dates, y=values, name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}: %{{y:+.2f}}%<extra></extra>",
        ))
        has_data = True

    if not has_data:
        return None

    if normalize:
        fig.add_hline(y=0, line_dash="dash", line_color="#444", opacity=0.6)

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#666")),
        height=height,
        template="plotly_dark",
        plot_bgcolor="#0a0a14",
        paper_bgcolor="#0a0a14",
        margin=dict(l=50, r=20, t=40 if title else 20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        yaxis_title="% Change" if normalize else "",
        yaxis=dict(gridcolor="#141420", zerolinecolor="#1e1e2e"),
        xaxis=dict(tickformat="%b %d", nticks=10, gridcolor="#141420"),
        hovermode="x unified",
    )
    return fig


def generate_template_summary(
    macro_data: dict,
    regime: dict,
    fii_dii: dict | None,
    fii_dii_flows: dict,
    sector_rankings: list[dict],
) -> str:
    """Rule-based market narrative (~4-5 sentences). Always works, no external deps."""
    parts = []

    # 1. Global tone
    global_indices = ["S&P 500", "Nasdaq", "Dow Jones", "FTSE 100", "DAX", "Nikkei 225", "Hang Seng", "Shanghai"]
    pos_count = sum(1 for idx in global_indices if macro_data.get(idx, {}).get("change_pct", 0) > 0)
    neg_count = sum(1 for idx in global_indices if macro_data.get(idx, {}).get("change_pct", 0) < 0)
    total_global = pos_count + neg_count
    if total_global > 0:
        if pos_count >= total_global * 0.7:
            parts.append(f"Global markets are risk-on with {pos_count}/{total_global} major indices in the green.")
        elif neg_count >= total_global * 0.7:
            parts.append(f"Global markets are risk-off with {neg_count}/{total_global} indices declining.")
        else:
            parts.append(f"Global markets are mixed — {pos_count} up, {neg_count} down across major indices.")

    # 2. VIX reading
    vix_data = macro_data.get("VIX", {})
    vix_price = vix_data.get("price", 0)
    if vix_price:
        if vix_price < 15:
            parts.append(f"VIX at {vix_price:.1f} signals extreme calm — complacency watch.")
        elif vix_price < 25:
            parts.append(f"VIX at {vix_price:.1f} reflects normal volatility conditions.")
        else:
            parts.append(f"VIX elevated at {vix_price:.1f} — fear is present, defensive posture warranted.")

    # 3. Dollar impact
    dxy_data = macro_data.get("Dollar Index", {})
    dxy_price = dxy_data.get("price", 0)
    if dxy_price:
        if dxy_price < 100:
            parts.append("Weak dollar supports EM flows and commodity prices.")
        elif dxy_price > 105:
            parts.append("Strong dollar pressures EM currencies and foreign flows into India.")
        else:
            parts.append(f"Dollar Index at {dxy_price:.1f} — neutral for EM flows.")

    # 4. India regime + breadth + FII
    regime_label = regime.get("label", "Unknown")
    breadth_trend = regime.get("breadth_trend", "stable")
    fii_1m = fii_dii_flows.get("1m", {}).get("fii_net")
    india_parts = [f"India regime is {regime_label} with breadth {breadth_trend}"]
    if fii_1m is not None:
        direction = "buying" if fii_1m > 0 else "selling"
        india_parts.append(f"FII monthly net {direction} of {abs(fii_1m):,.0f} Cr")
    parts.append(". ".join(india_parts) + ".")

    # 5. Crude impact
    crude_data = macro_data.get("Crude Oil", {})
    crude_price = crude_data.get("price", 0)
    if crude_price:
        if crude_price > 85:
            parts.append(f"Crude at ${crude_price:.0f} is a headwind for India's current account and inflation.")
        elif crude_price < 65:
            parts.append(f"Crude at ${crude_price:.0f} is a tailwind for India — lower import bill and inflation.")
        else:
            parts.append(f"Crude at ${crude_price:.0f} is within comfort range for India.")

    # 6. Top sectors
    if sector_rankings:
        top_names = [s.get("sector", "") for s in sector_rankings[:3]]
        parts.append(f"Sector leadership: {', '.join(top_names)}.")

    return " ".join(parts)


def build_mini_heatmap(sector_rankings: list[dict], top_n: int = 4) -> go.Figure:
    """Compact sector heatmap for home page. Returns a plotly heatmap figure."""
    if not sector_rankings:
        fig = go.Figure()
        fig.add_annotation(text="No sector data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(height=300, template="plotly_dark")
        return fig

    sectors = [s["sector"] for s in sector_rankings]
    periods = ["1w", "2w", "1m", "3m", "6m"]
    z = []
    for s in sector_rankings:
        row = [s["momentum"].get(p, 0) or 0 for p in periods]
        z.append(row)

    # Mark top sectors
    top_sector_names = [s["sector"] for s in sector_rankings[:top_n]]

    fig = go.Figure(go.Heatmap(
        z=z, x=periods, y=sectors,
        colorscale=[[0, "#ef5350"], [0.5, "#2a2a2a"], [1, "#26a69a"]],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 11},
        showscale=False,
    ))

    # Add highlight rectangles for top sectors
    for i, sector in enumerate(sectors):
        if sector in top_sector_names:
            fig.add_shape(
                type="rect",
                x0=-0.5, x1=len(periods) - 0.5,
                y0=i - 0.5, y1=i + 0.5,
                line=dict(color="#FFD700", width=2),
            )

    fig.update_layout(
        height=max(250, len(sectors) * 28 + 60),
        template="plotly_dark",
        margin=dict(l=120, r=10, t=10, b=30),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def compute_quality_radar(watchlist: list[dict]) -> dict:
    """
    Compute Quality Radar buckets from watchlist fundamentals.
    Only processes stocks that already have fundamentals fetched (post-screener).

    Returns dict with keys: roic_champions, earnings_accelerators,
                            fcf_bargains, margin_expanders
    Each value is a list of {ticker, company_name, metric_value} dicts.
    """
    roic = []
    earnings_acc = []
    fcf_bargains = []
    margin_exp = []

    for item in watchlist:
        fund = item.get("fundamentals", {})
        if not fund.get("data_available"):
            continue

        ticker = fund.get("ticker", item.get("ticker", ""))
        name = fund.get("company_name", ticker)

        # ROIC Champions: ROE > 20%
        roe = fund.get("roe")
        if roe is not None and roe > 0.20:
            roic.append({"ticker": ticker, "name": name, "value": f"{roe*100:.1f}%"})

        # Earnings Accelerators: earnings growth > revenue growth (operating leverage)
        eg = fund.get("earnings_growth")
        rg = fund.get("revenue_growth")
        if eg is not None and rg is not None and eg > rg and eg > 0:
            roic_val = f"EG {eg*100:.0f}% > RG {rg*100:.0f}%"
            earnings_acc.append({"ticker": ticker, "name": name, "value": roic_val})

        # FCF Yield Bargains: freeCashflow/marketCap > 5%
        fcf = fund.get("free_cashflow") or fund.get("freeCashflow")
        mcap = fund.get("market_cap")
        if fcf and mcap and mcap > 0:
            fcf_yield = fcf / mcap
            if fcf_yield > 0.05:
                fcf_bargains.append({"ticker": ticker, "name": name, "value": f"{fcf_yield*100:.1f}%"})

        # Margin Expanders: positive profit margin > 10%
        pm = fund.get("profit_margin")
        if pm is not None and pm > 0.10:
            margin_exp.append({"ticker": ticker, "name": name, "value": f"{pm*100:.1f}%"})

    # Sort each bucket by name
    roic.sort(key=lambda x: x["ticker"])
    earnings_acc.sort(key=lambda x: x["ticker"])
    fcf_bargains.sort(key=lambda x: x["ticker"])
    margin_exp.sort(key=lambda x: x["ticker"])

    return {
        "roic_champions": roic,
        "earnings_accelerators": earnings_acc,
        "fcf_bargains": fcf_bargains,
        "margin_expanders": margin_exp,
    }


def format_large_number(n: float | int | None) -> str:
    """Format large numbers for display (e.g., 1.5Cr, 250L)."""
    if n is None:
        return "N/A"
    if abs(n) >= 1e7:
        return f"{n / 1e7:.1f} Cr"
    if abs(n) >= 1e5:
        return f"{n / 1e5:.1f} L"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f} K"
    return f"{n:.0f}"


# ── Analyst Ratings Chart ──────────────────────────────────────


def build_analyst_ratings_chart(recs_df: pd.DataFrame) -> go.Figure | None:
    """Build horizontal stacked bar chart of analyst ratings by month.

    Args:
        recs_df: DataFrame from yfinance get_recommendations() with columns
                 like strongBuy, buy, hold, sell, strongSell indexed by period.
    """
    if recs_df is None or recs_df.empty:
        return None

    cols_map = {
        "strongBuy": ("#2E7D32", "Strong Buy"),
        "buy": ("#66BB6A", "Buy"),
        "hold": ("#FFC107", "Hold"),
        "sell": ("#EF5350", "Sell"),
        "strongSell": ("#B71C1C", "Strong Sell"),
    }

    fig = go.Figure()
    labels = [str(p) for p in recs_df.index]

    for col_key, (color, name) in cols_map.items():
        if col_key in recs_df.columns:
            fig.add_trace(go.Bar(
                y=labels, x=recs_df[col_key].values,
                name=name, orientation="h",
                marker_color=color,
            ))

    fig.update_layout(
        barmode="stack",
        title="Analyst Ratings by Period",
        height=max(250, len(labels) * 40 + 80),
        template="plotly_dark",
        margin=dict(l=80, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Number of Analysts",
    )
    return fig


# ── Financial Charts (Quarterly & Annual) ─────────────────────


def build_quarterly_financials_chart(df: pd.DataFrame) -> go.Figure | None:
    """Grouped bar: Revenue + Net Income (left Y), Line: EPS (right Y).

    Expects columns: date, revenue, net_income, diluted_eps (values in Cr for rev/NI).
    """
    if df is None or df.empty:
        return None
    if "revenue" not in df.columns:
        return None

    labels = _quarter_labels(df["date"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "revenue" in df.columns:
        fig.add_trace(go.Bar(
            x=labels, y=df["revenue"] / 1e7, name="Revenue (Cr)",
            marker_color="#2196F3", opacity=0.7,
        ), secondary_y=False)

    if "net_income" in df.columns:
        fig.add_trace(go.Bar(
            x=labels, y=df["net_income"] / 1e7, name="Net Profit (Cr)",
            marker_color="#26a69a", opacity=0.7,
        ), secondary_y=False)

    if "diluted_eps" in df.columns:
        fig.add_trace(go.Scatter(
            x=labels, y=df["diluted_eps"], name="EPS",
            line=dict(color="#FF9800", width=3), mode="lines+markers",
        ), secondary_y=True)

    fig.update_layout(
        title="Quarterly: Revenue, Net Profit & EPS",
        height=450, template="plotly_dark",
        margin=dict(l=50, r=50, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode="group",
    )
    fig.update_yaxes(title_text="Amount (Cr)", secondary_y=False)
    fig.update_yaxes(title_text="EPS (INR)", secondary_y=True)
    return fig


def build_annual_financials_chart(df: pd.DataFrame) -> go.Figure | None:
    """Same as quarterly but with year labels."""
    if df is None or df.empty or "revenue" not in df.columns:
        return None

    labels = [d.strftime("FY %Y") for d in df["date"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=df["revenue"] / 1e7, name="Revenue (Cr)",
        marker_color="#2196F3", opacity=0.7,
    ), secondary_y=False)

    if "net_income" in df.columns:
        fig.add_trace(go.Bar(
            x=labels, y=df["net_income"] / 1e7, name="Net Profit (Cr)",
            marker_color="#26a69a", opacity=0.7,
        ), secondary_y=False)

    if "diluted_eps" in df.columns:
        fig.add_trace(go.Scatter(
            x=labels, y=df["diluted_eps"], name="EPS",
            line=dict(color="#FF9800", width=3), mode="lines+markers",
        ), secondary_y=True)

    fig.update_layout(
        title="Annual: Revenue, Net Profit & EPS",
        height=450, template="plotly_dark",
        margin=dict(l=50, r=50, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode="group",
    )
    fig.update_yaxes(title_text="Amount (Cr)", secondary_y=False)
    fig.update_yaxes(title_text="EPS (INR)", secondary_y=True)
    return fig


def build_margin_trend_chart(df: pd.DataFrame, is_annual: bool = False) -> go.Figure | None:
    """Multi-line: OPM%, EBITDA Margin%, NPM% over quarters/years."""
    if df is None or df.empty:
        return None

    margin_cols = {
        "opm_pct": ("OPM %", "#2196F3"),
        "ebitda_margin_pct": ("EBITDA %", "#FF9800"),
        "npm_pct": ("NPM %", "#26a69a"),
    }
    has_any = any(c in df.columns for c in margin_cols)
    if not has_any:
        return None

    labels = [d.strftime("FY %Y") for d in df["date"]] if is_annual else _quarter_labels(df["date"])

    fig = go.Figure()
    for col_key, (name, color) in margin_cols.items():
        if col_key in df.columns:
            fig.add_trace(go.Scatter(
                x=labels, y=df[col_key], name=name,
                line=dict(color=color, width=2), mode="lines+markers",
            ))

    fig.update_layout(
        title="Margin Trends" + (" (Annual)" if is_annual else " (Quarterly)"),
        height=400, template="plotly_dark",
        yaxis_title="%",
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_growth_chart(df: pd.DataFrame, is_annual: bool = False) -> go.Figure | None:
    """Bar chart: YoY Revenue Growth%, YoY EPS Growth%."""
    if df is None or df.empty:
        return None
    if "revenue" not in df.columns:
        return None

    shift = 4 if not is_annual else 1
    rev_growth = safe_pct_change(df["revenue"], periods=shift)
    labels = [d.strftime("FY %Y") for d in df["date"]] if is_annual else _quarter_labels(df["date"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rev_growth, name="Revenue Growth % (YoY)",
        marker_color=["#26a69a" if v and v >= 0 else "#ef5350" for v in rev_growth],
        opacity=0.7,
    ))

    if "diluted_eps" in df.columns:
        eps_growth = safe_pct_change(df["diluted_eps"], periods=shift)
        fig.add_trace(go.Scatter(
            x=labels, y=eps_growth, name="EPS Growth % (YoY)",
            line=dict(color="#FF9800", width=2), mode="lines+markers",
        ))

    fig.update_layout(
        title="YoY Growth" + (" (Annual)" if is_annual else " (Quarterly)"),
        height=400, template="plotly_dark",
        yaxis_title="Growth %",
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    return fig


def build_return_ratios_chart(roe_data: list[tuple], roce_data: list[tuple], roa_data: list[tuple]) -> go.Figure | None:
    """Multi-line: ROE, ROCE, ROA over years.

    Each arg is a list of (label, value) tuples.
    """
    if not roe_data and not roce_data and not roa_data:
        return None

    fig = go.Figure()
    if roe_data:
        labels, vals = zip(*roe_data)
        fig.add_trace(go.Scatter(x=list(labels), y=list(vals), name="ROE %", line=dict(color="#2196F3", width=2), mode="lines+markers"))
    if roce_data:
        labels, vals = zip(*roce_data)
        fig.add_trace(go.Scatter(x=list(labels), y=list(vals), name="ROCE %", line=dict(color="#FF9800", width=2), mode="lines+markers"))
    if roa_data:
        labels, vals = zip(*roa_data)
        fig.add_trace(go.Scatter(x=list(labels), y=list(vals), name="ROA %", line=dict(color="#26a69a", width=2), mode="lines+markers"))

    fig.update_layout(
        title="Return Ratios (Annual)",
        height=400, template="plotly_dark",
        yaxis_title="%",
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Shareholding Chart ────────────────────────────────────────


def build_shareholding_chart(data: list[dict]) -> go.Figure | None:
    """Stacked area chart: Promoter% + FPI% + DII% + Public% = 100%."""
    if not data:
        return None

    labels = [d["date"].strftime("%b %Y") if hasattr(d["date"], "strftime") else str(d["date"]) for d in data]
    categories = [
        ("promoter_pct", "Promoter", "#1565C0"),
        ("fpi_pct", "FPI/FII", "#2E7D32"),
        ("dii_pct", "DII", "#FF9800"),
        ("public_pct", "Public", "#9E9E9E"),
    ]

    fig = go.Figure()
    for key, name, color in categories:
        vals = [d.get(key) or 0 for d in data]
        fig.add_trace(go.Scatter(
            x=labels, y=vals, name=name,
            mode="lines", stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(")", ",0.6)").replace("rgb", "rgba") if "rgb" in color else color + "99",
        ))

    fig.update_layout(
        title="Shareholding Pattern",
        height=450, template="plotly_dark",
        yaxis_title="Holding %", yaxis_range=[0, 100],
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Helper ─────────────────────────────────────────────────────


def _quarter_labels(dates: pd.Series) -> list[str]:
    """Convert dates to Indian FY quarter labels: Q3 FY25, Q2 FY25, etc."""
    labels = []
    for d in dates:
        if not hasattr(d, "month"):
            labels.append(str(d))
            continue
        month = d.month
        if month <= 3:
            fy = d.year
            q = 4
        elif month <= 6:
            fy = d.year + 1
            q = 1
        elif month <= 9:
            fy = d.year + 1
            q = 2
        else:
            fy = d.year + 1
            q = 3
        labels.append(f"Q{q} FY{fy % 100}")
    return labels


# ── Earnings Season Card ─────────────────────────────────────────


def build_portfolio_heat_bar_html(risk_pct: float, limit_pct: float) -> str:
    """Build a colored progress bar showing portfolio heat (risk % of capital vs limit)."""
    utilization = min(risk_pct / limit_pct * 100, 100) if limit_pct > 0 else 0
    if utilization < 50:
        bar_color = "#26a69a"
        label = "LOW"
    elif utilization < 80:
        bar_color = "#FF9800"
        label = "MODERATE"
    else:
        bar_color = "#ef5350"
        label = "HIGH"
    return (
        f'<div style="background:#0f0f1a;border-radius:6px;padding:14px 18px;border:1px solid #1e1e2e;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<span style="font-size:0.72em;color:#6a6a8a;text-transform:uppercase;letter-spacing:0.1em;">Portfolio Heat</span>'
        f'<span style="font-size:0.75em;font-weight:700;color:{bar_color};letter-spacing:0.08em;">{label}</span>'
        f'</div>'
        f'<div style="background:#1e1e2e;border-radius:4px;height:18px;overflow:hidden;position:relative;">'
        f'<div style="background:{bar_color};height:100%;width:{utilization:.0f}%;border-radius:4px;transition:width 0.5s;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        f'<span style="font-size:0.78em;color:#e8e8e8;font-family:monospace;">{risk_pct:.1f}% at risk</span>'
        f'<span style="font-size:0.72em;color:#555;">Limit: {limit_pct:.1f}%</span>'
        f'</div>'
        f'</div>'
    )


def build_pyramid_progress_html(filled_shares: int, target_shares: int, tranches: list[dict]) -> str:
    """Build a visual tranche tracker showing pyramid fill level."""
    if target_shares <= 0:
        return ""
    fill_pct = min(filled_shares / target_shares * 100, 100)
    # Build tranche segments
    seg_html = ""
    for i, t in enumerate(tranches):
        frac = t["shares"] / target_shares * 100 if target_shares > 0 else 0
        colors = ["#2196F3", "#26a69a", "#FF9800", "#AB7AFF"]
        color = colors[i % len(colors)]
        seg_html += f'<div style="width:{frac:.1f}%;background:{color};height:100%;display:inline-block;" title="{t.get("label","")} {t["shares"]} shares @ {t["price"]:.1f}"></div>'
    remaining_pct = max(0, 100 - fill_pct)
    return (
        f'<div style="background:#0f0f1a;border-radius:4px;padding:10px 14px;border:1px solid #1e1e2e;margin-top:8px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'<span style="font-size:0.72em;color:#6a6a8a;text-transform:uppercase;">Pyramid Fill</span>'
        f'<span style="font-size:0.78em;color:#e8e8e8;font-family:monospace;">{filled_shares}/{target_shares} shares ({fill_pct:.0f}%)</span>'
        f'</div>'
        f'<div style="background:#1e1e2e;border-radius:3px;height:14px;overflow:hidden;display:flex;">'
        f'{seg_html}'
        f'<div style="width:{remaining_pct:.1f}%;background:#1e1e2e;height:100%;display:inline-block;"></div>'
        f'</div>'
        f'</div>'
    )


def build_earnings_season_card_html(data: dict) -> str:
    """Build dark-themed HTML card for earnings season on home page.

    Single-line HTML to avoid Streamlit rendering issues.
    """
    if not data:
        return ""

    label = data.get("quarter_label", "?")
    reported = data.get("reported_count", 0)
    total = data.get("total_universe", 0)
    reported_pct = data.get("reported_pct", 0)
    agg = data.get("aggregate", {})
    rev_yoy = agg.get("revenue_yoy_pct")
    pat_yoy = agg.get("pat_yoy_pct")
    by_seg = data.get("by_segment", {})
    gd = data.get("growth_distribution", {})

    rev_color = "#26a69a" if rev_yoy and rev_yoy >= 0 else "#ef5350"
    pat_color = "#26a69a" if pat_yoy and pat_yoy >= 0 else "#ef5350"
    rev_str = f"{rev_yoy:+.1f}%" if rev_yoy is not None else "N/A"
    pat_str = f"{pat_yoy:+.1f}%" if pat_yoy is not None else "N/A"

    # Segment mini-cards
    seg_html = ""
    seg_labels = {"large": "Large Cap", "mid": "Mid Cap", "small": "Small Cap"}
    for seg_key, seg_label in seg_labels.items():
        sd = by_seg.get(seg_key, {})
        seg_pat = sd.get("pat_yoy")
        seg_color = "#26a69a" if seg_pat and seg_pat >= 0 else "#ef5350" if seg_pat is not None else "#555"
        seg_pat_str = f"{seg_pat:+.1f}%" if seg_pat is not None else "N/A"
        seg_reported = sd.get("reported", 0)
        seg_count = sd.get("count", 0)
        seg_html += (
            f'<div style="flex:1;background:#12121e;border-radius:4px;padding:8px 12px;text-align:center;margin:0 3px;">'
            f'<div style="font-size:0.65em;color:#666;text-transform:uppercase;letter-spacing:0.06em;">{seg_label}</div>'
            f'<div style="font-size:1.1em;font-weight:700;color:{seg_color};font-family:monospace;">{seg_pat_str}</div>'
            f'<div style="font-size:0.6em;color:#555;">{seg_reported}/{seg_count} reported</div>'
            f'</div>'
        )

    above_pct = gd.get("above_15pct_pct", 0)
    above_count = gd.get("above_15pct", 0)

    html = (
        f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:6px;padding:16px 20px;margin-bottom:12px;">'
        # Row 1: Header
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        f'<div style="font-size:0.72em;color:#6a6a8a;text-transform:uppercase;letter-spacing:0.1em;">Earnings Season &mdash; {label} (Consolidated)</div>'
        f'<div style="font-size:0.8em;color:#999;font-family:monospace;">{reported}/{total} reported ({reported_pct:.0f}%)</div>'
        f'</div>'
        # Row 2: Big numbers
        f'<div style="display:flex;gap:24px;margin-bottom:14px;">'
        f'<div>'
        f'<div style="font-size:0.6em;color:#555;text-transform:uppercase;letter-spacing:0.08em;">Revenue YoY</div>'
        f'<div style="font-size:1.5em;font-weight:700;color:{rev_color};font-family:monospace;">{rev_str}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:0.6em;color:#555;text-transform:uppercase;letter-spacing:0.08em;">PAT YoY</div>'
        f'<div style="font-size:1.5em;font-weight:700;color:{pat_color};font-family:monospace;">{pat_str}</div>'
        f'</div>'
        f'</div>'
        # Row 3: Segments
        f'<div style="display:flex;gap:6px;margin-bottom:10px;">'
        f'{seg_html}'
        f'</div>'
        # Row 4: Breadth
        f'<div style="font-size:0.72em;color:#888;font-family:monospace;">'
        f'{above_pct:.0f}% of universe ({above_count} stocks) reported 15%+ PAT growth</div>'
        f'</div>'
    )
    return html


# ══════════════════════════════════════════════════════════════════
# India Home Page Helpers
# ══════════════════════════════════════════════════════════════════

def compute_rsi(close: "pd.Series", period: int = 14) -> "pd.Series":
    """Compute Wilder's RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_fear_greed(regime: dict, macro_data: dict, fii_dii_flows: dict,
                       nifty_df: "pd.DataFrame") -> dict:
    """Compute India Fear & Greed index (0=Extreme Fear, 100=Extreme Greed)."""
    from config import FEAR_GREED_WEIGHTS
    from scoring_utils import normalize_score

    components = {}
    signals = regime.get("signals", {})

    # 1. VIX (inverted: low VIX = greed)
    vix_data = macro_data.get("India VIX", {})
    vix_price = vix_data.get("price", 18) if vix_data else 18
    vix_score = normalize_score(vix_price, 30, 10)
    components["vix"] = {"score": vix_score, "value": vix_price, "label": f"VIX {vix_price:.1f}"}

    # 2. Breadth 50 DMA
    b50 = signals.get("breadth_50dma", {}).get("value", 50)
    breadth_score = normalize_score(b50, 20, 80)
    components["breadth_50dma"] = {"score": breadth_score, "value": b50, "label": f"{b50:.0f}% > 50DMA"}

    # 3. Net new highs
    nnh = signals.get("net_new_highs", {})
    net = nnh.get("highs", 0) - nnh.get("lows", 0)
    nnh_score = normalize_score(net, -50, 50)
    components["net_new_highs"] = {"score": nnh_score, "value": net, "label": f"Net highs: {net:+d}"}

    # 4. Nifty vs 200 DMA
    close = nifty_df["Close"]
    ma200 = close.rolling(200).mean()
    pct_from_200 = ((close.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]) * 100
    dma_score = normalize_score(pct_from_200, -15, 15)
    components["nifty_vs_200dma"] = {"score": dma_score, "value": pct_from_200,
                                     "label": f"{pct_from_200:+.1f}% from 200DMA"}

    # 5. FII flow direction
    fii_1w = fii_dii_flows.get("1w", {}).get("fii_net", 0) if fii_dii_flows else 0
    fii_score = normalize_score(fii_1w, -5000, 5000)
    components["fii_flow"] = {"score": fii_score, "value": fii_1w,
                              "label": f"FII 1w: {fii_1w:+,.0f}Cr"}

    # 6. RSI
    rsi_series = compute_rsi(close, 14)
    rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else 50
    rsi_score = normalize_score(rsi_val, 20, 80)
    components["rsi"] = {"score": rsi_score, "value": rsi_val, "label": f"RSI: {rsi_val:.0f}"}

    # Weighted composite
    weights = FEAR_GREED_WEIGHTS
    total_weight = sum(weights.values())
    composite = sum(
        components[k]["score"] * weights.get(k, 0)
        for k in components if k in weights
    ) / total_weight
    composite = max(0, min(100, composite))

    if composite <= 20:
        label = "Extreme Fear"
    elif composite <= 40:
        label = "Fear"
    elif composite <= 60:
        label = "Neutral"
    elif composite <= 80:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {"score": round(composite, 1), "label": label, "components": components}


def build_fear_greed_gauge_html(score: float, label: str) -> str:
    """Build an SVG semicircle gauge for Fear & Greed."""
    import math
    angle = 180 - (score / 100 * 180)
    needle_x = 150 + 110 * math.cos(math.radians(angle))
    needle_y = 140 - 110 * math.sin(math.radians(angle))

    if score <= 25:
        color = "#ef5350"
    elif score <= 45:
        color = "#FF9800"
    elif score <= 55:
        color = "#888"
    elif score <= 75:
        color = "#8BC34A"
    else:
        color = "#26a69a"

    return (
        f'<div style="text-align:center;background:#0f0f1a;border:1px solid #1e1e2e;border-radius:8px;padding:16px;">'
        f'<svg viewBox="0 0 300 170" style="max-width:280px;">'
        f'<defs><linearGradient id="fg_grad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" style="stop-color:#ef5350"/>'
        f'<stop offset="25%" style="stop-color:#FF9800"/>'
        f'<stop offset="50%" style="stop-color:#888"/>'
        f'<stop offset="75%" style="stop-color:#8BC34A"/>'
        f'<stop offset="100%" style="stop-color:#26a69a"/>'
        f'</linearGradient></defs>'
        f'<path d="M 30 140 A 120 120 0 0 1 270 140" fill="none" stroke="url(#fg_grad)" stroke-width="18" stroke-linecap="round"/>'
        f'<line x1="150" y1="140" x2="{needle_x:.0f}" y2="{needle_y:.0f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="150" cy="140" r="6" fill="{color}"/>'
        f'<text x="30" y="165" fill="#666" font-size="10" text-anchor="start">FEAR</text>'
        f'<text x="270" y="165" fill="#666" font-size="10" text-anchor="end">GREED</text>'
        f'</svg>'
        f'<div style="font-size:2em;font-weight:700;color:{color};margin-top:-10px;">{score:.0f}</div>'
        f'<div style="font-size:0.85em;color:{color};font-weight:600;">{label}</div>'
        f'</div>'
    )


def build_cap_tier_card_html(name: str, price: float, change_pct: float,
                             dist_200dma: float, rsi: float, stage: str) -> str:
    """Build a market cap tier card (Nifty 50, Midcap 150, etc.)."""
    color = "#26a69a" if change_pct >= 0 else "#ef5350"
    arrow = "&#9650;" if change_pct >= 0 else "&#9660;"
    dma_color = "#26a69a" if dist_200dma > 2 else "#ef5350" if dist_200dma < -2 else "#FF9800"
    if rsi > 70:
        rsi_color, rsi_label = "#ef5350", "OB"
    elif rsi < 30:
        rsi_color, rsi_label = "#26a69a", "OS"
    else:
        rsi_color, rsi_label = "#888", ""
    stage_colors = {"S1": "#2196F3", "S2": "#26a69a", "S3": "#FF9800", "S4": "#ef5350"}
    stage_color = stage_colors.get(stage[:2] if stage else "", "#888")

    return (
        f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:8px;padding:14px;text-align:center;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<span style="font-size:0.75em;color:#666;text-transform:uppercase;letter-spacing:0.05em;">{name}</span>'
        f'<span style="font-size:0.65em;background:{stage_color}22;color:{stage_color};padding:2px 6px;border-radius:3px;">{stage}</span>'
        f'</div>'
        f'<div style="font-size:1.5em;font-weight:700;color:#e8e8e8;font-family:monospace;">{price:,.1f}</div>'
        f'<div style="font-size:0.9em;color:{color};font-family:monospace;">{arrow} {change_pct:+.2f}%</div>'
        f'<div style="display:flex;justify-content:center;gap:8px;margin-top:10px;">'
        f'<span style="font-size:0.7em;background:{dma_color}22;color:{dma_color};padding:3px 8px;border-radius:4px;">{dist_200dma:+.1f}% vs 200DMA</span>'
        f'<span style="font-size:0.7em;background:{rsi_color}22;color:{rsi_color};padding:3px 8px;border-radius:4px;">RSI {rsi:.0f} {rsi_label}</span>'
        f'</div></div>'
    )


def build_sector_heatmap_html(sector_rankings: list, top_sectors: list) -> str:
    """Build a compact sector heatmap table."""
    if not sector_rankings:
        return ""

    rows_html = ""
    for s in sector_rankings:
        name = s.get("sector") or s.get("name", "")
        is_top = name in top_sectors
        rs = s.get("mansfield_rs", s.get("rs_score", 0))
        rs_trend = s.get("rs_trend", "flat")
        trend_arrow = {"rising": "&#9650;", "falling": "&#9660;"}.get(rs_trend, "&#8212;")
        trend_color = {"rising": "#26a69a", "falling": "#ef5350"}.get(rs_trend, "#666")

        stage_info = s.get("sector_stage", {})
        if isinstance(stage_info, dict):
            stage = f"S{stage_info.get('stage', '?')}"
            substage = stage_info.get("substage", "")
            if substage:
                stage = f"{stage} {substage}"
        else:
            stage = str(stage_info) if stage_info else "-"
        stage_colors = {"S1": "#2196F3", "S2": "#26a69a", "S3": "#FF9800", "S4": "#ef5350"}
        s_color = stage_colors.get(stage[:2], "#888")

        mom = s.get("momentum", {})
        m1w = mom.get("1w", mom.get(5, 0)) if isinstance(mom, dict) else 0
        m1m = mom.get("1m", mom.get(21, 0)) if isinstance(mom, dict) else 0
        m3m = mom.get("3m", mom.get(63, 0)) if isinstance(mom, dict) else 0

        def _m_cell(v):
            c = "#26a69a" if v > 0.5 else "#ef5350" if v < -0.5 else "#666"
            return f'<td style="padding:4px 8px;text-align:right;color:{c};font-family:monospace;font-size:0.8em;">{v:+.1f}%</td>'

        rs_abs = min(abs(rs), 30)
        rs_bg = f"rgba(38,166,154,{rs_abs/60})" if rs > 0 else f"rgba(239,83,80,{rs_abs/60})"
        rs_tc = "#26a69a" if rs > 0 else "#ef5350" if rs < 0 else "#888"
        row_bg = "background:#26a69a08;" if is_top else ""
        star = "&#9733; " if is_top else ""

        signal = s.get("signal", {})
        sig_label = signal.get("label", "") if isinstance(signal, dict) else str(signal) if signal else ""
        sig_cm = {"Bullish Thrust": "#26a69a", "Bullish Inflection": "#8BC34A",
                  "Bearish Inflection": "#FF9800", "Bearish Breakdown": "#ef5350",
                  "Recovery Fading": "#ef5350", "Pullback Slowing": "#8BC34A", "Rolling Over": "#FF9800"}
        sig_c = sig_cm.get(sig_label, "#666")

        rows_html += (
            f'<tr style="border-bottom:1px solid #1a1a2e;{row_bg}">'
            f'<td style="padding:5px 8px;font-size:0.8em;color:#ccc;white-space:nowrap;">{star}{name}</td>'
            f'<td style="padding:4px 6px;text-align:center;"><span style="font-size:0.7em;background:{s_color}22;color:{s_color};padding:1px 5px;border-radius:3px;">{stage}</span></td>'
            f'<td style="padding:4px 8px;text-align:right;background:{rs_bg};color:{rs_tc};font-family:monospace;font-size:0.8em;font-weight:600;">{rs:+.1f}</td>'
            f'<td style="padding:4px 6px;text-align:center;color:{trend_color};font-size:0.8em;">{trend_arrow}</td>'
            f'{_m_cell(m1w)}{_m_cell(m1m)}{_m_cell(m3m)}'
            f'<td style="padding:4px 8px;font-size:0.7em;color:{sig_c};">{sig_label}</td>'
            f'</tr>'
        )

    return (
        f'<table style="width:100%;border-collapse:collapse;background:#0f0f1a;border-radius:6px;overflow:hidden;">'
        f'<thead><tr style="border-bottom:1px solid #2a2a3e;">'
        f'<th style="padding:6px 8px;color:#555;font-size:0.7em;text-align:left;">SECTOR</th>'
        f'<th style="padding:6px 6px;color:#555;font-size:0.7em;text-align:center;">STAGE</th>'
        f'<th style="padding:6px 8px;color:#555;font-size:0.7em;text-align:right;">RS</th>'
        f'<th style="padding:6px 6px;color:#555;font-size:0.7em;text-align:center;">TREND</th>'
        f'<th style="padding:6px 8px;color:#555;font-size:0.7em;text-align:right;">1W</th>'
        f'<th style="padding:6px 8px;color:#555;font-size:0.7em;text-align:right;">1M</th>'
        f'<th style="padding:6px 8px;color:#555;font-size:0.7em;text-align:right;">3M</th>'
        f'<th style="padding:6px 8px;color:#555;font-size:0.7em;text-align:left;">SIGNAL</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )


def build_risk_matrix_html(macro_data: dict, fii_dii: dict = None) -> str:
    """Build a one-line risk matrix summary for India."""
    factors = []

    vix = macro_data.get("India VIX", {})
    if vix:
        v = vix.get("price", 18)
        if v < 14: factors.append(("VIX calm", 1))
        elif v > 22: factors.append(("VIX elevated", -1))
        else: factors.append(("VIX normal", 0))

    crude = macro_data.get("Crude Oil", {})
    if crude:
        c = crude.get("price", 75)
        if c < 65: factors.append(("Crude cheap", 1))
        elif c > 85: factors.append(("Crude expensive", -1))
        else: factors.append(("Crude neutral", 0))

    inr = macro_data.get("USD/INR", {})
    if inr:
        r = inr.get("price", 83)
        if r < 82: factors.append(("INR stable", 1))
        elif r > 86: factors.append(("INR weak", -1))
        else: factors.append(("INR neutral", 0))

    if fii_dii:
        fii_net = fii_dii.get("fii_net", 0)
        if fii_net > 500: factors.append(("FII buying", 1))
        elif fii_net < -500: factors.append(("FII selling", -1))
        else: factors.append(("FII flat", 0))

    overall = sum(f[1] for f in factors)
    if overall >= 2:
        verdict, vc = "RISK-ON", "#26a69a"
    elif overall <= -2:
        verdict, vc = "RISK-OFF", "#ef5350"
    else:
        verdict, vc = "MIXED", "#FF9800"

    fstr = " + ".join(f'<span style="color:{"#26a69a" if s > 0 else "#ef5350" if s < 0 else "#888"};">{t}</span>' for t, s in factors)
    return (
        f'<div style="background:{vc}0d;border-left:3px solid {vc};'
        f'padding:8px 14px;border-radius:0 6px 6px 0;margin:8px 0;font-size:0.85em;">'
        f'{fstr} = <span style="color:{vc};font-weight:700;">{verdict}</span></div>'
    )


def build_valuation_card_html(nifty_price: float, pe: float, pe_bands: dict,
                              earnings_yield: float, us_10y: float = None) -> str:
    """Build Nifty valuation snapshot card with PE band bar."""
    if pe <= pe_bands["cheap"]:
        pe_label, pe_color = "CHEAP", "#26a69a"
    elif pe <= pe_bands["fair_low"]:
        pe_label, pe_color = "ATTRACTIVE", "#8BC34A"
    elif pe <= pe_bands["fair_high"]:
        pe_label, pe_color = "FAIR VALUE", "#888"
    elif pe <= pe_bands["expensive"]:
        pe_label, pe_color = "EXPENSIVE", "#FF9800"
    else:
        pe_label, pe_color = "BUBBLE", "#ef5350"

    eps = nifty_price / pe if pe > 0 else 0
    erp_html = ""
    if us_10y and us_10y > 0:
        erp = earnings_yield - us_10y
        erp_c = "#26a69a" if erp > 2 else "#ef5350" if erp < 0 else "#FF9800"
        erp_html = (
            f'<div style="margin-top:8px;font-size:0.75em;color:#888;">'
            f'Equity Risk Premium: <span style="color:{erp_c};font-weight:600;">{erp:.1f}%</span>'
            f' (EY {earnings_yield:.1f}% - US10Y {us_10y:.1f}%)</div>'
        )

    pe_min, pe_max = pe_bands["extreme_low"], pe_bands["bubble"]
    pe_pct = max(0, min(100, (pe - pe_min) / (pe_max - pe_min) * 100))
    avg_pct = (pe_bands["long_term_avg"] - pe_min) / (pe_max - pe_min) * 100

    return (
        f'<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:8px;padding:16px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<div>'
        f'<span style="font-size:0.7em;color:#666;text-transform:uppercase;">NIFTY 50 VALUATION</span>'
        f'<div style="font-size:1.6em;font-weight:700;color:{pe_color};margin-top:4px;">PE {pe:.1f}x</div>'
        f'<div style="font-size:0.8em;color:{pe_color};">{pe_label}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:0.8em;color:#999;">Implied EPS: <span style="color:#ccc;font-weight:600;">{eps:,.0f}</span></div>'
        f'<div style="font-size:0.8em;color:#999;">Earnings Yield: <span style="color:#ccc;font-weight:600;">{earnings_yield:.1f}%</span></div>'
        f'<div style="font-size:0.8em;color:#999;">LT Avg PE: <span style="color:#888;">{pe_bands["long_term_avg"]:.0f}x</span></div>'
        f'</div></div>'
        f'<div style="margin-top:12px;position:relative;height:20px;background:#1a1a2e;border-radius:10px;overflow:hidden;">'
        f'<div style="position:absolute;left:0;top:0;height:100%;width:{pe_pct:.0f}%;'
        f'background:linear-gradient(90deg,#26a69a,#8BC34A,#888,#FF9800,#ef5350);border-radius:10px;opacity:0.7;"></div>'
        f'<div style="position:absolute;left:{avg_pct:.0f}%;top:0;height:100%;width:2px;background:#fff;opacity:0.5;"></div>'
        f'<div style="position:absolute;left:{pe_pct:.0f}%;top:-2px;width:8px;height:24px;background:{pe_color};border-radius:4px;transform:translateX(-4px);"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.65em;color:#555;margin-top:3px;">'
        f'<span>{pe_bands["extreme_low"]}x</span><span>Avg {pe_bands["long_term_avg"]}x</span><span>{pe_bands["bubble"]}x</span></div>'
        f'{erp_html}</div>'
    )


def compute_advance_decline(all_stock_data: dict, lookback: int = 90) -> "pd.DataFrame":
    """Compute cumulative advance/decline line from stock data."""
    import pandas as pd
    all_closes = {}
    for ticker, df in all_stock_data.items():
        if len(df) >= lookback:
            all_closes[ticker] = df["Close"].iloc[-lookback:]
    if not all_closes:
        return pd.DataFrame()
    closes = pd.DataFrame(all_closes)
    daily_ret = closes.pct_change()
    advances = (daily_ret > 0).sum(axis=1)
    declines = (daily_ret < 0).sum(axis=1)
    ad_line = (advances - declines).cumsum()
    return pd.DataFrame({"AD_Line": ad_line, "Advances": advances, "Declines": declines})
