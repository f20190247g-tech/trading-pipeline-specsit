"""Page 4: Position Management — Track, manage, and close positions."""
import streamlit as st
import pandas as pd
from datetime import datetime

from position_manager import (
    add_position, close_position, load_positions, add_tranche,
    get_positions_summary, load_trade_history, get_trade_stats,
    calculate_position_size, get_portfolio_heat,
)
from dashboard_helpers import (
    format_large_number, build_portfolio_heat_bar_html,
    build_pyramid_progress_html,
)
from config import (
    POSITION_CONFIG, STOP_CONFIG, PROFIT_CONFIG,
    ALLOCATION_CONFIG, REGIME_POSTURE,
)

st.set_page_config(page_title="Positions", page_icon="📊", layout="wide")
st.title("Position Management")

stock_data = st.session_state.get("stock_data", {})
capital = st.session_state.get("capital", POSITION_CONFIG["total_capital"])
regime = st.session_state.get("regime", {})
regime_score = regime.get("regime_score", 0)
posture = REGIME_POSTURE.get(regime_score, REGIME_POSTURE[0])

# ── How It Works ───────────────────────────────────────────────
with st.expander("How Position Tracking Works", expanded=False):
    st.markdown(f"""
**Trailing Stop** — Automatically calculated as: *Highest Close Since Entry* minus
{STOP_CONFIG['atr_multiple']}x ATR(14). It only moves **up**, never down. If the current
price drops below the trailing stop, the system suggests SELL.

**Suggested Actions:**
- **HOLD** — Price is above the trailing stop and within normal parameters
- **ADD** — Price pulled back to the 10-day moving average and bounced (potential add point)
- **PARTIAL SELL** — Climax volume detected (daily volume >{PROFIT_CONFIG['climax_volume_multiple']}x average
  on the biggest up-day since entry). Suggests selling {PROFIT_CONFIG['partial_sell_pct']}% of the position
- **SELL** — Current price is below the trailing stop. Time to exit.
- **HOLD (8-week rule)** — Stock gained {PROFIT_CONFIG['first_target_gain_pct']}%+ in under
  {PROFIT_CONFIG['fast_gain_threshold_weeks']} weeks. Per O'Neil's rule, hold for at least
  {PROFIT_CONFIG['hold_min_weeks_if_fast']} weeks from entry to let the winner run.

**Important:** These are *suggestions* based on rules, not orders. Always apply your own judgement.
    """)


# ── Section A: Position Size Calculator ──────────────────────
st.markdown("### Position Size Calculator")
st.caption(
    "Calculate how many shares to buy based on risk, conviction, and current regime. "
    f"Regime: **{posture['label']}** | Risk/trade: **{posture['risk_per_trade_pct']}%** | "
    f"Capital: **{format_large_number(capital)}**"
)

tiers = ALLOCATION_CONFIG["conviction_tiers"]
tier_options = [f"{k} — {v['label']} ({v['target_pct']}% target)" for k, v in tiers.items()]
tier_keys = list(tiers.keys())

sz1, sz2, sz3 = st.columns(3)
with sz1:
    calc_entry = st.number_input("Entry Price", min_value=1.0, value=500.0, step=1.0, format="%.2f", key="calc_entry")
with sz2:
    calc_stop = st.number_input("Stop Price", min_value=0.1, value=460.0, step=1.0, format="%.2f", key="calc_stop")
with sz3:
    calc_tier_idx = st.selectbox("Conviction Tier", range(len(tier_options)), format_func=lambda i: tier_options[i], key="calc_tier")
    calc_conviction = tier_keys[calc_tier_idx]

if calc_stop >= calc_entry:
    st.warning("Stop must be below entry price.")
else:
    sizing = calculate_position_size(
        entry_price=calc_entry,
        stop_price=calc_stop,
        capital=capital,
        risk_pct=posture["risk_per_trade_pct"],
        conviction=calc_conviction,
    )

    pyramid_sizes = ALLOCATION_CONFIG["pyramid_sizes"]
    target_shares = sizing["target_shares"]
    initial_shares = sizing["initial_shares"]

    # Display sizing results
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Initial Shares", f"{initial_shares:,}", help=f"First tranche ({pyramid_sizes[0]*100:.0f}% of target)")
    r2.metric("Position Value", format_large_number(sizing["position_value"]))
    r3.metric("Risk Amount", format_large_number(sizing["risk_amount"]))
    r4.metric("Risk % of Capital", f"{sizing['risk_pct_of_capital']:.2f}%")

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Target Shares (Full)", f"{target_shares:,}", help=f"Full position after all pyramid adds")
    r6.metric("Target Value", format_large_number(target_shares * calc_entry))
    r7.metric("Target % of Capital", f"{sizing['target_pct']:.0f}%")
    r8.metric("Risk/Share", f"{sizing['risk_per_share']:.2f}")

    # Warnings
    max_single = POSITION_CONFIG["max_single_position_pct"]
    if sizing["position_pct_of_capital"] > max_single:
        st.warning(f"Initial position ({sizing['position_pct_of_capital']:.1f}%) exceeds max single position limit ({max_single}%).")

    # Store in session for pre-filling the add form
    st.session_state["calc_shares"] = initial_shares
    st.session_state["calc_conviction"] = calc_conviction
    st.session_state["calc_target_shares"] = target_shares

st.divider()

# ── Add Position Form ──────────────────────────────────────────
with st.expander("Add New Position", expanded=True):
    st.caption(
        "Enter the details of a position you've taken. The stop loss is a **price level** "
        "(not a percentage) — typically the low of the breakout base or a level where the "
        "trade thesis is invalidated."
    )
    with st.form("add_position_form"):
        ap1, ap2, ap3 = st.columns(3)
        with ap1:
            new_ticker = st.text_input(
                "Ticker",
                placeholder="RELIANCE.NS",
                help="Use the yfinance format: SYMBOL.NS for NSE stocks",
            )
        with ap2:
            new_entry_date = st.date_input("Entry Date", value=datetime.today())
        with ap3:
            new_entry_price = st.number_input(
                "Entry Price (INR)",
                min_value=1.0,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="The price you bought at",
            )

        ap4, ap5, ap6 = st.columns(3)
        with ap4:
            prefill_shares = st.session_state.get("calc_shares", 1)
            new_shares = st.number_input("Shares", min_value=1, step=1, value=prefill_shares)
        with ap5:
            new_stop = st.number_input(
                "Initial Stop Loss (Price)",
                min_value=1.0,
                value=90.0,
                step=1.0,
                format="%.2f",
                help="Price level where you'd exit if wrong — NOT a percentage. "
                     "Typically the base low or entry minus 2.5x ATR.",
            )
        with ap6:
            new_notes = st.text_input("Notes (optional)", placeholder="e.g. VCP breakout, 1st base")

        ap7, ap8 = st.columns(2)
        with ap7:
            add_tier_idx = st.selectbox(
                "Conviction Tier",
                range(len(tier_options)),
                format_func=lambda i: tier_options[i],
                key="add_tier",
            )
            add_conviction = tier_keys[add_tier_idx]
        with ap8:
            prefill_target = st.session_state.get("calc_target_shares", 0)
            new_target_shares = st.number_input(
                "Target Shares (Full Position)",
                min_value=0, step=1, value=prefill_target,
                help="Total shares after all pyramid adds. 0 = no pyramid tracking.",
            )

        submitted = st.form_submit_button("Add Position", type="primary")
        if submitted:
            if not new_ticker:
                st.error("Enter a ticker symbol.")
            elif new_stop >= new_entry_price:
                st.error("Stop loss must be below entry price.")
            elif new_entry_price <= 0 or new_stop <= 0:
                st.error("Price and stop must be positive.")
            else:
                risk_pct = ((new_entry_price - new_stop) / new_entry_price) * 100
                pos = add_position(
                    ticker=new_ticker.strip().upper(),
                    entry_date=new_entry_date.strftime("%Y-%m-%d"),
                    entry_price=new_entry_price,
                    shares=int(new_shares),
                    initial_stop=new_stop,
                    notes=new_notes,
                    conviction=add_conviction,
                    target_shares=int(new_target_shares),
                )
                st.success(
                    f"Added: **{pos['ticker']}** — {pos['shares']} shares @ "
                    f"{pos['entry_price']:.2f}, stop at {pos['initial_stop']:.2f} "
                    f"(risk: {risk_pct:.1f}%, conviction: {add_conviction})"
                )
                st.rerun()


# ── Active Positions ───────────────────────────────────────────
st.markdown("### Active Positions")

positions = load_positions()

if not positions:
    st.info("No active positions. Use the form above to add your first position.")
else:
    summaries = get_positions_summary(stock_data)

    if not stock_data:
        st.warning(
            "No price data in session — run a scan from the home page first so "
            "current prices, trailing stops, and suggested actions can be computed."
        )

    # ── Section B: Portfolio Heat Gauge ───────────────────────
    heat = get_portfolio_heat(summaries, capital, regime_score)
    st.markdown(
        build_portfolio_heat_bar_html(heat["total_risk_pct"], heat["regime_limit_pct"]),
        unsafe_allow_html=True,
    )

    # Per-position risk breakdown
    if heat["risk_per_position"]:
        with st.expander("Risk Breakdown by Position"):
            risk_rows = []
            for rp in heat["risk_per_position"]:
                risk_rows.append({
                    "Ticker": rp["ticker"],
                    "Current": f"{rp['current']:.1f}",
                    "Stop": f"{rp['stop']:.1f}",
                    "Shares": rp["shares"],
                    "Risk (INR)": f"{rp['risk']:,.0f}",
                    "Risk %": f"{rp['risk_pct']:.2f}%",
                })
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    st.markdown("")

    # Portfolio summary metrics
    total_positions = len(summaries)
    total_value = sum(
        s.get("current_price", s["entry_price"]) * s["shares"]
        for s in summaries if s.get("current_price")
    )
    total_cost = sum(s["entry_price"] * s["shares"] for s in summaries)
    total_pnl = sum(s.get("pnl", 0) for s in summaries)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positions", total_positions)
    m2.metric("Portfolio Value", format_large_number(total_value))
    m3.metric("Cost Basis", format_large_number(total_cost))
    pnl_delta = f"{(total_pnl / total_cost * 100):.1f}%" if total_cost > 0 else "0%"
    m4.metric("Total P&L", format_large_number(total_pnl), delta=pnl_delta)

    # Positions table
    rows = []
    for s in summaries:
        action = s.get("suggested_action", "HOLD")
        action_colors = {
            "SELL": "🔴", "PARTIAL SELL": "🟠", "ADD": "🟢", "HOLD": "⚪", "NO DATA": "⚫",
        }
        conv = s.get("conviction", "")
        rows.append({
            "Ticker": s["ticker"],
            "Conv": conv if conv else "-",
            "Entry Date": s["entry_date"],
            "Entry": f"{s['entry_price']:.1f}",
            "Current": f"{s.get('current_price', 0):.1f}" if s.get("current_price") else "N/A",
            "Shares": s["shares"],
            "P&L": f"{s.get('pnl', 0):,.0f}",
            "P&L %": f"{s.get('pnl_pct', 0):+.1f}%",
            "Days": s.get("days_held", 0),
            "Trail Stop": f"{s.get('trailing_stop', 0):.1f}",
            "Action": f"{action_colors.get(action, '')} {action}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Position Detail Cards
    st.markdown("### Position Details")
    st.caption("Expand a position for full details, action reasoning, and to close it.")

    for s in summaries:
        action = s.get("suggested_action", "HOLD")
        conv_label = s.get("conviction", "")
        conv_suffix = f" [{conv_label}]" if conv_label else ""
        with st.expander(f"{s['ticker']}{conv_suffix} — {action} | P&L: {s.get('pnl_pct', 0):+.1f}%", expanded=True):

            # Action reason — prominent at the top
            action_colors_css = {
                "SELL": "#ef5350", "PARTIAL SELL": "#FF9800",
                "ADD": "#26a69a", "HOLD": "#888", "NO DATA": "#555",
            }
            ac = action_colors_css.get(action, "#888")
            reason = s.get("action_reason", "")
            st.markdown(
                f'<div style="background:{ac}18; border-left:4px solid {ac}; '
                f'padding:10px 16px; border-radius:0 8px 8px 0; margin-bottom:12px;">'
                f'<span style="font-weight:700; color:{ac}; font-size:1.1em;">{action}</span>'
                f'<span style="color:#ccc; margin-left:12px;">{reason}</span></div>',
                unsafe_allow_html=True,
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Entry Price", f"{s['entry_price']:.2f}")
            d2.metric("Current", f"{s.get('current_price', 0):.2f}" if s.get("current_price") else "N/A")
            d3.metric("Initial Stop", f"{s['initial_stop']:.2f}",
                       help="The price level you set when entering — your 'I was wrong' point")
            d4.metric("Trailing Stop", f"{s.get('trailing_stop', 0):.2f}",
                       help=f"Auto-calculated: Highest close since entry ({s.get('highest_close', 0):.2f}) "
                            f"minus {STOP_CONFIG['atr_multiple']}x ATR ({s.get('atr', 0):.2f}). "
                            f"Only moves up, never down.")

            d5, d6, d7, d8 = st.columns(4)
            d5.metric("Shares", s["shares"])
            d6.metric("Days Held", s.get("days_held", 0))
            d7.metric("ATR (14)", f"{s.get('atr', 0):.2f}",
                       help="Average True Range over 14 days — measures daily volatility in INR. "
                            "Used to set the trailing stop distance.")
            d8.metric("High Since Entry", f"{s.get('highest_close', 0):.2f}",
                       help="Highest closing price since you entered. The trailing stop "
                            "is anchored to this value.")

            # ── Section C: Pyramid Status ─────────────────────
            tranches = s.get("tranches", [])
            target_shares = s.get("target_shares", 0)
            if target_shares > 0 and tranches:
                filled_shares = sum(t["shares"] for t in tranches)
                st.markdown(
                    build_pyramid_progress_html(filled_shares, target_shares, tranches),
                    unsafe_allow_html=True,
                )

                # Tranches table
                t_rows = []
                for i, t in enumerate(tranches):
                    t_rows.append({
                        "#": i + 1,
                        "Date": t.get("date", ""),
                        "Price": f"{t['price']:.2f}",
                        "Shares": t["shares"],
                        "Cost": f"{t['price'] * t['shares']:,.0f}",
                        "Label": t.get("label", ""),
                    })
                st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

                # Average cost across tranches
                total_cost_tranches = sum(t["price"] * t["shares"] for t in tranches)
                avg_cost = total_cost_tranches / filled_shares if filled_shares > 0 else 0
                st.caption(f"Avg cost across tranches: **{avg_cost:.2f}**")

                # Next add suggestion
                remaining = target_shares - filled_shares
                if remaining > 0 and len(tranches) < ALLOCATION_CONFIG["max_pyramid_adds"]:
                    pyramid_sizes = ALLOCATION_CONFIG["pyramid_sizes"]
                    next_idx = len(tranches)
                    next_fraction = pyramid_sizes[next_idx] if next_idx < len(pyramid_sizes) else pyramid_sizes[-1]
                    next_shares = int(target_shares * next_fraction)
                    current_price = s.get("current_price", s["entry_price"])
                    gain_pct = ((current_price / s["entry_price"]) - 1) * 100 if s["entry_price"] > 0 else 0
                    min_gain = ALLOCATION_CONFIG["pyramid_min_gain_pct"]
                    if gain_pct >= min_gain:
                        st.success(f"Next add: **{next_shares} shares** (tranche #{next_idx + 1}). "
                                   f"Stock is up {gain_pct:.1f}% — pullback to 10/21 EMA is ideal add point.")
                    else:
                        st.caption(f"Next add: {next_shares} shares — needs {min_gain}%+ gain first (currently {gain_pct:+.1f}%).")

                # Add tranche form
                with st.form(f"add_tranche_{s['id']}"):
                    st.markdown("**Add Pyramid Tranche**")
                    at1, at2, at3 = st.columns(3)
                    with at1:
                        tranche_date = st.date_input("Date", value=datetime.today(), key=f"tr_date_{s['id']}")
                    with at2:
                        tranche_price = st.number_input(
                            "Price", min_value=0.01, value=float(s.get("current_price", s["entry_price"])),
                            step=0.1, format="%.2f", key=f"tr_price_{s['id']}",
                        )
                    with at3:
                        tranche_shares = st.number_input(
                            "Shares", min_value=1, step=1, value=1, key=f"tr_shares_{s['id']}",
                        )
                    tranche_btn = st.form_submit_button("Add Tranche")
                    if tranche_btn:
                        result = add_tranche(
                            position_id=s["id"],
                            date=tranche_date.strftime("%Y-%m-%d"),
                            price=tranche_price,
                            shares=int(tranche_shares),
                        )
                        if result:
                            st.success(f"Added {tranche_shares} shares @ {tranche_price:.2f}")
                            st.rerun()

            # ── Pyramid Triggers ────────────────────────────
            triggers = s.get("pyramid_triggers", {})
            if triggers and any(triggers.values()):
                active = [k.replace("_", " ").title() for k, v in triggers.items() if v]
                trig_text = " · ".join(active)
                st.markdown(
                    f'<div style="background:#26a69a18; border-left:3px solid #26a69a; '
                    f'padding:8px 12px; border-radius:0 6px 6px 0; margin:8px 0;">'
                    f'<span style="font-weight:600; color:#26a69a;">Pyramid Triggers Active:</span> '
                    f'<span style="color:#ccc;">{trig_text}</span></div>',
                    unsafe_allow_html=True,
                )

            # 8-week hold warning
            if s.get("hold_until"):
                st.info(
                    f"**8-week hold rule active until {s['hold_until']}** — "
                    f"This stock gained 20%+ quickly. O'Neil's research shows these "
                    f"runners often become the biggest winners. Sit tight unless the "
                    f"stop is hit.",
                    icon="⏳",
                )

            # Notes
            if s.get("notes"):
                st.caption(f"Notes: {s['notes']}")

            # Close position form
            st.markdown("---")
            st.markdown("**Close this position**")
            st.caption("Fill in when you actually exit the trade.")
            with st.form(f"close_{s['id']}"):
                cl1, cl2, cl3 = st.columns(3)
                with cl1:
                    exit_date = st.date_input("Exit Date", value=datetime.today(), key=f"exit_date_{s['id']}")
                with cl2:
                    exit_price = st.number_input(
                        "Exit Price (INR)",
                        value=s.get("current_price", s["entry_price"]),
                        min_value=0.01, step=0.1, format="%.2f",
                        key=f"exit_price_{s['id']}",
                        help="The price you sold at",
                    )
                with cl3:
                    exit_reason = st.text_input(
                        "Exit Reason",
                        key=f"exit_reason_{s['id']}",
                        placeholder="e.g. Hit trailing stop, Took profit, Thesis broken",
                    )

                close_btn = st.form_submit_button("Close Position")
                if close_btn:
                    trade = close_position(
                        position_id=s["id"],
                        exit_date=exit_date.strftime("%Y-%m-%d"),
                        exit_price=exit_price,
                        reason=exit_reason,
                    )
                    if trade:
                        st.success(
                            f"Closed {trade['ticker']}: P&L {trade['pnl']:+,.0f} ({trade['pnl_pct']:+.1f}%) "
                            f"in {trade['days_held']} days"
                        )
                        st.rerun()


# ── Trade History ──────────────────────────────────────────────
st.divider()
st.markdown("### Trade History")
st.caption(
    "Every closed position is logged here. Use this to review your edge — "
    "are your winners bigger than your losers? Is your win rate above 40%?"
)

history = load_trade_history()
if not history:
    st.caption("No closed trades yet. Positions you close above will appear here with full P&L stats.")
else:
    # Summary stats
    stats = get_trade_stats()
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Trades", stats["total_trades"])
    s2.metric("Win Rate", f"{stats['win_rate']:.0f}%",
              help="% of trades closed at a profit. Aim for >40% with good risk/reward.")
    s3.metric("Avg Gain", f"{stats['avg_gain']:+.1f}%",
              help="Average % gain on winning trades")
    s4.metric("Avg Loss", f"{stats['avg_loss']:+.1f}%",
              help="Average % loss on losing trades. Should be smaller than avg gain.")
    s5.metric("Total P&L", format_large_number(stats["total_pnl"]))

    # Expectancy
    if stats["total_trades"] >= 5:
        wr = stats["win_rate"] / 100
        expectancy = (wr * stats["avg_gain"]) + ((1 - wr) * stats["avg_loss"])
        exp_color = "#26a69a" if expectancy > 0 else "#ef5350"
        st.markdown(
            f'<div style="font-size:0.9em; color:{exp_color};">'
            f'Expectancy per trade: <b>{expectancy:+.2f}%</b> — '
            f'{"positive edge, keep going" if expectancy > 0 else "negative edge, review your process"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # History table
    hist_ticker_search = st.text_input("Search ticker", key="history_ticker_search").strip().upper()
    h_rows = []
    for t in reversed(history):
        if hist_ticker_search and hist_ticker_search not in t["ticker"].upper():
            continue
        h_rows.append({
            "Ticker": t["ticker"],
            "Entry": f"{t['entry_price']:.1f}",
            "Exit": f"{t.get('exit_price', 0):.1f}",
            "Shares": t["shares"],
            "P&L": f"{t.get('pnl', 0):+,.0f}",
            "P&L %": f"{t.get('pnl_pct', 0):+.1f}%",
            "Days": t.get("days_held", 0),
            "Reason": t.get("exit_reason", ""),
            "Entry Date": t.get("entry_date", ""),
            "Exit Date": t.get("exit_date", ""),
        })
    st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
