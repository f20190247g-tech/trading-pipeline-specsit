#!/usr/bin/env python3
"""
Trading Pipeline — Main Orchestrator
Chains all layers: Regime → Sector RS → Stock Screen → Stage Filter →
Earnings + Value Analysis → Conviction Scoring → Fundamental Veto → R:R Scan

Usage:
    python pipeline.py                          # full scan
    python pipeline.py --capital 2000000        # set capital to Rs 20L
    python pipeline.py --sectors-only           # just show sector rankings
    python pipeline.py --regime-only            # just show market regime
    python pipeline.py --stock RELIANCE.NS      # analyze a single stock
"""
import argparse
import datetime as dt
import logging
import sys
import time

from config import POSITION_CONFIG, LOOKBACK_DAYS
from data_fetcher import (
    fetch_index_data,
    fetch_sector_data,
    fetch_stock_data_for_sectors,
    fetch_all_stock_data,
    NIFTY500_SECTOR_MAP,
)
from market_regime import compute_regime, print_regime, compute_macro_liquidity_score
from sector_rs import scan_sectors, get_top_sectors, print_sector_rankings
from stock_screener import screen_stocks, print_screener_results
from stage_filter import (
    filter_stage2_candidates,
    analyze_stock_stage,
    print_stage_results,
)
from fundamental_veto import (
    generate_final_watchlist,
    print_final_watchlist,
    fetch_fundamentals,
    apply_fundamental_veto,
)
from conviction_scorer import rank_candidates_by_conviction

logger = logging.getLogger(__name__)


def _enrich_candidates(candidates: list[dict]) -> list[dict]:
    """Enrich Stage 2 candidates with earnings and value analysis.

    Adds 'earnings_analysis' and 'value_analysis' dicts to each candidate.
    Failures are non-fatal — candidates keep neutral scores.
    """
    try:
        from earnings_analysis import compute_earnings_acceleration
    except ImportError:
        compute_earnings_acceleration = None

    try:
        from value_analysis import compute_value_score
    except ImportError:
        compute_value_score = None

    for c in candidates:
        ticker = c.get("ticker", "")

        # Earnings acceleration
        if compute_earnings_acceleration:
            try:
                c["earnings_analysis"] = compute_earnings_acceleration(ticker)
            except Exception as e:
                logger.debug("Earnings analysis failed for %s: %s", ticker, e)
                c["earnings_analysis"] = {"data_available": False}
        else:
            c["earnings_analysis"] = {"data_available": False}

        # Value analysis
        if compute_value_score:
            try:
                c["value_analysis"] = compute_value_score(ticker)
            except Exception as e:
                logger.debug("Value analysis failed for %s: %s", ticker, e)
                c["value_analysis"] = {"data_available": False}
        else:
            c["value_analysis"] = {"data_available": False}

    return candidates


def _compute_fii_gate(fii_dii_data: dict | None) -> dict | None:
    """Compute FII gating if module available."""
    try:
        from fii_gating import check_fii_gate
        return check_fii_gate(fii_dii_data)
    except ImportError:
        return None


def _compute_breadth(all_stock_data: dict) -> dict | None:
    """Compute breadth by stage if module available."""
    try:
        from breadth_analysis import compute_breadth_by_stage
        from stage_filter import classify_stage

        stage_results = []
        for ticker, df in all_stock_data.items():
            if len(df) < 230:
                continue
            stage_info = classify_stage(df)
            stage_results.append({"ticker": ticker, "stage": stage_info})

        if stage_results:
            return compute_breadth_by_stage(stage_results)
    except ImportError:
        pass
    return None


def _scan_rr(candidates: list[dict]) -> list[dict]:
    """Apply R:R scanner to candidates."""
    try:
        from rr_scanner import scan_asymmetric_setups
        return scan_asymmetric_setups(candidates)
    except ImportError:
        return candidates


def run_full_pipeline(capital: float = None, days: int = None):
    """Run the complete multi-layer pipeline."""
    if capital is None:
        capital = POSITION_CONFIG["total_capital"]
    if days is None:
        days = LOOKBACK_DAYS

    print("\n" + "#" * 65)
    print("  TRADING PIPELINE — FULL SCAN (Tri-Factor)")
    print(f"  Date: {dt.date.today()}")
    print(f"  Capital: Rs {capital:,.0f}")
    print("#" * 65)

    start_time = time.time()

    # ── STEP 1: Fetch Nifty 50 Index Data ───────────────────────
    print("\n[1/9] Fetching Nifty 50 index data...")
    nifty_df = fetch_index_data(days=days)
    print(f"  Nifty 50: {len(nifty_df)} trading days loaded.")

    # ── STEP 1b: Fetch Macro Data for Liquidity Scoring ────────
    print("  Fetching macro data for liquidity scoring...")
    from data_fetcher import fetch_macro_data
    macro_data = fetch_macro_data()

    # Try to get FII/DII flows
    fii_dii_data = None
    try:
        from nse_data_fetcher import get_nse_fetcher
        nse = get_nse_fetcher()
        fii_dii_data = nse.compute_fii_dii_flows()
    except Exception:
        pass

    macro_liquidity = compute_macro_liquidity_score(macro_data, fii_dii_data)
    print(f"  Macro Liquidity: {macro_liquidity['label']} (score {macro_liquidity['score']:.0f})")

    # FII gating
    fii_gate = _compute_fii_gate(fii_dii_data)
    if fii_gate and fii_gate.get("gated"):
        print(f"  FII Gate: {fii_gate['gate_level'].upper()} — {fii_gate['reason']}")

    # ── STEP 2: Fetch All Stock Data (for breadth) ──────────────
    print("\n[2/9] Fetching stock universe data...")
    all_stock_data = fetch_all_stock_data(days=days)

    # ── STEP 2b: Market Breadth by Stage ──────────────────────
    print("  Computing breadth by stage...")
    breadth = _compute_breadth(all_stock_data)
    if breadth:
        print(f"  Breadth: {breadth['breadth_label']} (score {breadth['breadth_score']:.0f}) | "
              f"S2: {breadth['stage_pcts'].get(2, 0):.0f}% | S4: {breadth['stage_pcts'].get(4, 0):.0f}%")

    # ── STEP 3: Market Regime ───────────────────────────────────
    print("\n[3/9] Computing market regime...")
    regime = compute_regime(nifty_df, all_stock_data, macro_liquidity=macro_liquidity)
    print_regime(regime)

    if regime["regime_score"] == -2:
        print("\n  REGIME IS CASH — No new positions recommended.")
        print("  Pipeline complete. Run again when conditions improve.\n")
        return

    # ── STEP 4: Sector Relative Strength ────────────────────────
    print("[4/9] Scanning sector relative strength...")
    sector_data = fetch_sector_data(days=days)
    sector_rankings = scan_sectors(sector_data, nifty_df)
    print_sector_rankings(sector_rankings)
    top_sectors = get_top_sectors(sector_rankings)
    print(f"  Target sectors: {', '.join(top_sectors)}")

    # ── STEP 5: Stock Screening ─────────────────────────────────
    print(f"\n[5/9] Screening stocks in target sectors...")
    stock_data = {}
    needed_tickers = set()
    for sector in top_sectors:
        for t in NIFTY500_SECTOR_MAP.get(sector, []):
            if t in all_stock_data:
                stock_data[t] = all_stock_data[t]
            else:
                needed_tickers.add(t)

    if needed_tickers:
        print(f"  Fetching {len(needed_tickers)} additional tickers...")
        from data_fetcher import fetch_price_data
        extra = fetch_price_data(list(needed_tickers), days=days)
        stock_data.update(extra)

    screened = screen_stocks(stock_data, nifty_df, sector_data, top_sectors)
    print_screener_results(screened)

    if not screened:
        print("  No stocks pass screening filters. Pipeline complete.\n")
        return

    # ── STEP 6: Stage 2 Filter ──────────────────────────────────
    print("[6/9] Applying Stage 2 + breakout filter...")
    stage2_candidates = filter_stage2_candidates(stock_data, screened)
    print_stage_results(stage2_candidates)

    if not stage2_candidates:
        print("  No Stage 2 candidates found. Check watchlist for developing setups.\n")
        return

    # ── STEP 7: Earnings + Value Enrichment ─────────────────────
    print("[7/9] Enriching candidates (earnings acceleration + value analysis)...")
    stage2_candidates = _enrich_candidates(stage2_candidates)
    enriched_count = sum(1 for c in stage2_candidates
                         if c.get("earnings_analysis", {}).get("data_available")
                         or c.get("value_analysis", {}).get("data_available"))
    print(f"  Enriched {enriched_count}/{len(stage2_candidates)} candidates with fundamental data.")

    # ── STEP 8: Tri-Factor Conviction Scoring ───────────────────
    print("[8/9] Computing tri-factor conviction scores...")
    ranked = rank_candidates_by_conviction(
        stage2_candidates,
        sector_rankings,
        macro_liquidity=macro_liquidity,
        fii_gate=fii_gate,
    )

    # Print conviction breakdown for top candidates
    for c in ranked[:5]:
        pillars = c.get("conviction_pillars", {})
        ticker = c.get("ticker", "")
        score = c.get("conviction_score", 0)
        trend = c.get("earnings_analysis", {}).get("trend", "n/a")
        print(f"  {ticker:<14} Conv: {score:5.1f} | "
              f"Tech: {pillars.get('technical', 0):.0f} "
              f"Val: {pillars.get('value', 0):.0f} "
              f"Macro: {pillars.get('macro', 0):.0f} | "
              f"Earnings: {trend}")

    # ── STEP 8b: R:R Scanner ──────────────────────────────────
    rr_candidates = _scan_rr(ranked)
    rr_count = sum(1 for c in rr_candidates if c.get("rr_ratio", 0) >= 3)
    if rr_count:
        print(f"  {rr_count} candidates with R:R >= 3:1")

    # ── STEP 9: Fundamental Veto + Final Output ─────────────────
    print("[9/9] Applying fundamental veto and computing position sizes...")

    # Apply FII gate size reduction if active
    effective_capital = capital
    if fii_gate and fii_gate.get("gated"):
        mult = fii_gate.get("size_multiplier", 1.0)
        effective_capital = capital * mult
        print(f"  FII gate active — effective capital reduced to Rs {effective_capital:,.0f} ({mult:.0%})")

    watchlist = generate_final_watchlist(rr_candidates, regime, effective_capital)
    print_final_watchlist(watchlist, regime)

    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")
    print(f"  {len([w for w in watchlist if w['action'] == 'BUY'])} buy signals | "
          f"{len([w for w in watchlist if w['action'] in ('WATCH', 'WATCHLIST')])} on watchlist | "
          f"{len([w for w in watchlist if w['action'] == 'VETOED'])} vetoed")

    # Summary of new analysis layers
    if breadth:
        print(f"  Market breadth: {breadth['breadth_label']} | ", end="")
    if macro_liquidity:
        print(f"Liquidity: {macro_liquidity['label']} | ", end="")
    if fii_gate:
        print(f"FII: {fii_gate['gate_level']}", end="")
    print()

    return watchlist


def run_regime_only(days: int = None):
    """Just show market regime."""
    if days is None:
        days = LOOKBACK_DAYS
    print("\n  Fetching data for regime analysis...")
    nifty_df = fetch_index_data(days=days)
    all_stock_data = fetch_all_stock_data(days=days)
    regime = compute_regime(nifty_df, all_stock_data)
    print_regime(regime)
    return regime


def run_sectors_only(days: int = None):
    """Just show sector rankings."""
    if days is None:
        days = LOOKBACK_DAYS
    print("\n  Fetching data for sector analysis...")
    nifty_df = fetch_index_data(days=days)
    sector_data = fetch_sector_data(days=days)
    rankings = scan_sectors(sector_data, nifty_df)
    print_sector_rankings(rankings)
    return rankings


def analyze_single_stock(ticker: str, days: int = None):
    """Deep-dive analysis on a single stock."""
    if days is None:
        days = LOOKBACK_DAYS

    print(f"\n  Analyzing {ticker}...")
    from data_fetcher import fetch_price_data
    data = fetch_price_data([ticker], days=days)
    if ticker not in data:
        print(f"  Error: Could not fetch data for {ticker}")
        return

    df = data[ticker]
    nifty_df = fetch_index_data(days=days)

    # Stage analysis
    stage = analyze_stock_stage(df, ticker)
    print(f"\n  Stage: {stage['stage']['detail']}")
    print(f"  S2 checks: {stage['stage'].get('s2_checks', {})}")
    print(f"  Bases found: {stage['bases_found']}, in Stage 2: {stage['base_count_in_stage2']}")

    # Weekly confirmation
    weekly = stage.get("weekly", {})
    if weekly:
        print(f"  Weekly: {weekly.get('weekly_detail', 'N/A')}")

    # Stage transitions
    transitions = stage.get("transitions", [])
    if transitions:
        t = transitions[0]
        print(f"  Transition: {t['transition']} ({t['signal']}) ~{t['days_ago']} days ago")

    if stage["breakout"]:
        bo = stage["breakout"]
        print(f"\n  BREAKOUT DETECTED!")
        print(f"    Date: {bo['breakout_date']}")
        print(f"    Price: {bo['breakout_price']}")
        print(f"    Volume: {bo['volume_ratio']}x average")

        if stage["entry_setup"]:
            es = stage["entry_setup"]
            print(f"    Entry: {es['entry_price']} | Stop: {es['effective_stop']} | Risk: {es['risk_pct']}%")

    if stage["vcp"]:
        vcp = stage["vcp"]
        print(f"\n  VCP: {'Yes' if vcp['is_vcp'] else 'No'} ({vcp['contractions']} contractions)")

    # Consolidation quality
    consolidation = stage.get("consolidation")
    if consolidation:
        print(f"  Consolidation: {consolidation.get('detail', 'N/A')}")

    # Earnings acceleration
    try:
        from earnings_analysis import compute_earnings_acceleration
        ea = compute_earnings_acceleration(ticker)
        if ea.get("data_available"):
            print(f"\n  Earnings: {ea.get('trend', 'N/A')} (score {ea.get('combined_score', 0):.0f})")
            if ea.get("eps_yoy_growth"):
                print(f"    EPS YoY: {ea['eps_yoy_growth'][:3]}")
    except Exception:
        pass

    # Value analysis
    try:
        from value_analysis import compute_value_score
        va = compute_value_score(ticker)
        if va.get("data_available"):
            print(f"\n  Value Score: {va.get('composite_score', 0):.0f}/100 (grade {va.get('grade', 'N/A')})")
            if va.get("roic_latest"):
                print(f"    ROIC: {va['roic_latest']:.1f}% | FCF Yield: {va.get('fcf_yield', 0):.1f}%")
    except Exception:
        pass

    # Fundamentals (legacy veto)
    print(f"\n  Fundamentals:")
    fund = fetch_fundamentals(ticker)
    veto = apply_fundamental_veto(fund)
    print(f"    Passes veto: {veto['passes']} ({veto['confidence']})")
    for r in veto["reasons"]:
        print(f"    - {r}")

    for key in ["pe_ratio", "peg_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth"]:
        val = fund.get(key)
        if val is not None:
            if "growth" in key or key == "roe":
                print(f"    {key}: {val*100:.1f}%")
            else:
                print(f"    {key}: {val:.2f}")

    # RS vs Nifty
    from stock_screener import compute_stock_rs
    rs = compute_stock_rs(df["Close"], nifty_df["Close"])
    print(f"\n  RS vs Nifty (6m): {rs:.2f}%")

    # R:R targets
    if stage.get("entry_setup"):
        try:
            from rr_scanner import compute_multi_level_targets
            targets = compute_multi_level_targets(
                stage["entry_setup"]["entry_price"],
                stage["entry_setup"]["effective_stop"],
            )
            if targets:
                print(f"\n  R:R Targets:")
                for t in targets:
                    print(f"    {t['r_multiple']}R: Rs {t['price']:.2f} (+{t['gain_pct']}%) — {t['action']}")
        except Exception:
            pass

    print()
    return stage


def main():
    parser = argparse.ArgumentParser(description="Trading Pipeline Scanner")
    parser.add_argument("--capital", type=float, default=None, help="Total capital in Rs")
    parser.add_argument("--days", type=int, default=None, help="Lookback days for historical data")
    parser.add_argument("--regime-only", action="store_true", help="Only show market regime")
    parser.add_argument("--sectors-only", action="store_true", help="Only show sector rankings")
    parser.add_argument("--stock", type=str, help="Analyze a single stock (e.g., RELIANCE.NS)")

    args = parser.parse_args()

    if args.stock:
        analyze_single_stock(args.stock, days=args.days)
    elif args.regime_only:
        run_regime_only(days=args.days)
    elif args.sectors_only:
        run_sectors_only(days=args.days)
    else:
        run_full_pipeline(capital=args.capital, days=args.days)


if __name__ == "__main__":
    main()
