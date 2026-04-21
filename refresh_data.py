#!/usr/bin/env python3
"""
Headless data refresh script — run via cron for daily updates.

Usage:
    python refresh_data.py

Cron example (9 AM on weekdays):
    0 9 * * 1-5 cd /Users/amitgupta/trading-pipeline && /Users/amitgupta/.pyenv/versions/3.10.6/bin/python refresh_data.py >> scan_cache/refresh.log 2>&1
"""
import sys
import os
import pickle
import logging
import datetime as dt
from pathlib import Path

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

from config import POSITION_CONFIG, SECTOR_CONFIG
from data_fetcher import (
    fetch_index_data, fetch_all_stock_data, fetch_sector_data,
    fetch_price_data, fetch_macro_data, get_sector_map,
    get_all_stock_tickers, get_sector_for_stock,
)
from market_regime import compute_regime
from sector_rs import scan_sectors, get_top_sectors
from stock_screener import screen_stocks
from stage_filter import filter_stage2_candidates
from fundamental_veto import generate_final_watchlist
from dashboard_helpers import compute_quality_radar

CACHE_DIR = Path(__file__).parent / "scan_cache"
CACHE_FILE = CACHE_DIR / "last_scan.pkl"

CACHE_KEYS = [
    "scan_date", "capital", "nifty_df", "all_stock_data", "sector_data",
    "regime", "sector_rankings", "top_sectors", "stock_data",
    "screened_stocks", "stage2_candidates", "final_watchlist",
    "macro_data", "quality_radar", "universe_count",
    "earnings_season",
]


def run_refresh():
    """Run full pipeline and save results to disk cache."""
    capital = POSITION_CONFIG["total_capital"]
    top_n = SECTOR_CONFIG["top_sectors_count"]
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  PIPELINE REFRESH — {timestamp}")
    print(f"{'='*60}\n")

    results = {}

    # Step 0: Macro data
    print("[0/9] Fetching macro data...")
    macro_data = fetch_macro_data()
    results["macro_data"] = macro_data

    # Step 1: Nifty
    print("[1/9] Fetching Nifty 50 index data...")
    nifty_df = fetch_index_data()
    results["nifty_df"] = nifty_df

    # Step 2: All stocks
    all_tickers = get_all_stock_tickers()
    results["universe_count"] = len(all_tickers)
    print(f"[2/9] Fetching stock universe data ({len(all_tickers)} stocks)...")
    all_stock_data = fetch_all_stock_data()
    results["all_stock_data"] = all_stock_data
    print(f"       Loaded {len(all_stock_data)} stocks")

    # Step 3: Market regime
    print("[3/9] Computing market regime...")
    regime = compute_regime(nifty_df, all_stock_data)
    results["regime"] = regime
    print(f"       Regime: {regime['label']} (score {regime['regime_score']:+d})")

    # Step 4: Sectors
    print("[4/9] Fetching sector data & computing RS...")
    sector_data = fetch_sector_data()
    results["sector_data"] = sector_data
    sector_rankings = scan_sectors(sector_data, nifty_df)
    results["sector_rankings"] = sector_rankings
    top_sectors = get_top_sectors(sector_rankings, n=top_n)
    results["top_sectors"] = top_sectors
    print(f"       Top sectors: {', '.join(top_sectors)}")

    # Step 5: Screen
    print("[5/9] Screening stocks in top sectors...")
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
    results["stock_data"] = stock_data

    screened = screen_stocks(stock_data, nifty_df, sector_data, top_sectors)
    results["screened_stocks"] = screened
    print(f"       {len(screened)} stocks passed screening")

    # Step 6: Stage filter
    print("[6/9] Running stage analysis...")
    stage2 = filter_stage2_candidates(stock_data, screened) if screened else []
    results["stage2_candidates"] = stage2
    print(f"       {len(stage2)} Stage 2 candidates")

    # Step 7: Fundamental veto
    print("[7/9] Applying fundamental veto & sizing...")
    watchlist = generate_final_watchlist(stage2, regime, capital) if stage2 else []
    results["final_watchlist"] = watchlist

    buy_count = sum(1 for w in watchlist if w.get("action") == "BUY")
    watch_count = sum(1 for w in watchlist if w.get("action") in ("WATCH", "WATCHLIST"))
    print(f"       {buy_count} BUY signals, {watch_count} watchlist")

    # Step 8: Quality Radar
    print("[8/9] Computing Quality Radar...")
    quality_radar = compute_quality_radar(watchlist)
    results["quality_radar"] = quality_radar

    # Step 9: Earnings season
    print("[9/9] Running earnings season scan...")
    from earnings_season import run_earnings_scan
    from data_fetcher import load_universe as _load_universe_refresh
    try:
        universe_df = _load_universe_refresh()
        earnings = run_earnings_scan(universe_df)
        results["earnings_season"] = earnings
        print(f"       {earnings.get('reported_count', 0)}/{earnings.get('total_universe', 0)} reported for {earnings.get('quarter_label', '?')}")
    except Exception as e:
        print(f"       Earnings scan failed: {e}")

    results["scan_date"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    results["capital"] = capital

    # Save to disk
    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)

    size_mb = CACHE_FILE.stat().st_size / 1024 / 1024
    print(f"\n  Saved to {CACHE_FILE} ({size_mb:.1f} MB)")
    print(f"  Refresh complete at {dt.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    
        # Auto-push to Gist for HTML dashboard
    from gist_updater import build_snapshot_from_cache, push_snapshot
    push_snapshot(build_snapshot_from_cache())
    
    run_refresh()
