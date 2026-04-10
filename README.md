# Trading Pipeline

A research and decision-support system for Indian equity markets, built around a multi-layer trading pipeline and Streamlit dashboard.

## What this repo does

This repository combines technical, sector, and fundamental analytics to identify high-probability ideas in the Indian market.
It includes:

- Market regime detection for overall market posture
- Sector relative strength scanning to find leadership sectors
- Stock leadership screening within target sectors
- Stage-based breakout and base analysis (Weinstein/Minervini style)
- Earnings acceleration and value enrichment
- Conviction scoring and candidate ranking
- Position sizing, risk management, and trade tracking
- Institutional flow checks for FII/DII/FPI
- Interactive Streamlit dashboard pages for analysis and monitoring

## Key entrypoints

- `app.py` — Streamlit dashboard entry point. Loads cached scan results, runs periodic scans, and renders the multi-page UI.
- `pipeline.py` — CLI orchestrator for full scan execution and printable results.
- `refresh_data.py` — Headless refresh script for rebuilding scan cache.
- `requirements.txt` — Python dependencies.

## Main modules and file roles

### Core pipeline

- `config.py`
  - Central configuration and strategy parameters.
  - Defines market regime thresholds, sector RS settings, screener and stage parameters, fundamental rules, profit-taking logic, and risk sizing.

- `data_fetcher.py`
  - Fetches historical price data from `yfinance`.
  - Loads the NSE Total Market universe from NSECSV and builds a sector map.
  - Caches downloaded data to reduce repeated downloads.

- `nse_data_fetcher.py`
  - NSE-specific scraper wrapper used for quarterly filings, shareholding data, and institutional flow extraction.
  - Uses cached NSE responses and custom request handling.

- `market_regime.py`
  - Computes the current market regime using Nifty price vs 200 DMA, 50/200 DMA crossover status, breadth above 50/200 and 200 DMAs, and net new highs/lows.
  - Maps signals to a 5-point regime score and a posture such as Aggressive/Normal/Cautious/Defensive/Cash.

- `sector_rs.py`
  - Computes sector-relative strength using Mansfield RS and multi-timeframe momentum.
  - Ranks sectors and selects target sectors for the next layer.

- `stock_screener.py`
  - Screens stocks within the target sectors for leadership.
  - Filters based on RS vs Nifty, RS vs sector, proximity to 52-week highs, average volume, and accumulation.

- `stage_filter.py`
  - Classifies stocks into Weinstein stages 1–4.
  - Detects Stage 2 breakout candidates and consolidation bases.

- `earnings_analysis.py`
  - Analyzes earnings acceleration, revenue and profit trends, and supports fundamental enrichment of candidates.

- `value_analysis.py`
  - Computes quality/value signals such as ROIC, free cash flow, and valuation metrics.

- `fundamental_veto.py`
  - Fetches basic company fundamentals and applies quality veto checks.
  - Computes position sizing, stop levels, and profit target recommendations.

- `conviction_scorer.py`
  - Combines technical, fundamental, and macro factors into a conviction score.
  - Ranks candidates to prioritize the strongest ideas.

- `rr_scanner.py`
  - Identifies asymmetric risk/reward setups and computes reward/risk metrics.

- `position_manager.py`
  - Manages open positions, trade history, pyramiding, and exit rules.

- `fii_gating.py` and `fpi_sector_flows.py`
  - Monitor institutional flows and apply gating logic when FII/DII flows are adverse.

- `breadth_analysis.py`
  - Generates market breadth metrics based on stage distributions.

- `dashboard_helpers.py`
  - Shared charting, HTML, and UI helper functions used by the Streamlit pages.

- `financial_utils.py`
  - Utility functions for parsing financial statements and computing YoY metrics.

### Streamlit pages

The `pages/` folder contains the app's UI sections:

- `1_Market_Regime.py` — Market regime pulse, breadth, macro dashboards.
- `2_Sector_Rotation.py` — Sector RS, sector ranking, and rotation insights.
- `3_Stock_Opportunities.py` — Stock screening and candidate lists.
- `4_Positions.py` — Current portfolio, position management, and holdings.
- `5_Stock_Deep_Dive.py` — Detailed stock-level analytics and charts.
- `6_Watchlist.py` — Final watchlist and monitored candidates.
- `7_Earnings_Season.py` — Earnings season tracking and sector-level earnings data.
- `8_FII_DII_Flows.py` — Institutional flow monitoring and heatmaps.

## How it works

The pipeline runs in layers:

1. Fetch market, sector, and stock data.
2. Determine the current market regime and macro liquidity posture.
3. Rank sectors by relative strength.
4. Screen stocks within the top sectors for leadership.
5. Apply Stage 2 / breakout filters.
6. Enrich candidates with earnings and value signals.
7. Compute conviction scores and apply fundamental veto logic.
8. Generate a final watchlist, position sizing, and risk/reward guidance.

## Usage

- `streamlit run app.py`
  - Launch the interactive dashboard.

- `python pipeline.py`
  - Run a full scan from the command line.

- `python refresh_data.py`
  - Refresh the cached scan results for scheduled or batch execution.

## Dependencies

Required packages are listed in `requirements.txt`, including:

- `streamlit`
- `yfinance`
- `pandas`
- `numpy`
- `plotly`
- `beautifulsoup4`
- `curl_cffi`

## Notes

- The repo is designed for Indian equity markets and uses NSE data plus Nifty sector indices.
- It supports both interactive visualization and batch refresh workflows.
- The `trading-pipeline-specsit` folder appears to be a duplicate or alternate copy of the same project.

---

This README summarizes the repo's purpose, mechanics, and module structure for easier onboarding and maintenance.