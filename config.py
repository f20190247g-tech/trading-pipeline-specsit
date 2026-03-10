"""
Trading Pipeline Configuration
All tunable parameters in one place.
"""

# ── Data Settings ──────────────────────────────────────────────
DATA_SOURCE = "yfinance"
LOOKBACK_DAYS = 0  # 0 = fetch max available history (yfinance period='max')
VOLUME_LOOKBACK = 50  # days for average volume

# ── Market Regime Thresholds ───────────────────────────────────
REGIME_CONFIG = {
    # Index vs 200 DMA
    "index_above_200dma_bullish": True,
    "index_near_200dma_pct": 2.0,  # within ±2% = neutral

    # Breadth: % of stocks above 50 DMA
    "breadth_50dma_bullish": 60,
    "breadth_50dma_bearish": 40,

    # Breadth: % of stocks above 200 DMA
    "breadth_200dma_bullish": 65,
    "breadth_200dma_bearish": 45,

    # Net new highs threshold
    "net_new_highs_bullish": 20,
    "net_new_highs_bearish": -10,
}

# Regime score to posture mapping
# Score ranges from -2 (full bear) to +2 (full bull)
REGIME_POSTURE = {
    2:  {"label": "Aggressive",  "max_capital_pct": 100, "risk_per_trade_pct": 1.5, "max_new_positions": 99},
    1:  {"label": "Normal",      "max_capital_pct": 80,  "risk_per_trade_pct": 1.0, "max_new_positions": 4},
    0:  {"label": "Cautious",    "max_capital_pct": 50,  "risk_per_trade_pct": 0.75, "max_new_positions": 2},
    -1: {"label": "Defensive",   "max_capital_pct": 20,  "risk_per_trade_pct": 0.5, "max_new_positions": 1},
    -2: {"label": "Cash",        "max_capital_pct": 10,  "risk_per_trade_pct": 0.0, "max_new_positions": 0},
}

# ── Sector RS Settings ─────────────────────────────────────────
SECTOR_CONFIG = {
    "rs_ma_period": 52 * 5,  # ~52 weeks in trading days for Mansfield RS zero line
    "momentum_periods": [5, 10, 21, 63, 126],  # 1w, 2w, 1m, 3m, 6m in trading days
    "top_sectors_count": 4,
    "min_rs_trend": 0,  # RS must be above zero-line (improving)
}

# ── NSE Sector Index Tickers (yfinance format) ─────────────────
NSE_SECTOR_INDICES = {
    "Nifty IT":           "^CNXIT",
    "Nifty Bank":         "^NSEBANK",
    "Nifty Pharma":       "^CNXPHARMA",
    "Nifty Auto":         "^CNXAUTO",
    "Nifty Metal":        "^CNXMETAL",
    "Nifty Realty":       "^CNXREALTY",
    "Nifty FMCG":         "^CNXFMCG",
    "Nifty Energy":       "^CNXENERGY",
    "Nifty Infra":        "^CNXINFRA",
    "Nifty PSU Bank":     "^CNXPSUBANK",
    "Nifty Media":        "^CNXMEDIA",
    "Nifty Pvt Bank":     "^CNXPRIVATEBANK",
    "Nifty Fin Service":  "^CNXFINANCE",
    "Nifty Consumption":  "^CNXCONSUMPTION",
    "Nifty Commodities":  "^CNXCOMMODITIES",
    "Nifty MNC":          "^CNXMNC",
    "Nifty Healthcare":   "^CNXHEALTHCARE",
}

NIFTY50_TICKER = "^NSEI"

# ── NSE Total Market Universe ────────────────────────────────
NSE_TM_CSV_URL = "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"

# ── Sector Index Constituent CSVs (for sectors poorly mapped by industry labels) ──
SECTOR_CONSTITUENT_URLS = {
    "Nifty Bank":        "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "Nifty Pharma":      "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "Nifty PSU Bank":    "https://archives.nseindia.com/content/indices/ind_niftypsubanklist.csv",
    "Nifty MNC":         "https://archives.nseindia.com/content/indices/ind_niftymnclist.csv",
    "Nifty Fin Service": "https://archives.nseindia.com/content/indices/ind_niftyfinancelist.csv",
    "Nifty Healthcare":  "https://archives.nseindia.com/content/indices/ind_niftyhealthcarelist.csv",
    "Nifty Consumption": "https://archives.nseindia.com/content/indices/ind_niftyconsumptionlist.csv",
    "Nifty Commodities": "https://archives.nseindia.com/content/indices/ind_niftycommoditieslist.csv",
}
UNIVERSE_CACHE_TTL_HOURS = 24  # re-download constituent CSV after this
MACRO_CACHE_TTL_HOURS = 4     # macro data refresh interval
NSE_DATA_CACHE_TTL_HOURS = 24  # NSE financials/shareholding cache TTL

# ── Chart Settings ─────────────────────────────────────────────
CHART_CONFIG = {
    "default_timeframe": "Weekly",  # Weekly | Daily | Monthly
}

# ── Macro Dashboard Tickers ──────────────────────────────────
MACRO_TICKERS = {
    "Nifty 50": "^NSEI",
    "S&P 500": "^GSPC",  "Nasdaq": "^IXIC",  "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",  "DAX": "^GDAXI",
    "Nikkei 225": "^N225",  "Hang Seng": "^HSI",  "Shanghai": "000001.SS",
    "VIX": "^VIX",  "India VIX": "^INDIAVIX",
    "Dollar Index": "DX-Y.NYB",  "USD/INR": "USDINR=X",
    "EUR/USD": "EURUSD=X",  "USD/JPY": "USDJPY=X",
    "US 10Y": "^TNX",  "US 5Y": "^FVX",  "US 30Y": "^TYX",
    "Crude Oil": "CL=F",  "Brent Crude": "BZ=F",
    "Gold": "GC=F",  "Silver": "SI=F",  "Copper": "HG=F",
}

MACRO_GROUPS = {
    "Global Indices": ["S&P 500", "Nasdaq", "Dow Jones", "FTSE 100", "DAX", "Nikkei 225", "Hang Seng", "Shanghai"],
    "Risk Gauges": ["VIX", "India VIX", "Dollar Index", "US 10Y", "US 5Y", "US 30Y"],
    "Currencies": ["USD/INR", "EUR/USD", "USD/JPY"],
    "Commodities": ["Crude Oil", "Brent Crude", "Gold", "Silver", "Copper"],
}

RISK_GAUGE_THRESHOLDS = {
    "VIX":          {"low": 15, "high": 25, "labels": ("CALM", "CAUTION", "FEAR")},
    "India VIX":    {"low": 14, "high": 22, "labels": ("CALM", "CAUTION", "FEAR")},
    "Dollar Index": {"low": 100, "high": 105, "labels": ("WEAK $", "NEUTRAL", "STRONG $")},
    "US 10Y":       {"low": 3.5, "high": 4.5, "labels": ("LOW", "NORMAL", "ELEVATED")},
    "Crude Oil":    {"low": 65, "high": 85, "labels": ("CHEAP", "NORMAL", "EXPENSIVE")},
    "Gold":         {"low": 1900, "high": 2100, "labels": ("RISK-ON", "NEUTRAL", "RISK-OFF")},
}

# ── Industry → Sector Index Mapping ──────────────────────────
# Maps NSE CSV "Industry" values to our sector indices (best-effort)
INDUSTRY_TO_SECTOR = {
    # IT
    "IT - Software": "Nifty IT",
    "IT - Services": "Nifty IT",
    "IT - Hardware": "Nifty IT",
    "Information Technology": "Nifty IT",
    # Banks
    "Banks": "Nifty Bank",
    "Banks - Private Sector": "Nifty Bank",
    "Banks - Public Sector": "Nifty PSU Bank",
    "Finance - Banks - Private Sector": "Nifty Bank",
    "Finance - Banks - Public Sector": "Nifty PSU Bank",
    # Pharma / Healthcare
    "Pharmaceuticals": "Nifty Pharma",
    "Pharmaceuticals & Biotechnology": "Nifty Pharma",
    "Healthcare": "Nifty Healthcare",
    "Healthcare Services": "Nifty Healthcare",
    # Auto
    "Automobiles": "Nifty Auto",
    "Auto Components": "Nifty Auto",
    "Automobile and Auto Components": "Nifty Auto",
    # Metal
    "Metals & Mining": "Nifty Metal",
    "Metals": "Nifty Metal",
    "Mining": "Nifty Metal",
    "Steel": "Nifty Metal",
    # Realty
    "Realty": "Nifty Realty",
    "Construction": "Nifty Realty",
    # FMCG
    "FMCG": "Nifty FMCG",
    "Fast Moving Consumer Goods": "Nifty FMCG",
    "Consumer Food": "Nifty FMCG",
    # Energy
    "Oil Gas & Consumable Fuels": "Nifty Energy",
    "Oil & Gas": "Nifty Energy",
    "Power": "Nifty Energy",
    "Energy": "Nifty Energy",
    # Infra
    "Construction Materials": "Nifty Infra",
    "Cement & Cement Products": "Nifty Infra",
    "Capital Goods": "Nifty Infra",
    "Industrial Manufacturing": "Nifty Infra",
    "Infrastructure": "Nifty Infra",
    # Financial Services
    "Financial Services": "Nifty Fin Service",
    "Finance": "Nifty Fin Service",
    "Finance - NBFC": "Nifty Fin Service",
    "Finance - Housing Finance": "Nifty Fin Service",
    "Insurance": "Nifty Fin Service",
    "Financial Technology (Fintech)": "Nifty Fin Service",
    # Consumption / Consumer
    "Consumer Durables": "Nifty Consumption",
    "Consumer Services": "Nifty Consumption",
    "Leisure Services": "Nifty Consumption",
    "Retailing": "Nifty Consumption",
    "Textiles": "Nifty Consumption",
    "Apparels & Accessories": "Nifty Consumption",
    # Media
    "Media": "Nifty Media",
    "Media Entertainment & Publication": "Nifty Media",
    # Commodities / Chemicals
    "Chemicals": "Nifty Commodities",
    "Fertilizers & Agrochemicals": "Nifty Commodities",
    # Telecom
    "Telecommunication": "Nifty Infra",
    "Telecom - Services": "Nifty Infra",
    # Misc
    "Diversified": "Nifty Infra",
    "Services": "Nifty Consumption",
    "Forest Materials": "Nifty Commodities",
    "Utilities": "Nifty Infra",
}

# ── Stock Screener Settings ────────────────────────────────────
SCREENER_CONFIG = {
    "rs_period": 126,  # 6-month RS lookback
    "min_rs_vs_nifty": 0,  # stock must outperform Nifty
    "min_rs_vs_sector": -0.05,  # slight underperformance OK if other signals strong
    "max_distance_from_52w_high_pct": 25,  # within 25% of 52-week high
    "min_avg_volume": 50000,  # minimum average daily volume (shares)
    "accumulation_days_lookback": 50,
    "min_accumulation_ratio": 1.1,  # up-day volume / down-day volume > 1.1
}

# ── Stage Analysis Settings ────────────────────────────────────
STAGE_CONFIG = {
    "ma_short": 50,
    "ma_mid": 150,
    "ma_long": 200,

    # Stage 2 criteria
    "price_above_150ma": True,
    "price_above_200ma": True,
    "ma150_above_ma200": True,
    "ma200_rising_days": 20,  # 200 MA must be rising for at least 20 days
    "price_above_52w_low_pct": 30,  # at least 30% above 52-week low
    "price_within_52w_high_pct": 25,  # within 25% of 52-week high

    # Breakout criteria
    "volume_surge_multiple": 1.5,  # breakout volume >= 1.5x 50-day avg
    "base_min_days": 20,  # minimum base length
    "base_max_days": 200,  # maximum base length
    "base_max_depth_pct": 35,  # base correction shouldn't exceed 35%
    "max_base_count": 3,  # prefer 1st-3rd base, skip 4th+

    # VCP (Volatility Contraction Pattern)
    "vcp_contractions_min": 2,
    "vcp_contraction_ratio": 0.6,  # each contraction should be < 60% of prior
}

# ── Fundamental Veto Thresholds ────────────────────────────────
FUNDAMENTAL_CONFIG = {
    "min_eps_growth_yoy_pct": 15,
    "min_revenue_growth_yoy_pct": 10,
    "min_roe_pct": 15,
    "max_debt_equity": 1.5,
    "max_peg_ratio": 2.0,
    "max_pe_vs_sector_avg_multiple": 2.0,  # PE shouldn't be > 2x sector avg
}

# ── Position Sizing ────────────────────────────────────────────
POSITION_CONFIG = {
    "total_capital": 1_000_000,  # ₹10L default, override at runtime
    "default_risk_per_trade_pct": 1.0,
    "max_portfolio_risk_pct": 6.0,  # total open risk shouldn't exceed 6%
    "max_single_position_pct": 15.0,  # no single stock > 15% of portfolio
}

# ── Trailing Stop Settings ─────────────────────────────────────
STOP_CONFIG = {
    "method": "atr",  # "atr" or "ma"
    "atr_period": 14,
    "atr_multiple": 2.5,  # trail at 2.5x ATR below high
    "ma_trail_period": 50,  # use 50 DMA as trailing stop
    "initial_stop_buffer_pct": 1.0,  # 1% below base low for initial stop
}

# ── Profit Taking Rules ───────────────────────────────────────
PROFIT_CONFIG = {
    "partial_sell_pct": 33,  # sell 1/3 at first target
    "first_target_gain_pct": 20,  # take partial at 20% gain
    "hold_min_weeks_if_fast": 8,  # if 20%+ in <3 weeks, hold 8 weeks
    "fast_gain_threshold_weeks": 3,
    "climax_volume_multiple": 3.0,  # 3x avg volume on biggest up-day = climax
}

# ── Conviction Scoring Weights (Tri-Factor) ──────────────────
CONVICTION_CONFIG = {
    # Tri-factor pillar weights (must sum to 100)
    "technical_weight": 50,         # Technical: sector + stage + RS + patterns
    "value_weight": 25,             # Value: ROIC + FCF + fortress + DCF + moat
    "macro_weight": 25,             # Macro: liquidity + earnings accel + FII
    # Legacy sub-weights (within technical pillar)
    "sector_rank_weight": 40,
    "stage2_score_weight": 20,
    "base_count_weight": 10,
    "rs_percentile_weight": 15,
    "accumulation_weight": 15,
}

# ── Value Analysis Config ─────────────────────────────────────
VALUE_ANALYSIS_CONFIG = {
    "roic_weight": 25,              # ROIC history (Buffett)
    "fcf_weight": 20,               # FCF yield & quality
    "fortress_weight": 20,          # Balance sheet strength
    "dcf_weight": 20,               # DCF margin of safety
    "moat_weight": 15,              # Competitive moat
    "wacc": 0.12,                   # Indian market WACC default
    "terminal_growth": 0.03,        # Terminal growth rate
    "min_roic_pct": 12,             # Minimum ROIC for quality
    "min_fcf_yield_pct": 3,         # Minimum FCF yield
}

# ── R:R Scanner Config ────────────────────────────────────────
RR_SCANNER_CONFIG = {
    "min_rr_ratio": 3.0,            # Minimum R:R to qualify
    "ideal_rr_ratio": 5.0,          # Sweet spot for Druckenmiller
    "max_risk_pct": 8.0,            # Max risk % per trade
    "target_r_multiples": [1.0, 2.0, 3.0, 5.0, 8.0],
}

# ── Smart Money Config ───────────────────────────────────────
SMART_MONEY_CONFIG = {
    "bulk_deal_lookback_days": 90,
    "block_deal_lookback_days": 90,
    "delivery_threshold_high": 50,   # green above this %
    "delivery_threshold_low": 30,    # red below this %
    "fii_dii_cache_ttl_hours": 1,
    "delivery_cache_ttl_hours": 4,
}

# ── Allocation & Pyramid Settings ─────────────────────────────
ALLOCATION_CONFIG = {
    "conviction_tiers": {
        "A+": {"target_pct": 35, "label": "Bet Big"},
        "A":  {"target_pct": 25, "label": "High Conviction"},
        "B":  {"target_pct": 15, "label": "Solid Setup"},
        "C":  {"target_pct": 10, "label": "Small / Speculative"},
    },
    "max_pyramid_adds": 3,
    "pyramid_sizes": [0.50, 0.30, 0.20],
    "pyramid_min_gain_pct": 2.0,
}

MACRO_DERIVATIVE_LABELS = ["VIX", "Dollar Index", "Crude Oil", "Gold", "US 10Y"]

FII_DII_CACHE_TTL_HOURS = 1

# ── Macro Liquidity Scoring ────────────────────────────────────
MACRO_LIQUIDITY_CONFIG = {
    "fii_flow_weight": 30,        # FII net flow trend (30%)
    "vix_trend_weight": 25,       # VIX trend direction (25%)
    "usdinr_trend_weight": 20,    # USD/INR weakening = positive (20%)
    "yield_curve_weight": 25,     # 10Y-5Y spread direction (25%)
    "fii_lookback_days": [5, 10, 21],  # lookback windows for flow trend
    "vix_overbought": 25,         # India VIX above this = fear
    "vix_oversold": 14,           # India VIX below this = calm
    "usdinr_strong_threshold": 1.0,  # % weakening = headwind
    "yield_curve_inversion_threshold": -0.1,  # negative spread = warning
}

# ── Weekly Stage Confirmation ──────────────────────────────────
WEEKLY_STAGE_CONFIG = {
    "weekly_ma_period": 30,       # 30-week MA (Weinstein's primary)
    "weekly_ma_rising_weeks": 4,  # must be rising for 4+ weeks
    "weekly_close_above_ma": True,
}

# ── Earnings Acceleration ──────────────────────────────────────
EARNINGS_ACCEL_CONFIG = {
    "min_quarters": 3,            # need at least 3 quarters of data
    "acceleration_bonus": 10,     # conviction bonus for accelerating earnings
    "deceleration_penalty": -5,   # conviction penalty for decelerating
}

# ── Consolidation Quality ──────────────────────────────────────
CONSOLIDATION_QUALITY_CONFIG = {
    "tight_range_threshold_pct": 15,
    "very_tight_threshold_pct": 10,
    "volume_dryup_threshold": 0.6,
    "min_quality_grade": "B",
}

# ── FII/DII Flow Gating ──────────────────────────────────────
FII_GATING_CONFIG = {
    "fii_5d_threshold": -2000,     # Cr; below this = red flag
    "fii_10d_threshold": -5000,    # Cr; sustained outflow = danger
    "dii_support_threshold": 1000,  # Cr; DII buying can offset FII
    "gate_action": "reduce_size",   # "reduce_size" or "block_new"
    "size_reduction_pct": 50,       # halve position size when gated
}

# ── Earnings Season ──────────────────────────────────────────
NIFTY100_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
NIFTY_MIDCAP150_CSV_URL = "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"

EARNINGS_SEASON_CONFIG = {
    "cache_ttl_hours": 12,
    "quarter_filing_lag_days": 30,
    "date_tolerance_days": 15,
    "high_growth_threshold_pct": 15,
}
