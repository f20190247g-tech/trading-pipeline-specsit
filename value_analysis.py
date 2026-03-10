"""
Value Analysis Engine — ROIC history, FCF calculation, DCF valuation,
Balance Sheet Fortress scoring, and Moat Strength analysis.

Buffett's lens: durable competitive advantages priced with margin of safety.
"""
import logging
import pandas as pd
import numpy as np

from financial_utils import (
    get_annual_financials,
    get_quarterly_financials,
    extract_row,
    compute_ttm,
    safe_divide,
)
from scoring_utils import normalize_score, weighted_composite, grade_score

logger = logging.getLogger(__name__)

# ── Row name variants for yfinance financial statements ───────
REVENUE_ROWS = ["Total Revenue", "Operating Revenue", "TotalRevenue", "Revenue"]
NET_INCOME_ROWS = ["Net Income", "NetIncome", "Net Income Common Stockholders",
                   "Net Income From Continuing Operations"]
EBIT_ROWS = ["EBIT", "Operating Income", "OperatingIncome"]
TOTAL_ASSETS_ROWS = ["Total Assets", "TotalAssets"]
CURRENT_LIAB_ROWS = ["Current Liabilities", "CurrentLiabilities",
                     "Total Current Liabilities"]
TOTAL_DEBT_ROWS = ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"]
CASH_ROWS = ["Cash And Cash Equivalents", "CashAndCashEquivalents",
             "Cash Cash Equivalents And Short Term Investments"]
CURRENT_ASSETS_ROWS = ["Current Assets", "CurrentAssets", "Total Current Assets"]
INTEREST_EXPENSE_ROWS = ["Interest Expense", "InterestExpense",
                         "Interest Expense Non Operating"]
STOCKHOLDER_EQUITY_ROWS = ["Stockholders Equity", "StockholdersEquity",
                           "Total Stockholder Equity", "Common Stock Equity"]
OCF_ROWS = ["Operating Cash Flow", "OperatingCashFlow",
            "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities"]
CAPEX_ROWS = ["Capital Expenditure", "CapitalExpenditure",
              "Purchase Of PPE", "Purchases of Property Plant and Equipment"]
SHARES_ROWS = ["Ordinary Shares Number", "Share Issued", "BasicAverageShares"]
DEPRECIATION_ROWS = ["Depreciation And Amortization", "DepreciationAndAmortization",
                     "Depreciation", "Reconciled Depreciation"]
TAX_ROWS = ["Tax Provision", "TaxProvision", "Income Tax Expense"]


# ═══════════════════════════════════════════════════════════════
# ROIC / ROCE History (Phase 2.1)
# ═══════════════════════════════════════════════════════════════

def compute_roic_history(ticker: str) -> dict:
    """Compute historical ROIC/ROCE from annual financial statements.

    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT * (1 - tax_rate)
    Invested Capital = Total Assets - Current Liabilities

    Returns:
        {
            "ticker": str,
            "data_available": bool,
            "roic_history": [{year, roic_pct}, ...],
            "roce_history": [{year, roce_pct}, ...],
            "avg_roic_3y": float,
            "avg_roic_5y": float,
            "roic_trend": "improving" | "stable" | "declining",
            "roic_consistency": int (years ROIC > 15%),
            "roic_score": 0-100,
        }
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        fin = get_annual_financials(ticker)
        if not fin["available"]:
            return result

        income = fin["income"]
        balance = fin["balance"]

        ebit = extract_row(income, EBIT_ROWS)
        tax = extract_row(income, TAX_ROWS)
        net_income = extract_row(income, NET_INCOME_ROWS)
        total_assets = extract_row(balance, TOTAL_ASSETS_ROWS)
        current_liab = extract_row(balance, CURRENT_LIAB_ROWS)
        equity = extract_row(balance, STOCKHOLDER_EQUITY_ROWS)

        if ebit is None or total_assets is None:
            # Fallback: use ROE = NI / Equity
            if net_income is not None and equity is not None:
                return _compute_roe_fallback(ticker, net_income, equity)
            return result

        result["data_available"] = True

        # Compute ROIC for each year
        roic_history = []
        roce_history = []
        years = min(len(ebit), len(total_assets))

        for i in range(years):
            year = str(ebit.index[i].year) if hasattr(ebit.index[i], "year") else str(ebit.index[i])

            ebit_val = float(ebit.iloc[i])
            assets_val = float(total_assets.iloc[i]) if i < len(total_assets) else None
            cl_val = float(current_liab.iloc[i]) if current_liab is not None and i < len(current_liab) else 0

            # Tax rate estimation
            tax_val = float(tax.iloc[i]) if tax is not None and i < len(tax) else 0
            ni_val = float(net_income.iloc[i]) if net_income is not None and i < len(net_income) else None

            if ni_val and ni_val > 0 and tax_val:
                tax_rate = tax_val / (ni_val + tax_val)
            else:
                tax_rate = 0.25  # default 25%

            nopat = ebit_val * (1 - tax_rate)

            # Invested Capital = Total Assets - Current Liabilities
            if assets_val and assets_val > 0:
                invested_capital = assets_val - cl_val
                roic_pct = safe_divide(nopat, invested_capital)
                if roic_pct is not None:
                    roic_history.append({"year": year, "roic_pct": round(roic_pct * 100, 1)})

            # ROCE = EBIT / Capital Employed (Assets - Current Liabilities)
            equity_val = float(equity.iloc[i]) if equity is not None and i < len(equity) else None
            if equity_val and equity_val > 0:
                roce_pct = safe_divide(ebit_val, equity_val)
                if roce_pct is not None:
                    roce_history.append({"year": year, "roce_pct": round(roce_pct * 100, 1)})

        result["roic_history"] = roic_history
        result["roce_history"] = roce_history

        # Averages
        roic_vals = [r["roic_pct"] for r in roic_history]
        if roic_vals:
            result["avg_roic_3y"] = round(np.mean(roic_vals[:3]), 1) if len(roic_vals) >= 3 else round(np.mean(roic_vals), 1)
            result["avg_roic_5y"] = round(np.mean(roic_vals[:5]), 1) if len(roic_vals) >= 5 else None

            # Trend
            if len(roic_vals) >= 3:
                recent = np.mean(roic_vals[:2])
                older = np.mean(roic_vals[2:4]) if len(roic_vals) >= 4 else roic_vals[-1]
                if recent > older + 2:
                    result["roic_trend"] = "improving"
                elif recent < older - 2:
                    result["roic_trend"] = "declining"
                else:
                    result["roic_trend"] = "stable"
            else:
                result["roic_trend"] = "stable"

            # Consistency (years > 15%)
            result["roic_consistency"] = sum(1 for r in roic_vals if r > 15)

            # Score
            avg = result["avg_roic_3y"]
            consistency = result["roic_consistency"]
            base_score = normalize_score(avg, 5, 30)
            consistency_bonus = min(consistency * 5, 20)
            trend_adj = {"improving": 5, "stable": 0, "declining": -5}.get(result["roic_trend"], 0)
            result["roic_score"] = round(min(100, base_score + consistency_bonus + trend_adj), 1)
        else:
            result["roic_score"] = 0

        return result

    except Exception as e:
        logger.warning("ROIC history failed for %s: %s", ticker, e)
        return result


def _compute_roe_fallback(ticker: str, net_income, equity) -> dict:
    """Fallback: compute ROE when ROIC data isn't available."""
    result = {"ticker": ticker, "data_available": True, "roic_history": [], "roce_history": []}
    roe_history = []

    years = min(len(net_income), len(equity))
    for i in range(years):
        year = str(net_income.index[i].year) if hasattr(net_income.index[i], "year") else str(net_income.index[i])
        roe = safe_divide(float(net_income.iloc[i]), float(equity.iloc[i]))
        if roe is not None:
            roe_history.append({"year": year, "roic_pct": round(roe * 100, 1)})

    result["roic_history"] = roe_history
    roic_vals = [r["roic_pct"] for r in roe_history]

    if roic_vals:
        result["avg_roic_3y"] = round(np.mean(roic_vals[:3]), 1) if len(roic_vals) >= 3 else round(np.mean(roic_vals), 1)
        result["avg_roic_5y"] = round(np.mean(roic_vals[:5]), 1) if len(roic_vals) >= 5 else None
        result["roic_trend"] = "stable"
        result["roic_consistency"] = sum(1 for r in roic_vals if r > 15)
        result["roic_score"] = round(normalize_score(result["avg_roic_3y"], 5, 30), 1)
    else:
        result["roic_score"] = 0

    return result


# ═══════════════════════════════════════════════════════════════
# FCF Calculator (Phase 2.2)
# ═══════════════════════════════════════════════════════════════

def compute_fcf(ticker: str) -> dict:
    """Compute Free Cash Flow from annual cash flow statement.

    FCF = Operating Cash Flow - Capital Expenditure

    Returns:
        {
            "ticker": str,
            "data_available": bool,
            "fcf_history": [{year, fcf, fcf_margin_pct}, ...],
            "fcf_ttm": float,
            "fcf_yield_pct": float,
            "fcf_trend": "growing" | "stable" | "declining",
            "cash_conversion": float (FCF/NI ratio),
            "fcf_score": 0-100,
        }
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        fin = get_annual_financials(ticker)
        if not fin["available"]:
            return result

        cashflow = fin["cashflow"]
        income = fin["income"]

        ocf = extract_row(cashflow, OCF_ROWS)
        capex = extract_row(cashflow, CAPEX_ROWS)
        revenue = extract_row(income, REVENUE_ROWS)
        net_income = extract_row(income, NET_INCOME_ROWS)

        if ocf is None:
            return result

        result["data_available"] = True

        # Compute FCF history
        fcf_history = []
        years = len(ocf)

        for i in range(years):
            year = str(ocf.index[i].year) if hasattr(ocf.index[i], "year") else str(ocf.index[i])
            ocf_val = float(ocf.iloc[i])
            capex_val = abs(float(capex.iloc[i])) if capex is not None and i < len(capex) else 0
            fcf_val = ocf_val - capex_val

            rev_val = float(revenue.iloc[i]) if revenue is not None and i < len(revenue) else None
            fcf_margin = safe_divide(fcf_val, rev_val)

            fcf_history.append({
                "year": year,
                "fcf": round(fcf_val, 0),
                "fcf_margin_pct": round(fcf_margin * 100, 1) if fcf_margin is not None else None,
            })

        result["fcf_history"] = fcf_history

        # TTM FCF (latest annual)
        fcf_values = [h["fcf"] for h in fcf_history]
        if fcf_values:
            result["fcf_ttm"] = fcf_values[0]

        # FCF Yield (need market cap from yfinance)
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap", 0)
            if mcap and mcap > 0 and fcf_values:
                result["fcf_yield_pct"] = round(fcf_values[0] / mcap * 100, 2)
        except Exception:
            pass

        # Cash conversion (FCF / Net Income)
        if net_income is not None and len(net_income) > 0:
            ni_val = float(net_income.iloc[0])
            if ni_val > 0 and fcf_values:
                result["cash_conversion"] = round(fcf_values[0] / ni_val, 2)

        # Trend
        if len(fcf_values) >= 3:
            recent = np.mean(fcf_values[:2])
            older = np.mean(fcf_values[2:4]) if len(fcf_values) >= 4 else fcf_values[-1]
            if recent > older * 1.1:
                result["fcf_trend"] = "growing"
            elif recent < older * 0.9:
                result["fcf_trend"] = "declining"
            else:
                result["fcf_trend"] = "stable"
        else:
            result["fcf_trend"] = "stable"

        # Score
        fcf_yield = result.get("fcf_yield_pct", 0)
        cash_conv = result.get("cash_conversion", 0)
        trend = result.get("fcf_trend", "stable")

        yield_score = normalize_score(fcf_yield, 0, 10)
        conv_score = normalize_score(cash_conv, 0, 1.5)
        trend_adj = {"growing": 10, "stable": 0, "declining": -10}.get(trend, 0)

        result["fcf_score"] = round(min(100, yield_score * 0.5 + conv_score * 0.5 + trend_adj), 1)

        return result

    except Exception as e:
        logger.warning("FCF computation failed for %s: %s", ticker, e)
        return result


# ═══════════════════════════════════════════════════════════════
# Balance Sheet Fortress Score (Phase 2.3)
# ═══════════════════════════════════════════════════════════════

def compute_fortress_score(ticker: str) -> dict:
    """Score balance sheet strength across multiple dimensions.

    Components (each 0-25, total 0-100):
    1. Current Ratio (liquidity)
    2. Debt/Equity (leverage)
    3. Interest Coverage (debt service)
    4. Cash/Debt ratio (safety buffer)

    Returns dict with fortress_score, grade, and component breakdown.
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        fin = get_annual_financials(ticker)
        if not fin["available"]:
            return result

        balance = fin["balance"]
        income = fin["income"]

        current_assets = extract_row(balance, CURRENT_ASSETS_ROWS)
        current_liab = extract_row(balance, CURRENT_LIAB_ROWS)
        total_debt = extract_row(balance, TOTAL_DEBT_ROWS)
        cash = extract_row(balance, CASH_ROWS)
        equity = extract_row(balance, STOCKHOLDER_EQUITY_ROWS)
        ebit = extract_row(income, EBIT_ROWS)
        interest = extract_row(income, INTEREST_EXPENSE_ROWS)

        result["data_available"] = True
        components = {}

        # 1. Current Ratio (0-25)
        cr = None
        if current_assets is not None and current_liab is not None:
            ca_val = float(current_assets.iloc[0])
            cl_val = float(current_liab.iloc[0])
            cr = safe_divide(ca_val, cl_val)
        if cr is not None:
            cr_score = normalize_score(cr, 0.5, 2.5) / 4  # scale to 0-25
            components["current_ratio"] = {"value": round(cr, 2), "score": round(cr_score, 1)}
            result["current_ratio"] = round(cr, 2)
        else:
            components["current_ratio"] = {"value": None, "score": 12.5}

        # 2. Debt/Equity (0-25, lower is better)
        de = None
        if total_debt is not None and equity is not None:
            debt_val = float(total_debt.iloc[0])
            eq_val = float(equity.iloc[0])
            de = safe_divide(debt_val, eq_val)
        if de is not None:
            de_score = normalize_score(de, 3.0, 0.0) / 4  # inverted, scale to 0-25
            components["debt_equity"] = {"value": round(de, 2), "score": round(de_score, 1)}
            result["debt_equity"] = round(de, 2)
        else:
            components["debt_equity"] = {"value": None, "score": 12.5}

        # 3. Interest Coverage (0-25)
        ic = None
        if ebit is not None and interest is not None:
            ebit_val = float(ebit.iloc[0])
            int_val = abs(float(interest.iloc[0]))
            ic = safe_divide(ebit_val, int_val)
        if ic is not None:
            ic_score = normalize_score(ic, 1, 15) / 4  # scale to 0-25
            components["interest_coverage"] = {"value": round(ic, 1), "score": round(ic_score, 1)}
            result["interest_coverage"] = round(ic, 1)
        else:
            # No interest expense could mean zero debt (good)
            components["interest_coverage"] = {"value": None, "score": 20}

        # 4. Cash/Debt ratio (0-25)
        cd = None
        if cash is not None and total_debt is not None:
            cash_val = float(cash.iloc[0])
            debt_val = float(total_debt.iloc[0])
            cd = safe_divide(cash_val, debt_val) if debt_val > 0 else 10.0  # no debt = max
        if cd is not None:
            cd_score = normalize_score(cd, 0, 1.5) / 4  # scale to 0-25
            components["cash_debt"] = {"value": round(cd, 2), "score": round(cd_score, 1)}
        else:
            components["cash_debt"] = {"value": None, "score": 12.5}

        # Composite fortress score
        total = sum(c["score"] for c in components.values())
        result["fortress_score"] = round(total, 1)
        result["fortress_grade"] = grade_score(total)
        result["components"] = components

        # Red flags
        red_flags = []
        if cr is not None and cr < 1.0:
            red_flags.append(f"Current ratio {cr:.1f} < 1.0 (liquidity risk)")
        if de is not None and de > 2.0:
            red_flags.append(f"D/E {de:.1f} > 2.0 (high leverage)")
        if ic is not None and ic < 3.0:
            red_flags.append(f"Interest coverage {ic:.1f}x < 3x (debt service risk)")
        result["red_flags"] = red_flags

        return result

    except Exception as e:
        logger.warning("Fortress score failed for %s: %s", ticker, e)
        return result


# ═══════════════════════════════════════════════════════════════
# Simple DCF + Margin of Safety (Phase 2.4)
# ═══════════════════════════════════════════════════════════════

def compute_simple_dcf(
    ticker: str,
    wacc: float = 0.12,
    terminal_growth: float = 0.03,
    high_growth_years: int = 5,
) -> dict:
    """Compute simplified DCF valuation + margin of safety.

    Uses latest FCF and estimated growth rate.
    Conservative defaults: 12% WACC, 3% terminal growth (India GDP+inflation proxy).

    Returns:
        {
            "ticker": str,
            "data_available": bool,
            "fcf_base": float,
            "growth_rate_used": float,
            "intrinsic_value": float,
            "current_price": float,
            "margin_of_safety_pct": float,
            "upside_pct": float,
            "verdict": "Undervalued" | "Fair" | "Overvalued",
            "dcf_score": 0-100,
        }
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        # Get FCF
        fcf_data = compute_fcf(ticker)
        if not fcf_data.get("data_available") or not fcf_data.get("fcf_ttm"):
            return result

        fcf_base = fcf_data["fcf_ttm"]
        if fcf_base <= 0:
            result["data_available"] = True
            result["verdict"] = "Negative FCF"
            result["dcf_score"] = 10
            return result

        # Get current price and shares
        import yfinance as yf
        info = yf.Ticker(ticker).info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        shares = info.get("sharesOutstanding", 0)

        if not current_price or not shares:
            return result

        result["data_available"] = True

        # Estimate growth rate from revenue/earnings growth
        revenue_growth = info.get("revenueGrowth", 0.10)
        earnings_growth = info.get("earningsGrowth", 0.10)
        if revenue_growth and earnings_growth:
            estimated_growth = min(float(revenue_growth), float(earnings_growth))
        elif revenue_growth:
            estimated_growth = float(revenue_growth)
        elif earnings_growth:
            estimated_growth = float(earnings_growth)
        else:
            estimated_growth = 0.08  # conservative default

        # Cap growth rate
        estimated_growth = max(0.02, min(estimated_growth, 0.30))

        # DCF: 5-year projection + terminal value
        pv_fcf = 0
        for yr in range(1, high_growth_years + 1):
            projected_fcf = fcf_base * (1 + estimated_growth) ** yr
            pv_fcf += projected_fcf / (1 + wacc) ** yr

        # Terminal value (Gordon Growth Model)
        terminal_fcf = fcf_base * (1 + estimated_growth) ** high_growth_years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / (1 + wacc) ** high_growth_years

        # Enterprise value
        enterprise_value = pv_fcf + pv_terminal

        # Per share intrinsic value
        intrinsic_per_share = enterprise_value / shares

        # Margin of safety
        mos_pct = (intrinsic_per_share - current_price) / intrinsic_per_share * 100
        upside_pct = (intrinsic_per_share / current_price - 1) * 100

        # Verdict
        if mos_pct >= 30:
            verdict = "Undervalued"
        elif mos_pct >= 0:
            verdict = "Fair"
        else:
            verdict = "Overvalued"

        # Score
        dcf_score = normalize_score(mos_pct, -50, 50)

        result.update({
            "fcf_base": round(fcf_base, 0),
            "growth_rate_used": round(estimated_growth * 100, 1),
            "wacc_used": round(wacc * 100, 1),
            "intrinsic_value": round(intrinsic_per_share, 2),
            "current_price": round(current_price, 2),
            "margin_of_safety_pct": round(mos_pct, 1),
            "upside_pct": round(upside_pct, 1),
            "verdict": verdict,
            "dcf_score": round(dcf_score, 1),
        })

        return result

    except Exception as e:
        logger.warning("DCF failed for %s: %s", ticker, e)
        return result


# ═══════════════════════════════════════════════════════════════
# Moat Strength Score (Phase 4.3)
# ═══════════════════════════════════════════════════════════════

def compute_moat_score(ticker: str) -> dict:
    """Compute competitive moat strength from financial consistency.

    Components (each 0-25, total 0-100):
    1. ROIC consistency (years ROIC > 15%)
    2. Margin stability (low variance = moat)
    3. Revenue growth consistency (positive streaks)
    4. FCF quality (cash conversion > 1.0)

    Returns dict with moat_score, moat_grade, components.
    """
    result = {"ticker": ticker, "data_available": False}

    try:
        # Get ROIC history
        roic_data = compute_roic_history(ticker)

        # Get FCF data
        fcf_data = compute_fcf(ticker)

        if not roic_data.get("data_available"):
            return result

        result["data_available"] = True
        components = {}

        # 1. ROIC consistency (0-25)
        consistency = roic_data.get("roic_consistency", 0)
        roic_vals = [r["roic_pct"] for r in roic_data.get("roic_history", [])]
        consistency_score = normalize_score(consistency, 0, 5) / 4
        components["roic_consistency"] = {
            "value": consistency,
            "years_available": len(roic_vals),
            "score": round(consistency_score, 1),
        }

        # 2. Margin stability (0-25) - lower variance = better
        if len(roic_vals) >= 3:
            roic_std = np.std(roic_vals)
            stability_score = normalize_score(roic_std, 15, 2) / 4  # lower std = higher score
        else:
            stability_score = 12.5
        components["margin_stability"] = {"value": round(roic_std, 1) if len(roic_vals) >= 3 else None,
                                          "score": round(stability_score, 1)}

        # 3. Revenue growth consistency (0-25)
        try:
            fin = get_annual_financials(ticker)
            revenue = extract_row(fin["income"], REVENUE_ROWS)
            if revenue is not None and len(revenue) >= 3:
                rev_vals = [float(revenue.iloc[i]) for i in range(min(5, len(revenue)))]
                positive_growth_years = sum(1 for i in range(len(rev_vals)-1) if rev_vals[i] > rev_vals[i+1])
                growth_score = normalize_score(positive_growth_years, 0, 4) / 4
            else:
                growth_score = 12.5
        except Exception:
            growth_score = 12.5
        components["revenue_consistency"] = {"score": round(growth_score, 1)}

        # 4. FCF quality (0-25)
        cash_conv = fcf_data.get("cash_conversion", 0)
        if cash_conv:
            fcf_quality_score = normalize_score(cash_conv, 0, 1.5) / 4
        else:
            fcf_quality_score = 12.5
        components["fcf_quality"] = {"value": cash_conv, "score": round(fcf_quality_score, 1)}

        # Composite
        total = sum(c["score"] for c in components.values())
        result["moat_score"] = round(total, 1)
        result["components"] = components

        # Grade
        if total >= 80:
            result["moat_grade"] = "Wide"
        elif total >= 55:
            result["moat_grade"] = "Narrow"
        else:
            result["moat_grade"] = "None"

        return result

    except Exception as e:
        logger.warning("Moat score failed for %s: %s", ticker, e)
        return result


# ═══════════════════════════════════════════════════════════════
# Composite Value Score (combines all value metrics)
# ═══════════════════════════════════════════════════════════════

def compute_value_score(ticker: str) -> dict:
    """Compute composite value score combining ROIC, FCF, Fortress, and DCF.

    Returns unified dict with all value metrics and a single 0-100 value score.
    """
    result = {"ticker": ticker, "data_available": False}

    roic = compute_roic_history(ticker)
    fcf = compute_fcf(ticker)
    fortress = compute_fortress_score(ticker)
    dcf = compute_simple_dcf(ticker)
    moat = compute_moat_score(ticker)

    any_available = any(d.get("data_available") for d in [roic, fcf, fortress, dcf, moat])
    if not any_available:
        return result

    result["data_available"] = True
    result["roic"] = roic
    result["fcf"] = fcf
    result["fortress"] = fortress
    result["dcf"] = dcf
    result["moat"] = moat

    # Composite value score (weighted)
    components = [
        (roic.get("roic_score", 50), 25),
        (fcf.get("fcf_score", 50), 20),
        (fortress.get("fortress_score", 50), 20),
        (dcf.get("dcf_score", 50), 20),
        (moat.get("moat_score", 50), 15),
    ]

    result["value_score"] = weighted_composite(components)
    result["value_grade"] = grade_score(result["value_score"])

    return result
