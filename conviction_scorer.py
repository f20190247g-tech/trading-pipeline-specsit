"""
Tri-Factor Conviction Scoring Engine — ranks candidates by composite score (0-100).

Three pillars:
1. TECHNICAL (50%): Sector rank, stage score, base count, RS, accumulation, pattern quality
2. VALUE (25%): ROIC, FCF, fortress score, DCF margin of safety, moat
3. MACRO (25%): Macro liquidity regime, earnings acceleration, FII gating

Plus bonus factors (VCP, bulk deals, volume surge, delivery %, weekly confirmation).
"""
import logging
from datetime import datetime, timedelta
from config import CONVICTION_CONFIG
from scoring_utils import normalize_score, weighted_composite, percentile_rank

logger = logging.getLogger(__name__)


def compute_technical_score(
    candidate: dict,
    sector_rankings: list[dict],
    all_candidates: list[dict],
) -> float:
    """Compute the technical pillar score (0-100).

    Factors: sector rank, S2 score, base count, RS percentile, accumulation,
    consolidation quality, weekly confirmation.
    """
    score = 0.0

    # ── Sector Rank (35 pts max) ──────────────────────────────
    sector = candidate.get("sector", "")
    top_sectors = [r.get("sector") or r.get("name", "") for r in sector_rankings]
    if sector in top_sectors:
        rank = top_sectors.index(sector) + 1
        sector_pts_map = {1: 35, 2: 30, 3: 22, 4: 12}
        score += sector_pts_map.get(rank, 5)

        # Bonus: sector itself in Stage 2
        sector_info = sector_rankings[rank - 1] if rank <= len(sector_rankings) else {}
        sector_stage = sector_info.get("sector_stage", {})
        if sector_stage.get("stage") == 2:
            score += 5  # sector in Stage 2 = extra conviction

    # ── Stage 2 Score (20 pts max) ────────────────────────────
    s2_score = candidate.get("stage", {}).get("s2_score", 0)
    score += (s2_score / 7) * 20

    # ── Base Count (10 pts max) ───────────────────────────────
    base_count = candidate.get("base_count_in_stage2", 1)
    # Check for numbered bases from Phase 3.1
    breakout = candidate.get("breakout")
    if breakout:
        base_count = breakout.get("base_number", base_count)
    base_pts_map = {1: 10, 2: 8, 3: 5, 0: 3}
    score += base_pts_map.get(base_count, 2)

    # ── RS Percentile (15 pts max) ────────────────────────────
    rs_val = candidate.get("rs_vs_nifty", 0)
    all_rs = [c.get("rs_vs_nifty", 0) for c in all_candidates]
    if all_rs:
        rs_pctl = percentile_rank(rs_val, all_rs) / 100
        score += rs_pctl * 15

    # ── Accumulation (10 pts max) ─────────────────────────────
    acc_val = candidate.get("accumulation_ratio", 1.0)
    all_acc = [c.get("accumulation_ratio", 1.0) for c in all_candidates]
    if all_acc:
        acc_pctl = percentile_rank(acc_val, all_acc) / 100
        score += acc_pctl * 10

    # ── Consolidation Quality bonus (up to 5 pts) ────────────
    consolidation = candidate.get("consolidation")
    if consolidation:
        grade = consolidation.get("quality_grade", "D")
        consol_pts = {"A+": 5, "A": 4, "B": 3, "C": 1, "D": 0}
        score += consol_pts.get(grade, 0)

    # ── Weekly confirmation bonus (up to 5 pts) ──────────────
    weekly = candidate.get("weekly")
    if weekly and weekly.get("weekly_confirmed"):
        score += 3
        substage = weekly.get("weekly_s2_substage", "")
        if substage == "Early":
            score += 2  # early S2 = best timing

    return round(min(score, 100), 1)


def compute_value_pillar_score(candidate: dict) -> float:
    """Compute the value pillar score (0-100) from pre-computed value analysis.

    Expects candidate to have a 'value_analysis' dict (from value_analysis.compute_value_score).
    If not available, returns a neutral 50.
    """
    va = candidate.get("value_analysis")
    if not va or not va.get("data_available", False):
        return 50.0  # neutral if no data — don't penalize

    return round(va.get("composite_score", 50.0), 1)


def compute_macro_pillar_score(
    candidate: dict,
    macro_liquidity: dict | None = None,
    fii_gate: dict | None = None,
) -> float:
    """Compute the macro pillar score (0-100).

    Components:
    - Macro liquidity regime score (40%)
    - Earnings acceleration (40%)
    - FII gating adjustment (20%)
    """
    components = []

    # ── Macro liquidity (40%) ─────────────────────────────────
    if macro_liquidity:
        macro_score = macro_liquidity.get("score", 50)
        components.append((macro_score, 0.4))
    else:
        components.append((50.0, 0.4))  # neutral

    # ── Earnings acceleration (40%) ───────────────────────────
    earnings = candidate.get("earnings_analysis")
    if earnings and earnings.get("data_available"):
        ea_score = earnings.get("combined_score", 50)
        components.append((ea_score, 0.4))
    else:
        components.append((50.0, 0.4))  # neutral

    # ── FII gating (20%) ─────────────────────────────────────
    if fii_gate:
        gate_level = fii_gate.get("gate_level", "none")
        fii_scores = {"none": 70, "caution": 40, "severe": 15}
        components.append((fii_scores.get(gate_level, 50), 0.2))
    else:
        components.append((50.0, 0.2))

    return weighted_composite(components)


def compute_conviction_score(
    candidate: dict,
    sector_rankings: list[dict],
    all_candidates: list[dict],
    bulk_deals: list[dict] | None = None,
    delivery_data: dict | None = None,
    macro_liquidity: dict | None = None,
    fii_gate: dict | None = None,
) -> float:
    """Compute a 0-100 tri-factor conviction score for a single candidate.

    Tri-factor weights (from config, defaults below):
        Technical: 50%  — sector + stage + RS + accumulation + patterns
        Value:     25%  — ROIC + FCF + fortress + DCF + moat
        Macro:     25%  — liquidity regime + earnings accel + FII flows

    Plus bonuses up to +10 (VCP, bulk deals, volume surge, delivery).

    Args:
        candidate: Stage 2 candidate dict.
        sector_rankings: Sector ranking dicts from sector_rs module.
        all_candidates: All candidates (for percentile computation).
        bulk_deals: Recent bulk deals list from NSE.
        delivery_data: Delivery data dict for this stock.
        macro_liquidity: Macro liquidity score dict.
        fii_gate: FII gating check result dict.

    Returns:
        Float score 0-100.
    """
    cfg = CONVICTION_CONFIG

    # ── Three Pillars ─────────────────────────────────────────
    tech_score = compute_technical_score(candidate, sector_rankings, all_candidates)
    value_score = compute_value_pillar_score(candidate)
    macro_score = compute_macro_pillar_score(candidate, macro_liquidity, fii_gate)

    # Weights from config (with defaults)
    tech_weight = cfg.get("technical_weight", 50)
    value_weight = cfg.get("value_weight", 25)
    macro_weight = cfg.get("macro_weight", 25)

    pillar_score = weighted_composite([
        (tech_score, tech_weight),
        (value_score, value_weight),
        (macro_score, macro_weight),
    ])

    # Store pillar breakdown on the candidate for transparency
    candidate["conviction_pillars"] = {
        "technical": tech_score,
        "value": value_score,
        "macro": macro_score,
    }

    # ── Bonuses (up to +10) ──────────────────────────────────
    bonus = 0.0

    # VCP pattern detected: +3
    vcp = candidate.get("vcp")
    if vcp and vcp.get("is_vcp"):
        bonus += 3

    # Bulk deal in last 30 days: +3
    if bulk_deals:
        ticker_clean = candidate.get("ticker", "").replace(".NS", "").replace(".BO", "").upper()
        for deal in bulk_deals:
            if deal.get("symbol", "").upper() == ticker_clean:
                bonus += 3
                break

    # Volume surge >2x on breakout: +2
    breakout = candidate.get("breakout")
    if breakout and breakout.get("breakout"):
        vol_ratio = breakout.get("volume_ratio", 0)
        if vol_ratio >= 2.0:
            bonus += 2

    # Delivery % >50%: +2
    if delivery_data and delivery_data.get("delivery_pct", 0) > 50:
        bonus += 2

    # Asymmetric R:R bonus: +2 if R:R >= 5
    rr = candidate.get("rr_ratio", 0)
    if rr >= 5:
        bonus += 2

    score = pillar_score + min(bonus, 10)
    return round(min(score, 100), 1)


def rank_candidates_by_conviction(
    candidates: list[dict],
    sector_rankings: list[dict],
    bulk_deals: list[dict] | None = None,
    macro_liquidity: dict | None = None,
    fii_gate: dict | None = None,
) -> list[dict]:
    """Score and rank all candidates by conviction. Returns sorted list (highest first).

    Adds 'conviction_score' key to each candidate dict.
    """
    for candidate in candidates:
        candidate["conviction_score"] = compute_conviction_score(
            candidate=candidate,
            sector_rankings=sector_rankings,
            all_candidates=candidates,
            bulk_deals=bulk_deals,
            macro_liquidity=macro_liquidity,
            fii_gate=fii_gate,
        )

    return sorted(candidates, key=lambda c: c.get("conviction_score", 0), reverse=True)


def get_top_conviction_ideas(
    ranked_candidates: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """Return top N actionable ideas (BUY action + has entry setup).

    Args:
        ranked_candidates: Candidates already sorted by conviction_score.
        top_n: Number of top ideas to return.

    Returns:
        List of up to top_n candidates with BUY action and entry setups.
    """
    ideas = []
    for c in ranked_candidates:
        action = c.get("action", "")
        entry_setup = c.get("entry_setup")
        if action in ("BUY", "WATCHLIST") and entry_setup:
            ideas.append(c)
            if len(ideas) >= top_n:
                break
    return ideas


def get_top_ideas_by_sector(
    ranked_candidates: list[dict],
    top_sectors: list[str],
    per_sector: int = 3,
) -> dict[str, list[dict]]:
    """Return top N actionable ideas grouped by sector.

    Args:
        ranked_candidates: Candidates already sorted by conviction_score.
        top_sectors: Ordered list of top sector names.
        per_sector: Max ideas per sector.

    Returns:
        OrderedDict-like dict {sector_name: [candidates...]} preserving
        top_sectors order.  Only sectors with at least 1 idea are included.
    """
    sector_ideas: dict[str, list[dict]] = {s: [] for s in top_sectors}

    for c in ranked_candidates:
        sector = c.get("sector", "")
        if sector not in sector_ideas:
            continue
        if len(sector_ideas[sector]) >= per_sector:
            continue
        action = c.get("action", "")
        entry_setup = c.get("entry_setup")
        # Accept BUY, WATCHLIST, or WATCH — show all actionable ideas
        if action in ("BUY", "WATCHLIST", "WATCH") or entry_setup:
            sector_ideas[sector].append(c)

    # Remove sectors with no ideas, preserve order
    return {s: ideas for s, ideas in sector_ideas.items() if ideas}
