"""
Market Breadth by Weinstein Stage — aggregate stage distribution across the universe.
When >60% of stocks are in Stage 2, it's a bull market.
When >50% are in Stage 4, it's a bear market.
"""
from scoring_utils import normalize_score


def compute_breadth_by_stage(
    stage_results: list[dict],
) -> dict:
    """Compute market breadth by stage distribution.

    Args:
        stage_results: List of dicts with at least {"stage": {"stage": 1-4}}.

    Returns:
        dict with stage counts, percentages, and market health assessment.
    """
    total = len(stage_results)
    if total == 0:
        return {
            "total_stocks": 0,
            "stage_counts": {},
            "stage_pcts": {},
            "breadth_score": 50,
            "breadth_label": "NO DATA",
        }

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in stage_results:
        stage = r.get("stage", {})
        if isinstance(stage, dict):
            s = stage.get("stage", 0)
        else:
            s = stage
        if s in counts:
            counts[s] += 1

    pcts = {s: round(c / total * 100, 1) for s, c in counts.items()}

    # Breadth score: weighted by bullish stages
    # S2 stocks are positive, S4 stocks are negative
    bullish_pct = pcts.get(2, 0)
    bearish_pct = pcts.get(4, 0)
    basing_pct = pcts.get(1, 0)

    # Score: 0-100 where 50 is neutral
    # High S2% pushes above 50, high S4% pushes below 50
    raw_score = 50 + (bullish_pct - bearish_pct) * 0.5 + (basing_pct - 20) * 0.1
    breadth_score = round(max(0, min(100, raw_score)), 1)

    # Label
    if breadth_score >= 70:
        label = "STRONG BULL"
    elif breadth_score >= 55:
        label = "BULLISH"
    elif breadth_score >= 45:
        label = "NEUTRAL"
    elif breadth_score >= 30:
        label = "BEARISH"
    else:
        label = "STRONG BEAR"

    # Additional metrics
    s2_with_breakouts = sum(
        1 for r in stage_results
        if r.get("stage", {}).get("stage") == 2 and r.get("breakout")
    )

    return {
        "total_stocks": total,
        "stage_counts": counts,
        "stage_pcts": pcts,
        "breadth_score": breadth_score,
        "breadth_label": label,
        "s2_with_breakouts": s2_with_breakouts,
        "s2_breakout_pct": round(s2_with_breakouts / total * 100, 1) if total > 0 else 0,
        "net_stage_score": round(bullish_pct - bearish_pct, 1),
    }


def compute_sector_breadth(
    stage_results: list[dict],
) -> dict[str, dict]:
    """Compute breadth by stage for each sector.

    Args:
        stage_results: List of dicts with "sector" and "stage" keys.

    Returns:
        Dict mapping sector name to breadth dict.
    """
    from collections import defaultdict

    by_sector = defaultdict(list)
    for r in stage_results:
        sector = r.get("sector", "Unknown")
        by_sector[sector].append(r)

    result = {}
    for sector, stocks in by_sector.items():
        result[sector] = compute_breadth_by_stage(stocks)

    # Sort by breadth score descending
    return dict(sorted(result.items(), key=lambda x: x[1]["breadth_score"], reverse=True))
