"""
Asymmetric Risk/Reward Scanner — identifies setups where upside vastly exceeds downside.
Druckenmiller's principle: bet big when R:R is asymmetric.
"""
from scoring_utils import normalize_score


def scan_asymmetric_setups(
    candidates: list[dict],
    min_rr: float = 3.0,
) -> list[dict]:
    """Scan candidates for asymmetric risk/reward setups.

    For each candidate with entry_setup, computes:
    - Multi-level R:R (1R, 2R, 3R, 5R targets)
    - Asymmetry score based on risk % and potential reward

    Args:
        candidates: List of Stage 2 candidates with entry_setup dicts.
        min_rr: Minimum R:R ratio to include (default 3.0).

    Returns:
        Filtered and scored list sorted by R:R ratio (best first).
    """
    scored = []

    for c in candidates:
        entry_setup = c.get("entry_setup")
        if not entry_setup:
            scored.append(c)
            continue

        entry = entry_setup.get("entry_price", 0)
        stop = entry_setup.get("effective_stop", 0)
        if entry <= 0 or stop <= 0 or stop >= entry:
            scored.append(c)
            continue

        risk_per_share = entry - stop
        risk_pct = round(risk_per_share / entry * 100, 2)

        # Multi-level targets
        targets = []
        for r_mult in [1.0, 2.0, 3.0, 5.0, 8.0]:
            target_price = round(entry + risk_per_share * r_mult, 2)
            gain_pct = round((target_price / entry - 1) * 100, 1)
            targets.append({
                "r_multiple": r_mult,
                "target_price": target_price,
                "gain_pct": gain_pct,
            })

        # Best realistic R:R (use 3R as primary target)
        rr_ratio = 3.0  # default
        if risk_pct > 0:
            # Estimate potential upside from RS and stage quality
            s2_score = c.get("stage", {}).get("s2_score", 4)
            rs = c.get("rs_vs_nifty", 0)

            # Better setups (high RS, perfect S2) get higher estimated R:R
            quality_mult = 1.0
            if s2_score >= 7:
                quality_mult += 0.5
            if rs > 10:
                quality_mult += 0.3
            vcp = c.get("vcp")
            if vcp and vcp.get("is_vcp"):
                quality_mult += 0.3

            rr_ratio = round(3.0 * quality_mult, 1)

        # Asymmetry score (0-100)
        # High R:R + low risk % = high asymmetry
        rr_score = normalize_score(rr_ratio, 1.0, 8.0)
        risk_score = normalize_score(risk_pct, 15.0, 2.0)  # lower risk = higher score
        asymmetry_score = round(rr_score * 0.6 + risk_score * 0.4, 1)

        updated = {**c}
        # Preserve fundamental_veto's targets dict, add R:R into it
        existing_targets = updated.get("targets")
        if isinstance(existing_targets, dict):
            existing_targets["reward_risk_ratio"] = rr_ratio
        else:
            updated["targets"] = {"reward_risk_ratio": rr_ratio}
        updated["rr_targets"] = targets
        updated["rr_ratio"] = rr_ratio
        updated["asymmetry_score"] = asymmetry_score
        scored.append(updated)

    scored.sort(key=lambda x: x["asymmetry_score"], reverse=True)
    return scored


def compute_multi_level_targets(
    entry_price: float,
    stop_loss: float,
) -> list[dict]:
    """Compute R-multiple profit targets with sell plan.

    Returns list of target levels with prices, gain %, and action.
    """
    if entry_price <= 0 or stop_loss >= entry_price:
        return []

    risk = entry_price - stop_loss

    levels = [
        (1.0, 25, "Sell 25% — lock in 1R"),
        (2.0, 25, "Sell 25% — 2R profit"),
        (3.0, 25, "Sell 25% — 3R profit"),
        (5.0, 25, "Hold final 25% with trailing stop"),
    ]

    targets = []
    for r_mult, sell_pct, action in levels:
        price = round(entry_price + risk * r_mult, 2)
        gain = round((price / entry_price - 1) * 100, 1)
        targets.append({
            "r_multiple": r_mult,
            "price": price,
            "gain_pct": gain,
            "sell_pct": sell_pct,
            "action": action,
        })

    return targets
