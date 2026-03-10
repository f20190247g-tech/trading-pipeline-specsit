"""
FII/DII Flow Gating — Druckenmiller's liquidity lens.
When FII outflows are severe and DII can't offset, reduce exposure.
"""
from config import FII_GATING_CONFIG


def check_fii_gate(fii_dii_data: dict | None) -> dict:
    """Check if FII outflows warrant gating new positions.

    Args:
        fii_dii_data: Dict with keys like 'fii_5d_net', 'fii_10d_net',
                      'dii_5d_net', 'dii_10d_net' (values in Cr).

    Returns:
        dict with:
            gated: bool — True if conditions warrant caution
            gate_level: "none" | "caution" | "severe"
            size_multiplier: float — 1.0 (full) or reduced
            reason: str
    """
    cfg = FII_GATING_CONFIG

    if not fii_dii_data:
        return {
            "gated": False,
            "gate_level": "none",
            "size_multiplier": 1.0,
            "reason": "No FII/DII data available",
        }

    fii_5d = fii_dii_data.get("fii_5d_net", 0)
    fii_10d = fii_dii_data.get("fii_10d_net", 0)
    dii_5d = fii_dii_data.get("dii_5d_net", 0)
    dii_10d = fii_dii_data.get("dii_10d_net", 0)

    reasons = []
    gate_level = "none"

    # Check severe outflow
    if fii_10d < cfg["fii_10d_threshold"]:
        gate_level = "severe"
        reasons.append(f"FII 10d outflow {fii_10d:,.0f} Cr (threshold {cfg['fii_10d_threshold']:,.0f})")
    elif fii_5d < cfg["fii_5d_threshold"]:
        gate_level = "caution"
        reasons.append(f"FII 5d outflow {fii_5d:,.0f} Cr (threshold {cfg['fii_5d_threshold']:,.0f})")

    # Check if DII support offsets
    if gate_level != "none":
        net_flow_5d = fii_5d + dii_5d
        net_flow_10d = fii_10d + dii_10d

        if dii_10d > cfg["dii_support_threshold"] and net_flow_10d > 0:
            # DII buying offsets FII selling — reduce gate severity
            if gate_level == "severe":
                gate_level = "caution"
                reasons.append(f"DII support {dii_10d:,.0f} Cr partially offsets")
            else:
                gate_level = "none"
                reasons.append(f"DII buying {dii_10d:,.0f} Cr offsets FII outflow")

    # Determine size multiplier
    if gate_level == "severe":
        size_mult = (100 - cfg["size_reduction_pct"]) / 100
    elif gate_level == "caution":
        size_mult = (100 - cfg["size_reduction_pct"] / 2) / 100
    else:
        size_mult = 1.0

    return {
        "gated": gate_level != "none",
        "gate_level": gate_level,
        "size_multiplier": round(size_mult, 2),
        "reason": "; ".join(reasons) if reasons else "FII flows normal",
        "fii_5d": fii_5d,
        "fii_10d": fii_10d,
        "dii_5d": dii_5d,
        "dii_10d": dii_10d,
    }
