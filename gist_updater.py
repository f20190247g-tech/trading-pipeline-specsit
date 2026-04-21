"""
gist_updater.py — Push scan snapshot to a GitHub Gist.

The HTML dashboard reads from this Gist to show live data.
Reads credentials from Streamlit secrets or environment variables.
"""
from __future__ import annotations
import json
import os


def _get_credentials() -> tuple[str | None, str | None]:
    """Return (github_pat, gist_id) from Streamlit secrets or env vars."""
    # 1. Streamlit Cloud secrets (st.secrets)
    try:
        import streamlit as st
        pat = st.secrets.get("GITHUB_PAT") or st.secrets.get("GIST_PAT")
        gid = st.secrets.get("GIST_ID")
        if pat and gid:
            return str(pat), str(gid)
    except Exception:
        pass

    # 2. Environment variables (local use)
    pat = os.environ.get("GITHUB_PAT") or os.environ.get("GIST_PAT")
    gid = os.environ.get("GIST_ID")
    if pat and gid:
        return pat, gid

    return None, None


def push_snapshot(snapshot: dict) -> tuple[bool, str]:
    """
    Push snapshot dict to GitHub Gist as sdwb_scan.json.
    Returns (success: bool, message: str).
    """
    import requests

    pat, gist_id = _get_credentials()

    if not pat or not gist_id:
        msg = "Skipped — GITHUB_PAT or GIST_ID not configured in Streamlit secrets."
        print(f"  [Gist] {msg}")
        return False, msg

    try:
        content = json.dumps(snapshot, default=str)
        resp = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"token {pat}",
                "Accept":        "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"files": {"sdwb_scan.json": {"content": content}}},
            timeout=20,
        )
        if resp.status_code == 200:
            raw_url = f"https://gist.githubusercontent.com/{_gist_owner(resp.json())}/{gist_id}/raw/sdwb_scan.json"
            msg = f"✓ Pushed — {raw_url}"
            print(f"  [Gist] {msg}")
            return True, raw_url
        else:
            msg = f"✗ GitHub API returned {resp.status_code}: {resp.text[:300]}"
            print(f"  [Gist] {msg}")
            return False, msg
    except Exception as e:
        msg = f"✗ Network error: {e}"
        print(f"  [Gist] {msg}")
        return False, msg


def _gist_owner(gist_json: dict) -> str:
    try:
        return gist_json["owner"]["login"]
    except Exception:
        return "unknown"


def build_snapshot_from_cache() -> dict | None:
    """Load the latest scan cache and convert to dashboard JSON format."""
    import pickle
    from pathlib import Path

    CACHE_DIR = Path(__file__).parent / "scan_cache"
    candidates = [
        CACHE_DIR / "last_weekly_scan.pkl",
        CACHE_DIR / "last_scan.pkl",
    ]
    data: dict = {}
    for path in candidates:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data.update(pickle.load(f))
            except Exception:
                pass

    daily = CACHE_DIR / "last_daily_check.pkl"
    if daily.exists():
        try:
            with open(daily, "rb") as f:
                data.update(pickle.load(f))
        except Exception:
            pass

    if not data:
        return None

    # Import the same transformers used by api.py
    try:
        from api import (
            _build_regime, _build_nifty, _build_breadth,
            _build_sectors, _build_stocks, _build_watchlist,
            _build_positions, _build_fii, _build_earnings,
            _flt, _int,
        )
    except ImportError:
        return {"error": "api.py not found", "is_live": False}

    regime_raw  = data.get("regime", {}) or {}
    signals_raw = regime_raw.get("signals", {}) or {}
    brs         = data.get("breadth_by_stage", {}) or {}
    fii_today   = data.get("daily_fii_dii") or {}
    fii_flows   = data.get("daily_fii_dii_flows") or {}

    stock_data = dict(data.get("stock_data", {}) or {})
    stock_data.update(data.get("daily_stock_data") or {})

    positions, trade_history = _build_positions(stock_data)

    return {
        "is_live":       True,
        "scan_date":     data.get("scan_date", data.get("last_weekly_scan_date", "")),
        "capital":       _flt(data.get("capital", 5_000_000), 5_000_000),
        "regime":        _build_regime(regime_raw),
        "nifty":         _build_nifty(data.get("nifty_df")),
        "breadth":       _build_breadth(brs, signals_raw),
        "sectors":       _build_sectors(data.get("sector_rankings", []), data.get("top_sectors", [])),
        "stocks":        _build_stocks(
                             data.get("screened_stocks", []),
                             data.get("stage2_candidates", []),
                             data.get("final_watchlist", []),
                         ),
        "watchlist":     _build_watchlist(data.get("final_watchlist", [])),
        "positions":     positions,
        "trade_history": trade_history,
        "fii_dii":       _build_fii(fii_today, fii_flows, data.get("fii_history")),
        "earnings":      _build_earnings(data.get("earnings_season") or {}),
    }
