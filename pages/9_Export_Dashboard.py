"""
Page 9: Export Dashboard Data
Pushes the latest scan snapshot to GitHub Gist so the HTML dashboard shows live data.
"""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Export Dashboard", page_icon="📡", layout="wide")
st.title("📡 Export to HTML Dashboard")

st.markdown("""
Push your latest scan data to GitHub Gist so the **SDWB HTML Dashboard** shows live data.

**How it works:**
1. Your scan runs and saves results to `scan_cache/`
2. Click the button below — data is pushed to your GitHub Gist
3. Open `SDWB Dashboard (Standalone).html` — it will show **● LIVE**
""")

# ── Check credentials ─────────────────────────────────────────
pat  = st.secrets.get("GITHUB_PAT") or st.secrets.get("GIST_PAT") if hasattr(st, "secrets") else None
gist = st.secrets.get("GIST_ID") if hasattr(st, "secrets") else None

if not pat or not gist:
    st.error("**Credentials not configured.** Add these to your Streamlit Cloud secrets:")
    st.code("""
GITHUB_PAT = "ghp_xxxxxxxxxxxxxxxxxxxx"
GIST_ID    = "xxxxxxxxxxxxxxxxxxxx"
""", language="toml")
    st.markdown("""
**How to get these:**
- **GITHUB_PAT** → github.com → Settings → Developer Settings → Personal Access Tokens → Generate new token → tick **gist** scope → copy the token
- **GIST_ID** → gist.github.com → New Gist → create with any content → copy the long ID from the URL (`gist.github.com/username/**THIS_PART**`)
""")
    st.stop()

st.success("✓ Credentials found in Streamlit secrets")

# ── Check cache ───────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "scan_cache"
weekly    = CACHE_DIR / "last_weekly_scan.pkl"
legacy    = CACHE_DIR / "last_scan.pkl"

cache_exists = weekly.exists() or legacy.exists()
if not cache_exists:
    st.warning("No scan cache found. Run a **Weekend Scan** from the home page first, then come back here.")
    st.stop()

scan_date = st.session_state.get("scan_date", st.session_state.get("last_weekly_scan_date", "unknown"))
st.info(f"Cache found · Last scan: **{scan_date}**")

# ── Gist URL display ──────────────────────────────────────────
raw_url = f"https://gist.githubusercontent.com/f20190247g-tech/{gist}/raw/sdwb_scan.json"
st.markdown("**Your Gist raw URL** (paste this into your HTML dashboard):")
st.code(raw_url)

# ── Push button ───────────────────────────────────────────────
st.divider()
col1, col2 = st.columns([1, 2])
with col1:
    push_btn = st.button("🚀 Push to Gist", type="primary", use_container_width=True)
with col2:
    st.caption("Pushes current scan data to GitHub Gist · Takes ~3 seconds")

if push_btn:
    with st.spinner("Building snapshot and pushing to GitHub Gist..."):
        try:
            from gist_updater import build_snapshot_from_cache, push_snapshot
            snapshot = build_snapshot_from_cache()
            if snapshot is None:
                st.error("Could not load scan cache. Run a Weekend Scan first.")
            else:
                ok, msg = push_snapshot(snapshot)
                if ok:
                    st.success(f"**Data pushed!** Open `SDWB Dashboard (Standalone).html` — it will show ● LIVE")
                    st.markdown(f"Gist URL: `{msg}`")
                    st.balloons()
                else:
                    st.error(f"Push failed: {msg}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Auto-push toggle ──────────────────────────────────────────
st.divider()
st.markdown("#### Auto-push after scans")
st.markdown("""
To push automatically every time you run a scan, add these 3 lines to the **end** of `run_refresh()` in `refresh_data.py`:

```python
    # Auto-push to Gist for HTML dashboard
    from gist_updater import build_snapshot_from_cache, push_snapshot
    push_snapshot(build_snapshot_from_cache())
```

Then every scan → Gist → HTML dashboard updates automatically. No manual button needed.
""")

# ── Instructions for HTML ─────────────────────────────────────
st.divider()
st.markdown("#### Set your Gist URL in the HTML dashboard")
st.markdown(f"""
Open `SDWB Dashboard (Standalone).html` in a text editor and find this line near the top:

```javascript
const GIST_URL = "";   // ← paste your Gist raw URL here
```

Replace it with:

```javascript
const GIST_URL = "{raw_url}";
```

Save the file. Done — the dashboard will now fetch live data from your Gist.
""")
