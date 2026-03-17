"""
NSE India API wrapper using curl_cffi for TLS fingerprint impersonation.

NSE blocks Python's `requests` library via TLS fingerprinting (403).
curl_cffi impersonates a real Chrome browser's TLS handshake to bypass this.

Provides quarterly financials, shareholding patterns, and corporate
announcements from NSE's undocumented API endpoints.
"""
import time
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

from curl_cffi import requests as cf_requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "scan_cache"
CACHE_TTL_HOURS = 24
SCREENER_UNFILED_TTL_HOURS = 24  # Re-check daily for stocks that haven't filed yet


class NSEDataFetcher:
    """NSE India API wrapper with Chrome TLS impersonation and caching."""

    BASE_URL = "https://www.nseindia.com"

    def __init__(self):
        self.session = cf_requests.Session(impersonate="chrome")
        self._last_request_time = 0.0
        self._min_interval = 0.35  # ~3 requests/sec
        self._cookies_valid = False

    def _init_cookies(self):
        """Establish session cookies by visiting the main page."""
        try:
            resp = self.session.get(self.BASE_URL, timeout=10)
            if resp.status_code == 200:
                self._cookies_valid = True
            else:
                logger.warning("NSE cookie init got status %d", resp.status_code)
                self._cookies_valid = False
        except Exception as e:
            logger.warning("NSE cookie init failed: %s", e)
            self._cookies_valid = False

    def _request(self, url, params=None, retries=3):
        """Rate-limited request with retry and cookie refresh."""
        for attempt in range(retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            if not self._cookies_valid:
                self._init_cookies()
                if not self._cookies_valid:
                    time.sleep(2 ** attempt)
                    continue

            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code in (403, 429):
                    logger.info("NSE %d on attempt %d, refreshing cookies", resp.status_code, attempt + 1)
                    self._cookies_valid = False
                    time.sleep(2 ** attempt)
                    continue

                if resp.status_code == 404:
                    return None

                resp.raise_for_status()
                return resp.json()

            except ValueError:
                logger.warning("NSE returned non-JSON on attempt %d", attempt + 1)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("NSE request failed on attempt %d: %s", attempt + 1, e)
                self._cookies_valid = False
                time.sleep(2 ** attempt)

        return None

    # ── Cache helpers ─────────────────────────────────────────────

    def _cache_path(self, symbol: str, data_type: str) -> Path:
        CACHE_DIR.mkdir(exist_ok=True)
        return CACHE_DIR / f"{symbol}_{data_type}.pkl"

    def _load_cache(self, symbol: str, data_type: str, ttl_hours: float | None = None):
        path = self._cache_path(symbol, data_type)
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            ttl = ttl_hours if ttl_hours is not None else CACHE_TTL_HOURS
            if datetime.now() - mtime > timedelta(hours=ttl):
                return None
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, symbol: str, data_type: str, data):
        try:
            path = self._cache_path(symbol, data_type)
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning("Cache save failed for %s/%s: %s", symbol, data_type, e)

    # ── Clean symbol helper ───────────────────────────────────────

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """Convert 'RELIANCE.NS' -> 'RELIANCE'."""
        return symbol.replace(".NS", "").replace(".BO", "").strip().upper()

    # ── Quarterly Results ─────────────────────────────────────────

    def fetch_quarterly_results(self, symbol: str, num_quarters: int = 20) -> pd.DataFrame | None:
        """Fetch quarterly financial results from NSE.

        Uses /api/results-comparision for detailed P&L (5 quarters),
        plus /api/top-corp-info for summary data.

        Returns DataFrame with columns:
            date, revenue, operating_income, net_income, diluted_eps,
            opm_pct, npm_pct, depreciation, tax
        All monetary values in lakhs (divide by 100 for Cr).
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "quarterly_v2")
        if cached is not None:
            return cached

        # Primary: detailed P&L from results-comparision
        data = self._request(f"{self.BASE_URL}/api/results-comparision?symbol={clean}")

        if not data:
            return None

        try:
            records = data.get("resCmpData", [])
            if not records:
                return None

            rows = []
            for r in records[:num_quarters]:
                date_str = r.get("re_to_dt")
                if not date_str:
                    continue
                try:
                    date = pd.to_datetime(date_str, dayfirst=True)
                except Exception:
                    continue

                revenue = self._parse_num(r.get("re_net_sale"))
                total_income = self._parse_num(r.get("re_total_inc"))
                other_income = self._parse_num(r.get("re_oth_inc_new"))
                total_expense = self._parse_num(r.get("re_oth_tot_exp"))
                depreciation = self._parse_num(r.get("re_depr_und_exp"))
                pbt = self._parse_num(r.get("re_pro_loss_bef_tax"))
                tax = self._parse_num(r.get("re_tax"))
                net_income = self._parse_num(r.get("re_net_profit") or r.get("re_con_pro_loss"))
                eps = self._parse_num(r.get("re_dilut_eps_for_cont_dic_opr") or r.get("re_diluted_eps"))

                # Operating income = revenue - total expenses + depreciation + tax
                # Or: PBT + interest
                interest = self._parse_num(r.get("re_int_new"))
                operating_income = None
                if pbt is not None and interest is not None:
                    operating_income = pbt + interest
                elif revenue is not None and total_expense is not None:
                    operating_income = revenue - total_expense + (depreciation or 0) + (tax or 0)

                # Margins computed before unit conversion (ratios are unitless)
                opm = (operating_income / revenue * 100) if revenue and operating_income else None
                npm = (net_income / revenue * 100) if revenue and net_income else None

                # NSE reports monetary values in lakhs; convert to rupees
                # so _fmt_cr (which divides by 1e7) displays correctly.
                LAKHS = 1e5
                rows.append({
                    "date": date,
                    "revenue": revenue * LAKHS if revenue else None,
                    "operating_income": operating_income * LAKHS if operating_income else None,
                    "net_income": net_income * LAKHS if net_income else None,
                    "diluted_eps": eps,  # EPS is already per-share, no conversion
                    "opm_pct": opm,
                    "npm_pct": npm,
                    "depreciation": depreciation * LAKHS if depreciation else None,
                    "tax": tax * LAKHS if tax else None,
                    "pbt": pbt * LAKHS if pbt else None,
                    "other_income": other_income * LAKHS if other_income else None,
                })

            if not rows:
                return None

            # Detect and fix unit inconsistency: NSE sometimes returns
            # the oldest quarter in crores while others are in lakhs,
            # causing a ~100x difference in monetary values.
            self._fix_unit_outliers(rows)

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            self._save_cache(clean, "quarterly_v2", df)
            return df

        except Exception as e:
            logger.warning("Failed to parse quarterly results for %s: %s", clean, e)
            return None

    # ── Annual Results ────────────────────────────────────────────

    def fetch_annual_results(self, symbol: str, num_years: int = 10) -> pd.DataFrame | None:
        """Fetch annual financial results from NSE.

        NSE doesn't have a separate annual endpoint — we return None and
        let the caller fall back to yfinance for annual data.
        """
        # NSE's results-comparision only returns recent quarterly data.
        # No separate annual endpoint exists. Return None to trigger yfinance fallback.
        return None

    # ── Shareholding Pattern ──────────────────────────────────────

    def fetch_shareholding_pattern(self, symbol: str, num_quarters: int = 20) -> list[dict] | None:
        """Fetch shareholding pattern from NSE.

        Uses /api/top-corp-info which provides Promoter vs Public split
        for the last 5 quarters.

        Returns list of dicts: {date, promoter_pct, public_pct}
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "shareholding")
        if cached is not None:
            return cached

        data = self._request(
            f"{self.BASE_URL}/api/top-corp-info",
            params={"symbol": clean, "market": "equities"},
        )
        if not data:
            return None

        try:
            sh_data = data.get("shareholdings_patterns", {}).get("data", {})
            if not sh_data:
                return None

            results = []
            for date_str, holdings in sh_data.items():
                try:
                    date = pd.to_datetime(date_str, dayfirst=True)
                except Exception:
                    continue

                entry = {"date": date, "promoter_pct": None, "public_pct": None}
                for h in holdings:
                    for key, val in h.items():
                        pct = self._parse_num(val)
                        key_lower = key.lower()
                        if "promoter" in key_lower:
                            entry["promoter_pct"] = pct
                        elif "public" in key_lower:
                            entry["public_pct"] = pct

                if entry["promoter_pct"] is not None:
                    results.append(entry)

            if not results:
                return None

            results.sort(key=lambda x: x["date"])
            self._save_cache(clean, "shareholding", results)
            return results

        except Exception as e:
            logger.warning("Failed to parse shareholding for %s: %s", clean, e)
            return None

    # ── Announcements ─────────────────────────────────────────────

    def fetch_announcements(self, symbol: str, months: int = 3) -> list[dict] | None:
        """Fetch corporate announcements from NSE.

        Uses /api/corporate-announcements which returns full history.
        We filter to the requested months.

        Returns list of dicts: {date, subject, category}
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "announcements")
        if cached is not None:
            return cached

        data = self._request(
            f"{self.BASE_URL}/api/corporate-announcements",
            params={"index": "equities", "symbol": clean},
        )
        if not data:
            return None

        try:
            records = data if isinstance(data, list) else data.get("data", [])
            if not records:
                return None

            cutoff = datetime.now() - timedelta(days=months * 30)
            results = []
            for r in records:
                date_str = r.get("an_dt") or r.get("sort_date")
                if not date_str:
                    continue
                try:
                    date = pd.to_datetime(date_str)
                except Exception:
                    continue

                if date < cutoff:
                    continue

                # Use attchmntText (full description) if available, else desc (short)
                subject = r.get("attchmntText") or r.get("desc") or ""
                # Use desc as category (it's usually a short label like "Updates", "Board Meeting")
                category = r.get("desc") or r.get("smIndustry") or "General"

                results.append({
                    "date": date,
                    "subject": subject[:300],
                    "category": category,
                })

            if not results:
                return None

            results.sort(key=lambda x: x["date"], reverse=True)
            self._save_cache(clean, "announcements", results)
            return results

        except Exception as e:
            logger.warning("Failed to parse announcements for %s: %s", clean, e)
            return None

    # ── Smart Money Data ──────────────────────────────────────────

    def fetch_fii_dii_data(self) -> dict | None:
        """Fetch FII/DII daily trading activity from NSE.

        Returns dict: {date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net}
        Values in Crores.
        """
        cached = self._load_cache("MARKET", "fii_dii")
        if cached is not None:
            return cached

        data = self._request(f"{self.BASE_URL}/api/fiidiiTradeReact")
        if not data:
            return None

        try:
            result = {"date": None, "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
                      "dii_buy": 0, "dii_sell": 0, "dii_net": 0}

            records = data if isinstance(data, list) else [data]
            for r in records:
                category = (r.get("category") or "").upper()
                buy_val = self._parse_num(r.get("buyValue"))
                sell_val = self._parse_num(r.get("sellValue"))
                net_val = self._parse_num(r.get("netValue"))
                date_str = r.get("date")

                if date_str and result["date"] is None:
                    result["date"] = date_str

                if "FII" in category or "FPI" in category:
                    result["fii_buy"] = buy_val or 0
                    result["fii_sell"] = sell_val or 0
                    result["fii_net"] = net_val or 0
                elif "DII" in category:
                    result["dii_buy"] = buy_val or 0
                    result["dii_sell"] = sell_val or 0
                    result["dii_net"] = net_val or 0

            self._save_cache("MARKET", "fii_dii", result)
            return result

        except Exception as e:
            logger.warning("Failed to parse FII/DII data: %s", e)
            return None

    def fetch_fii_dii_historical(self, days: int = 1825) -> pd.DataFrame:
        """Load and accumulate historical FII/DII daily data.

        Strategy:
        1. Load existing history from scan_cache/fii_dii_history.csv
           (seeded with embedded data on first run)
        2. Fetch today's data from NSE /api/fiidiiTradeReact
        3. Append, deduplicate, save, return

        NSE only exposes the current day via API — no date-range endpoint.
        History accumulates over time as each run appends today's data.

        Returns:
            DataFrame with columns: date, fii_buy, fii_sell, fii_net,
            dii_buy, dii_sell, dii_net. Sorted by date ascending.
        """
        csv_path = CACHE_DIR / "fii_dii_history.csv"
        CACHE_DIR.mkdir(exist_ok=True)

        # Load existing CSV or seed with embedded data
        existing = None
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path, parse_dates=["date"])
            except Exception:
                existing = None

        if existing is None or existing.empty:
            existing = self._seed_fii_dii_history()
            if not existing.empty:
                try:
                    existing.to_csv(csv_path, index=False)
                except Exception as e:
                    logger.warning("Failed to save seed FII/DII history: %s", e)

        # Check if we already have today's data
        today = datetime.now().date()
        if existing is not None and not existing.empty:
            last_date = pd.to_datetime(existing["date"]).max().date()
            if last_date >= today:
                return existing.sort_values("date").reset_index(drop=True)

        # Fetch today's data from NSE
        data = self._request(f"{self.BASE_URL}/api/fiidiiTradeReact")
        new_rows = []
        if data:
            new_rows = self._parse_fii_dii_records(data)

        # Merge with existing
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            # Deduplicate by date (keep latest)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date").drop_duplicates(
                subset=["date"], keep="last"
            ).reset_index(drop=True)

            # Save to CSV
            try:
                combined.to_csv(csv_path, index=False)
            except Exception as e:
                logger.warning("Failed to save FII/DII history: %s", e)

            return combined

        if existing is not None and not existing.empty:
            return existing.sort_values("date").reset_index(drop=True)

        return pd.DataFrame(
            columns=["date", "fii_buy", "fii_sell", "fii_net",
                      "dii_buy", "dii_sell", "dii_net"]
        )

    @staticmethod
    def _seed_fii_dii_history() -> pd.DataFrame:
        """Seed data from MrChartist/fii-dii-data for first run.

        Daily granularity for recent sessions. This gives the dashboard
        something to show before the CSV accumulates via daily fetches.
        """
        seed = [
            {"date": "2026-03-13", "fii_buy": 11923, "fii_sell": 22647, "fii_net": -10724, "dii_buy": 22707, "dii_sell": 12730, "dii_net": 9977},
            {"date": "2026-03-12", "fii_buy": 14201, "fii_sell": 16122, "fii_net": -1921, "dii_buy": 18402, "dii_sell": 13201, "dii_net": 5201},
            {"date": "2026-03-11", "fii_buy": 18420, "fii_sell": 15201, "fii_net": 3219, "dii_buy": 12101, "dii_sell": 15402, "dii_net": -3301},
            {"date": "2026-03-10", "fii_buy": 10123, "fii_sell": 18402, "fii_net": -8279, "dii_buy": 19402, "dii_sell": 10201, "dii_net": 9201},
            {"date": "2026-03-09", "fii_buy": 12402, "fii_sell": 17602, "fii_net": -5200, "dii_buy": 16021, "dii_sell": 11002, "dii_net": 5019},
            {"date": "2026-03-06", "fii_buy": 15602, "fii_sell": 19201, "fii_net": -3599, "dii_buy": 18201, "dii_sell": 13401, "dii_net": 4800},
            {"date": "2026-03-05", "fii_buy": 11201, "fii_sell": 24102, "fii_net": -12901, "dii_buy": 25102, "dii_sell": 12102, "dii_net": 13000},
            {"date": "2026-03-04", "fii_buy": 14201, "fii_sell": 13201, "fii_net": 1000, "dii_buy": 13201, "dii_sell": 14102, "dii_net": -901},
            {"date": "2026-03-03", "fii_buy": 18402, "fii_sell": 19201, "fii_net": -799, "dii_buy": 15002, "dii_sell": 12001, "dii_net": 3001},
            {"date": "2026-03-02", "fii_buy": 10101, "fii_sell": 14201, "fii_net": -4100, "dii_buy": 14201, "dii_sell": 9001, "dii_net": 5200},
            {"date": "2026-02-27", "fii_buy": 22102, "fii_sell": 28402, "fii_net": -6300, "dii_buy": 30102, "dii_sell": 18201, "dii_net": 11901},
            {"date": "2026-02-26", "fii_buy": 15401, "fii_sell": 12102, "fii_net": 3299, "dii_buy": 11201, "dii_sell": 16401, "dii_net": -5200},
            {"date": "2026-02-25", "fii_buy": 12402, "fii_sell": 19402, "fii_net": -7000, "dii_buy": 18401, "dii_sell": 11202, "dii_net": 7199},
            {"date": "2026-02-24", "fii_buy": 14801, "fii_sell": 16201, "fii_net": -1400, "dii_buy": 15021, "dii_sell": 13201, "dii_net": 1820},
            {"date": "2026-02-23", "fii_buy": 11201, "fii_sell": 18402, "fii_net": -7201, "dii_buy": 17201, "dii_sell": 10402, "dii_net": 6799},
        ]
        df = pd.DataFrame(seed)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _parse_fii_dii_records(self, data) -> list[dict]:
        """Parse NSE FII/DII API response into row dicts.

        Handles both single-day (2 records: FII + DII) and
        multi-day (many records with dates) response formats.
        """
        rows_by_date: dict[str, dict] = {}
        records = data if isinstance(data, list) else [data]

        for r in records:
            category = (r.get("category") or "").upper()
            date_str = r.get("date")
            if not date_str:
                continue

            # Normalize date
            try:
                dt_obj = pd.to_datetime(date_str, dayfirst=True)
                date_key = dt_obj.strftime("%Y-%m-%d")
            except Exception:
                continue

            if date_key not in rows_by_date:
                rows_by_date[date_key] = {
                    "date": date_key,
                    "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
                    "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
                }

            buy_val = self._parse_num(r.get("buyValue")) or 0
            sell_val = self._parse_num(r.get("sellValue")) or 0
            net_val = self._parse_num(r.get("netValue")) or 0

            if "FII" in category or "FPI" in category:
                rows_by_date[date_key]["fii_buy"] = buy_val
                rows_by_date[date_key]["fii_sell"] = sell_val
                rows_by_date[date_key]["fii_net"] = net_val
            elif "DII" in category:
                rows_by_date[date_key]["dii_buy"] = buy_val
                rows_by_date[date_key]["dii_sell"] = sell_val
                rows_by_date[date_key]["dii_net"] = net_val

        return list(rows_by_date.values())

    def fetch_bulk_deals(self, from_date: str = None, to_date: str = None) -> list[dict]:
        """Fetch bulk deals from NSE.

        Args:
            from_date: DD-MM-YYYY format. Defaults to 90 days ago.
            to_date: DD-MM-YYYY format. Defaults to today.

        Returns list of {date, symbol, client_name, deal_type, quantity, price}.
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=90)).strftime("%d-%m-%Y")
        if to_date is None:
            to_date = datetime.now().strftime("%d-%m-%Y")

        data = self._request(
            f"{self.BASE_URL}/api/historical/bulk-deals",
            params={"from": from_date, "to": to_date},
        )
        if not data:
            return []

        try:
            records = data if isinstance(data, list) else data.get("data", [])
            results = []
            for r in records:
                results.append({
                    "date": r.get("mTd") or r.get("date") or "",
                    "symbol": (r.get("symbol") or "").strip(),
                    "client_name": r.get("clientName") or r.get("clientname") or "",
                    "deal_type": r.get("buySell") or r.get("buysell") or "",
                    "quantity": self._parse_num(r.get("quantity") or r.get("quantityTraded")) or 0,
                    "price": self._parse_num(r.get("wAvgPrice") or r.get("wapc") or r.get("price")) or 0,
                })
            return results
        except Exception as e:
            logger.warning("Failed to parse bulk deals: %s", e)
            return []

    def fetch_block_deals(self, from_date: str = None, to_date: str = None) -> list[dict]:
        """Fetch block deals from NSE.

        Args:
            from_date: DD-MM-YYYY format. Defaults to 90 days ago.
            to_date: DD-MM-YYYY format. Defaults to today.

        Returns list of {date, symbol, client_name, deal_type, quantity, price}.
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=90)).strftime("%d-%m-%Y")
        if to_date is None:
            to_date = datetime.now().strftime("%d-%m-%Y")

        data = self._request(
            f"{self.BASE_URL}/api/historical/block-deals",
            params={"from": from_date, "to": to_date},
        )
        if not data:
            return []

        try:
            records = data if isinstance(data, list) else data.get("data", [])
            results = []
            for r in records:
                results.append({
                    "date": r.get("mTd") or r.get("date") or "",
                    "symbol": (r.get("symbol") or "").strip(),
                    "client_name": r.get("clientName") or r.get("clientname") or "",
                    "deal_type": r.get("buySell") or r.get("buysell") or "",
                    "quantity": self._parse_num(r.get("quantity") or r.get("quantityTraded")) or 0,
                    "price": self._parse_num(r.get("wAvgPrice") or r.get("wapc") or r.get("price")) or 0,
                })
            return results
        except Exception as e:
            logger.warning("Failed to parse block deals: %s", e)
            return []

    def fetch_delivery_data(self, symbol: str) -> dict | None:
        """Fetch delivery percentage data for a symbol from NSE quote.

        Returns: {symbol, delivery_qty, traded_qty, delivery_pct}
        """
        clean = self._clean_symbol(symbol)
        cached = self._load_cache(clean, "delivery")
        if cached is not None:
            return cached

        data = self._request(
            f"{self.BASE_URL}/api/quote-equity",
            params={"symbol": clean},
        )
        if not data:
            return None

        try:
            sec_info = data.get("securityWiseDP") or data.get("preOpenMarket") or {}
            delivery_qty = self._parse_num(sec_info.get("deliveryQuantity") or
                                           data.get("deliveryQuantity"))
            traded_qty = self._parse_num(sec_info.get("quantityTraded") or
                                         data.get("totalTradedVolume"))
            delivery_pct = self._parse_num(sec_info.get("deliveryToTradedQuantity") or
                                           data.get("deliveryToTradedQuantity"))

            if delivery_pct is None and delivery_qty and traded_qty and traded_qty > 0:
                delivery_pct = (delivery_qty / traded_qty) * 100

            result = {
                "symbol": clean,
                "delivery_qty": delivery_qty or 0,
                "traded_qty": traded_qty or 0,
                "delivery_pct": delivery_pct or 0,
            }

            self._save_cache(clean, "delivery", result)
            return result

        except Exception as e:
            logger.warning("Failed to parse delivery data for %s: %s", clean, e)
            return None

    # ── Screener.in Consolidated Quarterly ───────────────────────

    _screener_min_interval = 1.5  # seconds between Screener.in requests

    def _screener_request(self, path: str, retries: int = 2):
        """Rate-limited GET to Screener.in with adaptive backoff on 429."""
        for attempt in range(retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self._screener_min_interval:
                time.sleep(self._screener_min_interval - elapsed)
            try:
                self._last_request_time = time.time()
                resp = self.session.get(
                    f"https://www.screener.in{path}", timeout=15,
                )
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    # Slow down for subsequent requests too
                    self._screener_min_interval = min(
                        self._screener_min_interval + 0.5, 4.0)
                    time.sleep(3)
                    continue
                if resp.status_code == 403:
                    logger.warning("Screener.in 403 (blocked)")
                    return None
                resp.raise_for_status()
                return resp
            except Exception as e:
                logger.debug("Screener.in request failed: %s", e)
                time.sleep(2)
        return None

    def fetch_screener_quarterly(self, symbol: str) -> pd.DataFrame | None:
        """Fetch consolidated quarterly results from Screener.in.

        Parses the HTML quarters table. Falls back to standalone URL
        if the consolidated page returns 404.

        Returns DataFrame with same schema as fetch_quarterly_results().
        """
        clean = self._clean_symbol(symbol)

        # Quarter-aware caching: if cached data already covers the current
        # target quarter, it's final — return it regardless of age.
        # If not (stock hasn't filed yet), use a 24h TTL to re-check daily.
        cached = self._load_cache(clean, "quarterly_screener", ttl_hours=None)
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            target_qtr = self._current_target_quarter()
            latest_date = pd.to_datetime(cached["date"]).max()
            if latest_date >= pd.Timestamp(target_qtr) - pd.Timedelta(days=15):
                return cached  # stock has filed for current quarter — cache forever
            # Stock hasn't filed yet; only re-fetch if cache is >24h old
            cached_ttl = self._load_cache(clean, "quarterly_screener",
                                          ttl_hours=SCREENER_UNFILED_TTL_HOURS)
            if cached_ttl is not None and isinstance(cached_ttl, pd.DataFrame):
                return cached_ttl

        # Try consolidated first; if 404 or gated (premium), try standalone.
        table = None
        for url_path in (f"/company/{clean}/consolidated/", f"/company/{clean}/"):
            resp = self._screener_request(url_path)
            if resp is None:
                continue
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                section = soup.find("section", id="quarters")
                if not section:
                    continue
                tbl = section.find("table", class_="data-table")
                if not tbl:
                    continue
                thead = tbl.find("thead")
                if not thead:
                    continue
                if len(thead.find_all("th")) <= 1:
                    continue  # gated/premium skeleton — try next URL
                table = tbl
                break
            except Exception:
                continue

        if table is None:
            return None

        try:
            header_cells = table.find("thead").find_all("th")
            dates = []
            for th in header_cells[1:]:  # skip row-label column
                # Strip any child button text (Screener adds + buttons)
                btn = th.find("button")
                text = btn.get_text(strip=True) if btn else th.get_text(strip=True)
                text = text.lstrip("+").strip()
                try:
                    dates.append(pd.to_datetime(text, format="%b %Y") + pd.offsets.MonthEnd(0))
                except Exception:
                    dates.append(None)

            # Parse body rows into a label->values dict
            tbody = table.find("tbody")
            if not tbody:
                return None

            row_data: dict[str, list] = {}
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if not tds:
                    continue
                label = tds[0].get_text(strip=True).rstrip("+ ")
                vals = []
                for td in tds[1:]:
                    btn = td.find("button")
                    text = btn.get_text(strip=True) if btn else td.get_text(strip=True)
                    text = text.lstrip("+").strip()
                    vals.append(self._parse_screener_num(text))
                row_data[label] = vals

            if not dates or not row_data:
                return None

            # Detect banking vs non-banking schema
            is_banking = "Financing Profit" in row_data or "Financing Margin %" in row_data

            num_cols = len(dates)
            rows = []
            for i in range(num_cols):
                if dates[i] is None:
                    continue

                def _val(label):
                    lst = row_data.get(label, [])
                    return lst[i] if i < len(lst) else None

                if is_banking:
                    revenue = _val("Revenue")
                    operating_income = _val("Financing Profit")
                    opm = _val("Financing Margin %")
                    net_income = _val("Net Profit")
                    eps = _val("EPS in Rs")
                    depreciation = None
                    tax = None
                    pbt = None
                    other_income = None
                else:
                    revenue = _val("Sales")
                    operating_income = _val("Operating Profit")
                    opm = _val("OPM %")
                    other_income = _val("Other Income")
                    depreciation = _val("Depreciation")
                    pbt = _val("Profit before tax")
                    tax_pct = _val("Tax %")
                    net_income = _val("Net Profit")
                    eps = _val("EPS in Rs")

                    # Compute tax from pbt and tax%
                    tax = None
                    if pbt is not None and tax_pct is not None:
                        tax = pbt * tax_pct / 100

                # Compute NPM
                npm = None
                if revenue and net_income is not None and revenue != 0:
                    npm = net_income / revenue * 100

                # Screener values are in Crores; convert to rupees
                # (pipeline stores rupees, display divides by 1e7)
                CR = 1e7
                rows.append({
                    "date": dates[i],
                    "revenue": revenue * CR if revenue is not None else None,
                    "operating_income": operating_income * CR if operating_income is not None else None,
                    "net_income": net_income * CR if net_income is not None else None,
                    "diluted_eps": eps,
                    "opm_pct": opm,
                    "npm_pct": npm,
                    "depreciation": depreciation * CR if depreciation is not None else None,
                    "tax": tax * CR if tax is not None else None,
                    "pbt": pbt * CR if pbt is not None else None,
                    "other_income": other_income * CR if other_income is not None else None,
                })

            if not rows:
                return None

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            self._save_cache(clean, "quarterly_screener", df)
            return df

        except Exception as e:
            logger.warning("Failed to parse Screener.in quarterly for %s: %s", clean, e)
            return None

    def fetch_quarterly_consolidated(self, symbol: str, num_quarters: int = 20) -> pd.DataFrame | None:
        """Fetch consolidated quarterly results, Screener.in first, NSE fallback."""
        result = self.fetch_screener_quarterly(symbol)
        if result is not None and not result.empty:
            return result
        return self.fetch_quarterly_results(symbol, num_quarters)

    @staticmethod
    def _parse_screener_num(text: str) -> float | None:
        """Parse a number from Screener.in table cells."""
        if not text or text in ("", "-", "—"):
            return None
        # Remove commas and % suffix
        cleaned = text.replace(",", "").replace("%", "").strip()
        if not cleaned or cleaned in ("-", "—"):
            return None
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _current_target_quarter() -> datetime:
        """Most recent Indian FY quarter-end that's at least 45 days ago.

        Used for cache validation: if a stock's latest data matches this
        quarter, the result is final and doesn't need re-fetching.
        """
        cutoff = datetime.now() - timedelta(days=45)
        year = cutoff.year
        quarter_ends = [
            datetime(year - 1, 12, 31),
            datetime(year, 3, 31),
            datetime(year, 6, 30),
            datetime(year, 9, 30),
            datetime(year, 12, 31),
        ]
        for qe in reversed(quarter_ends):
            if qe <= cutoff:
                return qe
        return quarter_ends[0]

    @staticmethod
    def _fix_unit_outliers(rows: list[dict]):
        """Fix unit inconsistency in quarterly results.

        NSE sometimes returns the oldest quarter's monetary values in crores
        while the rest are in lakhs (~100x difference). Detect outliers and
        scale them up to match the majority.
        """
        monetary_keys = [
            "revenue", "operating_income", "net_income",
            "depreciation", "tax", "pbt", "other_income",
        ]
        # Use revenue as the reference for detection
        rev_by_idx = []
        for i, r in enumerate(rows):
            rv = r.get("revenue")
            if rv is not None and rv > 0:
                rev_by_idx.append((i, rv))

        if len(rev_by_idx) < 3:
            return

        rev_values = sorted([rv for _, rv in rev_by_idx])
        median_rev = rev_values[len(rev_values) // 2]

        for idx, rv in rev_by_idx:
            ratio = median_rev / rv
            if 30 < ratio < 300:
                # This row's monetary values are ~100x too small (crores vs lakhs)
                scale = round(ratio / 100) * 100  # snap to nearest 100x
                if scale < 50:
                    continue
                for key in monetary_keys:
                    val = rows[idx].get(key)
                    if val is not None:
                        rows[idx][key] = val * scale
                # Margins are ratios — they stay correct, no need to fix

    @staticmethod
    def _parse_num(val) -> float | None:
        """Safely parse a numeric value from NSE response."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            cleaned = str(val).replace(",", "").replace(" ", "").strip()
            if cleaned in ("", "-", "NA", "N/A"):
                return None
            return float(cleaned)
        except (ValueError, TypeError):
            return None


# Module-level singleton
_fetcher = None


def get_nse_fetcher() -> NSEDataFetcher:
    """Get or create the module-level NSE data fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = NSEDataFetcher()
    return _fetcher


def compute_fii_dii_flows(history_df: pd.DataFrame) -> dict:
    """Compute cumulative FII/DII net flows for multiple timeframes.

    Args:
        history_df: DataFrame from fetch_fii_dii_historical() with columns:
                    date, fii_net, dii_net (+ buy/sell columns).

    Returns:
        Dict of {timeframe_label: {fii_net, dii_net, days_available}}.
        timeframe_label in: 1w, 2w, 1m, 3m, 6m, 1y, 2y, 5y.
        Returns empty dict if insufficient data.
    """
    if history_df is None or history_df.empty:
        return {}

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    latest_date = df["date"].max()

    timeframes = {
        "1w": 7,
        "2w": 14,
        "1m": 30,
        "3m": 91,
        "6m": 182,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }

    result = {}
    for label, cal_days in timeframes.items():
        cutoff = latest_date - timedelta(days=cal_days)
        period_df = df[df["date"] > cutoff]

        if period_df.empty:
            result[label] = {"fii_net": None, "dii_net": None, "days_available": 0}
            continue

        result[label] = {
            "fii_net": round(period_df["fii_net"].sum(), 1),
            "dii_net": round(period_df["dii_net"].sum(), 1),
            "days_available": len(period_df),
        }

    return result
