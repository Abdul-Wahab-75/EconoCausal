"""
Currency Conversion Engine for EconoCausal
Provides automatic currency detection, live/cached exchange rates,
and multi-currency conversion (INR, USD, EUR, GBP, CAD, AUD, JPY, AED, SGD, etc.).
"""

import urllib.request
import json
import time
from typing import Dict, Tuple, Optional

# Supported major currencies with their symbol and display label
SUPPORTED_CURRENCIES: Dict[str, Dict[str, str]] = {
    "INR": {"name": "Indian Rupee", "symbol": "₹", "country": "India"},
    "USD": {"name": "US Dollar", "symbol": "$", "country": "United States"},
    "EUR": {"name": "Euro", "symbol": "€", "country": "European Union"},
    "GBP": {"name": "British Pound", "symbol": "£", "country": "United Kingdom"},
    "AED": {"name": "UAE Dirham", "symbol": "د.إ", "country": "United Arab Emirates"},
    "CAD": {"name": "Canadian Dollar", "symbol": "CA$", "country": "Canada"},
    "AUD": {"name": "Australian Dollar", "symbol": "A$", "country": "Australia"},
    "JPY": {"name": "Japanese Yen", "symbol": "¥", "country": "Japan"},
    "SGD": {"name": "Singapore Dollar", "symbol": "S$", "country": "Singapore"},
}

# Reliable fallback rates relative to 1 INR (in case user is offline)
FALLBACK_RATES_BASE_INR: Dict[str, float] = {
    "INR": 1.0,
    "USD": 0.01054,    # 1 USD ≈ 94.88 INR (or ~85-95 standard)
    "EUR": 0.00941,    # 1 EUR ≈ 106.27 INR
    "GBP": 0.00788,    # 1 GBP ≈ 126.90 INR
    "AED": 0.03870,    # 1 AED ≈ 25.84 INR
    "CAD": 0.01420,    # 1 CAD ≈ 70.42 INR
    "AUD": 0.01580,    # 1 AUD ≈ 63.29 INR
    "JPY": 1.58000,    # 1 JPY ≈ 0.63 INR
    "SGD": 0.01390,    # 1 SGD ≈ 71.94 INR
}

_RATES_CACHE: Optional[Dict[str, float]] = None
_CACHE_TIMESTAMP: float = 0.0
_CACHE_TTL_SECONDS: float = 3600.0  # 1 hour cache


def get_live_rates(base_currency: str = "INR") -> Dict[str, float]:
    """
    Fetches real-time exchange rates using the standard open exchange rates endpoint.
    Falls back smoothly to robust local fallback rates if offline.
    """
    global _RATES_CACHE, _CACHE_TIMESTAMP

    now = time.time()
    if _RATES_CACHE is not None and (now - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS:
        return _RATES_CACHE

    try:
        url = f"https://open.er-api.com/v6/latest/{base_currency}"
        req = urllib.request.Request(url, headers={"User-Agent": "EconoCausal/1.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("result") == "success" and "rates" in data:
                rates = data["rates"]
                _RATES_CACHE = rates
                _CACHE_TIMESTAMP = now
                return rates
    except Exception:
        pass

    # Return local fallback rates
    _RATES_CACHE = FALLBACK_RATES_BASE_INR
    _CACHE_TIMESTAMP = now
    return FALLBACK_RATES_BASE_INR


def convert_currency(
    amount: float,
    from_curr: str = "INR",
    to_curr: str = "INR"
) -> float:
    """
    Converts amount from from_curr to to_curr using live or cached exchange rates.
    """
    if from_curr == to_curr or amount == 0:
        return float(amount)

    rates = get_live_rates("INR")

    # Rate of from_curr per 1 INR: rates[from_curr]
    # Rate of to_curr per 1 INR: rates[to_curr]
    rate_from = rates.get(from_curr, FALLBACK_RATES_BASE_INR.get(from_curr, 1.0))
    rate_to = rates.get(to_curr, FALLBACK_RATES_BASE_INR.get(to_curr, 1.0))

    # Amount in INR = amount / rate_from
    amount_in_inr = amount / rate_from if rate_from > 0 else amount
    amount_converted = amount_in_inr * rate_to
    return float(amount_converted)


def get_currency_meta(currency_code: str) -> Dict[str, str]:
    """Returns currency symbol and display metadata."""
    return SUPPORTED_CURRENCIES.get(
        currency_code,
        {"name": currency_code, "symbol": currency_code + " ", "country": "International"}
    )
