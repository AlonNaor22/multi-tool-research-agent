"""Currency converter tool — real-time exchange rates via frankfurter.app."""

import re
import asyncio
import aiohttp
from typing import Optional, Dict

from langchain_core.tools import tool
from src.utils import async_retry_on_error, async_fetch, safe_tool_call, require_input
from src.constants import DEFAULT_HTTP_TIMEOUT


# ─── Module overview ───────────────────────────────────────────────
# Converts between currencies using real-time exchange rates from
# the frankfurter.app API. Parses natural-language conversion queries.
# ───────────────────────────────────────────────────────────────────

# Common currency codes and their aliases
CURRENCY_ALIASES = {
    # Major currencies
    "dollar": "USD", "dollars": "USD", "usd": "USD", "$": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR", "\u20ac": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "sterling": "GBP", "\u00a3": "GBP",
    "yen": "JPY", "jpy": "JPY", "\u00a5": "JPY",
    "yuan": "CNY", "cny": "CNY", "rmb": "CNY", "renminbi": "CNY",
    "franc": "CHF", "francs": "CHF", "chf": "CHF",
    "rupee": "INR", "rupees": "INR", "inr": "INR", "\u20b9": "INR",
    "ruble": "RUB", "rubles": "RUB", "rub": "RUB", "\u20bd": "RUB",

    # Other common currencies
    "cad": "CAD", "canadian": "CAD",
    "aud": "AUD", "australian": "AUD",
    "nzd": "NZD",
    "krw": "KRW", "won": "KRW", "\u20a9": "KRW",
    "brl": "BRL", "real": "BRL", "reais": "BRL",
    "mxn": "MXN", "peso": "MXN", "pesos": "MXN",
    "sgd": "SGD", "singapore": "SGD",
    "hkd": "HKD", "hong kong": "HKD",
    "sek": "SEK", "krona": "SEK",
    "nok": "NOK",
    "dkk": "DKK",
    "pln": "PLN", "zloty": "PLN",
    "thb": "THB", "baht": "THB",
    "idr": "IDR", "rupiah": "IDR",
    "try": "TRY", "lira": "TRY",
    "zar": "ZAR", "rand": "ZAR",
    "ils": "ILS", "shekel": "ILS", "shekels": "ILS", "\u20aa": "ILS",
    "aed": "AED", "dirham": "AED",
    "sar": "SAR", "riyal": "SAR",
    "btc": "BTC", "bitcoin": "BTC", "\u20bf": "BTC",
}

# List of valid ISO currency codes (most common ones)
VALID_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "INR", "RUB", "CAD", "AUD",
    "NZD", "KRW", "BRL", "MXN", "SGD", "HKD", "SEK", "NOK", "DKK", "PLN",
    "THB", "IDR", "TRY", "ZAR", "ILS", "AED", "SAR", "EGP", "PHP", "VND",
    "MYR", "CZK", "HUF", "RON", "BGN", "HRK", "ISK", "CLP", "COP", "PEN",
    "ARS", "TWD", "PKR", "BDT", "NGN", "KES", "UAH", "QAR", "KWD", "BHD",
}


# Takes (currency). Resolves aliases and names to a 3-letter ISO code.
# Returns the uppercase ISO code or None if unrecognized.
def normalize_currency(currency: str) -> Optional[str]:
    """Convert currency name/alias to ISO code."""
    currency_clean = currency.strip().lower()

    # Check aliases first
    if currency_clean in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[currency_clean]

    # Check if it's already a valid ISO code
    currency_upper = currency_clean.upper()
    if currency_upper in VALID_CURRENCIES:
        return currency_upper

    return None


# Takes (from_currency, to_currency). Calls the frankfurter.app API.
# Returns a dict containing the exchange rate and date.
@async_retry_on_error(max_retries=2, delay=1.0)
async def get_exchange_rate(from_currency: str, to_currency: str) -> Dict:
    """Fetch exchange rate from frankfurter.app API."""
    base_url = "https://api.frankfurter.app/latest"

    params = {
        "from": from_currency,
        "to": to_currency,
    }

    return await async_fetch(base_url, params=params, timeout=DEFAULT_HTTP_TIMEOUT)


# Takes (amount, from_currency, to_currency). Normalizes codes and fetches rate.
# Returns a formatted string with the converted amount and exchange rate.
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another."""
    # Normalize currency codes
    from_code = normalize_currency(from_currency)
    to_code = normalize_currency(to_currency)

    if not from_code:
        return f"Error: Unknown currency '{from_currency}'. Use ISO codes (USD, EUR, GBP) or common names (dollar, euro, pound)."
    if not to_code:
        return f"Error: Unknown currency '{to_currency}'. Use ISO codes (USD, EUR, GBP) or common names (dollar, euro, pound)."

    if from_code == to_code:
        return f"{amount} {from_code} = {amount} {to_code} (same currency)"

    try:
        data = await get_exchange_rate(from_code, to_code)

        if "rates" not in data or to_code not in data["rates"]:
            return f"Error: Could not get exchange rate for {from_code} to {to_code}"

        rate = data["rates"][to_code]
        result = amount * rate

        # Format nicely based on magnitude
        if result >= 1000:
            result_str = f"{result:,.2f}"
        elif result >= 1:
            result_str = f"{result:.2f}"
        else:
            result_str = f"{result:.4f}"

        return (
            f"{amount:,.2f} {from_code} = {result_str} {to_code}\n"
            f"Exchange rate: 1 {from_code} = {rate:.4f} {to_code}\n"
            f"(Rate as of {data.get('date', 'today')})"
        )

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return f"Error fetching exchange rate: {str(e)}"
    except Exception as e:
        return f"Error converting currency: {str(e)}"


# Takes (input_str). Parses natural-language conversion requests like "100 USD to EUR".
# Returns the converted amount, exchange rate, and date.
@safe_tool_call("converting currency")
async def currency_converter(input_str: str) -> str:
    """Convert between currencies using real-time exchange rates.

    FORMAT: '100 USD to EUR', 'convert 50 dollars to pounds', 'rate EUR GBP'

    SUPPORTED: All major world currencies (USD, EUR, GBP, JPY, CNY, etc.)

    Accepts currency codes (USD) or names (dollar, euro, pound, yen).

    Returns current exchange rate and converted amount."""
    err = require_input(input_str, "conversion request")
    if err:
        return err
    input_str = input_str.strip()

    # Command: help
    if input_str.lower() in ("help", "?", "list"):
        return _get_help()

    # Command: rate only (e.g., "rate USD EUR")
    rate_match = re.match(
        r'rate\s+([a-zA-Z$\u20ac\u00a3\u00a5\u20b9\u20bd\u20a9\u20aa\u20bf]+)\s+([a-zA-Z$\u20ac\u00a3\u00a5\u20b9\u20bd\u20a9\u20aa\u20bf]+)',
        input_str,
        re.IGNORECASE
    )
    if rate_match:
        from_curr = rate_match.group(1)
        to_curr = rate_match.group(2)
        return await convert_currency(1.0, from_curr, to_curr)

    # Parse conversion pattern: "[convert] AMOUNT FROM_CURRENCY to TO_CURRENCY"
    conversion_match = re.match(
        r'(?:convert\s+)?(-?[\d,]+\.?\d*)\s*([a-zA-Z$\u20ac\u00a3\u00a5\u20b9\u20bd\u20a9\u20aa\u20bf]+)\s+to\s+([a-zA-Z$\u20ac\u00a3\u00a5\u20b9\u20bd\u20a9\u20aa\u20bf]+)',
        input_str,
        re.IGNORECASE
    )

    if conversion_match:
        try:
            # Handle numbers with commas
            amount_str = conversion_match.group(1).replace(",", "")
            amount = float(amount_str)
            from_currency = conversion_match.group(2)
            to_currency = conversion_match.group(3)
            return await convert_currency(amount, from_currency, to_currency)
        except ValueError as e:
            return f"Error parsing amount: {str(e)}"

    return (
        "Error: Could not parse conversion request.\n"
        "Use format: '100 USD to EUR', 'convert 50 dollars to pounds', or 'rate EUR GBP'"
    )


# Returns help text listing supported formats and currency codes.
def _get_help() -> str:
    """Return help text."""
    return """Currency Converter Help:

FORMAT:
  100 USD to EUR
  convert 50 dollars to euros
  1000 yen to dollars
  rate EUR GBP (just shows exchange rate)

SUPPORTED CURRENCIES:

Major: USD ($), EUR (\u20ac), GBP (\u00a3), JPY (\u00a5), CNY, CHF, INR (\u20b9), RUB

Americas: CAD, BRL, MXN, ARS, CLP, COP, PEN

Asia-Pacific: AUD, NZD, SGD, HKD, KRW (\u20a9), TWD, THB, IDR, MYR, PHP, VND

Europe: SEK, NOK, DKK, PLN, CZK, HUF, RON, CHF, TRY

Middle East/Africa: ILS (\u20aa), AED, SAR, QAR, ZAR, EGP, NGN, KES

You can use currency names (dollar, euro, pound, yen) or ISO codes (USD, EUR, GBP, JPY).

Note: Exchange rates are fetched in real-time from frankfurter.app"""


currency_tool = tool(currency_converter)
