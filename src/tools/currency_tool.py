"""Currency converter tool for the research agent.

Converts between currencies using real-time exchange rates from a free API.

Uses the free exchangerate-api.com or falls back to frankfurter.app API.
"""

import re
import requests
from typing import Optional, Dict
from langchain_core.tools import Tool

from src.utils import retry_on_error


# Common currency codes and their aliases
CURRENCY_ALIASES = {
    # Major currencies
    "dollar": "USD", "dollars": "USD", "usd": "USD", "$": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR", "€": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "sterling": "GBP", "£": "GBP",
    "yen": "JPY", "jpy": "JPY", "¥": "JPY",
    "yuan": "CNY", "cny": "CNY", "rmb": "CNY", "renminbi": "CNY",
    "franc": "CHF", "francs": "CHF", "chf": "CHF",
    "rupee": "INR", "rupees": "INR", "inr": "INR", "₹": "INR",
    "ruble": "RUB", "rubles": "RUB", "rub": "RUB", "₽": "RUB",

    # Other common currencies
    "cad": "CAD", "canadian": "CAD",
    "aud": "AUD", "australian": "AUD",
    "nzd": "NZD",
    "krw": "KRW", "won": "KRW", "₩": "KRW",
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
    "ils": "ILS", "shekel": "ILS", "shekels": "ILS", "₪": "ILS",
    "aed": "AED", "dirham": "AED",
    "sar": "SAR", "riyal": "SAR",
    "btc": "BTC", "bitcoin": "BTC", "₿": "BTC",
}

# List of valid ISO currency codes (most common ones)
VALID_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "INR", "RUB", "CAD", "AUD",
    "NZD", "KRW", "BRL", "MXN", "SGD", "HKD", "SEK", "NOK", "DKK", "PLN",
    "THB", "IDR", "TRY", "ZAR", "ILS", "AED", "SAR", "EGP", "PHP", "VND",
    "MYR", "CZK", "HUF", "RON", "BGN", "HRK", "ISK", "CLP", "COP", "PEN",
    "ARS", "TWD", "PKR", "BDT", "NGN", "KES", "UAH", "QAR", "KWD", "BHD",
}


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


@retry_on_error(max_retries=2, delay=1.0)
def get_exchange_rate(from_currency: str, to_currency: str) -> Dict:
    """
    Fetch exchange rate from API.

    Uses frankfurter.app (free, no API key required).
    """
    base_url = "https://api.frankfurter.app/latest"

    params = {
        "from": from_currency,
        "to": to_currency,
    }

    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another.

    Args:
        amount: The amount to convert
        from_currency: Source currency code
        to_currency: Target currency code

    Returns:
        Formatted string with conversion result
    """
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
        data = get_exchange_rate(from_code, to_code)

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

    except requests.exceptions.RequestException as e:
        return f"Error fetching exchange rate: {str(e)}"
    except Exception as e:
        return f"Error converting currency: {str(e)}"


def currency_convert(input_str: str) -> str:
    """
    Parse and execute a currency conversion request.

    Supports formats:
    - "100 USD to EUR"
    - "convert 50 dollars to euros"
    - "500 yen to dollars"
    - "rate USD EUR" (just get the rate)

    Args:
        input_str: The conversion request

    Returns:
        Result string or error message
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: Empty conversion request"

    # Command: help
    if input_str.lower() in ("help", "?", "list"):
        return _get_help()

    # Command: rate only (e.g., "rate USD EUR")
    rate_match = re.match(
        r'rate\s+([a-zA-Z$€£¥₹₽₩₪₿]+)\s+([a-zA-Z$€£¥₹₽₩₪₿]+)',
        input_str,
        re.IGNORECASE
    )
    if rate_match:
        from_curr = rate_match.group(1)
        to_curr = rate_match.group(2)
        return convert_currency(1.0, from_curr, to_curr)

    # Parse conversion pattern: "[convert] AMOUNT FROM_CURRENCY to TO_CURRENCY"
    conversion_match = re.match(
        r'(?:convert\s+)?(-?[\d,]+\.?\d*)\s*([a-zA-Z$€£¥₹₽₩₪₿]+)\s+to\s+([a-zA-Z$€£¥₹₽₩₪₿]+)',
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
            return convert_currency(amount, from_currency, to_currency)
        except ValueError as e:
            return f"Error parsing amount: {str(e)}"

    return (
        "Error: Could not parse conversion request.\n"
        "Use format: '100 USD to EUR', 'convert 50 dollars to pounds', or 'rate EUR GBP'"
    )


def _get_help() -> str:
    """Return help text."""
    return """Currency Converter Help:

FORMAT:
  100 USD to EUR
  convert 50 dollars to euros
  1000 yen to dollars
  rate EUR GBP (just shows exchange rate)

SUPPORTED CURRENCIES:

Major: USD ($), EUR (€), GBP (£), JPY (¥), CNY, CHF, INR (₹), RUB

Americas: CAD, BRL, MXN, ARS, CLP, COP, PEN

Asia-Pacific: AUD, NZD, SGD, HKD, KRW (₩), TWD, THB, IDR, MYR, PHP, VND

Europe: SEK, NOK, DKK, PLN, CZK, HUF, RON, CHF, TRY

Middle East/Africa: ILS (₪), AED, SAR, QAR, ZAR, EGP, NGN, KES

You can use currency names (dollar, euro, pound, yen) or ISO codes (USD, EUR, GBP, JPY).

Note: Exchange rates are fetched in real-time from frankfurter.app"""


# Create the LangChain Tool wrapper
currency_tool = Tool(
    name="currency_converter",
    func=currency_convert,
    description=(
        "Convert between currencies using real-time exchange rates. "
        "\n\nFORMAT: '100 USD to EUR', 'convert 50 dollars to pounds', 'rate EUR GBP'"
        "\n\nSUPPORTED: All major world currencies (USD, EUR, GBP, JPY, CNY, etc.)"
        "\n\nAccepts currency codes (USD) or names (dollar, euro, pound, yen)."
        "\n\nReturns current exchange rate and converted amount."
    )
)
