"""Unit converter tool for the research agent.

A dedicated tool for converting between different units of measurement.

Supported categories:
- Length (m, km, mi, ft, in, etc.)
- Weight/Mass (kg, lb, oz, etc.)
- Volume (l, gal, cup, etc.)
- Time (s, min, h, day, etc.)
- Speed (m/s, km/h, mph, etc.)
- Area (sqm, sqft, acre, etc.)
- Temperature (C, F, K)
- Data (bytes, KB, MB, GB, etc.)
"""

import re
from typing import Optional, Tuple, Dict
from langchain_core.tools import Tool


# ============================================================================
# UNIT CONVERSION SYSTEM
# ============================================================================

# Conversion factors to base units
# Length: base unit = meters
LENGTH_TO_METERS = {
    "m": 1, "meter": 1, "meters": 1,
    "km": 1000, "kilometer": 1000, "kilometers": 1000,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "nm": 1852, "nautical_mile": 1852, "nautical_miles": 1852,
}

# Weight/Mass: base unit = kilograms
WEIGHT_TO_KG = {
    "kg": 1, "kilogram": 1, "kilograms": 1,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 0.000001, "milligram": 0.000001, "milligrams": 0.000001,
    "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    "ton": 1000, "tons": 1000, "tonne": 1000, "tonnes": 1000,
    "st": 6.35029, "stone": 6.35029, "stones": 6.35029,
}

# Volume: base unit = liters
VOLUME_TO_LITERS = {
    "l": 1, "liter": 1, "liters": 1, "litre": 1, "litres": 1,
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
    "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
    "cup": 0.236588, "cups": 0.236588,
    "floz": 0.0295735, "fl_oz": 0.0295735, "fluid_ounce": 0.0295735,
    "tbsp": 0.0147868, "tablespoon": 0.0147868, "tablespoons": 0.0147868,
    "tsp": 0.00492892, "teaspoon": 0.00492892, "teaspoons": 0.00492892,
    "m3": 1000, "cubic_meter": 1000, "cubic_meters": 1000,
}

# Time: base unit = seconds
TIME_TO_SECONDS = {
    "s": 1, "sec": 1, "second": 1, "seconds": 1,
    "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "min": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "wk": 604800, "week": 604800, "weeks": 604800,
    "mo": 2592000, "month": 2592000, "months": 2592000,  # ~30 days
    "yr": 31536000, "year": 31536000, "years": 31536000,  # 365 days
}

# Speed: base unit = meters per second
SPEED_TO_MPS = {
    "mps": 1, "m/s": 1,
    "kph": 0.277778, "km/h": 0.277778, "kmh": 0.277778,
    "mph": 0.44704,
    "knot": 0.514444, "knots": 0.514444,
    "fps": 0.3048, "ft/s": 0.3048,
}

# Area: base unit = square meters
AREA_TO_SQM = {
    "sqm": 1, "m2": 1, "sq_m": 1, "square_meter": 1, "square_meters": 1,
    "sqkm": 1000000, "km2": 1000000, "square_kilometer": 1000000,
    "sqft": 0.092903, "ft2": 0.092903, "square_foot": 0.092903, "square_feet": 0.092903,
    "sqmi": 2589988, "mi2": 2589988, "square_mile": 2589988, "square_miles": 2589988,
    "acre": 4046.86, "acres": 4046.86,
    "hectare": 10000, "hectares": 10000, "ha": 10000,
}

# Data: base unit = bytes
DATA_TO_BYTES = {
    "b": 1, "byte": 1, "bytes": 1,
    "kb": 1024, "kilobyte": 1024, "kilobytes": 1024,
    "mb": 1048576, "megabyte": 1048576, "megabytes": 1048576,
    "gb": 1073741824, "gigabyte": 1073741824, "gigabytes": 1073741824,
    "tb": 1099511627776, "terabyte": 1099511627776, "terabytes": 1099511627776,
    "pb": 1125899906842624, "petabyte": 1125899906842624,
}

# Map unit types to their conversion dicts
UNIT_CATEGORIES = {
    "length": LENGTH_TO_METERS,
    "weight": WEIGHT_TO_KG,
    "mass": WEIGHT_TO_KG,
    "volume": VOLUME_TO_LITERS,
    "time": TIME_TO_SECONDS,
    "speed": SPEED_TO_MPS,
    "area": AREA_TO_SQM,
    "data": DATA_TO_BYTES,
}


def find_unit_category(unit: str) -> Optional[Tuple[str, Dict]]:
    """Find which category a unit belongs to and return (category_name, conversion_dict)."""
    unit_lower = unit.lower().replace(" ", "_")
    for category, conversions in UNIT_CATEGORIES.items():
        if unit_lower in conversions:
            return category, conversions
    return None


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Handle temperature conversions separately (not linear scaling)."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Normalize unit names
    temp_aliases = {
        "c": "celsius", "celsius": "celsius",
        "f": "fahrenheit", "fahrenheit": "fahrenheit",
        "k": "kelvin", "kelvin": "kelvin",
    }

    from_unit = temp_aliases.get(from_unit)
    to_unit = temp_aliases.get(to_unit)

    if not from_unit or not to_unit:
        raise ValueError("Unknown temperature unit")

    # Convert to Celsius first
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Celsius to target
    if to_unit == "celsius":
        return celsius
    elif to_unit == "fahrenheit":
        return celsius * 9/5 + 32
    elif to_unit == "kelvin":
        return celsius + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a value from one unit to another.

    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., 'km', 'miles', 'kg')
        to_unit: Target unit (e.g., 'm', 'feet', 'pounds')

    Returns:
        Formatted string with the conversion result
    """
    from_unit_clean = from_unit.lower().replace(" ", "_")
    to_unit_clean = to_unit.lower().replace(" ", "_")

    # Check for temperature (special case)
    temp_units = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}
    if from_unit_clean in temp_units and to_unit_clean in temp_units:
        result = convert_temperature(value, from_unit_clean, to_unit_clean)
        return f"{value} {from_unit} = {result:.4g} {to_unit}"

    # Find the category for source unit
    from_info = find_unit_category(from_unit_clean)
    to_info = find_unit_category(to_unit_clean)

    if not from_info:
        return f"Error: Unknown unit '{from_unit}'"
    if not to_info:
        return f"Error: Unknown unit '{to_unit}'"

    from_category, from_conversions = from_info
    to_category, to_conversions = to_info

    if from_category != to_category:
        return f"Error: Cannot convert {from_category} ({from_unit}) to {to_category} ({to_unit})"

    # Convert: value -> base unit -> target unit
    base_value = value * from_conversions[from_unit_clean]
    result = base_value / to_conversions[to_unit_clean]

    return f"{value} {from_unit} = {result:.6g} {to_unit}"


def convert(input_str: str) -> str:
    """
    Parse and execute a unit conversion request.

    Supports formats:
    - "5 km to miles"
    - "convert 5 km to miles"
    - "100 fahrenheit to celsius"
    - "10 pounds to kg"

    Args:
        input_str: The conversion request

    Returns:
        Result string or error message
    """
    input_str = input_str.strip()

    # Handle empty input
    if not input_str:
        return "Error: Empty conversion request"

    # Command: help
    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Parse conversion pattern: "[convert] VALUE FROM_UNIT to TO_UNIT"
    conversion_match = re.match(
        r'(?:convert\s+)?(-?[\d.]+)\s*([a-zA-Z_/]+)\s+to\s+([a-zA-Z_/]+)',
        input_str,
        re.IGNORECASE
    )

    if conversion_match:
        try:
            value = float(conversion_match.group(1))
            from_unit = conversion_match.group(2)
            to_unit = conversion_match.group(3)
            return convert_units(value, from_unit, to_unit)
        except ValueError as e:
            return f"Error: {str(e)}"

    return (
        "Error: Could not parse conversion request. "
        "Use format: '5 km to miles' or 'convert 100 F to C'"
    )


def _get_help() -> str:
    """Return help text."""
    return """Unit Converter Help:

FORMAT:
  5 km to miles
  convert 100 F to C
  10 pounds to kg

SUPPORTED UNITS:

Length: m, km, cm, mm, mi (miles), yd, ft, in, nm (nautical miles)

Weight: kg, g, mg, lb (pounds), oz (ounces), ton, stone

Volume: l, ml, gal (gallons), qt (quarts), pt (pints), cup,
        floz (fluid oz), tbsp, tsp, m3 (cubic meters)

Time: s, ms, min, h (hours), d (days), wk (weeks), mo (months), yr (years)

Speed: m/s, km/h, mph, knots, ft/s

Area: sqm, sqft, sqmi, acre, hectare, km2, m2

Temperature: C (Celsius), F (Fahrenheit), K (Kelvin)

Data: b (bytes), kb, mb, gb, tb, pb"""


# Create the LangChain Tool wrapper
unit_converter_tool = Tool(
    name="unit_converter",
    func=convert,
    description=(
        "Convert between different units of measurement. "
        "\n\nFORMAT: '5 km to miles', 'convert 100 F to C', '10 pounds to kg'"
        "\n\nSUPPORTED UNITS:"
        "\n- Length: m, km, cm, mm, mi, ft, in, yd"
        "\n- Weight: kg, g, lb, oz, ton"
        "\n- Volume: l, ml, gal, cup, tbsp, tsp"
        "\n- Time: s, min, h, day, week, year"
        "\n- Speed: m/s, km/h, mph, knots"
        "\n- Area: sqm, sqft, acre, hectare"
        "\n- Temperature: C, F, K"
        "\n- Data: bytes, kb, mb, gb, tb"
    )
)
