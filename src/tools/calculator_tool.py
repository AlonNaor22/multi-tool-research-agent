"""Calculator tool for the research agent.

Enhanced calculator with support for:
- Basic arithmetic and math functions
- Variables (store and reuse values)
- Unit conversions (length, weight, temperature, etc.)

Uses a safe evaluation approach to prevent code injection attacks.
"""

import math
import re
from typing import Dict, Any, Optional
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


def find_unit_category(unit: str) -> Optional[tuple]:
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

    # Convert from Celsius to target
    if to_unit == "celsius":
        return celsius
    elif to_unit == "fahrenheit":
        return celsius * 9/5 + 32
    elif to_unit == "kelvin":
        return celsius + 273.15


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


# ============================================================================
# SAFE MATH EVALUATION
# ============================================================================

# Allowed math functions (safe subset)
SAFE_MATH_FUNCTIONS = {
    # Basic
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,

    # From math module
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1/3),  # Cube root
    "exp": math.exp,
    "log": math.log,      # Natural log
    "log10": math.log10,  # Base 10 log
    "log2": math.log2,    # Base 2 log

    # Trigonometric (radians)
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,

    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,

    # Angular conversion
    "degrees": math.degrees,  # radians to degrees
    "radians": math.radians,  # degrees to radians

    # Special functions
    "factorial": math.factorial,
    "gcd": math.gcd,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,

    # Constants (as functions for easy access)
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "tau": lambda: math.tau,
    "inf": lambda: math.inf,
}

# Allowed operators and their precedence
SAFE_OPERATORS = {
    "+", "-", "*", "/", "//", "%", "**",
    "(", ")", ",", ".",
    "<", ">", "<=", ">=", "==", "!=",
    "&", "|", "^", "~",
}


class AdvancedCalculator:
    """
    A calculator with variable storage and safe expression evaluation.

    Features:
    - Store and recall variables
    - Math functions (sqrt, sin, cos, factorial, etc.)
    - Unit conversions
    - Safe evaluation (no code injection)
    """

    def __init__(self):
        """Initialize calculator with empty variable storage."""
        self.variables: Dict[str, float] = {}

    def set_variable(self, name: str, value: float) -> str:
        """Store a variable."""
        # Validate variable name (alphanumeric, starts with letter)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return f"Error: Invalid variable name '{name}'. Use letters, numbers, underscore (start with letter)."

        # Don't allow overwriting function names
        if name.lower() in SAFE_MATH_FUNCTIONS:
            return f"Error: '{name}' is a reserved function name."

        self.variables[name] = float(value)
        return f"Stored: {name} = {value}"

    def get_variable(self, name: str) -> Optional[float]:
        """Retrieve a variable value."""
        return self.variables.get(name)

    def list_variables(self) -> str:
        """List all stored variables."""
        if not self.variables:
            return "No variables stored."
        lines = ["Stored variables:"]
        for name, value in self.variables.items():
            lines.append(f"  {name} = {value}")
        return "\n".join(lines)

    def clear_variables(self) -> str:
        """Clear all stored variables."""
        self.variables.clear()
        return "All variables cleared."

    def safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.

        This uses Python's eval() but with a restricted namespace
        containing only safe math functions and stored variables.
        """
        # Build the safe namespace
        safe_namespace = {}

        # Add math functions
        for name, func in SAFE_MATH_FUNCTIONS.items():
            # For constants like pi(), make them accessible as values
            if name in ("pi", "e", "tau", "inf"):
                safe_namespace[name] = func()
            else:
                safe_namespace[name] = func

        # Add stored variables
        safe_namespace.update(self.variables)

        # Validate expression doesn't contain dangerous patterns
        dangerous_patterns = [
            "__", "import", "exec", "eval", "compile", "open", "file",
            "input", "raw_input", "globals", "locals", "vars", "dir",
            "getattr", "setattr", "delattr", "hasattr",
            "type", "isinstance", "issubclass", "callable",
            "classmethod", "staticmethod", "property",
        ]

        expr_lower = expression.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                raise ValueError(f"Expression contains forbidden pattern: {pattern}")

        # Evaluate with restricted namespace
        # __builtins__ = {} prevents access to built-in functions
        result = eval(expression, {"__builtins__": {}}, safe_namespace)

        return float(result)

    def calculate(self, input_str: str) -> str:
        """
        Process a calculation request.

        Supports:
        - Simple expressions: "2 + 2"
        - Math functions: "sqrt(16)", "sin(pi/2)", "factorial(5)"
        - Variable assignment: "x = 10" or "set x = 10"
        - Variable usage: "x * 2" (after setting x)
        - Unit conversion: "convert 5 km to miles" or "5 km to miles"
        - List variables: "variables" or "vars"
        - Clear variables: "clear" or "clear variables"

        Args:
            input_str: The calculation request

        Returns:
            Result string or error message
        """
        input_str = input_str.strip()

        # Handle empty input
        if not input_str:
            return "Error: Empty expression"

        # Command: list variables
        if input_str.lower() in ("variables", "vars", "list"):
            return self.list_variables()

        # Command: clear variables
        if input_str.lower() in ("clear", "clear variables", "clear vars"):
            return self.clear_variables()

        # Command: help
        if input_str.lower() in ("help", "?"):
            return self._get_help()

        # Check for unit conversion: "convert 5 km to miles" or "5 km to miles"
        conversion_match = re.match(
            r'(?:convert\s+)?(-?[\d.]+)\s*([a-zA-Z_/]+)\s+to\s+([a-zA-Z_/]+)',
            input_str,
            re.IGNORECASE
        )
        if conversion_match:
            value = float(conversion_match.group(1))
            from_unit = conversion_match.group(2)
            to_unit = conversion_match.group(3)
            return convert_units(value, from_unit, to_unit)

        # Check for variable assignment: "x = 10" or "set x = 10"
        assignment_match = re.match(
            r'(?:set\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)',
            input_str
        )
        if assignment_match:
            var_name = assignment_match.group(1)
            expr = assignment_match.group(2)
            try:
                value = self.safe_eval(expr)
                return self.set_variable(var_name, value)
            except Exception as e:
                return f"Error evaluating expression: {str(e)}"

        # Otherwise, evaluate as expression
        try:
            result = self.safe_eval(input_str)

            # Format result nicely
            if result == int(result):
                return str(int(result))
            else:
                # Use general format, but limit decimal places
                return f"{result:.10g}"

        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: {str(e)}"
        except SyntaxError:
            return f"Error: Invalid syntax in expression"
        except NameError as e:
            return f"Error: Unknown variable or function - {str(e)}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    def _get_help(self) -> str:
        """Return help text."""
        return """Calculator Help:

BASIC MATH:
  2 + 2, 10 * 5, 100 / 4, 2 ** 10, 17 % 5

MATH FUNCTIONS:
  sqrt(16), cbrt(27), abs(-5), round(3.7)
  sin(x), cos(x), tan(x), asin(x), acos(x), atan(x)
  log(x), log10(x), log2(x), exp(x)
  factorial(5), gcd(12, 8), ceil(3.2), floor(3.8)
  degrees(pi), radians(180)
  min(1,2,3), max(1,2,3), sum([1,2,3])

CONSTANTS:
  pi = 3.14159..., e = 2.71828..., tau = 6.28318...

VARIABLES:
  x = 10        (store a value)
  x * 2         (use stored value)
  variables     (list all variables)
  clear         (clear all variables)

UNIT CONVERSION:
  5 km to miles
  100 fahrenheit to celsius
  convert 10 pounds to kg

SUPPORTED UNITS:
  Length: m, km, cm, mm, mi, yd, ft, in
  Weight: kg, g, mg, lb, oz, ton
  Volume: l, ml, gal, qt, pt, cup, tbsp, tsp
  Time: s, min, h, day, week, month, year
  Speed: m/s, km/h, mph, knots
  Area: sqm, sqft, acre, hectare
  Temperature: C, F, K
  Data: b, kb, mb, gb, tb"""


# Create a global calculator instance (persists across calls)
_calculator = AdvancedCalculator()


def calculate(expression: str) -> str:
    """
    Entry point for the calculator tool.

    This function is called by the LangChain agent.
    """
    return _calculator.calculate(expression)


# Create the LangChain Tool wrapper
calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description=(
        "Perform mathematical calculations, use variables, and convert units. "
        "\n\nBASIC MATH: '2 + 2', '100 * 0.15', '2 ** 10' (power), '17 % 5' (modulo)"
        "\n\nMATH FUNCTIONS: sqrt(16), sin(x), cos(x), tan(x), log(x), log10(x), "
        "exp(x), factorial(5), abs(-5), round(3.7), ceil(x), floor(x), "
        "min(1,2,3), max(1,2,3), gcd(12,8), degrees(pi), radians(180)"
        "\n\nCONSTANTS: pi, e, tau"
        "\n\nVARIABLES: 'x = 10' to store, then 'x * 2' to use, 'variables' to list, 'clear' to reset"
        "\n\nUNIT CONVERSION: '5 km to miles', '100 F to C', '10 pounds to kg'"
        "\n\nSupported units: length (m, km, mi, ft, in), weight (kg, lb, oz), "
        "volume (l, gal, cup), time (s, min, h, day), temperature (C, F, K), "
        "speed (m/s, km/h, mph), area (sqm, sqft, acre), data (kb, mb, gb)"
        "\n\nType 'help' for full documentation."
    )
)
