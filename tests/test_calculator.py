"""Tests for src/tools/calculator_tool.py — math calculations and variables."""

import math
import pytest
from src.tools.calculator_tool import AdvancedCalculator, calculate


@pytest.fixture
def calc():
    """Fresh calculator instance for each test."""
    return AdvancedCalculator()


class TestBasicArithmetic:
    """Test basic math operations."""

    async def test_addition(self, calc):
        assert calc.calculate("2 + 2") == "4"

    async def test_subtraction(self, calc):
        assert calc.calculate("10 - 3") == "7"

    async def test_multiplication(self, calc):
        assert calc.calculate("6 * 7") == "42"

    async def test_division(self, calc):
        assert calc.calculate("100 / 4") == "25"

    async def test_power(self, calc):
        assert calc.calculate("2 ** 10") == "1024"

    async def test_modulo(self, calc):
        assert calc.calculate("17 % 5") == "2"

    async def test_float_result(self, calc):
        result = calc.calculate("10 / 3")
        assert result.startswith("3.33333")

    async def test_negative_numbers(self, calc):
        assert calc.calculate("-5 + 3") == "-2"

    async def test_complex_expression(self, calc):
        assert calc.calculate("(2 + 3) * 4") == "20"


class TestMathFunctions:
    """Test built-in math functions."""

    async def test_sqrt(self, calc):
        assert calc.calculate("sqrt(16)") == "4"

    async def test_factorial(self, calc):
        assert calc.calculate("factorial(5)") == "120"

    async def test_abs(self, calc):
        assert calc.calculate("abs(-42)") == "42"

    async def test_round(self, calc):
        assert calc.calculate("round(3.7)") == "4"

    async def test_ceil(self, calc):
        assert calc.calculate("ceil(3.2)") == "4"

    async def test_floor(self, calc):
        assert calc.calculate("floor(3.8)") == "3"

    async def test_log(self, calc):
        result = float(calc.calculate("log(e)"))
        assert abs(result - 1.0) < 0.0001

    async def test_sin_of_zero(self, calc):
        assert calc.calculate("sin(0)") == "0"

    async def test_sin_of_pi_over_2(self, calc):
        result = float(calc.calculate("sin(pi / 2)"))
        assert abs(result - 1.0) < 0.0001

    async def test_gcd(self, calc):
        assert calc.calculate("gcd(12, 8)") == "4"

    async def test_min_max(self, calc):
        assert calc.calculate("min(3, 1, 2)") == "1"
        assert calc.calculate("max(3, 1, 2)") == "3"


class TestConstants:
    """Test math constants."""

    async def test_pi(self, calc):
        result = float(calc.calculate("pi"))
        assert abs(result - math.pi) < 0.0001

    async def test_e(self, calc):
        result = float(calc.calculate("e"))
        assert abs(result - math.e) < 0.0001

    async def test_tau(self, calc):
        result = float(calc.calculate("tau"))
        assert abs(result - math.tau) < 0.0001


class TestVariables:
    """Test variable storage and retrieval."""

    async def test_set_and_use_variable(self, calc):
        calc.calculate("x = 10")
        assert calc.calculate("x * 3") == "30"

    async def test_set_with_keyword(self, calc):
        result = calc.calculate("set y = 25")
        assert "y = 25" in result

    async def test_variable_persistence(self, calc):
        calc.calculate("a = 5")
        calc.calculate("b = 10")
        assert calc.calculate("a + b") == "15"

    async def test_variable_from_expression(self, calc):
        calc.calculate("result = 2 + 3")
        assert calc.calculate("result") == "5"

    async def test_list_variables(self, calc):
        calc.calculate("x = 42")
        result = calc.calculate("variables")
        assert "x" in result
        assert "42" in result

    async def test_list_empty_variables(self, calc):
        result = calc.calculate("vars")
        assert "No variables" in result

    async def test_clear_variables(self, calc):
        calc.calculate("x = 10")
        calc.calculate("clear")
        assert "No variables" in calc.calculate("vars")

    async def test_reserved_name_rejection(self, calc):
        result = calc.calculate("pi = 5")
        assert "Error" in result or "reserved" in result.lower()

    async def test_invalid_variable_name(self, calc):
        result = calc.set_variable("123abc", 10)
        assert "Error" in result


class TestSecurity:
    """Test that dangerous expressions are blocked."""

    async def test_blocks_import(self, calc):
        result = calc.calculate("__import__('os')")
        assert "Error" in result

    async def test_blocks_dunder(self, calc):
        result = calc.calculate("__builtins__")
        assert "Error" in result

    async def test_blocks_exec(self, calc):
        result = calc.calculate("exec('print(1)')")
        assert "Error" in result

    async def test_blocks_eval(self, calc):
        result = calc.calculate("eval('1+1')")
        assert "Error" in result

    async def test_blocks_open(self, calc):
        result = calc.calculate("open('file.txt')")
        assert "Error" in result


class TestErrorHandling:
    """Test error cases."""

    async def test_empty_input(self, calc):
        result = calc.calculate("")
        assert "Error" in result

    async def test_division_by_zero(self, calc):
        result = calc.calculate("1 / 0")
        assert "Error" in result and "zero" in result.lower()

    async def test_invalid_syntax(self, calc):
        result = calc.calculate("2 +* 3")
        assert "Error" in result

    async def test_unknown_variable(self, calc):
        result = calc.calculate("unknown_var + 1")
        assert "Error" in result

    async def test_help_command(self, calc):
        result = calc.calculate("help")
        assert "Calculator Help" in result


class TestModuleLevelFunction:
    """Test the module-level calculate() function used by LangChain."""

    async def test_calculate_function_works(self):
        result = calculate("2 + 2")
        assert result == "4"
