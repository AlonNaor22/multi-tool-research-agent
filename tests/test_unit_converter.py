"""Tests for src/tools/unit_converter_tool.py — unit conversions."""

import pytest
from src.tools.unit_converter_tool import convert


class TestLengthConversions:
    """Test length/distance conversions."""

    def test_km_to_miles(self):
        result = convert("10 km to miles")
        assert "6.21" in result

    def test_miles_to_km(self):
        result = convert("1 mile to km")
        assert "1.60" in result or "1.61" in result

    def test_meters_to_feet(self):
        result = convert("100 meters to feet")
        assert "328" in result

    def test_inches_to_cm(self):
        result = convert("12 inches to cm")
        assert "30" in result


class TestWeightConversions:
    """Test weight/mass conversions."""

    def test_kg_to_pounds(self):
        result = convert("1 kg to pounds")
        assert "2.2" in result

    def test_pounds_to_kg(self):
        result = convert("100 pounds to kg")
        assert "45" in result


class TestTemperatureConversions:
    """Test temperature conversions (non-linear)."""

    def test_freezing_c_to_f(self):
        result = convert("0 celsius to fahrenheit")
        assert "32" in result

    def test_boiling_c_to_f(self):
        result = convert("100 celsius to fahrenheit")
        assert "212" in result

    def test_body_temp_f_to_c(self):
        result = convert("98.6 fahrenheit to celsius")
        assert "37" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_value(self):
        result = convert("0 km to miles")
        assert "0" in result

    def test_negative_value(self):
        result = convert("-40 celsius to fahrenheit")
        assert "-40" in result

    def test_invalid_unit(self):
        result = convert("10 foobar to baz")
        assert "Error" in result or "not recognized" in result.lower() or "Unknown" in result

    def test_empty_input(self):
        result = convert("")
        assert "Error" in result or "help" in result.lower() or "Usage" in result

    def test_help_command(self):
        result = convert("help")
        # Should return help text or usage info
        assert len(result) > 20
