"""Tests for the date/time calculator tool."""

import pytest
from pydantic import ValidationError

from src.tools.datetime_tool import (
    datetime_calculator,
    datetime_tool,
    DatetimeInput,
)


class TestDatetimeCalculator:
    """Tests for the datetime calculator tool — typed parameters."""

    @pytest.mark.asyncio
    async def test_now(self):
        result = await datetime_calculator(operation="now", timezone="UTC")
        assert "Current Date/Time" in result
        assert "UTC" in result
        assert "Day:" in result

    @pytest.mark.asyncio
    async def test_add_days(self):
        result = await datetime_calculator(operation="add", date="2024-01-15", days=30)
        assert "2024-02-14" in result
        assert "30 day(s)" in result

    @pytest.mark.asyncio
    async def test_add_months(self):
        # Jan 31 + 1 month = Feb 29 (2024 is a leap year)
        result = await datetime_calculator(operation="add", date="2024-01-31", months=1)
        assert "2024-02-29" in result

    @pytest.mark.asyncio
    async def test_add_years(self):
        result = await datetime_calculator(operation="add", date="2024-03-01", years=2)
        assert "2026-03-01" in result

    @pytest.mark.asyncio
    async def test_add_business_days(self):
        # 2024-01-15 is a Monday; +5 business days = 2024-01-22 (next Monday)
        result = await datetime_calculator(
            operation="add", date="2024-01-15", business_days=5,
        )
        assert "2024-01-22" in result

    @pytest.mark.asyncio
    async def test_diff(self):
        result = await datetime_calculator(
            operation="diff", date_from="2024-01-01", date_to="2024-12-31",
        )
        assert "365 days" in result
        assert "Business days:" in result

    @pytest.mark.asyncio
    async def test_timezone_conversion(self):
        result = await datetime_calculator(
            operation="convert",
            convert_datetime="2024-01-15 14:00",
            from_tz="EST",
            to_tz="JST",
        )
        assert "EST" in result
        assert "JST" in result
        # EST is -5, JST is +9, so 14:00 EST = 04:00 next day JST
        assert "04:00" in result

    @pytest.mark.asyncio
    async def test_date_info(self):
        # July 4, 2024 is a Thursday
        result = await datetime_calculator(operation="info", date="2024-07-04")
        assert "Thursday" in result
        assert "Q3" in result
        assert "No" in result  # not a weekend

    @pytest.mark.asyncio
    async def test_weekend_info(self):
        # 2024-01-13 is a Saturday
        result = await datetime_calculator(operation="info", date="2024-01-13")
        assert "Saturday" in result
        assert "Yes" in result  # is weekend

    @pytest.mark.asyncio
    async def test_business_days_count(self):
        # 2024-01-01 (Mon) to 2024-01-05 (Fri) — business days strictly between
        result = await datetime_calculator(
            operation="business_days",
            date_from="2024-01-01",
            date_to="2024-01-05",
        )
        assert "Business days:" in result
        assert "Calendar days:" in result

    @pytest.mark.asyncio
    async def test_unknown_operation_message(self):
        # Direct calls bypass Pydantic; the function returns a friendly message.
        result = await datetime_calculator(operation="bogus")
        assert "Unknown operation" in result

    @pytest.mark.asyncio
    async def test_invalid_date(self):
        result = await datetime_calculator(operation="info", date="not-a-date")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_unknown_timezone(self):
        result = await datetime_calculator(operation="now", timezone="FAKE_TZ")
        assert "Error" in result or "Unknown timezone" in result


class TestDatetimeSchema:
    """Pydantic args_schema enforces shape; LangChain rejects bad calls at the boundary."""

    def test_missing_operation_rejected(self):
        with pytest.raises(ValidationError):
            DatetimeInput()

    def test_unknown_operation_rejected_by_literal(self):
        with pytest.raises(ValidationError):
            DatetimeInput(operation="bogus")

    def test_minimal_now_valid(self):
        parsed = DatetimeInput(operation="now")
        assert parsed.operation == "now"
        assert parsed.timezone == "UTC"  # default

    def test_full_add_valid(self):
        parsed = DatetimeInput(operation="add", date="2024-01-15", days=30, business_days=2)
        assert parsed.days == 30
        assert parsed.business_days == 2


class TestDatetimeTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        assert datetime_tool.name == "datetime_calculator"
        assert datetime_tool.args_schema is DatetimeInput
