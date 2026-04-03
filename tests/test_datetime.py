"""Tests for the date/time calculator tool."""

import json
import pytest


class TestDatetimeCalculator:
    """Tests for the datetime calculator tool."""

    @pytest.mark.asyncio
    async def test_now(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "now", "timezone": "UTC"})
        result = await datetime_calculate(query)
        assert "Current Date/Time" in result
        assert "UTC" in result
        assert "Day:" in result

    @pytest.mark.asyncio
    async def test_add_days(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "add", "date": "2024-01-15", "days": 30})
        result = await datetime_calculate(query)
        assert "2024-02-14" in result
        assert "30 day(s)" in result

    @pytest.mark.asyncio
    async def test_add_months(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "add", "date": "2024-01-31", "months": 1})
        result = await datetime_calculate(query)
        # Jan 31 + 1 month = Feb 29 (2024 is leap year)
        assert "2024-02-29" in result

    @pytest.mark.asyncio
    async def test_add_years(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "add", "date": "2024-03-01", "years": 2})
        result = await datetime_calculate(query)
        assert "2026-03-01" in result

    @pytest.mark.asyncio
    async def test_add_business_days(self):
        from src.tools.datetime_tool import datetime_calculate

        # 2024-01-15 is a Monday, adding 5 business days = 2024-01-22 (next Monday)
        query = json.dumps({"operation": "add", "date": "2024-01-15", "business_days": 5})
        result = await datetime_calculate(query)
        assert "2024-01-22" in result

    @pytest.mark.asyncio
    async def test_diff(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "diff", "from": "2024-01-01", "to": "2024-12-31"})
        result = await datetime_calculate(query)
        assert "365 days" in result
        assert "Business days:" in result

    @pytest.mark.asyncio
    async def test_timezone_conversion(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({
            "operation": "convert",
            "datetime": "2024-01-15 14:00",
            "from_tz": "EST",
            "to_tz": "JST"
        })
        result = await datetime_calculate(query)
        assert "EST" in result
        assert "JST" in result
        # EST is -5, JST is +9, so 14:00 EST = 04:00 next day JST
        assert "04:00" in result

    @pytest.mark.asyncio
    async def test_date_info(self):
        from src.tools.datetime_tool import datetime_calculate

        # July 4, 2024 is a Thursday
        query = json.dumps({"operation": "info", "date": "2024-07-04"})
        result = await datetime_calculate(query)
        assert "Thursday" in result
        assert "Q3" in result
        assert "No" in result  # not a weekend

    @pytest.mark.asyncio
    async def test_weekend_info(self):
        from src.tools.datetime_tool import datetime_calculate

        # 2024-01-13 is a Saturday
        query = json.dumps({"operation": "info", "date": "2024-01-13"})
        result = await datetime_calculate(query)
        assert "Saturday" in result
        assert "Yes" in result  # is weekend

    @pytest.mark.asyncio
    async def test_business_days_count(self):
        from src.tools.datetime_tool import datetime_calculate

        # 2024-01-01 (Mon) to 2024-01-05 (Fri) = 4 business days between (Tue-Fri)
        query = json.dumps({"operation": "business_days", "from": "2024-01-01", "to": "2024-01-05"})
        result = await datetime_calculate(query)
        assert "Business days:" in result
        assert "Calendar days:" in result

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "bogus"})
        result = await datetime_calculate(query)
        assert "Unknown operation" in result

    @pytest.mark.asyncio
    async def test_invalid_date(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "info", "date": "not-a-date"})
        result = await datetime_calculate(query)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_unknown_timezone(self):
        from src.tools.datetime_tool import datetime_calculate

        query = json.dumps({"operation": "now", "timezone": "FAKE_TZ"})
        result = await datetime_calculate(query)
        assert "Error" in result or "Unknown timezone" in result

    @pytest.mark.asyncio
    async def test_simple_string_input(self):
        from src.tools.datetime_tool import datetime_calculate

        result = await datetime_calculate("what time is it")
        assert "JSON input" in result  # should prompt for JSON

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        from src.tools.datetime_tool import datetime_calculate

        result = await datetime_calculate("{invalid json")
        assert "Error" in result

    def test_tool_wrapper_exists(self):
        from src.tools.datetime_tool import datetime_tool
        assert datetime_tool.name == "datetime_calculator"
        assert datetime_tool.coroutine is not None
