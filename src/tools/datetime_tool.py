"""Date/Time calculator — arithmetic, timezone conversion, business days."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import safe_tool_call

# ─── Module overview ───────────────────────────────────────────────
# Date/time operations: arithmetic, timezone conversion, business-day
# counting, and date info lookups. Schema is enforced by Anthropic's
# tool-use API via args_schema; the tool dispatches by `operation`.
# ───────────────────────────────────────────────────────────────────

# Common timezone offsets (no pytz dependency needed)
TIMEZONE_OFFSETS = {
    "UTC": 0, "GMT": 0,
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
    "CET": 1, "CEST": 2, "EET": 2, "EEST": 3,
    "IST": 5.5, "JST": 9, "KST": 9, "CST_CN": 8,
    "AEST": 10, "AEDT": 11, "NZST": 12, "NZDT": 13,
    "BRT": -3, "ART": -3, "GST": 4,
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# Takes (date_str). Tries several common date formats.
# Returns a datetime or raises ValueError.
def _parse_date(date_str: str) -> datetime:
    """Parse a date string in common formats (YYYY-MM-DD, MM/DD/YYYY, etc.)."""
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d %b %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: '{date_str}'. Use YYYY-MM-DD format.")


# Takes (name). Maps a timezone abbreviation to a fixed UTC offset.
# Returns a timezone object or raises ValueError.
def _get_tz(name: str) -> timezone:
    """Get a fixed-offset timezone by abbreviation."""
    name = name.upper().strip()
    if name not in TIMEZONE_OFFSETS:
        raise ValueError(
            f"Unknown timezone '{name}'. Supported: {', '.join(sorted(TIMEZONE_OFFSETS.keys()))}"
        )
    offset_hours = TIMEZONE_OFFSETS[name]
    return timezone(timedelta(hours=offset_hours))


# Takes (dt, months). Adds months to a date, clamping to month-end.
# Returns the adjusted datetime.
def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a date, clamping day to month-end when needed."""
    import calendar
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


# Takes (start, end). Counts weekdays between two dates, excluding endpoints.
# Returns the integer count of business days.
def _business_days_between(start: datetime, end: datetime) -> int:
    """Count business days (Mon-Fri) between two dates, excluding endpoints."""
    if start > end:
        start, end = end, start
    count = 0
    current = start + timedelta(days=1)
    while current < end:
        if current.weekday() < 5:  # Monday=0 through Friday=4
            count += 1
        current += timedelta(days=1)
    return count


# Takes (start, days). Skips weekends while adding days.
# Returns the resulting datetime.
def _add_business_days(start: datetime, days: int) -> datetime:
    """Add N business days to a date."""
    current = start
    added = 0
    direction = 1 if days >= 0 else -1
    target = abs(days)
    while added < target:
        current += timedelta(days=direction)
        if current.weekday() < 5:
            added += 1
    return current


# Dispatches to now/add/diff/convert/info/business_days based on `operation`.
# Returns a formatted result string. Errors are caught by safe_tool_call.
@safe_tool_call("calculating date/time")
async def datetime_calculator(
    operation: str,
    timezone: str = "UTC",
    date: str = "",
    days: int = 0,
    weeks: int = 0,
    months: int = 0,
    years: int = 0,
    business_days: int = 0,
    date_from: str = "",
    date_to: str = "",
    convert_datetime: str = "",
    from_tz: str = "UTC",
    to_tz: str = "UTC",
) -> str:
    """Run a single date/time operation with typed parameters."""
    op = operation.lower()
    biz_days = business_days  # avoid shadowing the parameter inside add-result text

    try:
        if op == "now":
            tz = _get_tz(timezone)
            now = datetime.now(tz)
            return (
                f"**Current Date/Time ({timezone})**\n"
                f"Date: {now.strftime('%Y-%m-%d')}\n"
                f"Time: {now.strftime('%H:%M:%S')}\n"
                f"Day: {DAY_NAMES[now.weekday()]}\n"
                f"Week: {now.isocalendar()[1]}\n"
                f"Quarter: Q{(now.month - 1) // 3 + 1}"
            )

        elif op == "add":
            d = _parse_date(date)
            result = d
            if years:
                result = _add_months(result, years * 12)
            if months:
                result = _add_months(result, months)
            if weeks:
                result += timedelta(weeks=weeks)
            if days:
                result += timedelta(days=days)
            if biz_days:
                result = _add_business_days(result, biz_days)

            parts_added = []
            if years: parts_added.append(f"{years} year(s)")
            if months: parts_added.append(f"{months} month(s)")
            if weeks: parts_added.append(f"{weeks} week(s)")
            if days: parts_added.append(f"{days} day(s)")
            if biz_days: parts_added.append(f"{biz_days} business day(s)")

            return (
                f"**Date Arithmetic**\n"
                f"Start: {d.strftime('%Y-%m-%d')} ({DAY_NAMES[d.weekday()]})\n"
                f"Added: {', '.join(parts_added)}\n"
                f"Result: {result.strftime('%Y-%m-%d')} ({DAY_NAMES[result.weekday()]})"
            )

        elif op == "diff":
            df = _parse_date(date_from)
            dt = _parse_date(date_to)
            delta = dt - df

            total_days = abs(delta.days)
            yrs = total_days // 365
            remaining = total_days % 365
            mos = remaining // 30
            dys = remaining % 30
            wks = total_days // 7
            biz_count = _business_days_between(df, dt)

            direction = "later" if delta.days >= 0 else "earlier"

            return (
                f"**Date Difference**\n"
                f"From: {df.strftime('%Y-%m-%d')} ({DAY_NAMES[df.weekday()]})\n"
                f"To: {dt.strftime('%Y-%m-%d')} ({DAY_NAMES[dt.weekday()]})\n"
                f"Total: {total_days} days ({direction})\n"
                f"Approx: {yrs}y {mos}m {dys}d\n"
                f"Weeks: {wks}\n"
                f"Business days: {biz_count}"
            )

        elif op == "convert":
            dt = _parse_date(convert_datetime)
            src_tz = _get_tz(from_tz)
            dst_tz = _get_tz(to_tz)

            dt_src = dt.replace(tzinfo=src_tz)
            dt_dst = dt_src.astimezone(dst_tz)

            return (
                f"**Timezone Conversion**\n"
                f"From: {dt_src.strftime('%Y-%m-%d %H:%M')} {from_tz}\n"
                f"To: {dt_dst.strftime('%Y-%m-%d %H:%M')} {to_tz}\n"
                f"Offset: {TIMEZONE_OFFSETS.get(to_tz.upper(), 0) - TIMEZONE_OFFSETS.get(from_tz.upper(), 0):+.1f} hours"
            )

        elif op == "info":
            d = _parse_date(date)
            iso = d.isocalendar()

            return (
                f"**Date Info: {d.strftime('%Y-%m-%d')}**\n"
                f"Day of week: {DAY_NAMES[d.weekday()]}\n"
                f"Day of year: {d.timetuple().tm_yday}\n"
                f"Week number: {iso[1]}\n"
                f"Quarter: Q{(d.month - 1) // 3 + 1}\n"
                f"Is weekend: {'Yes' if d.weekday() >= 5 else 'No'}\n"
                f"ISO format: {d.isoformat()}"
            )

        elif op == "business_days":
            df = _parse_date(date_from)
            dt = _parse_date(date_to)
            biz_count = _business_days_between(df, dt)
            total_days = abs((dt - df).days)

            return (
                f"**Business Days**\n"
                f"From: {df.strftime('%Y-%m-%d')} ({DAY_NAMES[df.weekday()]})\n"
                f"To: {dt.strftime('%Y-%m-%d')} ({DAY_NAMES[dt.weekday()]})\n"
                f"Business days: {biz_count}\n"
                f"Calendar days: {total_days}\n"
                f"Weekend days: {total_days - biz_count}"
            )

        else:
            return (
                f"Unknown operation: '{operation}'. "
                f"Supported: now, add, diff, convert, info, business_days"
            )

    except Exception as e:
        return f"Error: {str(e)}"


class DatetimeInput(BaseModel):
    """Inputs for the datetime_calculator tool."""
    operation: Literal["now", "add", "diff", "convert", "info", "business_days"] = Field(
        description="Which date/time operation to perform.",
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone abbreviation for 'now' (e.g. UTC, EST, JST). Defaults to UTC.",
    )
    date: str = Field(
        default="",
        description="Date for 'add' or 'info' (YYYY-MM-DD or other common formats).",
    )
    days: int = Field(default=0, description="Days to add (negative subtracts). Used by 'add'.")
    weeks: int = Field(default=0, description="Weeks to add. Used by 'add'.")
    months: int = Field(default=0, description="Months to add. Used by 'add'.")
    years: int = Field(default=0, description="Years to add. Used by 'add'.")
    business_days: int = Field(
        default=0,
        description="Business days to add (skips weekends). Used by 'add'.",
    )
    date_from: str = Field(
        default="",
        description="Start date for 'diff' or 'business_days' (YYYY-MM-DD).",
    )
    date_to: str = Field(
        default="",
        description="End date for 'diff' or 'business_days' (YYYY-MM-DD).",
    )
    convert_datetime: str = Field(
        default="",
        description="Datetime string to convert (e.g. '2024-01-15 14:00'). Used by 'convert'.",
    )
    from_tz: str = Field(
        default="UTC",
        description="Source timezone abbreviation for 'convert'.",
    )
    to_tz: str = Field(
        default="UTC",
        description="Target timezone abbreviation for 'convert'.",
    )


class DatetimeTool(BaseTool):
    name: str = "datetime_calculator"
    description: str = (
        "Perform date/time calculations: arithmetic, timezone conversion, business days, "
        "and date info."
        "\n\nOPERATIONS:"
        "\n- now: Current time. Optional: timezone (default UTC)."
        "\n- add: Add days/weeks/months/years/business_days to `date`."
        "\n- diff: Days between `date_from` and `date_to`."
        "\n- convert: Convert `convert_datetime` from `from_tz` to `to_tz`."
        "\n- info: Day-of-week, week number, quarter, weekend status of `date`."
        "\n- business_days: Count business days between `date_from` and `date_to`."
        "\n\nDO NOT USE FOR: simple arithmetic (use calculator), recurring schedules."
    )
    args_schema: Type[BaseModel] = DatetimeInput

    # Forwards every validated parameter to datetime_calculator.
    async def _arun(self, **kwargs) -> str:
        return await datetime_calculator(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


datetime_tool = DatetimeTool()
