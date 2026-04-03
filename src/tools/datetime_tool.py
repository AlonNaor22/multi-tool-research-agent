"""Date/Time calculator tool for the research agent.

Performs date arithmetic, timezone conversions, business day calculations,
and relative date computations. No external API required.

Features:
- Date arithmetic (add/subtract days, weeks, months, years)
- Timezone conversion
- Business days calculation (excludes weekends)
- Day of week, week number, quarter
- Duration between two dates
- Relative dates (next Friday, last Monday, etc.)
"""

import json
from datetime import datetime, timedelta, timezone
from langchain_core.tools import Tool
from src.utils import make_sync

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


def _parse_date(date_str: str) -> datetime:
    """Parse a date string in common formats."""
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


def _get_tz(name: str) -> timezone:
    """Get a timezone by name."""
    name = name.upper().strip()
    if name not in TIMEZONE_OFFSETS:
        raise ValueError(
            f"Unknown timezone '{name}'. Supported: {', '.join(sorted(TIMEZONE_OFFSETS.keys()))}"
        )
    offset_hours = TIMEZONE_OFFSETS[name]
    return timezone(timedelta(hours=offset_hours))


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a date, handling month-end edge cases."""
    import calendar
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _business_days_between(start: datetime, end: datetime) -> int:
    """Count business days (Mon-Fri) between two dates, excluding both endpoints."""
    if start > end:
        start, end = end, start
    count = 0
    current = start + timedelta(days=1)
    while current < end:
        if current.weekday() < 5:  # Monday=0 through Friday=4
            count += 1
        current += timedelta(days=1)
    return count


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


async def datetime_calculate(query: str) -> str:
    """Perform date/time calculations.

    Input is a JSON object specifying the operation:

    Operations:
    - now: Get current date/time: {"operation": "now", "timezone": "EST"}
    - add: Add time to a date: {"operation": "add", "date": "2024-01-15", "days": 30}
      Supports: days, weeks, months, years, business_days
    - diff: Difference between dates: {"operation": "diff", "from": "2024-01-01", "to": "2024-12-31"}
    - convert: Timezone conversion: {"operation": "convert", "datetime": "2024-01-15 14:00", "from_tz": "EST", "to_tz": "JST"}
    - info: Date info: {"operation": "info", "date": "2024-07-04"}
    - business_days: Count business days: {"operation": "business_days", "from": "2024-01-01", "to": "2024-01-31"}
    """
    try:
        if query.strip().startswith("{"):
            params = json.loads(query)
        else:
            # Try to interpret as a simple query
            return (
                "Please provide a JSON input. Examples:\n"
                '- Current time: {"operation": "now", "timezone": "EST"}\n'
                '- Add days: {"operation": "add", "date": "2024-01-15", "days": 30}\n'
                '- Difference: {"operation": "diff", "from": "2024-01-01", "to": "2024-12-31"}\n'
                '- Convert TZ: {"operation": "convert", "datetime": "2024-01-15 14:00", "from_tz": "EST", "to_tz": "JST"}\n'
                '- Date info: {"operation": "info", "date": "2024-07-04"}\n'
                '- Business days: {"operation": "business_days", "from": "2024-01-01", "to": "2024-01-31"}'
            )
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. See tool description for examples."

    operation = params.get("operation", "").lower()

    try:
        if operation == "now":
            tz_name = params.get("timezone", "UTC")
            tz = _get_tz(tz_name)
            now = datetime.now(tz)
            return (
                f"**Current Date/Time ({tz_name})**\n"
                f"Date: {now.strftime('%Y-%m-%d')}\n"
                f"Time: {now.strftime('%H:%M:%S')}\n"
                f"Day: {DAY_NAMES[now.weekday()]}\n"
                f"Week: {now.isocalendar()[1]}\n"
                f"Quarter: Q{(now.month - 1) // 3 + 1}"
            )

        elif operation == "add":
            date = _parse_date(params.get("date", ""))
            days = params.get("days", 0)
            weeks = params.get("weeks", 0)
            months = params.get("months", 0)
            years = params.get("years", 0)
            biz_days = params.get("business_days", 0)

            result = date
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
                f"Start: {date.strftime('%Y-%m-%d')} ({DAY_NAMES[date.weekday()]})\n"
                f"Added: {', '.join(parts_added)}\n"
                f"Result: {result.strftime('%Y-%m-%d')} ({DAY_NAMES[result.weekday()]})"
            )

        elif operation == "diff":
            date_from = _parse_date(params.get("from", ""))
            date_to = _parse_date(params.get("to", ""))
            delta = date_to - date_from

            total_days = abs(delta.days)
            years = total_days // 365
            remaining = total_days % 365
            months = remaining // 30
            days = remaining % 30
            weeks = total_days // 7
            biz_days = _business_days_between(date_from, date_to)

            direction = "later" if delta.days >= 0 else "earlier"

            return (
                f"**Date Difference**\n"
                f"From: {date_from.strftime('%Y-%m-%d')} ({DAY_NAMES[date_from.weekday()]})\n"
                f"To: {date_to.strftime('%Y-%m-%d')} ({DAY_NAMES[date_to.weekday()]})\n"
                f"Total: {total_days} days ({direction})\n"
                f"Approx: {years}y {months}m {days}d\n"
                f"Weeks: {weeks}\n"
                f"Business days: {biz_days}"
            )

        elif operation == "convert":
            dt_str = params.get("datetime", "")
            from_tz = params.get("from_tz", "UTC")
            to_tz = params.get("to_tz", "UTC")

            dt = _parse_date(dt_str)
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

        elif operation == "info":
            date = _parse_date(params.get("date", ""))
            iso = date.isocalendar()

            return (
                f"**Date Info: {date.strftime('%Y-%m-%d')}**\n"
                f"Day of week: {DAY_NAMES[date.weekday()]}\n"
                f"Day of year: {date.timetuple().tm_yday}\n"
                f"Week number: {iso[1]}\n"
                f"Quarter: Q{(date.month - 1) // 3 + 1}\n"
                f"Is weekend: {'Yes' if date.weekday() >= 5 else 'No'}\n"
                f"ISO format: {date.isoformat()}"
            )

        elif operation == "business_days":
            date_from = _parse_date(params.get("from", ""))
            date_to = _parse_date(params.get("to", ""))
            biz_days = _business_days_between(date_from, date_to)
            total_days = abs((date_to - date_from).days)

            return (
                f"**Business Days**\n"
                f"From: {date_from.strftime('%Y-%m-%d')} ({DAY_NAMES[date_from.weekday()]})\n"
                f"To: {date_to.strftime('%Y-%m-%d')} ({DAY_NAMES[date_to.weekday()]})\n"
                f"Business days: {biz_days}\n"
                f"Calendar days: {total_days}\n"
                f"Weekend days: {total_days - biz_days}"
            )

        else:
            return (
                f"Unknown operation: '{operation}'. "
                f"Supported: now, add, diff, convert, info, business_days"
            )

    except Exception as e:
        return f"Error: {str(e)}"


datetime_tool = Tool(
    name="datetime_calculator",
    func=make_sync(datetime_calculate),
    coroutine=datetime_calculate,
    description=(
        "Perform date/time calculations: arithmetic, timezone conversion, "
        "business days, and date info. All input is JSON."
        "\n\nUSE FOR:"
        "\n- Current time: '{\"operation\": \"now\", \"timezone\": \"EST\"}'"
        "\n- Add days: '{\"operation\": \"add\", \"date\": \"2024-01-15\", \"days\": 30}'"
        "\n- Date diff: '{\"operation\": \"diff\", \"from\": \"2024-01-01\", \"to\": \"2024-12-31\"}'"
        "\n- Timezone: '{\"operation\": \"convert\", \"datetime\": \"2024-01-15 14:00\", "
        "\"from_tz\": \"EST\", \"to_tz\": \"JST\"}'"
        "\n- Date info: '{\"operation\": \"info\", \"date\": \"2024-07-04\"}'"
        "\n- Business days: '{\"operation\": \"business_days\", \"from\": \"2024-01-01\", \"to\": \"2024-01-31\"}'"
        "\n\nSupports: days, weeks, months, years, business_days in 'add' operation"
        "\n\nDO NOT USE FOR: simple arithmetic (use calculator), recurring schedules"
    ),
)
