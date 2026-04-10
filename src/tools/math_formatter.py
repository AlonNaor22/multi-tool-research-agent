"""Math formatter — renders MATH_STRUCTURED JSON as Streamlit-compatible KaTeX markdown."""

import json
from typing import Optional
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Markdown generation helpers
# ---------------------------------------------------------------------------

def _matrix_to_markdown(data: list, caption: Optional[str] = None) -> str:
    """Render a 2D list as a centered markdown table."""
    if not data:
        return ""
    cols = len(data[0])
    # Header row (column indices)
    header = "| " + " | ".join(f"C{j+1}" for j in range(cols)) + " |"
    separator = "|" + "|".join(" :---: " for _ in range(cols)) + "|"
    rows = []
    for row in data:
        cells = " | ".join(_fmt_num(v) for v in row)
        rows.append(f"| {cells} |")

    parts = [header, separator] + rows
    if caption:
        parts.append(f"\n*{caption}*")
    return "\n".join(parts)


def _fmt_num(v) -> str:
    """Format a numeric value as a clean string."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


def _latex_inline(expr: str) -> str:
    """Wrap expr in KaTeX inline delimiters ($...$)."""
    if not expr:
        return ""
    return f"${expr}$"


def _latex_block(expr: str) -> str:
    """Wrap expr in KaTeX block delimiters ($$...$$)."""
    if not expr:
        return ""
    return f"\n\n$${expr}$$\n\n"


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------

def format_math(input_str: str) -> str:
    """Convert a MATH_STRUCTURED JSON string into Streamlit-compatible KaTeX markdown."""
    # Strip the prefix if present
    if input_str.startswith("MATH_STRUCTURED:"):
        json_str = input_str[len("MATH_STRUCTURED:"):]
    else:
        json_str = input_str

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return "**Error:** Invalid math data format"

    if data.get("error") and not data.get("steps"):
        return f"**Error:** {data['error']}"

    operation = data.get("operation", "")
    lines = []

    # Title
    title = data.get("title", "")
    if title:
        if "\\" in title:
            lines.append(f"#### {_latex_inline(title)}")
        else:
            lines.append(f"#### {title}")
    lines.append("")

    # Input matrix display
    if data.get("matrix_data") and operation.startswith("matrix"):
        lines.append(_matrix_to_markdown(data["matrix_data"], "Input matrix"))
        lines.append("")

    # Steps
    steps = data.get("steps", [])
    for step in steps:
        num = step.get("num", "")
        desc = step.get("desc", "")
        expr_latex = step.get("expr_latex", "")

        lines.append(f"**Step {num}:** {desc}")
        if expr_latex:
            lines.append(f"> {_latex_inline(expr_latex)}")
        lines.append("")

    # Result
    result_latex = data.get("result_latex", "")
    result_text = data.get("result", "")

    if data.get("result_matrix_data"):
        lines.append("---")
        lines.append("**Result:**")
        lines.append("")
        lines.append(_matrix_to_markdown(data["result_matrix_data"]))
    elif result_latex:
        lines.append("---")
        lines.append(f"**Result:** {_latex_inline(result_latex)}")
    elif result_text:
        lines.append("---")
        lines.append(f"**Result:** {result_text}")

    # Error note (non-fatal, e.g., singular matrix with steps)
    if data.get("error") and steps:
        lines.append("")
        lines.append(f"> **Note:** {data['error']}")

    return "\n".join(lines)


async def math_formatter(input_str: str) -> str:
    """Format mathematical results with properly rendered equations and matrices.

    Input: A string from calculator tool that starts with 'MATH_STRUCTURED:' followed by JSON data.

    Output: Clean formatted text with LaTeX equations and matrix tables.

    ALWAYS use this tool to format calculator output that starts with 'MATH_STRUCTURED:' before presenting results to the user. Pass the ENTIRE calculator output (including the 'MATH_STRUCTURED:' prefix) as input to this tool."""
    return format_math(input_str)


math_formatter_tool = tool(math_formatter)
