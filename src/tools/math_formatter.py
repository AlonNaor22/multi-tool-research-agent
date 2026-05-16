"""Math formatter — renders solver structured dicts as Streamlit-compatible KaTeX markdown."""

from typing import Optional

# ─── Module overview ───────────────────────────────────────────────
# Library helpers (no LangChain tool) that turn the step solver's
# structured dict into Streamlit-compatible KaTeX markdown: LaTeX
# equations, step lists, and matrix tables. Imported directly by
# calculator_tool so its returned output is already formatted.
# ───────────────────────────────────────────────────────────────────


def _fmt_num(v) -> str:
    """Format a numeric value as a clean string."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


# Takes a 2D list and optional caption. Renders a centered markdown table.
def _matrix_to_markdown(data: list, caption: Optional[str] = None) -> str:
    """Render a 2D list as a centered markdown table."""
    if not data:
        return ""
    cols = len(data[0])
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


# Wraps a LaTeX expression in inline KaTeX delimiters ($...$).
def _latex_inline(expr: str) -> str:
    """Wrap expr in KaTeX inline delimiters ($...$)."""
    if not expr:
        return ""
    return f"${expr}$"


# Takes the step solver's structured dict directly.
# Returns Streamlit-compatible KaTeX markdown (title, steps, matrices, result).
def format_math_from_dict(data: dict) -> str:
    """Convert a solver structured dict into KaTeX markdown."""
    if data.get("error") and not data.get("steps"):
        return f"**Error:** {data['error']}"

    operation = data.get("operation", "")
    lines = []

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
    for step in data.get("steps", []) or []:
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
    if data.get("error") and data.get("steps"):
        lines.append("")
        lines.append(f"> **Note:** {data['error']}")

    return "\n".join(lines)
