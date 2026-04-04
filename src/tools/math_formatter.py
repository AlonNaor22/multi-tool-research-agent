"""Math formatter tool — converts structured math data to rich HTML.

Takes MATH_STRUCTURED: JSON from the calculator tool and produces
self-contained HTML with:
- MathJax-rendered LaTeX equations
- Styled HTML tables for matrices
- Numbered step-by-step breakdowns
- Highlighted final results

Output is wrapped in <!-- MATH_HTML --> sentinels for app.py to detect.
"""

import json
from typing import Optional
from langchain_core.tools import Tool


# ---------------------------------------------------------------------------
# MathJax CDN + CSS (self-contained for Streamlit st.html() iframes)
# ---------------------------------------------------------------------------

MATHJAX_SCRIPT = (
    '<script id="MathJax-script" async '
    'src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
)

MATH_CSS = """
<style>
.math-result {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    margin: 0; padding: 16px 20px;
    background: #f8f9fa; border-radius: 10px;
    border-left: 5px solid #4C72B0;
    color: #1a1a2e;
    line-height: 1.6;
}
.math-result h4 {
    margin: 0 0 12px 0; color: #2c3e50;
    font-size: 1.1em; font-weight: 600;
}
.math-step {
    margin: 8px 0; padding: 6px 0 6px 12px;
    border-left: 2px solid #dee2e6;
}
.step-num {
    font-weight: 700; color: #4C72B0;
    margin-right: 4px;
}
.step-desc { color: #495057; }
.math-expr {
    margin: 4px 0 4px 20px;
    font-size: 1.05em;
    color: #212529;
}
.math-result-final {
    margin-top: 14px; padding: 10px 14px;
    background: #e8f4fd; border-radius: 6px;
    font-size: 1.1em; font-weight: 500;
    border: 1px solid #b8daff;
}
.math-matrix {
    border-collapse: collapse;
    margin: 8px auto;
    background: white;
}
.math-matrix td {
    padding: 6px 14px;
    text-align: center;
    border: 1px solid #dee2e6;
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
}
.math-matrix tr:nth-child(even) { background: #f1f3f5; }
.math-matrix-caption {
    text-align: center; font-size: 0.85em;
    color: #6c757d; margin-top: 4px;
}
.math-error {
    color: #dc3545; font-weight: 500;
    padding: 8px 12px; background: #f8d7da;
    border-radius: 6px; border: 1px solid #f5c6cb;
}
</style>
"""


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _matrix_to_html(data: list, caption: Optional[str] = None) -> str:
    """Render a 2D list as an HTML table."""
    if not data:
        return ""
    rows = []
    for row in data:
        cells = "".join(f"<td>{_fmt_num(v)}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    table = f'<table class="math-matrix">{"".join(rows)}</table>'
    if caption:
        table += f'<div class="math-matrix-caption">{caption}</div>'
    return table


def _fmt_num(v) -> str:
    """Format a number for display in a matrix cell."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


def _latex(expr: str) -> str:
    """Wrap an expression in MathJax inline delimiters."""
    if not expr:
        return ""
    return f"\\({expr}\\)"


def _latex_block(expr: str) -> str:
    """Wrap an expression in MathJax display (block) delimiters."""
    if not expr:
        return ""
    return f"$${expr}$$"


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------

def format_math(input_str: str) -> str:
    """Format a MATH_STRUCTURED: JSON string into rich HTML.

    Args:
        input_str: Either a raw JSON string or prefixed with "MATH_STRUCTURED:"

    Returns:
        HTML string wrapped in <!-- MATH_HTML --> sentinels.
    """
    # Strip the prefix if present
    if input_str.startswith("MATH_STRUCTURED:"):
        json_str = input_str[len("MATH_STRUCTURED:"):]
    else:
        json_str = input_str

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return _wrap_html('<div class="math-error">Error: Invalid math data format</div>')

    if data.get("error") and not data.get("steps"):
        return _wrap_html(f'<div class="math-error">Error: {data["error"]}</div>')

    operation = data.get("operation", "")
    html_parts = []

    # Title
    title = data.get("title", "")
    if title:
        # If title contains LaTeX (backslashes), render it
        if "\\" in title:
            html_parts.append(f'<h4>{_latex(title)}</h4>')
        else:
            html_parts.append(f"<h4>{title}</h4>")

    # Input matrix display
    if data.get("matrix_data") and operation.startswith("matrix"):
        html_parts.append(_matrix_to_html(data["matrix_data"], "Input matrix"))

    # Steps
    steps = data.get("steps", [])
    for step in steps:
        num = step.get("num", "")
        desc = step.get("desc", "")
        expr_latex = step.get("expr_latex", "")

        step_html = f'<div class="math-step">'
        step_html += f'<span class="step-num">Step {num}:</span> '
        step_html += f'<span class="step-desc">{desc}</span>'
        if expr_latex:
            step_html += f'<div class="math-expr">{_latex(expr_latex)}</div>'
        step_html += '</div>'
        html_parts.append(step_html)

    # Result
    result_latex = data.get("result_latex", "")
    result_text = data.get("result", "")

    # Result matrix
    if data.get("result_matrix_data"):
        html_parts.append('<div class="math-result-final">')
        html_parts.append("<strong>Result:</strong>")
        html_parts.append(_matrix_to_html(data["result_matrix_data"]))
        html_parts.append("</div>")
    elif result_latex:
        html_parts.append(f'<div class="math-result-final"><strong>Result:</strong> {_latex(result_latex)}</div>')
    elif result_text:
        html_parts.append(f'<div class="math-result-final"><strong>Result:</strong> {result_text}</div>')

    # Graph hint
    if data.get("has_function") and data.get("expression_str"):
        expr_str = data["expression_str"]
        html_parts.append(
            f'<div style="margin-top:8px;color:#6c757d;font-size:0.85em;">'
            f'Tip: This involves a function. Use create_chart to plot it: '
            f'<code>{{"chart_type":"function","data":{{"expression":"{expr_str}"}}}}</code>'
            f'</div>'
        )

    # Error note (non-fatal, e.g., singular matrix that still has steps)
    if data.get("error") and steps:
        html_parts.append(f'<div class="math-error" style="margin-top:8px;">Note: {data["error"]}</div>')

    content = "\n".join(html_parts)
    return _wrap_html(f'<div class="math-result">{content}</div>')


def _wrap_html(content: str) -> str:
    """Wrap content with sentinels, CSS, and MathJax script."""
    return (
        f"<!-- MATH_HTML -->\n"
        f"{MATH_CSS}\n"
        f"{MATHJAX_SCRIPT}\n"
        f"{content}\n"
        f"<!-- /MATH_HTML -->"
    )


async def async_format_math(input_str: str) -> str:
    """Async wrapper for the math formatter."""
    return format_math(input_str)


# LangChain Tool wrapper
math_formatter_tool = Tool(
    name="math_formatter",
    func=format_math,
    coroutine=async_format_math,
    description=(
        "Format mathematical results as rich HTML with properly rendered equations and matrices. "
        "\n\nInput: A string from calculator tool that starts with 'MATH_STRUCTURED:' "
        "followed by JSON data."
        "\n\nOutput: Beautiful HTML with MathJax equations, styled step-by-step solutions, "
        "and HTML tables for matrices."
        "\n\nALWAYS use this tool to format calculator output that starts with 'MATH_STRUCTURED:' "
        "before presenting results to the user. Pass the ENTIRE calculator output "
        "(including the 'MATH_STRUCTURED:' prefix) as input to this tool."
    ),
)
