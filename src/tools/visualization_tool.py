"""Visualization tool — matplotlib chart generation (bar, line, pie, scatter, etc.)."""

import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.tools import Tool
from src.constants import CHART_FIGSIZE, CHART_DPI


# Output directory for charts
OUTPUT_DIR = "output"

# Color palettes
COLOR_PALETTES = {
    "default": ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"],
    "warm": ["#E74C3C", "#E67E22", "#F1C40F", "#D35400", "#C0392B", "#F39C12"],
    "cool": ["#3498DB", "#1ABC9C", "#9B59B6", "#2980B9", "#16A085", "#8E44AD"],
    "pastel": ["#AEC6CF", "#FFB347", "#B39EB5", "#FF6961", "#77DD77", "#FDFD96"],
}

VALID_CHART_TYPES = [
    "bar", "stacked_bar", "line", "area", "pie",
    "scatter", "histogram", "box", "violin", "heatmap", "function",
]

# Safe namespace for evaluating function expressions
_FUNC_NAMESPACE = {
    "x": None,  # placeholder, replaced at eval time
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sqrt": np.sqrt, "abs": np.abs, "log": np.log, "log10": np.log10, "log2": np.log2,
    "exp": np.exp, "pi": np.pi, "e": np.e,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "ceil": np.ceil, "floor": np.floor,
}


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def get_colors(palette: str, count: int) -> list:
    colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])
    return [colors[i % len(colors)] for i in range(count)]


def generate_chart(input_str: str) -> str:
    """Generate a chart from a JSON specification and save as PNG."""
    try:
        spec = json.loads(input_str)
    except json.JSONDecodeError as e:
        return (
            f"Error: Invalid JSON input. {str(e)}\n\n"
            'Expected format: {"chart_type": "bar", "title": "My Chart", '
            '"data": {"labels": [...], "values": [...]}}'
        )

    if "chart_type" not in spec:
        return f"Error: Missing 'chart_type'. Options: {', '.join(VALID_CHART_TYPES)}"
    if "data" not in spec:
        return "Error: Missing 'data' field."

    chart_type = spec["chart_type"].lower()
    if chart_type not in VALID_CHART_TYPES:
        return f"Error: Unknown chart_type '{chart_type}'. Options: {', '.join(VALID_CHART_TYPES)}"

    title = spec.get("title", "Chart")
    data = spec["data"]
    palette = spec.get("palette", "default")
    show_grid = spec.get("grid", False)
    show_legend = spec.get("legend", True)
    single_color = spec.get("color")

    # Validate data keys
    validation_error = _validate_data(chart_type, data)
    if validation_error:
        return validation_error

    try:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        error_data = data.get("error")

        # --- Bar chart ---
        if chart_type == "bar":
            _draw_bar(ax, data, spec, palette, single_color, show_legend, error_data)

        # --- Stacked bar chart ---
        elif chart_type == "stacked_bar":
            _draw_stacked_bar(ax, data, spec, palette, single_color, show_legend)

        # --- Line chart ---
        elif chart_type == "line":
            _draw_line(ax, data, spec, palette, single_color, show_legend, error_data)

        # --- Area chart ---
        elif chart_type == "area":
            _draw_area(ax, data, spec, palette, single_color, show_legend)

        # --- Pie chart ---
        elif chart_type == "pie":
            _draw_pie(ax, data, palette)

        # --- Scatter plot ---
        elif chart_type == "scatter":
            _draw_scatter(ax, data, spec, palette, single_color)

        # --- Histogram ---
        elif chart_type == "histogram":
            _draw_histogram(ax, data, spec, palette, single_color)

        # --- Box plot ---
        elif chart_type == "box":
            _draw_box(ax, data, spec, palette)

        # --- Violin plot ---
        elif chart_type == "violin":
            _draw_violin(ax, data, spec, palette)

        # --- Heatmap ---
        elif chart_type == "heatmap":
            _draw_heatmap(ax, data, spec, fig)

        # --- Function plot ---
        elif chart_type == "function":
            _draw_function(ax, data, spec, palette, single_color, show_legend)

        # Grid and title
        if show_grid and chart_type not in ("pie", "heatmap"):
            ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Save
        ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{chart_type}_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=CHART_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return f"Chart saved successfully!\nFile: {filepath}\nType: {chart_type}\nTitle: {title}"

    except Exception as e:
        plt.close('all')
        return f"Error generating chart: {str(e)}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_data(chart_type: str, data: dict) -> str | None:
    """Return an error string if data is invalid for chart_type, else None."""
    if chart_type in ("bar", "stacked_bar", "line", "area"):
        if "series" not in data and ("labels" not in data or "values" not in data):
            return f"Error: {chart_type} charts need 'labels' and 'values' in data, or a 'series' array."
    elif chart_type == "pie":
        if "labels" not in data or "values" not in data:
            return "Error: Pie charts need 'labels' and 'values' in data."
    elif chart_type == "scatter":
        if not (("x" in data and "y" in data) or ("labels" in data and "values" in data)):
            return "Error: Scatter charts need 'x' and 'y' in data."
    elif chart_type == "histogram":
        if "values" not in data:
            return "Error: Histograms need 'values' in data."
    elif chart_type in ("box", "violin"):
        if "values" not in data and "series" not in data:
            return f"Error: {chart_type} charts need 'values' or 'series' in data."
    elif chart_type == "heatmap":
        if "matrix" not in data:
            return "Error: Heatmaps need 'matrix' (2D array) in data."
    elif chart_type == "function":
        if "expression" not in data and "expressions" not in data:
            return "Error: Function plots need 'expression' or 'expressions' in data."
    return None


# ---------------------------------------------------------------------------
# Individual chart drawing functions
# ---------------------------------------------------------------------------

def _draw_bar(ax, data, spec, palette, single_color, show_legend, error_data):
    if "series" in data:
        labels = data["labels"]
        x = np.arange(len(labels))
        width = 0.8 / len(data["series"])
        colors = get_colors(palette, len(data["series"]))
        for i, series in enumerate(data["series"]):
            offset = (i - len(data["series"]) / 2 + 0.5) * width
            color = single_color if single_color else colors[i]
            yerr = series.get("error") if series.get("error") else None
            ax.bar(x + offset, series["values"], width, label=series["name"],
                   color=color, yerr=yerr, capsize=3 if yerr else 0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if show_legend:
            ax.legend()
    else:
        labels, values = data["labels"], data["values"]
        color = single_color if single_color else get_colors(palette, 1)[0]
        ax.bar(labels, values, color=color,
               yerr=error_data, capsize=3 if error_data else 0)
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_stacked_bar(ax, data, spec, palette, single_color, show_legend):
    if "series" not in data:
        return _draw_bar(ax, data, spec, palette, single_color, show_legend, None)
    labels = data["labels"]
    colors = get_colors(palette, len(data["series"]))
    bottom = np.zeros(len(labels))
    for i, series in enumerate(data["series"]):
        color = single_color if single_color else colors[i]
        ax.bar(labels, series["values"], label=series["name"], color=color, bottom=bottom)
        bottom += np.array(series["values"])
    if show_legend:
        ax.legend()
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_line(ax, data, spec, palette, single_color, show_legend, error_data):
    labels = data["labels"]
    if "series" in data:
        colors = get_colors(palette, len(data["series"]))
        for i, series in enumerate(data["series"]):
            color = single_color if single_color else colors[i]
            yerr = series.get("error")
            if yerr:
                ax.errorbar(labels, series["values"], yerr=yerr, marker='o',
                            label=series["name"], color=color, capsize=3)
            else:
                ax.plot(labels, series["values"], marker='o', label=series["name"], color=color)
        if show_legend:
            ax.legend()
    else:
        color = single_color if single_color else get_colors(palette, 1)[0]
        if error_data:
            ax.errorbar(labels, data["values"], yerr=error_data, marker='o', color=color, capsize=3)
        else:
            ax.plot(labels, data["values"], marker='o', color=color)
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_area(ax, data, spec, palette, single_color, show_legend):
    labels = data["labels"]
    if "series" in data:
        colors = get_colors(palette, len(data["series"]))
        for i, series in enumerate(data["series"]):
            color = single_color if single_color else colors[i]
            ax.fill_between(range(len(labels)), series["values"], alpha=0.5, label=series["name"], color=color)
            ax.plot(range(len(labels)), series["values"], color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        if show_legend:
            ax.legend()
    else:
        color = single_color if single_color else get_colors(palette, 1)[0]
        ax.fill_between(range(len(labels)), data["values"], alpha=0.5, color=color)
        ax.plot(range(len(labels)), data["values"], color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_pie(ax, data, palette):
    labels, values = data["labels"], data["values"]
    colors = get_colors(palette, len(labels))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')


def _draw_scatter(ax, data, spec, palette, single_color):
    x_vals = data.get("x", data.get("labels", []))
    y_vals = data.get("y", data.get("values", []))
    color = single_color if single_color else get_colors(palette, 1)[0]
    ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=50)
    ax.set_xlabel(spec.get("xlabel", "X"))
    ax.set_ylabel(spec.get("ylabel", "Y"))


def _draw_histogram(ax, data, spec, palette, single_color):
    values = data["values"]
    bins = spec.get("bins", 10)
    color = single_color if single_color else get_colors(palette, 1)[0]
    ax.hist(values, bins=bins, color=color, edgecolor='white', alpha=0.7)
    ax.set_xlabel(spec.get("xlabel", "Value"))
    ax.set_ylabel(spec.get("ylabel", "Frequency"))


def _draw_box(ax, data, spec, palette):
    show_fliers = spec.get("showfliers", True)
    if "series" in data:
        all_values = [s["values"] for s in data["series"]]
        labels = [s["name"] for s in data["series"]]
        bp = ax.boxplot(all_values, tick_labels=labels, patch_artist=True, showfliers=show_fliers)
        colors = get_colors(palette, len(all_values))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        values = data["values"]
        # Single dataset or list of datasets
        if values and isinstance(values[0], list):
            bp = ax.boxplot(values, patch_artist=True, showfliers=show_fliers)
            colors = get_colors(palette, len(values))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            if data.get("labels"):
                ax.set_xticklabels(data["labels"])
        else:
            bp = ax.boxplot([values], patch_artist=True, showfliers=show_fliers)
            bp["boxes"][0].set_facecolor(get_colors(palette, 1)[0])
            bp["boxes"][0].set_alpha(0.7)
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_violin(ax, data, spec, palette):
    if "series" in data:
        all_values = [s["values"] for s in data["series"]]
        labels = [s["name"] for s in data["series"]]
    elif data.get("values") and isinstance(data["values"][0], list):
        all_values = data["values"]
        labels = data.get("labels", [str(i + 1) for i in range(len(all_values))])
    else:
        all_values = [data["values"]]
        labels = [""]

    parts = ax.violinplot(all_values, showmeans=True, showmedians=True)
    colors = get_colors(palette, len(all_values))
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_heatmap(ax, data, spec, fig):
    matrix = np.array(data["matrix"])
    cmap = spec.get("cmap", "viridis")
    annotate = spec.get("annotate", False)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax)

    if data.get("xlabels"):
        ax.set_xticks(range(len(data["xlabels"])))
        ax.set_xticklabels(data["xlabels"])
    if data.get("ylabels"):
        ax.set_yticks(range(len(data["ylabels"])))
        ax.set_yticklabels(data["ylabels"])

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = "white" if val < (matrix.max() + matrix.min()) / 2 else "black"
                ax.text(j, i, f"{val:.1f}" if isinstance(val, float) else str(val),
                        ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))


def _draw_function(ax, data, spec, palette, single_color, show_legend):
    x_range = data.get("x_range", [-10, 10])
    points = data.get("points", 500)
    x = np.linspace(x_range[0], x_range[1], points)

    expressions = data.get("expressions", [data.get("expression")])
    colors = get_colors(palette, len(expressions))

    namespace = {**_FUNC_NAMESPACE, "x": x}

    for i, expr_str in enumerate(expressions):
        try:
            y = eval(expr_str, {"__builtins__": {}}, namespace)
            color = single_color if (single_color and len(expressions) == 1) else colors[i]
            label = expr_str if len(expressions) > 1 else None
            ax.plot(x, y, color=color, label=label, linewidth=2)
        except Exception as e:
            ax.text(0.5, 0.5 - i * 0.1, f"Error plotting '{expr_str}': {e}",
                    transform=ax.transAxes, ha='center', color='red')

    # Reference lines
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)

    if show_legend and len(expressions) > 1:
        ax.legend()
    ax.set_xlabel(spec.get("xlabel", "x"))
    ax.set_ylabel(spec.get("ylabel", "y"))


# ---------------------------------------------------------------------------
# Async wrapper and Tool definition
# ---------------------------------------------------------------------------

async def async_generate_chart(input_str: str) -> str:
    return generate_chart(input_str)


visualization_tool = Tool(
    name="create_chart",
    func=generate_chart,
    coroutine=async_generate_chart,
    description=(
        "Generate charts and graphs from data. Saves PNG images to output/ folder. "
        "\n\nCHART TYPES: bar, stacked_bar, line, area, pie, scatter, histogram, box, violin, heatmap, function"
        "\n\nBASIC FORMAT:"
        '\n{"chart_type": "bar", "title": "My Chart", "data": {"labels": ["A", "B"], "values": [10, 20]}}'
        "\n\nOPTIONS: xlabel, ylabel, color, palette (default/warm/cool/pastel), grid, legend"
        "\n\nERROR BARS: Add \"error\": [1, 2] to data (works on bar and line charts)"
        "\n\nMULTIPLE SERIES: {\"data\": {\"labels\": [...], \"series\": [{\"name\": \"A\", \"values\": [...]}]}}"
        "\n\nBOX/VIOLIN: {\"data\": {\"series\": [{\"name\": \"Group A\", \"values\": [1,2,3,4]}]}}"
        "\n\nHEATMAP: {\"data\": {\"matrix\": [[1,2],[3,4]], \"xlabels\": [...], \"ylabels\": [...]}, \"annotate\": true}"
        "\n\nFUNCTION: {\"data\": {\"expression\": \"sin(x)\", \"x_range\": [-6.28, 6.28]}}"
        "\n\nSTACKED BAR: {\"chart_type\": \"stacked_bar\", \"data\": {\"labels\": [...], \"series\": [...]}}"
    )
)
