"""Data visualization tool for the research agent.

This tool generates charts and graphs from data using matplotlib.
Charts are saved as PNG images in the output/ directory.

Features:
- Multiple chart types: bar, line, pie, scatter, histogram, area
- Color customization
- Multiple data series support
- Grid and legend options

We use matplotlib with the 'Agg' backend - this means it works without
a display/GUI, which is important for running in a server or CLI.
"""

import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.tools import Tool


# Output directory for charts
OUTPUT_DIR = "output"

# Color palettes
COLOR_PALETTES = {
    "default": ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"],
    "warm": ["#E74C3C", "#E67E22", "#F1C40F", "#D35400", "#C0392B", "#F39C12"],
    "cool": ["#3498DB", "#1ABC9C", "#9B59B6", "#2980B9", "#16A085", "#8E44AD"],
    "pastel": ["#AEC6CF", "#FFB347", "#B39EB5", "#FF6961", "#77DD77", "#FDFD96"],
}


def ensure_output_dir():
    """Make sure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def get_colors(palette: str, count: int) -> list:
    """Get a list of colors from a palette."""
    colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])
    # Cycle colors if we need more than available
    return [colors[i % len(colors)] for i in range(count)]


def generate_chart(input_str: str) -> str:
    """
    Generate a chart from the provided data specification.

    INPUT FORMAT (JSON):
    {
        "chart_type": "bar" | "line" | "pie" | "scatter" | "histogram" | "area",
        "title": "Chart Title",
        "data": {
            "labels": ["A", "B", "C"],
            "values": [10, 20, 30]
        },
        "xlabel": "X Axis Label",     (optional)
        "ylabel": "Y Axis Label",     (optional)
        "color": "#4C72B0",           (optional, single color)
        "palette": "default",         (optional: default, warm, cool, pastel)
        "grid": true,                 (optional, show grid lines)
        "legend": true                (optional, show legend)
    }

    For scatter plots:
    {
        "chart_type": "scatter",
        "title": "Correlation",
        "data": {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 5, 4, 5]
        }
    }

    For histograms:
    {
        "chart_type": "histogram",
        "title": "Distribution",
        "data": {
            "values": [1, 1, 2, 2, 2, 3, 3, 4, 5]
        },
        "bins": 10  (optional)
    }

    For multiple series (line/bar):
    {
        "chart_type": "line",
        "title": "Sales Over Time",
        "data": {
            "labels": ["Jan", "Feb", "Mar"],
            "series": [
                {"name": "Product A", "values": [10, 15, 20]},
                {"name": "Product B", "values": [5, 10, 15]}
            ]
        }
    }

    Args:
        input_str: JSON string with chart specification

    Returns:
        Success message with file path, or error message.
    """
    try:
        spec = json.loads(input_str)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input. {str(e)}\n\nExpected format: {{\"chart_type\": \"bar\", \"title\": \"My Chart\", \"data\": {{\"labels\": [...], \"values\": [...]}}}}"

    # Validate required fields
    if "chart_type" not in spec:
        return "Error: Missing 'chart_type'. Options: bar, line, pie, scatter, histogram, area."
    if "data" not in spec:
        return "Error: Missing 'data' field."

    chart_type = spec["chart_type"].lower()
    title = spec.get("title", "Chart")
    data = spec["data"]
    palette = spec.get("palette", "default")
    show_grid = spec.get("grid", False)
    show_legend = spec.get("legend", True)
    single_color = spec.get("color")

    # Validate chart type
    valid_types = ["bar", "line", "pie", "scatter", "histogram", "area"]
    if chart_type not in valid_types:
        return f"Error: Unknown chart_type '{chart_type}'. Options: {', '.join(valid_types)}"

    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            if "series" in data:
                # Multiple series bar chart
                labels = data["labels"]
                x = np.arange(len(labels))
                width = 0.8 / len(data["series"])
                colors = get_colors(palette, len(data["series"]))

                for i, series in enumerate(data["series"]):
                    offset = (i - len(data["series"]) / 2 + 0.5) * width
                    color = single_color if single_color else colors[i]
                    ax.bar(x + offset, series["values"], width, label=series["name"], color=color)

                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                if show_legend:
                    ax.legend()
            else:
                # Single series bar chart
                labels = data["labels"]
                values = data["values"]
                color = single_color if single_color else get_colors(palette, 1)[0]
                ax.bar(labels, values, color=color)

            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif chart_type == "line":
            labels = data["labels"]

            if "series" in data:
                # Multiple series
                colors = get_colors(palette, len(data["series"]))
                for i, series in enumerate(data["series"]):
                    color = single_color if single_color else colors[i]
                    ax.plot(labels, series["values"], marker='o', label=series["name"], color=color)
                if show_legend:
                    ax.legend()
            else:
                # Single series
                color = single_color if single_color else get_colors(palette, 1)[0]
                ax.plot(labels, data["values"], marker='o', color=color)

            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif chart_type == "area":
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

        elif chart_type == "pie":
            labels = data["labels"]
            values = data["values"]
            colors = get_colors(palette, len(labels))
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')

        elif chart_type == "scatter":
            x_values = data.get("x", data.get("labels", []))
            y_values = data.get("y", data.get("values", []))
            color = single_color if single_color else get_colors(palette, 1)[0]
            ax.scatter(x_values, y_values, color=color, alpha=0.7, s=50)
            ax.set_xlabel(spec.get("xlabel", "X"))
            ax.set_ylabel(spec.get("ylabel", "Y"))

        elif chart_type == "histogram":
            values = data["values"]
            bins = spec.get("bins", 10)
            color = single_color if single_color else get_colors(palette, 1)[0]
            ax.hist(values, bins=bins, color=color, edgecolor='white', alpha=0.7)
            ax.set_xlabel(spec.get("xlabel", "Value"))
            ax.set_ylabel(spec.get("ylabel", "Frequency"))

        # Add grid if requested
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)

        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Ensure output directory exists
        ensure_output_dir()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{chart_type}_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Save the chart
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return f"Chart saved successfully!\nFile: {filepath}\nType: {chart_type}\nTitle: {title}"

    except Exception as e:
        plt.close('all')
        return f"Error generating chart: {str(e)}"


# Create the LangChain Tool wrapper
visualization_tool = Tool(
    name="create_chart",
    func=generate_chart,
    description=(
        "Generate charts and graphs from data. Saves PNG images to output/ folder. "
        "\n\nCHART TYPES: bar, line, pie, scatter, histogram, area"
        "\n\nBASIC FORMAT:"
        '\n{"chart_type": "bar", "title": "My Chart", "data": {"labels": ["A", "B"], "values": [10, 20]}}'
        "\n\nOPTIONS:"
        "\n- xlabel/ylabel: Axis labels"
        "\n- color: Single color (e.g., '#FF5733')"
        "\n- palette: 'default', 'warm', 'cool', 'pastel'"
        "\n- grid: true/false"
        "\n- legend: true/false"
        "\n\nMULTIPLE SERIES:"
        '\n{"chart_type": "line", "data": {"labels": [...], "series": [{"name": "A", "values": [...]}, ...]}}'
        "\n\nSCATTER: {\"data\": {\"x\": [...], \"y\": [...]}}"
        "\n\nHISTOGRAM: {\"data\": {\"values\": [...]}, \"bins\": 10}"
    )
)
