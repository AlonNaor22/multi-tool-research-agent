"""Data visualization tool for the research agent.

This tool generates charts and graphs from data using matplotlib.
Charts are saved as PNG images in the output/ directory.

KEY CONCEPT: Tools with File Output
-----------------------------------
Most tools return text. This tool creates a FILE (the chart image).
The tool returns the file path so the agent can tell the user where to find it.

We use matplotlib with the 'Agg' backend - this means it works without
a display/GUI, which is important for running in a server or CLI.
"""

import os
import json
from datetime import datetime
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from langchain_core.tools import Tool


# Output directory for charts
OUTPUT_DIR = "output"


def ensure_output_dir():
    """Make sure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def generate_chart(input_str: str) -> str:
    """
    Generate a chart from the provided data specification.

    HOW TO USE THIS TOOL:
    --------------------
    The input should be a JSON string with the following structure:

    {
        "chart_type": "bar" | "line" | "pie",
        "title": "Chart Title",
        "data": {
            "labels": ["A", "B", "C"],
            "values": [10, 20, 30]
        },
        "xlabel": "X Axis Label",  (optional, for bar/line)
        "ylabel": "Y Axis Label"   (optional, for bar/line)
    }

    For line charts with multiple series:
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
        # Parse the JSON input
        spec = json.loads(input_str)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input. {str(e)}\n\nExpected format: {{\"chart_type\": \"bar\", \"title\": \"My Chart\", \"data\": {{\"labels\": [...], \"values\": [...]}}}}"

    # Validate required fields
    if "chart_type" not in spec:
        return "Error: Missing 'chart_type'. Must be 'bar', 'line', or 'pie'."
    if "data" not in spec:
        return "Error: Missing 'data' field with labels and values."

    chart_type = spec["chart_type"].lower()
    title = spec.get("title", "Chart")
    data = spec["data"]

    # Validate chart type
    if chart_type not in ["bar", "line", "pie"]:
        return f"Error: Unknown chart_type '{chart_type}'. Use 'bar', 'line', or 'pie'."

    # Validate data structure
    if "labels" not in data:
        return "Error: data must contain 'labels' array."
    if "values" not in data and "series" not in data:
        return "Error: data must contain either 'values' array or 'series' array."

    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = data["labels"]

        if chart_type == "bar":
            values = data["values"]
            ax.bar(labels, values, color='steelblue')
            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif chart_type == "line":
            if "series" in data:
                # Multiple lines
                for series in data["series"]:
                    ax.plot(labels, series["values"], marker='o', label=series["name"])
                ax.legend()
            else:
                # Single line
                ax.plot(labels, data["values"], marker='o', color='steelblue')
            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif chart_type == "pie":
            values = data["values"]
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Makes the pie circular

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
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to free memory

        return f"Chart saved successfully!\nFile: {filepath}\nType: {chart_type}\nTitle: {title}"

    except Exception as e:
        plt.close('all')  # Clean up on error
        return f"Error generating chart: {str(e)}"


# Create the LangChain Tool wrapper
visualization_tool = Tool(
    name="create_chart",
    func=generate_chart,
    description=(
        "Generate charts and graphs from data. Creates PNG image files in the output/ folder. "
        "Input must be a JSON string with: "
        "chart_type ('bar', 'line', or 'pie'), "
        "title (string), "
        "data (object with 'labels' array and 'values' array). "
        "Optional: xlabel, ylabel for axis labels. "
        "Example for bar chart: "
        '{\"chart_type\": \"bar\", \"title\": \"Sales by Region\", \"data\": {\"labels\": [\"North\", \"South\", \"East\", \"West\"], \"values\": [100, 150, 80, 120]}} '
        "Example for pie chart: "
        '{\"chart_type\": \"pie\", \"title\": \"Market Share\", \"data\": {\"labels\": [\"Product A\", \"Product B\", \"Product C\"], \"values\": [45, 30, 25]}}'
    )
)
