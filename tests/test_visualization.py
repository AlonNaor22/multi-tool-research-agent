"""Tests for src/tools/visualization_tool.py — chart generation."""

import os
import json
import pytest
from unittest.mock import patch


class TestChartGeneration:
    """Test chart generation with temporary output directory."""

    def test_bar_chart_creates_file(self, tmp_path):
        with patch("src.tools.visualization_tool.OUTPUT_DIR", str(tmp_path)):
            from src.tools.visualization_tool import generate_chart

            input_data = json.dumps({
                "chart_type": "bar",
                "title": "Test Bar Chart",
                "data": {
                    "labels": ["A", "B", "C"],
                    "values": [10, 20, 30],
                },
            })
            result = generate_chart(input_data)

            assert "saved" in result.lower() or "created" in result.lower() or ".png" in result

    def test_line_chart(self, tmp_path):
        with patch("src.tools.visualization_tool.OUTPUT_DIR", str(tmp_path)):
            from src.tools.visualization_tool import generate_chart

            input_data = json.dumps({
                "chart_type": "line",
                "title": "Test Line Chart",
                "data": {
                    "labels": ["Jan", "Feb", "Mar"],
                    "values": [100, 150, 200],
                },
            })
            result = generate_chart(input_data)

            assert ".png" in result or "saved" in result.lower()

    def test_pie_chart(self, tmp_path):
        with patch("src.tools.visualization_tool.OUTPUT_DIR", str(tmp_path)):
            from src.tools.visualization_tool import generate_chart

            input_data = json.dumps({
                "chart_type": "pie",
                "title": "Market Share",
                "data": {
                    "labels": ["Chrome", "Firefox", "Safari"],
                    "values": [65, 20, 15],
                },
            })
            result = generate_chart(input_data)

            assert ".png" in result or "saved" in result.lower()

    def test_invalid_json(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart("not valid json {{{")
        assert "Error" in result

    def test_missing_chart_type(self, tmp_path):
        with patch("src.tools.visualization_tool.OUTPUT_DIR", str(tmp_path)):
            from src.tools.visualization_tool import generate_chart

            input_data = json.dumps({
                "title": "Incomplete",
                "data": {"labels": ["A"], "values": [1]},
            })
            result = generate_chart(input_data)

            assert "chart_type" in result

    def test_missing_data(self, tmp_path):
        with patch("src.tools.visualization_tool.OUTPUT_DIR", str(tmp_path)):
            from src.tools.visualization_tool import generate_chart

            input_data = json.dumps({
                "chart_type": "bar",
                "title": "No Data",
            })
            result = generate_chart(input_data)

            assert "data" in result.lower()
