"""Tests for src/tools/visualization_tool.py — chart generation (all types)."""

import os
import json
import pytest
from unittest.mock import patch


def _gen(chart_spec: dict, tmp_path) -> str:
    """Helper: generate a chart with patched output dir."""
    with patch("src.tools.visualization_tool.OUTPUT_DIR", str(tmp_path)):
        from src.tools.visualization_tool import generate_chart
        return generate_chart(json.dumps(chart_spec))


class TestBasicCharts:
    """Test the original chart types."""

    async def test_bar_chart(self, tmp_path):
        result = _gen({"chart_type": "bar", "title": "Bar", "data": {"labels": ["A", "B"], "values": [10, 20]}}, tmp_path)
        assert "saved" in result.lower()

    async def test_line_chart(self, tmp_path):
        result = _gen({"chart_type": "line", "title": "Line", "data": {"labels": ["Jan", "Feb"], "values": [100, 200]}}, tmp_path)
        assert ".png" in result

    async def test_pie_chart(self, tmp_path):
        result = _gen({"chart_type": "pie", "title": "Pie", "data": {"labels": ["A", "B"], "values": [60, 40]}}, tmp_path)
        assert ".png" in result

    async def test_scatter_chart(self, tmp_path):
        result = _gen({"chart_type": "scatter", "title": "Scatter", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}}, tmp_path)
        assert ".png" in result

    async def test_histogram(self, tmp_path):
        result = _gen({"chart_type": "histogram", "title": "Hist", "data": {"values": [1, 1, 2, 3, 3, 3, 4, 5]}}, tmp_path)
        assert ".png" in result

    async def test_area_chart(self, tmp_path):
        result = _gen({"chart_type": "area", "title": "Area", "data": {"labels": ["A", "B", "C"], "values": [10, 20, 15]}}, tmp_path)
        assert ".png" in result

    async def test_multi_series_bar(self, tmp_path):
        result = _gen({
            "chart_type": "bar", "title": "Multi",
            "data": {"labels": ["A", "B"], "series": [
                {"name": "S1", "values": [10, 20]},
                {"name": "S2", "values": [15, 25]},
            ]}
        }, tmp_path)
        assert ".png" in result


class TestNewChartTypes:
    """Test newly added chart types."""

    async def test_box_plot_single(self, tmp_path):
        result = _gen({
            "chart_type": "box", "title": "Box",
            "data": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        }, tmp_path)
        assert "saved" in result.lower()

    async def test_box_plot_multi_series(self, tmp_path):
        result = _gen({
            "chart_type": "box", "title": "Box Multi",
            "data": {"series": [
                {"name": "Group A", "values": [1, 2, 3, 4, 5]},
                {"name": "Group B", "values": [3, 4, 5, 6, 7]},
            ]}
        }, tmp_path)
        assert ".png" in result

    async def test_violin_plot(self, tmp_path):
        result = _gen({
            "chart_type": "violin", "title": "Violin",
            "data": {"series": [
                {"name": "A", "values": [1, 2, 2, 3, 3, 3, 4, 5]},
                {"name": "B", "values": [2, 3, 3, 4, 4, 4, 5, 6]},
            ]}
        }, tmp_path)
        assert ".png" in result

    async def test_heatmap(self, tmp_path):
        result = _gen({
            "chart_type": "heatmap", "title": "Heatmap",
            "data": {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        }, tmp_path)
        assert ".png" in result

    async def test_heatmap_with_labels_and_annotations(self, tmp_path):
        result = _gen({
            "chart_type": "heatmap", "title": "Annotated",
            "data": {
                "matrix": [[1, 2], [3, 4]],
                "xlabels": ["X1", "X2"],
                "ylabels": ["Y1", "Y2"],
            },
            "annotate": True,
        }, tmp_path)
        assert ".png" in result

    async def test_function_plot_single(self, tmp_path):
        result = _gen({
            "chart_type": "function", "title": "Sine",
            "data": {"expression": "sin(x)", "x_range": [-6.28, 6.28]}
        }, tmp_path)
        assert ".png" in result

    async def test_function_plot_multiple(self, tmp_path):
        result = _gen({
            "chart_type": "function", "title": "Trig",
            "data": {
                "expressions": ["sin(x)", "cos(x)"],
                "x_range": [0, 6.28],
            }
        }, tmp_path)
        assert ".png" in result

    async def test_stacked_bar(self, tmp_path):
        result = _gen({
            "chart_type": "stacked_bar", "title": "Stacked",
            "data": {"labels": ["Q1", "Q2"], "series": [
                {"name": "Product A", "values": [10, 20]},
                {"name": "Product B", "values": [5, 15]},
            ]}
        }, tmp_path)
        assert ".png" in result

    async def test_error_bars_on_bar(self, tmp_path):
        result = _gen({
            "chart_type": "bar", "title": "With Errors",
            "data": {"labels": ["A", "B", "C"], "values": [10, 20, 15], "error": [1, 2, 1.5]}
        }, tmp_path)
        assert ".png" in result

    async def test_error_bars_on_line(self, tmp_path):
        result = _gen({
            "chart_type": "line", "title": "Line Errors",
            "data": {"labels": ["A", "B", "C"], "values": [10, 20, 15], "error": [1, 2, 1.5]}
        }, tmp_path)
        assert ".png" in result


class TestValidation:
    """Test error handling and validation."""

    async def test_invalid_json(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart("not valid json")
        assert "Error" in result

    async def test_missing_chart_type_defaults_to_bar(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart(json.dumps({"data": {"labels": ["A"], "values": [1]}}))
        assert "saved" in result.lower() or ".png" in result

    async def test_missing_data(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart(json.dumps({"chart_type": "bar"}))
        assert "data" in result.lower()

    async def test_unknown_chart_type(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart(json.dumps({"chart_type": "invalid", "data": {}}))
        assert "Unknown" in result or "invalid" in result

    async def test_heatmap_missing_matrix(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart(json.dumps({"chart_type": "heatmap", "data": {"values": [1, 2]}}))
        assert "matrix" in result.lower()

    async def test_function_missing_expression(self):
        from src.tools.visualization_tool import generate_chart
        result = generate_chart(json.dumps({"chart_type": "function", "data": {"x_range": [0, 1]}}))
        assert "expression" in result.lower()
