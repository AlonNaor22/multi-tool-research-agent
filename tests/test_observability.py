"""Tests for src/observability.py — metrics tracking and persistence."""

import os
import json
import time
import tempfile
import pytest
from unittest.mock import MagicMock

from src.observability import (
    QueryMetrics,
    ObservabilityCallbackHandler,
    MetricsStore,
    format_query_metrics,
    MODEL_PRICING,
    DEFAULT_PRICING,
)


class TestQueryMetrics:
    """Tests for the QueryMetrics dataclass."""

    def test_defaults(self):
        m = QueryMetrics()
        assert m.total_tokens == 0
        assert m.estimated_cost_usd == 0.0
        assert m.tools_called == []

    def test_to_dict(self):
        m = QueryMetrics(query_id="abc", input_tokens=100, output_tokens=50)
        d = m.to_dict()
        assert d["query_id"] == "abc"
        assert d["input_tokens"] == 100
        assert isinstance(d, dict)

    def test_from_dict(self):
        data = {"query_id": "xyz", "input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
        m = QueryMetrics.from_dict(data)
        assert m.query_id == "xyz"
        assert m.input_tokens == 200

    def test_from_dict_ignores_unknown_fields(self):
        data = {"query_id": "xyz", "unknown_field": "ignored"}
        m = QueryMetrics.from_dict(data)
        assert m.query_id == "xyz"

    def test_roundtrip(self):
        original = QueryMetrics(
            query_id="test123",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            estimated_cost_usd=0.004,
            tools_called=[{"name": "calculator", "status": "success", "duration_s": 0.5}],
        )
        restored = QueryMetrics.from_dict(original.to_dict())
        assert restored.query_id == original.query_id
        assert restored.input_tokens == original.input_tokens
        assert restored.tools_called == original.tools_called


class TestObservabilityCallbackHandler:
    """Tests for the callback handler."""

    def test_reset_sets_query_start(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset(question="What is AI?")
        assert handler._query_start is not None
        assert handler._question == "What is AI?"

    def test_tool_tracking(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset()

        # Simulate tool call
        handler.on_tool_start({"name": "calculator"}, "2+2")
        time.sleep(0.01)
        handler.on_tool_end("4")

        metrics = handler.get_metrics()
        assert metrics.tool_success_count == 1
        assert metrics.tool_failure_count == 0
        assert len(metrics.tools_called) == 1
        assert metrics.tools_called[0]["name"] == "calculator"
        assert metrics.tools_called[0]["status"] == "success"

    def test_tool_error_tracking(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset()

        handler.on_tool_start({"name": "web_search"}, "query")
        handler.on_tool_error(Exception("timeout"))

        metrics = handler.get_metrics()
        assert metrics.tool_success_count == 0
        assert metrics.tool_failure_count == 1
        assert metrics.tools_called[0]["status"] == "error"

    def test_thinking_time_tracking(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset()

        handler.on_llm_start({}, "prompts")
        time.sleep(0.05)
        handler.on_llm_end(MagicMock(llm_output=None, generations=[]))

        metrics = handler.get_metrics()
        assert metrics.thinking_duration_s >= 0.04

    def test_cost_calculation(self):
        handler = ObservabilityCallbackHandler(model_name="claude-sonnet-4-5-20250929")
        handler.reset()
        handler._input_tokens = 1000
        handler._output_tokens = 500

        metrics = handler.get_metrics()
        # Sonnet: $3/M input + $15/M output
        expected = 1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000
        assert abs(metrics.estimated_cost_usd - expected) < 0.0001

    def test_cost_calculation_unknown_model(self):
        handler = ObservabilityCallbackHandler(model_name="unknown-model")
        handler.reset()
        handler._input_tokens = 1000
        handler._output_tokens = 500

        metrics = handler.get_metrics()
        # Should use DEFAULT_PRICING
        expected = 1000 * DEFAULT_PRICING["input"] / 1_000_000 + 500 * DEFAULT_PRICING["output"] / 1_000_000
        assert abs(metrics.estimated_cost_usd - expected) < 0.0001

    def test_multiple_tool_calls(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset()

        handler.on_tool_start({"name": "calculator"}, "2+2")
        handler.on_tool_end("4")
        handler.on_tool_start({"name": "web_search"}, "AI news")
        handler.on_tool_end("results...")
        handler.on_tool_start({"name": "weather"}, "London")
        handler.on_tool_error(Exception("API key missing"))

        metrics = handler.get_metrics()
        assert metrics.tool_success_count == 2
        assert metrics.tool_failure_count == 1
        assert len(metrics.tools_called) == 3

    def test_duration_tracking(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset()
        time.sleep(0.05)

        metrics = handler.get_metrics()
        assert metrics.total_duration_s >= 0.04

    def test_token_extraction_from_llm_output(self):
        handler = ObservabilityCallbackHandler(model_name="test-model")
        handler.reset()

        mock_response = MagicMock()
        mock_response.llm_output = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        mock_response.generations = []

        handler.on_llm_start({}, "prompts")
        handler.on_llm_end(mock_response)

        metrics = handler.get_metrics()
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 150


class TestMetricsStore:
    """Tests for metrics persistence."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)

            m = QueryMetrics(query_id="test1", input_tokens=100, output_tokens=50, total_tokens=150)
            store.save(m)

            loaded = store.load()
            assert len(loaded) == 1
            assert loaded[0].query_id == "test1"
            assert loaded[0].input_tokens == 100

    def test_multiple_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)

            for i in range(5):
                m = QueryMetrics(query_id=f"q{i}", input_tokens=100 * (i + 1))
                store.save(m)

            loaded = store.load()
            assert len(loaded) == 5

    def test_load_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)

            for i in range(10):
                store.save(QueryMetrics(query_id=f"q{i}"))

            loaded = store.load(limit=3)
            assert len(loaded) == 3
            assert loaded[-1].query_id == "q9"  # Most recent

    def test_load_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)
            loaded = store.load()
            assert loaded == []

    def test_summary_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)

            store.save(QueryMetrics(
                query_id="q1", total_tokens=1000, estimated_cost_usd=0.01,
                total_duration_s=5.0, tool_success_count=2, tool_failure_count=0,
                tools_called=[
                    {"name": "calculator", "status": "success", "duration_s": 0.1},
                    {"name": "web_search", "status": "success", "duration_s": 1.0},
                ],
            ))
            store.save(QueryMetrics(
                query_id="q2", total_tokens=2000, estimated_cost_usd=0.02,
                total_duration_s=10.0, tool_success_count=1, tool_failure_count=1,
                tools_called=[
                    {"name": "calculator", "status": "success", "duration_s": 0.2},
                    {"name": "weather", "status": "error", "duration_s": 0.5},
                ],
            ))

            stats = store.get_summary_stats()
            assert stats["total_queries"] == 2
            assert stats["total_tokens"] == 3000
            assert stats["total_cost_usd"] == 0.03
            assert stats["avg_tokens_per_query"] == 1500
            assert stats["tool_usage"]["calculator"] == 2
            assert stats["tool_success_rate"] == 75.0

    def test_summary_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)

            stats = store.get_summary_stats()
            assert stats["total_queries"] == 0
            assert stats["tool_success_rate"] == 0.0

    def test_format_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "metrics.jsonl")
            store = MetricsStore(filepath=filepath)

            store.save(QueryMetrics(
                query_id="q1", total_tokens=1000, estimated_cost_usd=0.01,
                total_duration_s=5.0, tool_success_count=1, tool_failure_count=0,
            ))

            result = store.format_summary()
            assert "Performance Summary" in result
            assert "1,000" in result


class TestFormatQueryMetrics:
    """Tests for the CLI formatting function."""

    def test_format_output(self):
        m = QueryMetrics(
            input_tokens=500, output_tokens=200, total_tokens=700,
            estimated_cost_usd=0.0045,
            total_duration_s=3.5, thinking_duration_s=1.2, tool_duration_s=2.3,
            tool_success_count=2, tool_failure_count=0,
        )
        result = format_query_metrics(m)
        assert "500" in result
        assert "200" in result
        assert "700" in result
        assert "$0.004500" in result
        assert "2 succeeded" in result
