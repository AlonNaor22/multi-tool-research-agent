"""Observability module for the research agent.

Tracks per-query metrics (token usage, costs, tool success/failure rates)
and persists them to JSONL files for historical analysis.

No external dependencies — uses LangChain callback hooks and local JSON storage.
"""

import os
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import Counter

from langchain_core.callbacks import BaseCallbackHandler


# ---------------------------------------------------------------------------
# Cost tables — price per 1M tokens (as of 2025)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # Sonnet
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    # Haiku
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Opus
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}

# Fallback pricing if model not in table
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}

OBSERVABILITY_DIR = "observability"
METRICS_FILE = os.path.join(OBSERVABILITY_DIR, "metrics.jsonl")


# ---------------------------------------------------------------------------
# QueryMetrics — per-query data
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    """Metrics collected for a single agent query."""

    query_id: str = ""
    timestamp: str = ""
    question: str = ""
    model_name: str = ""

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Timing
    total_duration_s: float = 0.0
    thinking_duration_s: float = 0.0
    tool_duration_s: float = 0.0

    # Tool metrics
    tools_called: List[Dict] = field(default_factory=list)
    tool_success_count: int = 0
    tool_failure_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "QueryMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ObservabilityCallbackHandler
# ---------------------------------------------------------------------------

class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Captures token usage, tool metrics, and timing for each query.

    Designed to run alongside TimingCallbackHandler and StreamingCallbackHandler.
    After the query completes, call get_metrics() to retrieve a QueryMetrics object.
    """

    def __init__(self, model_name: str = ""):
        super().__init__()
        self.model_name = model_name
        self._reset_state()

    def _reset_state(self):
        self._query_start: Optional[float] = None
        self._llm_start: Optional[float] = None
        self._thinking_time: float = 0.0
        self._input_tokens: int = 0
        self._output_tokens: int = 0

        # Tool tracking
        self._current_tool_name: Optional[str] = None
        self._current_tool_start: Optional[float] = None
        self._current_tool_input: str = ""
        self._tools_called: List[Dict] = []
        self._tool_successes: int = 0
        self._tool_failures: int = 0
        self._tool_total_time: float = 0.0

        self._question: str = ""

    def reset(self, question: str = ""):
        """Reset for a new query."""
        self._reset_state()
        self._query_start = time.time()
        self._question = question[:200]

    # --- LLM events ---

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, **kwargs) -> None:
        self._llm_start = time.time()

    def on_llm_end(self, response: Any, **kwargs) -> None:
        # Track thinking time
        if self._llm_start is not None:
            self._thinking_time += time.time() - self._llm_start
            self._llm_start = None

        # Extract token usage from the LLM response
        # LangChain Anthropic provider puts usage in llm_output or generations
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("usage", {})
            self._input_tokens += usage.get("input_tokens", 0)
            self._output_tokens += usage.get("output_tokens", 0)

        # Also check generations for usage_metadata (newer LangChain)
        if hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                        metadata = gen.message.usage_metadata
                        if metadata:
                            self._input_tokens += getattr(metadata, "input_tokens", 0)
                            self._output_tokens += getattr(metadata, "output_tokens", 0)

    # --- Tool events ---

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        self._current_tool_name = serialized.get("name", "unknown")
        self._current_tool_start = time.time()
        self._current_tool_input = str(input_str)[:200]

    def on_tool_end(self, output: str, **kwargs) -> None:
        if self._current_tool_start is not None:
            duration = time.time() - self._current_tool_start
            self._tool_total_time += duration
            self._tools_called.append({
                "name": self._current_tool_name,
                "status": "success",
                "duration_s": round(duration, 2),
                "input_size": len(self._current_tool_input),
                "output_size": len(str(output)[:5000]),
            })
            self._tool_successes += 1
            self._current_tool_start = None

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        if self._current_tool_start is not None:
            duration = time.time() - self._current_tool_start
            self._tool_total_time += duration
            self._tools_called.append({
                "name": self._current_tool_name,
                "status": "error",
                "duration_s": round(duration, 2),
                "error": str(error)[:200],
            })
            self._tool_failures += 1
            self._current_tool_start = None

    # --- Metrics compilation ---

    def get_metrics(self) -> QueryMetrics:
        """Compile all captured data into a QueryMetrics object."""
        total_duration = time.time() - self._query_start if self._query_start else 0.0
        total_tokens = self._input_tokens + self._output_tokens

        # Calculate cost
        pricing = MODEL_PRICING.get(self.model_name, DEFAULT_PRICING)
        cost = (
            self._input_tokens * pricing["input"] / 1_000_000
            + self._output_tokens * pricing["output"] / 1_000_000
        )

        return QueryMetrics(
            query_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            question=self._question,
            model_name=self.model_name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
            total_duration_s=round(total_duration, 2),
            thinking_duration_s=round(self._thinking_time, 2),
            tool_duration_s=round(self._tool_total_time, 2),
            tools_called=self._tools_called,
            tool_success_count=self._tool_successes,
            tool_failure_count=self._tool_failures,
        )


# ---------------------------------------------------------------------------
# MetricsStore — persistent storage
# ---------------------------------------------------------------------------

class MetricsStore:
    """Append-only JSONL store for query metrics."""

    def __init__(self, filepath: str = METRICS_FILE):
        self.filepath = filepath
        self._ensure_dir()

    def _ensure_dir(self):
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def save(self, metrics: QueryMetrics) -> None:
        """Append a single query's metrics to the JSONL file."""
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")

    def load(self, limit: int = 100) -> List[QueryMetrics]:
        """Load the most recent metrics entries."""
        if not os.path.exists(self.filepath):
            return []

        entries = []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(QueryMetrics.from_dict(json.loads(line)))
        except (json.JSONDecodeError, IOError):
            return []

        return entries[-limit:]

    def get_summary_stats(self) -> Dict:
        """Aggregate stats across all stored metrics."""
        entries = self.load(limit=500)
        if not entries:
            return {
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_tokens_per_query": 0,
                "avg_cost_per_query": 0.0,
                "avg_duration_s": 0.0,
                "tool_usage": {},
                "tool_success_rate": 0.0,
            }

        total_tokens = sum(e.total_tokens for e in entries)
        total_cost = sum(e.estimated_cost_usd for e in entries)
        total_duration = sum(e.total_duration_s for e in entries)
        total_successes = sum(e.tool_success_count for e in entries)
        total_failures = sum(e.tool_failure_count for e in entries)
        total_tool_calls = total_successes + total_failures

        # Tool usage distribution
        tool_counter: Counter = Counter()
        for entry in entries:
            for tool in entry.tools_called:
                tool_counter[tool["name"]] += 1

        n = len(entries)
        return {
            "total_queries": n,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_tokens_per_query": total_tokens // n if n else 0,
            "avg_cost_per_query": round(total_cost / n, 6) if n else 0.0,
            "avg_duration_s": round(total_duration / n, 2) if n else 0.0,
            "tool_usage": dict(tool_counter.most_common(20)),
            "tool_success_rate": round(total_successes / total_tool_calls * 100, 1) if total_tool_calls else 100.0,
        }

    def format_summary(self) -> str:
        """Format summary stats as a readable string for CLI."""
        stats = self.get_summary_stats()

        if stats["total_queries"] == 0:
            return "No query metrics recorded yet."

        lines = [
            "\n📊 Agent Performance Summary:",
            "-" * 40,
            f"  Queries:           {stats['total_queries']}",
            f"  Total tokens:      {stats['total_tokens']:,}",
            f"  Avg tokens/query:  {stats['avg_tokens_per_query']:,}",
            f"  Total cost:        ${stats['total_cost_usd']:.4f}",
            f"  Avg cost/query:    ${stats['avg_cost_per_query']:.6f}",
            f"  Avg duration:      {stats['avg_duration_s']:.1f}s",
            f"  Tool success rate: {stats['tool_success_rate']}%",
        ]

        if stats["tool_usage"]:
            lines.append("")
            lines.append("  Most used tools:")
            for tool, count in list(stats["tool_usage"].items())[:10]:
                lines.append(f"    {tool}: {count} calls")

        return "\n".join(lines)


def format_query_metrics(metrics: QueryMetrics) -> str:
    """Format a single query's metrics for CLI display."""
    lines = [
        f"\n📈 Query Metrics:",
        f"  Tokens: {metrics.input_tokens:,} in + {metrics.output_tokens:,} out = {metrics.total_tokens:,} total",
        f"  Cost: ${metrics.estimated_cost_usd:.6f}",
        f"  Duration: {metrics.total_duration_s:.1f}s (thinking: {metrics.thinking_duration_s:.1f}s, tools: {metrics.tool_duration_s:.1f}s)",
        f"  Tools: {metrics.tool_success_count} succeeded, {metrics.tool_failure_count} failed",
    ]
    return "\n".join(lines)
