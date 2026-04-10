"""Callback handlers for tool timing and real-time token streaming."""

import sys
import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler

# Fix Windows console encoding — default 'charmap' codec can't handle Unicode
# characters (emojis, special symbols) that the LLM frequently produces.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TimingCallbackHandler(BaseCallbackHandler):
    """Tracks execution time for each tool call."""

    def __init__(self):
        super().__init__()
        self.tool_times: List[Dict[str, Any]] = []
        self._current_tool_start: Optional[float] = None
        self._current_tool_name: Optional[str] = None

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        self._current_tool_start = time.time()
        self._current_tool_name = serialized.get("name", "unknown_tool")

        print(f"  ⏱️  [{self._current_tool_name}] Starting...")

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        if self._current_tool_start is not None:
            duration = time.time() - self._current_tool_start
            self.tool_times.append({
                "tool": self._current_tool_name,
                "duration": duration,
            })

            print(f"  ✅ [{self._current_tool_name}] Completed in {duration:.2f}s")
            self._current_tool_start = None
            self._current_tool_name = None

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        if self._current_tool_start is not None:
            duration = time.time() - self._current_tool_start
            print(f"  ❌ [{self._current_tool_name}] Failed after {duration:.2f}s: {error}")
            self._current_tool_start = None
            self._current_tool_name = None

    def get_summary(self) -> str:
        """Return a formatted multi-line summary of all tool execution times."""
        if not self.tool_times:
            return "No tools were used."

        lines = ["\n📊 Tool Execution Summary:"]
        lines.append("-" * 40)

        total_time = 0
        for entry in self.tool_times:
            tool = entry["tool"]
            duration = entry["duration"]
            total_time += duration
            lines.append(f"  {tool}: {duration:.2f}s")

        lines.append("-" * 40)
        lines.append(f"  Total tool time: {total_time:.2f}s")

        return "\n".join(lines)

    def reset(self):
        self.tool_times = []
        self._current_tool_start = None
        self._current_tool_name = None


class StreamingCallbackHandler(BaseCallbackHandler):
    """Streams LLM tokens and tool status to stdout in real time."""

    def __init__(self):
        super().__init__()
        self._tool_depth = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: Any,
        **kwargs: Any,
    ) -> None:
        if self._tool_depth == 0:
            print("\n🧠 Thinking...", flush=True)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self._tool_depth == 0:
            sys.stdout.write(token)
            sys.stdout.flush()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        self._tool_depth += 1
        tool_name = serialized.get("name", "unknown_tool")
        print(f"\n🔧 Using {tool_name}...", flush=True)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self._tool_depth = max(0, self._tool_depth - 1)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self._tool_depth = max(0, self._tool_depth - 1)
        print(f"\n⚠️  Tool error: {error}", flush=True)

    def reset(self):
        self._tool_depth = 0
