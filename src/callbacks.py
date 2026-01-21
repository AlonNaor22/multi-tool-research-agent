"""Custom callbacks for the research agent.

LangChain Callbacks let you "hook into" events during agent execution.
Events include: tool starts, tool ends, LLM starts, LLM ends, errors, etc.

We use this to track how long each tool takes to execute.
"""

import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler


class TimingCallbackHandler(BaseCallbackHandler):
    """
    A callback handler that tracks execution time for tools.

    How Callbacks Work:
    1. Agent starts running
    2. Agent decides to use a tool
    3. on_tool_start() is called <-- We record start time
    4. Tool executes...
    5. on_tool_end() is called   <-- We calculate duration
    6. Repeat for more tools...
    7. Agent finishes

    This gives us visibility into what's happening inside the agent.
    """

    def __init__(self):
        """Initialize the callback handler."""
        super().__init__()
        # Store timing data for each tool call
        self.tool_times: List[Dict[str, Any]] = []
        # Track when current tool started
        self._current_tool_start: Optional[float] = None
        self._current_tool_name: Optional[str] = None

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """
        Called when a tool starts executing.

        Args:
            serialized: Tool metadata (contains name, description, etc.)
            input_str: The input being passed to the tool
        """
        # Record the start time
        self._current_tool_start = time.time()
        # Get the tool name from serialized data
        self._current_tool_name = serialized.get("name", "unknown_tool")

        print(f"  ⏱️  [{self._current_tool_name}] Starting...")

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """
        Called when a tool finishes executing.

        Args:
            output: The output returned by the tool
        """
        if self._current_tool_start is not None:
            # Calculate how long the tool took
            duration = time.time() - self._current_tool_start

            # Store the timing data
            self.tool_times.append({
                "tool": self._current_tool_name,
                "duration": duration,
            })

            print(f"  ✅ [{self._current_tool_name}] Completed in {duration:.2f}s")

            # Reset for next tool
            self._current_tool_start = None
            self._current_tool_name = None

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when a tool encounters an error."""
        if self._current_tool_start is not None:
            duration = time.time() - self._current_tool_start
            print(f"  ❌ [{self._current_tool_name}] Failed after {duration:.2f}s: {error}")
            self._current_tool_start = None
            self._current_tool_name = None

    def get_summary(self) -> str:
        """
        Get a summary of all tool execution times.

        Returns:
            Formatted string with timing summary
        """
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
        """Reset timing data for a new query."""
        self.tool_times = []
        self._current_tool_start = None
        self._current_tool_name = None
