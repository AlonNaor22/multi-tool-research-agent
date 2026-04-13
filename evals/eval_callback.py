"""Evaluation callback handler that tracks which tools the agent selects.

Unlike TimingCallbackHandler (which tracks execution time), this callback
records tool *names* so the eval suite can verify the agent picked the
right tool for each question.
"""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler

# ─── Module overview ───────────────────────────────────────────────
# LangChain callback handler used by the eval suite.  Records tool
# names, inputs, outputs, and errors so the runner can verify that
# the agent selected the correct tool for each test case.
# ───────────────────────────────────────────────────────────────────


class EvalCallbackHandler(BaseCallbackHandler):
    """Records which tools were called during an agent run."""

    def __init__(self):
        super().__init__()
        self.tools_called: List[str] = []
        self.tool_inputs: List[str] = []
        self.tool_outputs: List[str] = []
        self.errors: List[str] = []

    # Takes (serialized, input_str). Appends the tool name and input
    # to their respective lists when a tool invocation begins.
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Record the tool name and input when a tool starts."""
        tool_name = serialized.get("name", "unknown_tool")
        self.tools_called.append(tool_name)
        self.tool_inputs.append(input_str)

    # Takes tool output string. Stores a truncated copy (max 500 chars).
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Record the tool output."""
        self.tool_outputs.append(str(output)[:500])  # Truncate for storage

    # Takes an exception. Appends its string representation to errors.
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Record any tool errors."""
        self.errors.append(str(error))

    # Clears all recorded data so the handler can be reused for the
    # next test case.
    def reset(self):
        """Clear all recorded data for the next test case."""
        self.tools_called = []
        self.tool_inputs = []
        self.tool_outputs = []
        self.errors = []

    # Returns a copy of the tool-name list collected so far.
    def get_tools_called(self) -> List[str]:
        """Return the list of tool names that were called."""
        return list(self.tools_called)
