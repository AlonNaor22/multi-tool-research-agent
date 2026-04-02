"""Streamlit-aware LangChain callback handler for real-time streaming.

The key problem: LangChain callbacks fire on internal threads that don't have
access to Streamlit's script-run context, so any st.write() / st.markdown()
calls inside a callback silently fail.

The solution (from the Streamlit-x-LangGraph-Cookbooks pattern):
1. Capture the current Streamlit script context at creation time.
2. Wrap every on_* callback method with that context so Streamlit
   recognizes the thread and allows UI writes.
"""

import inspect
from typing import Any, Callable, Dict, TypeVar

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from langchain_core.callbacks.base import BaseCallbackHandler


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """Create a callback handler that streams LLM tokens into a Streamlit
    container — with proper script-run context so writes render in real time.

    Tool calls are shown as a single compact status line (not expanded),
    keeping the chat clean. Detailed tool info goes to the callback inbox.

    Args:
        parent_container: The st.container() where output will be rendered.

    Returns:
        A configured BaseCallbackHandler instance.
    """

    class StreamHandler(BaseCallbackHandler):

        def __init__(self, container: DeltaGenerator):
            self.container = container
            # Single placeholder for the compact tool status line
            self.status_placeholder = self.container.empty()
            # Placeholder for streaming tokens (the actual answer)
            self.token_placeholder = self.container.empty()
            self.text = ""
            self._tool_depth = 0

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """Append each token and overwrite the placeholder — gives
            the classic 'typing' effect. Only stream when not inside
            a tool call (i.e., the final answer)."""
            if self._tool_depth == 0:
                self.text += token
                self.token_placeholder.markdown(self.text + "▌")

        def on_llm_end(self, response: Any, **kwargs) -> None:
            """Remove the cursor once generation is complete."""
            if self.text:
                self.token_placeholder.markdown(self.text)

        def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
        ) -> None:
            self._tool_depth += 1
            name = serialized.get("name", "tool")
            # Show a single compact status line — no expanded details
            self.status_placeholder.markdown(
                f"*🔧 Using {name}...*"
            )

        def on_tool_end(self, output: Any, **kwargs: Any) -> None:
            self._tool_depth = max(0, self._tool_depth - 1)
            if self._tool_depth == 0:
                # Clear the status line when all tools are done
                self.status_placeholder.empty()

        def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
            self._tool_depth = max(0, self._tool_depth - 1)
            self.status_placeholder.markdown(
                f"*⚠️ Tool error: {str(error)[:80]}*"
            )

    # --- Inject the Streamlit script-run context into every on_* method ---
    fn_return_type = TypeVar("fn_return_type")

    def add_streamlit_context(
        fn: Callable[..., fn_return_type],
    ) -> Callable[..., fn_return_type]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamHandler(parent_container)

    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith("on_"):
            setattr(st_cb, method_name, add_streamlit_context(method_func))

    return st_cb
