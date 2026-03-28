"""Streamlit Web UI for the Multi-Tool Research Agent.

A chat interface that showcases the agent's multi-tool capabilities
with real-time streaming feedback (thinking, tool calls, answers).
"""

import streamlit as st
import time
from typing import Any, Dict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

from src.agent import ResearchAgent
from src.session_manager import list_sessions
from src.tool_health import format_health_status
from config import ANTHROPIC_API_KEY, MODEL_NAME


# ---------------------------------------------------------------------------
# Streamlit callback handler — captures events for rendering in the UI
# ---------------------------------------------------------------------------

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Captures agent events (thinking, tool calls, errors) for Streamlit rendering."""

    def __init__(self):
        super().__init__()
        self.events = []  # List of {"type": ..., "data": ...}
        self._tool_depth = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, **kwargs) -> None:
        if self._tool_depth == 0:
            self.events.append({"type": "thinking"})

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        self._tool_depth += 1
        tool_name = serialized.get("name", "unknown_tool")
        self.events.append({"type": "tool_start", "tool": tool_name, "input": str(input_str)[:200]})

    def on_tool_end(self, output: str, **kwargs) -> None:
        self._tool_depth = max(0, self._tool_depth - 1)
        self.events.append({"type": "tool_end", "output": str(output)[:500]})

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        self._tool_depth = max(0, self._tool_depth - 1)
        self.events.append({"type": "tool_error", "error": str(error)[:300]})

    def reset(self):
        self.events = []
        self._tool_depth = 0


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Multi-Tool Research Agent")

# ---------------------------------------------------------------------------
# Check API key
# ---------------------------------------------------------------------------

if not ANTHROPIC_API_KEY:
    st.error("**ANTHROPIC_API_KEY not set.** Add it to your `.env` file and restart.")
    st.stop()

# ---------------------------------------------------------------------------
# Initialize agent in session state
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    with st.spinner("Initializing agent and checking tool health..."):
        st.session_state.agent = ResearchAgent()
    st.session_state.chat_history = []  # List of {"role": ..., "content": ...}
    st.session_state.tool_events = []   # Events from last query for display

agent: ResearchAgent = st.session_state.agent

# ---------------------------------------------------------------------------
# Sidebar — tool status & session management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Agent Info")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Tools: **{len(agent.tools)}** available, {len(agent.disabled_tools)} disabled")

    # Tool health status
    with st.expander("Tool Status", expanded=False):
        all_tool_names = [t.name for t in agent.tools] + agent.disabled_tools
        health_str = format_health_status(agent.tool_health, all_tool_names)
        st.text(health_str)

    st.divider()

    # Session management
    st.header("Sessions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", use_container_width=True):
            if agent.memory.history:
                path = agent.save_session()
                st.success(f"Saved!")
            else:
                st.warning("Nothing to save.")
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            agent.memory.clear()
            agent.current_session_id = None
            st.session_state.chat_history = []
            st.session_state.tool_events = []
            st.rerun()

    # Load session
    sessions = list_sessions()
    if sessions:
        with st.expander("Load Session", expanded=False):
            for s in sessions[:10]:
                label = f"{s['session_id']} ({s['message_count']} msgs)"
                if st.button(label, key=s["session_id"]):
                    if agent.load_session(s["session_id"]):
                        # Rebuild chat_history from agent memory
                        st.session_state.chat_history = []
                        for user_input, agent_output in agent.memory.history:
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            st.session_state.chat_history.append({"role": "assistant", "content": agent_output})
                        st.rerun()

    st.divider()

    # Last query timing
    if agent.timing_callback.tool_times:
        with st.expander("Last Query Timing", expanded=True):
            total = 0.0
            for entry in agent.timing_callback.tool_times:
                dur = entry["duration"]
                total += dur
                st.text(f"  {entry['tool']}: {dur:.2f}s")
            st.text(f"  Total: {total:.2f}s")

# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input and agent execution
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a research question..."):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent with streaming
    with st.chat_message("assistant"):
        # Show tool activity in a status widget
        status = st.status("Researching...", expanded=True)

        # Set up our Streamlit callback
        sl_callback = StreamlitCallbackHandler()
        agent.timing_callback.reset()

        # Build messages and stream
        messages = agent.memory.get_messages()
        messages.append(HumanMessage(content=prompt))

        start_time = time.time()
        final_result = None

        try:
            for chunk in agent.agent.stream(
                {"messages": messages},
                {"callbacks": [agent.timing_callback, sl_callback],
                 "recursion_limit": 20},
                stream_mode="values",
            ):
                final_result = chunk

                # Render new events in the status widget
                for event in sl_callback.events:
                    if event["type"] == "thinking":
                        status.write("🧠 Thinking...")
                    elif event["type"] == "tool_start":
                        status.write(f"🔧 Using **{event['tool']}**...")
                    elif event["type"] == "tool_end":
                        status.write(f"✅ Done")
                    elif event["type"] == "tool_error":
                        status.write(f"⚠️ Error: {event['error']}")
                sl_callback.events.clear()

            elapsed = time.time() - start_time

            # Extract answer
            if final_result:
                answer = agent._extract_answer(final_result)
            else:
                answer = "No answer was generated."

            # Save to memory
            agent.memory.add_exchange(prompt, answer)

            status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)

        except Exception as e:
            answer = f"Error: {str(e)}"
            status.update(label="Error", state="error", expanded=False)

        # Display the answer
        st.markdown(answer)

    # Save to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
