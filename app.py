"""Streamlit Web UI for the Multi-Tool Research Agent.

A chat interface that showcases the agent's multi-tool capabilities
with real-time streaming feedback (thinking, tool calls, answers).
"""

import asyncio
import os
import queue
import threading
import streamlit as st
import time
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

from src.agent import ResearchAgent
from src.session_manager import list_sessions
from src.tool_health import format_health_status
from src.observability import MetricsStore
from src.rate_limiter import RateLimitExceeded
from config import ANTHROPIC_API_KEY, MODEL_NAME, API_KEYS, update_env_key


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
# Sidebar — API Keys configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("API Keys")
    with st.expander("Manage API Keys", expanded=not bool(ANTHROPIC_API_KEY)):
        _key_saved = False
        for env_var, info in API_KEYS.items():
            current_value = os.getenv(env_var, "").strip()
            label = info["label"]
            tag = " (required)" if info["required"] else ""

            if current_value:
                st.markdown(f"**{label}** &nbsp; Configured")
            else:
                new_val = st.text_input(
                    f"{label}{tag}",
                    type="password",
                    key=f"apikey_{env_var}",
                    help=f"Get your key at {info['url']}",
                )
                if st.button("Save", key=f"save_{env_var}"):
                    if new_val.strip():
                        update_env_key(env_var, new_val.strip())
                        _key_saved = True
                    else:
                        st.warning("Key cannot be empty.")
        if _key_saved:
            st.success("Key saved! Reloading...")
            st.rerun()

    st.divider()

# ---------------------------------------------------------------------------
# Check API key
# ---------------------------------------------------------------------------

if not os.getenv("ANTHROPIC_API_KEY", "").strip():
    st.error("**ANTHROPIC_API_KEY not set.** Enter it in the sidebar and click Save.")
    st.stop()

# ---------------------------------------------------------------------------
# Initialize agent in session state
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    with st.spinner("Initializing agent and checking tool health..."):
        st.session_state.agent = ResearchAgent()
    st.session_state.chat_history = []  # List of {"role": ..., "content": ...}
    st.session_state.last_metrics = None
    st.session_state.callback_inbox = []  # Callback events for the inbox panel

agent: ResearchAgent = st.session_state.agent

# ---------------------------------------------------------------------------
# Sidebar — tool status, observability, session management
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

    # --- Observability: Last Query Metrics ---
    last_metrics = st.session_state.last_metrics
    if last_metrics:
        with st.expander("Last Query Metrics", expanded=True):
            col_in, col_out = st.columns(2)
            col_in.metric("Input Tokens", f"{last_metrics.input_tokens:,}")
            col_out.metric("Output Tokens", f"{last_metrics.output_tokens:,}")

            col_cost, col_dur = st.columns(2)
            col_cost.metric("Est. Cost", f"${last_metrics.estimated_cost_usd:.5f}")
            col_dur.metric("Duration", f"{last_metrics.total_duration_s:.1f}s")

            if last_metrics.tools_called:
                st.caption("Tool calls:")
                for t in last_metrics.tools_called:
                    icon = "✅" if t["status"] == "success" else "❌"
                    st.text(f"  {icon} {t['name']} ({t['duration_s']:.1f}s)")

    # --- Observability: Performance History ---
    store = MetricsStore()
    summary = store.get_summary_stats()
    if summary["total_queries"] > 0:
        with st.expander("Performance History", expanded=False):
            col_q, col_t = st.columns(2)
            col_q.metric("Total Queries", summary["total_queries"])
            col_t.metric("Total Cost", f"${summary['total_cost_usd']:.4f}")

            col_avg, col_rate = st.columns(2)
            col_avg.metric("Avg Tokens/Query", f"{summary['avg_tokens_per_query']:,}")
            col_rate.metric("Tool Success Rate", f"{summary['tool_success_rate']}%")

            # Tool usage bar chart
            if summary["tool_usage"]:
                st.caption("Tool usage distribution:")
                tool_df = pd.DataFrame(
                    list(summary["tool_usage"].items()),
                    columns=["Tool", "Calls"]
                ).set_index("Tool")
                st.bar_chart(tool_df)

    st.divider()

    # --- Rate Limiting ---
    st.header("Rate Limiting")
    rl_enabled = st.toggle("Enable token budget", value=agent.rate_limiter.enabled)
    rl_budget = st.number_input(
        "Token budget", min_value=1000, max_value=10_000_000,
        value=agent.rate_limiter.budget, step=10_000,
        disabled=not rl_enabled,
    )
    agent.rate_limiter.set_config(enabled=rl_enabled, budget=rl_budget)

    if rl_enabled:
        remaining = agent.rate_limiter.tokens_remaining
        spent = agent.rate_limiter.tokens_spent
        pct = agent.rate_limiter.usage_fraction

        st.progress(min(pct, 1.0))
        if pct >= 1.0:
            st.error(f"Budget exhausted: {spent:,} / {rl_budget:,} tokens")
        elif pct >= 0.8:
            st.warning(f"Tokens remaining: {remaining:,} / {rl_budget:,}")
        else:
            st.caption(f"Tokens remaining: {remaining:,} / {rl_budget:,}")

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
            st.session_state.last_metrics = None
            st.session_state.callback_inbox = []
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

# ---------------------------------------------------------------------------
# Helper — render a single callback inbox event as styled HTML
# ---------------------------------------------------------------------------

def _render_inbox_event(event: Dict) -> str:
    """Return an HTML div for one callback inbox event."""
    if event["is_error"]:
        return (
            f'<div style="background:#8B0000; color:#fff; padding:8px 12px;'
            f' border-radius:6px; margin:4px 0; font-size:0.85em;">'
            f'<small style="color:#ffcccc;">{event["time"]}</small> '
            f'{event["message"]}</div>'
        )
    return (
        f'<div style="background:#fff; color:#222; padding:8px 12px;'
        f' border-radius:6px; margin:4px 0; border:1px solid #ddd;'
        f' font-size:0.85em;">'
        f'<small style="color:#888;">{event["time"]}</small> '
        f'{event["message"]}</div>'
    )

# ---------------------------------------------------------------------------
# Main layout — chat (left) + callback inbox (right)
# ---------------------------------------------------------------------------

chat_col, inbox_col = st.columns([3, 1])

# ---------------------------------------------------------------------------
# Right column — Callback Inbox (rendered first so it appears during streaming)
# ---------------------------------------------------------------------------

with inbox_col:
    st.markdown("#### Callback Inbox")
    inbox_container = st.container(height=500)
    with inbox_container:
        if st.session_state.callback_inbox:
            html_parts = [_render_inbox_event(ev) for ev in st.session_state.callback_inbox]
            st.markdown("".join(html_parts), unsafe_allow_html=True)
        else:
            st.caption("No events yet. Ask a question to see callback activity.")

# ---------------------------------------------------------------------------
# Left column — Chat display + input
# ---------------------------------------------------------------------------

with chat_col:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -----------------------------------------------------------------------
    # Chat input and agent execution
    # -----------------------------------------------------------------------

    if prompt := st.chat_input("Ask a research question..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Clear inbox for the new query
        st.session_state.callback_inbox = []

        # Run agent with streaming
        with st.chat_message("assistant"):
            # Compact status bar — only shows the *latest* activity as one line
            status = st.status("Researching...", expanded=True)
            answer_placeholder = st.empty()

            # Set up callbacks
            sl_callback = StreamlitCallbackHandler()
            agent.timing_callback.reset()
            agent.observability_callback.reset(question=prompt)

            # Build messages
            messages = agent.memory.get_messages()
            messages.append(HumanMessage(content=prompt))

            start_time = time.time()
            answer = "No answer was generated."

            # -- Thread-based streaming: run async agent in a background thread,
            #    push events to a queue, and consume them on the Streamlit thread
            #    so status updates actually render in real time.
            token_queue: queue.Queue = queue.Queue()
            _DONE = object()  # sentinel

            def _run_in_thread():
                """Run the async agent loop in a dedicated thread."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(_async_stream())
                    token_queue.put(("result", result))
                except Exception as exc:
                    token_queue.put(("error", exc))
                finally:
                    token_queue.put(("done", _DONE))
                    loop.close()

            async def _async_stream():
                results = []
                async for chunk in agent.agent.astream(
                    {"messages": messages},
                    {"callbacks": [agent.timing_callback, sl_callback,
                                   agent.observability_callback],
                     "recursion_limit": 20},
                    stream_mode="values",
                ):
                    results.append(chunk)
                    # Push callback events to the queue for the main thread
                    for event in sl_callback.events:
                        token_queue.put(("event", event))
                    sl_callback.events.clear()
                return results[-1] if results else None

            try:
                # Check rate limit before starting
                agent.rate_limiter.check_budget()

                # Start the agent in a background thread
                worker = threading.Thread(target=_run_in_thread, daemon=True)
                worker.start()

                # Consume events on the Streamlit thread so UI updates flush
                final_result = None
                error_exc = None

                while True:
                    try:
                        kind, payload = token_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if kind == "event":
                        ts = datetime.now().strftime("%H:%M:%S")
                        if payload["type"] == "thinking":
                            status.update(label="🧠 Thinking...")
                            st.session_state.callback_inbox.append({
                                "time": ts, "type": "thinking",
                                "message": "🧠 Thinking...", "is_error": False,
                            })
                        elif payload["type"] == "tool_start":
                            status.update(label=f"🔧 Using {payload['tool']}...")
                            st.session_state.callback_inbox.append({
                                "time": ts, "type": "tool_start",
                                "message": f"🔧 Using <b>{payload['tool']}</b>...",
                                "is_error": False,
                            })
                        elif payload["type"] == "tool_end":
                            status.update(label="✅ Tool finished")
                            st.session_state.callback_inbox.append({
                                "time": ts, "type": "tool_end",
                                "message": "✅ Tool finished", "is_error": False,
                            })
                        elif payload["type"] == "tool_error":
                            status.update(label=f"⚠️ Error: {payload['error'][:80]}")
                            st.session_state.callback_inbox.append({
                                "time": ts, "type": "tool_error",
                                "message": f"⚠️ Error: {payload['error']}",
                                "is_error": True,
                            })
                    elif kind == "result":
                        final_result = payload
                    elif kind == "error":
                        error_exc = payload
                    elif kind == "done":
                        break

                worker.join(timeout=5)

                if error_exc is not None:
                    raise error_exc

                elapsed = time.time() - start_time

                # Extract answer
                if final_result:
                    answer = agent._extract_answer(final_result)
                else:
                    answer = "No answer was generated."

                # Save to memory
                agent.memory.add_exchange(prompt, answer)

                # Persist and store metrics
                metrics = agent.observability_callback.get_metrics()
                agent.metrics_store.save(metrics)
                agent.rate_limiter.record_tokens(metrics.total_tokens)
                st.session_state.last_metrics = metrics

                status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)

            except RateLimitExceeded as e:
                answer = str(e)
                status.update(label="Rate limit exceeded", state="error", expanded=False)
                st.session_state.callback_inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "rate_limit",
                    "message": f"⚠️ Rate limit exceeded: {e}",
                    "is_error": True,
                })

            except Exception as e:
                answer = f"Error: {str(e)}"
                status.update(label="Error", state="error", expanded=False)
                st.session_state.callback_inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "error",
                    "message": f"⚠️ {str(e)[:200]}",
                    "is_error": True,
                })

            # Display the answer
            answer_placeholder.markdown(answer)

        # Save to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Rerun to update the inbox column with collected events
        st.rerun()
