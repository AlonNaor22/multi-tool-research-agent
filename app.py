"""Streamlit Web UI for the Multi-Tool Research Agent.

A chat interface that showcases the agent's multi-tool capabilities
with real-time streaming feedback (thinking, tool calls, answers).
"""

import os
import streamlit as st
import time
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from src.agent import ResearchAgent
from src.session_manager import list_sessions
from src.tool_health import format_health_status
from src.observability import MetricsStore
from src.rate_limiter import RateLimitExceeded
from config import ANTHROPIC_API_KEY, MODEL_NAME, API_KEYS, update_env_key


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

        # Run agent with real-time streaming.
        # Uses graph.stream(stream_mode="messages") which yields
        # (AIMessageChunk, metadata) tuples directly — no callback
        # propagation needed. This bypasses the LangChain create_agent
        # limitation where invoke() doesn't forward callbacks to the LLM.
        with st.chat_message("assistant"):
            status_placeholder = st.empty()  # compact one-line tool status
            token_placeholder = st.empty()   # streaming answer text

            agent.timing_callback.reset()
            agent.observability_callback.reset(question=prompt)

            messages = agent.memory.get_messages()
            messages.append(HumanMessage(content=prompt))

            start_time = time.time()
            answer = "No answer was generated."
            streamed_text = ""
            inbox = st.session_state.callback_inbox

            try:
                agent.rate_limiter.check_budget()

                status_placeholder.markdown("*🧠 Thinking...*")
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "thinking",
                    "message": "🧠 Thinking...", "is_error": False,
                })

                for chunk, metadata in agent.agent.stream(
                    {"messages": messages},
                    config={
                        "callbacks": [
                            agent.timing_callback,
                            agent.observability_callback,
                        ],
                        "recursion_limit": 20,
                    },
                    stream_mode="messages",
                ):
                    node = metadata.get("langgraph_node", "")

                    # --- Token streaming from the model node ---
                    if node == "model" and isinstance(chunk, AIMessageChunk):
                        # Anthropic returns content as a list of blocks:
                        # [{"text": "...", "type": "text", "index": 0}]
                        content = chunk.content
                        text_part = ""
                        if isinstance(content, str) and content:
                            text_part = content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_part += block.get("text", "")

                        if text_part:
                            # Reveal word-by-word with a tiny delay so Streamlit
                            # renders each update visually instead of batching them.
                            # This creates a natural typing effect regardless of
                            # whether the API sends individual tokens or larger chunks.
                            words = text_part.split(" ")
                            for i, word in enumerate(words):
                                if i > 0:
                                    streamed_text += " "
                                streamed_text += word
                                token_placeholder.markdown(streamed_text + "▌")
                                if i < len(words) - 1:
                                    time.sleep(0.01)  # 10 ms between words
                            # Clear status when answer starts streaming
                            status_placeholder.empty()

                    # --- Tool results from the tools node ---
                    elif node == "tools" and isinstance(chunk, ToolMessage):
                        tool_name = chunk.name or "tool"
                        ts = datetime.now().strftime("%H:%M:%S")

                        status_placeholder.markdown(f"*🔧 Used {tool_name}*")
                        inbox.append({
                            "time": ts, "type": "tool_start",
                            "message": f"🔧 Using <b>{tool_name}</b>...",
                            "is_error": False,
                        })
                        inbox.append({
                            "time": ts, "type": "tool_end",
                            "message": "✅ Tool finished", "is_error": False,
                        })

                # Finalize: remove cursor, show clean text
                if streamed_text:
                    token_placeholder.markdown(streamed_text)
                    answer = streamed_text
                status_placeholder.empty()

                elapsed = time.time() - start_time

                # Save to memory
                agent.memory.add_exchange(prompt, answer)

                # Persist metrics
                metrics = agent.observability_callback.get_metrics()
                agent.metrics_store.save(metrics)
                agent.rate_limiter.record_tokens(metrics.total_tokens)
                st.session_state.last_metrics = metrics

                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "complete",
                    "message": f"✅ Done in {elapsed:.1f}s",
                    "is_error": False,
                })

            except RateLimitExceeded as e:
                answer = str(e)
                status_placeholder.empty()
                st.error(answer)
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "rate_limit",
                    "message": f"⚠️ Rate limit exceeded: {e}",
                    "is_error": True,
                })

            except Exception as e:
                answer = f"Error: {str(e)}"
                status_placeholder.empty()
                st.error(answer)
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "error",
                    "message": f"⚠️ {str(e)[:200]}",
                    "is_error": True,
                })

        # Save to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Rerun to update the inbox column with collected events
        st.rerun()
