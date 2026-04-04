"""Streamlit Web UI for the Multi-Tool Research Agent.

A chat interface that showcases the agent's multi-tool capabilities
with real-time streaming feedback (thinking, tool calls, answers).
"""

import os
import re as _re
import streamlit as st
import time
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from src.agent import ResearchAgent
from src.planner import is_simple_query, ResearchPlan
from src.multi_agent.supervisor import DelegationPlan
from src.session_manager import list_sessions
from src.tool_health import format_health_status
from src.observability import MetricsStore
from src.rate_limiter import RateLimitExceeded
from config import ANTHROPIC_API_KEY, MODEL_NAME, API_KEYS, update_env_key


# ---------------------------------------------------------------------------
# Smart content renderer — handles HTML math blocks and chart images
# ---------------------------------------------------------------------------

def _render_agent_content(text: str, container=None):
    """Render agent output, detecting HTML math blocks and chart file paths.

    - Splits on <!-- MATH_HTML --> sentinels and renders HTML blocks via st.html()
    - Detects CHART_FILE:path markers and renders images via st.image()
    - Passes regular text through st.markdown()
    """
    if container is None:
        container = st

    # Split on MATH_HTML sentinel blocks
    parts = _re.split(
        r'(<!-- MATH_HTML -->.*?<!-- /MATH_HTML -->)',
        text,
        flags=_re.DOTALL,
    )

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith('<!-- MATH_HTML -->'):
            # Extract the HTML content between sentinels
            html_content = part.replace('<!-- MATH_HTML -->', '').replace('<!-- /MATH_HTML -->', '').strip()
            container.html(html_content)

        elif 'CHART_FILE:' in part:
            # Split on chart file references
            chart_parts = _re.split(r'CHART_FILE:([\S]+)', part)
            for i, cp in enumerate(chart_parts):
                cp = cp.strip()
                if not cp:
                    continue
                if i % 2 == 0:
                    container.markdown(cp)
                else:
                    # This is a file path
                    if os.path.exists(cp):
                        container.image(cp, use_container_width=True)
                    else:
                        container.markdown(f"*Chart: {cp}*")
        else:
            container.markdown(part)



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
    st.session_state.research_mode = "Auto"  # Auto / Direct / Plan-and-Execute

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

    # --- Research Mode ---
    st.header("Research Mode")
    mode_options = ["Auto", "Direct", "Plan-and-Execute", "Multi-Agent"]
    research_mode = st.radio(
        "Mode",
        options=mode_options,
        index=mode_options.index(
            st.session_state.get("research_mode", "Auto")
        ),
        help=(
            "**Auto**: uses the complexity detector to choose.\n"
            "**Direct**: always runs the agent without a plan.\n"
            "**Plan-and-Execute**: always generates a multi-step research plan first.\n"
            "**Multi-Agent**: supervisor delegates to specialist agents that run in parallel."
        ),
    )
    st.session_state.research_mode = research_mode

    if research_mode == "Auto":
        st.caption("Simple questions → Direct. Complex ones → Plan-and-Execute.")
    elif research_mode == "Plan-and-Execute":
        st.caption("Every query gets a structured research plan.")
    elif research_mode == "Multi-Agent":
        st.caption("Supervisor delegates to specialist agents (parallel execution).")

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

def _render_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan as a compact markdown string for Streamlit."""
    STATUS_ICON = {"pending": "⏳", "in_progress": "🔄", "done": "✅"}
    lines = ["**Research Plan**", ""]
    for step in plan.steps:
        icon = STATUS_ICON.get(step.status, "⏳")
        tools_hint = f" *(tools: {', '.join(step.expected_tools)})*" if step.expected_tools else ""
        lines.append(f"{icon} **Step {step.step_number}**: {step.description}{tools_hint}")
        if step.status == "done" and step.findings:
            short = step.findings[:200].replace("\n", " ")
            if len(step.findings) > 200:
                short += "…"
            lines.append(f"   > {short}")
    return "\n".join(lines)


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

def _render_delegation_plan(plan: DelegationPlan, specialist_status: Dict[str, str] = None) -> str:
    """Render a DelegationPlan as compact markdown for Streamlit."""
    STATUS_ICON = {"pending": "\u23f3", "running": "\U0001f504", "done": "\u2705"}
    status = specialist_status or {}

    lines = ["**Multi-Agent Delegation Plan**", ""]
    if plan.rationale:
        lines.append(f"*{plan.rationale}*")
        lines.append("")

    for i, phase in enumerate(plan.execution_phases):
        parallel_note = " (parallel)" if len(phase) > 1 else ""
        lines.append(f"**Phase {i + 1}**{parallel_note}")
        for name in phase:
            icon = STATUS_ICON.get(status.get(name, "pending"), "\u23f3")
            task = plan.specialist_tasks.get(name, "")
            task_preview = f": {task[:80]}..." if task and len(task) > 80 else f": {task}" if task else ""
            lines.append(f"  {icon} **{name}**{task_preview}")
        lines.append("")

    return "\n".join(lines)


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
            _render_agent_content(msg["content"])

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

        # Determine which mode to use
        mode = st.session_state.get("research_mode", "Auto")
        use_multi_agent = (mode == "Multi-Agent")
        if mode == "Auto":
            use_plan = not is_simple_query(prompt)
        elif mode == "Plan-and-Execute":
            use_plan = True
        else:
            use_plan = False

        with st.chat_message("assistant"):
            inbox = st.session_state.callback_inbox
            start_time = time.time()
            answer = "No answer was generated."

            try:
                agent.rate_limiter.check_budget()

                # =======================================================
                # MULTI-AGENT MODE
                # =======================================================
                if use_multi_agent:
                    plan_placeholder = st.empty()
                    status_placeholder = st.empty()
                    token_placeholder = st.empty()

                    status_placeholder.markdown("*\U0001f9e0 Supervisor is analyzing the query...*")
                    inbox.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "supervisor",
                        "message": "\U0001f9e0 Supervisor analyzing query...",
                        "is_error": False,
                    })

                    streamed_text = ""
                    specialist_status: Dict[str, str] = {}

                    for event in agent.multi_agent_stream(prompt):
                        etype = event.get("type")

                        if etype == "plan_created":
                            ma_plan = event["plan"]
                            # Initialize all specialist statuses to pending
                            for phase in ma_plan.execution_phases:
                                for name in phase:
                                    specialist_status[name] = "pending"
                            plan_placeholder.markdown(
                                _render_delegation_plan(ma_plan, specialist_status)
                            )
                            parallel_count = sum(
                                1 for p in ma_plan.execution_phases if len(p) > 1
                            )
                            status_placeholder.markdown(
                                f"*\U0001f5fa\ufe0f Delegation plan: "
                                f"{len(ma_plan.execution_phases)} phases, "
                                f"{len(specialist_status)} specialists*"
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "plan",
                                "message": (
                                    f"\U0001f5fa\ufe0f Plan: {len(ma_plan.execution_phases)} phases, "
                                    f"{len(specialist_status)} specialists"
                                ),
                                "is_error": False,
                            })

                        elif etype == "phase_started":
                            phase_specialists = event.get("specialists", [])
                            parallel = len(phase_specialists) > 1
                            note = " (parallel)" if parallel else ""
                            status_placeholder.markdown(
                                f"*\U0001f504 Phase {event['phase_idx'] + 1}{note}: "
                                f"{', '.join(phase_specialists)}*"
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "phase",
                                "message": (
                                    f"\U0001f504 Phase {event['phase_idx'] + 1}{note}: "
                                    f"{', '.join(phase_specialists)}"
                                ),
                                "is_error": False,
                            })

                        elif etype == "specialist_started":
                            name = event.get("specialist", "")
                            specialist_status[name] = "running"
                            plan_placeholder.markdown(
                                _render_delegation_plan(ma_plan, specialist_status)
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "specialist",
                                "message": f"\U0001f504 <b>{name}</b> started",
                                "is_error": False,
                            })

                        elif etype == "specialist_done":
                            name = event.get("specialist", "")
                            specialist_status[name] = "done"
                            plan_placeholder.markdown(
                                _render_delegation_plan(ma_plan, specialist_status)
                            )
                            preview = event.get("result_preview", "")[:100]
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "specialist",
                                "message": f"\u2705 <b>{name}</b> done",
                                "is_error": False,
                            })

                        elif etype == "phase_done":
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "phase",
                                "message": f"\u2705 Phase {event['phase_idx'] + 1} complete",
                                "is_error": False,
                            })

                        elif etype == "synthesis_token":
                            token = event["token"]
                            words = token.split(" ")
                            for wi, word in enumerate(words):
                                if wi > 0:
                                    streamed_text += " "
                                streamed_text += word
                                token_placeholder.markdown(streamed_text + "\u258c")
                                if wi < len(words) - 1:
                                    time.sleep(0.01)
                            plan_placeholder.empty()
                            status_placeholder.empty()

                        elif etype == "done":
                            answer = event.get("answer", streamed_text)

                    if streamed_text:
                        token_placeholder.empty()
                        _render_agent_content(streamed_text, token_placeholder)
                    status_placeholder.empty()

                # =======================================================
                # PLAN-AND-EXECUTE MODE
                # =======================================================
                elif use_plan:
                    plan_placeholder = st.empty()    # live plan panel
                    status_placeholder = st.empty()  # one-line status
                    token_placeholder = st.empty()   # streamed synthesis

                    status_placeholder.markdown("*🗺️ Generating research plan...*")
                    inbox.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "planning",
                        "message": "🗺️ Generating research plan...",
                        "is_error": False,
                    })

                    streamed_text = ""

                    for event in agent.plan_and_execute_stream(prompt):
                        etype = event.get("type")

                        if etype == "plan_created":
                            plan = event["plan"]
                            if plan.is_simple:
                                status_placeholder.markdown("*🧠 Thinking (direct mode)...*")
                                inbox.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "thinking",
                                    "message": "🧠 Simple query — direct mode",
                                    "is_error": False,
                                })
                            else:
                                plan_placeholder.markdown(_render_plan(plan))
                                status_placeholder.markdown("*🔄 Starting research...*")
                                inbox.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "plan",
                                    "message": f"🗺️ Plan: {len(plan.steps)} steps",
                                    "is_error": False,
                                })

                        elif etype == "step_started":
                            plan = event["plan"]
                            step = plan.steps[event["step_idx"]]
                            plan_placeholder.markdown(_render_plan(plan))
                            status_placeholder.markdown(
                                f"*🔄 Step {step.step_number}: {step.description[:60]}...*"
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "step_start",
                                "message": f"🔄 Step {step.step_number}: {step.description[:60]}",
                                "is_error": False,
                            })

                        elif etype == "step_tool":
                            tool_name = event.get("tool_name", "tool")
                            ts = datetime.now().strftime("%H:%M:%S")
                            inbox.append({
                                "time": ts, "type": "tool_call",
                                "message": f"🔧 Tool: <b>{tool_name}</b>",
                                "is_error": False,
                            })

                        elif etype == "step_done":
                            plan = event["plan"]
                            step = plan.steps[event["step_idx"]]
                            plan_placeholder.markdown(_render_plan(plan))
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "step_done",
                                "message": f"✅ Step {step.step_number} done",
                                "is_error": False,
                            })

                        elif etype == "synthesis_token":
                            token = event["token"]
                            # Word-by-word reveal for natural typing effect
                            words = token.split(" ")
                            for wi, word in enumerate(words):
                                if wi > 0:
                                    streamed_text += " "
                                streamed_text += word
                                token_placeholder.markdown(streamed_text + "▌")
                                if wi < len(words) - 1:
                                    time.sleep(0.01)
                            plan_placeholder.empty()
                            status_placeholder.empty()

                        elif etype == "done":
                            answer = event.get("answer", streamed_text)

                    # Finalize
                    if streamed_text:
                        token_placeholder.empty()
                        _render_agent_content(streamed_text, token_placeholder)
                    status_placeholder.empty()

                # =======================================================
                # DIRECT MODE (existing streaming approach — unchanged)
                # =======================================================
                else:
                    status_placeholder = st.empty()
                    token_placeholder = st.empty()

                    agent.timing_callback.reset()
                    agent.observability_callback.reset(question=prompt)

                    messages = agent.memory.get_messages()
                    messages.append(HumanMessage(content=prompt))

                    streamed_text = ""

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
                            content = chunk.content
                            text_part = ""
                            if isinstance(content, str) and content:
                                text_part = content
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text_part += block.get("text", "")

                            if text_part:
                                words = text_part.split(" ")
                                for i, word in enumerate(words):
                                    if i > 0:
                                        streamed_text += " "
                                    streamed_text += word
                                    token_placeholder.markdown(streamed_text + "▌")
                                    if i < len(words) - 1:
                                        time.sleep(0.01)
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

                    if streamed_text:
                        token_placeholder.empty()
                        _render_agent_content(streamed_text, token_placeholder)
                        answer = streamed_text
                    status_placeholder.empty()

                    # Save memory + metrics (direct mode)
                    agent.memory.add_exchange(prompt, answer)
                    metrics = agent.observability_callback.get_metrics()
                    agent.metrics_store.save(metrics)
                    agent.rate_limiter.record_tokens(metrics.total_tokens)
                    st.session_state.last_metrics = metrics

                # ---------------------------------------------------
                elapsed = time.time() - start_time
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "complete",
                    "message": f"✅ Done in {elapsed:.1f}s",
                    "is_error": False,
                })

            except RateLimitExceeded as e:
                answer = str(e)
                st.error(answer)
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "rate_limit",
                    "message": f"⚠️ Rate limit exceeded: {e}",
                    "is_error": True,
                })

            except Exception as e:
                answer = f"Error: {str(e)}"
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
