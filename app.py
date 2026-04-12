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
from src.ui_strings import UI
from src.constants import (
    MODE_AUTO, MODE_DIRECT, MODE_PLAN_EXECUTE, MODE_MULTI_AGENT,
    RESEARCH_MODES,
    EVENT_PLAN_CREATED, EVENT_PHASE_STARTED, EVENT_SPECIALIST_STARTED,
    EVENT_SPECIALIST_DONE, EVENT_PHASE_DONE, EVENT_SYNTHESIS_TOKEN,
    EVENT_DONE, EVENT_STEP_STARTED, EVENT_STEP_TOOL, EVENT_STEP_DONE,
    STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_DONE,
)


# ---------------------------------------------------------------------------
# Smart content renderer — handles math formatting and chart images
# ---------------------------------------------------------------------------

def _stream_display(text: str, placeholder) -> None:
    """Display streaming text, hiding raw math content during token generation.

    While the LLM is generating tokens, MATH_STRUCTURED: JSON and LaTeX
    ($...$) look ugly as raw text. This function shows a clean placeholder
    whenever math content is detected mid-stream.
    """
    # Detect raw structured JSON
    if 'MATH_STRUCTURED:' in text:
        idx = text.index('MATH_STRUCTURED:')
        before = text[:idx].strip()
        if before:
            placeholder.markdown(before + "\n\n" + UI.status.formatting_math)
        else:
            placeholder.markdown(UI.status.formatting_math)
        return

    # Detect LaTeX with unclosed delimiters (partial streaming of $...$)
    # Count $ signs — odd count means we're mid-LaTeX expression
    dollar_count = text.count('$') - text.count('\\$')  # exclude escaped
    if dollar_count > 0 and dollar_count % 2 != 0:
        # We're mid-LaTeX — show text up to the last complete expression
        # Find the last unmatched $
        last_dollar = text.rfind('$')
        safe_text = text[:last_dollar].strip()
        if safe_text:
            placeholder.markdown(safe_text + " ...")
        else:
            placeholder.markdown(UI.status.rendering_math)
        return

    placeholder.markdown(text + "▌")

def _render_agent_content(text: str, container=None):
    """Render agent output with math formatting and chart embedding.

    - Auto-formats any raw MATH_STRUCTURED: JSON the agent passed through
    - Detects chart file paths (output/*.png) and renders images inline
    - Detects CHART_FILE:path markers and renders images inline
    - Passes text through st.markdown() (supports KaTeX $...$ natively)
    """
    if container is None:
        container = st

    # --- Fallback: auto-format any MATH_STRUCTURED: the agent didn't format ---
    if 'MATH_STRUCTURED:' in text:
        text = _auto_format_math_structured(text)

    # --- Convert markdown image syntax to st.image for local files ---
    # The agent may use ![alt](output/chart.png) which st.markdown can't render locally
    img_pattern = r'!\[[^\]]*\]\((output[/\\][^\)]+\.png)\)'
    if _re.search(img_pattern, text):
        parts = _re.split(f'({img_pattern})', text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _re.match(r'output[/\\].*\.png$', part):
                filepath = part.replace('\\', '/')
                if os.path.exists(filepath):
                    container.image(filepath, use_container_width=True)
            elif _re.match(img_pattern, part):
                pass  # Skip the full ![...]() match (already handled the path)
            else:
                container.markdown(part)
        return

    # --- Detect chart file paths (CHART_FILE: markers or bare paths) ---
    chart_pattern = r'(?:CHART_FILE:)?(output[/\\]chart[^\s\)\"\']+\.png)'
    if _re.search(chart_pattern, text):
        parts = _re.split(f'({chart_pattern})', text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _re.match(r'output[/\\]chart.*\.png$', part):
                filepath = part.replace('\\', '/')
                if os.path.exists(filepath):
                    container.image(filepath, use_container_width=True)
            else:
                cleaned = part.replace('CHART_FILE:', '').strip()
                if cleaned:
                    container.markdown(cleaned)
        return

    container.markdown(text)


def _auto_format_math_structured(text: str) -> str:
    """Find raw MATH_STRUCTURED: JSON in text and replace with formatted markdown.

    This is the safety net for direct mode: if the LLM includes raw
    MATH_STRUCTURED:{...} in its response instead of calling math_formatter,
    we format it automatically so the user never sees raw JSON.
    """
    import json as _json
    from src.tools.math_formatter import format_math

    result_parts = []
    remaining = text

    while 'MATH_STRUCTURED:' in remaining:
        prefix_marker = 'MATH_STRUCTURED:'
        idx = remaining.index(prefix_marker)

        # Add the text before the marker
        result_parts.append(remaining[:idx])

        # Find the JSON object by counting braces
        json_start = idx + len(prefix_marker)
        if json_start >= len(remaining) or remaining[json_start] != '{':
            result_parts.append(remaining[idx:])
            remaining = ""
            break

        depth = 0
        json_end = json_start
        for i in range(json_start, len(remaining)):
            if remaining[i] == '{':
                depth += 1
            elif remaining[i] == '}':
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        raw_block = remaining[idx:json_end]
        try:
            html = format_math(raw_block)
            result_parts.append(html)
        except Exception:
            # Fallback: try to extract plain_text from the JSON
            try:
                data = _json.loads(remaining[json_start:json_end])
                result_parts.append(data.get("plain_text", raw_block))
            except Exception:
                result_parts.append(raw_block)

        remaining = remaining[json_end:]

    result_parts.append(remaining)
    return "".join(result_parts)



# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=UI.page_title,
    page_icon=UI.page_icon,
    layout="wide",
)

st.title(UI.app_title)

# ---------------------------------------------------------------------------
# Sidebar — API Keys configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header(UI.sidebar.api_keys_header)
    with st.expander(UI.sidebar.manage_api_keys, expanded=not bool(ANTHROPIC_API_KEY)):
        _key_saved = False
        for env_var, info in API_KEYS.items():
            current_value = os.getenv(env_var, "").strip()
            label = info["label"]
            tag = " (required)" if info["required"] else ""

            if current_value:
                st.markdown(UI.sidebar.key_configured_fmt.format(label=label))
            else:
                new_val = st.text_input(
                    f"{label}{tag}",
                    type="password",
                    key=f"apikey_{env_var}",
                    help=UI.sidebar.key_input_help_fmt.format(url=info['url']),
                )
                if st.button(UI.sidebar.save_btn, key=f"save_{env_var}"):
                    if new_val.strip():
                        update_env_key(env_var, new_val.strip())
                        _key_saved = True
                    else:
                        st.warning(UI.sidebar.key_empty_warning)
        if _key_saved:
            st.success(UI.sidebar.key_saved)
            st.rerun()

    st.divider()

# ---------------------------------------------------------------------------
# Check API key
# ---------------------------------------------------------------------------

if not os.getenv("ANTHROPIC_API_KEY", "").strip():
    st.error(UI.errors.api_key_missing)
    st.stop()

# ---------------------------------------------------------------------------
# Initialize agent in session state
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    with st.spinner(UI.initializing):
        st.session_state.agent = ResearchAgent()
    st.session_state.chat_history = []  # List of {"role": ..., "content": ..., "charts": [...]}
    st.session_state.last_metrics = None
    st.session_state.callback_inbox = []  # Callback events for the inbox panel
    st.session_state.research_mode = MODE_AUTO  # Auto / Direct / Plan-and-Execute
    st.session_state.pending_charts = []  # Chart paths from current query

agent: ResearchAgent = st.session_state.agent

# ---------------------------------------------------------------------------
# Sidebar — tool status, observability, session management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header(UI.sidebar.agent_info_header)
    st.caption(UI.sidebar.model_caption_fmt.format(model=MODEL_NAME))
    st.caption(UI.sidebar.tools_caption_fmt.format(available=len(agent.tools), disabled=len(agent.disabled_tools)))

    # Tool health status
    with st.expander(UI.sidebar.tool_status, expanded=False):
        all_tool_names = [t.name for t in agent.tools] + agent.disabled_tools
        health_str = format_health_status(agent.tool_health, all_tool_names)
        st.text(health_str)

    st.divider()

    # --- Observability: Last Query Metrics ---
    last_metrics = st.session_state.last_metrics
    if last_metrics:
        with st.expander(UI.sidebar.last_query_metrics, expanded=True):
            col_in, col_out = st.columns(2)
            col_in.metric(UI.sidebar.input_tokens, f"{last_metrics.input_tokens:,}")
            col_out.metric(UI.sidebar.output_tokens, f"{last_metrics.output_tokens:,}")

            col_cost, col_dur = st.columns(2)
            col_cost.metric(UI.sidebar.est_cost, f"${last_metrics.estimated_cost_usd:.5f}")
            col_dur.metric(UI.sidebar.duration, f"{last_metrics.total_duration_s:.1f}s")

            if last_metrics.tools_called:
                st.caption(UI.sidebar.tool_calls_caption)
                for t in last_metrics.tools_called:
                    icon = "✅" if t["status"] == "success" else "❌"
                    st.text(f"  {icon} {t['name']} ({t['duration_s']:.1f}s)")

    # --- Observability: Performance History ---
    store = MetricsStore()
    summary = store.get_summary_stats()
    if summary["total_queries"] > 0:
        with st.expander(UI.sidebar.performance_history, expanded=False):
            col_q, col_t = st.columns(2)
            col_q.metric(UI.sidebar.total_queries, summary["total_queries"])
            col_t.metric(UI.sidebar.total_cost, f"${summary['total_cost_usd']:.4f}")

            col_avg, col_rate = st.columns(2)
            col_avg.metric(UI.sidebar.avg_tokens, f"{summary['avg_tokens_per_query']:,}")
            col_rate.metric(UI.sidebar.tool_success_rate, f"{summary['tool_success_rate']}%")

            # Tool usage bar chart
            if summary["tool_usage"]:
                st.caption(UI.sidebar.tool_usage_dist)
                tool_df = pd.DataFrame(
                    list(summary["tool_usage"].items()),
                    columns=["Tool", "Calls"]
                ).set_index("Tool")
                st.bar_chart(tool_df)

    st.divider()

    # --- Rate Limiting ---
    st.header(UI.sidebar.rate_limiting_header)
    rl_enabled = st.toggle(UI.sidebar.enable_budget_toggle, value=agent.rate_limiter.enabled)
    rl_budget = st.number_input(
        UI.sidebar.token_budget_label, min_value=1000, max_value=10_000_000,
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
            st.error(UI.sidebar.budget_exhausted_fmt.format(spent=spent, budget=rl_budget))
        elif pct >= 0.8:
            st.warning(UI.sidebar.tokens_remaining_fmt.format(remaining=remaining, budget=rl_budget))
        else:
            st.caption(UI.sidebar.tokens_remaining_fmt.format(remaining=remaining, budget=rl_budget))

    st.divider()

    # --- Research Mode ---
    st.header(UI.sidebar.research_mode_header)
    mode_options = list(RESEARCH_MODES)
    research_mode = st.radio(
        UI.sidebar.mode_label,
        options=mode_options,
        index=mode_options.index(
            st.session_state.get("research_mode", MODE_AUTO)
        ),
        help=UI.sidebar.mode_help,
    )
    st.session_state.research_mode = research_mode

    if research_mode == MODE_AUTO:
        st.caption(UI.sidebar.auto_caption)
    elif research_mode == MODE_PLAN_EXECUTE:
        st.caption(UI.sidebar.plan_caption)
    elif research_mode == MODE_MULTI_AGENT:
        st.caption(UI.sidebar.multi_agent_caption)

    st.divider()

    # Session management
    st.header(UI.sidebar.sessions_header)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(UI.sidebar.save_btn, use_container_width=True):
            if agent.memory.history:
                path = agent.save_session()
                st.success(UI.sidebar.saved)
            else:
                st.warning(UI.sidebar.nothing_to_save)
    with col2:
        if st.button(UI.sidebar.clear_chat_btn, use_container_width=True):
            agent.memory.clear()
            agent.current_session_id = None
            st.session_state.chat_history = []
            st.session_state.last_metrics = None
            st.session_state.callback_inbox = []
            st.rerun()

    # Load session
    sessions = list_sessions()
    if sessions:
        with st.expander(UI.sidebar.load_session, expanded=False):
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
    STATUS_ICON = {STATUS_PENDING: "⏳", STATUS_IN_PROGRESS: "🔄", STATUS_DONE: "✅"}
    lines = [UI.plan.research_plan_title, ""]
    for step in plan.steps:
        icon = STATUS_ICON.get(step.status, "⏳")
        tools_hint = f" *(tools: {', '.join(step.expected_tools)})*" if step.expected_tools else ""
        lines.append(f"{icon} **Step {step.step_number}**: {step.description}{tools_hint}")
        if step.status == STATUS_DONE and step.findings:
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
    STATUS_ICON = {STATUS_PENDING: "\u23f3", STATUS_IN_PROGRESS: "\U0001f504", STATUS_DONE: "\u2705"}
    status = specialist_status or {}

    lines = [UI.plan.delegation_plan_title, ""]
    if plan.rationale:
        lines.append(f"*{plan.rationale}*")
        lines.append("")

    for name in plan.specialists:
        icon = STATUS_ICON.get(status.get(name, STATUS_PENDING), "\u23f3")
        deps = plan.depends_on.get(name, [])
        dep_str = f" *(after {', '.join(deps)})*" if deps else ""
        task = plan.specialist_tasks.get(name, "")
        task_preview = f": {task[:80]}..." if task and len(task) > 80 else f": {task}" if task else ""
        lines.append(f"  {icon} **{name}**{task_preview}{dep_str}")
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
    st.markdown(UI.inbox.title)
    inbox_container = st.container(height=500)
    with inbox_container:
        if st.session_state.callback_inbox:
            html_parts = [_render_inbox_event(ev) for ev in st.session_state.callback_inbox]
            st.markdown("".join(html_parts), unsafe_allow_html=True)
        else:
            st.caption(UI.inbox.no_events)

# ---------------------------------------------------------------------------
# Left column — Chat display + input
# ---------------------------------------------------------------------------

with chat_col:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            _render_agent_content(msg["content"])
            # Render any chart images stored with this message
            for chart_path in msg.get("charts", []):
                if os.path.exists(chart_path):
                    st.image(chart_path, use_container_width=True)

    # -----------------------------------------------------------------------
    # Chat input and agent execution
    # -----------------------------------------------------------------------

    if prompt := st.chat_input(UI.chat.placeholder):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Clear inbox for the new query
        st.session_state.callback_inbox = []

        # Determine which mode to use
        mode = st.session_state.get("research_mode", MODE_AUTO)
        use_multi_agent = (mode == MODE_MULTI_AGENT)
        if mode == MODE_AUTO:
            use_plan = not is_simple_query(prompt)
        elif mode == MODE_PLAN_EXECUTE:
            use_plan = True
        else:
            use_plan = False

        with st.chat_message("assistant"):
            inbox = st.session_state.callback_inbox
            start_time = time.time()
            answer = UI.no_answer

            try:
                agent.rate_limiter.check_budget()

                # =======================================================
                # MULTI-AGENT MODE
                # =======================================================
                if use_multi_agent:
                    plan_placeholder = st.empty()
                    status_placeholder = st.empty()
                    token_placeholder = st.empty()

                    status_placeholder.markdown(UI.status.supervisor_analyzing)
                    inbox.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "supervisor",
                        "message": UI.inbox.supervisor_msg,
                        "is_error": False,
                    })

                    streamed_text = ""
                    specialist_status: Dict[str, str] = {}

                    for event in agent.multi_agent_stream(prompt):
                        etype = event.get("type")

                        if etype == EVENT_PLAN_CREATED:
                            ma_plan = event["plan"]
                            for name in ma_plan.specialists:
                                specialist_status[name] = STATUS_PENDING
                            plan_placeholder.markdown(
                                _render_delegation_plan(ma_plan, specialist_status)
                            )
                            status_placeholder.markdown(
                                UI.status.delegation_plan_fmt.format(phase_count=len(ma_plan.specialists), specialist_count=len(specialist_status))
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "plan",
                                "message": UI.inbox.plan_fmt.format(phase_count=len(ma_plan.specialists), specialist_count=len(specialist_status)),
                                "is_error": False,
                            })

                        elif etype == EVENT_PHASE_STARTED:
                            phase_specialists = event.get("specialists", [])
                            parallel = len(phase_specialists) > 1
                            note = " (parallel)" if parallel else ""
                            status_placeholder.markdown(
                                UI.status.phase_started_fmt.format(phase_number=event['phase_idx'] + 1, note=note, specialists=', '.join(phase_specialists))
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "phase",
                                "message": UI.inbox.phase_fmt.format(phase_number=event['phase_idx'] + 1, note=note, specialists=', '.join(phase_specialists)),
                                "is_error": False,
                            })

                        elif etype == EVENT_SPECIALIST_STARTED:
                            name = event.get("specialist", "")
                            specialist_status[name] = STATUS_IN_PROGRESS
                            plan_placeholder.markdown(
                                _render_delegation_plan(ma_plan, specialist_status)
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "specialist",
                                "message": UI.inbox.specialist_started_fmt.format(name=name),
                                "is_error": False,
                            })

                        elif etype == EVENT_SPECIALIST_DONE:
                            name = event.get("specialist", "")
                            specialist_status[name] = STATUS_DONE
                            plan_placeholder.markdown(
                                _render_delegation_plan(ma_plan, specialist_status)
                            )
                            preview = event.get("result_preview", "")[:100]
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "specialist",
                                "message": UI.inbox.specialist_done_fmt.format(name=name),
                                "is_error": False,
                            })

                        elif etype == EVENT_PHASE_DONE:
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "phase",
                                "message": UI.inbox.phase_complete_fmt.format(phase_number=event['phase_idx'] + 1),
                                "is_error": False,
                            })

                        elif etype == EVENT_SYNTHESIS_TOKEN:
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

                        elif etype == EVENT_DONE:
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

                    status_placeholder.markdown(UI.status.generating_plan)
                    inbox.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "planning",
                        "message": UI.inbox.generating_plan,
                        "is_error": False,
                    })

                    streamed_text = ""
                    math_tool_outputs = []  # Capture math/chart tool outputs

                    for event in agent.plan_and_execute_stream(prompt):
                        etype = event.get("type")

                        if etype == EVENT_PLAN_CREATED:
                            plan = event["plan"]
                            if plan.is_simple:
                                status_placeholder.markdown(UI.status.thinking_direct)
                                inbox.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "thinking",
                                    "message": UI.inbox.simple_query,
                                    "is_error": False,
                                })
                            else:
                                plan_placeholder.markdown(_render_plan(plan))
                                status_placeholder.markdown(UI.status.starting_research)
                                inbox.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "plan",
                                    "message": UI.inbox.plan_steps_fmt.format(step_count=len(plan.steps)),
                                    "is_error": False,
                                })

                        elif etype == EVENT_STEP_STARTED:
                            plan = event["plan"]
                            step = plan.steps[event["step_idx"]]
                            plan_placeholder.markdown(_render_plan(plan))
                            desc_short = step.description[:60]
                            status_placeholder.markdown(
                                UI.status.step_fmt.format(step_number=step.step_number, desc=desc_short)
                            )
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "step_start",
                                "message": UI.inbox.step_started_fmt.format(step_number=step.step_number, desc=desc_short),
                                "is_error": False,
                            })

                        elif etype == EVENT_STEP_TOOL:
                            tool_name = event.get("tool_name", "tool")
                            tool_output = event.get("tool_output", "")
                            ts = datetime.now().strftime("%H:%M:%S")
                            inbox.append({
                                "time": ts, "type": "tool_call",
                                "message": UI.inbox.tool_fmt.format(tool_name=tool_name),
                                "is_error": False,
                            })

                            # Capture math_formatter output for direct rendering
                            if tool_name == "math_formatter" and tool_output:
                                math_tool_outputs.append(("math", tool_output))

                            # Capture chart file paths for image embedding
                            if tool_name == "create_chart" and tool_output and ("output/" in tool_output or "output\\" in tool_output):
                                chart_match = _re.search(r'(output[/\\]\S+\.png)', tool_output)
                                if chart_match:
                                    math_tool_outputs.append(("chart", chart_match.group(1)))

                        elif etype == EVENT_STEP_DONE:
                            plan = event["plan"]
                            step = plan.steps[event["step_idx"]]
                            plan_placeholder.markdown(_render_plan(plan))
                            inbox.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "step_done",
                                "message": UI.inbox.step_done_fmt.format(step_number=step.step_number),
                                "is_error": False,
                            })

                        elif etype == EVENT_SYNTHESIS_TOKEN:
                            token = event["token"]
                            # Word-by-word reveal for natural typing effect
                            words = token.split(" ")
                            for wi, word in enumerate(words):
                                if wi > 0:
                                    streamed_text += " "
                                streamed_text += word
                                _stream_display(streamed_text, token_placeholder)
                                if wi < len(words) - 1:
                                    time.sleep(0.01)
                            plan_placeholder.empty()
                            status_placeholder.empty()

                        elif etype == EVENT_DONE:
                            answer = event.get("answer", streamed_text)

                    # Finalize: render answer + any charts/math from tool outputs
                    token_placeholder.empty()

                    # Collect chart paths for rendering
                    chart_paths = [
                        os.path.abspath(c.replace('\\', '/'))
                        for t, c in math_tool_outputs if t == "chart"
                    ]

                    # Fallback: scan for recent chart files if none were captured
                    if not chart_paths:
                        import glob
                        chart_files = sorted(glob.glob("output/chart_*.png"), key=os.path.getmtime, reverse=True)
                        if chart_files and time.time() - os.path.getmtime(chart_files[0]) < 60:
                            chart_paths = [os.path.abspath(chart_files[0])]

                    # Store charts in session state for persistence across reruns
                    st.session_state.pending_charts = [p for p in chart_paths if os.path.exists(p)]

                    # Render everything in order: text, then charts
                    if streamed_text:
                        _render_agent_content(streamed_text, st)

                    for chart_path in chart_paths:
                        if os.path.exists(chart_path):
                            st.image(chart_path, use_container_width=True)

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
                    math_tool_outputs = []  # Capture math_formatter and chart outputs

                    status_placeholder.markdown(UI.status.thinking)
                    inbox.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "thinking",
                        "message": UI.inbox.thinking, "is_error": False,
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
                                    _stream_display(streamed_text, token_placeholder)
                                    if i < len(words) - 1:
                                        time.sleep(0.01)
                                status_placeholder.empty()

                        # --- Tool results from the tools node ---
                        elif node == "tools" and isinstance(chunk, ToolMessage):
                            tool_name = chunk.name or "tool"
                            tool_content = chunk.content or ""
                            ts = datetime.now().strftime("%H:%M:%S")
                            status_placeholder.markdown(UI.status.used_tool_fmt.format(tool_name=tool_name))
                            inbox.append({
                                "time": ts, "type": "tool_start",
                                "message": UI.inbox.using_tool_fmt.format(tool_name=tool_name),
                                "is_error": False,
                            })
                            inbox.append({
                                "time": ts, "type": "tool_end",
                                "message": UI.inbox.tool_finished, "is_error": False,
                            })

                            # Capture math_formatter output for direct rendering
                            if tool_name == "math_formatter" and tool_content:
                                math_tool_outputs.append(("math", tool_content))

                            # Capture chart file paths for image embedding
                            if tool_name == "create_chart" and ("output/" in tool_content or "output\\" in tool_content):
                                chart_match = _re.search(r'(output[/\\]\S+\.png)', tool_content)
                                if chart_match:
                                    math_tool_outputs.append(("chart", chart_match.group(1)))

                    # --- Final render ---
                    if streamed_text:
                        token_placeholder.empty()
                        # Render the agent's prose text
                        _render_agent_content(streamed_text, token_placeholder)
                        answer = streamed_text

                    # Collect chart paths and store in session state for persistence
                    chart_paths = [
                        os.path.abspath(c.replace('\\', '/'))
                        for t, c in math_tool_outputs if t == "chart"
                    ]
                    st.session_state.pending_charts = [p for p in chart_paths if os.path.exists(p)]

                    # Render captured tool outputs that the agent didn't include
                    for output_type, output_content in math_tool_outputs:
                        if output_type == "math" and "$" in output_content:
                            if output_content[:50] not in (streamed_text or ""):
                                st.markdown("---")
                                st.markdown(output_content)
                        elif output_type == "chart":
                            filepath = os.path.abspath(output_content.replace('\\', '/'))
                            if os.path.exists(filepath):
                                st.image(filepath, use_container_width=True)

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
                    "message": UI.inbox.done_fmt.format(elapsed=elapsed),
                    "is_error": False,
                })

            except RateLimitExceeded as e:
                answer = str(e)
                st.error(answer)
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "rate_limit",
                    "message": UI.inbox.rate_limit_fmt.format(error=e),
                    "is_error": True,
                })

            except Exception as e:
                answer = UI.errors.error_prefix_fmt.format(error=str(e))
                st.error(answer)
                inbox.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "error",
                    "message": UI.inbox.error_fmt.format(error=str(e)[:200]),
                    "is_error": True,
                })

        # Save to chat history (include chart paths if any were generated)
        history_entry = {"role": "assistant", "content": answer}
        if "pending_charts" in st.session_state and st.session_state.pending_charts:
            history_entry["charts"] = st.session_state.pending_charts
            st.session_state.pending_charts = []
        st.session_state.chat_history.append(history_entry)

        # Rerun to update the inbox column with collected events
        st.rerun()
