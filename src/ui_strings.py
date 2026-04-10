"""User-facing strings for the Streamlit web UI.

Organized as a nested class for dot-access (``UI.sidebar.api_keys_header``).
All f-string templates use ``{named}`` placeholders so translators can
reorder variables freely — call ``.format(...)`` at the call site.

To add a new language, copy this file to e.g. ``ui_strings_he.py``, translate
every value, and import the desired module in ``app.py``.
"""


class UI:
    """Top-level container for all user-facing strings."""

    # -- Page config ----------------------------------------------------------
    page_title = "Research Agent"
    page_icon = "\U0001f52c"
    app_title = "\U0001f52c Multi-Tool Research Agent"

    # -- Initialization -------------------------------------------------------
    initializing = "Initializing agent and checking tool health..."
    no_answer = "No answer was generated."

    # -- Sidebar: API Keys ----------------------------------------------------
    class sidebar:
        api_keys_header = "API Keys"
        manage_api_keys = "Manage API Keys"
        key_configured_fmt = "**{label}** &nbsp; Configured"
        key_input_help_fmt = "Get your key at {url}"
        save_btn = "Save"
        key_empty_warning = "Key cannot be empty."
        key_saved = "Key saved! Reloading..."

        # -- Agent Info -------------------------------------------------------
        agent_info_header = "Agent Info"
        model_caption_fmt = "Model: `{model}`"
        tools_caption_fmt = "Tools: **{available}** available, {disabled} disabled"
        tool_status = "Tool Status"

        # -- Observability ----------------------------------------------------
        last_query_metrics = "Last Query Metrics"
        input_tokens = "Input Tokens"
        output_tokens = "Output Tokens"
        est_cost = "Est. Cost"
        duration = "Duration"
        tool_calls_caption = "Tool calls:"
        tool_call_fmt = "  {icon} {name} ({duration_s:.1f}s)"

        performance_history = "Performance History"
        total_queries = "Total Queries"
        total_cost = "Total Cost"
        avg_tokens = "Avg Tokens/Query"
        tool_success_rate = "Tool Success Rate"
        tool_usage_dist = "Tool usage distribution:"

        # -- Rate Limiting ----------------------------------------------------
        rate_limiting_header = "Rate Limiting"
        enable_budget_toggle = "Enable token budget"
        token_budget_label = "Token budget"
        budget_exhausted_fmt = "Budget exhausted: {spent:,} / {budget:,} tokens"
        tokens_remaining_fmt = "Tokens remaining: {remaining:,} / {budget:,}"

        # -- Research Mode ----------------------------------------------------
        research_mode_header = "Research Mode"
        mode_label = "Mode"
        mode_help = (
            "**Auto**: uses the complexity detector to choose.\n"
            "**Direct**: always runs the agent without a plan.\n"
            "**Plan-and-Execute**: always generates a multi-step "
            "research plan first.\n"
            "**Multi-Agent**: supervisor delegates to specialist agents "
            "that run in parallel."
        )
        auto_caption = "Simple questions \u2192 Direct. Complex ones \u2192 Plan-and-Execute."
        plan_caption = "Every query gets a structured research plan."
        multi_agent_caption = "Supervisor delegates to specialist agents (parallel execution)."

        # -- Sessions ---------------------------------------------------------
        sessions_header = "Sessions"
        clear_chat_btn = "Clear Chat"
        load_session = "Load Session"
        session_label_fmt = "{session_id} ({count} msgs)"
        saved = "Saved!"
        nothing_to_save = "Nothing to save."

    # -- Chat -----------------------------------------------------------------
    class chat:
        placeholder = "Ask a research question..."

    # -- Plan rendering -------------------------------------------------------
    class plan:
        research_plan_title = "**Research Plan**"
        step_fmt = "{icon} **Step {step_number}**: {description}{tools_hint}"
        findings_fmt = "   > {short}"

        delegation_plan_title = "**Multi-Agent Delegation Plan**"
        phase_header_fmt = "**Phase {phase_number}**{parallel_note}"
        specialist_line_fmt = "  {icon} **{name}**{task_preview}"
        parallel_note = " (parallel)"

    # -- Status messages (shown in main content area during streaming) ---------
    class status:
        thinking = "*\U0001f9e0 Thinking...*"
        thinking_direct = "*\U0001f9e0 Thinking (direct mode)...*"
        supervisor_analyzing = "*\U0001f9e0 Supervisor is analyzing the query...*"
        generating_plan = "*\U0001f5fa\ufe0f Generating research plan...*"
        starting_research = "*\U0001f504 Starting research...*"
        step_fmt = "*\U0001f504 Step {step_number}: {desc}...*"
        delegation_plan_fmt = (
            "*\U0001f5fa\ufe0f Delegation plan: "
            "{phase_count} phases, {specialist_count} specialists*"
        )
        phase_started_fmt = (
            "*\U0001f504 Phase {phase_number}{note}: {specialists}*"
        )
        used_tool_fmt = "*\U0001f527 Used {tool_name}*"
        formatting_math = "*Formatting math solution...*"
        rendering_math = "*Rendering math...*"

    # -- Callback inbox (right sidebar panel) ---------------------------------
    class inbox:
        title = "#### Callback Inbox"
        no_events = "No events yet. Ask a question to see callback activity."

        supervisor_msg = "\U0001f9e0 Supervisor analyzing query..."
        plan_fmt = "\U0001f5fa\ufe0f Plan: {phase_count} phases, {specialist_count} specialists"
        phase_fmt = "\U0001f504 Phase {phase_number}{note}: {specialists}"
        specialist_started_fmt = "\U0001f504 <b>{name}</b> started"
        specialist_done_fmt = "\u2705 <b>{name}</b> done"
        phase_complete_fmt = "\u2705 Phase {phase_number} complete"

        generating_plan = "\U0001f5fa\ufe0f Generating research plan..."
        simple_query = "\U0001f9e0 Simple query \u2014 direct mode"
        plan_steps_fmt = "\U0001f5fa\ufe0f Plan: {step_count} steps"
        step_started_fmt = "\U0001f504 Step {step_number}: {desc}"
        tool_fmt = "\U0001f527 Tool: <b>{tool_name}</b>"
        step_done_fmt = "\u2705 Step {step_number} done"

        thinking = "\U0001f9e0 Thinking..."
        using_tool_fmt = "\U0001f527 Using <b>{tool_name}</b>..."
        tool_finished = "\u2705 Tool finished"

        done_fmt = "\u2705 Done in {elapsed:.1f}s"
        rate_limit_fmt = "\u26a0\ufe0f Rate limit exceeded: {error}"
        error_fmt = "\u26a0\ufe0f {error}"

    # -- Error messages -------------------------------------------------------
    class errors:
        api_key_missing = (
            "**ANTHROPIC_API_KEY not set.** Enter it in the sidebar "
            "and click Save."
        )
        error_prefix_fmt = "Error: {error}"
