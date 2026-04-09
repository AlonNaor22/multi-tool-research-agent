"""Main research agent using Claude's native async tool calling.

The agent uses Claude's structured tool-use API to decide which tools to call
and in what order. All I/O is async — the agent uses ainvoke()/astream() for
non-blocking execution, the same pattern used in production agents.

Includes conversation memory for follow-up questions.

Plan-and-Execute mode (Item 12)
--------------------------------
A LangGraph StateGraph with three nodes drives complex research:
  create_plan  → generate a multi-step ResearchPlan
  execute_step → run one step using the existing tool-calling agent
                 and advance the step pointer in the same node
  synthesize   → combine all step findings into a final answer

Simple queries are detected by a complexity heuristic and bypass planning
entirely, falling back to direct-mode streaming (same as before).
"""

import sys
from typing import Annotated, Generator, List, Optional
import operator

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

from src.callbacks import TimingCallbackHandler, StreamingCallbackHandler
from src.observability import ObservabilityCallbackHandler, MetricsStore, format_query_metrics
from src.rate_limiter import RateLimiter
from src.tool_health import check_tool_health, get_available_tools
from config import (
    ANTHROPIC_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_ITERATIONS,
    VERBOSE,
)
from src.tools.calculator_tool import calculator_tool
from src.tools.unit_converter_tool import unit_converter_tool
from src.tools.equation_solver_tool import equation_solver_tool
from src.tools.wikipedia_tool import wikipedia_tool
from src.tools.search_tool import search_tool
from src.tools.weather_tool import weather_tool
from src.tools.news_tool import news_tool
from src.tools.url_tool import url_tool
from src.tools.arxiv_tool import arxiv_tool
from src.tools.python_repl_tool import python_repl_tool
from src.tools.wolfram_tool import wolfram_tool
from src.tools.visualization_tool import visualization_tool
from src.tools.parallel_tool import parallel_tool
from src.tools.currency_tool import currency_tool
from src.tools.youtube_tool import youtube_tool
from src.tools.pdf_tool import pdf_tool
from src.tools.google_scholar_tool import google_scholar_tool
from src.tools.translation_tool import translation_tool
from src.tools.reddit_tool import reddit_tool
from src.tools.wikidata_tool import wikidata_tool
from src.tools.github_tool import github_tool
from src.tools.scraper_tool import scraper_tool
from src.tools.csv_tool import csv_tool
from src.tools.datetime_tool import datetime_tool
from src.tools.math_formatter import math_formatter_tool


# Tool categories for hierarchical selection — included in the system prompt
# so the LLM can navigate 20 tools effectively.
TOOL_CATEGORIES = {
    "MATH & COMPUTATION": {
        "tools": ["calculator", "unit_converter", "equation_solver", "currency_converter", "wolfram_alpha", "datetime_calculator", "math_formatter", "create_chart"],
        "guidance": "Use calculator for arithmetic, step-by-step solutions (derivatives, integrals, equations, matrix ops). When calculator returns MATH_STRUCTURED: output, ALWAYS pass it to math_formatter. When the user asks to graph/plot a function or when a visual would help, use create_chart with chart_type 'function'. Use equation_solver for symbolic algebra, systems, eigenvalues, RREF. unit_converter for unit changes, currency_converter for exchange rates, datetime_calculator for date arithmetic. wolfram_alpha is for REFERENCE DATA lookups only."
    },
    "INFORMATION RETRIEVAL": {
        "tools": ["web_search", "wikipedia", "news_search", "arxiv_search", "youtube_search", "google_scholar", "github_search"],
        "guidance": "web_search for general web results from diverse sources. news_search for journalism/news articles with dates and sources. wikipedia for explanations, history, and context. arxiv_search for latest STEM pre-prints (unpublished). google_scholar for published peer-reviewed papers with citations. youtube_search for videos/tutorials. github_search for code repositories and open-source projects."
    },
    "WEB CONTENT": {
        "tools": ["fetch_url", "pdf_reader", "web_scraper"],
        "guidance": "fetch_url reads a page/PDF as plain text. pdf_reader handles complex multi-column PDFs better (use for academic papers). web_scraper extracts structured data — tables, lists, links — from HTML pages. Choose based on what you need: text -> fetch_url, structure -> web_scraper, complex PDF -> pdf_reader."
    },
    "SOCIAL & DISCUSSION": {
        "tools": ["reddit_search"],
        "guidance": "Use reddit_search for community opinions, discussions, experiences, and recommendations. Great for 'what do people think about X' or 'best X for Y' questions."
    },
    "KNOWLEDGE BASE": {
        "tools": ["wikidata"],
        "guidance": "Use wikidata for structured entity facts — population, GDP, coordinates, founding dates, relationships. Different from wikipedia (which gives explanations) and wolfram_alpha (which gives scientific/physical constants)."
    },
    "TRANSLATION": {
        "tools": ["translate"],
        "guidance": "Use translate when you encounter non-English text or need to translate content for the user. Supports 100+ languages."
    },
    "CODE EXECUTION": {
        "tools": ["python_repl"],
        "guidance": "Use for complex calculations, data manipulation, algorithms, or when other tools are insufficient."
    },
    "VISUALIZATION": {
        "tools": ["create_chart"],
        "guidance": "Use to visualize data as bar, line, or pie charts."
    },
    "DATA FILES": {
        "tools": ["csv_reader"],
        "guidance": "Use csv_reader to read and analyze CSV/Excel/TSV files — get column info, statistics, sample data, filtering, and aggregation."
    },
    "MULTI-SOURCE": {
        "tools": ["parallel_search"],
        "guidance": "Use when you need to gather information from multiple sources at once for efficiency."
    },
    "WEATHER": {
        "tools": ["weather"],
        "guidance": "Use for weather forecasts and current conditions."
    },
}


def _build_system_prompt(disabled_tools: list = None) -> str:
    """Build the system prompt with tool selection guidance.

    Args:
        disabled_tools: List of tool names that are unavailable (missing API keys, etc.).
                        These are removed from the category listings so the LLM doesn't try them.
    """
    disabled = set(disabled_tools or [])

    lines = [
        "You are a helpful research assistant with access to various tools.",
        "Your goal is to answer questions thoroughly by gathering information from multiple sources when needed.",
        "",
        "TOOL SELECTION PROCESS:",
        "1. Identify what TYPE of task you need (math? information lookup? code execution?)",
        "2. Look at the matching CATEGORY below",
        "3. Read the category guidance to pick the right tool",
        "4. Choose the most specific tool for your need",
        "",
    ]

    # Build categories, filtering out disabled tools
    for category_name, category_info in TOOL_CATEGORIES.items():
        available_tools = [t for t in category_info['tools'] if t not in disabled]
        if not available_tools:
            continue  # Skip entire category if all its tools are disabled
        lines.append(f"## {category_name}")
        lines.append(f"Tools: {', '.join(available_tools)}")
        lines.append(f"Guidance: {category_info['guidance']}")
        lines.append("")

    lines.extend([
        "Important guidelines:",
        "- Always identify the CATEGORY first, then select the tool",
        "- Use multiple tools when necessary to gather comprehensive information",
        "- For calculations: use calculator (simple) or python_repl (complex) - never do math in your head",
        "- For facts: prefer wikipedia (established) over web_search (current/recent)",
        "- If the user asks a follow-up question, use the conversation history for context",
        "- Synthesize information from multiple sources into a coherent answer",
        "",
        "MATH WORKFLOW:",
        "- When calculator returns output starting with MATH_STRUCTURED:, pass the ENTIRE output to math_formatter",
        "- MANDATORY: If the user says 'graph', 'plot', 'show', 'visualize', or asks to SEE a function, you MUST call create_chart. Do NOT just describe the graph in words.",
        "- MANDATORY: When solving equations with roots (e.g. x^2 - 4 = 0), also graph the function to show where it crosses the x-axis.",
        "- create_chart example: {\"chart_type\":\"function\",\"data\":{\"expression\":\"x**2 - 4\",\"x_range\":[-5,5]},\"title\":\"f(x) = x^2 - 4\"}",
        "- The chart file path from create_chart output will be automatically embedded as an image in the UI.",
        "",
        "ERROR RECOVERY:",
        "- If a tool returns an error, do NOT repeat the same call. Try an alternative tool from the same category.",
        "- If a search returns no results, try a shorter or broader query before giving up.",
        "- Fallback options: if wolfram_alpha is unavailable, use calculator or python_repl instead.",
        "- If web_search fails, try wikipedia or news_search for the same information.",
        "- Always give the user a useful answer, even if some tools are unavailable.",
    ])

    return "\n".join(lines)


# Default system prompt (no disabled tools). Overridden per-instance in __init__
# if the health check finds disabled tools.
SYSTEM_PROMPT = _build_system_prompt()


class SimpleMemory:
    """
    A simple conversation memory implementation.

    Stores ALL conversation exchanges for saving, but only uses the last k
    exchanges for the prompt (to avoid context overflow).
    """

    def __init__(self, k: int = 5):
        """
        Initialize the memory.

        Args:
            k: Number of recent exchanges to include in prompt (default: 5)
        """
        self.k = k
        self.history = []  # List of ALL (input, output) tuples

    def add_exchange(self, user_input: str, agent_output: str):
        """Add a conversation exchange to memory."""
        self.history.append((user_input, agent_output))

    def get_messages(self) -> list:
        """Get the recent conversation history as LangChain messages (last k exchanges)."""
        if not self.history:
            return []

        recent_history = self.history[-self.k:]
        messages = []
        for user_input, agent_output in recent_history:
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content=agent_output))
        return messages

    def get_history_string(self) -> str:
        """Get the recent conversation history as a string (for display/compatibility)."""
        if not self.history:
            return "No previous conversation."

        recent_history = self.history[-self.k:]
        lines = []
        for user_input, agent_output in recent_history:
            lines.append(f"Human: {user_input}")
            lines.append(f"Assistant: {agent_output}")
        return "\n".join(lines)

    def clear(self):
        """Clear all conversation history."""
        self.history = []

    @property
    def buffer(self) -> str:
        """Get the memory buffer (for compatibility)."""
        return self.get_history_string()


class ResearchAgent:
    """
    A research agent using Claude's native tool calling.

    Uses LangChain's create_agent (LangGraph-based) which leverages Claude's
    structured tool-use API instead of text-based ReAct parsing. This means:
    - Tool calls are structured JSON, not fragile text parsing
    - The LLM natively understands tool schemas
    - More reliable tool selection and argument passing
    """

    def __init__(self):
        """Initialize the agent with memory and health-checked tools."""
        # Initialize Claude as the LLM
        self.llm = ChatAnthropic(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY,
            streaming=True,
        )

        # Collect all our tools
        all_tools = [
            # Math & Computation
            calculator_tool,
            unit_converter_tool,
            equation_solver_tool,
            currency_tool,
            wolfram_tool,
            math_formatter_tool,
            # Information Retrieval
            wikipedia_tool,
            search_tool,
            news_tool,
            arxiv_tool,
            youtube_tool,
            google_scholar_tool,
            # Web Content
            url_tool,
            pdf_tool,
            # Social & Discussion
            reddit_tool,
            # Knowledge Base
            wikidata_tool,
            # Translation
            translation_tool,
            # Code Execution
            python_repl_tool,
            # Visualization
            visualization_tool,
            # Multi-Source
            parallel_tool,
            # Weather
            weather_tool,
            # GitHub
            github_tool,
            # Web Scraping
            scraper_tool,
            # Data Files
            csv_tool,
            # Date/Time
            datetime_tool,
        ]

        # Run health check and filter out tools with missing dependencies.
        # Disabled tools are removed so the agent never wastes a turn calling them.
        self.tool_health = check_tool_health()
        self.tools, self.disabled_tools = get_available_tools(all_tools, self.tool_health)

        # Create our simple conversation memory
        self.memory = SimpleMemory(k=5)

        # Track current session ID (for saving to the same file)
        self.current_session_id = None

        # Create callback handlers
        self.timing_callback = TimingCallbackHandler()
        self.streaming_callback = StreamingCallbackHandler()
        self.observability_callback = ObservabilityCallbackHandler(model_name=MODEL_NAME)
        self.metrics_store = MetricsStore()
        self.rate_limiter = RateLimiter()

        # Build system prompt with disabled tools removed from categories
        system_prompt = _build_system_prompt(disabled_tools=self.disabled_tools)

        # Create the agent using native tool calling (LangGraph-based)
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=VERBOSE,
        )

    async def query(self, question: str, show_timing: bool = True) -> str:
        """
        Run a research query asynchronously and return the answer.
        Memory is automatically updated.

        Args:
            question: The research question to answer.
            show_timing: Whether to print timing summary (default: True)

        Returns:
            The agent's final answer as a string.
        """
        try:
            # Check rate limit before starting
            self.rate_limiter.check_budget()

            # Reset callbacks for new query
            self.timing_callback.reset()
            self.observability_callback.reset(question=question)

            # Build messages: conversation history + new question
            messages = self.memory.get_messages()
            messages.append(HumanMessage(content=question))

            # Run the agent with native async tool calling
            result = await self.agent.ainvoke(
                {"messages": messages},
                {"callbacks": [self.timing_callback, self.observability_callback],
                 "recursion_limit": MAX_ITERATIONS * 2},
            )

            # Extract the final answer from the last AI message
            answer = self._extract_answer(result)

            # Save this exchange to memory
            self.memory.add_exchange(question, answer)

            # Persist observability metrics and update rate limiter
            metrics = self.observability_callback.get_metrics()
            self.metrics_store.save(metrics)
            self.rate_limiter.record_tokens(metrics.total_tokens)

            # Print timing summary if requested
            if show_timing:
                print(self.timing_callback.get_summary())
                print(format_query_metrics(metrics))

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    async def stream_query(self, question: str, show_timing: bool = True) -> str:
        """
        Run a research query with real-time async streaming output.

        Instead of blocking until the full answer is ready, this streams
        intermediate steps (thinking, tool calls) and the final answer
        token-by-token so the user sees progress in real-time.

        Args:
            question: The research question to answer.
            show_timing: Whether to print timing summary (default: True)

        Returns:
            The agent's final answer as a string.
        """
        try:
            # Check rate limit before starting
            self.rate_limiter.check_budget()

            # Reset state from previous query
            self.timing_callback.reset()
            self.streaming_callback.reset()
            self.observability_callback.reset(question=question)

            # Build messages: conversation history + new question
            messages = self.memory.get_messages()
            messages.append(HumanMessage(content=question))

            # Stream the agent execution asynchronously
            final_result = None
            async for chunk in self.agent.astream(
                {"messages": messages},
                {"callbacks": [self.timing_callback, self.streaming_callback,
                               self.observability_callback],
                 "recursion_limit": MAX_ITERATIONS * 2},
                stream_mode="values",
            ):
                final_result = chunk

            # Extract the final answer
            answer = self._extract_answer(final_result) if final_result else "No answer was generated."

            # Save this exchange to memory
            self.memory.add_exchange(question, answer)

            # Persist observability metrics and update rate limiter
            metrics = self.observability_callback.get_metrics()
            self.metrics_store.save(metrics)
            self.rate_limiter.record_tokens(metrics.total_tokens)

            # Print timing summary if requested
            if show_timing:
                print(self.timing_callback.get_summary())
                print(format_query_metrics(metrics))

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    def _extract_answer(self, result: dict) -> str:
        """Extract the final text answer from the agent result."""
        messages = result.get("messages", [])
        # Walk backwards to find the last AI message with text content
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Content can be a string or a list of blocks
                if isinstance(msg.content, str):
                    return msg.content
                # If it's a list of content blocks, extract text blocks
                if isinstance(msg.content, list):
                    text_parts = [
                        block["text"] for block in msg.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if text_parts:
                        return "\n".join(text_parts)
        return "No answer was generated."

    def get_last_timing(self) -> str:
        """Get the timing summary from the last query."""
        return self.timing_callback.get_summary()

    def get_last_metrics(self):
        """Get the observability metrics from the last query."""
        return self.observability_callback.get_metrics()

    def clear_memory(self):
        """Clear the conversation history, reset rate limiter, and start a new session."""
        self.memory.clear()
        self.rate_limiter.reset()
        self.current_session_id = None  # Next save will create a new file
        print("Conversation memory cleared.")

    def get_memory(self) -> str:
        """Get the current conversation history."""
        return self.memory.buffer

    def save_session(self, session_id: str = None, description: str = None) -> str:
        """
        Save the current session to a file.

        Uses the same session file throughout a session. Only creates
        a new file on the first save or if explicitly given a new ID.

        Args:
            session_id: Optional session ID. If None, reuses current or generates new.
            description: Short description for new sessions (3 words max).

        Returns:
            Path to the saved session file.
        """
        from src.session_manager import save_session

        # Reuse current session ID if we have one and none was provided
        if session_id is None and self.current_session_id is not None:
            session_id = self.current_session_id

        filepath = save_session(self.memory.history, session_id, description)

        # Store the session ID for future saves (extract from filepath if new)
        if self.current_session_id is None:
            # Extract session ID from the filepath
            import os
            filename = os.path.basename(filepath)
            self.current_session_id = filename.replace('.json', '')

        return filepath

    def load_session(self, session_id: str) -> bool:
        """
        Load a previously saved session.

        Args:
            session_id: The session ID to load.

        Returns:
            True if loaded successfully, False otherwise.
        """
        from src.session_manager import load_session
        history = load_session(session_id)

        if history is None:
            return False

        # Restore the history to memory
        self.memory.history = list(history)

        # Track this as the current session (so future saves go to same file)
        self.current_session_id = session_id

        return True


    # ------------------------------------------------------------------
    # Multi-Agent Orchestration
    # ------------------------------------------------------------------

    def _get_orchestrator(self):
        """Lazily build the multi-agent orchestrator on first use."""
        if not hasattr(self, '_multi_agent_orchestrator') or self._multi_agent_orchestrator is None:
            from src.multi_agent.orchestrator import MultiAgentOrchestrator
            self._multi_agent_orchestrator = MultiAgentOrchestrator(
                llm=self.llm,
                all_tools=self.tools,
                tool_health=self.tool_health,
                callbacks=[self.timing_callback, self.observability_callback],
            )
        return self._multi_agent_orchestrator

    async def multi_agent_query(self, query: str, verbose: bool = True) -> str:
        """Run a query using multi-agent orchestration (for CLI).

        The supervisor delegates sub-tasks to specialist agents, which
        run in parallel when independent.  Results are synthesized into
        a final answer.

        Args:
            query:   The research question.
            verbose: Whether to print progress to stdout.

        Returns:
            The final synthesized answer.
        """
        orchestrator = self._get_orchestrator()
        if verbose:
            answer = await orchestrator.run_verbose(query)
        else:
            answer = await orchestrator.run(query)
        self.memory.add_exchange(query, answer)
        return answer

    def multi_agent_stream(self, query: str):
        """Sync generator for Streamlit — yields typed multi-agent events.

        Event types: plan_created, phase_started, specialist_started,
        specialist_done, phase_done, synthesis_token, done.
        """
        orchestrator = self._get_orchestrator()
        final_answer = ""
        for event in orchestrator.stream(query):
            yield event
            if event.get("type") == "done":
                final_answer = event.get("answer", "")
        if final_answer:
            self.memory.add_exchange(query, final_answer)

    # ------------------------------------------------------------------
    # Plan-and-Execute: LangGraph StateGraph
    # ------------------------------------------------------------------

    def _build_plan_execute_graph(self):
        """Build the plan-and-execute LangGraph StateGraph.

        The graph has three nodes:
          create_plan  — call the planner to generate a ResearchPlan
          execute_step — run one pending step via the inner agent AND
                         advance the step pointer in the same return dict
          synthesize   — combine all findings into a final answer

        Conditional edge after *execute_step*:
          more steps remaining → execute_step (self-loop)
          all steps done       → synthesize

        Previously a separate ``replan`` node existed that only incremented
        the step counter. It was a pure no-op hop and has been folded into
        ``execute_step``. Real replanning (revising remaining steps based on
        findings) would live here if/when implemented.
        """
        from src.planner import generate_plan, ResearchPlan

        # ---- State schema -----------------------------------------------
        class PlanExecuteState(TypedDict):
            query: str
            plan_data: dict                              # ResearchPlan.model_dump()
            current_step: int
            all_findings: Annotated[List[str], operator.add]
            final_answer: str

        # ---- Node: create_plan ------------------------------------------
        async def create_plan_node(state: PlanExecuteState) -> dict:
            plan = generate_plan(state["query"], self.llm)
            return {
                "plan_data": plan.model_dump(),
                "current_step": 0,
                "all_findings": [],
                "final_answer": "",
            }

        # ---- Node: execute_step -----------------------------------------
        async def execute_step_node(state: PlanExecuteState) -> dict:
            """Run the current step and advance the step pointer.

            Returns a delta containing the updated plan, an appended finding,
            and ``current_step + 1``. The self-loop condition in
            ``should_continue`` inspects the already-incremented pointer to
            decide whether to re-enter this node or route to synthesize.
            """
            plan = ResearchPlan(**state["plan_data"])
            idx = state["current_step"]

            if idx >= len(plan.steps):
                # Defensive: should never happen because should_continue
                # routes to synthesize once idx == len(steps).
                return {"plan_data": plan.model_dump()}

            step = plan.steps[idx]
            plan.steps[idx] = step.model_copy(update={"status": "in_progress"})

            step_messages = [
                HumanMessage(
                    content=(
                        f"Research task: {step.description}\n\n"
                        f"Focus specifically on this task. Be thorough but concise."
                    )
                )
            ]
            result = await self.agent.ainvoke(
                {"messages": step_messages},
                {"recursion_limit": 20},
            )
            findings = self._extract_answer(result)
            plan.steps[idx] = step.model_copy(
                update={"status": "done", "findings": findings}
            )
            return {
                "plan_data": plan.model_dump(),
                "all_findings": [
                    f"Step {step.step_number} — {step.description}:\n{findings}"
                ],
                "current_step": idx + 1,
            }

        # ---- Node: synthesize -------------------------------------------
        async def synthesize_node(state: PlanExecuteState) -> dict:
            plan = ResearchPlan(**state["plan_data"])
            findings = state.get("all_findings", [])

            if not findings:
                # Simple / direct mode — answer from scratch
                prompt = state["query"]
            else:
                steps_text = "\n\n".join(findings)
                prompt = (
                    f"Synthesize these research findings into a comprehensive answer.\n\n"
                    f"Original question: {state['query']}\n\n"
                    f"Research findings:\n{steps_text}"
                )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return {"final_answer": content}

        # ---- Conditional edge after execute_step ------------------------
        def should_continue(state: PlanExecuteState) -> str:
            """Decide whether to loop back for another step or synthesize.

            Inspects the already-incremented ``current_step`` that
            ``execute_step_node`` just returned.
            """
            plan = ResearchPlan(**state["plan_data"])
            if state["current_step"] >= len(plan.steps):
                return "synthesize"
            return "execute_step"

        # ---- Conditional edge after create_plan -------------------------
        def after_create_plan(state: PlanExecuteState) -> str:
            plan = ResearchPlan(**state["plan_data"])
            if plan.is_simple or not plan.steps:
                return "synthesize"
            return "execute_step"

        # ---- Wire the graph ---------------------------------------------
        workflow = StateGraph(PlanExecuteState)
        workflow.add_node("create_plan", create_plan_node)
        workflow.add_node("execute_step", execute_step_node)
        workflow.add_node("synthesize", synthesize_node)

        workflow.add_edge(START, "create_plan")
        workflow.add_conditional_edges("create_plan", after_create_plan)
        workflow.add_conditional_edges("execute_step", should_continue)
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Plan-and-Execute: async method (for CLI / tests)
    # ------------------------------------------------------------------

    async def plan_and_execute(self, query: str, verbose: bool = True) -> str:
        """Run a query using the plan-and-execute strategy.

        1. Generates a structured research plan.
        2. Executes each step sequentially with the inner tool-calling agent.
        3. Synthesizes all findings into a final answer (streamed to stdout
           if *verbose* is True).

        Simple queries (detected by the complexity heuristic) fall back to
        ``stream_query()`` automatically.

        Args:
            query:   The research question.
            verbose: Whether to print plan and synthesis progress.

        Returns:
            The final synthesized answer as a string.
        """
        from src.planner import generate_plan

        if verbose:
            print("\n\U0001f5fa\ufe0f  Generating research plan...")

        plan = generate_plan(query, self.llm)

        if plan.is_simple:
            if verbose:
                print("(Simple query \u2014 using direct mode)\n")
            return await self.stream_query(query)

        if verbose:
            print(f"\nResearch Plan ({len(plan.steps)} steps):")
            for step in plan.steps:
                print(f"  {step.step_number}. {step.description}")
                if step.expected_tools:
                    print(f"     Tools: {', '.join(step.expected_tools)}")
            print()

        all_findings: List[str] = []

        for i, step in enumerate(plan.steps):
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}: {step.description}")
                print(f"{'=' * 60}")

            step_messages = [
                HumanMessage(
                    content=(
                        f"Research task: {step.description}\n\n"
                        f"Focus specifically on this task. Be thorough but concise."
                    )
                )
            ]
            result = await self.agent.ainvoke(
                {"messages": step_messages},
                {"recursion_limit": 20},
            )
            findings = self._extract_answer(result)
            plan.steps[i] = step.model_copy(
                update={"status": "done", "findings": findings}
            )
            all_findings.append(
                f"Step {step.step_number} \u2014 {step.description}:\n{findings}"
            )
            if verbose:
                print(f"\u2705 Step {step.step_number} complete")

        # ------ Synthesis (streamed) ------------------------------------
        steps_text = "\n\n".join(all_findings)
        synthesis_prompt = (
            f"Synthesize these research findings into a comprehensive answer.\n\n"
            f"Original question: {query}\n\n"
            f"Research findings:\n{steps_text}"
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print("Synthesizing all findings...")
            print(f"{'=' * 60}\n")

        final_answer = ""
        async for chunk in self.llm.astream([HumanMessage(content=synthesis_prompt)]):
            text = self._extract_chunk_text(chunk)
            if text:
                final_answer += text
                if verbose:
                    sys.stdout.write(text)
                    sys.stdout.flush()

        if verbose:
            print()  # newline after streaming

        self.memory.add_exchange(query, final_answer)
        return final_answer

    # ------------------------------------------------------------------
    # Plan-and-Execute: sync streaming generator (for Streamlit)
    # ------------------------------------------------------------------

    def plan_and_execute_stream(self, query: str) -> Generator[dict, None, None]:
        """Sync generator that drives plan-and-execute and yields typed events.

        Designed for the Streamlit UI.  Each yielded dict has a ``"type"`` key:

        * ``plan_created``   — plan is ready; contains ``"plan"``
        * ``step_started``   — step N began; contains ``"step_idx"``, ``"plan"``
        * ``step_tool``      — a tool was called; contains ``"step_idx"``, ``"tool_name"``
        * ``step_done``      — step N finished; contains ``"step_idx"``, ``"plan"``
        * ``synthesis_token``— one text chunk of the synthesis; contains ``"token"``
        * ``done``           — final; contains ``"answer"``, ``"plan"``
        """
        from src.planner import generate_plan

        plan = generate_plan(query, self.llm)
        yield {"type": "plan_created", "plan": plan}

        # ---- Simple / direct mode -----------------------------------
        if plan.is_simple:
            messages_in = [HumanMessage(content=query)]
            final_answer = ""
            for chunk, metadata in self.agent.stream(
                {"messages": messages_in},
                config={"recursion_limit": 20},
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node == "model" and isinstance(chunk, AIMessageChunk):
                    text = self._extract_chunk_text(chunk)
                    if text:
                        final_answer += text
                        yield {"type": "synthesis_token", "token": text}
                elif node == "tools" and isinstance(chunk, ToolMessage):
                    yield {"type": "step_tool", "step_idx": -1, "tool_name": chunk.name or "tool", "tool_output": chunk.content or ""}

            yield {"type": "done", "answer": final_answer, "plan": plan}
            return

        # ---- Multi-step execution ------------------------------------
        for i, step in enumerate(plan.steps):
            plan.steps[i] = step.model_copy(update={"status": "in_progress"})
            yield {"type": "step_started", "step_idx": i, "plan": plan}

            step_messages = [
                HumanMessage(
                    content=(
                        f"Research task: {step.description}\n\n"
                        f"Focus specifically on this task. Be thorough but concise."
                    )
                )
            ]

            # Stream the inner agent to capture tool calls as they happen
            inner_result = None
            for update in self.agent.stream(
                {"messages": step_messages},
                config={"recursion_limit": 20},
                stream_mode="updates",
            ):
                inner_result = update
                # Yield tool call events from the tools node
                for node_name, node_output in update.items():
                    if node_name == "tools":
                        msgs = node_output.get("messages", [])
                        for msg in msgs:
                            if isinstance(msg, ToolMessage):
                                yield {
                                    "type": "step_tool",
                                    "step_idx": i,
                                    "tool_name": msg.name or "tool",
                                    "tool_output": msg.content or "",
                                }

            # Extract findings from the final state
            findings = ""
            if inner_result:
                # stream_mode="updates" yields {node_name: state_update}
                for node_name, node_output in inner_result.items():
                    msgs = node_output.get("messages", [])
                    for msg in reversed(msgs):
                        if isinstance(msg, AIMessage) and msg.content:
                            if isinstance(msg.content, str):
                                findings = msg.content
                            elif isinstance(msg.content, list):
                                findings = "\n".join(
                                    block.get("text", "")
                                    for block in msg.content
                                    if isinstance(block, dict) and block.get("type") == "text"
                                )
                            if findings:
                                break
                    if findings:
                        break

            plan.steps[i] = plan.steps[i].model_copy(
                update={"status": "done", "findings": findings}
            )
            yield {"type": "step_done", "step_idx": i, "plan": plan}

        # ---- Synthesis (streamed token-by-token) ---------------------
        steps_text = "\n\n".join(
            f"Step {s.step_number} \u2014 {s.description}:\n{s.findings}"
            for s in plan.steps
        )
        synthesis_prompt = (
            f"Synthesize these research findings into a comprehensive answer.\n\n"
            f"Original question: {query}\n\n"
            f"Research findings:\n{steps_text}"
        )

        final_answer = ""
        for chunk in self.llm.stream([HumanMessage(content=synthesis_prompt)]):
            text = self._extract_chunk_text(chunk)
            if text:
                final_answer += text
                yield {"type": "synthesis_token", "token": text}

        self.memory.add_exchange(query, final_answer)
        yield {"type": "done", "answer": final_answer, "plan": plan}

    # ------------------------------------------------------------------
    # Helper: extract text from an LLM chunk (AIMessageChunk or AIMessage)
    # ------------------------------------------------------------------

    def _extract_chunk_text(self, chunk) -> str:
        """Return the text content from an LLM message or chunk."""
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""


def create_research_agent():
    """Create and return a research agent. Kept for backward compatibility."""
    return ResearchAgent()


async def run_research_query(query: str) -> str:
    """Run a single research query (without memory)."""
    agent = ResearchAgent()
    return await agent.query(query)


# For testing
if __name__ == "__main__":
    import asyncio

    async def _test():
        print("Testing agent with memory...\n")
        agent = ResearchAgent()

        q1 = "What is the population of France?"
        print(f"Q1: {q1}")
        print("=" * 60)
        a1 = await agent.query(q1)
        print("=" * 60)
        print(f"A1: {a1}\n")

        q2 = "How does that compare to Germany?"
        print(f"Q2: {q2}")
        print("=" * 60)
        a2 = await agent.query(q2)
        print("=" * 60)
        print(f"A2: {a2}")

    asyncio.run(_test())
