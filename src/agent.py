"""LangGraph research agent with memory, plan-and-execute, and multi-agent orchestration."""

import asyncio
import logging
import queue
import sys
import threading
from typing import Dict, Generator, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from src.callbacks import TimingCallbackHandler, StreamingCallbackHandler
from src.constants import (
    EVENT_PLAN_CREATED, EVENT_SYNTHESIS_TOKEN, EVENT_DONE,
    EVENT_STEP_STARTED, EVENT_STEP_TOOL, EVENT_STEP_DONE,
    EVENT_WAVE_STARTED, STATUS_IN_PROGRESS, STATUS_DONE,
)

logger = logging.getLogger(__name__)
from src.observability import ObservabilityCallbackHandler, MetricsStore, format_query_metrics
from src.rate_limiter import RateLimiter
from src.tool_health import check_tool_health, get_available_tools
from src.utils import flatten_content, extract_chunk_text, extract_ai_answer
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
    """Build system prompt with tool categories, excluding disabled_tools from listings."""
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
    """Conversation memory storing all exchanges but prompting with only the last k."""

    def __init__(self, k: int = 5):
        self.k = k
        self.history = []

    def add_exchange(self, user_input: str, agent_output: str):
        self.history.append((user_input, agent_output))

    def get_messages(self) -> list:
        """Return last k exchanges as HumanMessage/AIMessage pairs."""
        if not self.history:
            return []

        recent_history = self.history[-self.k:]
        messages = []
        for user_input, agent_output in recent_history:
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content=agent_output))
        return messages

    def get_history_string(self) -> str:
        """Return last k exchanges as a formatted 'Human:/Assistant:' string."""
        if not self.history:
            return "No previous conversation."

        recent_history = self.history[-self.k:]
        lines = []
        for user_input, agent_output in recent_history:
            lines.append(f"Human: {user_input}")
            lines.append(f"Assistant: {agent_output}")
        return "\n".join(lines)

    def clear(self):
        self.history = []

    @property
    def buffer(self) -> str:
        return self.get_history_string()


class ResearchAgent:
    """LangGraph-based research agent with tool calling, memory, and multi-agent modes."""

    def __init__(self):
        self.llm = ChatAnthropic(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY,
            streaming=True,
        )

        all_tools = [
            calculator_tool, unit_converter_tool, equation_solver_tool,
            currency_tool, wolfram_tool, math_formatter_tool,
            wikipedia_tool, search_tool, news_tool, arxiv_tool,
            youtube_tool, google_scholar_tool,
            url_tool, pdf_tool, reddit_tool, wikidata_tool,
            translation_tool, python_repl_tool, visualization_tool,
            parallel_tool, weather_tool, github_tool, scraper_tool,
            csv_tool, datetime_tool,
        ]

        # Disabled tools are removed so the agent never wastes a turn calling them
        self.tool_health = check_tool_health()
        self.tools, self.disabled_tools = get_available_tools(all_tools, self.tool_health)

        self.memory = SimpleMemory(k=5)
        self.current_session_id = None
        self.timing_callback = TimingCallbackHandler()
        self.streaming_callback = StreamingCallbackHandler()
        self.observability_callback = ObservabilityCallbackHandler(model_name=MODEL_NAME)
        self.metrics_store = MetricsStore()
        self.rate_limiter = RateLimiter()

        system_prompt = _build_system_prompt(disabled_tools=self.disabled_tools)
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=VERBOSE,
        )

    async def query(self, question: str, show_timing: bool = True) -> str:
        """Run a research query async, update memory, and return the answer string."""
        try:
            self.rate_limiter.check_budget()
            self.timing_callback.reset()
            self.observability_callback.reset(question=question)

            messages = self.memory.get_messages()
            messages.append(HumanMessage(content=question))
            result = await self.agent.ainvoke(
                {"messages": messages},
                {"callbacks": [self.timing_callback, self.observability_callback],
                 "recursion_limit": MAX_ITERATIONS * 2},
            )

            answer = extract_ai_answer(result)
            self.memory.add_exchange(question, answer)

            metrics = self.observability_callback.get_metrics()
            self.metrics_store.save(metrics)
            self.rate_limiter.record_tokens(metrics.total_tokens)

            if show_timing:
                print(self.timing_callback.get_summary())
                print(format_query_metrics(metrics))

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    async def stream_query(self, question: str, show_timing: bool = True) -> str:
        """Stream a research query with real-time output and return the final answer."""
        try:
            self.rate_limiter.check_budget()
            self.timing_callback.reset()
            self.streaming_callback.reset()
            self.observability_callback.reset(question=question)

            messages = self.memory.get_messages()
            messages.append(HumanMessage(content=question))

            final_result = None
            async for chunk in self.agent.astream(
                {"messages": messages},
                {"callbacks": [self.timing_callback, self.streaming_callback,
                               self.observability_callback],
                 "recursion_limit": MAX_ITERATIONS * 2},
                stream_mode="values",
            ):
                final_result = chunk

            answer = extract_ai_answer(final_result) if final_result else "No answer was generated."
            self.memory.add_exchange(question, answer)

            metrics = self.observability_callback.get_metrics()
            self.metrics_store.save(metrics)
            self.rate_limiter.record_tokens(metrics.total_tokens)

            if show_timing:
                print(self.timing_callback.get_summary())
                print(format_query_metrics(metrics))

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    def get_last_timing(self) -> str:
        return self.timing_callback.get_summary()

    def get_last_metrics(self):
        return self.observability_callback.get_metrics()

    def clear_memory(self):
        """Clear conversation history, reset rate limiter, and start a new session."""
        self.memory.clear()
        self.rate_limiter.reset()
        self.current_session_id = None
        print("Conversation memory cleared.")

    def get_memory(self) -> str:
        return self.memory.buffer

    def save_session(self, session_id: str = None, description: str = None) -> str:
        """Save session history to JSON, reusing current session ID if set."""
        from src.session_manager import save_session

        if session_id is None and self.current_session_id is not None:
            session_id = self.current_session_id

        filepath = save_session(self.memory.history, session_id, description)

        if self.current_session_id is None:
            import os
            filename = os.path.basename(filepath)
            self.current_session_id = filename.replace('.json', '')

        return filepath

    def load_session(self, session_id: str) -> bool:
        """Load a saved session into memory by session_id; return True on success."""
        from src.session_manager import load_session
        history = load_session(session_id)

        if history is None:
            return False

        self.memory.history = list(history)
        self.current_session_id = session_id

        return True


    def route_query(self, query: str, mode: str = "Auto"):
        """Sync generator that routes to the right mode and yields typed events.

        Centralizes mode selection so app.py and main.py don't duplicate it.
        Yields the same event types as plan_and_execute_stream / multi_agent_stream.
        """
        from src.planner import is_simple_query
        from src.constants import MODE_AUTO, MODE_DIRECT, MODE_PLAN_EXECUTE, MODE_MULTI_AGENT

        if mode == MODE_MULTI_AGENT:
            yield from self.multi_agent_stream(query)
        elif mode == MODE_PLAN_EXECUTE or (mode == MODE_AUTO and not is_simple_query(query)):
            yield from self.plan_and_execute_stream(query)
        else:
            yield from self._direct_stream(query)

    def _direct_stream(self, query: str):
        """Sync generator for direct mode that yields the same event types as other modes."""
        self.timing_callback.reset()
        self.observability_callback.reset(question=query)

        messages = self.memory.get_messages()
        messages.append(HumanMessage(content=query))

        final_answer = ""
        for chunk, metadata in self.agent.stream(
            {"messages": messages},
            config={
                "callbacks": [self.timing_callback, self.observability_callback],
                "recursion_limit": MAX_ITERATIONS * 2,
            },
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node", "")
            if node == "model" and isinstance(chunk, AIMessageChunk):
                text = extract_chunk_text(chunk)
                if text:
                    final_answer += text
                    yield {"type": EVENT_SYNTHESIS_TOKEN, "token": text}
            elif node == "tools" and isinstance(chunk, ToolMessage):
                yield {
                    "type": EVENT_STEP_TOOL,
                    "step_idx": -1,
                    "tool_name": chunk.name or "tool",
                    "tool_output": chunk.content or "",
                }

        self.memory.add_exchange(query, final_answer)
        metrics = self.observability_callback.get_metrics()
        self.metrics_store.save(metrics)
        self.rate_limiter.record_tokens(metrics.total_tokens)
        yield {"type": EVENT_DONE, "answer": final_answer}

    def _get_orchestrator(self):
        """Lazily build and cache the multi-agent orchestrator."""
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
        """Run query via multi-agent orchestration; return synthesized answer."""
        self.observability_callback.reset(question=query)
        orchestrator = self._get_orchestrator()
        if verbose:
            answer = await orchestrator.run_verbose(query)
        else:
            answer = await orchestrator.run(query)
        self.memory.add_exchange(query, answer)
        metrics = self.observability_callback.get_metrics()
        self.metrics_store.save(metrics)
        self.rate_limiter.record_tokens(metrics.total_tokens)
        return answer

    def multi_agent_stream(self, query: str):
        """Yield typed multi-agent events (sync generator) for Streamlit UI."""
        self.observability_callback.reset(question=query)
        orchestrator = self._get_orchestrator()
        final_answer = ""
        for event in orchestrator.stream(query):
            yield event
            if event.get("type") == EVENT_DONE:
                final_answer = event.get("answer", "")
        if final_answer:
            self.memory.add_exchange(query, final_answer)
        metrics = self.observability_callback.get_metrics()
        self.metrics_store.save(metrics)
        self.rate_limiter.record_tokens(metrics.total_tokens)

    async def _run_step(
        self,
        step,
        plan,
        completed_findings: Dict[int, str],
    ) -> str:
        """Execute a single plan step, injecting context from declared dependencies."""
        task = f"Research task: {step.description}\n\nFocus specifically on this task. Be thorough but concise."

        deps = plan.depends_on.get(step.step_number, [])
        if deps:
            dep_parts = [
                f"[Step {d} findings]: {completed_findings[d]}"
                for d in deps if d in completed_findings
            ]
            if dep_parts:
                task += "\n\nContext from prior steps:\n" + "\n".join(dep_parts)

        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            {"recursion_limit": 20, "callbacks": [self.observability_callback]},
        )
        return extract_ai_answer(result)

    async def plan_and_execute(self, query: str, verbose: bool = True) -> str:
        """Generate a plan, execute steps in dependency-driven waves, synthesize."""
        from src.planner import generate_plan

        self.observability_callback.reset(question=query)

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
                deps = plan.depends_on.get(step.step_number, [])
                dep_info = f"  (depends on: {deps})" if deps else ""
                print(f"  {step.step_number}. {step.description}{dep_info}")
                if step.expected_tools:
                    print(f"     Tools: {', '.join(step.expected_tools)}")
            print()

        completed_findings: Dict[int, str] = {}
        pending = {s.step_number for s in plan.steps}
        step_map = {s.step_number: (i, s) for i, s in enumerate(plan.steps)}
        wave = 0

        while pending:
            ready = [
                sn for sn in sorted(pending)
                if all(d in completed_findings for d in plan.depends_on.get(sn, []))
            ]
            if not ready:
                logger.warning("No ready steps — possible circular dependency; breaking")
                break

            parallel_note = " (parallel)" if len(ready) > 1 else ""
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Wave {wave + 1}{parallel_note}: Steps {ready}")
                print(f"{'=' * 60}")

            async def _exec(sn: int) -> tuple:
                idx, step = step_map[sn]
                try:
                    findings = await self._run_step(step, plan, completed_findings)
                except Exception as e:
                    findings = f"Error in step {sn}: {str(e)}"
                return sn, idx, findings

            results = await asyncio.gather(*[_exec(sn) for sn in ready])

            for sn, idx, findings in results:
                completed_findings[sn] = findings
                pending.discard(sn)
                plan.steps[idx] = plan.steps[idx].model_copy(
                    update={"status": STATUS_DONE, "findings": findings}
                )
                if verbose:
                    print(f"\u2705 Step {sn} complete")

            wave += 1

        steps_text = "\n\n".join(
            f"Step {s.step_number} \u2014 {s.description}:\n{s.findings}"
            for s in plan.steps
        )
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
        try:
            async for chunk in self.llm.astream([HumanMessage(content=synthesis_prompt)]):
                text = extract_chunk_text(chunk)
                if text:
                    final_answer += text
                    if verbose:
                        sys.stdout.write(text)
                        sys.stdout.flush()
        except Exception as e:
            final_answer = final_answer or f"Error during synthesis: {str(e)}"

        if verbose:
            print()

        self.memory.add_exchange(query, final_answer)
        metrics = self.observability_callback.get_metrics()
        self.metrics_store.save(metrics)
        self.rate_limiter.record_tokens(metrics.total_tokens)
        return final_answer

    def plan_and_execute_stream(self, query: str) -> Generator[dict, None, None]:
        """Yield typed plan-and-execute events (sync generator) for Streamlit UI."""
        from src.planner import generate_plan

        self.observability_callback.reset(question=query)
        plan = generate_plan(query, self.llm)
        yield {"type": EVENT_PLAN_CREATED, "plan": plan}

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
                    text = extract_chunk_text(chunk)
                    if text:
                        final_answer += text
                        yield {"type": EVENT_SYNTHESIS_TOKEN, "token": text}
                elif node == "tools" and isinstance(chunk, ToolMessage):
                    yield {"type": EVENT_STEP_TOOL, "step_idx": -1, "tool_name": chunk.name or "tool", "tool_output": chunk.content or ""}

            yield {"type": EVENT_DONE, "answer": final_answer, "plan": plan}
            return

        # --- Wave-based parallel execution via async-to-sync bridge ---
        _SENTINEL = object()
        q: queue.Queue = queue.Queue()
        agent_ref = self

        async def _drive() -> None:
            completed_findings: Dict[int, str] = {}
            pending = {s.step_number for s in plan.steps}
            step_map = {s.step_number: (i, s) for i, s in enumerate(plan.steps)}
            wave = 0

            while pending:
                ready = [
                    sn for sn in sorted(pending)
                    if all(d in completed_findings for d in plan.depends_on.get(sn, []))
                ]
                if not ready:
                    logger.warning("No ready steps — possible circular dependency; breaking")
                    break

                q.put({"type": EVENT_WAVE_STARTED, "wave_idx": wave, "step_numbers": ready})

                for sn in ready:
                    idx, step = step_map[sn]
                    plan.steps[idx] = step.model_copy(update={"status": STATUS_IN_PROGRESS})
                    q.put({"type": EVENT_STEP_STARTED, "step_idx": idx, "plan": plan})

                async def _exec(sn: int) -> tuple:
                    idx, step = step_map[sn]
                    try:
                        findings = await agent_ref._run_step(step, plan, completed_findings)
                    except Exception as e:
                        findings = f"Error in step {sn}: {str(e)}"
                    return sn, idx, findings

                results = await asyncio.gather(*[_exec(sn) for sn in ready])

                for sn, idx, findings in results:
                    completed_findings[sn] = findings
                    pending.discard(sn)
                    plan.steps[idx] = plan.steps[idx].model_copy(
                        update={"status": STATUS_DONE, "findings": findings}
                    )
                    q.put({"type": EVENT_STEP_DONE, "step_idx": idx, "plan": plan})

                wave += 1

        def _runner() -> None:
            try:
                asyncio.run(_drive())
            except BaseException as exc:
                q.put(("error", exc))
                return
            q.put(("done", _SENTINEL))

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()

        while True:
            item = q.get()
            if isinstance(item, dict):
                yield item
            elif isinstance(item, tuple) and item[0] == "error":
                worker.join()
                raise item[1]
            else:
                break
        worker.join()

        # --- Synthesis phase (sync, same as before) ---
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
            text = extract_chunk_text(chunk)
            if text:
                final_answer += text
                yield {"type": EVENT_SYNTHESIS_TOKEN, "token": text}

        self.memory.add_exchange(query, final_answer)
        metrics = self.observability_callback.get_metrics()
        self.metrics_store.save(metrics)
        self.rate_limiter.record_tokens(metrics.total_tokens)
        yield {"type": EVENT_DONE, "answer": final_answer, "plan": plan}

def create_research_agent():
    """Create and return a ResearchAgent instance."""
    return ResearchAgent()


async def run_research_query(query: str) -> str:
    """Run a single stateless research query; return the answer string."""
    agent = ResearchAgent()
    return await agent.query(query)


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
