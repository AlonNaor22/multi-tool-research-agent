"""Multi-agent orchestrator using LangGraph StateGraph.

Wires together the supervisor and specialist agents into a phased
execution graph:

  supervisor_plan  →  dispatch_phases  →  synthesize  →  END

The dispatch node iterates through execution phases. Within each phase,
independent specialists run concurrently via ``asyncio.gather()``.
"""

import asyncio
import sys
from typing import Annotated, Dict, Generator, List, Optional
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

from src.multi_agent.supervisor import Supervisor, DelegationPlan
from src.multi_agent.specialists import SpecialistAgent, build_specialists


class MultiAgentState(TypedDict):
    """State schema for the multi-agent orchestration graph."""
    query: str
    delegation_plan: dict                                   # DelegationPlan.model_dump()
    specialist_results: dict                                # {name: result_text}
    fact_check_report: str
    final_answer: str
    agent_activity: Annotated[List[dict], operator.add]     # UI event log


class MultiAgentOrchestrator:
    """Orchestrates multiple specialist agents via a supervisor.

    The orchestrator:
    1. Asks the supervisor to create a DelegationPlan.
    2. Dispatches specialists in phases — each phase runs its agents
       concurrently via ``asyncio.gather()``.
    3. Optionally runs the fact-checker on prior findings.
    4. Asks the supervisor to synthesize all outputs into a final answer.
    """

    def __init__(
        self,
        llm: ChatAnthropic,
        all_tools: list,
        tool_health: dict,
        callbacks: Optional[list] = None,
    ):
        self.llm = llm
        self.callbacks = callbacks or []
        self.supervisor = Supervisor(llm)
        self.specialists = build_specialists(all_tools, llm, tool_health)
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph StateGraph for multi-agent orchestration."""

        # ---- Node: supervisor_plan --------------------------------------
        async def supervisor_plan_node(state: MultiAgentState) -> dict:
            plan = self.supervisor.create_delegation_plan(state["query"])
            return {
                "delegation_plan": plan.model_dump(),
                "specialist_results": {},
                "fact_check_report": "",
                "final_answer": "",
                "agent_activity": [{
                    "type": "plan_created",
                    "plan": plan.model_dump(),
                    "rationale": plan.rationale,
                }],
            }

        # ---- Node: dispatch_phases --------------------------------------
        async def dispatch_phases_node(state: MultiAgentState) -> dict:
            plan = DelegationPlan(**state["delegation_plan"])
            all_results = dict(state.get("specialist_results", {}))
            activity = []

            for phase_idx, phase in enumerate(plan.execution_phases):
                activity.append({
                    "type": "phase_started",
                    "phase_idx": phase_idx,
                    "specialists": phase,
                })

                # Build context string from prior phase results
                prior_context = ""
                if all_results:
                    parts = [
                        f"[{name}]: {text[:500]}"
                        for name, text in all_results.items()
                    ]
                    prior_context = (
                        "\n\nContext from previous agents:\n" +
                        "\n".join(parts)
                    )

                # Run all specialists in this phase concurrently
                async def _run_specialist(name: str) -> tuple:
                    task = plan.specialist_tasks.get(name, plan.query)

                    # If this is the fact-checker, include prior findings
                    if name == "fact_checker" and all_results:
                        findings_text = "\n\n".join(
                            f"[{n}]: {r}" for n, r in all_results.items()
                            if n != "fact_checker"
                        )
                        task = (
                            f"Verify the key claims from these findings:\n\n"
                            f"{findings_text}\n\n"
                            f"Original question: {plan.query}"
                        )
                    elif prior_context and name != "fact_checker":
                        task = task + prior_context

                    activity.append({
                        "type": "specialist_started",
                        "specialist": name,
                        "task": task[:200],
                    })

                    specialist = self.specialists.get(name)
                    if specialist is None:
                        return name, f"[{name}] Unknown specialist."

                    result = await specialist.run(task, callbacks=self.callbacks)

                    activity.append({
                        "type": "specialist_done",
                        "specialist": name,
                        "result_preview": result[:200],
                    })
                    return name, result

                # asyncio.gather for parallel execution within the phase
                phase_results = await asyncio.gather(
                    *[_run_specialist(name) for name in phase]
                )

                for name, result in phase_results:
                    all_results[name] = result

                activity.append({
                    "type": "phase_done",
                    "phase_idx": phase_idx,
                })

            # Separate fact-check report if present
            fact_check_report = all_results.pop("fact_checker", "")

            return {
                "specialist_results": all_results,
                "fact_check_report": fact_check_report,
                "agent_activity": activity,
            }

        # ---- Node: synthesize -------------------------------------------
        async def synthesize_node(state: MultiAgentState) -> dict:
            answer = await self.supervisor.synthesize(
                query=state["query"],
                specialist_results=state["specialist_results"],
                fact_check_report=state.get("fact_check_report", ""),
            )
            return {
                "final_answer": answer,
                "agent_activity": [{"type": "synthesis_done"}],
            }

        # ---- Wire the graph ---------------------------------------------
        workflow = StateGraph(MultiAgentState)
        workflow.add_node("supervisor_plan", supervisor_plan_node)
        workflow.add_node("dispatch_phases", dispatch_phases_node)
        workflow.add_node("synthesize", synthesize_node)

        workflow.add_edge(START, "supervisor_plan")
        workflow.add_edge("supervisor_plan", "dispatch_phases")
        workflow.add_edge("dispatch_phases", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str) -> str:
        """Run the multi-agent pipeline and return the final answer.

        Args:
            query: The user's research question.

        Returns:
            The synthesized answer string.
        """
        result = await self.graph.ainvoke(
            {
                "query": query,
                "delegation_plan": {},
                "specialist_results": {},
                "fact_check_report": "",
                "final_answer": "",
                "agent_activity": [],
            }
        )
        return result["final_answer"]

    async def run_verbose(self, query: str) -> str:
        """Run with verbose CLI output showing each phase and specialist."""
        print(f"\n{'=' * 60}")
        print("Multi-Agent Orchestration")
        print(f"{'=' * 60}")
        print(f"\nQuery: {query}\n")

        # Step 1: Supervisor creates delegation plan
        print("Supervisor is analyzing the query...")
        plan = self.supervisor.create_delegation_plan(query)
        print(f"\nDelegation Plan ({len(plan.execution_phases)} phases):")
        print(f"  Rationale: {plan.rationale}")
        for i, phase in enumerate(plan.execution_phases):
            parallel_note = " (parallel)" if len(phase) > 1 else ""
            print(f"  Phase {i + 1}{parallel_note}: {', '.join(phase)}")
        if plan.needs_fact_check:
            print("  Fact-checking: enabled")
        print()

        # Step 2: Execute phases
        all_results = {}
        for phase_idx, phase in enumerate(plan.execution_phases):
            parallel_note = " (parallel)" if len(phase) > 1 else ""
            print(f"{'─' * 40}")
            print(f"Phase {phase_idx + 1}{parallel_note}: {', '.join(phase)}")
            print(f"{'─' * 40}")

            # Build context from prior results
            prior_context = ""
            if all_results:
                parts = [f"[{n}]: {t[:500]}" for n, t in all_results.items()]
                prior_context = "\n\nContext from previous agents:\n" + "\n".join(parts)

            async def _run_one(name):
                task = plan.specialist_tasks.get(name, query)
                if name == "fact_checker" and all_results:
                    findings = "\n\n".join(
                        f"[{n}]: {r}" for n, r in all_results.items()
                        if n != "fact_checker"
                    )
                    task = (
                        f"Verify the key claims from these findings:\n\n"
                        f"{findings}\n\nOriginal question: {query}"
                    )
                elif prior_context and name != "fact_checker":
                    task = task + prior_context

                print(f"  Starting {name}...")
                specialist = self.specialists.get(name)
                if specialist is None:
                    return name, f"[{name}] Unknown specialist."
                result = await specialist.run(task, callbacks=self.callbacks)
                print(f"  {name} done.")
                return name, result

            phase_results = await asyncio.gather(
                *[_run_one(n) for n in phase]
            )
            for name, result in phase_results:
                all_results[name] = result

        # Step 3: Synthesize
        fact_check_report = all_results.pop("fact_checker", "")
        print(f"\n{'=' * 60}")
        print("Synthesizing all findings...")
        print(f"{'=' * 60}\n")

        final_answer = ""
        async for chunk in self.llm.astream([
            HumanMessage(content=(
                f"Synthesize these specialist findings into a comprehensive answer.\n\n"
                f"Original question: {query}\n\n" +
                "\n\n".join(
                    f"--- {n.upper()} AGENT ---\n{r}"
                    for n, r in all_results.items()
                ) +
                (f"\n\n--- FACT-CHECK REPORT ---\n{fact_check_report}"
                 if fact_check_report else "")
            ))
        ]):
            text = _extract_chunk_text(chunk)
            if text:
                final_answer += text
                sys.stdout.write(text)
                sys.stdout.flush()

        print()  # newline after streaming
        return final_answer

    def stream(self, query: str) -> Generator[dict, None, None]:
        """Sync generator that yields typed events for the Streamlit UI.

        Event types:

        * ``plan_created``       — delegation plan ready
        * ``phase_started``      — a new phase begins
        * ``specialist_started`` — a specialist is running
        * ``specialist_done``    — a specialist finished
        * ``phase_done``         — a phase completed
        * ``synthesis_token``    — one chunk of the final answer
        * ``done``               — final answer ready
        """
        import asyncio as _asyncio

        # Step 1: Supervisor creates plan
        plan = self.supervisor.create_delegation_plan(query)
        yield {
            "type": "plan_created",
            "plan": plan,
        }

        # Step 2: Execute phases
        all_results = {}
        for phase_idx, phase in enumerate(plan.execution_phases):
            yield {
                "type": "phase_started",
                "phase_idx": phase_idx,
                "specialists": phase,
            }

            prior_context = ""
            if all_results:
                parts = [f"[{n}]: {t[:500]}" for n, t in all_results.items()]
                prior_context = "\n\nContext from previous agents:\n" + "\n".join(parts)

            for name in phase:
                yield {
                    "type": "specialist_started",
                    "specialist": name,
                    "phase_idx": phase_idx,
                }

            # Run specialists in parallel using asyncio
            async def _run_phase():
                results = []
                for name in phase:
                    task = plan.specialist_tasks.get(name, query)
                    if name == "fact_checker" and all_results:
                        findings = "\n\n".join(
                            f"[{n}]: {r}" for n, r in all_results.items()
                            if n != "fact_checker"
                        )
                        task = (
                            f"Verify the key claims from these findings:\n\n"
                            f"{findings}\n\nOriginal question: {query}"
                        )
                    elif prior_context and name != "fact_checker":
                        task = task + prior_context

                    specialist = self.specialists.get(name)
                    if specialist is None:
                        results.append((name, f"[{name}] Unknown specialist."))
                    else:
                        results.append((name, specialist.run(task, callbacks=self.callbacks)))

                # Gather coroutines
                coros = []
                names = []
                plain_results = {}
                for name, val in results:
                    if isinstance(val, str):
                        plain_results[name] = val
                    else:
                        coros.append(val)
                        names.append(name)

                if coros:
                    gathered = await _asyncio.gather(*coros)
                    for n, r in zip(names, gathered):
                        plain_results[n] = r

                return plain_results

            # Run the async phase in a new event loop if needed
            try:
                loop = _asyncio.get_running_loop()
                # Already in an async context — shouldn't happen in Streamlit sync
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    phase_results = loop.run_in_executor(
                        pool, lambda: _asyncio.run(_run_phase())
                    )
                    # This path is unlikely in practice
                    phase_results = _asyncio.run(_run_phase())
            except RuntimeError:
                # No event loop — expected in Streamlit sync context
                phase_results = _asyncio.run(_run_phase())

            for name, result in phase_results.items():
                all_results[name] = result
                yield {
                    "type": "specialist_done",
                    "specialist": name,
                    "phase_idx": phase_idx,
                    "result_preview": result[:300],
                }

            yield {
                "type": "phase_done",
                "phase_idx": phase_idx,
            }

        # Step 3: Synthesize (streamed token-by-token)
        fact_check_report = all_results.pop("fact_checker", "")

        synthesis_input = (
            f"Synthesize these specialist findings into a comprehensive answer.\n\n"
            f"Original question: {query}\n\n" +
            "\n\n".join(
                f"--- {n.upper()} AGENT ---\n{r}"
                for n, r in all_results.items()
            ) +
            (f"\n\n--- FACT-CHECK REPORT ---\n{fact_check_report}"
             if fact_check_report else "")
        )

        final_answer = ""
        for chunk in self.llm.stream([HumanMessage(content=synthesis_input)]):
            text = _extract_chunk_text(chunk)
            if text:
                final_answer += text
                yield {"type": "synthesis_token", "token": text}

        yield {
            "type": "done",
            "answer": final_answer,
            "plan": plan,
        }


def _extract_chunk_text(chunk) -> str:
    """Extract text from an LLM chunk."""
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
