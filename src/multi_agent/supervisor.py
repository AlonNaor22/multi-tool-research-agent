"""Supervisor agent for multi-agent orchestration.

The supervisor has NO tools. It uses the LLM to:
1. Analyze a query and produce a DelegationPlan (which specialists to invoke,
   in which phases, with what tasks).
2. Synthesize all specialist outputs into a final answer.
"""

import json
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.multi_agent.prompts import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_SYNTHESIZE_PROMPT
from src.multi_agent.specialists import SPECIALIST_DEFINITIONS


class DelegationPlan(BaseModel):
    """A structured delegation plan produced by the supervisor.

    ``execution_phases`` is a list of lists — each inner list contains
    specialist names that can run in parallel.  Phases execute sequentially
    so later phases can depend on earlier outputs.

    Example::

        execution_phases = [["research", "math"], ["analysis"]]
        # Phase 1: research + math run concurrently
        # Phase 2: analysis runs after (may use Phase 1 outputs)
    """
    query: str
    execution_phases: List[List[str]] = Field(default_factory=list)
    specialist_tasks: Dict[str, str] = Field(default_factory=dict)
    needs_fact_check: bool = False
    rationale: str = ""


class Supervisor:
    """LLM-only supervisor that delegates tasks and synthesizes results."""

    def __init__(self, llm: ChatAnthropic):
        self.llm = llm

    @staticmethod
    def _plan_messages(query: str) -> List[BaseMessage]:
        """Build the prompt messages for a delegation-plan request."""
        return [
            SystemMessage(content=SUPERVISOR_PLAN_PROMPT),
            HumanMessage(content=f"Create a delegation plan for: {query}"),
        ]

    @staticmethod
    def _extract_text(content) -> str:
        """Flatten Anthropic content blocks into a plain string."""
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content

    @staticmethod
    def _parse_plan_response(content: str, query: str) -> DelegationPlan:
        """Parse LLM JSON output into a validated DelegationPlan.

        Raises on malformed output so callers can decide whether to fall
        back to the single-research-agent default.
        """
        data = json.loads(content)

        execution_phases = data.get("execution_phases", [])
        specialist_tasks = data.get("specialist_tasks", {})
        needs_fact_check = data.get("needs_fact_check", False)
        rationale = data.get("rationale", "")

        valid_names = set(SPECIALIST_DEFINITIONS.keys())
        execution_phases = [
            [s for s in phase if s in valid_names]
            for phase in execution_phases
        ]
        execution_phases = [p for p in execution_phases if p]

        if not execution_phases:
            raise ValueError("No valid specialists in phases")

        if needs_fact_check:
            execution_phases = [
                [s for s in phase if s != "fact_checker"]
                for phase in execution_phases
            ]
            execution_phases = [p for p in execution_phases if p]
            execution_phases.append(["fact_checker"])

            if "fact_checker" not in specialist_tasks:
                specialist_tasks["fact_checker"] = (
                    f"Verify the key claims from the research findings "
                    f"about: {query}"
                )

        return DelegationPlan(
            query=query,
            execution_phases=execution_phases,
            specialist_tasks=specialist_tasks,
            needs_fact_check=needs_fact_check,
            rationale=rationale,
        )

    @staticmethod
    def _fallback_plan(query: str) -> DelegationPlan:
        """Single-research-agent fallback when parsing fails."""
        return DelegationPlan(
            query=query,
            execution_phases=[["research"]],
            specialist_tasks={"research": query},
            needs_fact_check=False,
            rationale="Fallback — could not parse delegation plan.",
        )

    def create_delegation_plan(self, query: str) -> DelegationPlan:
        """Analyze *query* and produce a DelegationPlan (sync).

        Prefer ``acreate_delegation_plan`` when calling from an async
        context so the event loop isn't blocked on the LLM round-trip.

        Falls back to a single-phase ``["research"]`` plan if LLM output
        cannot be parsed (same resilience pattern as ``generate_plan`` in
        ``src/planner.py``).
        """
        try:
            response = self.llm.invoke(self._plan_messages(query))
            content = self._extract_text(response.content)
            return self._parse_plan_response(content, query)
        except Exception:
            return self._fallback_plan(query)

    async def acreate_delegation_plan(self, query: str) -> DelegationPlan:
        """Async version of :meth:`create_delegation_plan`.

        Uses ``await self.llm.ainvoke(...)`` so the event loop stays free
        during the planner LLM round-trip. Use this from inside any
        ``async def`` node or generator.
        """
        try:
            response = await self.llm.ainvoke(self._plan_messages(query))
            content = self._extract_text(response.content)
            return self._parse_plan_response(content, query)
        except Exception:
            return self._fallback_plan(query)

    async def synthesize(
        self,
        query: str,
        specialist_results: Dict[str, str],
        fact_check_report: str = "",
    ) -> str:
        """Combine all specialist outputs into a single answer.

        Args:
            query: The original user query.
            specialist_results: Dict mapping specialist name to its output.
            fact_check_report: Optional fact-checker output.

        Returns:
            The synthesized final answer.
        """
        parts = [f"Original question: {query}\n"]

        for name, result in specialist_results.items():
            if name == "fact_checker":
                continue  # handled separately
            parts.append(f"--- {name.upper()} AGENT FINDINGS ---\n{result}\n")

        if fact_check_report:
            parts.append(
                f"--- FACT-CHECK REPORT ---\n{fact_check_report}\n"
            )

        combined = "\n".join(parts)
        messages = [
            SystemMessage(content=SUPERVISOR_SYNTHESIZE_PROMPT),
            HumanMessage(content=combined),
        ]

        response = await self.llm.ainvoke(messages)
        return self._extract_text(response.content)
