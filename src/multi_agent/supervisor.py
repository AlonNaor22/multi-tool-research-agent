"""Supervisor that produces delegation plans and synthesizes specialist outputs."""

import json
import logging
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.constants import SPECIALIST_FACT_CHECKER, SPECIALIST_RESEARCH
from src.multi_agent.prompts import SUPERVISOR_PLAN_PROMPT
from src.multi_agent.specialists import SPECIALIST_DEFINITIONS
from src.utils import flatten_content

logger = logging.getLogger(__name__)

# ─── Module overview ───────────────────────────────────────────────
# LLM-based supervisor that converts a user query into a DelegationPlan
# (which specialists to run, their tasks, and dependency ordering)
# and provides a fallback single-specialist plan on parse failure.
# ───────────────────────────────────────────────────────────────────


class DelegationPlan(BaseModel):
    """Dependency-driven specialist delegation plan."""
    query: str
    specialists: List[str] = Field(default_factory=list)
    specialist_tasks: Dict[str, str] = Field(default_factory=dict)
    depends_on: Dict[str, List[str]] = Field(default_factory=dict)
    needs_fact_check: bool = False
    rationale: str = ""


class Supervisor:
    """LLM-only supervisor for task delegation and result synthesis."""

    def __init__(self, llm: ChatAnthropic):
        self.llm = llm

    # Takes (query). Builds the system + human message pair for the planning LLM call.
    @staticmethod
    def _plan_messages(query: str) -> List[BaseMessage]:
        return [
            SystemMessage(content=SUPERVISOR_PLAN_PROMPT),
            HumanMessage(content=f"Create a delegation plan for: {query}"),
        ]

    # Takes (content, query). Parses raw LLM JSON into a validated DelegationPlan,
    # filtering unknown specialists and wiring up fact-checker dependencies.
    @staticmethod
    def _parse_plan_response(content: str, query: str) -> DelegationPlan:
        """Parse LLM JSON into a DelegationPlan; raises ValueError on malformed output."""
        data = json.loads(content)

        specialist_tasks = data.get("specialist_tasks", {})
        depends_on = data.get("depends_on", {})
        needs_fact_check = data.get("needs_fact_check", False)
        rationale = data.get("rationale", "")

        valid_names = set(SPECIALIST_DEFINITIONS.keys())
        specialists = [s for s in data.get("specialists", []) if s in valid_names]

        if not specialists:
            raise ValueError("No valid specialists")

        # Validate depends_on: remove references to unknown specialists
        depends_on = {
            s: [d for d in deps if d in valid_names and d in specialists]
            for s, deps in depends_on.items()
            if s in specialists
        }
        # Ensure every specialist has a depends_on entry (default: no deps)
        for s in specialists:
            depends_on.setdefault(s, [])

        if needs_fact_check:
            if SPECIALIST_FACT_CHECKER not in specialists:
                specialists.append(SPECIALIST_FACT_CHECKER)
            # Fact-checker depends on all other specialists
            others = [s for s in specialists if s != SPECIALIST_FACT_CHECKER]
            depends_on[SPECIALIST_FACT_CHECKER] = others
            if SPECIALIST_FACT_CHECKER not in specialist_tasks:
                specialist_tasks[SPECIALIST_FACT_CHECKER] = (
                    f"Verify the key claims from the research findings "
                    f"about: {query}"
                )

        return DelegationPlan(
            query=query,
            specialists=specialists,
            specialist_tasks=specialist_tasks,
            depends_on=depends_on,
            needs_fact_check=needs_fact_check,
            rationale=rationale,
        )

    # Takes (query). Returns a single-research-specialist plan used when parsing fails.
    @staticmethod
    def _fallback_plan(query: str) -> DelegationPlan:
        return DelegationPlan(
            query=query,
            specialists=[SPECIALIST_RESEARCH],
            specialist_tasks={SPECIALIST_RESEARCH: query},
            depends_on={SPECIALIST_RESEARCH: []},
            rationale="Fallback — could not parse delegation plan.",
        )

    # Takes (query). Invokes the LLM synchronously and returns a DelegationPlan.
    def create_delegation_plan(self, query: str) -> DelegationPlan:
        """Produce a DelegationPlan synchronously; falls back on parse failure."""
        try:
            response = self.llm.invoke(self._plan_messages(query))
            content = flatten_content(response.content)
            return self._parse_plan_response(content, query)
        except Exception:
            logger.warning(
                "Failed to parse delegation plan for query %r; "
                "falling back to single-research-agent plan.",
                query[:120],
                exc_info=True,
            )
            return self._fallback_plan(query)

    # Takes (query). Async version of create_delegation_plan.
    async def acreate_delegation_plan(self, query: str) -> DelegationPlan:
        """Async version of create_delegation_plan; falls back on parse failure."""
        try:
            response = await self.llm.ainvoke(self._plan_messages(query))
            content = flatten_content(response.content)
            return self._parse_plan_response(content, query)
        except Exception:
            logger.warning(
                "Failed to parse delegation plan for query %r; "
                "falling back to single-research-agent plan.",
                query[:120],
                exc_info=True,
            )
            return self._fallback_plan(query)

