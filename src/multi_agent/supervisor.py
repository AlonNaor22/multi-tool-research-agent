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


class DelegationPlan(BaseModel):
    """Phased specialist delegation plan with parallel execution per phase."""
    query: str
    execution_phases: List[List[str]] = Field(default_factory=list)
    specialist_tasks: Dict[str, str] = Field(default_factory=dict)
    needs_fact_check: bool = False
    rationale: str = ""


class Supervisor:
    """LLM-only supervisor for task delegation and result synthesis."""

    def __init__(self, llm: ChatAnthropic):
        self.llm = llm

    @staticmethod
    def _plan_messages(query: str) -> List[BaseMessage]:
        return [
            SystemMessage(content=SUPERVISOR_PLAN_PROMPT),
            HumanMessage(content=f"Create a delegation plan for: {query}"),
        ]

    @staticmethod
    def _parse_plan_response(content: str, query: str) -> DelegationPlan:
        """Parse LLM JSON into a DelegationPlan; raises ValueError on malformed output."""
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
                [s for s in phase if s != SPECIALIST_FACT_CHECKER]
                for phase in execution_phases
            ]
            execution_phases = [p for p in execution_phases if p]
            execution_phases.append([SPECIALIST_FACT_CHECKER])

            if SPECIALIST_FACT_CHECKER not in specialist_tasks:
                specialist_tasks[SPECIALIST_FACT_CHECKER] = (
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
        return DelegationPlan(
            query=query,
            execution_phases=[[SPECIALIST_RESEARCH]],
            specialist_tasks={SPECIALIST_RESEARCH: query},
            needs_fact_check=False,
            rationale="Fallback — could not parse delegation plan.",
        )

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

