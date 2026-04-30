"""Supervisor that produces delegation plans and synthesizes specialist outputs."""

import logging
import re
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.constants import (
    SPECIALIST_FACT_CHECKER, SPECIALIST_RESEARCH,
    SPECIALIST_MATH, SPECIALIST_ANALYSIS, SPECIALIST_TRANSLATION,
)
from src.multi_agent.prompts import SUPERVISOR_PLAN_PROMPT
from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

logger = logging.getLogger(__name__)

# ─── Module overview ───────────────────────────────────────────────
# LLM-based supervisor that converts a user query into a DelegationPlan
# (which specialists to run, their tasks, and dependency ordering)
# using Anthropic's native structured outputs. Provides a domain-aware
# heuristic fallback when the API call itself fails.
# ───────────────────────────────────────────────────────────────────

# Keyword routing for the heuristic fallback. Order matters: the first
# matching domain wins.
_DOMAIN_KEYWORDS: Dict[str, tuple] = {
    SPECIALIST_MATH: (
        "derivative", "integral", "differentiate", "integrate",
        "graph", "plot", "visualize",
        "equation", "solve", "calculate", "compute",
        "matrix", "matrices", "eigenvalue", "rref",
        "polynomial", "quadratic", "cubic", "function",
        "calculus", "algebra", "trig", "trigonometric",
        "sin(", "cos(", "tan(", "log(", "sqrt(",
        "convert", "currency",
    ),
    SPECIALIST_ANALYSIS: (
        "csv", "dataframe", "dataset", "data analysis",
        "histogram", "boxplot", "correlation", "regression",
    ),
    SPECIALIST_TRANSLATION: (
        "translate", "translation", "in french", "in spanish",
        "in german", "in chinese", "in japanese",
    ),
}


def _matches_keyword(query: str, keyword: str) -> bool:
    """Word-boundary match for word tokens; substring match for tokens with '('."""
    if "(" in keyword:
        return keyword in query  # e.g. "sin(" — already specific enough
    return re.search(rf"\b{re.escape(keyword)}\b", query) is not None


# Takes (query). Picks a specialist by keyword match for fallback routing.
# Returns SPECIALIST_RESEARCH when no domain keyword matches.
def _heuristic_specialist(query: str) -> str:
    """Pick a specialist by keyword match for fallback routing."""
    q = query.lower()
    for specialist, keywords in _DOMAIN_KEYWORDS.items():
        if any(_matches_keyword(q, kw) for kw in keywords):
            return specialist
    return SPECIALIST_RESEARCH


class _PlanResponse(BaseModel):
    """LLM-facing schema for structured-output delegation planning."""
    specialists: List[str] = Field(
        default_factory=list,
        description="Names of specialists to dispatch (research, math, analysis, fact_checker, translation).",
    )
    specialist_tasks: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of specialist name to its task description.",
    )
    depends_on: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of specialist name to its upstream dependencies.",
    )
    needs_fact_check: bool = Field(
        default=False,
        description="True if the answer benefits from independent fact-checking.",
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of the delegation strategy.",
    )


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
        self._planner = llm.with_structured_output(_PlanResponse)

    # Takes (query). Invokes the LLM synchronously and returns a DelegationPlan.
    def create_delegation_plan(self, query: str) -> DelegationPlan:
        """Produce a DelegationPlan synchronously; falls back on API failure."""
        try:
            response = self._planner.invoke(self._plan_messages(query))
            return self._finalize_plan(response, query)
        except Exception:
            return self._fallback_plan(query)

    # Takes (query). Async version of create_delegation_plan.
    async def acreate_delegation_plan(self, query: str) -> DelegationPlan:
        """Async version of create_delegation_plan; falls back on API failure."""
        try:
            response = await self._planner.ainvoke(self._plan_messages(query))
            return self._finalize_plan(response, query)
        except Exception:
            return self._fallback_plan(query)

    # Takes (query). Builds the system + human message pair for the planning LLM call.
    @staticmethod
    def _plan_messages(query: str) -> List[BaseMessage]:
        return [
            SystemMessage(content=SUPERVISOR_PLAN_PROMPT),
            HumanMessage(content=f"Create a delegation plan for: {query}"),
        ]

    # Takes (response, query). Validates the LLM's structured plan, drops unknown
    # specialists, and wires up the fact-checker dependency graph.
    @staticmethod
    def _finalize_plan(response: _PlanResponse, query: str) -> "DelegationPlan":
        """Convert a validated _PlanResponse into a DelegationPlan."""
        valid_names = set(SPECIALIST_DEFINITIONS.keys())
        specialists = [s for s in response.specialists if s in valid_names]

        if not specialists:
            raise ValueError("Plan returned no valid specialists")

        depends_on = {
            s: [d for d in response.depends_on.get(s, []) if d in specialists]
            for s in specialists
        }

        specialist_tasks = dict(response.specialist_tasks)

        if response.needs_fact_check:
            if SPECIALIST_FACT_CHECKER not in specialists:
                specialists.append(SPECIALIST_FACT_CHECKER)
            others = [s for s in specialists if s != SPECIALIST_FACT_CHECKER]
            depends_on[SPECIALIST_FACT_CHECKER] = others
            specialist_tasks.setdefault(
                SPECIALIST_FACT_CHECKER,
                f"Verify the key claims from the research findings about: {query}",
            )

        return DelegationPlan(
            query=query,
            specialists=specialists,
            specialist_tasks=specialist_tasks,
            depends_on=depends_on,
            needs_fact_check=response.needs_fact_check,
            rationale=response.rationale,
        )

    # Takes (query). Routes by keyword heuristic when the structured-output call fails.
    @staticmethod
    def _fallback_plan(query: str) -> DelegationPlan:
        """Domain-aware fallback used only when the LLM API itself fails."""
        specialist = _heuristic_specialist(query)
        logger.warning(
            "Falling back to heuristic routing → %s for query %r",
            specialist, query[:120],
            exc_info=True,
        )
        return DelegationPlan(
            query=query,
            specialists=[specialist],
            specialist_tasks={specialist: query},
            depends_on={specialist: []},
            rationale=f"Fallback — structured-output call failed; routed to {specialist} by keyword heuristic.",
        )
