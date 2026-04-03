"""Multi-agent orchestration system.

Implements a supervisor/worker pattern where specialist agents collaborate
on complex queries. The supervisor delegates tasks, specialists execute
them (potentially in parallel), and results are synthesized into a final answer.
"""

from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.supervisor import Supervisor, DelegationPlan
from src.multi_agent.specialists import SpecialistAgent, build_specialists

__all__ = [
    "MultiAgentOrchestrator",
    "Supervisor",
    "DelegationPlan",
    "SpecialistAgent",
    "build_specialists",
]
