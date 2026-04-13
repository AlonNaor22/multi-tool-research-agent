"""Multi-agent orchestrator: supervisor planning, parallel specialist dispatch, synthesis."""

import asyncio
import queue
import sys
import threading
from typing import AsyncGenerator, Dict, Generator, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import (
    SPECIALIST_FACT_CHECKER,
    EVENT_PLAN_CREATED, EVENT_PHASE_STARTED, EVENT_SPECIALIST_STARTED,
    EVENT_SPECIALIST_DONE, EVENT_PHASE_DONE, EVENT_SYNTHESIS_TOKEN,
    EVENT_DONE,
)
from src.utils import extract_chunk_text

from src.multi_agent.prompts import SUPERVISOR_SYNTHESIZE_PROMPT
from src.multi_agent.supervisor import Supervisor, DelegationPlan
from src.multi_agent.specialists import SpecialistAgent, SpecialistResult, build_specialists


class MultiAgentOrchestrator:
    """Phased parallel specialist orchestrator driven by a supervisor plan."""

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

    async def _astream_events(self, query: str) -> AsyncGenerator[dict, None]:
        """Yield typed events as the dependency-driven pipeline executes."""
        plan = await self.supervisor.acreate_delegation_plan(query)
        yield {"type": EVENT_PLAN_CREATED, "plan": plan}

        completed: Dict[str, SpecialistResult] = {}
        pending = set(plan.specialists)
        wave = 0

        while pending:
            # Find specialists whose dependencies are all satisfied
            ready = [
                s for s in plan.specialists
                if s in pending and all(d in completed for d in plan.depends_on.get(s, []))
            ]
            if not ready:
                break  # circular dependency safeguard

            yield {
                "type": EVENT_PHASE_STARTED,
                "phase_idx": wave,
                "specialists": ready,
            }
            for name in ready:
                yield {"type": EVENT_SPECIALIST_STARTED, "specialist": name, "phase_idx": wave}

            async def _run_specialist(name: str) -> SpecialistResult:
                task = plan.specialist_tasks.get(name, query)

                # Inject context only from declared dependencies
                deps = plan.depends_on.get(name, [])
                if name == SPECIALIST_FACT_CHECKER and completed:
                    findings_text = "\n\n".join(
                        f"[{n}]: {r.content}" for n, r in completed.items()
                        if n != SPECIALIST_FACT_CHECKER
                    )
                    task = (
                        f"Verify the key claims from these findings:\n\n"
                        f"{findings_text}\n\n"
                        f"Original question: {query}"
                    )
                elif deps:
                    dep_context = "\n".join(
                        f"[{d}]: {completed[d].content}" for d in deps if d in completed
                    )
                    task = task + f"\n\nContext from prior agents:\n{dep_context}"

                specialist = self.specialists.get(name)
                if specialist is None:
                    return SpecialistResult(name=name, content=f"[{name}] Unknown specialist.", error=True)
                return await specialist.run(task, callbacks=self.callbacks)

            results = await asyncio.gather(*[_run_specialist(n) for n in ready])

            for result in results:
                completed[result.name] = result
                pending.discard(result.name)
                yield {
                    "type": EVENT_SPECIALIST_DONE,
                    "specialist": result.name,
                    "phase_idx": wave,
                    "result_preview": result.content[:300],
                    "timed_out": result.timed_out,
                    "error": result.error,
                }

            yield {"type": EVENT_PHASE_DONE, "phase_idx": wave}
            wave += 1

        fact_check = completed.pop(SPECIALIST_FACT_CHECKER, None)
        fact_check_report = fact_check.content if fact_check and not fact_check.timed_out else ""

        # Skip timed-out and errored specialists in synthesis
        usable = {n: r for n, r in completed.items() if not r.timed_out and not r.error}

        synthesis_input = (
            f"Synthesize these specialist findings into a comprehensive answer.\n\n"
            f"Original question: {query}\n\n"
            + "\n\n".join(
                f"--- {n.upper()} AGENT ---\n{r.content}"
                for n, r in usable.items()
            )
            + (
                f"\n\n--- FACT-CHECK REPORT ---\n{fact_check_report}"
                if fact_check_report
                else ""
            )
        )

        final_answer = ""
        async for chunk in self.llm.astream([
            SystemMessage(content=SUPERVISOR_SYNTHESIZE_PROMPT),
            HumanMessage(content=synthesis_input),
        ]):
            text = extract_chunk_text(chunk)
            if text:
                final_answer += text
                yield {"type": EVENT_SYNTHESIS_TOKEN, "token": text}

        yield {
            "type": EVENT_DONE,
            "answer": final_answer,
            "plan": plan,
        }

    async def run(self, query: str) -> str:
        """Run the full pipeline and return the final synthesized answer."""
        final_answer = ""
        async for event in self._astream_events(query):
            if event.get("type") == EVENT_DONE:
                final_answer = event.get("answer", "")
        return final_answer

    async def run_verbose(self, query: str) -> str:
        """Run with verbose CLI output for each phase and specialist."""
        print(f"\n{'=' * 60}")
        print("Multi-Agent Orchestration")
        print(f"{'=' * 60}")
        print(f"\nQuery: {query}\n")

        final_answer = ""
        synthesis_started = False

        async for event in self._astream_events(query):
            etype = event.get("type")

            if etype == EVENT_PLAN_CREATED:
                plan = event["plan"]
                print("Supervisor is analyzing the query...")
                print(f"\nDelegation Plan ({len(plan.specialists)} specialists):")
                print(f"  Rationale: {plan.rationale}")
                for s in plan.specialists:
                    deps = plan.depends_on.get(s, [])
                    dep_str = f" (after {', '.join(deps)})" if deps else ""
                    print(f"  • {s}{dep_str}")
                if plan.needs_fact_check:
                    print("  Fact-checking: enabled")
                print()

            elif etype == EVENT_PHASE_STARTED:
                phase = event["specialists"]
                parallel_note = " (parallel)" if len(phase) > 1 else ""
                print(f"{'─' * 40}")
                print(
                    f"Phase {event['phase_idx'] + 1}{parallel_note}: "
                    f"{', '.join(phase)}"
                )
                print(f"{'─' * 40}")

            elif etype == EVENT_SPECIALIST_STARTED:
                print(f"  Starting {event['specialist']}...")

            elif etype == EVENT_SPECIALIST_DONE:
                print(f"  {event['specialist']} done.")

            elif etype == EVENT_SYNTHESIS_TOKEN:
                if not synthesis_started:
                    print(f"\n{'=' * 60}")
                    print("Synthesizing all findings...")
                    print(f"{'=' * 60}\n")
                    synthesis_started = True
                sys.stdout.write(event["token"])
                sys.stdout.flush()

            elif etype == EVENT_DONE:
                final_answer = event.get("answer", "")

        print()  # newline after streaming
        return final_answer

    def stream(self, query: str) -> Generator[dict, None, None]:
        """Sync generator bridging async events via a thread+queue for Streamlit."""
        _SENTINEL = object()
        q: "queue.Queue" = queue.Queue()

        def _runner() -> None:
            async def _drive() -> None:
                try:
                    async for event in self._astream_events(query):
                        q.put(("event", event))
                except BaseException as exc:  # propagate to consumer
                    q.put(("error", exc))
                finally:
                    q.put(("done", _SENTINEL))

            asyncio.run(_drive())

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()

        while True:
            kind, payload = q.get()
            if kind == "event":
                yield payload
            elif kind == "error":
                worker.join()
                raise payload
            else:  # "done"
                break

        worker.join()
