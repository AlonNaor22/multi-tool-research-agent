# TODO — Consolidated Project Backlog

Within each category, items are ordered by **importance** (highest first).

## Status legend
- [ ] not started
- [~] in progress
- [x] done

---

## Architecture audit (latest pass)

Dead code, missing wiring, and inconsistencies found across all three
research modes.

1. [x] **Delete dead `_build_plan_execute_graph()`** — `src/agent.py`
   Removed 90 lines of unused LangGraph StateGraph + `StateGraph`/`END`/
   `START` imports, `TypedDict`, `Annotated`, `operator`. The plan-execute
   methods use their own inline loops and never called this graph.

2. [x] **Delete dead `Supervisor.synthesize()`** — `src/multi_agent/supervisor.py`
   Removed the method and its `SUPERVISOR_SYNTHESIZE_PROMPT` import.

3. [x] **Wire `SUPERVISOR_SYNTHESIZE_PROMPT` into orchestrator synthesis** —
   `src/multi_agent/orchestrator.py`
   Synthesis now uses `[SystemMessage(SUPERVISOR_SYNTHESIZE_PROMPT),
   HumanMessage(content)]` instead of a bare HumanMessage.

4. [x] **Add metrics save to multi-agent mode** — `src/agent.py`
   Both `multi_agent_query()` and `multi_agent_stream()` now reset
   the observability callback before, and save metrics + record tokens
   after execution.

5. [x] **Add callbacks to plan-execute methods** — `src/agent.py`
   Both `plan_and_execute()` and `plan_and_execute_stream()` now reset
   the observability callback, pass it to agent calls, and save
   metrics + record tokens after completion.

6. [x] **Centralize mode selection into `route_query()`** — `src/agent.py`
   New `route_query(query, mode)` sync generator routes to the correct
   mode (multi-agent, plan-execute, or direct) based on mode string.
   New `_direct_stream(query)` implements direct mode as a sync generator
   yielding the same event types as other modes. `app.py` and `main.py`
   can delegate to `route_query()` instead of duplicating routing logic.

7. [x] **Add error handling to plan-execute** — `src/agent.py`
   `plan_and_execute()` now wraps each step's `ainvoke` and the
   synthesis `astream` in try/except, returning degraded error strings
   instead of crashing the pipeline.

---

## LangGraph architecture & orchestration

1. [ ] **Phases serial even when independent; prior context injected blindly**
   (audit #2) — `src/multi_agent/orchestrator.py`
   Add `depends_on: Dict[str, List[str]]` to `DelegationPlan`; inject
   context only from declared dependencies.

2. [ ] **Plan-execute runs steps strictly sequentially** (audit #4) —
   `src/agent.py`
   Add `depends_on` to `PlanStep`; use `asyncio.gather` to batch
   independent steps.

3. [ ] **Pydantic `model_dump` round-trip on every node call** (audit #5) —
   `src/agent.py`
   Store plan data once; keep mutable step state as separate keys.

---

## Context & token efficiency

1. [ ] **`prior_context` truncates at 500 chars arbitrarily** (audit #6) —
   `src/multi_agent/orchestrator.py`
   Pass full prior outputs, or generate a supervisor-summarized rollup
   per phase boundary.

2. [ ] **`prior_context` grows quadratically** (audit #10) —
   Same mitigation as above. Fix together.

---

## Code cleanup & refactoring

1. [ ] **`SPECIALIST_DEFINITIONS` is a stringly-typed dict** —
   `src/multi_agent/specialists.py`
   Convert to a `TypedDict` or dataclass schema.

2. [ ] **Sync-tool cancellation on specialist timeout** —
   `src/multi_agent/specialists.py`
   Sync tools in executor threads keep running past the timeout.

3. [ ] **Timeout sentinel vs plain degraded string** —
   `src/multi_agent/specialists.py` + orchestrator
   Return a structured signal so synthesis can skip timed-out specialists.

4. [ ] **Research agent prompt vague about tool names** —
   `src/multi_agent/prompts.py`
   `RESEARCH_AGENT_PROMPT` doesn't mention `parallel_search` or list
   exact tool names matching `SPECIALIST_DEFINITIONS`.

---

## Capabilities — new features

1. [ ] **RAG Pipeline & Semantic Memory**
   - Vector store (ChromaDB/FAISS) for embedding-based retrieval
   - Store past research results as embeddings
   - Query semantic memory before external API calls
   - Document ingestion (PDFs/URLs → chunk → embed → store)

2. [ ] **Human-in-the-Loop Interaction**
   - Agent asks clarifying questions before proceeding
   - Intermediate results for user approval
   - Confidence-based escalation

---

## Deployment & production readiness

1. [ ] FastAPI wrapper exposing the agent as a REST API
2. [ ] Dockerfile + docker-compose for containerized deployment
3. [ ] CI/CD pipeline (GitHub Actions) for tests, linting, build validation
4. [ ] Environment-based configuration (dev / staging / prod)
5. [ ] API authentication and rate limiting at the endpoint level

---

## Recommended ordering (quick reference)

Architecture audit fixes (do first — these are bugs/dead code):
1. Architecture #1 — delete dead plan-execute graph
2. Architecture #2 — delete dead Supervisor.synthesize()
3. Architecture #3 — wire SUPERVISOR_SYNTHESIZE_PROMPT into orchestrator
4. Architecture #4 — add metrics to multi-agent mode
5. Architecture #5 — add callbacks to plan-execute
6. Architecture #6 — centralize mode selection
7. Architecture #7 — error handling for plan-execute + synthesis

Then deeper improvements:
8. LangGraph #1 — depends_on + fan-out
9. Context #1+#2 — prior_context strategy
10. Cleanup items

---

## Done (for context)

### Architecture audit
- [x] **Unify orchestrator around single async event generator**
  (audit #1, #7, #12) — `_astream_events` as single source of truth
- [x] **Fold no-op `replan` node into `execute_step`** (audit #3, #13)
- [x] **Async `create_delegation_plan`** (audit #8)
- [x] **Per-specialist `recursion_limit` + `timeout_seconds`** (audit #9)
- [x] **Consolidate content-block flattener** (audit #11)
- [x] **Log supervisor fallback exceptions** (audit #14)

### Codebase improvements
- [x] Migrate tools to `@tool` decorator + `BaseTool` subclass
- [x] Extract hardcoded strings to constants + localization file
- [x] Consolidate duplicated boilerplate across 21 tool files
- [x] Apply `@safe_tool_call` + `require_input` across 21 tools
- [x] Trim all docstrings to 2 lines
- [x] Optimize test suite: 84s → 12s (parallel, fast retries, dedup)

### IMPROVEMENTS.md (items 1–13, all done)
- [x] #1 Tests
- [x] #2 Migrate to native tool calling
- [x] #3 Evaluation suite
- [x] #4 Streaming output
- [x] #5 Better error recovery
- [x] #6 Web UI
- [x] #7 Observability
- [x] #8 Async / true parallelism
- [x] #9 Config cleanup
- [x] #10 Rate limiting
- [x] #11 Session schema versioning
- [x] #12 Plan-and-execute loop
- [x] #13 Multi-agent orchestration
