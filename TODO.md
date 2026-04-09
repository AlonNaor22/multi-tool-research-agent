# TODO — Consolidated Project Backlog

Aggregates open items from `IMPROVEMENTS.md` and `docs/LANGGRAPH_AUDIT.md`.
Within each category, items are ordered by **importance** (highest first).

## Status legend
- [ ] not started
- [~] in progress
- [x] done

> Completed historical items from `IMPROVEMENTS.md` (items 1–13) and the
> completed audit fix #1 are listed at the bottom under **Done** for context.

---

## LangGraph architecture & orchestration

Source: `docs/LANGGRAPH_AUDIT.md`. Focused on making the multi-agent and
plan-execute graphs faster, more parallel, and less duplicative.

1. [ ] **Phases serial even when independent; prior context injected blindly**
   (audit #2) — `src/multi_agent/orchestrator.py`
   Add `depends_on: Dict[str, List[str]]` to `DelegationPlan`; inject context
   only from declared dependencies. Consider LangGraph's `Send` API for native
   fan-out instead of hand-rolled `asyncio.gather` inside a single node.

2. [x] **Plan-execute graph has a no-op `replan` node** (audit #3) —
   `src/agent.py`
   Folded `replan_node` into `execute_step_node` (now returns
   `current_step: idx + 1` in its delta). `should_continue` moved to the
   post-`execute_step` conditional edge, which self-loops until all steps
   are done. Graph shrank from 4 → 3 nodes; one graph hop saved per step.
   Also closes audit #13 (edge simplification). All 460 tests still pass;
   verified end-to-end with a mocked 3-step plan.

3. [ ] **Plan-execute runs steps strictly sequentially** (audit #4) —
   `src/agent.py:628-659`
   Add `depends_on` to `PlanStep`; use `asyncio.gather` or LangGraph `Send`
   to batch independent steps.

4. [x] **`create_delegation_plan` is sync inside async nodes** (audit #8) —
   `src/multi_agent/supervisor.py`
   Added `async def acreate_delegation_plan` that uses
   `await self.llm.ainvoke(...)`. Orchestrator's `_astream_events` now
   awaits it instead of blocking the event loop. Sync variant kept for
   backward compat with 6 existing test call sites (standard LangChain
   `invoke`/`ainvoke` convention). Shared logic factored into
   `_plan_messages`, `_extract_text`, `_parse_plan_response`, and
   `_fallback_plan` static helpers. `Supervisor.synthesize` also updated
   to use `_extract_text` (killed one inline copy). All 460 tests pass.

5. [ ] **No per-specialist recursion_limit or timeouts** (audit #9) —
   `src/multi_agent/specialists.py:115`
   Move `recursion_limit` into `SPECIALIST_DEFINITIONS`. Wrap
   `specialist.run()` in `asyncio.wait_for` with a degraded-result fallback
   so one stuck specialist doesn't block the whole phase.

6. [ ] **Pydantic `model_dump` round-trip on every node call** (audit #5) —
   `src/agent.py:629,651,655`; `src/multi_agent/orchestrator.py:80`
   Store plan data once; keep mutable step state as separate keys with
   commutative reducers (`operator.or_` on dicts keyed by step index).

7. [x] **Plan-execute edges convoluted once `replan` is removed**
   (audit #13) — `src/agent.py`
   Closed as part of fix #2 above. Edges now: `START → create_plan →
   (synthesize | execute_step) → (execute_step | synthesize) → END`.

---

## Context & token efficiency

Source: `docs/LANGGRAPH_AUDIT.md`. Reduce wasted tokens and irrelevant
context passed between agents.

1. [ ] **`prior_context` truncates at 500 chars arbitrarily** (audit #6) —
   `src/multi_agent/orchestrator.py` (now inside `_astream_events`)
   500 chars ≈ 100 tokens — too short to carry findings, too long to be free.
   Pass full prior outputs, or generate a supervisor-summarized rollup per
   phase boundary.

2. [ ] **`prior_context` grows quadratically** (audit #10) —
   `src/multi_agent/orchestrator.py`
   Fine today, but same mitigation as the item above. Worth fixing together.

---

## Error handling & observability

Source: `docs/LANGGRAPH_AUDIT.md`.

1. [ ] **`DelegationPlan` fallback silently swallows exceptions** (audit #14) —
   `src/multi_agent/supervisor.py:112-120`
   Bare `except` hides parser failures. Log the exception with context before
   falling back to the single-research-agent default.

---

## Code cleanup & refactoring

Source: `docs/LANGGRAPH_AUDIT.md`.

1. [ ] **Anthropic content-block flattener duplicated four times**
   (audit #11) — `src/multi_agent/orchestrator.py` (module-level
   `_extract_chunk_text`), `src/multi_agent/specialists.py:127`
   (`_extract_answer`), `src/multi_agent/supervisor.py`
   (`Supervisor._extract_text`, added in audit #8 fix), `src/agent.py`
   (inline), and `src/planner.py` (inline inside `generate_plan`).
   Consolidate into one helper in `src/utils.py` (e.g.
   `flatten_anthropic_content`) and import everywhere.

---

## Capabilities — new features

Source: `IMPROVEMENTS.md`. Larger product-facing additions.

1. [ ] **RAG Pipeline & Semantic Memory** (improvements #14)
   - Add a vector store (ChromaDB or FAISS) for embedding-based retrieval
   - Store past research results and retrieved documents as embeddings
   - Let the agent query semantic memory before making external API calls
   - Support document ingestion (upload PDFs/URLs → chunk → embed → store)
   - Demonstrate long-term knowledge accumulation across sessions

2. [ ] **Human-in-the-Loop Interaction** (improvements #15)
   - Let the agent ask clarifying questions before proceeding
   - Present intermediate results for user approval at key decision points
   - Support user corrections mid-workflow ("focus more on X, skip Y")
   - Confidence-based escalation — agent asks for help when uncertain

---

## Deployment & production readiness

Source: `IMPROVEMENTS.md` #16.

1. [ ] FastAPI wrapper exposing the agent as a REST API
2. [ ] Dockerfile + docker-compose for containerized deployment
3. [ ] CI/CD pipeline (GitHub Actions) for tests, linting, build validation
4. [ ] Environment-based configuration (dev / staging / prod)
5. [ ] API authentication and rate limiting at the endpoint level

---

## Recommended ordering (quick reference)

For the next sprint focused on agent quality:
1. ~~LangGraph #2 — fold `replan` into `execute_step`~~ ✅
2. ~~LangGraph #4 — async `create_delegation_plan`~~ ✅
3. LangGraph #5 — per-specialist timeouts
4. LangGraph #1 — `depends_on` + `Send` fan-out (bigger refactor)
5. Context #1 + #2 — fix `prior_context` strategy
6. Cleanup #1 + Error handling #1

---

## Done (for context)

### LangGraph audit
- [x] **#1 Unify orchestrator around single async event generator**
  (audit #1, #7, #12, dead `loop.run_in_executor` branch)
  Introduced `_astream_events` as single source of truth; `run`,
  `run_verbose`, and sync `stream` all consume it. File shrank 453 → ~290
  lines; all 23 multi-agent tests pass. Committed as `1cd2654`.
- [x] **#3 + #13 Fold no-op `replan` node into `execute_step`**
  (`src/agent.py`) Plan-execute graph shrank from 4 → 3 nodes.
  `execute_step` now advances `current_step` in its own return dict; the
  post-step conditional edge self-loops until done. Verified end-to-end
  with a mocked 3-step plan; all 460 other tests still pass.
- [x] **#8 Async `create_delegation_plan`** (`src/multi_agent/supervisor.py`)
  Added `acreate_delegation_plan` using `await self.llm.ainvoke(...)`.
  Orchestrator `_astream_events` awaits it instead of blocking on the
  sync version. Shared logic factored into four `@staticmethod` helpers;
  `Supervisor.synthesize` also switched to the new `_extract_text` helper.
  Sync method retained for the 6 existing test sites. All 460 tests pass.

### IMPROVEMENTS.md (items 1–13, all done)
- [x] #1 Tests (283 across tools, memory, sessions, callbacks, observability)
- [x] #2 Migrate to native tool calling (`create_agent` / LangGraph)
- [x] #3 Evaluation suite (15 curated cases, tool + answer scoring)
- [x] #4 Streaming output (`StreamingCallbackHandler`, `stream_query`)
- [x] #5 Better error recovery (`check_tool_health`, fallback mapping)
- [x] #6 Web UI (Streamlit chat with live tool status)
- [x] #7 Observability (`ObservabilityCallbackHandler`, cost/token metrics)
- [x] #8 Async / true parallelism (native `aiohttp`, `asyncio.gather`)
- [x] #9 Config cleanup (selectable model, full-tool banner)
- [x] #10 Rate limiting (per-session token budget)
- [x] #11 Session schema versioning (`version: "1.0"`)
- [x] #12 Plan-and-execute loop (LangGraph StateGraph + UI wiring)
- [x] #13 Multi-agent orchestration (supervisor + 5 specialists + fact-checker)
