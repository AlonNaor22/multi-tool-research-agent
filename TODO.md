# TODO — Consolidated Project Backlog

Within each category, items are ordered by **importance** (highest first).

## Status legend
- [ ] not started
- [~] in progress
- [x] done

---

## Structured output & tool-schema enforcement (audit)

Background: `src/multi_agent/supervisor.py` was converted from `json.loads(llm.invoke(...))` to
`llm.with_structured_output(_PlanResponse)` so Anthropic's tool-use API enforces the Pydantic
schema. The same fragile pattern exists elsewhere — both in direct LLM calls and in tool
definitions that take a single `query: str` and parse it as JSON inside the function.

1. [x] **Convert `src/planner.py` to `with_structured_output`** —
   `src/planner.py`
   Replaced `llm.invoke()` + `json.loads()` with `llm.with_structured_output(_PlanResponse).invoke()`.
   Added LLM-facing `_StepResponse` / `_PlanResponse` schemas separate from the runtime
   `ResearchStep` / `ResearchPlan` (so the LLM never sees `status`/`findings`). Trimmed
   the JSON-formatting block from `_PLANNER_SYSTEM`. `_parse_depends_on` now takes the
   typed dict directly (Pydantic handled the `int(key)` parsing). Fallback path now logs
   a warning, matching the supervisor pattern.

2. [x] **Migrate tools that REQUIRE JSON input to `BaseTool` + `args_schema`** —
   `src/tools/parallel_tool.py`, `src/tools/datetime_tool.py`
   - `parallel_tool.py`: now `ParallelSearchTool(BaseTool)` with `ParallelSearchInput`
     (`searches: List[SearchSpec]`, `min_length=1`, `max_length=10`); `SearchSpec.type`
     is `Literal["web","wikipedia","news","arxiv"]`. `_arun` normalizes validated
     `SearchSpec` instances to dicts so the inner `parallel_search(searches)` stays
     Pydantic-agnostic.
   - `datetime_tool.py`: now `DatetimeTool(BaseTool)` with `DatetimeInput`; `operation`
     is `Literal["now","add","diff","convert","info","business_days"]`. Renamed JSON
     keys to Python-safe params (`from`→`date_from`, `to`→`date_to`,
     `datetime`→`convert_datetime`).
   - Tests rewritten: dropped JSON-parsing fixtures, added `TestParallelSearchSchema` /
     `TestDatetimeSchema` for Pydantic boundary validation. End-to-end `tool.ainvoke({...})`
     verified — schema rejects bad operations and out-of-range list sizes.

3. [x] **Migrate string-or-JSON tools to `BaseTool` + `args_schema`** —
   `src/tools/search_tool.py`, `wikipedia_tool.py`, `news_tool.py`, `arxiv_tool.py`,
   `google_scholar_tool.py`, `reddit_tool.py`, `github_tool.py`, `csv_tool.py`, `scraper_tool.py`
   All nine converted to `<Tool>Class(BaseTool)` with a Pydantic `<Tool>Input` schema
   wired via `args_schema`. The underlying async function on each module kept `query`
   (or `path`/`url`) as the first positional arg with typed defaults for the previously
   JSON-encoded options, so `parallel_tool.py`'s `await func(query)` callers still
   work. `weather_tool.get_weather()` was a tenth `parse_tool_input` caller — its
   helper was rewritten to take typed kwargs. `parse_tool_input` and the now-unused
   `json` import were deleted from `src/utils.py`. Tests rewritten: JSON-string input
   tests replaced with typed-kwarg tests, plus a `Test<Tool>Schema` class per tool
   asserting Pydantic `ValidationError` on missing/invalid fields, and a
   `Test<Tool>Tool` class asserting the `name` and `args_schema` wiring. Full suite:
   489/489 passing.

4. [x] **Eliminate the `MATH_STRUCTURED:` text-channel** —
   `app.py`, `main.py`, `src/tools/math_formatter.py`, `src/tools/calculator_tool.py`
   Chose option A: `calculator_tool` now calls `format_math_from_dict()` directly
   and returns ready-to-display KaTeX markdown. The `"MATH_STRUCTURED:" + json.dumps(...)`
   prefix channel is gone, and with it the brittle `_auto_format_math_structured()`
   brace-counter in `app.py`, the mid-stream MATH_STRUCTURED interception in
   `_stream_display()`, and `main.py`'s `_clean_math_output()`. `math_formatter.py`
   is now a private helper module (no LangChain tool); `math_formatter` was removed
   from the agent's tool list, the multi-agent math specialist's tool list, and
   `src/tools/__init__.py`. Prompts updated: "When calculator returns
   MATH_STRUCTURED:, ALWAYS pass to math_formatter" → "calculator output is already
   formatted KaTeX markdown — include it verbatim." Dead-code cleanup: removed the
   `<!-- MATH_HTML -->` HTML-sentinel stripper that nothing produced, and the
   now-unused `UI.status.formatting_math` string. Tests rewritten for the
   `format_math_from_dict(dict)` signature, plus a new
   `TestCalculatorReturnsFormattedMarkdown` class verifying calculator output
   contains no `MATH_STRUCTURED` prefix and is renderable markdown. Full suite:
   490/490 passing.

5. [ ] **Structured fact-checker output** (optional, only if surfacing in UI) —
   `src/multi_agent/prompts.py:160-166`, `src/multi_agent/orchestrator.py:119-120`
   `FACT_CHECKER_PROMPT` asks for free-text `CONFIRMED:` / `CONTRADICTED:` / `UNVERIFIABLE:`
   lines. The fact-checker is a tool-using agent so its main call can't use
   `with_structured_output` directly, but a one-shot post-pass via
   `llm.with_structured_output(FactCheckReport)` (list of `{claim, verdict: Literal[...], source}`)
   would unlock per-verdict UI badges, telemetry on confidence ratios, and reliable
   downstream consumption by the synthesizer.

---

## LangGraph architecture & orchestration

1. [x] **Plan-execute runs steps strictly sequentially** (audit #4) —
   `src/agent.py`
   Added `depends_on` to `ResearchPlan`; wave-based parallel execution
   with `asyncio.gather` and async-to-sync bridge for Streamlit streaming.
   (commit 3cac7e3)

---

## Agent state & graph improvements (from state audit)

Do in this order — each builds on the previous:

1. [x] **Explicit state schema** — `src/agent.py`
   Defined `ResearchAgentState(AgentState)` and passed as `state_schema`
   to `create_agent()`. Specialists use base `AgentState` explicitly.

2. [x] **Unified persistence with SqliteSaver** — `src/agent.py`, `src/session_manager.py`
   Replaced SimpleMemory + JSON session files with SqliteSaver
   checkpointing to `sessions/checkpoints.db`. Auto-persists full
   graph state after every node. Sessions listed/loaded/deleted from
   SQLite. No manual "Save" button needed.

3. [x] **Inject dependency context as messages** — `src/agent.py`
   `_run_step` now replays each declared dependency as a
   `(HumanMessage, AIMessage)` pair before the current step's task,
   so prior findings arrive as real prior assistant turns. Also
   replaced sync `SqliteSaver` with `AsyncSqliteSaver` (bootstrapped
   on a persistent daemon loop) to unblock async `ainvoke`/`astream`.

4. [x] **History management as agent middleware** — `src/agent.py`
   Original framing was stale (`SimpleMemory` was already deleted by the
   SqliteSaver migration). Reframed: SqliteSaver let history grow
   without bound. Added `_HistorySummarizerMiddleware` (subclass of
   `langchain.agents.middleware.AgentMiddleware`) with both
   `before_model` and `abefore_model` hooks. When the messages channel
   exceeds `HISTORY_TRIM_THRESHOLD_TOKENS` (8000), the middleware
   summarizes the older portion via the LLM into one marked
   `AIMessage` and keeps the active turn (last HumanMessage onward)
   verbatim plus as much earlier history as fits in
   `HISTORY_KEEP_RECENT_TOKENS` (4000). Tool-use/tool-result pairing
   is preserved across the drop/keep split. Telemetry: new
   `summarized_message_count` field on `ResearchAgentState`.

---

## Code cleanup & refactoring

*All items completed.*

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

1. [x] **FastAPI wrapper exposing the agent as a REST API** —
   `src/api/`, `serve.py`
   Built a FastAPI app with `lifespan`-managed `ResearchAgent` singleton and a
   per-app `asyncio.Lock` that serializes mutating operations (the agent has
   mutable state — `current_session_id`, callbacks, rate limiter — so concurrent
   requests need to queue).

   Endpoints: `GET /health` (status + enabled/disabled tool lists),
   `POST /query` (blocking, returns answer + tokens_used + duration),
   `POST /query/stream` (SSE via `sse-starlette`, emits typed events including
   `synthesis_token`, `step_tool`, `phase_started`, `done`),
   `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` (all backed by
   the AsyncSqliteSaver checkpoint DB).

   Mode parameter: `auto | direct | plan | multi` (lowercase, REST-idiomatic).
   The dependency module maps these to the internal `MODE_*` constants so
   the route handlers never see unknown values.

   New `ResearchAgent.set_session_id()` lets the API scope each request to a
   caller-supplied thread without requiring an existing checkpoint
   (cf. `load_session()` which expects one). `serve.py` wraps uvicorn with
   `--host/--port/--reload` flags. Swagger UI auto-served at `/docs`.

   18 new tests in `tests/test_api.py` use FastAPI's `TestClient` over a
   stub agent (no LLM, no SQLite, no tool probes) and cover: health
   ok/degraded; direct/plan/multi dispatch; caller-supplied session_id;
   Pydantic 422 on invalid mode / empty query; 429 on RateLimitExceeded;
   500 on unexpected exceptions; SSE typed events terminated by `done`;
   stream error becomes `error` event; sessions list/get/delete + 404 paths.

   Drive-by fix: rewrote `tests/test_news.py` to patch
   `src.tools.news_tool.DDGS` per-test instead of mutating
   `sys.modules['duckduckgo_search']` (matching `test_search.py`'s pattern) —
   the module-level mock was racing with `test_search.py` /
   `test_parallel.py` under `pytest-xdist` and producing 2-3 flaky failures.
   Full suite now 508/508, stable across reruns.

2. [x] **Dockerfile + docker-compose for containerized deployment** —
   `Dockerfile`, `.dockerignore`, `docker-compose.yml`
   Multi-stage `Dockerfile`: a `python:3.12-slim` builder installs deps into
   an isolated `/opt/venv`, then a slim runtime stage copies just the venv +
   source. Build tools (`gcc`, `g++`) never reach the final image. Final size
   is ~250 MB vs ~750 MB single-stage. Runs as non-root `app` (UID 1000) for
   safer volume mounts. `HEALTHCHECK` probes `/health` via stdlib `urllib`
   (no `curl` dependency).

   `.dockerignore` keeps the build context lean — excludes `__pycache__/`,
   `venv/`, `.git/`, `.claude/`, `tests/`, `*.md` (except README), and the
   host-side state dirs that get volume-mounted at runtime
   (`sessions/`, `output/`, `observability/`). Also blocks `.env` so secrets
   never bake into the image; compose reads them via `env_file: .env`.

   `docker-compose.yml` is a one-service stack: builds the image, maps port
   8000, sources env from `.env`, bind-mounts the three state dirs so SQLite
   checkpoints, chart PNGs, and metrics JSONL survive `docker compose down`.
   `restart: unless-stopped` for resilience, healthcheck matching the
   Dockerfile.

   README updated with a Docker quickstart section.

   Not smoke-tested in this sandbox (Docker daemon unavailable here).
   Verify locally with `docker compose up --build` + `curl localhost:8000/health`.
3. [ ] CI/CD pipeline (GitHub Actions) for tests, linting, build validation
4. [ ] Environment-based configuration (dev / staging / prod)
5. [x] **API authentication and rate limiting at the endpoint level** —
   `src/api/auth.py`, `src/api/rate_limit.py`, `config.py`,
   `src/api/routes/*`, `docker-compose.yml`
   Bearer-token auth via FastAPI's `HTTPBearer` security scheme (so
   `/docs` gets an Authorize button for free). The `verify_token`
   dependency reads `API_AUTH_TOKEN` per-request, so tests can flip the
   env var via monkeypatch and the runtime can hot-swap the token
   without restart. When unset, auth is a no-op (dev mode) and the
   startup log emits a prominent WARNING. `/health` is intentionally
   left open so Docker/k8s liveness probes work without credentials;
   `/query`, `/query/stream`, and every `/sessions/*` route require auth.

   Rate limiting via `slowapi` (in-memory, per remote IP, X-Forwarded-For
   aware): `/query` and `/query/stream` 10/min (LLM-expensive),
   `/sessions` reads 60/min, `/sessions/{id}` DELETE 30/min. A custom
   429 handler in `src/api/rate_limit.py` augments slowapi's default
   with a `Retry-After` header so HTTP clients can back off cleanly
   (slowapi only emits `X-RateLimit-*` out of the box). Tests flip
   `limiter.enabled = False` autouse so the existing 18 API tests run
   without hitting limits; dedicated `TestRateLimit` re-enables and
   asserts the 11th `/query` request returns 429 with `Retry-After`.

   6 new auth tests (missing/invalid/valid token; /health stays open;
   sessions also protected; default-disabled in tests) + 2 rate-limit
   tests. README + `docker-compose.yml` documentation updated. Full
   suite: 516/516.

---

## Done (for context)

### Architecture audit
- [x] **Delete dead `_build_plan_execute_graph()`** — `src/agent.py`
- [x] **Delete dead `Supervisor.synthesize()`** — `src/multi_agent/supervisor.py`
- [x] **Wire `SUPERVISOR_SYNTHESIZE_PROMPT` into orchestrator synthesis** — `src/multi_agent/orchestrator.py`
- [x] **Add metrics save to multi-agent mode** — `src/agent.py`
- [x] **Add callbacks to plan-execute methods** — `src/agent.py`
- [x] **Centralize mode selection into `route_query()`** — `src/agent.py`
- [x] **Add error handling to plan-execute** — `src/agent.py`
- [x] **Unify orchestrator around single async event generator** (audit #1, #7, #12)
- [x] **Fold no-op `replan` node into `execute_step`** (audit #3, #13)
- [x] **Async `create_delegation_plan`** (audit #8)
- [x] **Per-specialist `recursion_limit` + `timeout_seconds`** (audit #9)
- [x] **Consolidate content-block flattener** (audit #11)
- [x] **Log supervisor fallback exceptions** (audit #14)

### LangGraph architecture & orchestration
- [x] **depends_on + fan-out for orchestrator phases** — wave-based parallel dispatch with `asyncio.gather`, context only from declared deps (commit 2d43a74)
- [x] **Pydantic `model_dump` round-trip** — code already uses `model_copy()` efficiently; no round-trip issue exists
- [x] **Plan-execute parallel steps** — added `depends_on` to `ResearchPlan`, wave-based execution with async-to-sync bridge (commit 3cac7e3)

### Context & token efficiency
- [x] **`prior_context` truncation + quadratic growth** — resolved by `depends_on` approach; no 500-char truncation exists, context injected only from declared dependencies

### Code cleanup
- [x] **Research agent prompt vague about tool names** — prompt now lists specific tools (web_search, arxiv_search, google_scholar, etc.)
- [x] **SPECIALIST_DEFINITIONS stringly-typed dict** — converted to `SpecialistConfig` frozen dataclass
- [x] **Timeout sentinel vs plain degraded string** — `SpecialistResult` dataclass with `timed_out`/`error` flags; synthesis skips timed-out specialists
- [x] **Sync-tool cancellation on timeout** — logged warning on timeout; Python threads can't be forcibly killed, documented as known limitation

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
