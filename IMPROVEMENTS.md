# Planned Improvements

Prioritized list of improvements to make this project stand out as a portfolio piece for agent engineering roles.

## High Impact

### 1. ~~Tests~~ (DONE)
- [x] 188 tests across all tools, memory, sessions, callbacks, and utilities

### 2. ~~Migrate to Native Tool Calling~~ (DONE)
- [x] Replaced `create_react_agent` (string-based ReAct parsing) with `create_agent` (LangGraph-based)
- [x] Uses Claude's native structured tool-use API instead of fragile text parsing
- [x] Messages-based conversation history instead of string concatenation
- [x] Updated requirements to langchain 1.x + langgraph

### 3. ~~Evaluation Suite~~ (DONE)
- [x] 15 curated test cases across all tool categories (math, search, weather, code, multi-step, ambiguous)
- [x] Two independent scores: tool selection accuracy + answer quality
- [x] CLI runner with `--case` and `--category` filtering, timestamped JSON reports
- [x] First run: 100% tool selection, 93% answer quality

### 4. ~~Streaming Output~~ (DONE)
- [x] `StreamingCallbackHandler` streams thinking/tool-use/answer tokens in real-time
- [x] `stream_query()` method uses LangGraph `stream()` instead of blocking `invoke()`
- [x] Original `query()` preserved for backward compatibility (tests/evals)

### 5. ~~Better Error Recovery~~ (DONE)
- [x] `check_tool_health()` probes API keys and libraries at startup
- [x] `get_available_tools()` filters out disabled tools so the agent never calls them
- [x] System prompt includes explicit fallback mapping and retry guidance
- [x] Startup banner shows tool status (available vs disabled with fallback info)
- [x] Fail-fast on missing `ANTHROPIC_API_KEY` with clear setup instructions

## Medium Impact

### 6. ~~Web UI~~ (DONE)
- [x] Streamlit chat interface with `st.chat_message` / `st.chat_input`
- [x] Real-time streaming feedback (thinking, tool calls) via `st.status` widget
- [x] Sidebar: tool health status, session save/load/clear, query timing
- [x] Run with `streamlit run app.py`

### 7. ~~Observability~~ (DONE)
- [x] `ObservabilityCallbackHandler` captures token usage, tool metrics, and timing per query
- [x] `QueryMetrics` dataclass with cost calculation (model-aware pricing tables)
- [x] `MetricsStore` persists metrics to `observability/metrics.jsonl` (JSONL append-only)
- [x] Streamlit dashboard: token counts, cost, tool calls, performance history with bar chart
- [x] CLI: metrics printed after each query + `stats` command for aggregate stats

### 8. ~~Async / True Parallelism~~ (DONE)
- [x] All I/O replaced with native async: `aiohttp` for HTTP, `asyncio.to_thread()` for blocking libraries
- [x] Agent uses `ainvoke()`/`astream()` (LangGraph's native async API)
- [x] `parallel_tool` uses `asyncio.gather()` instead of ThreadPoolExecutor
- [x] `async_retry_on_error` with `asyncio.sleep()`, `async_run_with_timeout` with `asyncio.wait_for()`
- [x] All 20 tools have `coroutine=` for native async execution
- [x] CLI and Streamlit use `asyncio.run()` entry points

## Nice to Have

### 9. ~~Config Cleanup~~ (DONE)
- [x] Make model selectable (currently hardcoded to claude-sonnet-4-5-20250929)
- [x] Fix banner to list all 17 tools

### 10. Rate Limiting
- [ ] Prevent burning API credits if someone hammers the agent

### 11. Session Schema Versioning
- [ ] Add version field to session JSON for forward compatibility
