# Multi-Tool Research Agent

An AI-powered research agent that autonomously selects and uses 20 tools (web search, Wikipedia, academic papers, Reddit, Wikidata, translation, code execution, and more) to gather information and answer complex questions. Built with Claude and LangGraph using native async tool calling.

## Features

### Core Features
- **Native Tool Calling**: Uses Claude's structured tool-use API via LangGraph for reliable tool selection
- **Fully Async**: All I/O uses aiohttp and asyncio for non-blocking execution
- **Conversation Memory**: Remembers all exchanges for follow-up questions
- **Session Persistence**: Save and load research sessions with schema versioning
- **Observability**: Token tracking, cost estimation, tool success rates, and query timing
- **Rate Limiting**: Optional per-session token budget with real-time UI controls
- **Web UI**: Streamlit chat interface with streaming feedback and performance dashboard

### Available Tools (20 Total)

| Category | Tool | Description |
|----------|------|-------------|
| **Math & Computation** | `calculator` | Arithmetic, algebra, math functions with variables |
| | `unit_converter` | Convert between measurement units (8 categories) |
| | `equation_solver` | Symbolic math via SymPy |
| | `currency_converter` | Real-time exchange rates (Frankfurter API) |
| | `wolfram_alpha` | Computational knowledge engine |
| **Information Retrieval** | `web_search` | Web search via DuckDuckGo |
| | `wikipedia` | Encyclopedic content |
| | `news_search` | Recent news articles |
| | `arxiv_search` | STEM/AI/ML academic preprints |
| | `youtube_search` | Video search via yt-dlp |
| | `google_scholar` | Academic papers via Semantic Scholar API |
| **Web Content** | `fetch_url` | Extract content from web pages |
| | `pdf_reader` | PDF text extraction via pdfplumber |
| **Social & Discussion** | `reddit_search` | Reddit posts and discussions |
| **Knowledge Base** | `wikidata` | Structured facts via SPARQL |
| **Translation** | `translate` | 100+ languages via Google Translate |
| **Code Execution** | `python_repl` | Sandboxed Python execution |
| **Visualization** | `create_chart` | Bar, line, and pie charts |
| **Multi-Source** | `parallel_search` | Run multiple searches concurrently with asyncio.gather |
| **Weather** | `weather` | Current conditions and forecasts |

### Engineering Highlights
- **Custom tool implementations** over LangChain community built-ins — adds async retry with rate-limit detection, timeout protection, structured output formatting, JSON input parsing with advanced options (category filtering, region, sorting), and help commands that the built-in tools lack
- **283 tests** with pytest-asyncio
- **Evaluation suite** with tool selection accuracy and answer quality scoring
- **TTL caching** on search tools to reduce redundant API calls
- **Tool health checks** at startup with automatic fallback guidance
- **Rate-limit-aware retries** (detects HTTP 429, backs off aggressively)
- **Cost tracking** with model-aware pricing tables

## How It Works

The agent uses Claude as a reasoning engine with LangGraph's native tool-calling loop:

1. **Analyze** the question and identify which category of tool is needed
2. **Select** the most appropriate tool based on system prompt guidance
3. **Execute** the tool call asynchronously (via `ainvoke`)
4. **Synthesize** results from multiple tool calls into a coherent answer

The agent's system prompt includes hierarchical tool categories with guidance on when to use each tool, error recovery strategies, and fallback options.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlonNaor22/multi-tool-research-agent.git
   cd multi-tool-research-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**

   Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   OPENWEATHER_API_KEY=your-openweather-key-here      # Optional (weather tool)
   WOLFRAM_ALPHA_APP_ID=your-wolfram-app-id-here       # Optional (wolfram tool)
   ```

   Required: [Anthropic](https://console.anthropic.com/)
   Optional: [OpenWeatherMap](https://openweathermap.org/api), [Wolfram Alpha](https://developer.wolframalpha.com/)

## Usage

### CLI
```bash
python main.py
```

Commands: `clear`, `save`, `load`, `sessions`, `stats`, `quit`

### Web UI (Streamlit)
```bash
streamlit run app.py
```

Features: chat interface, streaming feedback, tool health status, query metrics, performance history, rate limiting controls, session management.

## Example Queries

- "What is 15% of 250?" (calculator)
- "Find papers about transformers on ArXiv" (arxiv_search)
- "Translate 'hello world' to japanese" (translate)
- "What do people on Reddit think about Python vs Rust?" (reddit_search)
- "What's the population of Tokyo?" (wikidata)
- "Search for Tesla on web, wikipedia, and news at the same time" (parallel_search)
- "What is the population of France?" then "How does that compare to Germany?" (follow-up with memory)

## Project Structure

```
multi-tool-research-agent/
├── main.py                        # Async CLI entry point
├── app.py                         # Streamlit web UI
├── config.py                      # Configuration (model, API keys, limits)
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Test configuration
├── IMPROVEMENTS.md                # Improvement roadmap (all items done)
├── src/
│   ├── agent.py                   # ResearchAgent with async ainvoke/astream
│   ├── callbacks.py               # Timing and streaming callback handlers
│   ├── observability.py           # Token tracking, cost estimation, metrics store
│   ├── rate_limiter.py            # Optional per-session token budget
│   ├── session_manager.py         # Session save/load with schema versioning
│   ├── tool_health.py             # Startup health checks and fallback guidance
│   ├── constants.py               # Shared constants (timeouts, URLs, limits)
│   ├── utils.py                   # Async retry, timeout, aiohttp session, TTL cache
│   └── tools/                     # 20 tools (all async with aiohttp/asyncio)
│       ├── calculator_tool.py
│       ├── unit_converter_tool.py
│       ├── equation_solver_tool.py
│       ├── currency_tool.py
│       ├── wolfram_tool.py
│       ├── wikipedia_tool.py
│       ├── search_tool.py
│       ├── news_tool.py
│       ├── arxiv_tool.py
│       ├── youtube_tool.py
│       ├── google_scholar_tool.py
│       ├── url_tool.py
│       ├── pdf_tool.py
│       ├── reddit_tool.py
│       ├── wikidata_tool.py
│       ├── translation_tool.py
│       ├── python_repl_tool.py
│       ├── visualization_tool.py
│       ├── parallel_tool.py
│       └── weather_tool.py
├── tests/                         # 283 async tests
├── evals/                         # Evaluation framework
├── sessions/                      # Saved research sessions
└── observability/                 # Query metrics (JSONL)
```

## Technologies

- **LangGraph** / **LangChain** - Async agent orchestration with native tool calling
- **Claude (Anthropic)** - LLM for reasoning and tool selection
- **aiohttp** - Async HTTP client (replaces requests)
- **DuckDuckGo** - Free web and news search
- **Semantic Scholar API** - Academic paper search across all fields
- **yt-dlp** - YouTube video search
- **pdfplumber** - High-quality PDF text extraction
- **deep-translator** - Translation (100+ languages)
- **Wikidata SPARQL** - Structured knowledge queries
- **Streamlit** - Web UI with chat interface
- **pytest-asyncio** - Async test framework

## Configuration

Edit `config.py` or set environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | claude-sonnet-4-5-20250929 | Claude model to use |
| `TEMPERATURE` | 0.2 | Lower = more focused reasoning |
| `MAX_TOKENS` | 2000 | Maximum response length |
| `MAX_ITERATIONS` | 10 | Max reasoning steps per query |

## License

MIT
