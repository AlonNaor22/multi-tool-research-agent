# Multi-Tool Research Agent

An AI-powered research agent that autonomously uses multiple tools (web search, Wikipedia, calculator, weather, news, academic papers, Python code execution, Wolfram Alpha, and more) to gather information and answer complex questions. Built with Claude and LangChain using the ReAct (Reasoning + Acting) pattern.

## Features

### Core Features
- **Autonomous Tool Selection**: The agent decides which tools to use based on the question
- **ReAct Pattern**: Shows reasoning steps (Thought → Action → Observation)
- **Conversation Memory**: Remember all exchanges for follow-up questions
- **Session Persistence**: Save and load research sessions to continue later

### Available Tools (11 Total)

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for current information via DuckDuckGo |
| `wikipedia` | Get encyclopedic background information |
| `calculator` | Perform accurate mathematical calculations |
| `weather` | Get real-time weather data for any city |
| `news_search` | Find recent news articles on any topic |
| `fetch_url` | Read and extract content from web pages |
| `arxiv_search` | Search academic papers on ArXiv |
| `python_repl` | Execute Python code for complex calculations |
| `wolfram_alpha` | Computational knowledge (math, science, facts) |
| `create_chart` | Generate bar, line, and pie charts |
| `parallel_search` | Run multiple searches simultaneously |

### Additional Features
- **Execution Timing**: See how long each tool takes
- **Retry Logic**: Automatic retries for failed network requests
- **Data Visualization**: Create charts and save as PNG images
- **Parallel Execution**: Speed up research with concurrent searches

## How It Works

The agent uses Claude as a "reasoning engine" that follows the ReAct loop:

```
Question: What is the population of Japan as a percentage of world population?

Thought: I need to find Japan's population, then calculate the percentage...
Action: web_search
Action Input: Japan population 2024
Observation: Japan's population is approximately 123.4 million...

Thought: Now I need to calculate the percentage of 8 billion...
Action: calculator
Action Input: (123.4 / 8000) * 100
Observation: 1.54

Thought: I now know the final answer
Final Answer: Japan's population is ~123.4 million, which is about 1.54% of the world's 8 billion people.
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/multi-tool-research-agent.git
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
   OPENWEATHER_API_KEY=your-openweather-key-here
   WOLFRAM_ALPHA_APP_ID=your-wolfram-app-id-here
   ```

   Get your API keys from:
   - Anthropic: [console.anthropic.com](https://console.anthropic.com/)
   - OpenWeatherMap: [openweathermap.org/api](https://openweathermap.org/api) (free tier available)
   - Wolfram Alpha: [developer.wolframalpha.com](https://developer.wolframalpha.com/) (free tier available)

## Usage

Run the CLI:
```bash
python main.py
```

Then ask questions:
```
============================================================
        Multi-Tool Research Agent
        Powered by Claude + LangChain
============================================================

Available tools: web_search, wikipedia, calculator, weather, news_search,
fetch_url, arxiv_search, python_repl, wolfram_alpha, create_chart, parallel_search

Commands:
  Type your question to research
  'clear'    - Clear conversation memory
  'save'     - Save current session
  'load'     - Load a previous session
  'sessions' - List all saved sessions
  'quit'     - Exit the program

Your question: What's the weather in Tokyo and what are some facts about the city?
```

### Session Management

Save your research sessions to continue later:

```
Your question: What is quantum computing?
[...answer...]

Your question: save
Session description (3 words max): quantum computing basics
New session created: sessions/22.01.2026_quantum_computing_basics.json

Your question: quit
```

Later, load the session:
```
Your question: load
Available sessions:
  1. 22.01.2026_quantum_computing_basics (3 messages)

Enter session number or ID: 1
Loaded session: 22.01.2026_quantum_computing_basics
Restored 3 messages.

Your question: tell me more about qubits
[...continues with context from previous session...]
```

## Example Queries

**Simple queries:**
- "What is 15% of 250?"
- "Who invented the telephone?"
- "What's the weather in New York?"

**Academic research:**
- "Find papers about transformer neural networks on ArXiv"
- "What are the latest research papers on quantum computing?"

**Computational queries (Wolfram Alpha):**
- "Solve x^2 + 2x - 8 = 0"
- "What is the integral of x^2?"
- "Convert 100 miles to kilometers"

**Python code execution:**
- "Calculate the factorial of 20 using Python"
- "Generate the first 10 Fibonacci numbers"

**Data visualization:**
- "Create a pie chart showing market share: Apple 40%, Samsung 30%, Others 30%"
- "Make a bar chart of sales by region: North 100, South 150, East 80, West 120"

**Multi-tool queries:**
- "What's the population of Tokyo compared to New York?"
- "Search for Tesla on web, wikipedia, and news at the same time"

**Follow-up questions (uses memory):**
- "What is the population of France?" → "How does that compare to Germany?"
- "Tell me about climate change" → "What are the main causes?"

## Project Structure

```
multi-tool-research-agent/
├── main.py                     # CLI entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not in repo)
├── src/
│   ├── agent.py               # Main agent with ReAct pattern & memory
│   ├── callbacks.py           # Timing callback handler
│   ├── report_generator.py    # Markdown report export
│   ├── session_manager.py     # Session save/load functionality
│   ├── utils.py               # Utility functions (retry logic)
│   └── tools/
│       ├── calculator_tool.py  # Math calculations
│       ├── wikipedia_tool.py   # Wikipedia lookups
│       ├── search_tool.py      # Web search (DuckDuckGo)
│       ├── weather_tool.py     # Weather data
│       ├── news_tool.py        # News search
│       ├── url_tool.py         # URL content fetcher
│       ├── arxiv_tool.py       # Academic paper search
│       ├── python_repl_tool.py # Python code execution
│       ├── wolfram_tool.py     # Wolfram Alpha queries
│       ├── visualization_tool.py # Chart generation
│       └── parallel_tool.py    # Parallel search execution
├── sessions/                   # Saved research sessions
├── output/                     # Generated charts and reports
└── docs/
    └── CUSTOM_TOOLS_GUIDE.md   # Guide for creating new tools
```

## Technologies

- **LangChain** - Agent orchestration framework
- **Claude (Anthropic)** - LLM for reasoning
- **DuckDuckGo** - Free web and news search
- **Wikipedia API** - Encyclopedia lookups
- **OpenWeatherMap** - Weather data
- **ArXiv API** - Academic paper search
- **Wolfram Alpha API** - Computational knowledge
- **matplotlib** - Chart generation
- **BeautifulSoup** - Web page parsing
- **numexpr** - Safe math evaluation

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | claude-sonnet-4-5 | Claude model to use |
| `TEMPERATURE` | 0.2 | Lower = more focused reasoning |
| `MAX_ITERATIONS` | 10 | Max reasoning steps |
| `VERBOSE` | True | Show reasoning steps |

## Key Concepts

### ReAct Pattern
The agent follows a loop of **Re**asoning and **Act**ing:
1. **Thought**: Analyze what information is needed
2. **Action**: Choose and use a tool
3. **Observation**: Process the tool's result
4. Repeat until the answer is complete

### Conversation Memory
The agent remembers all exchanges in a session. For the AI prompt, only the last 5 exchanges are included to avoid context overflow, but all history is saved when you use the `save` command.

### Session Persistence
Sessions are saved as JSON files in the `sessions/` folder. The filename includes the date and a description you provide:
- Format: `dd.mm.yyyy_description_words.json`
- Example: `22.01.2026_quantum_computing_basics.json`

### Parallel Execution
The `parallel_search` tool uses Python's `ThreadPoolExecutor` to run multiple searches simultaneously, significantly speeding up research that requires multiple sources.

### Creating Custom Tools
See `docs/CUSTOM_TOOLS_GUIDE.md` for a comprehensive guide on creating your own tools.

## License

MIT
