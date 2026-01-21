# Multi-Tool Research Agent

An AI-powered research agent that autonomously uses multiple tools (web search, Wikipedia, calculator, weather, news, URL fetcher) to gather information and answer complex questions. Built with Claude and LangChain using the ReAct (Reasoning + Acting) pattern.

## Features

- **Autonomous Tool Selection**: The agent decides which tools to use based on the question
- **Web Search**: Access current information from the internet via DuckDuckGo
- **Wikipedia Lookup**: Get encyclopedic background information
- **Calculator**: Perform accurate mathematical calculations
- **Weather**: Get real-time weather data for any city
- **News Search**: Find recent news articles on any topic
- **URL Fetcher**: Read and extract content from web pages
- **Conversation Memory**: Remember previous exchanges for follow-up questions
- **Markdown Reports**: Export research results as formatted reports
- **Execution Timing**: See how long each tool takes to execute
- **Retry Logic**: Automatic retries for failed network requests
- **ReAct Pattern**: Shows reasoning steps (Thought → Action → Observation)

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
   ```

   Get your API keys from:
   - Anthropic: [console.anthropic.com](https://console.anthropic.com/)
   - OpenWeatherMap: [openweathermap.org/api](https://openweathermap.org/api) (free tier available)

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

Available tools: web_search, wikipedia, calculator, weather, news_search, fetch_url

Commands:
  Type your question to research
  'clear'  - Clear conversation memory
  'quit'   - Exit the program

Your question: What's the weather in Tokyo and what are some facts about the city?
```

After each answer, you can save the research as a Markdown report.

## Available Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| `web_search` | Search the web for current info | "Tesla stock price 2024" |
| `wikipedia` | Encyclopedic information | "Who was Albert Einstein?" |
| `calculator` | Mathematical calculations | "What is 15% of 850?" |
| `weather` | Current weather data | "Weather in London" |
| `news_search` | Recent news articles | "Latest AI news" |
| `fetch_url` | Read web page content | "Summarize this article: https://..." |

## Example Queries

**Simple queries:**
- "What is 15% of 250?"
- "Who invented the telephone?"
- "What's the weather in New York?"

**Multi-tool queries:**
- "What's the population of Tokyo compared to New York?"
- "What is the GDP of Germany and how does it rank globally?"
- "What's the latest news about artificial intelligence?"

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
├── .env.example                # Example environment file
├── src/
│   ├── agent.py               # Main agent with ReAct pattern & memory
│   ├── callbacks.py           # Timing callback handler
│   ├── report_generator.py    # Markdown report export
│   ├── utils.py               # Utility functions (retry logic)
│   └── tools/
│       ├── calculator_tool.py # Math calculations
│       ├── wikipedia_tool.py  # Wikipedia lookups
│       ├── search_tool.py     # Web search
│       ├── weather_tool.py    # Weather data
│       ├── news_tool.py       # News search
│       └── url_tool.py        # URL content fetcher
└── output/                     # Saved research reports
```

## Technologies

- **LangChain** - Agent orchestration framework
- **Claude (Anthropic)** - LLM for reasoning
- **DuckDuckGo** - Free web and news search
- **Wikipedia API** - Encyclopedia lookups
- **OpenWeatherMap** - Weather data
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
The agent remembers the last 5 exchanges, enabling follow-up questions like "How does that compare to X?" without repeating context.

### Callbacks
LangChain callbacks let us hook into agent events. We use this to track tool execution times.

### Retry Logic
Network-dependent tools (weather, search, URL fetcher) automatically retry on timeout or connection errors.

## License

MIT
