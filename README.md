# Multi-Tool Research Agent

An AI-powered research agent that autonomously uses multiple tools (web search, Wikipedia, calculator) to gather information and answer complex questions. Built with Claude and LangChain using the ReAct (Reasoning + Acting) pattern.

## Features

- **Autonomous Tool Selection**: The agent decides which tools to use based on the question
- **Web Search**: Access current information from the internet via DuckDuckGo
- **Wikipedia Lookup**: Get encyclopedic background information
- **Calculator**: Perform accurate mathematical calculations
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

4. **Set up API key**

   Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```

   Get your API key from [console.anthropic.com](https://console.anthropic.com/)

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

Available tools: web_search, wikipedia, calculator
Type 'quit' or 'exit' to stop.

Your question: Who was Albert Einstein and how many years ago was he born?
```

## Example Queries

- "What is 15% of 250?"
- "Who invented the telephone and when?"
- "What's the population of Tokyo compared to New York?"
- "What is the GDP of Germany and how does it rank globally?"

## Project Structure

```
multi-tool-research-agent/
├── main.py                 # CLI entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not in repo)
├── src/
│   ├── agent.py           # Main agent with ReAct pattern
│   └── tools/
│       ├── calculator_tool.py
│       ├── wikipedia_tool.py
│       └── search_tool.py
└── output/                 # For saved reports
```

## Technologies

- **LangChain** - Agent orchestration framework
- **Claude (Anthropic)** - LLM for reasoning
- **DuckDuckGo** - Free web search
- **Wikipedia API** - Encyclopedia lookups
- **numexpr** - Safe math evaluation

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | claude-sonnet-4-5 | Claude model to use |
| `TEMPERATURE` | 0.2 | Lower = more focused reasoning |
| `MAX_ITERATIONS` | 10 | Max reasoning steps |
| `VERBOSE` | True | Show reasoning steps |

## License

MIT
