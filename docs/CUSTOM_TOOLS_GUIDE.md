# Custom Tool Creation Guide

This guide explains how to create custom tools for the Multi-Tool Research Agent. After reading this, you'll be able to add any new capability to the agent.

## Table of Contents
1. [What is a Tool?](#what-is-a-tool)
2. [Tool Anatomy](#tool-anatomy)
3. [Three Ways to Create Tools](#three-ways-to-create-tools)
4. [Step-by-Step: Creating a New Tool](#step-by-step-creating-a-new-tool)
5. [Best Practices](#best-practices)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## What is a Tool?

A **tool** is a function that the AI agent can call to interact with the outside world. The agent (Claude) can:
- Read text and reason about problems
- BUT cannot browse the web, do precise math, or access APIs on its own

Tools extend the agent's capabilities by giving it access to external functions.

### The ReAct Loop

```
User Question
     ↓
┌─────────────────────────────────────────┐
│  Agent thinks: "I need to search the web" │
│  Agent acts: calls web_search tool        │
│  Agent observes: tool returns results     │
│  Agent thinks: "Now I need to calculate..." │
│  Agent acts: calls calculator tool         │
│  Agent observes: calculation result        │
│  Agent thinks: "I have enough info"        │
│  Agent responds: Final answer              │
└─────────────────────────────────────────┘
```

---

## Tool Anatomy

Every LangChain tool has three parts:

```python
from langchain_core.tools import Tool

my_tool = Tool(
    name="tool_name",           # 1. Name (what agent calls it)
    func=my_function,           # 2. Function (what it does)
    description="..."           # 3. Description (when to use it)
)
```

### 1. Name
- Short, descriptive identifier
- Use snake_case: `web_search`, `calculate`, `get_weather`
- The agent uses this name to call the tool

### 2. Function
- A Python function that takes a string input and returns a string output
- Signature: `def my_function(input: str) -> str`
- Should handle errors gracefully (return error message, don't crash)

### 3. Description
- **CRITICAL**: This is what the agent reads to decide when to use the tool
- Be specific about:
  - What the tool does
  - When to use it (and when NOT to)
  - What input format is expected
  - Example inputs

---

## Three Ways to Create Tools

### Method 1: Tool() Class (Recommended)

This is what we use in this project. Most flexible and explicit.

```python
from langchain_core.tools import Tool

def my_function(query: str) -> str:
    """Do something with the query."""
    # Your logic here
    return "result"

my_tool = Tool(
    name="my_tool",
    func=my_function,
    description="Description of what this tool does..."
)
```

### Method 2: @tool Decorator

Simpler syntax, good for quick tools.

```python
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """This docstring becomes the description."""
    return "result"
```

### Method 3: Subclassing BaseTool

Most control, good for complex tools with multiple methods.

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    query: str = Field(description="The search query")

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Description here..."
    args_schema: type[BaseModel] = MyToolInput

    def _run(self, query: str) -> str:
        # Your logic here
        return "result"
```

---

## Step-by-Step: Creating a New Tool

Let's create a tool that gets stock prices (example).

### Step 1: Create the Tool File

Create `src/tools/stock_tool.py`:

```python
"""Stock price tool for the research agent.

Gets current stock prices using a free API.
"""

import requests
from langchain_core.tools import Tool
from src.utils import retry_on_error


# API endpoint (example - you'd use a real API)
STOCK_API_URL = "https://api.example.com/stock"


@retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL")

    Returns:
        Stock price information or error message.
    """
    try:
        # Clean the input
        symbol = symbol.strip().upper()

        # Make API request
        response = requests.get(
            f"{STOCK_API_URL}/{symbol}",
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            price = data.get("price", "N/A")
            change = data.get("change", "N/A")
            return f"{symbol}: ${price} ({change})"
        else:
            return f"Could not find stock: {symbol}"

    except requests.exceptions.Timeout:
        return "Request timed out. Try again."
    except Exception as e:
        return f"Error getting stock price: {str(e)}"


# Create the tool
stock_tool = Tool(
    name="stock_price",
    func=get_stock_price,
    description=(
        "Get current stock price for a company. "
        "Input should be a stock ticker symbol (e.g., 'AAPL' for Apple, "
        "'GOOGL' for Google, 'MSFT' for Microsoft). "
        "Returns the current price and daily change."
    )
)
```

### Step 2: Add to Agent

Edit `src/agent.py`:

```python
# Add import at top
from src.tools.stock_tool import stock_tool

# Add to tools list in ResearchAgent.__init__
self.tools = [
    calculator_tool,
    wikipedia_tool,
    # ... other tools ...
    stock_tool,  # Add your new tool
]
```

### Step 3: Update Requirements (if needed)

If your tool needs new packages, add them to `requirements.txt`:

```
requests>=2.31.0    # If not already there
```

### Step 4: Test

Run the agent and try a query that should trigger your tool:
```
Your question: What is Apple's current stock price?
```

---

## Best Practices

### 1. Write Excellent Descriptions

The description is how the agent decides when to use your tool. Be specific!

**Bad:**
```python
description="Gets information"
```

**Good:**
```python
description=(
    "Get current stock price for a publicly traded company. "
    "Use this when the user asks about stock prices, share prices, "
    "or market value of companies. "
    "Input should be a stock ticker symbol (e.g., 'AAPL', 'GOOGL'). "
    "Do NOT use for private companies or cryptocurrency."
)
```

### 2. Handle Errors Gracefully

Never let your tool crash. Always return a helpful error message.

```python
def my_tool(query: str) -> str:
    try:
        # Your logic
        return result
    except SpecificError as e:
        return f"Could not process: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
```

### 3. Use the Retry Decorator

For network requests, use the retry decorator from `src/utils.py`:

```python
from src.utils import retry_on_error

@retry_on_error(max_retries=2, delay=1.0, exceptions=(Exception,))
def my_api_call(query: str) -> str:
    # This will retry up to 2 times if it fails
    ...
```

### 4. Validate and Clean Input

Don't trust the input. Clean and validate it.

```python
def my_tool(query: str) -> str:
    # Clean whitespace
    query = query.strip()

    # Check for empty input
    if not query:
        return "Error: Please provide a query."

    # Validate format if needed
    if len(query) > 1000:
        query = query[:1000]  # Truncate

    # Continue with clean input
    ...
```

### 5. Limit Response Size

Don't return huge responses - they fill up the agent's context.

```python
def my_tool(query: str) -> str:
    result = get_large_result()

    # Truncate if too long
    if len(result) > 2000:
        result = result[:2000] + "... (truncated)"

    return result
```

### 6. Add Timeouts

For network requests, always use timeouts:

```python
response = requests.get(url, timeout=10)  # 10 second timeout
```

---

## Common Patterns

### Pattern 1: API Integration

```python
import requests
import os

API_KEY = os.getenv("MY_API_KEY")
API_URL = "https://api.example.com"

def call_api(query: str) -> str:
    if not API_KEY:
        return "Error: API key not configured."

    response = requests.get(
        f"{API_URL}/endpoint",
        params={"q": query, "key": API_KEY},
        timeout=10
    )

    if response.status_code == 200:
        return response.json()["result"]
    else:
        return f"API error: {response.status_code}"
```

### Pattern 2: File Output (like visualization_tool)

```python
import os
from datetime import datetime

OUTPUT_DIR = "output"

def create_file(query: str) -> str:
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Create the file
    with open(filepath, "w") as f:
        f.write("Content here")

    # Return the path so user knows where to find it
    return f"File created: {filepath}"
```

### Pattern 3: JSON Input (like visualization_tool)

For tools that need structured input:

```python
import json

def complex_tool(input_str: str) -> str:
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError:
        return "Error: Invalid JSON. Expected format: {...}"

    # Validate required fields
    if "field1" not in data:
        return "Error: Missing 'field1' in input."

    # Process the structured data
    ...
```

### Pattern 4: Using LangChain Built-in Wrappers

LangChain has many pre-built integrations:

```python
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

# These handle all the API complexity for you
wikipedia = WikipediaAPIWrapper()
result = wikipedia.run("Python programming")
```

---

## Troubleshooting

### Problem: Agent never uses my tool

**Causes:**
1. Description doesn't match what user is asking
2. Another tool's description is a better match
3. Tool name conflicts with existing tool

**Solutions:**
1. Make description more specific with examples
2. Add "Use this when..." and "Do NOT use for..." clauses
3. Ensure unique tool name

### Problem: Tool returns errors

**Debug steps:**
1. Test the function directly in Python first
2. Add print statements to see what input is received
3. Check if API keys are loaded correctly
4. Verify network connectivity

```python
def my_tool(query: str) -> str:
    print(f"DEBUG: Received query: {repr(query)}")  # Debug
    # ... rest of function
```

### Problem: Agent uses tool incorrectly

**Causes:**
1. Description is ambiguous about input format
2. Agent is passing wrong type of data

**Solutions:**
1. Add explicit input format examples in description
2. Add input validation and return helpful error messages

### Problem: Tool is too slow

**Solutions:**
1. Add caching for repeated queries
2. Use shorter timeouts
3. Return partial results if full results take too long
4. Consider the parallel_search tool pattern

---

## Example Tools in This Project

Study these files for reference:

| Tool File | Key Concept |
|-----------|-------------|
| `calculator_tool.py` | Simple function, safe expression evaluation |
| `search_tool.py` | Using LangChain's built-in wrappers |
| `weather_tool.py` | API integration with API key |
| `wolfram_tool.py` | External API, error handling |
| `visualization_tool.py` | JSON input, file output |
| `python_repl_tool.py` | Security considerations, code execution |
| `parallel_tool.py` | Meta-tool, threading, combining tools |
| `arxiv_tool.py` | Academic API, result formatting |

---

## Quick Reference

### Minimal Tool Template

```python
"""Description of what this tool does."""

from langchain_core.tools import Tool

def my_function(query: str) -> str:
    """Process the query and return result."""
    try:
        # Your logic here
        result = f"Processed: {query}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

my_tool = Tool(
    name="my_tool",
    func=my_function,
    description="What this tool does and when to use it."
)
```

### Adding to Agent Checklist

- [ ] Create `src/tools/my_tool.py`
- [ ] Implement function with error handling
- [ ] Create Tool wrapper with good description
- [ ] Import in `src/agent.py`
- [ ] Add to `self.tools` list
- [ ] Add any new packages to `requirements.txt`
- [ ] Update `main.py` banner (optional)
- [ ] Test with relevant queries

---

Happy tool building! 🛠️
