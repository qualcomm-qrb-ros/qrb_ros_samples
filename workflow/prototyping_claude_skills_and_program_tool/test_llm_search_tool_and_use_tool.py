#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# VIBE CODING WARNING:
# This is a minimal prototype for controlled/educational use.
# It is NOT a complete security boundary. Do not run untrusted code in production.

# Reference Article : https://www.anthropic.com/engineering/advanced-tool-use
# Reference Notebook : https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_search_with_embeddings.ipynb

import yaml

# Line #70 of Notebook
import numpy as np
# Define our tool library with 2 domains
TOOL_LIBRARY = [
    # Weather Tools
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for multiple days ahead",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state",
                },
                "days": {
                    "type": "number",
                    "description": "Number of days to forecast (1-10)",
                },
            },
            "required": ["location", "days"],
        },
    },
    {
        "name": "get_timezone",
        "description": "Get the current timezone and time for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or timezone identifier",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_air_quality",
        "description": "Get current air quality index and pollutant levels for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates",
                }
            },
            "required": ["location"],
        },
    },
    # Finance Tools
    {
        "name": "get_stock_price",
        "description": "Get the stock price for a given ticker symbol at a specified timestamp (UTC+8) .",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, GOOGL)",
                },
                "timestamp": {
                    "type": "string",
                    "description": "to locate the ticket price with exact timestamp from NASDAQ database",
                },
            },
            "required": ["ticker", "timestamp"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount to convert",
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., USD)",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., EUR)",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate_compound_interest",
        "description": "Calculate compound interest for investments over time",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "Initial investment amount",
                },
                "rate": {
                    "type": "number",
                    "description": "Annual interest rate (as percentage)",
                },
                "years": {"type": "number", "description": "Number of years"},
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "monthly", "quarterly", "annually"],
                    "description": "Compounding frequency",
                },
            },
            "required": ["principal", "rate", "years"],
        },
    },
    {
        "name": "get_market_news",
        "description": "Get recent financial news and market updates for a specific company or sector",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Company name, ticker symbol, or sector",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of news articles to return",
                },
            },
            "required": ["query"],
        },
    },
]

print(f"✓ Defined {len(TOOL_LIBRARY)} tools in the library")

# Line #71 of Notebook
def tool_to_text(tool) -> str:
    """
    Convert a tool definition into a text representation for embedding.
    Combines the tool name, description, and parameter information.
    """
    text_parts = [
        f"Tool: {tool['name']}",
        f"Description: {tool['description']}",
    ]

    # Add parameter information
    if "input_schema" in tool and "properties" in tool["input_schema"]:
        params = tool["input_schema"]["properties"]
        param_descriptions = []
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "")
            param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")

        if param_descriptions:
            text_parts.append("Parameters: " + ", ".join(param_descriptions))

    return "\n".join(text_parts)


# Test with one tool
# sample_text = tool_to_text(TOOL_LIBRARY[0])
# print("Sample tool text representation:")
# print(sample_text)

def create_all_tools_embedding():
    """
    Create tools text embedding in TOOL_LIBRARY

    Args: 
        None

    Returns:
        List of tool embedding numpy array
    """
    # Create embeddings for all tools
    print("Creating embeddings for all tools...")

    tool_texts = [tool_to_text(tool) for tool in TOOL_LIBRARY]

    # my tool cannot return np array , so I need to do my self
    # this method may not be very efficient , but it is easy to understand
    # finally you will have a similarities list, and you can sort with index
    tool_embeddings = []
    temp_obj = emb.embeddings(tool_texts)
    for item in temp_obj.data:
        t_array = np.array(item.embedding)
        tool_embeddings.append(t_array)
        
    return tool_embeddings

tool_embeddings = create_all_tools_embedding()

# Refer from 
# Line #73 of : https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_search_with_embeddings.ipynb
def search_tools(query: str, top_k: int = 5) -> list[dict]:
    """
    Search for tools using semantic similarity.

    Args:
        query: Natural language description of what tool is needed
        top_k: Number of top tools to return

    Returns:
        List of tool definitions most relevant to the query
    """
    query_embedding = emb.embeddings(query)
    q_array = np.array(query_embedding.data[0].embedding)
    similarities = []
    for index, value in enumerate(tool_embeddings):
        tmp = np.dot(value, q_array)
        similarities.append(tmp)
        
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({"tool": TOOL_LIBRARY[idx], "similarity_score": float(similarities[idx])})

    return results

# Debug Only
# test_query = "I need to check the weather"
# test_results = search_tools(test_query, top_k=3)
# 
# print(f"Search query: '{test_query}'\n")
# print("Top 3 matching tools:")
# for i, result in enumerate(test_results, 1):
#     tool_name = result["tool"]["name"]
#     score = result["similarity_score"]
#     print(f"{i}. {tool_name} (similarity: {score:.3f})")

# The tool_search tool definition
TOOL_SEARCH_DEFINITION = {
    "name": "tool_search",
    "description": "Search for available tools that can help with a task. Returns tool definitions for matching tools. Use this when you need a tool but don't have it available yet.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what kind of tool you need (e.g., 'weather information', 'currency conversion', 'stock prices')",
            },
            "top_k": {
                "type": "number",
                "description": "Number of tools to return (default: 5)",
            },
        },
        "required": ["query"],
    },
}

print("✓ Tool search definition created")


# in Line #75 of https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_search_with_embeddings.ipynb
# there is no input schema for tool search, adding it for my local experiment
def handle_tool_search(query: str, top_k: int = 5) -> list[dict[str, any]]:
    """
    Handle a tool_search invocation and return tool references.

    Returns a list of tool_reference content blocks for discovered tools.
    """
    # Search for relevant tools
    results = search_tools(query, top_k=top_k)

    # Create tool_reference objects instead of full definitions
    tool_references = [
        {"type": "tool_reference", "tool_name": result["tool"]["name"], "input_schema": result["tool"]["input_schema"]} for result in results
    ]

    # the print is only for debug , not changing tool_references contents
    print(f"\n🔍 Tool search: '{query}'")
    print(f"   Found {len(tool_references)} tools:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['tool']['name']} (similarity: {result['similarity_score']:.3f})")

    return tool_references


# Test the handler
# test_result = handle_tool_search("stock market data", top_k=3)
# print(f"\nReturned {len(test_result)} tool references:")
# for ref in test_result:
#     print(f"  {ref}")


# question = "I need to check latest NVDA price, and last 10 days of NVDA price. Also compare with QCOM price and MSFT price with a svg gram."
question = "I need to check latest NVDA price"

# Now is the time to combine with PocketFlow
prompt = f"""
#### PROMPT START
#### CONTEXT
You are an assistant node that process tool search and feedback to orchestration node
Question: {question}
####
:::: ACTION SPACE
[1] tool_search
{TOOL_SEARCH_DEFINITION}

#### NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: tool_search
reason: <why you chose this action>
query: <your_query_string>
top_k: <your_top_k>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character

#### END OF PROMPT
"""

response = call_llm("", prompt)
yaml_str = response.split("```yaml")[1].split("```")[0].strip()
tool_input = yaml.safe_load(yaml_str)


# Now process tool_search 
query = tool_input["query"]
top_k = tool_input.get("top_k", 5)
tool_search_res = []
# Get tool references
tool_references = handle_tool_search(query, top_k)

# Create tool result with tool_reference content blocks
tool_search_res.append(
    {
        "type": "tool_search_result",
        "content": tool_references,
    }
)

tool_exec_res = "no tools called"

prompt = f"""
#### PROMPT START
#### CONTEXT
You are an assistant.
You can use direct tool calling.
You can also use code execution tool, and code execution tool can arrange call tool based on python language.
Your target language is python.
You need carefully evalute task complexity, and if it needs complex tool calling cycles, you need to consider use code execution first.
Question: 
{question}
Tool Search Results: 
{tool_search_res}
Tool Calling Results: 
{tool_exec_res}

#### ACTION SPACE
[1] code_exec
  Description: This tool will run python code clock in sandbox and only return execution results. Result string will be taken into your next inference in "Tool Use Results" part.
  Parameters:
    - file_name (str): the file name to store your code, you should name your file intuitively, so LLM could read the file name and understood what is the context inside.
    - code_block (str): the complete code block that can run in a single python file.
    - markdown_block (str): describe how your code_block could be reused in python language.

[2] answer
  Description: Answer the question based on code execution tool
  Parameters:
    - answer (str): Final answer to the question

[3] tool_use
  Description: based on tool search result, you can choose which tool to use to answer user queries
  Parameters:
    - None

#### NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: code_exec OR answer
reason: <why you chose this action>
answer: <if action is answer>
file_name: <your_file_name if action is code_exec>
code_block: <your_code if action is code_exec>
tool_name: <your_chosen_tool_name if action is tool_use>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character

#### END OF PROMPT
"""

response = call_llm("", prompt)
yaml_str = response.split("```yaml")[1].split("```")[0].strip()
act_dict = yaml.safe_load(yaml_str)
print(act_dict["code_block"])