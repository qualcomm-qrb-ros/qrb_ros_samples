# Prototype Claude Skills And Program Tools (Plan + Code Execution + Tool Search Tool)

This implementation is based directly on the tutorial: [LLM Agents are simply Graph — Tutorial For Dummies](https://zacharyhuang.substack.com/p/llm-agent-internal-as-a-graph-tutorial).

---
This specific project is inspired by : https://www.anthropic.com/engineering/advanced-tool-use

**One of the critical problem of Context Engineering is solving the massive tooling prompt, and opensource world had lots of ideas about it. the key concept is to avoid loading metadata directly but also finding the right tool while needed.**

**There are also explorations targetting a more efficient way than MCP, such as Claude Skills.**

Refering and inspired by : https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_search_with_embeddings.ipynb

https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/programmatic_tool_calling_ptc.ipynb

---

Reference : https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-agent

## Features (planing)

- Allow high-performance LLM to search available tools with usage 
- Allow high-performance LLM to generate code blocks to use available tools exactly and check tool execution results
- Tool search tool will use text embedding and compute dotproduct to get the best matching tools
- Code execution tool will based on pre-defined tool JSON schema and generate code
- Currently this project will use dummy tool calling

## Example Outputs


## Getting Started

TBD
## How It Works?

TBD