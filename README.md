# SoleMate — Single-Agent Assistant with MCP Server & Memory

A complete, end-to-end AI Agent application built using **Google Agent Development Kit (ADK)**, **Model Context Protocol (MCP)**, and local LLMs via **Ollama (`gemma4`)**.

This project demonstrates how to build a production-grade single agent with tool calling over MCP, persistent memory (semantic + episodic), retries, session management, and automated evaluation.

---

## 🌟 Key Features

- **Model Context Protocol (MCP) Server**: Exposes tools (`search_catalog`, `track_order`), resources (`catalog://featured`), and prompt templates over stdio.
- **Google ADK Integration**: Uses `google.adk` for agent definition, ReAct loop execution, runner session context, and tool binding.
- **Local Brain (Zero API Cost)**: Runs locally using Ollama and the `gemma4` model via `LiteLlm`.
- **Multi-Layer Memory System**:
  - **Session State**: Short-term, turn-by-turn memory (`InMemorySessionService`).
  - **Semantic Store**: Durable shopper preferences (e.g., shoe size, brand preference).
  - **Episodic Store**: Searchable summary of past interactions and orders.
- **Resilient Execution**: 3-2-1 exponential backoff retry wrapper for LLM calls.
- **Automated Golden Evals**: Built-in test suite to evaluate agent correctness against benchmark questions.

---

## 📁 Repository Structure

```
single_agent/
├── agent.py          # Google ADK agent definition, memory hooks, ReAct runner, and eval suite
├── mcp_server.py      # MCP server providing tools, resources, and prompt templates
├── README.md         # Documentation & setup guide
└── .adk/
    └── memory.json   # Persistent JSON memory store (semantic & episodic)
```

---

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **Ollama**: Installed and running locally.
   - Install the model:
     ```bash
     ollama pull gemma4
     ```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install google-adk "google-adk[extensions]" mcp litellm pydantic httpx
```

### 2. Verify Ollama Service

Ensure Ollama is active on port `11434`:

```bash
ollama list
```

### 3. Run the Agent (Interactive REPL)

Launch the agent CLI in interactive chat mode:

```bash
python agent.py
```

*Example interaction:*
```
SoleMate assistant · powered by local gemma4 · ctrl-c to quit
you > where is my order O-7421?
agent > I've checked on your order O-7421! It is currently in transit with BlueDart and is estimated to arrive in 2 days.
```

---

## 🧪 Running Evaluations

To run the automated golden eval benchmark suite:

### Windows (PowerShell)
```powershell
$env:EVAL="1"; python agent.py
```

### macOS / Linux
```bash
EVAL=1 python agent.py
```

---

## 🛠️ MCP Tools & Primitives

The MCP server ([`mcp_server.py`](file:///c:/Users/k16na/Desktop/single_agent/mcp_server.py)) exposes:

| Type | Name / URI | Description |
| :--- | :--- | :--- |
| **Tool** | `search_catalog` | Searches catalog by query, size, and stock availability. |
| **Tool** | `track_order` | Fetches live order status, carrier, ETA, and items. |
| **Resource** | `catalog://featured` | Returns featured catalog items in JSON format. |
| **Prompt** | `recommend_for_outfit` | Reusable outfit recommendation prompt. |

Inspect the MCP server interactively with the MCP CLI:

```bash
mcp dev mcp_server.py
```

---

## 📜 License

MIT License. Free to use and modify for learning and production applications.
