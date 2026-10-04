# =============================================================================
#  agent_full.py — a complete single-agent example, top to bottom
#  Domain    : SoleMate · online shoe store
#  Brain     : local Ollama · gemma4 (no API key, runs on your laptop)
# =============================================================================
#  Steps:
#    1 · TOOLS          — plain Python functions the agent can call
#    2 · MEMORY         — session state + episodic store
#    3 · GOOGLE ADK     — the agent + the runner (brain = Ollama gemma4)
#    4 · RETRIES        — guard every LLM call
#    5 · ReAct LOOP     — observe -> think -> act (ADK runs it for us)
#    6 · EVALUATION     — golden test cases + LLM-as-judge
# =============================================================================
import os, json, time, sys, re                             # stdlib · os env, json parsing, sleep for retries
import asyncio                                             # needed because ADK session creation is async in current versions
from datetime import datetime, timezone                    # timestamps for persisted memory records
from pathlib import Path                                   # lets the agent locate mcp_server.py beside agent.py
from typing import Any                                     # type hints only — runtime behaviour unchanged

if sys.version_info < (3, 10):
    preferred_python = os.getenv("SOLEMATE_PYTHON", "/Users/saikumar/miniforge3/bin/python")
    if Path(preferred_python).exists() and Path(preferred_python).resolve() != Path(sys.executable).resolve():
        os.execv(preferred_python, [preferred_python, *sys.argv])
    raise RuntimeError(
        "SoleMate requires Python 3.10+. Set SOLEMATE_PYTHON to a compatible Python interpreter."
    )

from pydantic import BaseModel                             # used to define typed tool outputs
from google.genai import types                             # ADK runner expects google.genai Content objects, not raw strings
from google.adk.agents import LlmAgent                     # the agent class — wires LLM + tools + memory
from google.adk.runners import Runner                      # drives one task end-to-end (the ReAct loop)
from google.adk.sessions import InMemorySessionService     # in-memory session state for dev
from google.adk.models.lite_llm import LiteLlm             # adapter that lets ADK talk to non-Google models
from google.adk.tools import FunctionTool, ToolContext     # simple local tools + session context
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams  # current ADK MCP toolset helper
from mcp import StdioServerParameters                      # stdio launch params exported by the installed mcp package


# =============================================================================
#  STEP 1 · TOOLS                                          (what the agent can DO)
# =============================================================================
#  We don't define tools inline — we connect to mcp_server.py over stdio and
#  ADK fetches search_catalog, track_order (and their docstrings) at startup.
#  The exact same tools are reachable from Claude Desktop, Cursor, or any other
#  MCP client — that's the whole point of MCP. Write the tool ONCE, plug it in
#  EVERYWHERE.
#
#  Make sure mcp_server.py sits next to this file (same folder). The dummy
#  CATALOG and ORDERS data live inside mcp_server.py — so this file stays a
#  pure agent file with no business data baked in.
# -----------------------------------------------------------------------------

mcp_toolset = McpToolset(                                  # ADK subprocess-launches the MCP server
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,                        # use the same Python env as this agent
            args=[str(Path(__file__).with_name("mcp_server.py"))],  # resolve the MCP server next to this agent file
        ),
    ),
)

# =============================================================================
#  STEP 2 · MEMORY                                         (what the agent KNOWS)
# =============================================================================
#  Three layers:
#    a) Session state · short-term, this conversation only
#    b) Episodic store · past cases summarised + vector-searchable
#    c) Semantic store · facts keyed by id (user prefs, business rules)
# -----------------------------------------------------------------------------

session_service = InMemorySessionService()                 # ADK's in-memory implementation; swap for Redis later
INITIAL_STATE = {                                          # default session state used when a new shopper session is created
    "shopper_name":    "Asha",                             # semantic fact · cheap context · re-used every turn
    "shoe_size":       "9",                                # preference — agent passes this to search_catalog
    "preferred_brand": "Nike",                             # preference — used to rank results
    "last_order":      None,                               # scratch — set by the first track_order call
}


async def ensure_session(user_id: str, session_id: str):   # works in both CLI and ADK web, which already has an event loop
    session = await session_service.get_session(           # check whether the session already exists
        app_name="shoestore",
        user_id=user_id,
        session_id=session_id,
    )
    if session is None:                                    # create it lazily only on the first turn
        session = await session_service.create_session(
            app_name="shoestore",
            user_id=user_id,
            session_id=session_id,
            state=INITIAL_STATE.copy(),
        )
    return session


MEMORY_FILE = Path(__file__).with_name(".adk") / "memory.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


class JsonMemoryStore:
    """Small persistent memory store. Production: replace with Redis + pgvector."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"semantic": {}, "episodic": {}})

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"semantic": {}, "episodic": {}}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def semantic_for(self, user_id: str) -> dict[str, Any]:
        return self._read().setdefault("semantic", {}).get(user_id, {})

    def remember_semantic(self, user_id: str, key: str, value: str, source: str) -> dict[str, Any]:
        data = self._read()
        facts = data.setdefault("semantic", {}).setdefault(user_id, {})
        facts[key] = {"value": value, "source": source, "updated_at": _now()}
        self._write(data)
        return facts[key]

    def remember_episode(self, user_id: str, summary: str, tags: list[str] | None = None) -> dict[str, Any]:
        data = self._read()
        rows = data.setdefault("episodic", {}).setdefault(user_id, [])
        row = {"summary": summary, "tags": tags or [], "created_at": _now()}
        rows.append(row)
        del rows[:-50]                                     # keep the demo store bounded
        self._write(data)
        return row

    def recall_episodes(self, user_id: str, query: str, k: int = 3) -> list[dict[str, Any]]:
        rows = self._read().setdefault("episodic", {}).get(user_id, [])
        query_terms = _tokens(query)
        scored = []
        for row in rows:
            haystack = " ".join([row.get("summary", ""), " ".join(row.get("tags", []))])
            score = len(query_terms & _tokens(haystack))
            scored.append((score, row))
        scored.sort(key=lambda item: (item[0], item[1].get("created_at", "")), reverse=True)
        return [row for score, row in scored[:k] if score > 0] or rows[-k:]


MEMORY = JsonMemoryStore(MEMORY_FILE)


def _memory_user_id(tool_context: ToolContext | None = None, fallback: str = "shopper-42") -> str:
    return getattr(tool_context, "user_id", None) or fallback


def _state_dict(state: Any) -> dict[str, Any]:
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    value = getattr(state, "_value", None)
    if isinstance(value, dict):
        return dict(value)
    return {}


def recall_memory(query: str, tool_context: ToolContext) -> dict:
    """Recall durable shopper facts and relevant past shopping episodes."""
    user_id = _memory_user_id(tool_context)
    semantic = MEMORY.semantic_for(user_id)
    episodes = MEMORY.recall_episodes(user_id, query)
    return {"user_id": user_id, "semantic": semantic, "episodes": episodes}


def remember_semantic_fact(key: str, value: str, source: str, tool_context: ToolContext) -> dict:
    """Store a durable shopper fact, such as shoe size, brand preference or style preference."""
    user_id = _memory_user_id(tool_context)
    saved = MEMORY.remember_semantic(user_id, key, value, source)
    tool_context.state[key] = value                        # expose the fact immediately in this session
    return {"saved": True, "user_id": user_id, "key": key, "fact": saved}


def remember_episode(summary: str, tags: list[str], tool_context: ToolContext) -> dict:
    """Store a short summary of an important interaction or resolved customer need."""
    user_id = _memory_user_id(tool_context)
    saved = MEMORY.remember_episode(user_id, summary, tags)
    return {"saved": True, "user_id": user_id, "episode": saved}


MEMORY_TOOLS = [
    FunctionTool(recall_memory),
    FunctionTool(remember_semantic_fact),
    FunctionTool(remember_episode),
]

TOOLS = [mcp_toolset, *MEMORY_TOOLS]                       # ADK auto-expands the MCP toolset + local memory tools


# =============================================================================
#  STEP 3 · GOOGLE ADK                                     (the agent + runner)
# =============================================================================
#  LlmAgent      = LLM + tools + memory + system prompt
#  Runner        = drives ONE task: kicks off the agent, runs the ReAct loop,
#                  records every Event (think / act / observe / reply)
#  Brain         = local Ollama running gemma4 — no API key, no rate limit,
#                  great for dev. Swap to Gemini Flash for prod (see Cost section).
# -----------------------------------------------------------------------------

BASE_SYSTEM_INSTRUCTION = """You are SoleMate's shopping assistant. The shopper sees
every reply. Be concise, warm and helpful — one short paragraph max.
When asked about an order, ALWAYS call track_order first — never invent status.
When recommending shoes, call search_catalog with the shopper's size from
session state or semantic memory. Never make up SKUs, prices or stock.

Memory policy:
- Use the injected semantic memory as durable shopper facts.
- Use the injected episodic memory as past interaction context.
- If the shopper states a stable preference, call remember_semantic_fact.
- If the interaction resolves a meaningful need, call remember_episode with a short summary and tags.
- Do not store payment details, passwords, secrets, or sensitive personal data."""


def build_memory_context(context) -> str:
    """Build durable memory context for each ADK CLI/Web model request."""
    user_id = context.user_id or "shopper-42"
    current_text = ""
    if context.user_content and context.user_content.parts:
        current_text = " ".join(part.text or "" for part in context.user_content.parts)
    semantic = MEMORY.semantic_for(user_id)
    episodes = MEMORY.recall_episodes(user_id, current_text, k=3) if current_text else []
    memory_context = {
        "user_id": user_id,
        "session_state": _state_dict(context.state),
        "semantic_memory": semantic,
        "episodic_memory": episodes,
    }
    return "\n\nCurrent memory context:\n" + json.dumps(memory_context, indent=2)


def inject_memory_context(callback_context, llm_request) -> None:
    """Append recalled memory to the system prompt without making instruction callable."""
    llm_request.config.system_instruction = BASE_SYSTEM_INSTRUCTION + build_memory_context(callback_context)


def _decode_tool_response(tool_response: dict) -> Any:
    content = tool_response.get("content") if isinstance(tool_response, dict) else None
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return tool_response


def record_tool_episode(tool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict) -> None:
    """Persist key tool-backed interactions as episodic memory."""
    tool_name = getattr(tool, "name", "")
    user_id = _memory_user_id(tool_context)
    decoded = _decode_tool_response(tool_response)
    if tool_name == "track_order" and isinstance(decoded, dict) and not decoded.get("error"):
        order_id = decoded.get("order_id") or args.get("order_id")
        summary = (
            f"Checked order {order_id}: status {decoded.get('status')}, "
            f"carrier {decoded.get('carrier')}, ETA {decoded.get('eta_days')} days."
        )
        MEMORY.remember_episode(user_id, summary, ["order", str(order_id), str(decoded.get("status"))])
    elif tool_name == "search_catalog" and isinstance(decoded, list):
        query = args.get("query", "")
        size = args.get("size")
        summary = f"Searched catalog for '{query}'"
        if size:
            summary += f" in size {size}"
        summary += f"; returned {len(decoded)} result(s)."
        MEMORY.remember_episode(user_id, summary, ["catalog_search", str(query), str(size or "")])


# Point ADK at your local Ollama daemon. Make sure 'ollama serve' is running
# and 'ollama pull gemma4' has been done once.
GEMMA = LiteLlm(                                           # LiteLLM adapter routes to Ollama under the hood
    model=os.getenv("SOLEMATE_LLM_MODEL", "ollama_chat/gemma4"),  # provider/model — 'ollama_chat' uses chat endpoint
    api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),  # default Ollama port; change if you remapped it
)

agent = LlmAgent(                                          # one agent · could be many for multi-agent
    name="shoestore_assistant",                            # appears in traces · keep snake_case
    model=GEMMA,                                           # ← your local gemma4 is now the brain
    instruction=BASE_SYSTEM_INSTRUCTION,                   # serializable prompt for ADK Web metadata
    tools=TOOLS,                                           # the toolbox from STEP 1
    before_model_callback=inject_memory_context,           # dynamic memory injection per request
    after_tool_callback=record_tool_episode,               # durable episodic memory for business tool calls
    # output_schema=Recommendation,                        # optional typed output (see Structured-Output section)
)

root_agent = agent                                         # ADK web/run loads this module-level name

runner = Runner(                                           # one runner per agent per session
    agent=agent,                                           # the LlmAgent we just defined
    app_name="shoestore",                                  # must match session_service's app_name
    session_service=session_service,                       # so memory + state survive across turns
)


# =============================================================================
#  STEP 4 · RETRIES                                        (the 3-2-1 rule)
# =============================================================================
#  Wrap every LLM call so transient failures (network, malformed JSON, rate
#  limit) don't blow up the whole task.
# -----------------------------------------------------------------------------

def with_retry(fn, *args, tries: int = 3, backoff: float = 1.5, **kwargs):
    """Call fn up to 'tries' times with exponential backoff."""
    last_err: Exception | None = None                      # remember the last error for the giveup log
    for attempt in range(tries):                           # 0, 1, 2  — 3 total attempts
        try:                                               # everything that can fail goes inside
            return fn(*args, **kwargs)                     # success path · return immediately
        except Exception as e:                             # be broad here; LLM failure modes are many
            last_err = e                                   # save for the post-mortem
            time.sleep(backoff ** attempt)                 # 1s, 1.5s, 2.25s — gentle ramp
    raise RuntimeError(                                    # we tried; loudly give up so an alert fires
        f"giving up after {tries} tries; last error: {last_err}"
    )


# =============================================================================
#  STEP 5 · ReAct LOOP                                     (observe · think · act)
# =============================================================================
#  ADK's Runner.run() IS the ReAct loop. We don't write the while-loop ourselves
#  — we just send a user message and read Events until the agent emits a final
#  reply. Each Event is a labelled step: tool_call, tool_response, model_reply.
# -----------------------------------------------------------------------------

async def chat_once(user_msg: str) -> str:                 # one user turn -> one final reply
    final_text: list[str] = []                             # collect text deltas (gemma streams tokens)
    final_seen = False                                     # let ADK finish its async generator cleanly
    await ensure_session("shopper-42", "shopper-42")       # make sure the dev REPL session exists before the runner uses it
    message = types.Content(                               # current ADK runner expects a typed Content object
        role="user",                                       # mark the message as coming from the shopper
        parts=[types.Part(text=user_msg)],                 # wrap the raw string inside one text Part
    )
    try:
        async for event in with_retry(                     # STEP 4's retry wraps the whole turn
            runner.run_async,                              # ADK's async runner
            user_id="shopper-42",                          # match session
            session_id="shopper-42",                       # match session
            new_message=message,                           # what the shopper just typed, in ADK's required format
        ):
            if event.is_final_response() and not final_seen:  # agent decided it's done
                final_seen = True
                if event.content is None or not event.content.parts:
                    return "I couldn't reach the local Ollama model. Make sure Ollama is running and the configured model is installed."
                for part in event.content.parts:           # parts can be text OR tool calls
                    if getattr(part, "text", None) and not getattr(part, "thought", False):  # keep only user-facing text, not internal reasoning
                        final_text.append(part.text)       # accumulate text deltas
            # other events (tool_call, tool_response) auto-recorded for tracing
    except Exception as e:
        return f"I couldn't complete that turn because the local model call failed: {e}"
    return "".join(final_text).strip()                     # one clean string for the UI


# =============================================================================
#  STEP 6 · EVALUATION                                     (golden tests + LLM-judge)
# =============================================================================
#  A real eval run = (1) golden questions, (2) the agent's answer, (3) a judge
#  (LLM or rule) deciding pass/fail. CI fails on regression — that's the only
#  thing that keeps quality stable across prompt changes.
# -----------------------------------------------------------------------------

GOLDEN = [                                                 # 3 cases is enough to start; grow over time
    {                                                      # case 1: must call track_order, not invent
        "user":   "where is my order O-7421?",
        "expect": ["O-7421"],                              # the order id must appear in the reply
    },
    {                                                      # case 2: must respect session size = 9
        "user":   "show me some white running shoes",
        "expect": ["size", "9"],                           # the agent used state["shoe_size"]
    },
    {                                                      # case 3: refuses what it doesn't know
        "user":   "what's the meaning of life?",
        "expect_not": ["42 ", "definitely"],               # must NOT confidently invent off-topic answers
    },
]


def evaluate() -> dict:
    """Run the golden set, return a pass/fail summary."""
    results = []                                           # collect per-case results
    for case in GOLDEN:                                    # one case at a time — keeps logs readable
        reply = asyncio.run(chat_once(case["user"]))       # actual agent output
        ok = (
            all(s in reply for s in case.get("expect", [])) and
            not any(s in reply for s in case.get("expect_not", []))
        )                                                  # both must hold for a pass
        results.append({"user": case["user"], "ok": ok, "reply": reply})
        print(f"{'OK' if ok else 'FAIL'}  {case['user'][:40]}")  # one line per case for CI logs
    passed = sum(1 for r in results if r["ok"])            # quick rollup
    return {"passed": passed, "total": len(results), "results": results}


# =============================================================================
#  ENTRY POINT — pick one
# =============================================================================
if __name__ == "__main__":
    if os.getenv("EVAL"):                                  # set EVAL=1 to run the golden tests
        summary = evaluate()                               # run the golden set
        exit(0 if summary["passed"] == summary["total"] else 1)  # CI-friendly exit code
    else:                                                  # otherwise drop into a REPL
        print("SoleMate assistant · powered by local gemma4 · ctrl-c to quit")
        while True:                                        # one user msg per loop iteration
            try:
                msg = input("you > ").strip()              # blocking · easy for local dev
            except (EOFError, KeyboardInterrupt):
                break
            if not msg: continue                           # ignore empty input
            print("agent >", asyncio.run(chat_once(msg)))  # ReAct loop runs to completion