# =============================================================================
#  mcp_server.py — exposes the SoleMate shoe-store via Model Context Protocol
# =============================================================================
#  Any MCP-compatible client (Claude Desktop, Cursor, an ADK agent, etc.) can
#  connect to this process and call the same two tools as the agent file —
#  plus a RESOURCE and a PROMPT so all three MCP primitives are demonstrated.
# -----------------------------------------------------------------------------
#  Install:   pip install "mcp[cli]"
#  Run:       python mcp_server.py            # stdio transport (default)
#  Inspect:   mcp dev mcp_server.py           # opens the MCP inspector UI
# -----------------------------------------------------------------------------
#  Teaching notes:
#    - FastMCP is a lightweight helper that lets you define an MCP server with
#      Python decorators instead of writing the protocol wiring by hand.
#    - @mcp.tool()     — exposes an action the model can call (e.g. search the
#      catalog, look up an order).
#    - @mcp.resource() — exposes read-only reference data at a URI-like address
#      (catalogs, docs, prices, status boards).
#    - @mcp.prompt()   — exposes a reusable prompt template; some clients
#      (Claude Desktop, Cursor) surface it as a slash-command.
#    - 'mcp dev mcp_server.py' is the fastest way to inspect the server, see
#      the registered primitives, and test them from the MCP inspector UI.
# =============================================================================
import json                                              # for the resource that returns JSON
from mcp.server.mcpserver import MCPServer as FastMCP    # MCPServer in mcp 2.x (FastMCP alias)

# Create the server. Name appears in the client's connection log + tool picker.
mcp = FastMCP("solemate-shoestore")


# =============================================================================
#  IN-MEMORY "DATABASE" — runnable with zero setup
# =============================================================================
#  Production: swap CATALOG/ORDERS for SQL/Postgres/etc. — function bodies
#  change, the @mcp.tool() signatures stay the same so clients don't break.
# -----------------------------------------------------------------------------

CATALOG = [
    {"sku": "SM-001", "name": "Air Runner X",  "brand": "Nike",     "size": "9",  "price_paise": 899000,  "stock": 12, "featured": True},
    {"sku": "SM-002", "name": "Cloud Boost",   "brand": "Adidas",   "size": "9",  "price_paise": 749000,  "stock": 5,  "featured": False},
    {"sku": "SM-003", "name": "Court Classic", "brand": "Puma",     "size": "9",  "price_paise": 549000,  "stock": 0,  "featured": False},
    {"sku": "SM-004", "name": "Trail Pro",     "brand": "Nike",     "size": "10", "price_paise": 1099000, "stock": 8,  "featured": True},
    {"sku": "SM-005", "name": "Daily Walker",  "brand": "Skechers", "size": "9",  "price_paise": 399000,  "stock": 22, "featured": False},
    {"sku": "SM-006", "name": "Heritage Low",  "brand": "Reebok",   "size": "8",  "price_paise": 459000,  "stock": 3,  "featured": False},
]

ORDERS = {
    "O-7421": {"status": "in_transit", "carrier": "BlueDart",  "eta_days": 2,
               "items": [{"sku": "SM-001", "qty": 1}]},
    "O-7300": {"status": "delivered",  "carrier": "Delhivery", "eta_days": 0,
               "items": [{"sku": "SM-005", "qty": 2}]},
    "O-7500": {"status": "shipped",    "carrier": "BlueDart",  "eta_days": 4,
               "items": [{"sku": "SM-002", "qty": 1}, {"sku": "SM-003", "qty": 1}]},
}


# =============================================================================
#  PRIMITIVE 1 · TOOL — an action the LLM can take
# =============================================================================
#  Function name + docstring is what the LLM sees when deciding whether to
#  call it — write the docstring in plain English, like instructions for a
#  new hire.
# -----------------------------------------------------------------------------

@mcp.tool()
def search_catalog(
    query: str,
    size: str | None = None,
    in_stock: bool = True,
) -> list[dict]:
    """Search the shoe catalog. Use when the shopper asks for a shoe by
    type, brand, colour or price. Pass 'size' if it was mentioned."""
    q = query.lower()                                    # case-insensitive match
    out = [                                              # plain list-comprehension over the in-memory list
        s for s in CATALOG
        if (q in s["name"].lower() or q in s["brand"].lower())  # match name OR brand
        and (size is None or s["size"] == size)          # respect size filter if provided
        and (not in_stock or s["stock"] > 0)             # in-stock-only by default
    ]
    return out[:10]                                      # cap to 10 results · keeps prompt small


@mcp.tool()
def track_order(order_id: str) -> dict:
    """Get the live status of ONE order. Use when the shopper asks where
    their order is, when it'll arrive, or for a tracking update."""
    order = ORDERS.get(order_id)                         # dict lookup — None if missing
    if order is None:                                    # explicit None check — never trust input
        return {"error": f"order {order_id} not found"}  # the LLM relays this back to the shopper
    return {                                             # plain dict — keep tool returns simple + typed
        "order_id": order_id,
        "status":   order["status"],                     # "shipped" · "in_transit" · "delivered" · "returned"
        "carrier":  order["carrier"],                    # e.g. "BlueDart"
        "eta_days": order["eta_days"],                   # days until delivery
        "items":    order["items"],                      # list of {sku, qty}
    }


# =============================================================================
#  PRIMITIVE 2 · RESOURCE — read-only data the LLM can fetch by URI
# =============================================================================
#  Think of it like an API endpoint, addressable as catalog://featured.
#  Use resources for stable reference data (catalogs, docs, prices, status
#  boards) — NOT for actions with side effects (use a tool for those).
# -----------------------------------------------------------------------------

@mcp.resource("catalog://featured")
def featured_shoes() -> str:
    """The current featured shoes — flagged by the merchandising team.
    Returns a JSON array of {sku, name, brand, price_paise}."""
    rows = [                                             # pick the featured ones · project the public columns
        {"sku": s["sku"], "name": s["name"],
         "brand": s["brand"], "price_paise": s["price_paise"]}
        for s in CATALOG if s["featured"]
    ]
    return json.dumps(rows, indent=2)                    # MCP resources return strings · JSON is fine


# =============================================================================
#  PRIMITIVE 3 · PROMPT — a reusable prompt template
# =============================================================================
#  In Claude Desktop, registered prompts show up as slash-commands. They let
#  you ship "expert intros" that nudge the LLM to use your tools correctly.
# -----------------------------------------------------------------------------

@mcp.prompt()
def recommend_for_outfit(outfit: str) -> str:
    """Recommend 3 shoes that go with the described outfit.
    Triggers as /recommend_for_outfit in Claude Desktop."""
    return (
        f"You are SoleMate's stylist. Recommend 3 shoes that match this outfit: {outfit}. "
        "Use the search_catalog tool first — never invent SKUs, prices or stock. "
        "Reply with one line per shoe: name · brand · ₹price · why it works."
    )


# =============================================================================
#  ENTRY POINT — stdio is the default transport
# =============================================================================
#  Claude Desktop, Cursor, ADK MCP-client — all speak stdio. Switch to
#  transport="sse" if you need to expose the server over HTTP.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")