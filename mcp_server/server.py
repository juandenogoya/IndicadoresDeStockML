"""
MCP Server db-consultor -- Entry point.

Uso:
  python -m mcp_server.server            # stdio (default, clientes locales)
  python -m mcp_server.server --transport http --port 8765  # HTTP/SSE (fase 3)
"""

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.calendar import (
    CALENDAR_ANNOTATIONS,
    check_trading_day,
    get_last_trading_day,
)
from mcp_server.tools.exploration import (
    EXPLORATION_ANNOTATIONS,
    describe_table,
    list_tables,
    list_tickers,
)

__version__ = "0.1.0"

mcp = FastMCP(
    name="db-consultor",
    instructions=(
        "Servidor MCP consultivo del proyecto IndicadoresDeStockML. "
        "Lee datos de la DB PostgreSQL local (activos_ml): precios, "
        "indicadores tecnicos, opciones, alertas ML y backtesting. "
        "SOLO LECTURA: no modifica datos ni infraestructura. "
        "Reglas de uso completas en mcp_server/INSTRUCTIONS.md."
    ),
)

# ── Herramienta de diagnostico ────────────────────────────────────────────────

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def ping() -> dict:
    """
    Verifica que el server MCP esta activo y devuelve su version.

    Usar para confirmar conectividad antes de llamar otras tools.
    No requiere conexion a DB.

    Returns:
        dict con status y version. Ej: {"status": "ok", "version": "0.1.0"}
    """
    return {"status": "ok", "version": __version__}


# ── Calendario bursatil NYSE ──────────────────────────────────────────────────

mcp.tool(annotations=CALENDAR_ANNOTATIONS)(check_trading_day)
mcp.tool(annotations=CALENDAR_ANNOTATIONS)(get_last_trading_day)

# ── Exploracion DB ────────────────────────────────────────────────────────────

mcp.tool(annotations=EXPLORATION_ANNOTATIONS)(list_tables)
mcp.tool(annotations=EXPLORATION_ANNOTATIONS)(describe_table)
mcp.tool(annotations=EXPLORATION_ANNOTATIONS)(list_tickers)


if __name__ == "__main__":
    mcp.run()
