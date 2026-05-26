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
from mcp_server.tools.stocks import (
    STOCKS_ANNOTATIONS,
    get_market_structure,
    get_price_action,
    get_price_history,
    get_technical_indicators,
)
from mcp_server.tools.options import (
    OPTIONS_ANNOTATIONS,
    get_options_analysis,
)
from mcp_server.tools.overview import (
    OVERVIEW_ANNOTATIONS,
    get_ticker_overview,
)
from mcp_server.tools.screener import (
    SCREENER_ANNOTATIONS,
    screen_tickers,
)
from mcp_server.tools.alerts import (
    ALERTS_ANNOTATIONS,
    get_ml_alert_history,
)
from mcp_server.tools.sintesis import (
    SINTESIS_ANNOTATIONS,
    get_ticker_sintesis,
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

# ── Datos historicos de acciones ──────────────────────────────────────────────

mcp.tool(annotations=STOCKS_ANNOTATIONS)(get_price_history)
mcp.tool(annotations=STOCKS_ANNOTATIONS)(get_technical_indicators)
mcp.tool(annotations=STOCKS_ANNOTATIONS)(get_price_action)
mcp.tool(annotations=STOCKS_ANNOTATIONS)(get_market_structure)

# ── Analisis de opciones ──────────────────────────────────────────────────────

mcp.tool(annotations=OPTIONS_ANNOTATIONS)(get_options_analysis)

# ── Overview: snapshot completo de un ticker ──────────────────────────────────

mcp.tool(annotations=OVERVIEW_ANNOTATIONS)(get_ticker_overview)

# ── Screener: filtrado multi-criterio sobre 199 tickers ───────────────────────

mcp.tool(annotations=SCREENER_ANNOTATIONS)(screen_tickers)

# ── Historial de alertas ML ────────────────────────────────────────────────────

mcp.tool(annotations=ALERTS_ANNOTATIONS)(get_ml_alert_history)

# ── Sintesis: tecnico D+W x opciones por plazo x sector + reglas ──────────────

mcp.tool(annotations=SINTESIS_ANNOTATIONS)(get_ticker_sintesis)


if __name__ == "__main__":
    mcp.run()
