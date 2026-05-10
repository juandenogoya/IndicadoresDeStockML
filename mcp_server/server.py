"""
MCP Server db-consultor -- Entry point.

Uso:
  python -m mcp_server.server            # stdio (default, clientes locales)
  python -m mcp_server.server --transport http --port 8765  # HTTP/SSE (fase 3)
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool

mcp = FastMCP(
    name="db-consultor",
    version="0.1.0",
    instructions=(
        "Servidor MCP consultivo del proyecto IndicadoresDeStockML. "
        "Lee datos de la DB PostgreSQL local (activos_ml): precios, "
        "indicadores tecnicos, opciones, alertas ML y backtesting. "
        "SOLO LECTURA: no modifica datos ni infraestructura. "
        "Reglas de uso completas en mcp_server/INSTRUCTIONS.md."
    ),
)


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
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    mcp.run()
