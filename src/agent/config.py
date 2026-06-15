"""
Config del orquestador de chat.

Lee el .env del proyecto (a diferencia del MCP server, que recibe el env del
cliente). Variables:
  GEMINI_API_KEY        (requerida) API key de Gemini -- proyecto IndicadoresStockML.
  MCP_READER_LOCAL_DSN  (requerida) DSN del rol mcp_reader (SELECT-only) a la DB local.

No imprime ni loggea los valores (regla de manejo de secretos del proyecto).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modelo por defecto: barato y suficiente (las tools ya devuelven datos
# sintetizados). Cambiable a "gemini-2.5-pro" para mejor razonamiento multi-tool.
DEFAULT_MODEL = "gemini-2.5-flash"

_loaded = False


def _ensure_env() -> None:
    """Carga el .env del proyecto una sola vez (sin pisar variables ya seteadas)."""
    global _loaded
    if not _loaded:
        load_dotenv(REPO_ROOT / ".env", override=False)
        _loaded = True


def get_gemini_api_key() -> str:
    _ensure_env()
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY no esta en el .env del proyecto. "
            "Agregar la linea GEMINI_API_KEY=<valor> y reintentar."
        )
    return key


def get_reader_dsn() -> str:
    _ensure_env()
    dsn = (os.getenv("MCP_READER_LOCAL_DSN") or "").strip()
    if not dsn:
        raise RuntimeError(
            "MCP_READER_LOCAL_DSN no esta en el .env del proyecto. "
            "Es el DSN del rol mcp_reader (SELECT-only) a la DB local."
        )
    return dsn
