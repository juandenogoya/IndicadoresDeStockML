"""
Cliente MCP (lado cliente) del orquestador.

Lanza el server db-consultor como subproceso stdio -- exactamente como lo hace
Gemini CLI -- pasandole el DSN del rol mcp_reader (SELECT-only). Hereda toda la
seguridad del server: el chat solo puede LEER.

Es LLM-agnostico: expone listar tools y ejecutar una tool. La conversion de
schemas a un proveedor concreto (Gemini) vive en orchestrator.py.
"""

import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agent.config import REPO_ROOT, get_reader_dsn


def _server_params() -> StdioServerParameters:
    """Parametros para lanzar el MCP server como subproceso stdio."""
    env = dict(os.environ)
    env["DATABASE_URL"] = get_reader_dsn()   # rol mcp_reader: SELECT-only
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("GEMINI_API_KEY", None)          # el server no necesita la key del LLM
    return StdioServerParameters(
        command=sys.executable,              # el mismo python del venv
        args=["-m", "mcp_server.server"],
        env=env,
        cwd=str(REPO_ROOT),
    )


@asynccontextmanager
async def open_session():
    """Context manager async: abre una sesion MCP inicializada lista para usar."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools(session) -> list:
    """Devuelve la lista de tools (objetos Tool del SDK con name/description/inputSchema)."""
    resp = await session.list_tools()
    return list(resp.tools)


def _result_to_text(result) -> str:
    """Concatena el contenido textual de un CallToolResult."""
    parts = []
    for item in (result.content or []):
        text = getattr(item, "text", None)
        parts.append(text if text is not None else str(item))
    return "\n".join(parts)


def _result_payload(result) -> dict:
    """
    Convierte un CallToolResult a un dict JSON-serializable para pasar de vuelta
    al LLM como function_response. Si el texto es JSON, lo parsea; si no, lo
    envuelve en {"text": ...}.
    """
    text = _result_to_text(result)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = {"text": text}
    if not isinstance(data, dict):
        data = {"result": data}
    return data


async def call_tool(session, name: str, arguments: dict | None) -> dict:
    """Ejecuta una tool del MCP y devuelve su payload como dict."""
    result = await session.call_tool(name, arguments=arguments or {})
    payload = _result_payload(result)
    if getattr(result, "isError", False):
        return {"error": payload}
    return payload
