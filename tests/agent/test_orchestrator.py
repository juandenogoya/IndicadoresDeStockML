"""
Tests del orquestador de chat -- piezas PURAS (sin DB, sin API, sin subproceso MCP).

Cubre:
  - _clean_schema: aplana opcionales (anyOf con null) y descarta claves de adorno.
  - _to_gemini_tools: arma function_declarations a partir de tools MCP simuladas.
  - mcp_bridge._result_payload: convierte el CallToolResult a dict para el LLM.
"""

import json
from types import SimpleNamespace

from src.agent import mcp_bridge
from src.agent.orchestrator import _clean_schema, _to_gemini_tools


def test_clean_schema_aplana_opcional_anyof():
    # Forma tipica de FastMCP para un parametro `str | None`.
    schema = {
        "type": "object",
        "title": "args",
        "properties": {
            "ticker": {"type": "string", "title": "Ticker"},
            "sector": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Sector"},
        },
        "required": ["ticker"],
        "additionalProperties": False,
    }
    out = _clean_schema(schema)

    assert "title" not in out
    assert "additionalProperties" not in out
    # El opcional quedo aplanado al subtipo no-null.
    assert out["properties"]["sector"] == {"type": "string"}
    assert out["properties"]["ticker"] == {"type": "string"}
    assert out["required"] == ["ticker"]


def test_to_gemini_tools_construye_declaraciones():
    fake_tools = [
        SimpleNamespace(
            name="get_price_history",
            description="OHLCV diario.",
            inputSchema={"type": "object", "properties": {"ticker": {"type": "string"}}},
        ),
        SimpleNamespace(
            name="ping",
            description="diagnostico",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]
    tools = _to_gemini_tools(fake_tools)

    assert len(tools) == 1  # un unico Tool agrupa todas las declaraciones
    decls = tools[0].function_declarations
    nombres = {d.name for d in decls}
    assert nombres == {"get_price_history", "ping"}


def test_to_gemini_tools_inputschema_none_no_rompe():
    fake_tools = [SimpleNamespace(name="x", description="", inputSchema=None)]
    tools = _to_gemini_tools(fake_tools)
    assert tools[0].function_declarations[0].name == "x"


def test_result_payload_parsea_json():
    result = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps({"status": "ok", "n": 3}))],
        isError=False,
    )
    payload = mcp_bridge._result_payload(result)
    assert payload == {"status": "ok", "n": 3}


def test_result_payload_texto_no_json_se_envuelve():
    result = SimpleNamespace(content=[SimpleNamespace(text="hola mundo")], isError=False)
    payload = mcp_bridge._result_payload(result)
    assert payload == {"text": "hola mundo"}


def test_result_payload_json_no_dict_se_envuelve():
    # Una lista JSON valida no es dict -> se envuelve en {"result": ...}.
    result = SimpleNamespace(content=[SimpleNamespace(text="[1, 2, 3]")], isError=False)
    payload = mcp_bridge._result_payload(result)
    assert payload == {"result": [1, 2, 3]}
