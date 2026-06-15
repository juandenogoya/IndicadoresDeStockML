"""
Orquestador del chat: loop pregunta -> Gemini (con las tools del MCP) -> respuesta.

Reusable y sin dependencia de Streamlit. El frontend pasa el mensaje del usuario
+ el historial de la conversacion (lista de Content de google-genai) y recibe el
texto final + el historial actualizado + las tools usadas.

Flujo de una llamada (answer):
  1. Abre una sesion MCP (subproceso stdio, rol mcp_reader read-only).
  2. Lista las tools del server y las convierte a function_declarations de Gemini.
  3. Loop: Gemini responde -> si pide tool calls, se ejecutan via MCP y se le
     devuelven los resultados -> repite hasta que responde texto (o se agota el
     presupuesto de rounds).
  4. Devuelve el texto final.
"""

import asyncio
import datetime as _dt
import sys

from google import genai
from google.genai import types

from src.agent import mcp_bridge
from src.agent.config import DEFAULT_MODEL, REPO_ROOT, get_gemini_api_key

MAX_TOOL_ROUNDS = 8
MAX_OUTPUT_TOKENS = 1024   # techo de salida (eficiencia: key pospago)
THINKING_BUDGET = 0        # desactiva el "thinking" de gemini-2.5-flash (ahorra tokens)

# System prompt CONDENSADO (destilacion de mcp_server/INSTRUCTIONS.md a lo
# esencial). Se manda en cada ronda, por eso es corto: menos tokens de entrada.
# La fuente de verdad completa de las reglas sigue siendo INSTRUCTIONS.md (la
# usan los clientes MCP como Gemini CLI); aca va solo lo critico para el chat.
_REGLAS = (
    "Sos un asistente consultivo de SOLO LECTURA sobre la DB local del proyecto "
    "IndicadoresDeStockML (199 tickers de acciones US). Respondes preguntas con "
    "datos REALES obtenidos via las tools del MCP db-consultor.\n\n"
    "REGLAS:\n"
    "1. FECHAS: para una fecha relativa (hoy, ayer, hace N dias) resolvela con las "
    "tools de calendario. Pero para 'ultimo cierre/dato disponible' NO uses el "
    "calendario: llama directo a la tool de datos (get_ticker_overview / "
    "get_price_history / etc.) y reporta la fecha de la fila mas reciente que devuelva.\n"
    "2. Universo CERRADO de 199 tickers. Si preguntan por uno fuera, decilo; no inventes.\n"
    "3. NO inventes numeros: si una tool no trae el dato, decilo.\n"
    "4. Score: 'score'/'alerta' = ML (alertas_scanner.alert_score, 0-100). El "
    "rule-based (scoring_tecnico) esta vacio en local.\n"
    "5. PCR volumen: <0.7 ALCISTA, 0.7-1.0 NEUTRO, >1.0 BAJISTA. El OI es del cierre previo (T-1).\n"
    "6. Fundamentales: get_fundamentals ya devuelve los % como porcentaje; en perfil "
    "'financiero' margenes/ROIC/liquidez vienen NULL por diseno (no es falta de dato).\n"
    "7. DATA MANUAL (Plan C): la DB local puede ir 1-2 dias habiles detras del "
    "calendario. La tool de datos SIEMPRE devuelve la fila mas reciente disponible "
    "(p.ej. del viernes aunque hoy sea lunes). RESPONDE con esa fila e informa SU "
    "fecha. NUNCA digas 'no hay datos' por falta del dato de hoy: el dato existe en "
    "una fecha anterior y es la respuesta valida.\n"
    "8. Describis los datos; no das recomendaciones de inversion.\n\n"
    "ESTILO (importante): responde en espanol, BREVE y al grano. Pocas frases o una "
    "tabla compacta. Sin preambulos, sin repetir la pregunta, sin disclaimers largos."
)


def _system_prompt() -> str:
    return f"{_REGLAS}\n\nHoy es {_dt.date.today().isoformat()}."


def _prune_history(contents) -> list:
    """
    Deja el historial liviano para el proximo turno: conserva solo las partes de
    TEXTO (la conversacion) y descarta function_call/function_response (el JSON
    crudo de las tools), que solo hacen falta dentro del turno que las genero.
    Ahorra muchos tokens de entrada en conversaciones largas.
    """
    pruned = []
    for c in contents:
        text_parts = [p for p in (c.parts or []) if getattr(p, "text", None)]
        if text_parts:
            pruned.append(types.Content(role=c.role, parts=text_parts))
    return pruned


def _clean_schema(node):
    """
    Adapta un JSON Schema de FastMCP a lo que Gemini acepta sin friccion:
    aplana 'anyOf: [T, null]' (opcionales) al subtipo no-null y descarta claves
    que el motor de function-calling no usa (title, default, additionalProperties).
    """
    if isinstance(node, list):
        return [_clean_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, val in node.items():
        if key in ("title", "$schema", "additionalProperties", "default"):
            continue
        if key == "anyOf" and isinstance(val, list):
            variants = [s for s in val if isinstance(s, dict) and s.get("type") != "null"]
            chosen = variants[0] if variants else (val[0] if val else {})
            out.update(_clean_schema(chosen))
            continue
        out[key] = _clean_schema(val)
    return out


def _short_desc(desc: str, limit: int = 320) -> str:
    """
    Recorta la descripcion de una tool a su primer parrafo (resumen) para no
    mandar el docstring completo en cada ronda. El nombre + resumen + el schema
    de parametros alcanzan para que el modelo elija bien y se ahorran tokens.
    """
    desc = (desc or "").strip()
    if not desc:
        return ""
    para = desc.split("\n\n", 1)[0].strip().replace("\n", " ")
    if len(para) <= limit:
        return para
    cut = para[:limit]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot > 80 else cut


def _to_gemini_tools(mcp_tools) -> list:
    """Convierte las tools del MCP en una lista [Tool] con function_declarations."""
    decls = []
    for tool in mcp_tools:
        schema = _clean_schema(tool.inputSchema or {"type": "object", "properties": {}})
        decls.append(
            types.FunctionDeclaration(
                name=tool.name,
                description=_short_desc(tool.description),
                parameters_json_schema=schema,
            )
        )
    return [types.Tool(function_declarations=decls)]


def _final_text(resp) -> str:
    """Extrae el texto final de la respuesta de Gemini, de forma defensiva."""
    try:
        if resp.text:
            return resp.text
    except (ValueError, AttributeError):
        pass
    out = []
    for cand in (resp.candidates or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "text", None):
                out.append(part.text)
    return "\n".join(out) if out else "(El modelo no devolvio texto.)"


async def answer(user_message: str, contents=None, model: str | None = None) -> dict:
    """
    Procesa un mensaje del usuario y devuelve la respuesta del asistente.

    Args:
        user_message: texto del usuario.
        contents: historial previo (lista de types.Content) o None.
        model: id de modelo Gemini; default DEFAULT_MODEL.

    Returns:
        dict con keys: text, contents (historial actualizado), tools_used, error.
    """
    model = model or DEFAULT_MODEL
    contents = list(contents or [])
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    client = genai.Client(api_key=get_gemini_api_key())
    tools_used: list[str] = []
    tokens_in = tokens_out = tokens_total = 0
    n_rondas = 0
    resp = None

    async with mcp_bridge.open_session() as session:
        mcp_tools = await mcp_bridge.list_tools(session)
        cfg = types.GenerateContentConfig(
            system_instruction=_system_prompt(),
            tools=_to_gemini_tools(mcp_tools),
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        for _ in range(MAX_TOOL_ROUNDS):
            resp = await client.aio.models.generate_content(
                model=model, contents=contents, config=cfg
            )
            n_rondas += 1
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                tokens_in += getattr(um, "prompt_token_count", 0) or 0
                tokens_out += (getattr(um, "candidates_token_count", 0) or 0) \
                    + (getattr(um, "thoughts_token_count", 0) or 0)
                tokens_total += getattr(um, "total_token_count", 0) or 0
            if not resp.candidates:
                break
            cand = resp.candidates[0]
            parts = (getattr(cand.content, "parts", None) or [])
            fcalls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            contents.append(cand.content)  # registrar el turno del modelo
            if not fcalls:
                break
            tool_parts = []
            for fc in fcalls:
                args = dict(fc.args) if fc.args else {}
                tools_used.append(fc.name)
                payload = await mcp_bridge.call_tool(session, fc.name, args)
                tool_parts.append(types.Part.from_function_response(name=fc.name, response=payload))
            contents.append(types.Content(role="user", parts=tool_parts))

    return {
        "text": _final_text(resp) if resp is not None else "(Sin respuesta.)",
        "contents": _prune_history(contents),
        "tools_used": tools_used,
        "tokens": tokens_total,        # compat: total
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "n_rondas": n_rondas,
        "model": model,
        "error": None,
    }


def answer_sync(user_message: str, contents=None, model: str | None = None) -> dict:
    """
    Wrapper sincrono para frontends sync (Streamlit). Atrapa errores.

    En Windows usa ProactorEventLoop (necesario para lanzar el subproceso MCP por
    stdio) sin tocar la policy global del proceso.
    """
    try:
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(answer(user_message, contents=contents, model=model))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as exc:  # noqa: BLE001 -- el frontend muestra el error al usuario
        return {
            "text": f"Hubo un error procesando la consulta: {exc}",
            "contents": list(contents or []),
            "tools_used": [],
            "tokens": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "n_rondas": 0,
            "model": model or DEFAULT_MODEL,
            "error": str(exc),
        }
