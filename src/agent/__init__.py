"""
src/agent -- Orquestador de chat en lenguaje natural sobre el MCP server.

Loop agentico: pregunta del usuario -> LLM (Gemini) con las tools del MCP
db-consultor -> ejecucion de tools (read-only) -> respuesta. Reusable: no
depende de Streamlit. El frontend (dashboard hoy; Telegram en Fase 3) pasa el
mensaje + historial y recibe el texto final.

Modulos:
  config       -- lectura de .env (GEMINI_API_KEY, MCP_READER_LOCAL_DSN), modelo.
  mcp_bridge   -- cliente MCP por stdio (lanza el server, lista/ejecuta tools).
  orchestrator -- loop pregunta->LLM->tools->respuesta (entrypoint answer/answer_sync).
"""
