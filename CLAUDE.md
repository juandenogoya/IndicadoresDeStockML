# Indicadores y Machine Learning

Este repositorio contiene un sistema de analisis tecnico/ML sobre 199 tickers
con DB PostgreSQL, pipeline diario, scanner ML y backtest historico.

## Documentos de memoria por dominio

Antes de trabajar en un dominio, leer el .md correspondiente:

- memory/MEMORY.md            : indice general, estado del sistema
- memory/AGENDA.md             : tareas activas
- memory/opciones.md           : opciones, PCR, IV, Z-scores
- memory/ml_modelos.md         : Random Forest V3, scoring ML
- memory/datos_pipeline.md     : pipeline diario, cron, 1W
- memory/indicadores_tecnicos.md
- memory/estructuras_velas.md  : SMC, patrones de vela
- memory/scanner_alertas.md
- memory/bt_postgresql_local.md
- memory/mcp_server.md         : servidor MCP consultivo (NUEVO)

## Reglas no negociables

1. Fechas: nunca asumir si una fecha es habil. Usar
   src/utils/trading_calendar.py o scripts/manual/check_fecha.py.
2. Encoding cp1252 en Windows: ASCII puro en strings, no Unicode arrows.
3. Antes de modificar cualquier archivo en src/ o scripts/, leer el .md
   del dominio relevante.
4. El subdirectorio mcp_server/ es SOLO consultivo: no modifica DB, no
   importa funciones con side effects, no toca src/ ni scripts/.
5. Solo consultivo: el MCP nunca modifica DB, scripts, parametros ni
   infraestructura del proyecto.
6. Rol PostgreSQL mcp_reader con SELECT-only + read-only transactions.
7. Validacion SQL con sqlglot en run_select (cuando se implemente en 1F).
8. Catalogo de queries vive afuera del repo, en ~/queries-catalog/.
9. mcp_server/INSTRUCTIONS.md es autocontenido: NO referencia rutas
   externas al repo.

## Antes de codear

Para cualquier tarea, listar archivos que se van a crear/modificar y
pedir aprobacion antes de implementar.

## -------------------------- Actualizaciones 15/05/2026

## Manejo de secretos

NUNCA incluir en respuestas, documentacion, commits, mensajes de error
o cualquier output:
- Passwords, tokens, API keys
- DSN completos con credenciales embebidas
- Contenido literal de archivos .env

Referenciar siempre por nombre de la env var o el archivo:
- Correcto: "el password vive en MCP_READER_LOCAL_DSN del .env"
- Incorrecto: "el password es <valor real>"

Si necesitas mostrar la estructura de un DSN, usar placeholders:
postgresql://<user>:<password>@<host>:<port>/<db>

## Estado del MCP server (subdirectorio mcp_server/)

Diseno completo: docs/mcp_server.md
Reglas de uso del propio server: mcp_server/INSTRUCTIONS.md

Fases completadas:
- Fase 0: skeleton + tool ping
- Fase 1A: calendar tools (check_trading_day, get_last_trading_day)
- Fase 1B: exploration tools (list_tables, describe_table, list_tickers)

Tools registradas hasta hoy (6):
- ping
- check_trading_day
- get_last_trading_day
- list_tables
- describe_table
- list_tickers

Fases pendientes:
- Fase 1C: tools de datos sobre acciones (price_history, technical_indicators,
  price_action, market_structure)
- Fase 1D: tools de opciones (options_summary, options_zscore)
- Fase 1E: tool de alertas (ml_alert_history)
- Fase 1F: composicion (ticker_overview) + run_select con validacion sqlglot
- Fase 2: catalogo de queries (save_query, list_saved, recall_query)
- Fase 3: cliente custom + bot Telegram en Oracle Cloud

## Patrones decididos para el MCP server

1. **Imports desde src/**: el MCP importa funciones puras de src/utils/
   y src/indicators/. Lista cerrada documentada en docs/mcp_server.md.
   NUNCA importar de scripts/, src/pipeline/, src/trading/.

2. **PYTHONPATH es responsabilidad del entorno**: el codigo del MCP
   hace `from src.utils.X import Y` sin sys.path manipulation.
   PYTHONPATH lo setea el cliente (settings.json de Gemini CLI, archivo
   systemd en Oracle Cloud) apuntando al repo root.

3. **DATABASE_URL es responsabilidad del entorno**: el MCP lee la env
   var via pydantic-settings. NO leer .env del proyecto directamente
   desde el codigo del server (el cliente MCP setea el env).

4. **Imports puros, no logica con side effects**: si una funcion de
   src/ mezcla calculo y persistencia (ej. zscore_pipeline que escribe
   DB), NO la importes. Refactorizar primero en src/ antes de consumirla.

5. **Queries parametrizadas con asyncpg ($1, $2, ...)**: nunca f-strings
   con valores del usuario en SQL.

6. **Conversion explicita asyncpg.Record → dict** antes de retornar
   (MCP requiere JSON-serializable).

7. **Tools llevan annotations declarativas**: readOnlyHint=True,
   destructiveHint=False, idempotentHint=True, openWorldHint=False.
   Estas annotations las usan los clientes MCP para decidir si piden
   confirmacion al usuario.

8. **Tests con pytest-asyncio en mode STRICT**: cada test async lleva
   @pytest.mark.asyncio explicito. Tests de integracion marcados con
   @pytest.mark.integration para poder skipearlos sin DB.
