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

## Antes de codear

Para cualquier tarea, listar archivos que se van a crear/modificar y
pedir aprobacion antes de implementar.