# Indicadores y Machine Learning

Sistema de analisis tecnico/ML sobre 199 tickers con PostgreSQL (local + Railway),
pipeline diario, scanner ML, snapshot de opciones US/AR y backtest historico.

## Arquitectura (Plan C, 14/5/2026)

- **LOCAL PostgreSQL** = fuente de verdad para OHLCV, indicadores, features,
  scanner, ML y backtesting historico.
- **Railway PostgreSQL** = SOLO opciones_snapshot (data irrecuperable post-
  mercado siguiente, justifica almacenamiento remoto siempre disponible).
- **Oracle Cloud VM** = cron de snapshot opciones US (3 intentos) + opciones AR.
- **GitHub Actions** = bots Alpaca (3) + intento 3 (backup IP distinta) del
  snapshot opciones US.
- **Windows** = recovery local manual post-cierre (recovery_incremental.bat).
- **Streamlit** = pausado/no critico. Reportes via CSV/Excel/HTML local.

## Documentos de dominio

Documentacion que existe hoy en docs/:

- docs/mcp_server.md      : servidor MCP consultivo (diseno + estado)
- docs/parametros_mcp.md  : registro de umbrales/parametros de las tools MCP
                            y su interpretacion (para re-tunear a futuro)
- docs/reportes.md        : modulo scripts/reports/ -- generador de PDF e
                            infografias para compartir analisis en X
- docs/estrategias_ft.md  : estrategias de forward testing
- docs/forward_testing/   : detalle de forward testing
- docs/checklist_recovery_manual.md : flujos de recovery manual
- dashboard/README.md     : spec del Dashboard (informe descriptivo por ticker).
                            Diseno cerrado 27/5/2026, desarrollo pendiente en
                            rama feature/dashboard. Ver tambien memory/dashboard.md

Politica de documentacion: un doc de dominio se crea SOLO cuando hay
conocimiento real que no se puede derivar leyendo el codigo (ej.
thresholds no obvios, convenciones de columnas, decisiones historicas).
No crear placeholders vacios. El estado general del sistema vive en
este CLAUDE.md; el codigo es la fuente de verdad para arquitectura,
rutas y estructura.

## Reglas no negociables

1. **Fechas**: nunca asumir si una fecha es habil. Usar
   `src/utils/trading_calendar.py` o `scripts/manual/check_fecha.py`.
2. **Encoding cp1252 en Windows**: ASCII puro en strings de codigo, no Unicode
   arrows. Telegram messages pueden usar UTF-8.
3. **Antes de modificar cualquier archivo en src/ o scripts/**: leer el .md
   del dominio relevante.
4. **MCP server (`mcp_server/`)**: SOLO consultivo. No modifica DB, no importa
   funciones con side effects, no toca src/ ni scripts/. Nunca modifica DB,
   scripts, parametros ni infraestructura del proyecto.
5. **Rol PostgreSQL `mcp_reader`** con SELECT-only + read-only transactions.
6. **Validacion SQL con sqlglot** en `run_select` (cuando se implemente).
7. **Catalogo de queries** vive afuera del repo, en `~/queries-catalog/`.
8. **mcp_server/INSTRUCTIONS.md es autocontenido**: NO referencia rutas
   externas al repo.
9. **NO correr 2 scripts yfinance concurrentes**: usar
   `src/utils/yfinance_lock.acquire()` al inicio. Rate limit es por IP.
10. **NO correr scripts yfinance pre-mercado** (antes de 13:30 UTC) sin
    entender el efecto. Algunos scripts etiquetan data del cierre anterior
    con la fecha de hoy -> datos corruptos.

## Antes de codear

Para cualquier tarea: listar archivos que se van a crear/modificar y pedir
aprobacion antes de implementar. Ver `docs/checklist_recovery_manual.md` para
flujos comunes de recovery.

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

## Patrones criticos del proyecto

### DB connection (src/data/database.py:get_engine())
- Chequea `os.getenv("DATABASE_URL")` PRIMERO.
- Si esta seteada (viene de .env.local) -> usa esa = **Railway**.
- Si NO esta seteada -> cae a `DB_CONFIG` (DB_HOST/PORT/USER/PASSWORD del .env) = **local**.
- `pool_pre_ping=True` activo desde 14/5/2026 (previene "server closed connection
  unexpectedly" en scripts largos -- Railway cierra idle ~5min).

### Forzar target LOCAL en scripts que cargan .env.local
Los scripts que hacen `load_dotenv('.env.local', override=True)` setean
DATABASE_URL=Railway sin importar el shell env. Opciones para forzar local:

1. **Renombrar .env.local temporalmente** (mas simple, atomico):
   ```bash
   mv .env.local .env.local.bak
   python scripts/X.py
   mv .env.local.bak .env.local
   ```
2. **Patron setup_target_env()** (ver `scripts/recovery_incremental.py`):
   Parsea ambos .env files explicitamente, elimina DATABASE_URL de os.environ
   cuando --target=local.
3. **Helper ft_env.py** (Forward Testing): `scripts/forward_testing/ft_env.py`
   expone `configurar_entorno_local()`. Los 9 bots FT + ft_setup lo llaman al
   inicio: carga solo `.env` y elimina DATABASE_URL -> get_engine() cae a local.
   FT corre 100% en local (no escribe en Railway).

### yfinance rate limit
- En yfinance 0.2.x, `YFRateLimitError` **NO se levanta como excepcion**.
  Yfinance imprime "Failed downloads: ..." a stdout/stderr y devuelve
  DataFrame vacio. Para detectar rate limit hay que capturar stdout/stderr
  con `contextlib.redirect_stdout/redirect_stderr` y parsear el texto.
- Rate limit es **por IP**, no por proceso. Si Oracle cron + Windows manual
  corren simultaneos, suman carga. Usar `src/utils/yfinance_lock.py`.
- `fast_info.last_price` usa `history(period='1y')` internamente = mismo
  endpoint que `yf.download` /v8/finance/chart/. SIN ventaja de rate limit.

### Opciones snapshot -- irrecuperabilidad
- yfinance/Yahoo solo expone la chain VIGENTE. Una vez abre el mercado del dia
  siguiente (13:30 UTC durante DST), las strikes/contratos del cierre
  anterior dejan de estar disponibles. **Snapshot debe correr antes**.
- Esquema actual: 4 intentos (23:00, 02:00, 04:00 GH, 06:00 UTC).
- **Engine yahooquery (20/5/2026)**: yfinance dejo de servir confiablemente el
  endpoint de opciones (~18/5/2026, verificado en 5 IPs). `33_opciones_snapshot.py`
  y `recovery_incremental.py` aceptan `--engine yfinance|yahooquery` (default
  yfinance, retrocompatible). Los 4 cron pasan `--engine yahooquery`. yahooquery
  trae el chain entero en 1 call/ticker (vs ~30 de yfinance).
- **precio_subyacente desde yahooquery (26/5/2026)**: `_get_precios_subyacentes`
  ANTES leia el ultimo close de precios_diarios, pero el snapshot corre en Oracle
  con .env.local -> lee precios_diarios de RAILWAY, que bajo Plan C esta CONGELADO
  (solo opciones se escriben ahi). El precio quedaba pegado a una fecha vieja,
  contaminando moneyness/muros/expected_move. AHORA toma el precio de yahooquery
  (regularMarketPrice si mercado cerrado, regularMarketPreviousClose si abierto),
  que coincide con el close de precios_diarios LOCAL.
- **NO asumir** que precio_subyacente sale de precios_diarios: sale de yahooquery.

### PostgreSQL ON CONFLICT
- Requiere unique index **FULL** (sin clausula WHERE).
- Partial unique index (`WHERE col IS NOT NULL`) NO sirve para ON CONFLICT
  sin declarar el predicate en el INSERT.
- LOCAL DB puede tener constraints faltantes vs Railway (ej: ticker_zscore_diario
  estaba sin unique en local). Verificar via `pg_indexes` antes de upsert.

### SQLAlchemy
- `text()` con named params (`:param`) + dict, NO `%s` + tuple.
- Alias en CTEs (WITH) deben estar en scope del SELECT/WHERE final.
- Bug detectado 14/5: `WHERE p.fecha` cuando el FROM es `stats s` -> alias
  `p` fuera de scope -> `UndefinedTable`. Ver commit 06118a8.

### Telegram
- No usar backslash dentro de f-string (Python 3.11). Asignar la variable antes.
- `src/pipeline/telegram_notifier.py` con `_send()` para enviar mensajes.

## Scripts clave (referencia rapida)

| Script | Para que |
|--------|----------|
| `scripts/manual/recovery_incremental.bat` | Recovery incremental local (precios + futuros) |
| `scripts/manual/status_local.bat` / `status.bat` | Status DB local / Railway |
| `scripts/manual/poblar_opciones.bat` | Carga manual opciones US (UNA pasada, sin dry-run) |
| `scripts/manual/recover_opciones_tickers.py` | Recovery quirurgico de tickers especificos |
| `scripts/sync_local.bat` | Sync Railway -> Local |
| `scripts/sync_to_railway.bat` | Sync Local -> Railway (paso a paso) |
| `scripts/migrations/clean_ticker_fantasma_se.py` | Limpieza generica ticker fantasma |
| `scripts/migrations/clean_railway_may12.py` | One-shot one-off |
| `scripts/manual/check_fecha.py` | CLI valida dia habil NYSE |
| `scripts/manual/ft_run_diario.bat` | Corre los 10 bots de Forward Testing en local + reporte HTML |
| `scripts/forward_testing/ft_reporte_html.py` | Reporte HTML autocontenido de FT (reportes/ft_reporte.html) |
| `scripts/refresh_earnings_calendar.py` | Refresh earnings_calendar desde Nasdaq (cron Oracle semanal) |
| `scripts/migrations/create_earnings_calendar.py` | Crea la tabla earnings_calendar (Railway + local) |
| `scripts/migrations/migrate_ft_railway_to_local.py` | Migracion puntual de tablas ft_* Railway -> local |
| `scripts/reports/make_infografia.bat <TICKER>` | Infografia PNG para X (datos del MCP, sin LLM). Ver docs/reportes.md |
| `scripts/reports/build_yaml.bat <TICKER>` + `make_report.bat <yaml>` | Reporte PDF detallado con narrativa del LLM |

## Tablas DB principales

Listado completo: usar `describe_table` del MCP o `information_schema`.
Las criticas:
- `precios_diarios` (OHLCV) | `indicadores_tecnicos`
- `features_precio_accion` | `features_market_structure`
- `alertas_scanner` (col: `scan_fecha`, `precio_fecha`)
- `ticker_zscore_diario` | `opciones_zscore_diario`
- `opciones_snapshot` | `opciones_resumen_diario`
- `opciones_sector_zscore_diario` (PCR_vol+vol agregados por sector, z-score)
- `opciones_pcr_plazo_diario` (PCR vol/OI + muros S/R por ventana corto/medio/largo,
  por ticker; fuente src/utils/opciones_plazo.py)
- `opciones_sector_pcr_plazo_diario` (PCR sectorial por ventana + z-score)
- `indicadores_tecnicos_1w` (RSI/MACD semanal)
- `futuros_diarios` | `indicadores_tecnicos_futuros`
- `features_regimen_macro` | `features_ml` | `features_sector`
- `earnings_calendar` (ticker PK, earnings_date DATE NULL; refrescada semanal
  desde Nasdaq por `refresh_earnings_calendar.py`)
- `ft_*` (5 tablas Forward Testing: estrategias, operaciones, candidatos_diarios,
  metricas_diarias, posiciones_diarias) -- LOCAL es fuente de verdad

## Flujo de recovery manual (caso comun: Oracle cron fallo)

```
1. status.bat               (ver Railway: que dias faltan)
2. recovery_incremental.bat (LOCAL: bajar precios faltantes via yfinance)
3. status_local.bat         (verificar 0 tickers desactualizados)
4. cron_diario --step features  (calcular features sobre los nuevos precios)
5. cron_diario --step scanner   (generar alertas)
6. Para z-scores: usar src/utils/zscore_pipeline.backfill_zscore_tickers(engine, desde=date)
```

Ver `docs/checklist_recovery_manual.md` para casos detallados (A: Oracle cron
pipeline, B: snapshot opciones, C: scanner faltante, D: dia perdido completo).

## Estado del MCP server (subdirectorio mcp_server/)

Diseno completo: docs/mcp_server.md
Reglas de uso del propio server: mcp_server/INSTRUCTIONS.md

FASE 1 COMPLETA (15/05/2026). 14 tools registradas y validadas contra
la DB local via Gemini CLI.

Fases completadas:
- Fase 0: skeleton + tool ping
- Fase 1A: calendar tools (check_trading_day, get_last_trading_day)
- Fase 1B: exploration tools (list_tables, describe_table, list_tickers)
- Fase 1C: stocks tools (price_history, technical_indicators,
  price_action, market_structure)
- Fase 1D: opciones (get_options_analysis -- una sola tool en vez de las
  dos planeadas; combina resumen, zscore, PCR por vencimiento, delta OI)
  Rediseñada 16/05/2026: las 3 secciones devuelven metricas computadas en
  vez de series crudas -- ~8.900 tokens menos por llamada (ver docs).
- Fase 1E: alertas ML (get_ml_alert_history)
- Fase 1F: composicion (get_ticker_overview)
- Extra: screener multi-criterio (screen_tickers, opcion B con 17
  parametros nullable) -- no estaba en el plan original de 12 tools

- Fase 1G: sintesis (get_ticker_sintesis, 26/05/2026) -- une tecnico D+W
  (RSI/MACD diario y semanal clasificados) x opciones por plazo (PCR_vol,
  muros de OI como S/R) x sentimiento sectorial, mas reglas de interpretacion.
  Recalcula los muros con el close real (defensa ante precio_subyacente viejo).

Tools registradas hasta hoy (16):
- ping
- check_trading_day, get_last_trading_day
- list_tables, describe_table, list_tickers
- get_price_history, get_technical_indicators, get_price_action,
  get_market_structure
- get_options_analysis
- get_ticker_overview
- screen_tickers
- get_ml_alert_history
- get_ticker_sintesis

Fases pendientes:
- run_select con validacion sqlglot (postergado: screen_tickers cubre
  la mayoria de las consultas cross-ticker; ver regla 6)
- safety.py (validacion SQL) -- pendiente, depende de run_select
- Fase 2: catalogo de queries (save_query, list_saved, recall_query)
- Fase 3: bot de Telegram (MVP local en Windows; ver docs/mcp_server.md)

Patrones aprendidos en Fase 1 (importantes para futuras tools):
- Columnas flag (choch_*, bos_*, patron_*, es_alcista, vol_spike) pueden
  ser smallint(0/1) o boolean segun version de la DB. En CASE de SQL usar
  ::int != 0 para normalizar ambos tipos. PostgreSQL valida tipos de CASE
  en parse-time, antes de evaluar parametros NULL.
- Tools que devuelven al LLM deben sintetizar columnas booleanas crudas
  en campos legibles (patron_activo, señal_smc) y NO incluir las columnas
  0/1 originales -- los modelos basicos las vuelcan sin interpretar.
- Eficiencia de tokens: computar conclusiones en Python (gratis, local) y
  enviar al LLM resumenes, no data cruda voluminosa. El LLM paga tokens por
  cada fila de entrada y razona peor que una formula. PERO: si el dato ES
  una serie temporal, preservar la trayectoria -- un min/max/promedio es
  ciego a la direccion. Resumir la conclusion, no aplanar la serie.
  Caso de referencia: rediseño de get_options_analysis (commit 0d92516).

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

6. **Conversion explicita asyncpg.Record -> dict** antes de retornar
   (MCP requiere JSON-serializable).

7. **Tools llevan annotations declarativas**: readOnlyHint=True,
   destructiveHint=False, idempotentHint=True, openWorldHint=False.
   Estas annotations las usan los clientes MCP para decidir si piden
   confirmacion al usuario.

8. **Tests con pytest-asyncio en mode STRICT**: cada test async lleva
   @pytest.mark.asyncio explicito. Tests de integracion marcados con
   @pytest.mark.integration para poder skipearlos sin DB.
