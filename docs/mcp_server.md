# MCP Server -- Servidor de consulta para activos_ml
# Ultima actualizacion: 2026-05-15

## Que es

Servidor MCP (Model Context Protocol) que expone la base de datos del proyecto
IndicadoresDeStockML como herramientas consumibles por clientes LLM. Permite
hacer preguntas en lenguaje natural sobre el sistema (precios, indicadores,
opciones, alertas, backtests) y recibir respuestas con datos extraidos
directamente de las tablas existentes.

**Why:** facilitar el acceso conversacional a la informacion ya calculada por
el pipeline, sin tener que escribir SQL ni abrir Streamlit. Hoy via Gemini CLI
en consola; manana via Telegram para consultas remotas.

**How to apply:** este archivo se carga al inicio de cualquier sesion de
implementacion o mantenimiento del MCP server. Es la fuente de verdad del
diseno; cualquier cambio de arquitectura se refleja aca primero.

---

## Estado de implementacion (2026-05-16) -- FASE 1 COMPLETA

14 tools registradas y validadas contra la DB local via Gemini CLI.
El resto de esta seccion describe el DISEÑO; abajo se aclara que se
construyo realmente y donde se desvio del plan original.

### Tools en produccion (14)

| Tool | Fase | Estado |
|---|---|---|
| ping | 0 | OK |
| check_trading_day, get_last_trading_day | 1A | OK |
| list_tables, describe_table, list_tickers | 1B | OK |
| get_price_history, get_technical_indicators, get_price_action, get_market_structure | 1C | OK |
| get_options_analysis | 1D | OK |
| get_ticker_overview | 1F | OK |
| screen_tickers | extra | OK |
| get_ml_alert_history | 1E | OK |

### Desviaciones del diseño original

1. **Opciones: 1 tool en vez de 2.** El plan tenia `get_options_summary`
   y `get_options_zscore` separadas. Se implemento una sola
   `get_options_analysis` que combina resumen diario, z-scores, PCR por
   vencimiento y delta de OI por contrato en el periodo.

   Rediseñada 16/05/2026 (commit 0d92516) para reducir consumo de tokens
   ~8.900 por llamada. Las 3 secciones devuelven metricas computadas en
   Python en vez de series crudas:
   - tendencia_diaria -> {actual, serie}: ultimo dia completo + serie
     diaria recortada (conserva la trayectoria, recorta campos redundantes).
   - pcr_por_vencimiento -> resumen por vencimiento: PCR OI inicio/actual,
     delta call/put OI, sesgo y tendencia. Ya NO devuelve la serie diaria.
   - acumulacion_oi -> top 10 calls + 10 puts, sin precio_subyacente ni
     oi_inicio por fila (redundantes/derivables).

2. **screen_tickers: tool nueva no planeada.** Screener multi-criterio
   sobre los 199 tickers (opcion B: 17 parametros nullable, cada filtro
   usa el patron `$N IS NULL OR condicion`). Cubre las consultas
   cross-ticker que originalmente se pensaba resolver con `run_select`.

3. **get_ticker_overview sin `days_back`.** Ancla siempre en el ultimo
   dato disponible por tabla. Un solo parametro: `ticker`.

4. **get_ml_alert_history con filtros ampliados.** En vez de
   `(ticker, days_back)` toma `ticker, desde, hasta, alert_nivel,
   alert_score_min, solo_verificados, limit`. Incluye retornos
   post-facto (retorno_Nd_real) para evaluar performance del modelo.

5. **run_select y safety.py: postergados.** `screen_tickers` cubre la
   mayoria de los casos cross-ticker. `run_select` con validacion
   sqlglot queda pendiente; mientras no exista, safety.py tampoco.

### Lecciones tecnicas de Fase 1

- **Columnas flag boolean vs smallint:** choch_*, bos_*, patron_*,
  es_alcista, vol_spike pueden estar tipadas como smallint(0/1) o boolean
  segun version de la DB. En expresiones CASE de SQL usar `::int != 0`
  para normalizar. PostgreSQL valida el tipo de retorno del CASE en
  parse-time, antes de evaluar que parametros son NULL -- por eso un CASE
  mal tipado falla aunque su filtro este inactivo.
- **Sintesis para el LLM:** las tools deben convertir columnas booleanas
  crudas en campos legibles (patron_activo, señal_smc, señal_reciente) y
  NO devolver las columnas 0/1 originales. Los modelos basicos
  (gemini-2.5-flash) vuelcan los datos crudos sin interpretarlos.
- **Eficiencia de tokens:** el LLM paga tokens por cada fila de entrada y
  razona peor que una formula. Computar conclusiones en Python (gratis,
  local) y enviar resumenes, no data cruda voluminosa. Excepcion critica:
  si el dato ES una serie temporal, preservar la trayectoria -- un
  min/max/promedio es ciego a la direccion (subida sostenida vs zigzag dan
  el mismo resumen). Resumir la conclusion, no aplanar la serie. Caso de
  referencia: rediseño de get_options_analysis (commit 0d92516).

---

## Principio rector -- SOLO CONSULTIVO (condicion critica)

El MCP server es UNICAMENTE consultivo. No modifica datos, scripts, parametros
ni infraestructura del proyecto. Esta restriccion es estructural, no opcional,
y se enforce en cuatro capas independientes:

1. **Rol PostgreSQL `mcp_reader`** con permisos SELECT-only y
   `default_transaction_read_only = on`. Aunque el codigo intentara escribir,
   PostgreSQL lo rechaza.
2. **Validacion SQL con sqlglot** en cada llamada a `run_select`: solo
   sentencias SELECT o WITH...SELECT pasan. DML/DDL rechazado antes de llegar
   a DB.
3. **Imports desde `src/` solo de funciones puras o de lectura.** No se importa
   `zscore_pipeline.calcular_zscore_*` (escribe DB) ni ningun script con side
   effects. Si una funcion mezcla calculo y persistencia, queda afuera del MCP.
4. **MCP annotations declarativas:** todas las tools del server llevan
   `readOnlyHint: true` y `destructiveHint: false`. Los clientes MCP usan
   estas anotaciones para decidir si requieren confirmacion del usuario.

El catalogo de queries (fase 2) escribe SOLO en `~/queries-catalog/`, repo
separado del proyecto. Nunca toca el repo principal ni la DB.

---

## Stack y decisiones de arquitectura

### Lenguaje y libreria
- **Python 3.11+** (el mismo del proyecto principal)
- **MCP Python SDK** con FastMCP (libreria oficial de Anthropic)
- **asyncpg** para conexion a PostgreSQL (async nativo, sin overhead)
- **sqlglot** para validacion SQL declarativa (entiende dialecto Postgres)
- **pydantic-settings** para configuracion (lee `.env` existente del proyecto)
- **structlog** para logging estructurado a stderr

### Por que Python y no TypeScript
El skill oficial de MCP recomienda TypeScript por madurez de SDK. Para este
caso elegimos Python por tres razones:
1. El proyecto IndicadoresDeStockML ya es Python; reutilizamos imports.
2. Las funciones del proyecto (`trading_calendar.is_trading_day`,
   `precio_accion.detectar_patron`, thresholds documentados) viven en Python.
3. Python es el lenguaje de trabajo del usuario (analista de datos).

### Ubicacion en el repo
Subdirectorio `mcp_server/` dentro de `IndicadoresDeStockML/`. NO es repo
separado. Razones:
- Reutiliza `src/utils/trading_calendar.py`, thresholds en `src/utils/config.py`,
  funciones puras de `src/indicators/*`.
- Comparte `.env` y `.env.local`.
- Si la logica del proyecto cambia (ej. nuevos thresholds de PCR), el MCP
  hereda automaticamente.

### Catalogo afuera
Repo Git separado en `~/queries-catalog/` con ciclo de vida independiente.
Razones:
- Las queries crecen todos los dias; el codigo del MCP cambia poco.
- Facilita versionado, sharing y backup independientes.
- Evita ensuciar el repo principal con archivos de exploracion.

### Transports
- **stdio** como default (clientes locales: Gemini CLI, Claude Desktop,
  Claude Code, Cursor).
- **Streamable HTTP** opcional via flag `--transport http` (clientes remotos
  para fase 3, despliegue en Oracle Cloud).

---

## Estructura de directorios

```
IndicadoresDeStockML/                       # repo principal (existente)
|-- src/                                     # SIN CAMBIOS
|-- scripts/                                 # SIN CAMBIOS
|-- app/                                     # Streamlit, SIN CAMBIOS
|-- mcp_server/                              # NUEVO subdirectorio
|   |-- __init__.py
|   |-- server.py                            # FastMCP entry point
|   |-- config.py                            # pydantic-settings, lee .env existente
|   |-- safety.py                            # validacion SQL con sqlglot
|   |-- INSTRUCTIONS.md                      # reglas de uso (fuente unica)
|   |-- db/
|   |   |-- __init__.py
|   |   |-- pool.py                          # asyncpg pool, lifecycle
|   |   `-- queries.py                       # SQL templates parametrizados
|   `-- tools/
|       |-- __init__.py
|       |-- exploration.py                   # list_tickers, describe_table, run_select
|       |-- calendar.py                      # check_trading_day, get_last_trading_day
|       |-- price.py                         # get_price_history, get_price_action
|       |-- structure.py                     # get_market_structure
|       |-- indicators.py                    # get_technical_indicators
|       |-- options.py                       # get_options_summary, get_options_zscore
|       |-- alerts.py                        # get_ml_alert_history
|       |-- overview.py                      # get_ticker_overview (orquesta otras)
|       |-- macro.py                         # get_market_regime, get_futures_snapshot (1.5)
|       |-- backtest.py                      # list_strategies, get_strategy_* (1.5)
|       |-- forward.py                       # ft_* tools (1.5)
|       `-- catalog.py                       # save_query, list_saved (fase 2)
|-- tests/
|   `-- mcp_server/
`-- requirements.txt                         # + mcp + sqlglot + asyncpg

~/queries-catalog/                           # repo Git SEPARADO
|-- README.md                                # convenciones, glosario
|-- adhoc/                                   # exploraciones efimeras, fechadas
|   `-- 2026-05-09_exploracion_BAC.sql
`-- catalog/                                 # queries utiles, organizadas
    |-- velas/
    |   |-- bac_velas_3d.sql
    |   `-- bac_velas_3d.md                  # contexto, supuestos
    |-- opciones/
    |-- alertas/
    `-- backtest/
```

---

## Capa de seguridad -- condiciones necesarias y suficientes

Tres capas independientes. Si una falla, las otras dos siguen protegiendo.

### 1. Rol PostgreSQL (primera linea)

```sql
-- Crear rol con permisos minimos
CREATE ROLE mcp_reader WITH LOGIN PASSWORD 'usar_password_robusto';

-- Permisos de conexion
GRANT CONNECT ON DATABASE activos_ml TO mcp_reader;
GRANT USAGE ON SCHEMA public TO mcp_reader;

-- Permisos de lectura sobre tablas
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO mcp_reader;

-- Restricciones de sesion
ALTER ROLE mcp_reader SET statement_timeout = '30s';
ALTER ROLE mcp_reader SET default_transaction_read_only = on;
ALTER ROLE mcp_reader SET idle_in_transaction_session_timeout = '60s';

-- Restriccion de conexiones simultaneas (defensa contra abuso)
ALTER ROLE mcp_reader CONNECTION LIMIT 5;
```

Verificacion: `psql -U mcp_reader -d activos_ml -c "DELETE FROM precios_diarios"`
debe fallar con error de permisos. Si no falla, el rol esta mal configurado.

### 2. Validacion SQL con sqlglot (segunda linea)

Implementada en `mcp_server/safety.py`. Para cada SQL recibido por
`run_select`:

- Parsea con `sqlglot.parse_one(sql, dialect='postgres')`.
- Verifica que sea exclusivamente `SELECT` o `WITH...SELECT`.
- Rechaza si encuentra: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE,
  COPY, GRANT, REVOKE, VACUUM, ANALYZE.
- Rechaza llamadas a funciones peligrosas: `pg_read_file`, `pg_ls_dir`,
  `dblink`, `lo_export`, `lo_import`, cualquier `pg_*` que toque filesystem.
- Inyecta `LIMIT 1000` automaticamente si la query no tiene LIMIT y la tabla
  fuente puede tener mas de 1000 filas (precios_diarios, opciones_snapshot).
- Loggea el SQL hasheado (SHA-256) + duracion + filas devueltas a structlog.

### 3. Imports puros desde src/ (tercera linea)

Lista cerrada de modulos que el MCP server puede importar:

| Modulo | Funciones permitidas | Razon |
|---|---|---|
| `src.utils.trading_calendar` | `is_trading_day`, `prev_trading_day`, `next_trading_day`, `describe_date`, `holiday_name` | Solo lectura de constantes |
| `src.utils.config` | `ALL_TICKERS`, `TICKER_SECTOR`, `SECTORES_ML`, `FUTUROS_TICKERS`, `RSI_PERIOD`, etc. | Constantes |
| `src.indicators.precio_accion` | `detectar_patron_doji`, `detectar_hammer`, etc. | Funciones puras (calculo sobre DataFrame) |

**Prohibido importar:**
- `src.utils.zscore_pipeline` (escribe DB)
- `scripts/*` (todos tienen side effects)
- `src.pipeline.*` (orquesta escritura)
- `src.trading.*` (logica de bots, side effects via Alpaca API)

Si una funcion deseable mezcla calculo y persistencia, refactorizarla en el
proyecto principal antes de importarla. NO duplicar logica en el MCP server.

### 4. MCP annotations declarativas (cuarta linea)

Cada tool registrada en el server lleva annotations en su definicion:

```python
@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_price_history(ticker: str, days_back: int = 10) -> dict:
    ...
```

Los clientes MCP (Gemini CLI, Claude Desktop, etc.) usan estas annotations
para decidir si requieren confirmacion del usuario antes de ejecutar la tool.
Las herramientas de catalogo (`save_query`) llevaran `readOnlyHint: false` y
`idempotentHint: false`, indicandoles a los clientes que pidan confirmacion.

---

## Tools por fase

### Convencion de naming

- Snake_case en ingles (convencion MCP internacional).
- Verbo de accion al inicio: `get_*`, `list_*`, `describe_*`, `check_*`,
  `save_*`.
- Sin prefijo de dominio (los nombres son auto-explicativos por el verbo y el
  objeto).

### Annotations por categoria

| Categoria de tool | readOnlyHint | destructiveHint | idempotentHint |
|---|---|---|---|
| Exploracion (list_*, describe_*) | true | false | true |
| Lectura de datos (get_*) | true | false | true |
| Calendar utilities | true | false | true |
| `save_query` (catalogo) | false | false | false |

### Fase 1 -- MVP (12 tools)

#### Exploracion y utilidad

```
list_tickers(sector: str | None = None,
             only_ml_active: bool = False) -> list[dict]
    Devuelve tickers del universo (199 totales). Filtros opcionales por sector
    o por flag de activo en ML.
    Source: tabla `activos`.

list_tables(pattern: str | None = None) -> list[str]
    Devuelve tablas accesibles. Filtra por patron LIKE si se provee.

describe_table(table_name: str) -> dict
    Devuelve schema de la tabla: columnas, tipos, indices, comentarios si los
    hay.

run_select(sql: str, limit: int = 1000) -> dict
    Ejecuta SELECT validado. Rechaza cualquier SQL que no sea SELECT puro.
    Inyecta LIMIT automaticamente.
```

#### Calendar

```
check_trading_day(date: str) -> dict
    Verifica si una fecha es habil NYSE. Retorna:
    {"date": "2026-05-09", "is_trading_day": false,
     "reason": "weekend", "next_trading_day": "2026-05-11"}
    Importa src.utils.trading_calendar.

get_last_trading_day() -> dict
    Retorna el ultimo dia habil hasta hoy.
```

#### Acciones

```
get_price_history(ticker: str, days_back: int = 10) -> list[dict]
    OHLCV diario de los ultimos N dias habiles. Source: precios_diarios.

get_technical_indicators(ticker: str, days_back: int = 10) -> list[dict]
    18 indicadores tecnicos: SMA21/50/200, dist_sma*, RSI14, MACD/signal/hist,
    ATR14, BB upper/middle/lower, OBV, vol_relativo, ADX, momentum.
    Source: indicadores_tecnicos.

get_price_action(ticker: str, days_back: int = 10,
                 group: str | None = None) -> list[dict]
    Features de precio_accion. group puede ser:
    - "anatomia": 9 features de anatomia de vela
    - "patrones": 8 patrones clasicos (doji, hammer, etc.)
    - "rolling":  8 features de estructura rolling
    - "volumen":  7 features de volumen direccional
    - None: las 32 features.
    Source: features_precio_accion.

get_market_structure(ticker: str, days_back: int = 10,
                     ventana: int = 5) -> list[dict]
    Features SMC: estructura, BOS, CHoCH, swings. ventana: 5 (tactico) o 10
    (estrategico).
    Source: features_market_structure.
```

#### Opciones

```
get_options_summary(ticker: str, days_back: int = 10) -> list[dict]
    PCR vol, PCR OI, IV calls/puts, n contratos, max OI strike, precio
    subyacente. Incluye label PCR derivada de los thresholds del proyecto.
    Source: opciones_resumen_diario.

get_options_zscore(ticker: str, days_back: int = 10) -> list[dict]
    Z-scores de volumen calls/puts/total, PCR, IV vs ventana 60d.
    Source: opciones_zscore_diario.
```

#### Senales

```
get_ml_alert_history(ticker: str, days_back: int = 20) -> list[dict]
    Historico de alertas del scanner ML: alert_nivel, alert_score,
    alert_detalle.
    Source: alertas_scanner. Columna fecha: scan_fecha.
```

#### Vista cruzada

```
get_ticker_overview(ticker: str, days_back: int = 5) -> dict
    Orquesta get_price_history + get_price_action (solo grupo "patrones") +
    get_market_structure + get_options_summary + get_ml_alert_history.
    Devuelve dict con shape unificado por fecha.
    Es la tool de facto para preguntas tipo "como esta BAC".
```

### Fase 1.5 -- Extensiones obvias

Implementar conforme aparezca demanda real. No bloqueantes para fase 1.

```
# Macro
get_market_regime(date: str | None = None) -> dict
    es_mercado_alcista, es_nq_liderando, divergencia_russell,
    vol_spike_macro, dist_sma50_es. Source: features_regimen_macro.

get_futures_snapshot(tickers: list[str] | None = None,
                     days_back: int = 5) -> list[dict]
    OHLCV + indicadores tecnicos de futuros. Source: futuros_diarios JOIN
    indicadores_tecnicos_futuros.

# Z-scores y actividad inusual
get_unusual_activity(date: str | None = None,
                     sector: str | None = None,
                     z_threshold: float = 2.0) -> list[dict]
    Cross ticker_zscore_diario + opciones_zscore_diario donde ambos > threshold.
    Util para detectar actividad institucional probable.

get_ticker_zscore(ticker: str, days_back: int = 10) -> list[dict]
    Z-scores de volumen y retorno de la accion. Source: ticker_zscore_diario.

# Backtesting historico
list_strategies() -> list[dict]
    Estrategias del backtesting historico. Source: bt_hist_estrategias.

get_strategy_metrics(strategy: str,
                     period: str | None = None) -> dict
    Metricas resumen: retorno total, drawdown, win rate, profit factor.

get_strategy_trades(strategy: str,
                    sort_by: str = "pnl",
                    top_n: int = 10) -> list[dict]
    Top trades por PnL, fecha o duracion. Source: bt_hist_operaciones.

get_equity_curve(strategy: str) -> list[dict]
    Equity curve diaria. Source: bt_hist_metricas_diarias.

# Forward testing en vivo
list_ft_strategies() -> list[dict]
get_ft_open_positions(strategy: str | None = None) -> list[dict]
get_ft_metrics(strategy: str, days_back: int = 30) -> dict

# Rule-based scoring (cuando se regenere local)
get_rule_based_score(ticker: str, days_back: int = 10) -> list[dict]
    Score 0-100 rule-based pre-ML. ATENCION: tabla scoring_tecnico esta vacia
    en local; consultar Railway o regenerar antes de usar.
```

### Fase 2 -- Catalogo selectivo

```
save_query(name: str,
           sql: str,
           description: str,
           tags: list[str] | None = None) -> dict
    Guarda query en ~/queries-catalog/catalog/<tag>/<name>.sql con metadata
    en .md adyacente. Requiere confirmacion del usuario via annotation
    `readOnlyHint: false`.

list_saved_queries(tag: str | None = None) -> list[dict]
    Lista queries guardadas con su descripcion y tags.

recall_query(name: str) -> dict
    Devuelve SQL + metadata de una query guardada. Util para que el modelo
    chequee antes de generar una nueva.
```

---

## System prompt y reglas de uso

Las reglas viven en `mcp_server/INSTRUCTIONS.md` como fuente unica. Cada
cliente las consume a su manera (symlink, copia, include).

### Reglas criticas (no negociables)

1. **REGLA ABSOLUTA -- FECHAS:** nunca asumir si una fecha es habil. Llamar
   `check_trading_day` o `get_last_trading_day` antes de razonar sobre
   ventanas temporales.

2. **Universo cerrado:** 199 tickers conocidos. Si el usuario pregunta por
   uno fuera, responder que no esta en el universo en lugar de inventar
   datos.

3. **Distincion de scores:** son DOS cosas distintas:
   - `scoring_tecnico.score`: rule-based (RSI/MACD/SMA200/SMA50/Momentum/SMA21).
     VACIA en local, 37k filas en Railway.
   - `alertas_scanner.alert_score`: ML (Random Forest). POBLADA en local.
   Si piden "score" sin aclarar, default al ML y notar la distincion.

4. **Convenciones de columnas no obvias:**
   - `precios_semanales.fecha_semana` (NO "fecha")
   - `alertas_scanner.scan_fecha` (NO "fecha_alerta", ese nombre no existe)

5. **OI = T-1 en yfinance.** Cuando se reporte open_interest, agregar nota:
   "OI corresponde al cierre del dia anterior".

6. **IV con cobertura baja** (< 30%) marcar la respuesta con advertencia:
   "IV cov: X%, baja cobertura, tomar con reserva".

7. **Gaps de datos conocidos en opciones:** 2026-04-23 y 2026-04-25 sin data
   recuperable. Si caen en la ventana consultada, marcar y no inventar.

8. **scoring_tecnico vacia en local** -- si la tool `get_rule_based_score` se
   llama con DSN local, advertir y sugerir regeneracion o consulta a Railway.

### Reglas de granularidad

- Por defecto: respuesta tabular markdown densa + parrafo descriptivo breve.
- NO interpretar. Describir que muestran los datos. Las conclusiones las saca
  el usuario.
- Si el usuario pide profundizar ("explicame mas", "por que", "detallame X"),
  ampliar usando los thresholds y convenciones documentadas del proyecto.
- Cuando una senal no este disponible (gap, cobertura baja, tabla vacia),
  marcarlo explicitamente. No inventar ni omitir en silencio.

### Reglas de composicion de "sentimiento"

El sistema tiene multiples senales que indican sentimiento. Cuando el usuario
pide "sentimiento" sin aclarar, devolver TODAS las disponibles para la
ventana consultada y dejar que el usuario interprete:

| Senal | Tabla / columna | Interpretacion documentada |
|---|---|---|
| Estructura SMC corto plazo | features_market_structure.estructura_5 | -1 bajista / 0 neutral / +1 alcista |
| Estructura SMC medio plazo | features_market_structure.estructura_10 | idem |
| Tendencia semanal (RSI/MACD) | AL VUELO desde precios_diarios via src.utils.weekly_tf (resample W-FRI + ta) | get_ticker_sintesis lo computa en el momento (29/5/2026). Ya NO lee indicadores_tecnicos_1w (congelada, pipeline 1w deprecado 28/5/2026) ni necesita staleness guard. Fuente unica compartida con el dashboard (dashboard/sintesis_data._semanal_bundle). |
| Score ML | alertas_scanner.alert_score | 0-100, niveles documentados |
| Nivel cualitativo ML | alertas_scanner.alert_nivel | COMPRA_FUERTE/COMPRA/NEUTRO/VENTA/VENTA_FUERTE |
| Sesgo PCR volumen | opciones_resumen_diario.pcr_vol | < 0.7 ALCISTA / 0.7-1.0 NEUTRO / > 1.0 BAJISTA |
| Z-score PCR | opciones_zscore_diario.pcr_vol_zscore | abs > 2 = inusual |
| Patrones de vela | features_precio_accion.patron_* | bool, segun el patron |
| Regimen macro | features_regimen_macro.es_mercado_alcista | 1 alcista / 0 bajista (ES vs SMA50w) |

---

## Portabilidad entre clientes MCP

### Lo que NO cambia entre clientes

El servidor MCP en si: codigo, tools, capa de DB, validacion. Cualquier
cliente compatible con el protocolo lo consume sin modificacion.

### Lo que SI cambia (y como mitigarlo)

| Aspecto | Mitigacion |
|---|---|
| Archivo de configuracion del cliente | Documentar el JSON de cada cliente en este archivo (anexo) |
| Ubicacion del system prompt | Fuente unica en `mcp_server/INSTRUCTIONS.md`; cada cliente la importa via symlink/copia |
| Quirks del modelo cliente | Tool descriptions exhaustivas, schemas estrictos, naming consistente |

### Matriz de clientes principales

| Cliente | Config | Transport | System prompt |
|---|---|---|---|
| Gemini CLI | `~/.gemini/settings.json` (key `mcpServers`) | stdio/SSE/HTTP | `~/.gemini/GEMINI.md` o `.gemini/GEMINI.md` |
| Claude Desktop | `claude_desktop_config.json` | stdio/HTTP | Limitado, via app |
| Claude Code | `.mcp.json` (proyecto o global) | stdio/SSE/HTTP | `CLAUDE.md` del proyecto |
| Cursor | `.cursor/mcp.json` | stdio/SSE | `.cursor/rules/*.md` |
| Cline / Continue (VSCode) | Settings de la extension | stdio | Settings de la extension |

Nota: El estado de soporte MCP en herramientas de GitHub (Copilot CLI,
Codespaces) esta evolucionando. Verificar al momento de migrar.

### Diez practicas que aseguran portabilidad

1. **Transport doble desde el dia uno**: stdio default + flag
   `--transport http` para HTTP/SSE.
2. **Nombres de tools en ingles, snake_case, action-oriented**
   (`get_options_summary`, no `obtener_resumen_opciones`).
3. **Descripciones extensas en cada tool**: cuando, por que, ejemplos de uso.
4. **JSON Schema estricto en inputs** via Pydantic. Tipos, defaults,
   descripciones por campo.
5. **Outputs estructurados** (dicts con shape predecible), no strings libres.
6. **No depender de quirks de un modelo**: cada tool valida sus propios
   inputs.
7. **Configuracion via env vars**, nada hardcoded.
8. **Logging a stderr** (stdio mode usa stdout para protocolo).
9. **Versionado del server** en metadata (`version: "0.1.0"`).
10. **`mcp_server/INSTRUCTIONS.md` como fuente unica** de las reglas de uso.

### Procedimiento de migracion (ej. Gemini CLI -> Claude Code)

1. Editar `.mcp.json` del proyecto Claude Code con la misma config que tenia
   en `~/.gemini/settings.json` (cambian un par de keys, el resto es igual).
2. Crear `CLAUDE.md` en el proyecto que haga `include` o copia de
   `mcp_server/INSTRUCTIONS.md`.
3. Verificar con el comando equivalente a `/mcp` del cliente nuevo que las
   tools aparecen.
4. Listo. Total: 10-15 minutos.

---

## Plan de implementacion por fases

### Fase 0 -- Setup (60-90 min)

1. Crear rama `mcp-server` desde main de `IndicadoresDeStockML`.
2. Crear rol PostgreSQL `mcp_reader` con DDL del anexo. Verificar que un
   `DELETE` falla.
3. Crear estructura `mcp_server/` con archivos vacios (skeleton).
4. Agregar `mcp`, `sqlglot`, `asyncpg`, `structlog` a `requirements.txt`.
5. Crear `mcp_server/INSTRUCTIONS.md` con las reglas criticas.
6. Crear repo Git separado en `~/queries-catalog/` con README de
   convenciones.
7. Crear `mcp_server/server.py` con UNA tool `ping()` que retorne
   `{"status": "ok"}`. Esto es para validar el descubrimiento del cliente.
8. Configurar `~/.gemini/settings.json` con el server.
9. Verificar con `/mcp` dentro de Gemini CLI que aparece y la tool `ping`
   responde.

Criterio de salida: `/mcp` lista el server `db-consultor` con la tool
`ping`, y al ejecutarla retorna `{"status": "ok"}`.

### Fase 1 -- MVP (12 tools, ~3 sesiones)

Implementar tools en este orden, una a la vez, con test manual desde Gemini
CLI antes de pasar a la siguiente:

1. Calendar: `check_trading_day`, `get_last_trading_day`.
2. Exploracion: `list_tables`, `describe_table`, `list_tickers`.
3. Acciones: `get_price_history`, `get_technical_indicators`,
   `get_price_action`, `get_market_structure`.
4. Opciones: `get_options_summary`, `get_options_zscore`.
5. Senales: `get_ml_alert_history`.
6. Composicion: `get_ticker_overview` (orquesta las anteriores).
7. Generica con validacion: `run_select`.

Criterio de salida: la pregunta de ejemplo de BAC (anexo) se responde
correctamente desde Gemini CLI.

### Fase 1.5 -- Extensiones (segun demanda)

Agregar tools de fase 1.5 cuando aparezcan preguntas que las requieran. No
implementar todas de una vez: cada una se justifica con un caso de uso real.

### Fase 2 -- Catalogo (~1 sesion)

1. Implementar `save_query`, `list_saved_queries`, `recall_query`.
2. Definir convenciones del catalogo en `~/queries-catalog/README.md`:
   estructura de directorios, formato de metadata .md, convencion de naming.
3. Promover queries utiles de fase 1 al catalogo.

### Fase 3 -- Bot de Telegram (MVP local primero)

REVISADO 16/05/2026. El plan original (desplegar en Oracle Cloud contra
Railway) quedo obsoleto con el Plan C: la DB local de Windows es la fuente
de verdad para todo menos opciones_snapshot. Un MCP server en Oracle contra
Railway solo veria opciones -- no precios, indicadores, features ni scanner.

Se adopta un enfoque MVP local primero.

**Concepto clave:** hoy Gemini CLI hace de cliente MCP -- recibe el mensaje,
llama al LLM, orquesta las tool calls contra el server y devuelve la
respuesta. Para Telegram hay que CONSTRUIR ese cliente/orquestador; el LLM
nunca usa el MCP directamente, solo pide "llamar tool X" y el orquestador
ejecuta.

#### MVP -- alcance (Fase 3a)

Todo local en Windows, sin IP publica ni infraestructura cloud:

- Bot de Telegram con long polling (sin webhook, sin nginx, sin tunel).
- Corre en la notebook Windows (debe estar encendida).
- MCP server por stdio: el bot lo lanza como subproceso, igual que Gemini CLI.
- DB local directa (la fuente de verdad del Plan C).
- Whitelist estricta de chat_id -- solo la cuenta del usuario; sin whitelist
  el bot ni responde a /start.
- Loop agentico: mensaje -> LLM -> tool calls -> MCP -> LLM -> respuesta.
- Sin memoria de conversacion (cada mensaje fresco -- mas simple y barato).

Componentes a construir:

| Componente | Funcion | Herramienta |
|---|---|---|
| Bot Telegram | Recibe/envia mensajes | python-telegram-bot (long polling) |
| Cliente LLM | Llama a Gemini con tool-calling | SDK google-genai |
| Cliente MCP | Conecta al server, ejecuta tool calls | MCP Python SDK (lado cliente) |
| Orquestador | Loop agentico mensaje->LLM->MCP->LLM | codigo propio |

Decision de diseño pendiente: **formato del mensaje de salida**. Telegram no
es una consola -- las tablas densas no se ven bien en movil, hay limite de
4096 caracteres por mensaje, soporta Markdown/HTML limitado. Definir si la
respuesta es narrativa corta, bullets, o mensajes partidos. Afecta como se
instruye al LLM.

El costo en tokens NO cambia respecto a Gemini CLI: cada mensaje sigue siendo
un round-trip (o varios, por el loop agentico). El trabajo de reduccion de
tokens en las tools sigue valiendo.

#### Fuera del MVP (Fase 3b, segun necesidad)

- Server siempre-online: requiere resolver el acceso a la DB local desde un
  host remoto (tunel Tailscale / SSH reverse, o mover el bot a un host con
  linea de vista a la DB). NO desplegar contra Railway -- no tiene los datos.
- Memoria de conversacion multi-turno.
- Transporte HTTP/SSE (el MVP usa stdio; long polling no necesita IP publica).

---

## Catalogo de queries -- repo separado

### Ubicacion
`~/queries-catalog/`, repo Git independiente.

### Estructura
```
~/queries-catalog/
|-- README.md                # convenciones, glosario de tablas
|-- adhoc/                   # queries efimeras de exploracion
|   `-- YYYY-MM-DD_<topic>.sql
`-- catalog/                 # queries utiles, persistentes
    |-- velas/
    |   |-- <name>.sql
    |   `-- <name>.md        # contexto, supuestos, fecha
    |-- opciones/
    |-- alertas/
    |-- backtest/
    `-- macro/
```

### Convencion de metadata (.md adyacente al .sql)

```markdown
# <name>

## Que devuelve
<descripcion en una frase>

## Cuando usar
<casos donde es util>

## Supuestos
- <ej. asume DB local sincronizada hasta hoy>
- <ej. requiere ticker en universo de 199>

## Parametros
- <listado si la query tiene placeholders>

## Creada
YYYY-MM-DD por <quien>

## Tablas que toca
- <tabla1>
- <tabla2>
```

### Workflow

1. Usuario hace una pregunta al MCP.
2. MCP arma SQL ad-hoc o usa una tool de dominio.
3. Si el resultado es util y la pregunta puede repetirse, el modelo sugiere
   guardar.
4. Usuario confirma. MCP ejecuta `save_query` que escribe `.sql` + `.md` en
   `~/queries-catalog/catalog/<tag>/`.
5. La proxima vez que se haga una pregunta similar, el modelo puede llamar
   `recall_query` antes de generar una nueva.

---

## Despliegue (fase 3)

OBSOLETO el plan original de desplegar en Oracle Cloud contra Railway.
Razon: Plan C (14/5/2026) hizo de la DB local de Windows la fuente de
verdad. Un MCP server en Oracle contra Railway solo veria opciones_snapshot
-- no precios, indicadores, features, scanner ni alertas ML. El MCP server
DEBE correr donde la DB local sea alcanzable.

### MVP -- local en Windows (Fase 3a)

Sin infraestructura cloud. El bot, el orquestador y el MCP server corren en
la notebook Windows; la notebook debe estar encendida para que el bot
responda.

- Bot Telegram con long polling -- no necesita IP publica, ni nginx, ni
  puertos abiertos, ni tunel.
- MCP server por stdio: el orquestador lo lanza como subproceso (igual que
  Gemini CLI hoy). No hace falta `--transport http`.
- DB local directa via el DSN del rol mcp_reader.
- Whitelist de chat_id en variable de entorno.

### Server siempre-online (Fase 3b -- futuro, segun necesidad)

Si se quiere que el bot responda sin depender de la notebook encendida,
el problema a resolver es el acceso a la DB local desde un host remoto:

- Opcion A: tunel (Tailscale o SSH reverse) entre el host del bot y la DB
  local de Windows. El MCP server corre remoto pero consulta la DB local
  por el tunel.
- Opcion B: mover la DB / replicar -- fuera de alcance por ahora.

NO desplegar el MCP server contra Railway: Railway no tiene los datos
(solo opciones). Esa era la premisa del plan viejo y es la que quedo mal.

### Seguridad (aplica a cualquier despliegue)

- Bot Telegram con whitelist estricta de `chat_id`. Sin whitelist el bot no
  responde, ni siquiera a `/start`.
- Rol PostgreSQL `mcp_reader` SELECT-only (primera linea de defensa, ya
  vigente).
- Si en Fase 3b se expone un puerto: nunca a internet abierta -- VPN
  (Tailscale), reverse proxy con auth, o SSH tunnel.
- Logs estructurados a archivo, rotados.

---

## Evaluacion y testing

### Tests unitarios

`tests/mcp_server/` con pytest. Cubrir:
- `safety.py`: que sqlglot rechaza DML/DDL correctamente. Casos: INSERT,
  UPDATE, DELETE, DROP, mixto, comentarios maliciosos, encoding tricks.
- `db/queries.py`: que las queries SQL templates se parametrizan
  correctamente.
- Cada `tools/*.py`: mock del pool de DB, verificar shape del output.

### Suite de evaluacion (10 preguntas)

Antes de declarar fase 1 completa, validar con 10 preguntas reales que el
usuario haria, cada una con respuesta verificable. Ejemplos:

1. Cual fue el ultimo dia habil antes de hoy?
2. Cuantos tickers hay en sector Financials?
3. Que patron de vela tiene BAC el 2026-05-08?
4. Cual es el PCR de volumen de NVDA del ultimo dia disponible?
5. Cuales son los top 5 tickers por alert_score del ML el ultimo dia?
6. Que tickers tienen Z-score de volumen > 2 hoy?
7. La estructura SMC de TSLA en ventana 10 hoy es alcista o bajista?
8. Cuantos contratos call y put se operaron de AAPL el 2026-05-07?
9. La SMA50 de SPY esta sobre o bajo el cierre actual?
10. Cuales son los 3 dias mas recientes de datos en alertas_scanner?

Para cada una, ejecutar manualmente el SQL equivalente y comparar con la
respuesta del MCP. Si las 10 son correctas y consistentes, fase 1 completa.

---

## Anexos

### Anexo A -- DDL completo del rol mcp_reader

Ver seccion "Capa de seguridad -- Rol PostgreSQL". Ejecutar como superuser
una sola vez en local y, eventualmente, en Railway.

### Anexo B -- Configuracion inicial Gemini CLI

Archivo `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "db-consultor": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/juand/path/to/IndicadoresDeStockML",
      "env": {
        "DATABASE_URL": "$MCP_READER_LOCAL_DSN",
        "QUERIES_CATALOG_PATH": "C:/Users/juand/queries-catalog"
      },
      "timeout": 30000
    }
  }
}
```

Donde `MCP_READER_LOCAL_DSN` es una variable de entorno con la cadena de
conexion del rol `mcp_reader` apuntando a la DB local. NO usar el DSN del
usuario `postgres` u otro con permisos de escritura.

### Anexo C -- Pregunta de ejemplo resuelta (BAC)

**Pregunta:** *"Estructura de velas de los ultimos 3 dias para BAC, indicando
tipo y sentimiento, vs PCR de volumen de opciones de BAC para los ultimos 3
dias. Resumen al final."*

**Hoy:** 2026-05-09 (sabado, no habil).

**Tools llamadas (en orden):**
1. `get_last_trading_day()` -> `2026-05-08`
2. `get_ticker_overview(ticker="BAC", days_back=3)` -> dict con todos los
   datos cruzados.

**Respuesta esperada (formato):**

```
Para BAC, ultimos 3 dias habiles:

| Fecha       | Patron vela     | estructura_5 | estructura_10 | Alert ML       | PCR_vol | PCR label |
|-------------|-----------------|--------------|---------------|----------------|---------|-----------|
| 2026-05-06  | engulfing_bull  | +1           | +1            | COMPRA (62)    | 0.84    | NEUTRO    |
| 2026-05-07  | doji            | +1           | +1            | NEUTRO  (51)   | 1.02    | BAJISTA   |
| 2026-05-08  | (sin patron)    |  0           |  0            | VENTA   (38)   | 1.41    | BAJISTA   |

Notas tecnicas:
- estructura_N: -1 bajista / 0 neutral / +1 alcista
- PCR label: <0.7 ALCISTA, 0.7-1.0 NEUTRO, >1.0 BAJISTA
- Alert ML: COMPRA_FUERTE>=70, COMPRA>=60, NEUTRO 50-59, VENTA<50, VENTA_FUERTE<30

Resumen: BAC mostro un patron de envolvente alcista el 06/5 con estructura SMC
alcista en ambas ventanas y respaldo del scanner ML (62, COMPRA). El 07/5
aparece un doji (indecision) y el PCR cruza al lado bajista. El 08/5, sin
patron destacado, estructura SMC neutral, alerta ML cae a VENTA y PCR sube a
1.41 (claramente bajista). Tres dias muestran un cambio gradual de tono
alcista a bajista.
```

### Anexo D -- Lista completa de tools con firma y annotations

Por brevedad, ver seccion "Tools por fase" arriba. La lista canonica esta
en `mcp_server/tools/__init__.py` cuando se implemente.

---

## Relacion con otros modulos del proyecto

- **opciones.md:** define semantica de PCR, IV, OI=T-1. Las tools de
  opciones del MCP respetan los thresholds documentados ahi.
- **estructuras_velas.md:** define patrones de vela y estructuras SMC. Las
  tools `get_price_action` y `get_market_structure` exponen las columnas
  pre-calculadas por los scripts referenciados ahi.
- **indicadores_tecnicos.md:** detalle de las 18 columnas de
  indicadores_tecnicos y de scoring_tecnico (rule-based, vacio en local).
- **scanner_alertas.md:** define alertas_scanner, niveles del scoring ML.
  La tool `get_ml_alert_history` lee esta tabla.
- **ml_modelos.md:** describe el modelo V3 de Random Forest y sus features.
  El MCP no toca el entrenamiento; solo lee outputs en alertas_scanner.
- **datos_pipeline.md:** describe el pipeline que llena las tablas que el
  MCP consume. El MCP no participa del pipeline; depende de el.
- **bt_postgresql_local.md:** define el universo de bt_hist_*. Las tools de
  fase 1.5 de backtest leen estas tablas.
- **AGENDA.md:** este nuevo modulo se sumaria a la lista de tareas activas
  como "Tarea 10 -- MCP Server consultivo".

---

## Backup
- Versiones anteriores de este archivo: ninguna (primera version).
- Documento padre del diseno: conversacion teorica del 2026-05-09.
