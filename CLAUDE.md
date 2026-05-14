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

## Documentos de memoria por dominio

Antes de trabajar en un dominio, leer el .md correspondiente:

- memory/MEMORY.md            : indice general, estado del sistema, patrones criticos
- memory/AGENDA.md             : tareas activas
- memory/opciones.md           : opciones, PCR, IV, Z-scores, 4 intentos diarios
- memory/ml_modelos.md         : Random Forest V3, scoring ML
- memory/datos_pipeline.md     : pipeline diario, cron, recovery_incremental, yfinance_lock
- memory/indicadores_tecnicos.md
- memory/estructuras_velas.md  : SMC, patrones de vela
- memory/scanner_alertas.md    : scanner ML, fixes schema local
- memory/bt_postgresql_local.md
- memory/mcp_server.md         : servidor MCP consultivo

## Reglas no negociables

1. **Fechas**: nunca asumir si una fecha es habil. Usar
   `src/utils/trading_calendar.py` o `scripts/manual/check_fecha.py`.
2. **Encoding cp1252 en Windows**: ASCII puro en strings de codigo, no Unicode
   arrows (`-->`, `=>`, `—`). Telegram messages pueden usar UTF-8.
3. **Antes de modificar cualquier archivo en src/ o scripts/**: leer el .md
   del dominio relevante.
4. **MCP server (`mcp_server/`)**: SOLO consultivo. No modifica DB, no importa
   funciones con side effects, no toca src/ ni scripts/.
5. **NO correr 2 scripts yfinance concurrentes**: usar
   `src/utils/yfinance_lock.acquire()` al inicio. Rate limit es por IP.
6. **NO correr scripts yfinance pre-mercado** (antes de 13:30 UTC) sin entender
   el efecto. Algunos scripts etiquetan data del cierre anterior con la fecha
   de hoy -> datos corruptos.

## Antes de codear

Para cualquier tarea: listar archivos que se van a crear/modificar y pedir
aprobacion antes de implementar. Ver `docs/checklist_recovery_manual.md` para
flujos comunes de recovery.

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

### yfinance rate limit
- En yfinance 0.2.x, `YFRateLimitError` **NO se levanta como excepcion**.
  Yfinance imprime "Failed downloads: ..." a stdout/stderr y devuelve
  DataFrame vacio. Para detectar rate limit hay que capturar stdout/stderr
  con `contextlib.redirect_stdout/redirect_stderr` y parsear el texto.
- Rate limit es **por IP**, no por proceso. Si Oracle cron + Windows manual
  corren simultaneos, suman carga. Usar `src/utils/yfinance_lock.py`.
- `fast_info.last_price` usa `history(period='1y')` internamente = mismo
  endpoint que `yf.download` /v8/finance/chart/. SIN ventaja de rate limit.

### Opciones snapshot — irrecuperabilidad
- yfinance solo expone la chain VIGENTE. Una vez abre el mercado del dia
  siguiente (13:30 UTC durante DST), las strikes/contratos del cierre
  anterior dejan de estar disponibles. **Snapshot debe correr antes**.
- Esquema actual: 4 intentos (23:00, 02:00, 04:00 GH, 06:00 UTC).

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

## Tablas DB principales

Ver `memory/MEMORY.md` para listado completo. Las criticas:
- `precios_diarios` (OHLCV) | `indicadores_tecnicos`
- `features_precio_accion` | `features_market_structure`
- `alertas_scanner` (col: `scan_fecha`, `precio_fecha`)
- `ticker_zscore_diario` | `opciones_zscore_diario`
- `opciones_snapshot` | `opciones_resumen_diario`
- `futuros_diarios` | `indicadores_tecnicos_futuros`
- `features_regimen_macro` | `features_ml` | `features_sector`

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
