# Indicadores y Machine Learning

Sistema de analisis tecnico/ML sobre 199 tickers con PostgreSQL (local + Railway),
pipeline diario, scanner ML, snapshot de opciones US/AR y backtest historico.

## Arquitectura (Plan C, 14/5/2026)

- **LOCAL PostgreSQL** = fuente de verdad para OHLCV, indicadores, features,
  scanner, ML y backtesting historico.
- **Railway PostgreSQL** = SOLO opciones_snapshot (data irrecuperable post-
  mercado siguiente, justifica almacenamiento remoto siempre disponible).
- **Oracle Cloud VM** = cron de snapshot opciones US (3 intentos) + opciones AR.
- **GitHub Actions** = intento 3 (backup IP distinta) del snapshot opciones US.
  Bots Alpaca (3): rediseñados Plan B y ACTIVOS desde 4/6/2026 (Pasos 1-5 hechos).
  Workflows repuntados a scripts/alpaca/ y habilitados. Ver memory/bots_trading.md.
- **Windows** = recovery local manual post-cierre (recovery_incremental.bat).
- **Streamlit** = la app Cloud vieja (app/, indicadoresat) DECOMISIONADA (5/6/2026,
  Paso 7): directorio app/ eliminado del repo; pendiente borrar la app en
  share.streamlit.io (manual). Quedan SOLO apps Streamlit LOCALES: dashboard/
  (informe por ticker) y scripts/reports/app.py (reportes/infografia).

### Bots Alpaca -- rediseño Plan B (4/6/2026, ACTIVO; Pasos 1-5 hechos)
Los 3 bots leian tablas de mercado CONGELADAS en Railway (Plan C apago el pipeline
que las alimentaba ~12/5) -> operaban con señales viejas. Reset a cero + rediseño,
ya implementado y en produccion (paper):
- 3 estrategias: Bot1 ML, Bot2 TECH_SECTOR_v1, Bot3 TECH_SECTOR_OPTIONS_v2 ($100k/bot,
  shares enteras, homologables a FT). v1 vs v2 aisla el aporte del dato de opciones.
- Tabla masticada `senales_bot_diaria` (Railway): el bot "solo opera", lee señales
  pre-computadas (no tablas crudas). Productor `scripts/push_senales_bot.py` la arma
  desde LOCAL (conexion dual) tras ft_run_diario y la sube a Railway (paso final del
  .bat). Rutina nocturna MANUAL (decision del usuario); guard de frescura evita operar
  con masticada rancia -> saltarse una noche es seguro (los bots no tradean ese dia).
- Cerebro de decision COMPARTIDO FT<->Alpaca (Opcion B) en `src/strategies/`
  (scoring/sectorial/ml_scanner, PURO). Adapters en `src/trading/` (senales_adapter
  lee la masticada; ejecucion_bot ejecuta+persiste por bot). Entrypoints en
  `scripts/alpaca/` (bot_ml/bot_tech_sector/bot_options). Viejos scripts/30,31,32
  JUBILADOS (sin borrar).
- PENDIENTE (Pasos 6-7): verificar 1-2 dias paper; dropear crudas de Railway (~330 MB)
  + retencion de opciones; limpiar Streamlit Cloud. Detalle: memory/bots_trading.md.

## Documentos de dominio

Documentacion que existe hoy en docs/:

- docs/mcp_server.md      : servidor MCP consultivo (diseno + estado)
- docs/parametros_mcp.md  : registro de umbrales/parametros de las tools MCP
                            y su interpretacion (para re-tunear a futuro)
- docs/reportes.md        : modulo scripts/reports/ -- generador de PDF e
                            infografias para compartir analisis en X
- docs/estrategias_ft.md  : estrategias de forward testing
- docs/earnings_reaccion.md : vista "Reaccion a balances" del dashboard +
                            tabla earnings_historico. Fecha de anuncio por Q
                            desde Alpha Vantage (la variable que faltaba: no la
                            daban earnings_calendar ni fundamentales), regla del
                            dia 0 (pre/post-market), backfill reanudable/cuota-
                            aware (key free 25/dia). LOCAL-only.
- docs/gestion_universo.md : alta/baja de tickers del universo (Tarea 14).
                            Fuente unica via tabla activos (src/data/universo),
                            CLI universo.py (add/remove/list), insight del backfill
                            2a (ticker nace caliente), dual-write Railway, tabla
                            universo_cambios, point-in-time por construccion.
- docs/bots_alpaca.md     : arquitectura de produccion de los 3 bots Alpaca
                            (Plan B): masticada senales_bot_diaria, cerebro
                            COMPARTIDO src/strategies/, adapters src/trading/,
                            mapeo bot->cuenta->tabla, guard de frescura, decisiones.
- docs/forward_testing/   : detalle de forward testing
- docs/infografia_fundamental.md : spec de diseno de la infografia fundamental
                            para X/redes (formato 4:5, layout 5 bloques, set de
                            indicadores por perfil banco/no-banco, decisiones).
                            Motor: scripts/reports/make_infografia_fundamental.py
- docs/infografia_simple.md : spec de la TERCERA infografia (simple/social, 4:5):
                            1 tarjeta combinada tecnico+fundamental con 2 GRAFICOS
                            (precio+muros mas fuertes; PER del ticker vs PER mediano
                            del sector por trim. calendario) + chips de opciones por
                            plazo + cinta de veredicto. SVG inline (sin matplotlib),
                            sin LLM. Motor: make_infografia_simple.py + boton en el
                            dashboard (vista "Informe por ticker")
- docs/fundamentales_calculo.md : diseno v2 del calculo de ratios fundamentales
                            -- inventario de claves crudas de yahooquery,
                            perfiles banco/no-banco (clasificacion multi-Q +
                            override curado), formulas por perfil. PREVIO a
                            codificar. Validado contra balances oficiales
                            MU/XP/JPM (crudos exactos al millon).
- docs/fuentes_fundamentales.md : evaluacion de DE DONDE traer los balances
                            trimestrales (27/8/2026). Auditoria que encontro
                            97 de 200 tickers sin su ultimo balance + metodo
                            de deteccion por cadencia propia; los 3 modos de
                            falla del estado actual (datos viejos, filas stub
                            parciales, contaminacion silenciosa de medianas);
                            comparativa yahooquery / Alpha Vantage / SEC XBRL
                            con pros, contras y costos; prototipo de
                            normalizacion SEC validado (147 tickers, ~70 Q c/u)
                            y los 3 errores SILENCIOSOS que encontro el cruce.
                            NINGUNA decision de fuente tomada. Prototipo en
                            scripts/oneshot/sec_xbrl_prototipo.py
                            FASES 1-3 IMPLEMENTADAS (29/8/2026). Secciones 13-15
                            con la capa derivada (multiplos diarios, base de
                            SPLIT, percentil estricto), la identidad
                            ProfitLoss-minoritarios que recupero el resultado
                            neto de 8 tickers, y la investigacion de REVENUE:
                            el problema es SEMANTICO, no aritmetico (122 de 147
                            sin ambiguedad, 23 requieren mapeo curado). El
                            encabezado tiene una tabla de atajos por pregunta.
- docs/ficha_empresa.md : CUARTA infografia (5/8/2026) -- tarjeta "presentacion
                            de empresa" fondo OSCURO (4:5). La empresa contra SI
                            MISMA: ultimo Q reportado + variacion INTERANUAL (sin
                            pares/benchmark). Adapta secciones por perfil
                            (banco: ROE/ROTCE/efficiency; no-banco: margenes/ROIC/
                            FCF). Ancla al ultimo Q con income real (evita el stub
                            recien reportado). Motor: make_ficha_empresa.py.
- docs/checklist_recovery_manual.md : flujos de recovery manual
- dashboard/README.md     : spec del Dashboard (informe descriptivo por ticker).
                            v1 + Fase 2 v1 desarrollados 28/5/2026 en rama
                            feature/dashboard: Streamlit local (modos Informe y
                            Radar del dia), export JPG (informe) y PDF (papel de
                            trabajo). Corre bajo el venv. Ver memory/dashboard.md
                            ("Como correrlo / retomarlo").

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

### Universo de tickers -- fuente UNICA: tabla `activos` (18/6/2026, Tarea 14)
- El universo se sirve de `src/data/universo.get_universo()` (lee `activos`
  WHERE activo=TRUE de la DB a la que apunte get_engine; activos vive en local Y
  Railway, sincronizada). Fallback a `config.ALL_TICKERS` si la tabla falla.
- NO usar `config.ALL_TICKERS` en codigo vivo nuevo -> usar get_universo(). Los
  consumidores migrados: snapshot opciones, scanner, refresh fundamentales/pais,
  cron_diario. config.ALL_TICKERS queda solo como fallback/legacy ML-BT.
- FALLBACK EN 3 NIVELES (20/6/2026, incidente Railway): `activos` -> **cache en
  disco** (`data/cache/universo.json`, refrescado en cada lectura exitosa) ->
  `config.ALL_TICKERS` (ultimo recurso). El cache existe porque ALL_TICKERS se
  DESINCRONIZA EN SILENCIO: `universo.py add` escribe la tabla y no toca config
  -> HOOD (alta 18/6) nunca entro a la lista, y con Railway caido el snapshot
  habria capturado 199/200 sin avisar. El cache se auto-cura. Es POR MAQUINA
  (refleja la DB que ve ese host). Sembrarlo: `universo.py cache`.
- Alta/baja: `scripts/manual/universo.py add|remove`. ADD dual-writea `activos`
  a local+Railway (el snapshot corre Oracle->Railway y debe ver el ticker).
  REMOVE = soft delete (activo=FALSE), conserva historia.
- POINT-IN-TIME: los agregados sectoriales (calcular_pcr_sector_plazo etc.) se
  computan por fecha sobre los tickers que TIENEN DATO ese dia -> alta/baja NO
  reescribe la historia. SALVEDAD: cambiar el sector de un ticker existente
  (reclasificacion) SI regrupa su historia (el JOIN toma el sector actual de
  activos) -> setear sector una vez. Detalle: docs/gestion_universo.md.

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

### yahooquery_loader -- barra en curso + timezones mezcladas (futuros)
- `src/utils/yahooquery_loader.py` (download_batch para recovery_incremental).
- yahooquery devuelve, ADEMAS de las barras diarias completas, la barra EN CURSO
  del dia de hoy cuando la sesion esta abierta (notorio en FUTUROS, que cotizan
  ~24h). Esa barra viene como `datetime.datetime` tz-aware (America/New_York)
  mezclada con los `datetime.date` tz-naive de las completas.
- pandas 3.x ya NO coacciona ese mix: `pd.to_datetime(index_mixto)` lanza
  "ValueError: Mixed timezones detected". El loader normaliza cada entrada a su
  fecha local (tz_localize(None)+normalize, sin pasar a UTC) y RECORTA a `<= end`
  (contrato end-inclusivo) -> descarta la barra parcial de hoy. Sin el recorte,
  esa parcial se persistiria con la fecha de hoy y bloquearia la barra real al
  cierre (regla #10). Fix 18/6/2026, commit c3c1892.

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
- **SPOOL en disco (20/6/2026, incidente Railway)**: Railway se detuvo por limite de
  consumo y el snapshot perdia el dato: `persistir_filas()` escribia por ticker y,
  con la DB caida, cada ticker levantaba excepcion y la chain YA DESCARGADA se
  descartaba. Ahora `33_opciones_snapshot.py` vuelca el crudo a
  `data/opciones_spool/opciones_YYYY-MM-DD.csv.gz` **ANTES** de intentar la DB
  (`src/utils/opciones_spool.py`, modulo puro, escritura en streaming). Un fallo de
  DB ya NO aborta la captura. INVARIANTE: **archivo en el spool = dato pendiente de
  persistir**; si la DB tomo todo, el spool se borra solo. Recuperacion:
  `scripts/manual/replay_opciones_spool.py` (upsert idempotente). Alerta Telegram
  automatica cuando queda spool pendiente (antes el fallo era SILENCIOSO).
  Desactivable con `--no-spool` (no recomendado).

### FT asincronico -- usar `fecha_datos`, NO `fecha_entrada` (21/7/2026)
- Los bots FT deciden y ejecutan con el OHLCV del **ultimo cierre disponible**,
  no con el del dia en que corren. Es la convencion del proyecto, no un bug.
- **El desfase NO es fijo**: depende de cuan rancia estaba `precios_diarios`
  cuando corrio el bot (rutina nocturna manual). Medido sobre 1.811 ops:
  16.7% mismo dia, 73.2% 1 dia, 6.2% 2 dias, 3.4% 5 dias, 0.6% 6 dias.
- `ft_operaciones.fecha_entrada`/`fecha_salida` = fecha de REGISTRO.
  `fecha_datos`/`fecha_datos_salida` = fecha del OHLCV usado. **Para cruzar con
  precios_diarios, indicadores_tecnicos o cualquier tabla de mercado hay que
  usar `fecha_datos`**; con la de registro se lee el dia equivocado, en silencio.
- Las escribe `ft_utils.obtener_fecha_datos()` (MAX(fecha) del ticker). Los bots
  no la manejan: `ft_utils` es el UNICO lugar que escribe `ft_operaciones`.

### Splits -- precios_diarios NO se re-ajusta hacia atras (21/7/2026)
- El pipeline diario solo trae los dias NUEVOS (ya ajustados por Yahoo). Cuando
  un ticker hace split, la historia previa queda en la escala VIEJA -> la serie
  del ticker queda **partida en dos** y todo lo que la cruza se rompe: SMA/RSI/
  ATR, features de ML, y las posiciones abiertas de FT.
- **Incidente 21/7/2026**: KLAC 10:1 (11/6) y CRWD 4:1 (30/6) sin aplicar. Ocho
  operaciones de FT se cerraron al precio post-split con la cantidad pre-split
  -> -12.709 USD de perdidas FICTICIAS y ~3.9 puntos de retorno de menos en 4
  estrategias, con stops disparados por un derrumbe que no existio.
- Herramienta: `scripts/manual/splits.py detectar|corregir`. Deteccion en 2
  etapas: (1) barrido local de variaciones diarias > umbral, barato; (2)
  verificacion contra Yahoo de los candidatos. **La etapa 2 es imprescindible**:
  un movimiento real y un split se ven IDENTICOS en la etapa 1 (CAR -38%/-48% y
  FISV -44% son movimientos REALES, confirmado).
- **El ratio observado en el salto NO es el ratio del split** salvo que el
  precio no se haya movido ese dia: KLAC dio 8.856 siendo 10:1 porque ademas
  subio +12.9% real. No usar el ratio del salto como filtro.
- **Un ratio constante NO alcanza para declarar split: tiene que ser ademas un
  ratio PLAUSIBLE** (2,3,4,5,10,20...). ORCL (0.9841) y DELL (0.9744) dan
  constante contra Yahoo y NO son splits -- corregirlos con divisor habria roto
  datos sanos. El script los reporta como "DISCREPANCIA (no split)". Regla: el
  detector recomienda una accion DESTRUCTIVA, asi que un falso positivo cuesta
  mas que un falso negativo.
- El detector cubre splits forward (caida) **e inversos** (suba).
- **Correccion por DIVISOR, no re-descargando**: `precios_diarios` guarda el
  close CRUDO tal como se bajo y nunca se re-ajusta por dividendos; el `Close`
  de yahooquery viene ajustado por split Y dividendos (KLAC da 9.8369 en vez de
  10; CRWD, sin dividendos, da 4.0000 exacto). Sobrescribir dejaria esos tickers
  en otra base que el resto del universo.
- **AUTOMATIZADO (21/7/2026)**: `splits.chequeo_diario()` corre dentro de
  `recovery_incremental.py` (target=local), JUSTO DESPUES de traer los precios
  nuevos, que es cuando el split se manifiesta. Barre los ultimos 7 dias
  (query local, gratis) y solo sale a Yahoo si hay candidatos -- en un dia
  normal, cero. Alerta por Telegram con el comando de correccion listo.
  Standalone: `splits.py chequeo [--dias N] [--sin-alerta]`.
- **DETECTA Y AVISA, NO CORRIGE.** Corregir reescribe precios_diarios (fuente de
  verdad) y el detector ya dio falsos positivos una vez. La correccion la
  dispara una persona con `splits.py corregir`.
- OJO al integrarlo en otro script: `chequeo_diario(usar_lock=False)` cuando el
  proceso YA tiene el lock de yfinance (caso recovery_incremental) -- pedirlo de
  nuevo aborta el proceso entero.

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
| `scripts/manual/poblar_opciones_yq.bat` | Carga manual opciones US via yahooquery (UNA pasada) |
| `scripts/manual/recover_opciones_tickers.py` | Recovery quirurgico de tickers especificos |
| `scripts/manual/replay_opciones_spool.py` | Reinyecta a la DB los snapshots de opciones que quedaron en disco por DB caida (`--list` / `--dry-run` / `--target local\|railway`). Upsert idempotente |
| `scripts/manual/retencion_opciones_railway.py` | Purga opciones_snapshot en RAILWAY dejando los ultimos 10 dias (causa raiz incidente 20/7: +580 MB/mes sin retencion). SOLO borra fechas verificadas replicadas en local (compara COUNT por fecha). Incluye VACUUM FULL (DELETE solo no devuelve disco). Encadenado al final de sync_opciones_railway_to_local.bat |
| `src/utils/opciones_spool.py` | Red de seguridad en disco del snapshot de opciones (modulo puro, .csv.gz en streaming) |
| `scripts/sync_local.bat` | Sync Railway -> Local |
| `scripts/sync_to_railway.bat` | Sync Local -> Railway (paso a paso) |
| `scripts/manual/universo.py` (+ `.bat`) | Alta/baja de tickers del universo (Tarea 14). `add` (backfill 2a + indicadores/features/z-scores/fundamentales + dual-write activos local+Railway + log), `remove` (soft delete + guard posiciones FT), `list`. Solo acciones. Ver docs/gestion_universo.md |
| `src/data/universo.py` | Fuente UNICA del universo: get_universo()/get_universo_sectores() leen de `activos` (fallback config.ALL_TICKERS). Lo usan snapshot/scanner/refresh/cron |
| `scripts/migrations/clean_ticker_fantasma_se.py` | Limpieza generica ticker fantasma |
| `scripts/oneshot/clean_railway_may12.py` | One-shot one-off (archivado en scripts/oneshot/) |
| `scripts/manual/check_fecha.py` | CLI valida dia habil NYSE |
| `scripts/manual/ft_run_diario.bat` | Corre los 10 bots de Forward Testing en local + reporte HTML + push senales_bot_diaria |
| `scripts/manual/splits.py` (detectar/corregir) | Deteccion y correccion de splits no aplicados en precios_diarios. 2 etapas (barrido local + verificacion Yahoo). Corrige por divisor y recomputa indicadores/features/z-scores. Ver "Splits" en Patrones criticos |
| `scripts/forward_testing/ft_compute_equity.py` | Reconstruye la equity MARCADA A MERCADO (`ft_equity_diaria`) desde ft_operaciones + precios_diarios. Idempotente, `--rebuild`/`--check`. Control de cuadre del cash contra ft_estrategias |
| `src/utils/ft_metricas.py` | Modulo PURO de metricas de riesgo (max DD, Sharpe con IC95%, Sortino, IR, beta) y de trade (expectancy, profit factor, payoff). Sin DB ni config |
| `scripts/push_senales_bot.py` | Productor de la tabla masticada senales_bot_diaria (Plan B). Lee LOCAL (tecnico/scanner/PCR_VOL), UPSERT a RAILWAY. Conexion dual. Hermano de FT (paso final de ft_run_diario.bat). Standalone via push_senales_bot.bat |
| `scripts/alpaca/bot_ml.py` / `bot_tech_sector.py` / `bot_options.py` | Los 3 bots Alpaca Plan B (entrypoints GH Actions). Leen la masticada, deciden con el cerebro src/strategies/, ejecutan via src/trading/ejecucion_bot. `--dry-run` / `--ignore-frescura`. Ver docs/bots_alpaca.md |
| `src/strategies/` | Cerebro de decision COMPARTIDO FT<->Alpaca (PURO): scoring (calcular_score_tecnico), sectorial (v1/v2), ml_scanner |
| `src/trading/senales_adapter.py` / `ejecucion_bot.py` | Adapters Alpaca: data (masticada->cerebro) + ejecucion (alpaca_client + posiciones_bot*/operaciones_bot*) |
| `scripts/forward_testing/ft_reporte_html.py` | Reporte HTML autocontenido de FT (reportes/ft_reporte.html) |
| `scripts/refresh_earnings_calendar.py` | Refresh earnings_calendar desde Nasdaq (cron Oracle semanal) |
| `scripts/refresh_earnings_historico.py` | Puebla earnings_historico (fecha de anuncio por Q) desde Alpha Vantage. REANUDABLE y cuota-aware (key free 25/dia, 5/min): `--backfill` (llena faltantes+desactualizados, <=20/corrida), sin flags = incremental, `--ticker X` (alta), `--status`, `--target local\|railway`. Backfill inicial corre en Oracle->Railway (cron temporal); incremental en Windows (target local). Ver docs/earnings_reaccion.md |
| `dashboard/earnings_reaccion.py` | Vista "Reaccion a balances": ventana simetrica pre+post (N ruedas por lado, 1-10) alrededor del balance. 3 paneles (precio USD, precio %, volumen x prom 50). Filtros por anio y trimestre (Q1-Q4). Dia 0 ajustado por pre/post-market |
| `scripts/manual/refresh_fundamentales.bat` | Refresh fundamentales (income/balance/cashflow/valuation) desde yahooquery. LOCAL-only, manual. ~3.5 min. Encadena 5 pasos derivados: ratios -> ticker_pais -> vs_sector -> **multiplos_px -> vs_sector --valuacion-px** (los 2 ultimos agregados 27/8/2026: sin ellos el trimestre nuevo queda sin `*_px` y el dashboard muestra la valuacion vacia). `set REFRESH_NO_PAUSE=1` para correrlo desatendido |
| `scripts/refresh_fundamentales.py` | Motor del refresh fundamentales (4 tablas, 8 Q, UPSERT con restatements) |
| `scripts/manual/refresh_fundamentales_sec.bat` (+ `scripts/refresh_fundamentales_sec.py`) | Refresh de la fuente SEC XBRL (PARALELA a yahooquery, LOCAL-only, ~147 tickers USA). INCREMENTAL: consulta `submissions` (~164 KB) y solo baja `companyfacts` (~4 MB) si cambio el accession del ultimo 10-Q/10-K -> sin balances nuevos mueve ~24 MB en vez de ~522 MB. REQUIERE `SEC_USER_AGENT` en el .env (SEC devuelve 403 sin User-Agent con mail de contacto). `--solo-normalizar` / `--forzar` / `--tickers` / `--dry-run`. ENCADENA 2 pasos derivados (refresh_acciones_circulacion + compute_sec_multiplos completo), solo si el refresh anduvo; `set SEC_NO_DERIVADOS=1` los saltea (necesario con --solo-normalizar, que es offline). Ver docs/fuentes_fundamentales.md |
| `src/utils/sec_xbrl.py` | Normalizador PURO de SEC XBRL -> serie trimestral (stdlib, sin DB/red). Resuelve sinonimos por concepto, tags que cambian dentro de la misma empresa, desacumulacion YTD (Q2=H1-Q1, Q3=9M-H1, Q4=FY-9M) y restatements. Etiqueta fiscal_year/fiscal_quarter reales. `hasta_filed` = point-in-time. Emite avisos como red contra el error silencioso. Incluye la identidad `net_income = ProfitLoss - minoritarios` para los 8 filers que no tagean NetIncomeLoss (MA/CAT/SCCO/AVAV/AVGO/F/FCX/AMT): solo rellena huecos, exige el hecho de minoritarios EN ESE PERIODO (nunca asume cero) y cruza contra ...AvailableToCommonStockholders. Acepta `tags_curados={concepto: tag}` por ticker (el tag curado REEMPLAZA la lista de sinonimos, no se antepone: preferimos el hueco visible al numero mezclado invisible). Aviso `mezcla_en_ejercicio` = los Q de un mismo ejercicio salieron de tags distintos Y sus anuales difieren; si coinciden son sinonimos y calla (109 de 126 mezclas son inocuas). Ver docs/fuentes_fundamentales.md sec. 14 y 16 |
| `src/utils/fundamentales_ttm.py` | TTM rodante PURO sobre la serie SEC + as-of por `filed_primero`. Rechaza ventanas que cruzan un hueco (un rolling(4) sobre serie con huecos suma 5-6 trimestres en silencio). Deriva ebitda/fcf/net_debt/bvps |
| `src/utils/sec_acciones.py` | Serie POINT-IN-TIME de acciones desde la portada `dei` (nunca re-expresada por splits). Descarta errores de unidad y picos; invariante `fecha > filed` |
| `src/utils/acciones_series.py` | Combina yahooquery (base de split ACTUAL) + SEC (base de su momento) y VALIDA que coincidan antes de mezclarlas. Yahoo manda donde llega; SEC solo extiende hacia atras; ESCALON nunca interpolacion; la discrepancia AVISA, no corrige |
| `scripts/refresh_acciones_circulacion.py` | Puebla `acciones_circulacion` (yahooquery + extension SEC validada). LOCAL-only. Usa yfinance_lock |
| `scripts/compute_sec_multiplos.py` | Serie DIARIA de multiplos sobre la fuente SEC (`fundamentales_sec_multiplos_d`). Capa derivada pura y recomputable. Percentil trailing ESTRICTO (exige ventana llena en tiempo); `--percentil-permisivo` la afloja. `--incremental` (paso diario en recovery_incremental) escribe solo la rueda nueva pero CALCULA la serie entera -- el percentil es rodante de 756 ruedas; no propaga restatements hacia atras, de eso se encarga la corrida completa del .bat. Motor expuesto como `computar()` |
| `src/data/sec/tags_curados.py` | Mapeo CURADO ticker -> tag XBRL (modulo de DATOS, sin logica). Los 23 tickers donde "revenue" es ambiguo porque dos tags valen cosas distintas. 10 decididos por ARBITRAJE contra yahooquery, 13 por CRITERIO contable (yahooquery tiene filas stub ahi). Cada entrada anota su base y la cifra 2025 |
| `scripts/manual/sec_avisos.py` | UNICO lector de `fundamentales_sec_avisos`. Ordena por SEVERIDAD y no por volumen: DEFECTO / HUECO / SOSPECHA / info. `--defectos` / `--detalle` / `--ticker` / `--alertar` (Telegram, SOLO defectos) |
| `scripts/oneshot/revenue_tags_reporte.py` | Regenera el diagnostico de ambiguedad de revenue: por ejercicio, cuanto da cada tag candidato, que tags usaron los 4 Q y cuanto da yahooquery de arbitro. Solo lee (cache SEC + DB). Correrlo cuando `mezcla_en_ejercicio` senale un ticker nuevo |
| `src/data/sec/client.py` | Descarga de data.sec.gov con cache en disco (`data/sec_cache/`, gitignoreado). REGLA: nada en `src/data/sec/` importa del lado de trading. OJO: SEC corta los loops de `curl` (conexion nueva por pedido) -> usar `requests.Session` con keep-alive |
| `scripts/compute_fundamentales_ratios.py` | Computa fundamentales_ratios_q (capa derivada, pura, recomputable sin re-fetch). Encadenado al refresh .bat |
| `scripts/compute_multiplos_px.py` | Recalcula PER/PB/PS/EV-EBITDA *_px con el cierre del dia (numerador=precio hoy, denominador TTM). DIARIO via recovery_incremental Y al final de refresh_fundamentales.bat. Ver docs/fundamentales_calculo.md |
| `src/utils/multiplos_px.py` | Logica pura del recalculo de multiplos al cierre (sin DB) |
| `scripts/refresh_ticker_pais.py` | Trae country/region por ticker (yahooquery assetProfile) -> tabla ticker_pais. Encadenado al refresh .bat |
| `scripts/manual/refresh_industria.py` | Rellena activos.industry desde yahooquery assetProfile (antes 62% NULL desde yfinance; ahora 200/200). Dual-write local+Railway, `--status`/`--dry-run`, avisa si un sector difiere pero NO lo toca. Usa yfinance_lock |
| `scripts/compute_perfiles_carteras.py` (+ `.bat`) | Perfilado de carteras (Fase 3): corre el motor puro perfil_metricas (ATR% multi-TF+beta+drawdown) + perfil_riesgo (clasificacion data-driven por percentil del universo, perfil puro) sobre los 200 y UPSERT en perfiles_ticker. MENSUAL, LOCAL-only. `--dry-run`/`--fecha`. Ver docs/perfiles_carteras.md |
| `src/utils/perfil_riesgo.py` | Clasificador PURO de perfil de riesgo: percentil por eje -> composite -> caja por cuartil (perfil=comportamiento); sector = contexto (caja_base) + flag excepcion. perfilar_universo(rows) necesita el universo entero |
| `scripts/compute_fundamentales_sector.py` | Computa fundamentales_ticker_vs_sector (ticker vs mediana de pares regionales; parametrizable --regions). Encadenado al refresh .bat |
| `scripts/oneshot/create_fundamentales_ratios_table.py` | Crea la tabla fundamentales_ratios_q (one-shot, archivado en scripts/oneshot/) |
| `scripts/oneshot/create_fundamentales_sector_table.py` | Crea la tabla fundamentales_ticker_vs_sector (one-shot, archivado en scripts/oneshot/) |
| `scripts/oneshot/create_earnings_calendar.py` | Crea la tabla earnings_calendar (one-shot, archivado en scripts/oneshot/) |
| `scripts/oneshot/create_fundamentales_tables.py` | Crea las 4 tablas fundamentales_* (one-shot, archivado en scripts/oneshot/) |
| `scripts/oneshot/migrate_ft_railway_to_local.py` | Migracion puntual ft_* Railway -> local (one-shot, scripts/oneshot/) |
| `scripts/reports/make_infografia.bat <TICKER>` | Infografia PNG para X (datos del MCP, sin LLM). Ver docs/reportes.md |
| `scripts/reports/build_yaml.bat <TICKER>` + `make_report.bat <yaml>` | Reporte PDF detallado con narrativa del LLM |
| `scripts/reports/make_ficha_empresa.py <TICKER>` | Ficha "presentacion de empresa" PNG (fondo oscuro 4:5): ultimo Q reportado + variacion interanual, la empresa contra si misma (sin pares). Adapta por perfil banco/no-banco. Ver docs/ficha_empresa.md |

## Tablas DB principales

Listado completo: usar `describe_table` del MCP o `information_schema`.
Las criticas:
- `activos` (ticker PK, nombre, sector, industry, activo BOOL, modelo_asignado) --
  FUENTE UNICA del universo. activo=TRUE = universo vivo. Leida via
  src/data/universo.get_universo(). En local Y Railway, sincronizada (dual-write
  en alta/baja). 200 tickers (HOOD incorporado 18/6).
- `universo_cambios` -- log de alta/baja (ticker, accion ALTA/BAJA, fecha, sector,
  motivo, detalle JSONB). local+Railway. Auditoria/reproducibilidad point-in-time.
- `precios_diarios` (OHLCV) | `indicadores_tecnicos`
- `features_precio_accion` | `features_market_structure`
- `alertas_scanner` (col: `scan_fecha`, `precio_fecha`)
- `ticker_zscore_diario` | `opciones_zscore_diario`
- `opciones_snapshot` | `opciones_resumen_diario` -- en RAILWAY, opciones_snapshot
  tiene RETENCION de 10 dias (purga verificada post-sync, 20/7/2026); la historia
  completa vive en LOCAL. Sin retencion crecia ~19 MB/dia y detuvo Railway por
  limite de consumo (incidente 20/7).
- `opciones_sector_zscore_diario` (PCR_vol+vol agregados por sector, z-score)
- `opciones_pcr_plazo_diario` (PCR vol/OI + muros S/R por ventana corto/medio/largo,
  por ticker; fuente src/utils/opciones_plazo.py)
- `opciones_sector_pcr_plazo_diario` (PCR sectorial por ventana + z-score)
- `indicadores_tecnicos_1w` (RSI/MACD semanal) -- CONGELADA 2026-04-02: pipeline
  semanal (scripts 23-30) deprecado 28/5/2026 (Plan C), movido a scripts/legacy_1w/.
  El timeframe semanal se calcula AL VUELO desde precios_diarios (dashboard,
  mtf_context y el MCP get_ticker_sintesis). Ningun flujo vivo la lee ya:
  src/utils/weekly_tf.py es la fuente unica del RSI/MACD semanal (29/5/2026).
- `futuros_diarios` | `indicadores_tecnicos_futuros`
- `features_regimen_macro` | `features_ml` | `features_sector`
- `earnings_calendar` (ticker PK, earnings_date DATE NULL; refrescada semanal
  desde Nasdaq por `refresh_earnings_calendar.py`)
- `earnings_historico` (LOCAL) -- fecha de anuncio de cada balance por trimestre
  (ticker, fiscal_period_end, announcement_date, report_time pre/post-market).
  PK (ticker, fiscal_period_end); JOIN con fundamentales_*_q por esa clave.
  Fuente Alpha Vantage EARNINGS (backfill reanudable/cuota-aware, key free
  25/dia). Base de la vista "Reaccion a balances" del dashboard. La variable que
  faltaba: earnings_calendar solo tiene la proxima fecha y fundamentales tiene
  el CIERRE fiscal, no el anuncio. Ver docs/earnings_reaccion.md
- `fundamentales_income_q` | `fundamentales_balance_q` | `fundamentales_cashflow_q`
  | `fundamentales_valuation_q` -- 4 tablas de analisis fundamental trimestral,
  ultimos 8 Q por ticker (income/balance/cashflow + ratios PE/PB/PS/PEG/EV-EBITDA).
  Schema wide con ~12-15 cols dedicadas + raw_json JSONB. PK natural
  (ticker, fiscal_period_end). LOCAL-only (Plan C: yahooquery sirve historicos
  recuperables, no necesita Railway). Refresh manual via
  scripts/manual/refresh_fundamentales.bat (~3.5 min full universo).
  Multi-moneda: reporting_currency por fila (170 USD + 29 monedas locales en
  ADRs); filtrar por USD o normalizar via FX para analisis cross-ticker.
- `fundamentales_sec_q` | `fundamentales_sec_avisos` | `fundamentales_sec_ingesta`
  (LOCAL) -- fuente SEC XBRL, **PARALELA a yahooquery** (28/8/2026). Las dos
  conviven; ningun consumidor las lee todavia y las `fundamentales_*_q` de
  yahooquery siguen intactas. `fundamentales_sec_q`: 1 fila por (ticker,
  period_end), 31 columnas de concepto + fiscal_year/fiscal_quarter + `origen`
  JSONB (rastro de auditoria: que tag produjo cada numero y si se derivo por
  desacumulacion). 147 tickers USA, 4.781 filas, ventana 2018-2026 (5 anios + runway
  del TTM; el cache en disco conserva 2007+ y es re-derivable con --desde). UNA tabla y no
  cuatro porque SEC no organiza por estado contable: publica hechos sueltos.
  Las columnas se generan DESDE src/utils/sec_xbrl.py para que el esquema no
  se desincronice. El point-in-time NO se almacena: se re-deriva con
  `normalizar(hasta_filed=...)` sobre el cache. Ver docs/fuentes_fundamentales.md
  CALIDAD MEDIDA (29/8/2026, control = la suma de 4 Q contra el anual que
  publico la empresa): net_income 99,6% | operating_income 98,6% | cfo 97,0% |
  revenue 87,8%. El revenue fallaba en 30 tickers por MEZCLA DE TAGS dentro
  del mismo ejercicio (siempre el Q4, que sale del 10-K y viene etiquetado
  distinto que los 10-Q). NO es un problema aritmetico y ningun algoritmo lo
  resuelve: 122 de 147 tickers no tienen ambiguedad, 23 necesitan un mapeo
  CURADO ticker -> tag. Las 4 salidas automaticas ya descartadas (resta contra
  el anual, `frame`, solo-publicado, pasarse al anual): doc sec. 15.
  RESUELTO 29/8/2026 (doc sec. 16): el mapeo vive en src/data/sec/tags_curados.py
  y lo aplica el refresh. Queda 1 mezcla consecuente en revenue (LNC, ejercicio
  2018, borde de la transicion ASC 606 y fuera de la ventana 2021+ de la capa
  derivada). P/S de SEC ya es consumible.
  **PENDIENTE de la misma clase: `d_and_a` mezcla de forma consecuente en 79 de
  147 tickers** -- sus sinonimos NO son equivalentes (Depreciation a secas no es
  DepreciationDepletionAndAmortization). Alimenta el EBITDA, asi que
  **EV-EBITDA de SEC sigue siendo el multiplo a consumir con cuidado**; explica
  su mal acuerdo contra yahooquery (mediana 9,04%, p90 38,7%). Verlo con
  `python scripts/manual/sec_avisos.py --defectos`.
- `acciones_circulacion` (+ `acciones_circulacion_validacion`, LOCAL) --
  acciones en circulacion por (ticker, fecha) en base de split **ACTUAL**, que
  es la unica apareable con `precios_diarios` (que se corrige retroactivamente
  por divisor). Fuente primaria yahooquery `OrdinarySharesNumber`; se extiende
  hacia atras con la portada SEC SOLO en los tickers donde se VALIDA que no
  cambiaron de base. La tabla `_validacion` guarda el veredicto por ticker
  (extendido, ratio_min/max, motivo). Motor puro: src/utils/acciones_series.py.
  200 tickers / 2.379 puntos: 101 arrancan en 2021, 83 en 2022, 16 en 2023+.
- `fundamentales_sec_multiplos_d` (LOCAL) -- serie DIARIA de multiplos sobre la
  fuente SEC. Capa DERIVADA y recomputable (funcion de fundamentales_sec_q +
  acciones_circulacion + precios_diarios). 1 fila por (ticker, rueda):
  market_cap, EV, PER/PB/PS/EV-EBITDA, fcf_yield + percentil TRAILING de cada
  uno dentro de su propia historia ("caro vs si misma"). 152.054 filas / 144
  tickers / 2021+. **Todos los multiplos salen de AGREGADOS** (market_cap /
  net_income_ttm, etc.), nunca de magnitudes por accion: SEC re-expresa lo "por
  accion" ante un split y precios_diarios tambien, pero con horizontes
  distintos -- `eps_ttm` y BVPS se sacaron a proposito. `shares_dias` = la
  antiguedad del conteo (el error del escalon anual es mediana 0,24% pero p99
  11,35%, y sobreestima). El percentil es ESTRICTO: NULL si la ventana de 756
  ruedas no esta llena en tiempo. El limitante es `precios_diarios`, no SEC:
  solo 84 tickers llegan a 756 ruedas.
- `fundamentales_ratios_q` -- capa DERIVADA (funcion pura de las 4 raw,
  recomputable sin re-fetch). 1 fila por (ticker, fiscal_period_end). Vista
  PARALELA/DESCRIPTIVA del fundamental: NO se mezcla con el score tecnico ni
  los bots. Crecimiento en base TRIMESTRAL (QoQ vs Q-1, YoY vs Q-4);
  rentabilidad/retornos/margenes en base TTM. Incluye PER/P-B/P-S/EV-EBITDA,
  BVPS, BPA (eps_q/eps_ttm), ROE/ROA/ROIC, margenes bruto/op/neto + deltas YoY,
  opex/revenue, FCF (ttm/margen/growth), current_ratio/working_capital/D-E/
  net_debt. ROIC standard: NOPAT_ttm=EBIT_ttm*(1-tax) / (deuda+equity-caja),
  NULL si pretax<=0 o sin EBIT. sector/industry denormalizados de activos
  (habilita GROUP BY sector). Los RATIOS son inmunes a escala/moneda; los
  ABSOLUTOS (BVPS/eps/fcf/working_capital/net_debt/*_ttm) quedan en moneda de
  reporte (no comparables cross-ticker sin FX). Caveat ADR: BVPS (por accion
  ordinaria) y EPS de Yahoo (por ADR) estan en bases distintas -- usar ratios
  para comparar. ~4 tickers semestrales (HMY/RIO/UL/VOD) sin Q -> sin ratios.
  Compute: scripts/compute_fundamentales_ratios.py (encadenado al refresh .bat;
  validacion de escala con WARN).
  v2 (1/6/2026 -- perfiles banco/no-banco, ver docs/fundamentales_calculo.md):
  columna `profile` por estructura contable SOSTENIDA (multi-Q: gross 0/N y
  opinc 0/N + NII -> financiero) + override curado. 18 financieros (incl. XP por
  override) / 177 no. Financieras: margenes industriales/ROIC/liquidez/WC/FCF
  = NULL; en su lugar rotce_ttm (NI_common/TangibleBookValue) y
  efficiency_ratio_ttm ((rev-pretax)/rev, aprox). NIM descartado (yahooquery sin
  cartera de prestamos). Ambos perfiles usan CommonStockEquity (NO equity total)
  y NetIncomeCommonStockholders en ROE/BVPS/ROIC -> corrige preferentes. As-of
  join del balance (mas reciente <= fecha income) -> ROE no queda NULL si el
  balance del Q exacto no salio. Validado vs balances OFICIALES MU/XP/JPM (JPM
  BVPS 128.38 = oficial; XP ROE 22.3% vs ROAE oficial 21.7%). PER/PB/PS/EV-EBITDA
  siguen de Yahoo (validados OK).
- `ticker_pais` (ticker PK, country, region, fetched_at) -- pais real de cada
  ticker desde yahooquery assetProfile.country, con region derivada
  (USA/Europa/China/Resto via REGION_MAP en refresh_ticker_pais.py). Base para
  el comparativo sectorial regional (reporting_currency es proxy imperfecto:
  empresas extranjeras que reportan en USD caerian mal clasificadas). LOCAL-only.
  Distribucion: USA 146, Resto 25, Europa 21, China 6 (+1 sin pais: FISV).
- `fundamentales_ticker_vs_sector` -- comparativa de cada metrica del ticker vs
  la mediana de sus PARES de la MISMA region (no mezclar prima de riesgo pais).
  Formato LONG (1 fila por ticker x metrica; 10 metricas: PER/P-B/P-S/EV-EBITDA
  + ROE/ROA/ROIC/net_margin/operating_margin + revenue_yoy). Columnas: value,
  peer_median/p25/p75, vs_median_pct, percentile, peer_n, peer_basis, low_sample.
  Politica de peer-set (umbral N=5): (1) bucket (sector,region) n>=5 ->
  basis='region'; (2) no-USA en bucket chico con sector USA n>=5 ->
  basis='usa_fallback' (flag honesto); (3) resto -> basis='none' (leyenda
  "pocas empresas"). Resultado: 169 tickers region / 22 usa_fallback / 4 none
  (REITs+utility USA). Snapshot del ultimo Q de cada ticker (earnings
  escalonados -> fechas difieren unos dias). Motor PARAMETRIZABLE por region
  (--regions USA,Europa) para curaduria del usuario. Compute:
  scripts/compute_fundamentales_sector.py (encadenado al refresh .bat).
  Consumo: vista "Analisis Financiero" del dashboard (dashboard/financiero.py:
  modo "Por ticker" = 4 bloques valuacion/calidad/crecimiento/solvencia con
  coloreo vs mediana de pares; modo "Screener sectorial" = tabla ordenable por
  sector+region con fila mediana). Solo lectura, no recalcula.
- `ft_*` (5 tablas Forward Testing: estrategias, operaciones, candidatos_diarios,
  metricas_diarias, posiciones_diarias) -- LOCAL es fuente de verdad
- `ft_equity_diaria` (LOCAL) -- equity curve MARCADA A MERCADO por estrategia,
  capa DERIVADA (funcion pura de ft_operaciones + precios_diarios, recomputable).
  Existe porque `ft_metricas_diarias.capital_total` esta a COSTO de entrada
  (`capital_inmovilizado = SUM(capital_entrada)`) y solo se mueve al cerrar una
  operacion: es una curva de PnL realizado, no una equity curve -> sobre ella el
  max drawdown de posiciones abiertas es INVISIBLE y toda metrica de riesgo
  subestima el riesgo. Solo dias habiles NYSE, sin huecos (la rutina nocturna es
  manual y le faltaba el 34% de los dias). **Las metricas de riesgo se calculan
  SOLO desde aca.** `ft_metricas_diarias` queda intacta como log operativo.
  La escribe `ft_compute_equity.py`. Ver docs/forward_testing/METRICAS.md
- `senales_bot_diaria` (RAILWAY) -- tabla MASTICADA Plan B para los 3 bots Alpaca
  (Tarea 16). 1 fila por (ticker, fecha), ~18 cols, PK (ticker, fecha). El bot
  "solo opera": lee senales pre-computadas, no las crudas. Columnas: close, sector,
  alert_nivel/alert_score (ML), sma21/50/200/rsi14/macd/macd_signal/atr14 (tecnico),
  pcr_score(0-3)/pcr_valido/pcr_corto/medio/largo (opciones PCR_VOL). La produce
  scripts/push_senales_bot.py desde LOCAL (conexion dual) y la UPSERTea a Railway,
  como paso final de ft_run_diario.bat. Es la unica tabla de mercado US que los
  bots leeran de Railway (habilita el cleanup de crudas, ver bots_trading.md).
- `llm_uso_tokens` (LOCAL) -- registro de consumo del chat del dashboard (vista
  "Consultas (IA)", orquestador src/agent). 1 fila por consulta: tokens_entrada/salida
  REALES (usage_metadata de Gemini), modelo, n_rondas, tools, pregunta, fecha y `usuario`
  (default 'local'; columna pensada para cuotas multiusuario a futuro). LOCAL-only
  (Plan C: log de frontend local). La escribe el dashboard con el engine local normal
  (NO el rol mcp_reader). Creada por scripts/oneshot/create_llm_uso_tokens_table.py.
- `perfiles_ticker` (LOCAL) -- snapshot del PERFIL DE RIESGO de cada ticker para
  segmentar carteras (Conservadora/Moderada/Arriesgada/Especulativa). Capa DERIVADA
  y recomputable (funcion de precios_diarios + futuros ES + activos via el motor puro
  perfil_metricas/perfil_riesgo). PK (ticker, fecha) = historia mensual (habilita ver
  DRIFT de caja). El perfil = COMPORTAMIENTO cuantitativo puro (percentil composite de
  ATR%_w/m+beta+drawdown dentro del universo -> caja por cuartil); el sector es CONTEXTO
  (caja_base, prior top-down) + flag `excepcion` (comportamiento se despega 2+ cajas del
  sector). La pobla scripts/compute_perfiles_carteras.py, cadencia MENSUAL (no va en el
  recovery diario). LOCAL-only (Plan C). Ver docs/perfiles_carteras.md.

## Flujo de recovery manual (caso comun: Oracle cron fallo)

```
1. status.bat               (ver Railway: que dias faltan)
2. recovery_incremental.bat (LOCAL: bajar precios faltantes via yfinance/yahooquery)
   -> incluye z-scores de acciones automaticamente al final (target=local):
      backfill_zscore_tickers desde MAX(fecha) de ticker_zscore_diario. Ya NO es
      paso manual. (28/5/2026; antes era el paso 6 de abajo.)
   -> incluye multiplos al cierre del dia (2/6/2026, target=local):
      compute_multiplos_px (PER/PB/PS/EV-EBITDA *_px en fundamentales_ratios_q con
      el cierre actual) + compute_sector_valuacion_px (comparativo de valuacion).
      Recompute DB->local, sin Yahoo. Ver docs/fundamentales_calculo.md.
   -> incluye multiplos SEC diarios (29/8/2026, target=local): compute_sec_multiplos
      --incremental (fundamentales_sec_multiplos_d). Fuente PARALELA a la de arriba.
      Escribe solo la rueda nueva pero calcula la serie entera (el percentil es
      rodante). Recompute DB->local, sin red. Ver docs/fuentes_fundamentales.md.
3. status_local.bat         (verificar 0 tickers desactualizados)
4. cron_diario --step features  (calcular features sobre los nuevos precios)
5. cron_diario --step scanner   (generar alertas)
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
  Semanal AL VUELO (29/5/2026, Tarea 11): el RSI/MACD semanal se computa al
  momento desde precios_diarios via src/utils/weekly_tf.py (resample W-FRI + ta,
  modulo PURO sin config/DB, fuente unica compartida con el dashboard). Reemplaza
  la lectura de indicadores_tecnicos_1w (congelada) y elimina el staleness guard
  previo: el semanal siempre esta fresco, sin tabla intermedia.

- Fase 1H: fundamentales (get_fundamentals, 10/06/2026) -- analisis fundamental
  trimestral sobre la capa derivada de Tarea 12 (fundamentales_ratios_q +
  fundamentales_ticker_vs_sector). Valuacion (al_cierre *_px + fiscal),
  rentabilidad SEGUN PERFIL (banco: ROTCE/efficiency; no-banco: ROIC/margenes),
  crecimiento con trayectoria (serie N trimestres), solvencia/FCF, vs pares
  regionales (percentil + peer_basis) y caveats automaticos (moneda/ADR/
  financiero). UNIDADES: la tool convierte los rate-metrics de fraccion (asi
  estan en DB, incluso columnas *_pct) a porcentaje. Validada vs JPM/XP/MU.
  Cierra el pendiente "integracion al MCP" de Tarea 12.

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
- get_fundamentals

Fases pendientes:
- run_select con validacion sqlglot (postergado: screen_tickers cubre
  la mayoria de las consultas cross-ticker; ver regla 6)
- safety.py (validacion SQL) -- pendiente, depende de run_select
- Fase 2: catalogo de queries (save_query, list_saved, recall_query)
- Fase 3: bot de Telegram (MVP local en Windows; ver docs/mcp_server.md)

Cliente/orquestador del MCP (NUEVO 15/6/2026, MERGEADO a main: merge 464f00c,
rama borrada): el dashboard incorpora una vista de chat en lenguaje natural
("Consultas (IA)") que es el PRIMER cliente propio del MCP server (hasta ahora
solo lo consumia Gemini CLI). El orquestador vive en src/agent/ (mcp_bridge =
cliente MCP por stdio con el rol mcp_reader; orchestrator = loop agentico
Gemini<->tools; config; uso_tokens = registro en llm_uso_tokens). Es REUTILIZABLE
y adelanta el grueso de la Fase 3 (Telegram solo cambiaria el frontend). Detalle:
memory/dashboard.md, docs/mcp_server.md (seccion frontend Streamlit).

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
