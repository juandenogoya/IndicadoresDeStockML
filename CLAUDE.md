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
  Bots Alpaca (3): APAGADOS el 13/9/2026 por decision del usuario (paper, sin valor
  frente a las estrategias FT): workflows deshabilitados y push de la masticada
  fuera de ft_run_diario. El codigo queda. Ver memory/bots_trading.md.
- **Windows** = rutina local manual post-cierre: UN doble clic en
  `scripts/manual/rutina_diaria.bat` (sync opciones + pasos 1-3 + FT, con log por
  paso, registro en `rutina_corridas` y resumen por Telegram; 13/9/2026).
- **Streamlit** = la app Cloud vieja (app/, indicadoresat) DECOMISIONADA (5/6/2026,
  Paso 7): directorio app/ eliminado del repo; pendiente borrar la app en
  share.streamlit.io (manual). Quedan SOLO apps Streamlit LOCALES: dashboard/
  (informe por ticker) y scripts/reports/app.py (reportes/infografia).

### Bots Alpaca -- rediseño Plan B (4/6/2026; APAGADOS el 13/9/2026)
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
- docs/estrategias_ft.md  : estrategias de forward testing (spec conceptual de cada
                            logica, registro de instancias activas y tabla de
                            instancias DISCONTINUADAS con el motivo y el link al
                            cierre). Criterios de alta/baja: este archivo, patrones
                            criticos, "Alta y baja de estrategias FT" 
- docs/earnings_reaccion.md : vista "Reaccion a balances" del dashboard +
                            tabla earnings_historico. Fecha de anuncio por Q
                            desde Alpha Vantage (la variable que faltaba: no la
                            daban earnings_calendar ni fundamentales), regla del
                            dia 0 (pre/post-market), backfill reanudable/cuota-
                            aware (key free 25/dia). LOCAL-only. Incluye el
                            diagnostico del 19/9/2026: el incremental era CIEGO
                            (miraba earnings_calendar, que solo guarda la PROXIMA
                            fecha) y por que la deteccion pasa a ser por CADENCIA
                            propia, mas el paso nocturno que la mantiene al dia.
- docs/gestion_universo.md : alta/baja de tickers del universo (Tarea 14).
                            Fuente unica via tabla activos (src/data/universo),
                            CLI universo.py (add/remove/list), insight del backfill
                            2a (ticker nace caliente), dual-write Railway, tabla
                            universo_cambios, point-in-time por construccion.
- docs/ml_reentrenamiento.md : diagnostico del ML + esquema de reentrenamiento
                            (Tarea 20). Global V3 (RF, 53 feats) desplegado, stale,
                            cubre 123/200 pero POSITIVO en vivo. Modelos SECTORIALES
                            rechazados por evidencia (el challenger desplego el global
                            en los 6 sectores). Decisiones: motor RF-global, no
                            reabrir concurso de algoritmos, lineal+calibracion como
                            controles, walk-forward PURGADO (no split unico),
                            ponderador sectorial validado o peso=1. 5 fases con
                            compuertas. OJO: los numeros de las secciones 8 y 8c
                            estan inflados por el leakage de market structure
                            (addendum sec. 10) y, mas fuerte, el intento de v3 con
                            features honestas NO PASA la compuerta pre-registrada
                            (sec. 10.1, 17/9/2026): AUC 0,51 contra 0,62 con la
                            tabla que mira al futuro. Sin cambiar la HIPOTESIS
                            (label relativo, otro horizonte) no hay v3. Por que da
                            0,51: sec. 10.2 y docs/features_ml.md.
- docs/estructura_velas.md : revision MEDIDA (17/9/2026) de velas y estructura de
                            mercado (SMC), diario y semanal. La historia de
                            features_market_structure mira 10 ruedas al futuro
                            (swings con ventana centrada): ML v1/v2 dan AUC 0,65
                            con ella y 0,52 con lo que se sabia cada dia. Velas:
                            envolvente que no envuelve (73%), martillo sin
                            contexto; ningun patron anticipa retorno. Impacto por
                            consumidor, plan de 4 fases (swings confirmados en
                            paralelo -> ML v3 -> migrar consumidores), particion
                            de datos para la v3 (walk-forward en el 80% + lockbox
                            20%) y reglas (test de invariancia). Fases 1-2 HECHAS:
                            modulos estructura.py/velas.py + tablas paralelas. Las
                            senales de entrada SMC solas no tienen ventaja; el
                            backtest con salidas esta hecho (sec. 9.3). Sec. 12:
                            AUDITORIA de las 3 tablas de features contra el OHLCV
                            (velas y estructura correctas al 100%; los patrones de
                            precio_accion mal definidos y 2 columnas que inventan
                            valores en las costuras de backfill).
- docs/features_ml.md     : inventario MEDIDO (17/9/2026) de las 53 features del ML
                            (tabla de origen, descripcion, calculo), cuales aportan
                            (ablacion por familia: ninguna que describa al ticker;
                            solo el contexto sectorial, que es regimen), redundancia,
                            las 71 tablas de la DB con su historia y su cobertura
                            sobre el dataset (el techo lo pone precios_diarios: 122
                            tickers desde 2021, los 200 recien desde 2024-04), el
                            label absoluto que mide el regimen (base 0,30-0,69 por
                            trimestre), la familia de valuacion/PER (no aporta) y la
                            seleccion propuesta para modelos nuevos. Leer ANTES de
                            proponer features o un modelo nuevo.
- docs/bots_alpaca.md     : arquitectura de produccion de los 3 bots Alpaca
                            (Plan B): masticada senales_bot_diaria, cerebro
                            COMPARTIDO src/strategies/, adapters src/trading/,
                            mapeo bot->cuenta->tabla, guard de frescura, decisiones.
- docs/forward_testing/   : detalle de forward testing. Incluye ANALISIS_SALIDAS.md
                            (19/9/2026): que hizo el precio despues de cada salida
                            de FT contra el universo (ninguna se distingue del
                            azar), la anatomia de la salida de TECH_SECTOR_v1 y sus
                            pasos 1 (quitar condiciones) y 2 (grilla de pesos, 589
                            reglas), y la de SMC_v1 (96 combinaciones de stop, CHoCH,
                            estructura y time stop): ninguna se confirma. Leer ANTES
                            de tocar una regla de salida.
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
- **LINEA SEC XBRL: DORMIDA desde el 30/8/2026.** Construida, medida y
  mergeada a main, con el paso diario APAGADO y sin consumidores en uso.
  Las tablas `fundamentales_sec_*`, `acciones_circulacion` y
  `polygon_*` quedan CONGELADAS en su ultima fecha (no se corrompen).
  El dashboard NO tiene consumidor: la vista "Comparativo Fundamental"
  se quito para no dejar una pantalla de algo que no esta en uso. Su
  codigo vive en el historico (commit fe56dde) si alguna vez se revive.
  Reactivar: `set SEC_MULTIPLOS_DIARIO=1` + una corrida de
  `scripts/manual/refresh_fundamentales_sec.bat` (el incremental no
  rellena el hueco hacia atras).
  El analisis fundamental EN USO sigue siendo el de yahooquery
  (`fundamentales_*_q` + `fundamentales_ratios_q`), que cubre los 200.
  Veredicto y pendientes medidos: docs/fuentes_fundamentales.md, cierre.

- docs/arquitectura_fuentes.md : QUE FUENTE SE USA PARA QUE, y cual manda
                            cuando dos difieren. Las 4 fuentes medidas
                            (yahooquery libreria / SEC companyfacts / Alpha
                            Vantage / EDGAR-sec-api.io, key EDGAR_API_KEY).
                            Reparto por EJE y no por metrica: temporal
                            ("caro vs si misma") = SEC; transversal ("vs
                            pares hoy") = yahooquery (unica que cubre los 53
                            no-USA); eventos = Alpha Vantage; EDGAR = bisturi,
                            no fuente. Incluye la decision abierta de la deuda
                            neta cuantificada (56 de 144 tickers sin EV; 33 se
                            arreglan GRATIS con tags propios) y las 5 reglas
                            de convivencia. Leer ANTES de agregar una fuente
                            o de resolver "de donde saco este numero".
- docs/ficha_empresa.md : CUARTA infografia (5/8/2026) -- tarjeta "presentacion
                            de empresa" fondo OSCURO (4:5). La empresa contra SI
                            MISMA: ultimo Q reportado + variacion INTERANUAL (sin
                            pares/benchmark). Adapta secciones por perfil
                            (banco: ROE/ROTCE/efficiency; no-banco: margenes/ROIC/
                            FCF). Ancla al ultimo Q con income real (evita el stub
                            recien reportado). Motor: make_ficha_empresa.py.
- docs/checklist_recovery_manual.md : flujos de recovery manual
- docs/perfiles_carteras.md : segmentacion del universo en 4 perfiles de riesgo
                            (Conservadora/Moderada/Arriesgada/Especulativa) por
                            COMPORTAMIENTO cuantitativo (percentil de ATR% multi-TF
                            + beta + drawdown), con el sector como contexto y
                            fuente del flag de excepcion. Fases 0-5 hechas.
                            Motor: compute_perfiles_carteras.py -> perfiles_ticker.
                            SECCION 15 (15/9/2026) = el Proyecto 2 (rotacion) medido
                            y DESCARTADO: por que el volumen agregado por cartera es
                            ruido y que haria falta para retomarlo (historia de
                            snapshots -> drift, no volumen).
- dashboard/README.md     : spec del Dashboard (informe descriptivo por ticker).
                            v1 + Fase 2 v1 desarrollados 28/5/2026 en rama
                            feature/dashboard: Streamlit local (modos Informe y
                            Radar del dia), export JPG (informe) y PDF (papel de
                            trabajo). Corre bajo el venv. Ver memory/dashboard.md
                            ("Como correrlo / retomarlo"). Incluye la seccion
                            "Alertas de comportamiento inusual" (15/9/2026): la
                            medicion sobre 13.863 ticker-dias que decidio NO
                            construir una vista nueva, y las 2 reglas que deja para
                            cualquier alerta futura -- umbral por Z-SCORE (nunca
                            multiplo fijo) y SIN indicacion de direccion (medido:
                            la direccion no existe a 5 ruedas).

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
  Eso vale para la CAPTURA (en la nube no hay precios_diarios). Para CALCULAR en
  local manda otra cosa, ver el punto siguiente.
- **PRECIO DE REFERENCIA (10/9/2026)**: todo calculo en LOCAL que necesita "el
  precio del subyacente en la rueda D" (muros de OI, expected move, resumen,
  moneyness del MCP, put wall de FT oiexit) usa el **close de `precios_diarios`**;
  `precio_subyacente` de la captura solo TAPA EL HUECO si falta el close. Regla en
  UN lugar: `src/utils/precio_referencia.py` (replicada en SQL solo en
  `mcp_server/db/queries.py`, ver abajo). La fuente usada
  viaja en `precio_fuente` de opciones_resumen_diario y opciones_pcr_plazo_diario.
  Motivo: el 2026-09-09 yahooquery `.price` devolvio 0 de 200 SIN excepcion ->
  crudo sin precio -> muros vacios en todo el universo con el close disponible; y
  el crudo previo al 26/5 tiene precios de Railway congelado (2.102 de 4.225 pares
  ticker-fecha >0,5%). Cuando ambos existen coinciden (max 0,055%). **ESCALA DE
  SPLIT**: splits.py corrige precios_diarios HACIA ATRAS, pero los strikes de una
  cadena vieja estan en la escala de SU dia -> el close se lleva a esa escala
  multiplicando por los splits REALES (ratio >=1,5) ejecutados despues de la rueda,
  segun el registro `splits_aplicados` (`precio_fuente='precios_x_split'`; lo
  escribe `splits.py corregir`, ver "Splits" abajo). Validado: captura /
  (close x factor) = 1,0000 exacto en 81 ruedas frescas de KLAC y 81 de CRWD. El
  factor sale del REGISTRO y no de la captura: con captura rancia (pre-11/5) el
  ratio observado no es exacto (KLAC x9,17..x10,70) y el primer diseno, que lo
  deducia de ahi, lo dejaba pasar. Los ajustes chicos (SCCO 1,01/1,012 de polygon,
  spinoffs tipo HON 1,061) NO se aplican: con ellos el cruce empeora (SCCO 0,988).
  Si captura y close difieren por un split EXACTO que el registro no explica, el
  paso 0 avisa. Las queries
  del MCP replican la regla en SQL sin factor: solo miran el ULTIMO snapshot. El snapshot
  ALERTA por Telegram si trae precio para <90% del universo, y el paso 0 de
  `compute_opciones_derivadas.py` avisa cuantos tickers caen al precio de la
  captura: funciona como detector de huecos de precios_diarios (asi aparecio el
  2026-08-28 con 157 de 200 tickers sin close).
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
- **`ft_posiciones_diarias.fecha_datos`** (12/9/2026): misma trampa. `fecha` es el
  dia en que corrio el bot y `precio_cierre` es el ultimo close disponible (18,8%
  del mismo dia, 79,6% de la rueda anterior). La escribe
  `ft_utils.registrar_estado_posiciones()`; la historia se backfilleo con
  `scripts/oneshot/add_fecha_datos_ft_posiciones.py`.

### FT: todo cambio que toque decisiones se REGISTRA en `ft_cambios` (13/9/2026)
- Antes de desplegar algo que cambie logica, parametros, modelo, un dato o infra
  con los que decide alguna estrategia FT: `ft_cambios.py add --dry-run`, revisar
  avisos, registrar, y la linea `**Registro**` en el JOURNAL. Sin registro, el
  cambio no se puede medir despues.
- La fecha efectiva se razona como `fecha_datos`: la rueda con la que va a decidir
  la proxima corrida. La del commit lee el tramo equivocado, en silencio.
- Se mide CONTRA EL GRUPO DE CONTROL (estrategias no afectadas, mismos dias), no
  contra cero ni solo contra el universo: en el fix del 29/5 la expectancy cruda de
  TECH_SECTOR_v1 daba EMPEORA con IC que excluia el cero y era el mercado del tramo
  (contra el control, NO CONCLUYENTE). Un cambio que afecta a TODAS no tiene control.
- Una estrategia nueva en paralelo (ej. ML_SCANNER_v2) no corta a la vieja: se
  comparan en el mismo periodo.

### Alta y baja de estrategias FT -- como se decide y que queda escrito (17/9/2026)
Destilado de las altas de SMC_v3 y de las bajas de COMBO_v1 y SMC_v2. Leer ANTES de
proponer, desplegar o apagar una estrategia. Detalle: docs/estrategias_ft.md,
docs/forward_testing/README.md y las fichas de docs/forward_testing/estrategias/.

**Una estrategia existe para responder UNA pregunta.** Se escribe en la ficha ANTES de
desplegarla ("?agregar X mejora la seleccion?"), junto con el CONTROL con el que se va
a comparar: la misma logica base sin X. Sin control de la misma familia no hay
conclusion posible -- comparar contra otra logica mide el mercado del tramo.

**Toda version nueva va EN PARALELO y la vieja sigue como control.** No se reemplaza
una logica en su lugar: se crea una instancia nueva con su id y las dos corren los
mismos dias. Los ids no se reutilizan (la historia cuelga del id).

**Antes de desplegar una regla nueva: backtest con entradas Y SALIDAS.** Medir el
retorno forward de la senal de ENTRADA a plazo fijo es una prueba de PREDICCION y no
evalua una estrategia de reglas: SMC y COMBO leen la estructura para entrar y para
salir. Se pre-registra por escrito, antes de correr: periodo, variantes, metricas y la
REGLA DE LECTURA (que resultado cuenta como "pasa"). La usada: gana plata en el total
Y le gana al universo equal-weight ajustado por exposicion (retorno / exposicion
media) en al menos 4 de los 6 tramos anuales. Declarar que no hay costos y cuantas
operaciones por anio hace (960/anio con costos no es lo mismo que 80/anio).

**Reglas fijas no tienen sesgo de "el modelo vio los datos", pero los PARAMETROS si.**
Elegidos mirando resultados son sobreajuste hecho a mano. Corolario: si dos variantes
del mismo parametro empatan en el backtest (N=5 y N=3), NO se elige la mejor: van las
dos a FT y decide la operacion real.

**Para dar de baja hacen falta las tres:**
1. la pregunta de la estrategia ya tiene respuesta y es "no aporta";
2. evidencia de dos mediciones independientes que apuntan al mismo lado (backtest de
   anios + FT contra el control), o una medicion de fondo que explique por que no
   puede aportar (ej. ningun patron de vela tiene exceso distinto de cero);
3. no es un problema de calibracion: mover el umbral no cambia un aporte medido en cero.
**Con la potencia estadistica de FT hay que ser honesto**: con 27 o 450 operaciones la
diferencia contra el control casi nunca es distinguible de cero (IC95 incluye el cero).
Eso se ESCRIBE, y la baja se apoya en el backtest y en los insumos, no en el resultado
de FT. **Una regla que nunca dispara no es una regla**: la salida por agotamiento de
SMC_v2 no se activo ni una vez en 94 ruedas. Contar activaciones antes de creerle algo.
**NO son motivo de baja por si solos**: un mal tramo, o ser peor que una estrategia de
otra logica.

**Que queda cuando se da de baja** (nada se borra):
- la FICHA con el cierre: periodo exacto, parametros con los que corrio, metricas
  (equity, max DD, Sortino, ops, aciertos, expectancy, profit factor), tabla de
  motivos de salida, los motivos de la baja, **que NO dice el cierre** y como se
  reabriria;
- la historia en `ft_operaciones` / `ft_equity_diaria`, con las posiciones abiertas
  liquidadas al ultimo cierre con `motivo_salida = ESTRATEGIA_DISCONTINUADA` (salida
  ARTIFICIAL, etiquetada para poder excluirla). Sin liquidar, la equity se seguiria
  marcando a mercado para siempre sin nadie que decida;
- `ft_estrategias.activa = FALSE` (el bot no puede operar aunque lo corran suelto);
- la entrada en `ft_setup_estrategias.ESTRATEGIAS` con la clave `discontinuada` y el
  motivo (no se re-inserta, pero los parametros no se pierden);
- el bloque del bot en `ft_run_diario.bat` como comentario con el motivo;
- el registro en `ft_cambios` (tipo INFRA, `cambia_decisiones=FALSE`);
- el codigo del bot, sin borrar.
Herramienta: `scripts/oneshot/discontinuar_estrategias_ft.py` (generico, `--dry-run`).

### Analisis de salidas FT: cruzar con `ft_cambios` y enumerar la regla (19/9/2026)
- **La historia de TECH_SECTOR_v1 antes del 29/5/2026 no mide su regla**: hasta el
  arreglo `fix_score_cero_salida` a la consulta de salida le faltaba el close y el score
  daba 0 todos los dias (357 de 412 salidas "SCORE_DEGRADADO_0.0" tenian score real
  >= 4). Toco tambien a TECH_SECTOR_v2 y OPTIONS v1/v2. La ficha de la v1 lo diagnostico
  como "exit binario" y la v2 se diseno sobre esa premisa. Antes de analizar la historia
  de una estrategia, cortar las ventanas que `ft_cambios` marca como bug.
- **Un score ponderado con umbral es una regla de si/no.** En TECH_SECTOR_v1 ningun
  score cae entre 3,5 y 4,0: salir con <= 3,5 es exactamente dejar de cumplir la entrada
  (SMA200 y SMA50 y 2 de 3 entre SMA21/MACD/RSI) y los pesos no mueven ninguna salida.
  Antes de "ajustar ponderadores", enumerar las combinaciones (`src/utils/ft_salidas`).
- Las salidas se miden contra el universo, en desvios del ticker y con el DIA como
  unidad. Al 19/9 ninguna salida de ninguna estrategia de FT se distingue de salir al
  azar. Cualquier palanca se elige con el backtest de anios, no con los meses de FT.
- Resultados de cada corrida en `reportes/analisis_salidas/AAAAMMDD_<etiqueta>/` (fuera
  de git, con `parametros.json`); los numeros que importan quedan en el doc.
- **Paso 1 de TECH_SECTOR_v1 (19/9, pre-registrado): sacar SMA21, MACD o RSI de la salida
  NO mejora.** Entradas fijas del motor 2021-2026 (4.902), salida re-simulada: sin SMA21
  el exceso por operacion no cambia (-0,02 pp, IC95 [-0,16; +0,11]) y la cola empeora
  (p5 -6,37% -> -8,07%). La salida rapida es control de riesgo. La re-simulacion se valido
  antes contra las salidas reales de FT (paso 0). Relajar una salida se juzga por la cola.
- **Paso 2 (19/9, pre-registrado): ninguna combinacion de pesos mejora la salida.** Grilla
  de pesos {0; 1; 1,5; 2; 3} por condicion, SMA200 obligatoria o con peso -> 589 reglas;
  metrica = exceso contra el universo en las 10 ruedas DESPUES de salir, seleccion 2021-24 y
  confirmacion 2025-26. 0 de 588 candidatas. Despues de la salida actual la accion hace lo
  que el universo (post10 -0,20 / +0,12 pp, desvio 7,26). Los pesos no eligen el momento:
  eligen cuanto se queda la posicion (correlacion +0,95 con la cola) y el orden entre reglas
  no se sostiene de un periodo al otro (-0,16). La mejor en 2021-24 ("mas peso al RSI") da
  cero en 2025-26. Con cientos de reglas, leer con seleccion y confirmacion separadas.
- **SMC_v1 (19/9, pre-registrado): ninguna de 96 salidas se confirma.** La historia de
  `features_market_structure` mira al futuro -> se reconstruye, rueda por rueda, lo que el bot
  veia (modulo viejo sobre las ultimas 250 barras; verificado 200/200 contra la tabla y 45/46
  entradas de FT). CHOCH_BEAR salio primero 1 vez en 920: el trailing stop esta en el mismo
  swing low y va antes. El time stop de 10 dias (hoy 20) mejora post y tramo en 2021-24 con IC
  que excluye el cero y va al mismo lado en 2025-26 sin alcanzar; hipotesis sin confirmar.
- **`earnings_historico` puede ir atrasada** (carga con cuota de Alpha Vantage): al 19/9/2026
  estaba completa hasta el 20/7 (julio-agosto: 86 de 200 tickers). Una re-simulacion que marca
  balances desde ahi atraviesa balances que el bot evita, y no igual en todas las reglas:
  cortarla antes (`ft_analisis_salidas_smc.fin_balances`). Invalido el FT de control del
  analisis de TECH_SECTOR_v1 (sec. 7.1 del doc). ARREGLADO el 19/9 (ver el patron siguiente);
  antes de re-correr un analisis con balances, mirar `--status`.

### Una tabla de EVENTOS se vigila por CADENCIA, no por antiguedad (19/9/2026)
- `earnings_historico` quedo 6 semanas atrasada mientras su propio diagnostico decia
  "Desactualizados: 0". La deteccion preguntaba si `earnings_calendar.earnings_date` ya
  habia pasado, pero esa tabla guarda **solo la PROXIMA fecha** de cada ticker y se refresca
  semanalmente: el dia que la empresa reporta, el refresh la empuja al trimestre siguiente y
  el ticker no vuelve a figurar como desactualizado NUNCA. La ventana de deteccion era el
  hueco entre el anuncio y el proximo refresh. Corriendo el incremental todas las noches
  tampoco habria traido nada.
- **Entre temporadas de balances, "vieja" es lo correcto**: por eso una tabla de eventos no
  se vigila con un umbral de antiguedad (y por eso `earnings_historico` NO entra en
  `estado_pipeline`, que mide alineacion de ruedas). Se vigila con la CADENCIA PROPIA de
  cada fila-sujeto: la mediana de dias entre sus eventos, que es un hecho suyo. Modulo puro
  `src/utils/earnings_cobertura.py`, FUENTE UNICA del script y del dashboard. Mediana y no
  promedio (un cierre fiscal corrido deja un intervalo raro); margen 15%; con cadencia FIJA
  un semestral daria falso positivo todos los trimestres.
- Misma familia que `scan_fecha`, `features_sector` y `fecha_datos`: un guard que nunca falla
  y devuelve un numero plausible. Si un diagnostico da 0 problemas, comprobar que PUEDE dar
  distinto de 0.
- El dato faltante no avisa solo: **la vista se ve igual de completa con la tabla atrasada**
  (un trimestre que falta no se distingue de uno que no existe). Por eso la vista ahora
  muestra la cobertura y avisa si al ticker elegido le falta un balance.
- Nasdaq sirve para dias PASADOS y no tiene cuota, pero no reemplaza a Alpha Vantage aca:
  devuelve el trimestre fiscal como MES y `time` siempre `'time-not-supplied'` -> se pierde
  `report_time` y con el la regla del dia 0 (post-market = la rueda siguiente).

### alertas_scanner: `scan_fecha` NO es la fecha de datos (incidente 2/9/2026)
- La fecha de datos de una alerta es **`precio_fecha`** (sobre que cierre se
  calculo). `scan_fecha` y `created_at` son CUANDO CORRIO el scanner. Es la
  misma trampa que `fecha_datos` vs `fecha_entrada` en FT: cruzar por la fecha
  de registro lee el dia equivocado, en silencio.
- Es la UNICA tabla del pipeline con esa dualidad. En las demas la columna
  `fecha` / `fecha_snapshot` es la rueda, y el reloj de escritura vive aparte
  (`created_at` / `computed_at` / `calculado_en`).
- Como se manifesto: el 2/9 `chequeo_rutina.py` informo "todo al dia y
  alineado" mientras las alertas eran del cierre del 31/8 y el resto del 1/9
  -- el scanner habia corrido el 1/9 (registro fresco) sobre datos de anteayer.
  El ft_run de ese dia y la masticada de los bots usaron senal ML de una rueda
  y tecnico de otra. El guard fallo en el primer caso real que le toco.
- Arreglado: `estado_pipeline.TABLAS` mide alertas_scanner por `precio_fecha`;
  el reloj de corrida se informa en una columna aparte. Hay test de regresion
  con el caso exacto y un guard de nombres que rechaza usar una columna de
  reloj como fecha de diagnostico.
- REGLA al agregar una tabla al registro: la pregunta no es "cual es su columna
  de fecha" sino **"cual de sus fechas dice a que rueda pertenece el contenido"**.

### 4 tickers sin contexto sectorial -- el modelo es el correcto, la entrada no (2/9/2026)
- Los features sectoriales son z-scores del ticker CONTRA SUS PARES. Con n<=3 no
  miden nada (con n=1 el ticker es su propio promedio y da 0 por construccion),
  asi que Real Estate (AMT/EQIX/PLD) y Utilities (VST) quedaron afuera del
  calculo en la Tarea 20. Son 4 de 200.
- **Son dos problemas distintos y solo uno estaba resuelto.** (1) QUE MODELO se
  usa: el global, y esta bien -- con 1 y 3 tickers no hay con que entrenar uno
  sectorial. (2) QUE DATOS recibe: **11 de las 53 features llegan NaN**. El
  global fue ENTRENADO con esas 11 presentes. Elegir el modelo global resuelve
  (1) y no toca (2). Verificado: AAPL recibe 11/11, EQIX 0/11.
- No es teorico: al 1/9/2026 EQIX estaba en COMPRA_FUERTE, que es el nivel con
  el que abren posicion `ft_bot_ml_scanner` y el bot_ml de Alpaca.
- **DECISION (2/9/2026): se marca, no se filtra.** La probabilidad se sigue
  publicando con la etiqueta "Sin contexto sectorial"; los bots operan esos 4
  igual que antes. Razon: FT y Alpaca son paper y existen para EVALUAR
  estrategias -- filtrarlos le sacaria casos al experimento. Revisar si se
  incorporan tickers a esos sectores.
- La marca se DERIVA del sector en cada lectura (`src/utils/contexto_sectorial`),
  no se almacena: sin migracion, sin backfill, y aplica retroactivamente a toda
  la historia de `alertas_scanner` (que ya tiene la columna `sector`). Cuando un
  sector crezca y salga de la lista, la marca deja de emitirse sola.
- La lista de sectores vive en UN lugar y el WHERE del productor se arma desde
  ahi. Con dos listas, el dia que un sector entre al calculo la marca seguiria
  apareciendo sobre tickers que ya tienen contexto -- o dejaria de aparecer
  sobre los que no.

### features_sector: el scanner lee la fila de SU rueda (incidente 13/9/2026)
- `feature_calculator._obtener_zscore_sectorial` tomaba la ULTIMA fila de
  `features_sector` sin mirar la fecha, y ningun paso diario actualizaba la tabla
  (solo el legacy 05, a mano: 24/2, 30/3, 9-10/4 y 2/7). El modelo ML recibio 11
  de sus 53 features con semanas de antiguedad durante toda la vida de
  FT_ML_SCANNER_v1, sin un solo error. Medido sobre la rueda 11/9: con el dato de
  la rueda, 57 de 200 tickers cambian de nivel.
- Ahora: el Paso 2 (`cron_diario --step features`, paso 2b) recalcula
  `scoring_tecnico` + `features_sector` de las ultimas 10 ruedas (calcula sobre
  toda la historia y persiste desde ahi); el scanner lee `fecha = rueda de la
  barra` y si falta deja NaN y lo avisa en su resumen; `estado_pipeline` vigila la
  tabla como insumo critico.
- Misma familia que `scan_fecha` y `fecha_datos`: una lectura "la ultima que
  haya" nunca falla y devuelve un numero plausible. Para decidir sobre la rueda
  D, leer la fila de D.
- Afecto SOLO a FT_ML_SCANNER_v1 (la unica estrategia que lee `alertas_scanner`),
  al Bot 1 de Alpaca mientras estuvo prendido, al Telegram del scanner y al MCP.
  Registrado en `ft_cambios` (DATOS). Detalle: docs/ml_reentrenamiento.md sec. 8b.
- Los legacy 03/05/06 (`scripts/legacy_ml/`) se corren desde la raiz con
  `PYTHONPATH=.`: su `sys.path` apunta a `scripts/`.

### features_market_structure: la HISTORIA mira 10 ruedas al futuro (medido 17/9/2026)
- `market_structure._calcular_estructura_n` (y la copia `_1w`) detecta swings con
  `rolling(2N+1, center=True)` y los registra en SU barra, cuando recien se conocen
  N barras despues. El Paso 2 recalcula la tabla entera cada dia: cada fila historica
  queda escrita con lo que paso despues. En la historia guardada, un swing high da
  -4,7% de exceso a 5 ruedas; con lo que se sabia ese dia, ~0.
- `min_periods=n+1`: la ULTIMA barra si se marca (swing provisional, max de las
  ultimas 11) y puede desaparecer al dia siguiente. El docstring dice NaN: es falso.
- Lo que opera hoy NO mira el futuro: scanner y bots FT usan la ultima fila. Lo
  contaminado: entrenamiento y validacion de ML v1/v2 (AUC 0,65 guardada vs 0,52
  real), walk-forward de la Tarea 20, backtests SMC/COMBO, historia en dashboard/MCP.
- REGLA: no entrenar, validar ni backtestear sobre esta tabla. La historia sin futuro
  es `features_estructura` (swings CONFIRMADOS, `src/indicators/estructura.py`) y
  `features_velas` (`src/indicators/velas.py`), en paralelo desde el 17/9/2026 y todavia
  sin consumidores de decision. Toda feature nueva pasa el test de invariancia:
  `calcular(datos[:t+1]).iloc[-1] == calcular(datos).iloc[t]`. Misma familia que
  `scan_fecha` y `features_sector`. Detalle y plan: docs/estructura_velas.md.
- Con historia sin futuro ninguna senal de entrada SMC, sola y a plazo fijo, elige
  acciones mejores que el universo (doc sec. 9.2). Eso NO evalua las estrategias SMC /
  COMBO como operan (salidas estructurales). Backtest 2021-09 -> 2026-09 con
  entradas y salidas (doc sec. 9.3): SMC_v1 con N=10 da +61% con la historia vieja y
  +13,5% con la nueva (dependia del futuro); con N=5 / N=3 confirmados da ~+58-60% y
  pasa la regla pre-registrada. COMBO no mejora por las velas (TECH_SECTOR sin velas
  +24,4% vs COMBO +22,3%). Sin costos. Cualquier cambio de regla se valida en FT.
- ML v3 CORRIDO y NO PASA (17/9/2026, doc sec. 9.6): con `features_estructura` (honesta)
  la config congelada de la Tarea 20 da AUC media **0,5099** en 6 folds purgados (3/6 por
  encima de 0,52; los dos folds con MAS datos quedan debajo de 0,50), y las 24 features de
  estructura aportan +0,0035 de AUC -> no entran. Control con los mismos folds y la tabla
  vieja: **0,6189** (6/6). El brazo de 29 features da identico en las dos corridas, lo que
  valida el montaje. **No hay modelo v3 desplegable; el lockbox (2025-07-15 -> 2026-08-13)
  quedo SIN ABRIR** para una hipotesis nueva. Los ~11 puntos de AUC del modelo eran el
  look-ahead. FT_ML_SCANNER_v1/v2 siguen corriendo: en vivo leen la ultima fila, sin futuro,
  y su resultado en FT es la medicion honesta -- lo que quedo sin respaldo es el "AUC 0,65".
- Y NO es por ser un modelo GLOBAL (doc sec. 9.7-9.8, pre-registrado): sobre las 96.534
  predicciones fuera de muestra de esos folds, ningun sector discrimina (mejor Basic
  Materials AUC 0,5446 con IC95 [0,476; 0,614] que incluye 0,50; peor Industrials 0,4761
  con 1/6 folds > 0,50) y la dispersion ENTRE sectores (0,0225) es un tercio de la
  dispersion DENTRO del sector entre ventanas (0,0581): lo que parece un sector bueno es
  la ventana de 6 meses que toco mirar. Por eso NO se entrenan modelos sectoriales (esto
  confirma el rechazo de ml_reentrenamiento sec. 2.5, cuyo head-to-head estaba medido con
  las features contaminadas) y por industria no se prueba: 13 industrias con 5+ tickers,
  31 con uno solo. Lo que queda por cambiar es la HIPOTESIS (label relativo al universo,
  horizonte mas corto), no el alcance del modelo.
- Y NO es por falta de features (17/9/2026, docs/features_ml.md): quitando cada familia del
  set de 53 en los mismos folds, ninguna que describa al TICKER aporta (sin los 4
  indicadores base el AUC MEJORA); lo unico que mueve el AUC es el contexto sectorial
  (+0,0126), que vale lo mismo para todo el sector en una fecha: es regimen. El label
  absoluto `retorno_20d > +1%` tiene base rate de 0,302 a 0,694 segun el trimestre: la
  pregunta la contesta el mercado. REGLAS para un modelo nuevo: (1) se empieza por el
  label (relativo al universo del dia) y por pre-registrar el UNIVERSO (hoy el fold 6
  tiene 196 tickers y los otros 122), no por las features; (2) una feature que vale lo
  mismo para todos los tickers de una fecha no puede ordenar tickers; (3) mirar la
  direccion y el retorno por decil, no solo el AUC (el unico indicio de la familia de
  valuacion, EV/EBITDA, va AL REVES del valor: lo caro subio mas); (4) contar las
  comparaciones: con 24 pruebas un 6/6 folds aparece por azar el ~31% de las veces.
- DECIDIDO con ese resultado (17/9/2026, doc sec. 9.4): **alta** de `FT_SMC_v3_N5` y
  `FT_SMC_v3_N3` (ids 12 y 13; misma regla de FT_SMC_v1 leyendo `features_estructura` +
  `features_velas`; las dos ventanas en paralelo porque el N no se elige mirando el
  backtest; FT_SMC_v1 sigue como control) y **baja** de `FT_COMBO_v1` (las velas no
  aportan) y `FT_SMC_v2` (peor equity, filtros sobre insumos sin valor, su salida nueva
  nunca disparo). Ver "Alta y baja de estrategias FT" arriba.

### features_precio_accion: patrones mal definidos y valores INVENTADOS (auditado 17/9/2026)
- `scripts/manual/auditar_features_tablas.py` recomputa las 3 tablas de features desde
  `precios_diarios` (las 3 reproducen exacto) y audita cada definicion con una
  implementacion INDEPENDIENTE contra el OHLCV. `features_velas` y `features_estructura`:
  100% en todo. `features_precio_accion`: el 71% de las envolventes no envuelve (solo
  compara tamano de cuerpo), el 52% de los martillos son hanging man y el marubozu no
  tiene direccion. Los patrones salen de `features_velas`; no se reparan en origen. Los
  leen todavia FT_SMC_v1 (a proposito, es el control), scanner, Telegram, dashboard y
  MCP (Fase 4 de docs/estructura_velas.md).
- **Valores inventados**: `tendencia_velas = velas_alcistas_5d.fillna(0)*2-5` escribe -5
  (el maximo bajista) donde no hay 5 barras, y `rango_expansion` (comparacion con NaN ->
  `astype(int)`) escribe 0. Caen en las COSTURAS DE CADA BACKFILL (32 fechas: las 3
  cohortes de alta del universo), no en el arranque de la serie, y se van a repetir en el
  proximo backfill. Ademas `tendencia_velas` es exactamente `2*velas_alcistas_5d - 5`:
  cero informacion. Lo que SI esta bien: flujo de volumen (`ad_flow`, `chaikin_mf_20`,
  `up_vol_5d`, `vol_ratio_5d`) y microestructura (`clv`, `gap_apertura_pct`,
  `rango_rel_atr`), la familia sin usar que propone docs/features_ml.md.
- REGLA: "la tabla reproduce el codigo" no es "la tabla esta bien". Un `fillna(0)` o un
  `astype(int)` sobre NaN no falla: escribe un numero plausible donde no hay dato. Correr
  el auditor despues de un backfill, de `splits.py corregir` o de tocar
  velas/estructura/precio_accion. Detalle: docs/estructura_velas.md sec. 12.

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
- **REGISTRO `splits_aplicados` (12/9/2026)**: `splits.py corregir` anota cada
  split en la MISMA transaccion que corrige precios_diarios -> no hay correccion
  sin registro. Guarda la fecha REAL de ejecucion que informa Yahoo
  (`yahooquery_loader.eventos_split`: history() trae la columna `splits`, que
  download_batch descarta), NO la fecha de corte de la DB, que depende de cuando
  se bajo cada rueda (KLAC corte 11/6 / ejecucion 12/6; CRWD 30/6 / 2/7). Si
  Yahoo no la informa, no corrige: `--fecha-ejecucion`. Lo lee el factor de escala
  del precio de referencia de opciones (antes polygon_splits: congelado desde el
  30/8, y un split listado pero sin corregir se habria escalado dos veces). Carga
  inicial validada: `scripts/oneshot/create_splits_aplicados.py`.
- **FT y bots Alpaca ante un split: se REGISTRA, NO se corrige** (decision
  12/9/2026: son paper para evaluar). Una posicion abierta durante el split
  cierra por un stop ficticio y queda con perdida ficticia; los analisis la marcan
  cruzando con splits_aplicados (consulta en docs/forward_testing/METRICAS.md).
  Corregido precios_diarios, las decisiones nuevas vuelven a ser correctas solas.
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

### Batch .bat: parentesis dentro de un bloque IF (2/9/2026)
- Adentro de `IF ... ( ... )`, un `)` SIN escapar CIERRA el bloque antes de
  tiempo. `echo Corre lo que falta (arriba) y volve.` dejo ` y volve.` suelto y
  cmd aborto con "no se esperaba **y** en este momento" -- un error que no
  nombra ni la linea ni el parentesis, y que en este repo confunde mas porque
  la ruta del proyecto tiene una "y" ("Indicadores y Machine Learning").
- Al escribir mensajes adentro de un bloque: sin parentesis, o escapados
  `^( ^)`.
- El bloque se parsea ENTERO al llegar al `IF`, asi que falla aunque la rama
  del error nunca se ejecute. Corolario: un `.bat` que "anduvo siempre" no
  garantiza nada sobre una rama nueva -- el guard de ft_run_diario nacio roto
  y no se noto hasta la primera corrida que lo ejercito, 4 horas despues.
- **Editar .bat con `sed -i` los rompe**: MSYS trabaja en modo texto y se come
  los CR, dejando el archivo LF-only. Por el mismo motivo `grep -c $'\r'` /
  `cat -A` MIENTEN sobre los finales de linea de un .bat. Para verificar de
  verdad: `tr -cd '\r' < f | wc -c`. Para editar: Python en modo binario.
- **`python ... | tee log` pierde el codigo de salida** (13/9/2026): `%ERRORLEVEL%`
  queda con el del `tee`, que siempre es 0. `recovery_incremental.bat` dijo
  "RECOVERY COMPLETO" con tickers pendientes durante meses (medido: Python sale 3,
  el .bat lee 0). Para log + codigo real: `scripts/manual/rutina_diaria.py`
  (`paso` / `correr`), que hace el tee en Python.
- **La logica de orquestacion no va en un .bat**: condiciones, politicas y
  resumenes viven en Python (`rutina_diaria.py` + `src/utils/rutina.py`). Los .bat
  de la rutina quedaron sin bloques IF: IF de una linea y GOTO.
- Un .bat llamado desde otro proceso no puede terminar en `pause`: queda esperando
  una tecla. `ft_run_diario.bat` saltea sus pausas con `RUTINA_ORQUESTADA=1`.
- **`exit /b N` adentro de un bloque `IF ( ... )` no llega a quien lanzo el .bat
  con `cmd /c`: sale 0** (medido 13/9/2026 en el guard de ft_run_diario, que asi
  habria dado "OK" con los bots sin correr). Salir del bloque con `goto :etiqueta`
  y hacer el `exit /b` fuera. El `exit /b` de nivel superior si devuelve bien.

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
| `src/utils/precio_referencia.py` | Modulo PURO (stdlib): regla del precio de REFERENCIA del subyacente para calcular en local -- el close de `precios_diarios` manda, `precio_subyacente` de la captura solo tapa el hueco. `resolver_precio` / `factor_escala` (escala de split desde splits_aplicados) / `elegir_evento_split` (fecha real del split para el registro) / `medir_divergencias` / `escalas_sin_registro` / `cobertura_baja`. Lo usan opciones_plazo, compute_opciones_derivadas, FT oiexit y el snapshot (alerta de cobertura) |
| `scripts/compute_opciones_derivadas.py` | Derivadas de opciones en LOCAL desde el crudo (HV, resumen, z-scores, PCR+muros por plazo). Paso 0 = diagnostico del precio de referencia (cuantos tickers caen al precio de la captura = huecos de precios_diarios). `--fecha` / `--desde` (recalcula en orden: los z-scores usan la historia previa). Paso [0b] de ft_run_diario.bat |
| `scripts/sync_local.bat` | Sync Railway -> Local |
| `scripts/sync_to_railway.bat` | Sync Local -> Railway (paso a paso) |
| `scripts/manual/universo.py` (+ `.bat`) | Alta/baja de tickers del universo (Tarea 14). `add` (backfill 2a + indicadores/features/z-scores/fundamentales + dual-write activos local+Railway + log), `remove` (soft delete + guard posiciones FT), `list`. Solo acciones. Ver docs/gestion_universo.md |
| `src/data/universo.py` | Fuente UNICA del universo: get_universo()/get_universo_sectores() leen de `activos` (fallback config.ALL_TICKERS). Lo usan snapshot/scanner/refresh/cron |
| `scripts/migrations/clean_ticker_fantasma_se.py` | Limpieza generica ticker fantasma |
| `scripts/oneshot/clean_railway_may12.py` | One-shot one-off (archivado en scripts/oneshot/) |
| `scripts/manual/check_fecha.py` | CLI valida dia habil NYSE |
| `scripts/manual/ft_run_diario.bat` | Corre los 11 bots ACTIVOS de Forward Testing en local (FT_ML_SCANNER_v2 desde el 14/9/2026, bloque [1b/11], sale 1 sin operar si el scanner no trajo la v2; FT_SMC_v3_N5/N3 desde el 17/9/2026, bloques [3b/11] y [3c/11]; los bloques de COMBO_v1 y SMC_v2 quedaron comentados con el motivo de la baja) + reporte HTML + precomputo de veredictos del dashboard. El push de senales_bot_diaria se saco el 13/9/2026 (bots Alpaca apagados). Sale 0 OK / 1 el guard freno / 2 algun bot fallo. Con `RUTINA_ORQUESTADA=1` (lo setea rutina_diaria) no pausa ni se registra solo; suelto se anota en `rutina_corridas` |
| `scripts/manual/rutina_diaria.bat` (+ `.py`) | La rutina diaria COMPLETA (13/9/2026): sync opciones + purga, Paso 1, Paso 2, Paso 3, ft_run_diario y (19/9/2026) `earnings`, en orden. Si falla el Paso 1 con mas de 10 tickers pendientes, el 2 o el 3, FRENA antes de los bots; el sync sigue. Log por paso en `logs/rutina/AAAAMMDD_HHMM/` + `resumen.txt`, registro en `rutina_corridas`, resumen por Telegram con los tickers pendientes y su ultimo dato. `--desde pasoN` / `--sin-telegram`. El .py tambien es el ejecutor de cada .bat de paso (`paso <clave>`, `correr --nombre recovery_incremental`): codigo de salida REAL (0 OK, 2 avisos, 1 error) |
| `src/utils/rutina.py` | Modulo PURO de la rutina: orden de pasos, politica ante una falla, clasificacion del Paso 1 (PARCIAL vs caida, via `recovery_incremental.py --resumen-json`), resumen de texto y mensaje de Telegram |
| `scripts/manual/chequeo_rutina.py` | Guard de coherencia de la rutina diaria (LOCAL). Distingue ANTIGUEDAD (todo viejo pero alineado = la convencion del proyecto, NO frena) de MEZCLA (tablas con fechas distintas entre si = decisiones con datos cruzados, SI frena). Reporta que .bat arregla cada tabla y aparte el caso IRRECUPERABLE (falta el crudo de opciones). Lo corre `ft_run_diario.bat` despues del paso [0b] y ANTES del primer bot; `set FT_IGNORAR_FRESCURA=1` lo saltea. Motor puro: `src/utils/estado_pipeline.py`. Informa ademas los HUECOS en el medio de la serie (ultimas 252 ruedas de precios/indicadores/features; avisa, no frena; excepciones verificadas en `HUECOS_CONOCIDOS`; `--solo-huecos`) y la ultima corrida de cada paso (`rutina_corridas`) |
| `src/utils/contexto_sectorial.py` | Modulo PURO (stdlib) con los sectores que quedan SIN features sectoriales (Real Estate n=3, Utilities n=1) y la marca "Sin contexto sectorial". FUENTE UNICA: la importan el productor (`sector_features` arma su WHERE desde la constante), `feature_calculator` (las 11 columnas), el scanner, Telegram y el MCP. La marca se DERIVA del sector en cada lectura -- sin columna nueva y retroactiva sobre toda la historia de `alertas_scanner` |
| `src/utils/estado_pipeline.py` | Modulo PURO del diagnostico de la rutina (sin DB ni Streamlit). Registro de tablas -> etiqueta / si es INSUMO de decisiones / que .bat la arregla, mas `diagnosticar()` y `resumen()`. FUENTE UNICA: lo comparten chequeo_rutina.py y la banda de estado del dashboard, para que no haya dos definiciones de "estan alineados los datos". **`Tabla.columna` es SIEMPRE la fecha de DATOS**; el reloj de corrida va aparte en `columna_registro` y se informa pero NO entra en el diagnostico (ver patrones criticos: incidente 2/9/2026) |
| `src/indicators/estructura.py` / `src/indicators/velas.py` | Modulos PUROS (numpy+pandas) de estructura de mercado con swings CONFIRMADOS (el swing existe N barras despues de su barra; sin swings provisionales) y patrones de vela con definicion clasica y contexto. INVARIANTES por test: la fila de una fecha no cambia con barras nuevas. Reemplazo en paralelo de market_structure.py (historia con futuro) y de los patrones de precio_accion.py. Sirven para diario y semanal (`tope_dias`). Los usa ya el semanal de mtf_context y del dashboard. Ver docs/estructura_velas.md |
| `scripts/compute_estructura_velas.py` | Escribe `features_estructura` + `features_velas` (LOCAL). Calcula sobre la historia completa de cada ticker (segundos) y persiste desde la primera rueda que falta, con 10 de solape; ticker sin filas se escribe entero. `--crear --completo` (carga inicial) / `--tickers` / `--dry-run` / `--status`. Paso 2c de cron_diario (no frena si falla); `splits.py corregir` lo corre con `--completo` |
| `scripts/ml/entrenar_ml_v3.py` | Fase 3a de la Tarea 23: entrena y valida el modelo v3 sobre `features_estructura` con la particion y compuertas PRE-REGISTRADAS (doc sec. 9.5). Walk-forward purgado de 6 folds en el 80% de desarrollo, dos brazos (53 vs 29 features), lockbox del 20% final que solo se abre si pasa la compuerta 1. `--solo-wf` / `--dry-run` / `--estructura vieja` (control de diagnostico). RESULTADO: no pasa, no hay v3, lockbox sin abrir |
| `scripts/ml/screen_sectorial_v3.py` | Paso 1 de la pregunta sectorial (solo lee las predicciones fuera de muestra del walk-forward): AUC/lift/exceso por sector con IC95 calculado sobre las 6 mediciones POR FOLD (la fila no es unidad independiente: dentro de un fold comparten mercado) + prueba de heterogeneidad (dispersion entre sectores vs dentro del sector). Sale 2 si ningun sector califica. RESULTADO: ninguno |
| `scripts/ml/auditar_invariancia_features.py` | Auditoria numerica de las 29 features NO estructurales (solo lee): invariancia `calcular(datos[:t+1]).iloc[-1] == calcular(datos).iloc[t]` y skew de ventana (dataset con historia completa vs scanner con las ultimas 500 barras). Las dos dan diferencia 0,00e+00 |
| `scripts/ml/analizar_features_ml.py` | Mediciones de docs/features_ml.md (solo lee): inventario de las 71 tablas, historia real por ticker, cobertura de cada fuente sobre el dataset, redundancia, composicion de los folds, base rate por trimestre (label absoluto vs relativo) y familia valor/PER contra los dos labels. `--seccion ablacion` (~20 min) quita cada familia del set de 53 y re-corre los 6 folds registrados (importa particion y folds de entrenar_ml_v3). IC95 sobre las mediciones POR FOLD |
| `scripts/manual/auditar_features_tablas.py` | Audita features_velas / features_estructura / features_precio_accion contra el OHLCV (solo lee): reproducibilidad, definiciones con implementacion INDEPENDIENTE, swings reales + invariancia, valores inventados. Sale 0 si velas y estructura pasan todo y las 3 reproducen; los defectos de precio_accion se informan como CONOCIDOS. Correrlo despues de un backfill o de `splits.py corregir` |
| `scripts/ml/medir_leakage_estructura.py` / `scripts/ml/medir_valor_velas.py` | Mediciones de la Tarea 23 (solo leen): tabla vieja vs lo que se sabia ese dia (loop por rueda, ~13 s/ticker), impacto en AUC de ML v1/v2, semanal, valor de eventos con swings confirmados y regla de entrada de FT_SMC (`--solo FG`, compuerta de la Fase 2); patrones de vela viejos vs clasicos |
| `scripts/compute_veredictos_universo.py` | Precomputa el veredicto sintetico de los ~200 tickers a `veredictos_universo_diario` (LOCAL). El screener del dashboard lo calculaba EN VIVO: 121 s medidos, cache solo en memoria del proceso Streamlit. Ahora lee la tabla: 323 ms. Idempotente (UPSERT), `--dry-run` / `--status`. Paso final de ft_run_diario.bat, DESPUES de [0b] (el veredicto vota con opciones_pcr_plazo_diario) |
| `scripts/manual/splits.py` (detectar/corregir) | Deteccion y correccion de splits no aplicados en precios_diarios. 2 etapas (barrido local + verificacion Yahoo). Corrige por divisor, REGISTRA el split en `splits_aplicados` (misma transaccion, fecha real de Yahoo o `--fecha-ejecucion`) y recomputa indicadores/features/z-scores. Ver "Splits" en Patrones criticos |
| `scripts/forward_testing/ft_bot_smc_v3.py` (+ `ft_scoring_estructura.py`) | Estrategias FT_SMC_v3_N5 / FT_SMC_v3_N3 (`--ventana 5\|3`): la MISMA regla de FT_SMC_v1 (el score se importa de `ft_scoring`, no se reimplementa) leyendo `features_estructura` (swings CONFIRMADOS) + `features_velas`. El loader aliasa las columnas de N a los nombres `*_10` que espera el score, ancla el lookback a la ultima rueda de DATOS (no al reloj) y valida que la ventana este persistida. Ver docs/forward_testing/estrategias/SMC_v3.md |
| `scripts/oneshot/discontinuar_estrategias_ft.py` | Baja de una estrategia FT (generico): liquida las posiciones abiertas al ultimo cierre con `motivo_salida=ESTRATEGIA_DISCONTINUADA`, `activa=FALSE` y imprime el cierre en numeros para la ficha. `--dry-run` / `--estrategias`. Se uso el 17/9/2026 con FT_COMBO_v1 y FT_SMC_v2. Despues: `ft_compute_equity.py --rebuild`, `ft_cambios.py add` y sacar el bot del .bat |
| `scripts/forward_testing/ft_compute_equity.py` | Reconstruye la equity MARCADA A MERCADO (`ft_equity_diaria`) desde ft_operaciones + precios_diarios. Idempotente, `--rebuild`/`--check`. Control de cuadre del cash contra ft_estrategias |
| `src/utils/ft_metricas.py` | Modulo PURO de metricas de riesgo (max DD, Sharpe con IC95%, Sortino, IR, beta) y de trade (expectancy, profit factor, payoff). Sin DB ni config |
| `src/utils/ft_tramos.py` | Modulo PURO para MEDIR UN CAMBIO: corta la historia de cada estrategia en tramos por `ft_cambios` y compara antes/despues contra un GRUPO DE CONTROL (las no afectadas, mismos dias) y contra el universo, con IC95 de Welch; expectancy en diferencia-en-diferencias. INSUFICIENTE sin numero por debajo de 20 ruedas / 10 ops por lado. Tambien `ic95_bootstrap` (Sortino). Ver docs/forward_testing/METRICAS.md sec. 12 |
| `scripts/forward_testing/ft_cambios.py` | CLI del registro `ft_cambios`: `add` (valida dia habil/estrategias/clave; avisa si ya hubo corridas con ese dato o si queda sin control; `--dry-run`) y `list`. Registrar ANTES de desplegar cualquier cambio que toque decisiones de FT |
| `scripts/forward_testing/ft_foto_base.py` | Foto de BASE congelada por rueda (`reportes/ft_foto_base_<rueda>.json/.md`): riesgo con Sortino IC95 por bootstrap y Sharpe IC95, metricas de operacion, tramo vigente y cambios. No pisa salvo `--forzar`. Primera: rueda 2026-09-11 |
| `src/ml/ml_v2.py` | Modelo ML v2 del scanner (Etapa 3c, 13/9/2026). Carga VALIDADA contra `models_ml_v2/metadata.json` (sha256 + orden de features), `matriz_x` (las features de market structure vacias van a 0, como al entrenar), `prob_v2`, `cortes_equivalentes` (cada corte de probabilidad de la v2 deja arriba la misma fraccion de filas que el de la v1 sobre las mismas filas). Sin DB |
| `scripts/ml/entrenar_ml_v2.py` | Entrena la v2 con la config congelada de la Tarea 20 (RF global + isotonica, label absoluto). Holdout 126 ruedas + embargo 20 con COMPUERTA fijada antes de ver resultados, cortes contra la v1, reentrenado final. Artefacto `models_ml_v2/rf_cal_global.joblib` FUERA de git + `metadata.json` EN git. `--dry-run` / `--forzar`. La v2 queda congelada durante la Etapa 4. Ver docs/ml_reentrenamiento.md sec. 8c |
| `scripts/push_senales_bot.py` | SIN USO desde el 13/9/2026 (bots Alpaca apagados; se conserva por si se reactivan). Productor de la tabla masticada senales_bot_diaria (Plan B). Lee LOCAL (tecnico/scanner/PCR_VOL), UPSERT a RAILWAY. Conexion dual. Hermano de FT (paso final de ft_run_diario.bat). Standalone via push_senales_bot.bat |
| `scripts/alpaca/bot_ml.py` / `bot_tech_sector.py` / `bot_options.py` | Los 3 bots Alpaca Plan B, APAGADOS el 13/9/2026 (workflows deshabilitados). Leen la masticada, deciden con el cerebro src/strategies/, ejecutan via src/trading/ejecucion_bot. `--dry-run` / `--ignore-frescura`. Ver docs/bots_alpaca.md |
| `src/strategies/` | Cerebro de decision COMPARTIDO FT<->Alpaca (PURO): scoring (calcular_score_tecnico), sectorial (v1/v2), ml_scanner |
| `src/trading/senales_adapter.py` / `ejecucion_bot.py` | Adapters Alpaca: data (masticada->cerebro) + ejecucion (alpaca_client + posiciones_bot*/operaciones_bot*) |
| `scripts/forward_testing/ft_reporte_html.py` | Reporte HTML autocontenido de FT (reportes/ft_reporte.html). Incluye la seccion "Antes y despues de cada cambio" (lee `ft_cambios`, calcula con `ft_tramos`): un bloque por cambio que corta, tramo vigente por estrategia y marcas. Y la seccion "ML v1 vs v2: por que difieren" (`ft_comparar`), que si falla deja el aviso sin tumbar el reporte |
| `src/utils/ft_comparar.py` | Modulo PURO de la Etapa 3f: POR QUE DIFIEREN FT_ML_SCANNER_v1 y v2 (no cual rinde mas). Senales sobre las MISMAS filas de alertas_scanner (ambas / solo v1 / solo v2 por rueda, Jaccard, retorno real a 5/20 ruedas y exceso contra el universo de la rueda; decide exclusivas v2 contra exclusivas v1), atribucion (tickers fuera del entrenamiento de la v1, nivel que dio la otra version, sector), operaciones compartidas/exclusivas, oportunidades que dejo afuera el tope y cartera pareada. INSUFICIENTE sin numero: 10 senales EN 5 RUEDAS distintas (14 exclusivas de una sola rueda son una sola observacion de mercado) / 10 ops / 20 ruedas. `tablas()` = el texto comun del .md y el HTML. Ver docs/forward_testing/METRICAS.md sec. 13 |
| `scripts/forward_testing/ft_analisis_salidas.py` (+ `src/utils/ft_salidas.py`) | Analisis de SALIDAS de FT (solo lee; docs/forward_testing/ANALISIS_SALIDAS.md). `panorama` = que hizo el precio despues de cada salida, contra el universo y en desvios del ticker, IC95 por dia, clasificacion a tiempo/temprano/indiferente vs salir al azar; `balances` = eventos unicos contra todos los balances del tramo; `tech_sector_v1` = combinaciones de la regla, ventana del bug del score 0,0 y que condicion disparo cada salida. Excluye las ventanas de `ft_cambios` que invalidan la historia. Escribe en `reportes/analisis_salidas/AAAAMMDD_<etiqueta>/`. `p0` (la re-simulacion reproduce a FT), `p1` (quitar condiciones) y `p2` (grilla de pesos: 589 reglas re-simuladas vectorizadas, con control interno contra la maquina del paso 0; seleccion 2021-24 / confirmacion 2025-26 / FT de control) son los pasos pre-registrados de TECH_SECTOR_v1. El modulo puro tiene la regla de la v1 como funcion de sus 5 condiciones, las variantes del paso 1 (`VARIANTES`) y la grilla del paso 2 como mascaras de 32 estados (`reglas_grilla`) |
| `scripts/forward_testing/ft_analisis_salidas_smc.py` (+ `src/utils/ft_salidas_smc.py`) | Analisis de SALIDAS de FT_SMC_v1 (solo lee; ANALISIS_SALIDAS.md sec. 10). Reconstruye rueda por rueda lo que el bot veia en `features_market_structure` (su historia mira al futuro): modulo viejo sobre las ultimas 250 barras, en paralelo, ~26 min, copia en `reportes/analisis_salidas/cache/` que se reusa mientras no haya rueda nueva (`--recalcular`). `--seccion p0` (fidelidad contra FT + anatomia) / `grilla` (96 combinaciones de stop, CHoCH, estructura rota y time stop; todas las senales, muestra con tope de 5 y FT de control) / `todas`. Corta la re-simulacion donde `earnings_historico` deja de estar completa (`fin_balances`). El modulo puro tiene la regla del bot con sus prioridades (`primera_salida`, referencia) y la version vectorizada (`salidas_reglas`), verificadas iguales por test y en cada corrida |
| `scripts/forward_testing/ft_comparar_ml.py` | Carga y reporte de la comparacion v1 vs v2 (`reportes/ft_comparar_ml.md`; `--desde`, default el inicio de la v2). Solo lee. Los retornos salen de `precios_diarios` por `precio_fecha`, NO de `retorno_Nd_real` (sin llenar desde mayo). Entrenados en la v1 = tickers con precio hasta fin de 2021 (123; `modelo_asignado` da 125). La seccion del HTML usa su `cargar_insumos()` |
| `scripts/refresh_earnings_calendar.py` | Refresh earnings_calendar desde Nasdaq (cron Oracle semanal) |
| `scripts/refresh_earnings_historico.py` (+ `scripts/manual/refresh_earnings_historico.bat`) | Puebla earnings_historico (fecha de anuncio por Q) desde Alpha Vantage. REANUDABLE y cuota-aware (key free 25/dia, 5/min): `--backfill` y el incremental comparten UNA cola (sin historia primero, despues del mas atrasado al menos), `--ticker X` (alta), `--status` (informa la cobertura real), `--target local\|railway`. Desde el 19/9/2026 corre SOLO como ultimo paso de la rutina (paso `earnings`, INFORMAR: nunca frena; ~4,5 min por las pausas de 13s). Ver docs/earnings_reaccion.md |
| `src/utils/earnings_cobertura.py` | Modulo PURO (stdlib): quien DEBE un balance y cuan al dia esta `earnings_historico`, por CADENCIA propia de cada ticker (mediana de dias entre sus anuncios, margen 15%) y no por antiguedad absoluta. `cadencia` / `estado` / `cobertura` / `a_traer` / `resumen`. FUENTE UNICA del script que la puebla y de la vista del dashboard. Ver el patron critico "Una tabla de EVENTOS se vigila por CADENCIA" |
| `dashboard/earnings_reaccion.py` | Vista "Reaccion a balances": ventana simetrica pre+post (N ruedas por lado, 1-10) alrededor del balance. 3 paneles (precio USD, precio %, volumen x prom 50). Filtros por anio y trimestre (Q1-Q4). Dia 0 ajustado por pre/post-market. Muestra la COBERTURA de la tabla y avisa si al ticker elegido le falta un balance (`_aviso_cobertura`, via src/utils/earnings_cobertura): un trimestre que falta no se distingue de uno que no existe |
| `scripts/manual/refresh_fundamentales.bat` | Refresh fundamentales (income/balance/cashflow/valuation) desde yahooquery. LOCAL-only, manual. ~3.5 min. Encadena 5 pasos derivados: ratios -> ticker_pais -> vs_sector -> **multiplos_px -> vs_sector --valuacion-px** (los 2 ultimos agregados 27/8/2026: sin ellos el trimestre nuevo queda sin `*_px` y el dashboard muestra la valuacion vacia). `set REFRESH_NO_PAUSE=1` para correrlo desatendido |
| `scripts/refresh_fundamentales.py` | Motor del refresh fundamentales (4 tablas, 8 Q, UPSERT con restatements) |
| `scripts/manual/refresh_fundamentales_sec.bat` (+ `scripts/refresh_fundamentales_sec.py`) | Refresh de la fuente SEC XBRL (PARALELA a yahooquery, LOCAL-only, ~147 tickers USA). INCREMENTAL: consulta `submissions` (~164 KB) y solo baja `companyfacts` (~4 MB) si cambio el accession del ultimo 10-Q/10-K -> sin balances nuevos mueve ~24 MB en vez de ~522 MB. REQUIERE `SEC_USER_AGENT` en el .env (SEC devuelve 403 sin User-Agent con mail de contacto). `--solo-normalizar` / `--forzar` / `--tickers` / `--dry-run`. ENCADENA 2 pasos derivados (refresh_acciones_circulacion + compute_sec_multiplos completo), solo si el refresh anduvo; `set SEC_NO_DERIVADOS=1` los saltea (necesario con --solo-normalizar, que es offline). Ver docs/fuentes_fundamentales.md |
| `src/utils/sec_xbrl.py` | Normalizador PURO de SEC XBRL -> serie trimestral (stdlib, sin DB/red). Resuelve sinonimos por concepto, tags que cambian dentro de la misma empresa, desacumulacion YTD (Q2=H1-Q1, Q3=9M-H1, Q4=FY-9M) y restatements. Etiqueta fiscal_year/fiscal_quarter reales. `hasta_filed` = point-in-time. Emite avisos como red contra el error silencioso. Incluye la identidad `net_income = ProfitLoss - minoritarios` para los 8 filers que no tagean NetIncomeLoss (MA/CAT/SCCO/AVAV/AVGO/F/FCX/AMT): solo rellena huecos, exige el hecho de minoritarios EN ESE PERIODO (nunca asume cero) y cruza contra ...AvailableToCommonStockholders. Acepta `tags_curados={concepto: tag}` por ticker (el tag curado REEMPLAZA la lista de sinonimos, no se antepone: preferimos el hueco visible al numero mezclado invisible). Aviso `mezcla_en_ejercicio` = los Q de un mismo ejercicio salieron de tags distintos Y sus anuales difieren; si coinciden son sinonimos y calla (109 de 126 mezclas son inocuas). Ver docs/fuentes_fundamentales.md sec. 14 y 16 |
| `src/utils/fundamentales_ttm.py` | TTM rodante PURO sobre la serie SEC + as-of por `filed_primero`. Rechaza ventanas que cruzan un hueco (un rolling(4) sobre serie con huecos suma 5-6 trimestres en silencio). Deriva ebitda/fcf/net_debt/bvps |
| `src/utils/sec_acciones.py` | Serie POINT-IN-TIME de acciones desde la portada `dei` (nunca re-expresada por splits). Descarta errores de unidad y picos; invariante `fecha > filed` |
| `src/utils/acciones_series.py` | Combina yahooquery (base de split ACTUAL) + los respaldos point-in-time (SEC / Polygon, base de su momento) y VALIDA que coincidan antes de mezclarlas. Yahoo manda donde llega; los otros solo EXTIENDEN hacia atras; ESCALON nunca interpolacion; la discrepancia AVISA, no corrige. `rebasar()` lleva una serie a la base de hoy con los splits de Polygon y **corta por `filed`, no por la fecha de periodo** (un numero esta en la base vigente cuando se PRESENTO: GOOG declara 658 MM al 2022-03-31 y 13.078 MM al 2022-06-30, las dos previas al split del 18/7). `TOL_BASE=0.12` medido, no a ojo. Ver docs/arquitectura_fuentes.md sec. 7 |
| `scripts/refresh_acciones_circulacion.py` | Puebla `acciones_circulacion`. Base yahooquery + 4 candidatos de extension probados EN ORDEN (portada / balance / promedio diluido de SEC, y Polygon ultimo): se queda con el primero que VALIDA, nunca elige a ciegas. Cada nivel se prueba sin rebase y con rebase -- Polygon mezcla splits reales con ajustes de precio por spinoff (HON 1,061 = Solstice) y no hay como distinguirlos por el ratio, asi que decide el validador. LOCAL-only. Usa yfinance_lock |
| `scripts/refresh_polygon.py` (+ `src/data/polygon/`) | Ingesta Polygon: `--cobertura` (existe el ticker), `--splits` (lista AUTORITATIVA de ratios), `--acciones` (conteo por fecha, la mas cara: 1 pedido POR FECHA). REANUDABLE via `polygon_ingesta` (distingue "no pedido" de "pedido, sin resultados"); cupo por ventana deslizante 4/min. Key `POLYGON_APY_KEY`. Usa `weighted_shares_outstanding` (TOTAL), NUNCA `share_class_shares_outstanding` (UNA clase: en V da 0,79 del total) |
| `scripts/compute_sec_multiplos.py` | Serie DIARIA de multiplos sobre la fuente SEC (`fundamentales_sec_multiplos_d`). Capa derivada pura y recomputable. Percentil trailing ESTRICTO (exige ventana llena en tiempo); `--percentil-permisivo` la afloja. `--incremental` (paso diario en recovery_incremental) escribe solo la rueda nueva pero CALCULA la serie entera -- el percentil es rodante de 756 ruedas; no propaga restatements hacia atras, de eso se encarga la corrida completa del .bat. Motor expuesto como `computar()` |
| `scripts/manual/valuacion_implicita.py` | CALCULADORA de escenarios sobre la fuente SEC (solo lee). Toma una tesis de precio (`--precio` / `--variacion`) y responde 3 cosas: que multiplos implica, en que percentil de la propia historia caen, y **cuanto tiene que crecer el negocio** para que ese precio sea un multiplo normal (`--referencia`, default mediana) contra lo que la empresa logro alguna vez. NO predice retornos. Ver docs/fuentes_fundamentales.md sec. 17 |
| `src/utils/valuacion_implicita.py` | Motor PURO de la calculadora (stdlib, sin DB). `multiplos`/`precio_para` (mover el precio) y `denominador_para`/`exigencia` (mover el negocio): UNA sola incognita por escenario. `crecimiento_historico` mide sobre la serie TRIMESTRAL, no la diaria (el TTM es una escalera: por rueda el n queda inflado x63). `roe_implicito` es un TECHO (patrimonio de hoy). ROIC/ROTCE ausentes a proposito: piden columnas que la capa derivada no tiene, y ninguno se mueve con el precio |
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
- `features_precio_accion` | `features_market_structure` (OJO: su historia mira 10
  ruedas al futuro, ver patrones criticos). En `features_precio_accion` los PATRONES de
  vela estan mal definidos y `tendencia_velas` / `rango_expansion` inventan valores en las
  costuras de backfill (auditado, docs/estructura_velas.md sec. 12); el flujo de volumen
  y la microestructura de la vela si estan bien
- `features_estructura` | `features_velas` (LOCAL, 17/9/2026) -- reemplazo en PARALELO
  de las dos anteriores, sin informacion futura (invariantes). Las 24 columnas de
  estructura con swings confirmados (`is_sh_N` = se confirmo HOY un swing) para N=5 y
  N=10, mas 12 de N=3 (36 en total; `estructura.VENTANAS_TABLA`, agregadas para
  FT_SMC_v3_N3/N5), y 11 patrones clasicos con contexto (hammer/hanging_man,
  shooting_star/inverted_hammer, envolventes que envuelven, marubozu con direccion).
  PK (ticker, fecha). Las escribe el Paso 2c (`compute_estructura_velas.py`); agregar
  una ventana nueva NO rehace la tabla (`--crear` hace ADD COLUMN IF NOT EXISTS, y
  despues `--completo` rellena). Consumidores de decision: FT_SMC_v3_N5 y FT_SMC_v3_N3
  (via `ft_scoring_estructura.py`); el scanner, los modelos v1/v2 y el resto de FT
  siguen en las tablas viejas
- `alertas_scanner` (col: `scan_fecha`, `precio_fecha`). Desde el 13/9/2026 (Etapa 3d)
  lleva ademas `ml_prob_v2` / `ml_modelo_v2` / `alert_score_v2` / `alert_nivel_v2`:
  el modelo ML v2 calculado EN PARALELO en la misma fila (mismas senales de price
  action, score tecnico y bajistas; cambian la probabilidad y los cortes). NULL en la
  historia previa y si el artefacto no carga. Las lee FT_ML_SCANNER_v2
- `ticker_zscore_diario` | `opciones_zscore_diario`
- `opciones_snapshot` | `opciones_resumen_diario` -- en RAILWAY, opciones_snapshot
  tiene RETENCION de 10 dias (purga verificada post-sync, 20/7/2026); la historia
  completa vive en LOCAL. Sin retencion crecia ~19 MB/dia y detuvo Railway por
  limite de consumo (incidente 20/7).
  **TECHO DE HISTORIA: las dos arrancan el 2026-04-18** (99 ruedas al 14/9/2026).
  Lo previo es irrecuperable (Yahoo solo expone la chain vigente) -> **no hay base
  de 52 semanas para opciones hasta ~abril/2027**. Cualquier ventana de referencia
  de opciones queda en ~40-60 ruedas y dentro de UN regimen: declararlo. Al cruzar
  opciones con precio/volumen de la accion (1.300+ ruedas), esta es la limitante.
- `opciones_sector_zscore_diario` (PCR_vol+vol agregados por sector, z-score)
- `opciones_pcr_plazo_diario` (PCR vol/OI + muros S/R por ventana corto/medio/largo,
  por ticker; fuente src/utils/opciones_plazo.py). `precio_sub` = precio de
  REFERENCIA y `precio_fuente` dice de donde salio ('precios_diarios' |
  'precios_x_split' | 'snapshot');
  idem en `opciones_resumen_diario` (10/9/2026)
- `opciones_sector_pcr_plazo_diario` (PCR sectorial por ventana + z-score)
- `indicadores_tecnicos_1w` (RSI/MACD semanal) -- CONGELADA 2026-04-02: pipeline
  semanal (scripts 23-30) deprecado 28/5/2026 (Plan C), movido a scripts/legacy_1w/.
  El timeframe semanal se calcula AL VUELO desde precios_diarios (dashboard,
  mtf_context y el MCP get_ticker_sintesis). Ningun flujo vivo la lee ya:
  src/utils/weekly_tf.py es la fuente unica del RSI/MACD semanal (29/5/2026).
- `futuros_diarios` | `indicadores_tecnicos_futuros`
- `features_regimen_macro` | `features_ml` | `features_sector` (z-scores del ticker
  contra su sector; INSUMO del scanner ML, 11 de las 53 features. La actualiza el
  Paso 2 desde el 13/9/2026: ultimas 10 ruedas. Ver patrones criticos)
- `earnings_calendar` (ticker PK, earnings_date DATE NULL; refrescada semanal
  desde Nasdaq por `refresh_earnings_calendar.py`)
- `earnings_historico` (LOCAL) -- fecha de anuncio de cada balance por trimestre
  (ticker, fiscal_period_end, announcement_date, report_time pre/post-market).
  PK (ticker, fiscal_period_end); JOIN con fundamentales_*_q por esa clave.
  Fuente Alpha Vantage EARNINGS (backfill reanudable/cuota-aware, key free
  25/dia). Base de la vista "Reaccion a balances" del dashboard. La variable que
  faltaba: earnings_calendar solo tiene la proxima fecha y fundamentales tiene
  el CIERRE fiscal, no el anuncio. NO es insumo de decisiones: el filtro de
  balances de los bots lee earnings_calendar. La pone al dia sola el paso
  `earnings` de la rutina; su atraso se mide por CADENCIA propia
  (src/utils/earnings_cobertura), no por antiguedad. Ver docs/earnings_reaccion.md
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
  EV/EBITDA VALIDADO 29/8/2026 (doc sec. 16.5-16.7). Dos defectos silenciosos
  corregidos: (a) `Depreciation` a secas salio de los sinonimos de d_and_a --
  es un SUBCONJUNTO (mediana 73% de la D&A completa), subestimaba el EBITDA y
  sobreestimaba el multiplo; mezclas 476 -> 2. (b) MAS GRAVE: el EV se
  calculaba con deuda PARCIAL. La API de companyfacts descarta los hechos
  dimensionados y 52 tickers no entregan `debt_long` (VZ daba 19.479 MM en vez
  de ~150.000; T daba deuda neta NEGATIVA). Ahora si la empresa tagea esa deuda
  en algun periodo y no en este, net_debt = NULL y el EV desaparece.
  Acuerdo vs yahooquery: EV/EBITDA mediana 9,24% -> 6,38%, dentro del 5%
  34% -> 46%, cobertura 78 -> 50 tickers (precio buscado). **El EV/EBITDA de
  SEC es internamente consistente y sirve para "caro vs si misma", pero NO es
  intercambiable con el de yahooquery**: alla es NormalizedEBITDA (sin
  one-offs), aca EBIT+D&A. Hueco de cobertura real en streamers (NFLX/WBD): la
  amortizacion de CONTENIDO no esta en los tags de D&A.
  Estado de los avisos: `python scripts/manual/sec_avisos.py --defectos`.
- `acciones_circulacion` (+ `acciones_circulacion_validacion`, LOCAL) --
  acciones en circulacion por (ticker, fecha) en base de split **ACTUAL**, que
  es la unica apareable con `precios_diarios` (que se corrige retroactivamente
  por divisor). Fuente primaria yahooquery `OrdinarySharesNumber`; se extiende
  hacia atras con el primer respaldo que VALIDA -- portada / balance /
  promedio diluido de SEC, y Polygon como ultimo recurso (unico que cubre a
  los filers multiclase: en V los TRES niveles de SEC vienen VACIOS porque
  companyfacts descarta los hechos dimensionados). La tabla `_validacion` guarda el veredicto por ticker
  (extendido, ratio_min/max, motivo). Motor puro: src/utils/acciones_series.py.
  200 tickers / 2.379 puntos: 101 arrancan en 2021, 83 en 2022, 16 en 2023+.
- `splits_aplicados` (LOCAL, 12/9/2026) -- registro de splits YA reflejados en
  precios_diarios (UNIQUE ticker, execution_date; `ratio` = acciones nuevas por
  vieja; `origen` corregido | historia_ajustada; `fecha_corte_db` = primera rueda
  que la DB tuvo en la escala nueva). `execution_date` es la fecha REAL de Yahoo,
  no el corte. La escribe `splits.py corregir` en la misma transaccion que la
  correccion; la lee el factor de escala del precio de referencia de opciones
  (`opciones_plazo.cargar_factores_escala`). Es tambien el registro de sucesos
  para marcar operaciones de FT/Alpaca cruzadas por un split
  (docs/forward_testing/METRICAS.md: la frontera es fecha_corte_db).
- `polygon_splits` / `polygon_acciones` / `polygon_ingesta` (LOCAL) --
  fuente Polygon, incorporada 30/8/2026 para cerrar las acciones en
  circulacion. `polygon_splits` (UNIQUE ticker, execution_date; `ratio` =
  split_to/split_from) es la lista AUTORITATIVA que alimenta el rebase (NO el
  factor de escala de opciones, que lee splits_aplicados) --
  OJO: el endpoint mezcla splits reales con ajustes de PRECIO por spinoff
  (HON 1,061 Solstice, IBM 1,046 Kyndryl, MMM 1,196 Solventum, DELL 1,973
  VMware, GSK 0,8 Haleon), que NO mueven el conteo; filtrar por "ratio
  plausible" no alcanza porque BBD 1,1 e ITUB 1,03 son bonificaciones que
  SI cuentan, asi que el refresh prueba con y sin rebase y gana la que
  valida. `polygon_acciones` guarda los DOS campos de conteo: se usa
  `weighted_shares` (total), nunca `share_class_shares` (una clase).
  `polygon_ingesta` (UNIQUE ticker, tarea) hace la corrida reanudable.
  Los 200 tickers existen en Polygon, incluidos los 53 sin SEC -- pero
  existir da acciones y splits, NO el denominador fundamental.
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
- `ft_cambios` (LOCAL, 13/9/2026) -- registro de los cambios que afectan a las
  estrategias FT (UNIQUE clave; `fecha_efectiva`, `tipo`, `estrategias` INTEGER[],
  `cambia_decisiones`, titulo/detalle/ref). `fecha_efectiva` = primera rueda de
  DATOS con la que la estrategia decidio ya con el cambio, NO la del commit ni la
  de la corrida. Solo `cambia_decisiones=TRUE` (logica/parametros/modelo) corta
  tramos; datos, medicion, refactor e infra quedan como marca. La lee el reporte
  (`ft_tramos`). Alta con `ft_cambios.py add`; carga inicial fechada con evidencia
  en `scripts/oneshot/create_ft_cambios.py`. Ver METRICAS.md sec. 12
- `rutina_corridas` (LOCAL, 13/9/2026) -- una fila por paso ejecutado de la rutina
  diaria (sync/paso1/paso2/paso3/ft/recovery_incremental): origen rutina|suelto,
  inicio, fin, duracion, exit_code, resultado (OK/PARCIAL/ERROR/SALTEADO/
  INTERRUMPIDO/EN_CURSO), rueda de DATOS antes y despues, detalle JSONB (notas,
  tickers pendientes, huecos), log_path, git_commit. Se inserta al arrancar y se
  actualiza al terminar: si queda EN_CURSO, el proceso murio. La escribe
  `rutina_diaria.py`; responde "cuando corrio cada paso" sin reconstruirlo desde
  timestamps de filas y commits (lo que hubo que hacer en la Etapa 1)
- `senales_bot_diaria` (RAILWAY) -- SIN PRODUCTOR desde el 13/9/2026 (bots Alpaca apagados). Tabla MASTICADA Plan B para los 3 bots Alpaca
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
- `veredictos_universo_diario` (LOCAL) -- veredicto sintetico (ALCISTA/NEUTRAL/
  BAJISTA) + frase de cada ticker, precomputado por la rutina nocturna. PK
  (ticker, fecha). Capa DERIVADA y recomputable (funcion de precios/indicadores/
  opciones via dashboard.sintesis_data + dashboard_sintesis.sintetizar). OJO:
  `fecha` es la fecha de DATOS, no la de corrida -- misma convencion que
  ft_operaciones.fecha_datos; con la de corrida, cruzarla con precios_diarios
  leeria el dia equivocado en silencio. Guarda historia por fecha (~200 filas/
  dia) para poder comparar el reparto del universo contra la rueda anterior,
  que es lo que consume el bloque "Clima" de la vista Hoy del dashboard.
  La escribe scripts/compute_veredictos_universo.py. LOCAL-only (Plan C).
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

La rutina diaria NORMAL es `scripts/manual/rutina_diaria.bat` (todo en orden, con
log, registro y resumen por Telegram; retomar con `--desde pasoN`). Lo de abajo es
para reconstruir a mano. Huecos en el MEDIO de la serie (invisibles para el MAX de
fecha): `chequeo_rutina.py` los informa; relleno en
docs/checklist_recovery_manual.md, CASO E.

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
4. cron_diario --step features  (features PA/SMC + scoring_tecnico y features_sector
                                 de las ultimas 10 ruedas, insumo del scanner;
                                 2c: features_estructura/velas, no frena si falla)
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
