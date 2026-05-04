# MANUAL_SCRIPTS — Glosario de ejecutables manuales

Referencia rapida de todos los scripts .bat disponibles para ejecucion manual.
Ubicacion de los archivos: `scripts/manual/` y `scripts/`

---

## DIAGNOSTICO Y ESTADO

### `status.bat`
- **Tarea:** Muestra el estado actual de todas las tablas de la DB: precios_diarios, alertas_scanner, opciones_snapshot, opciones_resumen_diario, futuros_diarios, y el modulo Argentina (alertas_scanner_ar, opciones_ar_gregas). Indica si los datos estan actualizados o si falta algun dia habil.
- **Cuando ejecutar:** Antes o despues de cualquier proceso manual. Util como primer paso para diagnosticar si el cron automatico corrio correctamente.
- **Prerequisito:** Ninguno.

---

### `check_yfinance.bat`
- **Tarea:** Verifica si `yf.download()` esta disponible y que fechas retorna. No escribe nada en DB. Resultado: `[OK]` / `[PARCIAL]` / `[FALLO]`.
- **Cuando ejecutar:** Antes de lanzar el pipeline si hay dudas de conectividad con Yahoo Finance, o para confirmar que el rate limit se limpió despues de ejecutar `limpiar_cookies_yfinance.bat`.
- **Prerequisito:** Ninguno.

---

### `limpiar_cookies_yfinance.bat`
- **Tarea:** Elimina `cookies.db` y `yfinance.cache` del directorio local de yfinance. Fuerza una sesion nueva sin la cookie de rate-limit.
- **Cuando ejecutar:** Cuando `check_yfinance.bat` da `[FALLO]` desde una IP residencial (PC o celular). La cookie de rate-limit queda guardada en disco y bloquea todas las requests siguientes sin importar la red.
- **Prerequisito:** Ninguno.
- **Paso siguiente:** Cambiar de red (WiFi -> 4G o viceversa) y correr `check_yfinance.bat` para verificar.

---

## PIPELINE DIARIO (Scanner) — Pasos individuales

Usar estos bats cuando el cron automatico de GitHub Actions falla en un paso especifico y se necesita re-correr solo ese paso sin repetir todo el proceso.

### `precios_paso1_fast_info.bat` *(PREFERIDO para uso diario normal)*
- **Tarea:** Igual que `cron_paso1_precios.bat` pero usa el endpoint de cotizaciones (`/v8/finance/quote/`) en lugar del historico OHLCV (`/v8/finance/chart/`). Funciona en GH Actions y desde IPs residenciales sin disparar el rate limit agresivo de Yahoo Finance.
- **Cuando ejecutar:** Uso diario normal post-cierre NYSE (despues 21:00 UTC). Captura la sesion mas reciente cerrada (hoy si es dia habil, viernes si es fin de semana).
- **Limitacion:** No sirve para backfill de multiples dias atrasados. Para eso usar `cron_paso1_precios.bat`.
- **Prerequisito:** Ninguno. Despues de las 21:00 UTC.
- **Paso siguiente:** `cron_paso2_features.bat`

---

### `cron_paso1_precios.bat` *(FALLBACK — backfill de multiples dias)*
- **Tarea:** Descarga precios de cierre (EOD) desde yfinance para los 199 tickers via endpoint historico OHLCV. Recalcula indicadores tecnicos y actualiza futuros. Sujeto a rate limit de Yahoo Finance en IPs cloud.
- **Cuando ejecutar:** Cuando faltan MULTIPLES dias de datos (backfill). Para uso diario normal preferir `precios_paso1_fast_info.bat`.
- **Prerequisito:** Ninguno.
- **Paso siguiente:** `cron_paso2_features.bat`

---

### `cron_paso2_features.bat`
- **Tarea:** Calcula y hace upsert de las features de precio-accion (`features_precio_accion`) y market structure (`features_market_structure`) para los 199 tickers. El scanner ML del Paso 3 depende de estos datos para generar senales.
- **Cuando ejecutar:** Inmediatamente despues del Paso 1, o cuando el Paso 2 del cron automatico fallo (y el Paso 1 ya corrio correctamente ese dia).
- **Prerequisito:** `cron_paso1_precios.bat` debe haber corrido hoy con exito (precios actualizados).
- **Paso siguiente:** `cron_paso3_scanner.bat`

---

### `cron_paso3_scanner.bat`
- **Tarea:** Corre el scanner ML sobre los 199 tickers, clasifica alertas (COMPRA_FUERTE / COMPRA / NEUTRAL / VENTA), persiste resultados en la tabla `alertas_scanner` y envia el resumen diario a Telegram (comportamiento identico al cron automatico).
- **Cuando ejecutar:** Despues del Paso 2, o cuando el Paso 3 del cron automatico fallo (y los pasos 1 y 2 ya corrieron correctamente ese dia). Tiempo estimado: ~60 minutos.
- **Prerequisito:** `cron_paso2_features.bat` debe haber corrido hoy con exito.
- **Paso siguiente:** Una vez finalizado, se puede ejecutar `ft_run_diario.bat` para correr los bots de forward testing.

---

### `poblar_scanner.bat`
- **Tarea:** Ejecuta el pipeline COMPLETO en un solo proceso (equivalente a correr los Pasos 1 + 2 + 3 + 4 en secuencia). Incluye precios, futuros, features, scanner ML, Telegram y verificacion post-facto.
- **Cuando ejecutar:** Cuando el cron del dia no corrio en absoluto y se necesita re-correr todo desde cero. Si solo fallo un paso especifico, preferir los bats individuales (cron_paso1/2/3) para no repetir trabajo innecesariamente.
- **Prerequisito:** Ninguno. Corre todo.
- **Tiempo estimado:** ~105 minutos.

---

## FUTUROS DE INDICES

### `poblar_futuros.bat`
- **Tarea:** Actualiza los precios de los futuros de indices (ES=F, YM=F, NQ=F, RTY=F) en la tabla `futuros_diarios`. Muestra el estado de la DB antes de ejecutar y pide confirmacion.
- **Cuando ejecutar:** Si el cron automatico fallo en el paso de futuros, o si `status.bat` muestra que los futuros tienen mas de 3 dias de atraso.
- **Prerequisito:** Ninguno.

---

## OPCIONES

### `poblar_opciones.bat`
- **Tarea:** Ejecuta el snapshot de opciones para una fecha especifica. Incluye: (1) verificacion de que la fecha es dia habil NYSE, (2) verificacion de que yfinance tiene los datos del dia correcto, (3) DRY RUN con confirmacion, (4) carga real a DB.
- **Cuando ejecutar:** Cuando el cron automatico de opciones (21:00 UTC) fallo o no corrio. La ventana valida para datos correctos es entre las 21:00 y 03:00 UTC (post-cierre NYSE, antes del mantenimiento de Yahoo).
- **Prerequisito:** Ninguno. El script pide la fecha por input.

---

## FORWARD TESTING

### `ft_run_diario.bat`
- **Tarea:** Ejecuta los 5 bots de forward testing en secuencia: FT_ML_SCANNER_v1, FT_TECH_v1, FT_SMC_v1, FT_TECH_SECTOR_v1, FT_COMBO_v1. Cada bot evalua posiciones abiertas (cierres por degradacion de score), evalua nuevas entradas y registra candidatos del dia. Genera un log en `logs/forward_testing/ft_YYYYMMDD_HHMM.log`.
- **Cuando ejecutar:** Una vez por dia habil, despues de que el Paso 3 del scanner haya terminado (los bots usan las alertas y features del dia). Tiempo recomendado: despues de las 15:00 UTC.
- **Prerequisito:** El Paso 3 del pipeline diario (scanner) debe haber corrido hoy. Los precios del dia deben estar cargados en DB.

---

## GOOGLE SHEETS

### `actualizar_sheets.bat` *(ubicado en `scripts/`)*
- **Tarea:** Exporta los datos de alertas y scoring a Google Sheets. Corre `sheets_export.py` y muestra el resultado en pantalla.
- **Cuando ejecutar:** Manualmente, cuando se necesite actualizar el dashboard de Sheets. No tiene schedule automatico.
- **Prerequisito:** El scanner del dia debe haber corrido para que los datos exportados sean los mas recientes.

---

## PRECIOS (Task Scheduler local)

### `actualizar_precios.bat` *(ubicado en `scripts/`)*
- **Tarea:** Descarga precios EOD y los sube a Railway. Disenado para correr via Task Scheduler de Windows a las 09:00 ARG (12:00 UTC), antes del cron de GH Actions. Guarda log en `logs/actualizar_precios.log`.
- **Cuando ejecutar:** Automaticamente via Task Scheduler. No requiere intervencion manual en condiciones normales.
- **Prerequisito:** Ninguno. Lee DATABASE_URL desde `.env.local`.

---

---

## ARGENTINA (IOL) — BCBA

Scripts para el modulo de mercado argentino. Usan la API REST de InvertirOnline (IOL)
con credenciales en `.env` (IOL_USERNAME, IOL_PASSWORD). NO usan yfinance.

### `scanner_ar.bat`
- **Tarea:** Descarga historico de IOL para los tickers de `activos_ar`, calcula indicadores (SMA, RSI, MACD, ATR, ADX, estructura de mercado BOS/CHoCH), puntua cada ticker y persiste senales en la tabla `alertas_scanner_ar`. Flujo: DRY RUN con confirmacion -> carga real -> status.
- **Cuando ejecutar:** Post-cierre BCBA, despues de las 17:00 ART (20:00 UTC). El cron automatico de GH Actions corre a las 20:30 UTC.
- **Prerequisito:** Credenciales IOL en `.env`. Tablas AR creadas (`init_tablas_ar.py`).
- **Opciones:** Pide si filtrar tickers especificos o procesar todos.

---

### `opciones_ar_snapshot.bat`
- **Tarea:** Descarga el chain de opciones desde IOL para los tickers con `tiene_opciones=TRUE`, filtra strikes liquidos (IV no nula, is_stale=False) y guarda Greeks (Delta, Gamma, Theta, Vega, Rho, IV) en `opciones_ar_gregas`. Incluye advertencia de horario y opcion de forzar.
- **Cuando ejecutar:** DURANTE el horario BCBA: L-V entre 10:30-17:00 ART (13:30-20:00 UTC). Fuera de ese horario los Greeks seran null. El cron automatico de GH Actions corre a las 16:30 UTC (13:30 ART).
- **Prerequisito:** Credenciales IOL en `.env`. Tablas AR creadas. Mercado BCBA abierto.
- **Opciones:** Permite forzar ejecucion fuera de horario (responder "f" cuando pregunta).

---

## FLUJO DE CONTINGENCIA DIARIO

Secuencia a seguir si el cron automatico de GH Actions no corrio en el dia:

```
PASO 1   →  cron_paso1_precios.bat        (despues de 21:00 UTC)
PASO 2   →  cron_paso2_features.bat       (inmediatamente despues)
PASO 3   →  cron_paso3_scanner.bat        (inmediatamente despues, ~60 min)
FT       →  ft_run_diario.bat             (cuando el Paso 3 termina)
OPCIONES →  poblar_opciones.bat           (entre 21:00 y 03:00 UTC)
```

**Modulo AR (si el cron scanner_ar.yml o opciones_ar_snapshot.yml no corrio):**

```
SCANNER AR  →  scanner_ar.bat             (despues de 17:00 ART = 20:00 UTC)
GREGAS AR   →  opciones_ar_snapshot.bat  (DURANTE horario BCBA: 10:30-17:00 ART)
```

Si solo fallo un paso especifico del cron, correr solo ese paso y los que le siguen.
Usar `status.bat` para confirmar el estado antes y despues de cada ejecucion.

---

## TROUBLESHOOTING: YFRateLimitError en IP residencial

Sintoma: `yf.download()` devuelve `YFRateLimitError('Too Many Requests...')` desde
la PC o el celular, incluso cambiando de red.

Causa: yfinance 0.2.x guarda las cookies de Yahoo Finance en disco
(`%LOCALAPPDATA%\py-yfinance\cookies.db`). Cuando Yahoo devuelve un error 429,
guarda una cookie de rate-limit que bloquea todas las requests siguientes
independientemente de la red o la IP.

Solucion:

```
PASO 1  ->  limpiar_cookies_yfinance.bat
PASO 2  ->  Cambiar de red (WiFi -> 4G o viceversa)
PASO 3  ->  check_yfinance.bat  (verificar [OK])
PASO 4  ->  Lanzar el pipeline normalmente
```

Nota: este fix aplica solo a IPs residenciales. Para IPs de cloud providers
(GH Actions = Microsoft Azure, Railway = Amazon AWS) el bloqueo es permanente
a nivel de red por parte de Yahoo Finance y no tiene fix local.
