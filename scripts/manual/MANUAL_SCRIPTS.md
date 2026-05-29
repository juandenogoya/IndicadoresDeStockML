# MANUAL_SCRIPTS -- Glosario de ejecutables manuales

Referencia de los `.bat` de ejecucion manual. Actualizado 28/5/2026 (Plan C).
Para el FLUJO de recovery ver `docs/checklist_recovery_manual.md`. La tabla
canonica de scripts del proyecto esta en `CLAUDE.md`.

Plan C: **local es la fuente de verdad** (todo menos opciones). Los bats de
pipeline apuntan a local; solo los de opciones tocan Railway.

---

## Estado / diagnostico

### `status_local.bat`  (local -- el que importa)
Estado de las tablas en la DB local: precios_diarios, indicadores, features,
alertas_scanner, opciones, z-scores. Primer comando para diagnosticar.

### `status.bat`  (Railway)
Estado de Railway. Bajo Plan C sirve sobre todo para ver el snapshot de opciones.

---

## Pipeline diario (local) -- pasos individuales

Correr post-cierre NYSE (>=21:00 UTC), en orden 1 -> 2 -> 3.

### `cron_paso1_precios_yq.bat`  *(Paso 1)*
Precios EOD + futuros + indicadores tecnicos + z-scores de acciones, via
**yahooquery** (engine actual; yfinance dejo de servir confiable). Motor:
`recovery_incremental.py --target local --engine yahooquery` -> detecta tickers
atrasados (MAX(fecha)) y baja SOLO los pendientes. Idempotente.

### `cron_paso2_features.bat`  *(Paso 2)*
Features de precio-accion (`features_precio_accion`) y market structure
(`features_market_structure`). Prerequisito: Paso 1 al dia.

### `cron_paso3_scanner.bat`  *(Paso 3)*
Scanner ML sobre los 199 tickers -> `alertas_scanner` + resumen a Telegram.
Prerequisito: Paso 2 al dia. ~60 min. La tendencia MTF (1w/1m) del mensaje de
Telegram se calcula al vuelo (mtf_context), no depende de tablas semanales.

### `recovery_incremental.bat`
El motor del Paso 1 invocado directo (mismos efectos). Acepta `--dry-run`,
`--target railway`, `--engine yfinance|yahooquery`, `--skip-futuros`, etc.

---

## Opciones (Railway)

### `poblar_opciones_yq.bat`
Carga manual del snapshot de opciones US a Railway via yahooquery (1 call por
ticker, ~30x menos carga que yfinance). Idempotente. Solo valido post-cierre y
antes de la apertura siguiente (la chain vieja se pierde al abrir el mercado).

### `sync_opciones_railway_to_local.bat`
Baja a local las 3 tablas de opciones (snapshot, resumen_diario, zscore_diario)
desde Railway. Incremental, sin yfinance, sin rate limit. Es el paso 4 del flujo
diario.

---

## Sincronizacion

### `sync_local.bat`   (Railway -> local, completo)
Sync incremental de todas las tablas Railway -> local. Forward Testing NO se
sincroniza (corre 100% en local).

### `sync_to_railway.bat`   (local -> Railway)
Subir local -> Railway. Poco usado bajo Plan C (Railway solo recibe opciones,
que las escribe el cron).

---

## Forward Testing

### `ft_run_diario.bat`
Corre los 10 bots de forward testing en local + regenera el reporte HTML
(`reportes/ft_reporte.html`). Despues del Paso 3 del scanner.

---

## Modulo Argentina (IOL / BCBA)

### `scanner_ar.bat`
Scanner BCBA: historico IOL de `activos_ar`, indicadores, senales ->
`alertas_scanner_ar`. Post-cierre BCBA (>=20:00 UTC). Requiere credenciales IOL
en `.env`.

### `opciones_ar_snapshot.bat`
Chain de opciones AR desde IOL con Greeks -> `opciones_ar_gregas`. DURANTE
horario BCBA (13:30-20:00 UTC); fuera de horario los Greeks son null.

---

## Reportes (scripts/reports/)

### `make_infografia.bat <TICKER>`
Infografia PNG para X (100% datos del MCP, sin LLM).

### `build_yaml.bat <TICKER>` + `make_report.bat <yaml>`
Reporte PDF detallado con narrativa del LLM. Ver `docs/reportes.md`.

---

## Utilitarios (.py)

- `check_fecha.py` -- valida si una fecha es dia habil NYSE (exit code).
- `db_status.py --target local|railway` -- backend de los status.bat.
- `recover_opciones_tickers.py` -- recovery quirurgico de opciones de tickers puntuales.

---

## Notas

- **Rate limit por IP**: nunca correr dos scripts contra Yahoo a la vez
  (`yfinance_lock` aborta si detecta concurrencia).
- **Nunca pre-mercado**: bajar precios antes de la apertura NYSE puede etiquetar
  el cierre anterior con la fecha de hoy -> datos corruptos.
- Para el dashboard (informe descriptivo + radar): `dashboard/run_dashboard.bat`
  (corre bajo el venv). Ver `dashboard/README.md`.
