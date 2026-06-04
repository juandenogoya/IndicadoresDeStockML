# Bots Alpaca — arquitectura Plan B

Estado: ACTIVO en paper desde 2026-06-04 (Tarea 16, Pasos 1-5).
Estado operativo vivo y pendientes: `memory/bots_trading.md`.

Este documento describe la arquitectura de produccion de los 3 bots Alpaca
rediseñados (Plan B). Captura las decisiones y convenciones que NO se derivan
leyendo el codigo (mapeos heredados, contrato del seam, por que ciertas cosas
son como son).

## Por que Plan B (contexto)

Los 3 bots viejos (`scripts/30_,31_,32_`) leian tablas de mercado CRUDAS de
Railway que el Plan C dejo de alimentar (~12/5/2026) -> operaban con señales
congeladas de 3-4 semanas. Se reseteo todo a cero y se rediseño con un principio:

> **"El bot solo opera"**: recibe señales pre-computadas (una tabla masticada),
> no consulta tablas crudas ni recalcula indicadores.

## Las 3 estrategias

| Bot | Estrategia            | Cerebro (variante)        | Aisla |
|-----|-----------------------|---------------------------|-------|
| 1   | ML_SCANNER_v1         | ml_scanner                | paradigma ML |
| 2   | TECH_SECTOR_v1        | sectorial (sin opciones)  | base tecnica sectorial |
| 3   | TECH_SECTOR_OPTIONS_v2| sectorial (gate PCR_VOL)  | aporte del dato de opciones |

$100k por bot, shares enteras (no fraccional), homologables a los bots de
Forward Testing del mismo nombre. **v1 vs v2 es un experimento controlado**:
entrada/salida identicas salvo el gate de opciones -> mide el valor del dato
de opciones en paper real.

## El seam: logica COMPARTIDA FT <-> Alpaca (Opcion B)

Un solo "cerebro" de decision, consumido por dos lados. Garantiza que FT (que
corre en local sobre la DB) y Alpaca (que corre en GH Actions sobre la masticada)
**decidan identico** -> la comparacion $100k FT vs $100k Alpaca es real.

Tres capas:

```
 1. ACCESO A DATOS (cada lado el suyo)
      FT     -> lee DB local (indicadores/precios/scanner/opciones)
      Alpaca -> lee la masticada senales_bot_diaria (src/trading/senales_adapter.py)

 2. DECISION (COMPARTIDA, el nucleo)  ->  src/strategies/  (PURO, sin DB ni broker)
      scoring.py      calcular_score_tecnico (SMA/MACD/RSI, 3 capas, max 5.5)
      sectorial.py    evaluar_entradas_sectorial / evaluar_cierres_sectorial
                      (una implementacion para v1 y v2, seleccionada por ConfigSectorial)
      ml_scanner.py   evaluar_entradas_ml / evaluar_cierres_ml (sizing incluido)
      El CONTRATO del seam = la forma de los dicts, que calza con las columnas
      de senales_bot_diaria.

 3. EJECUCION (cada lado el suyo)
      FT     -> ft_* (operaciones virtuales)
      Alpaca -> alpaca_client + posiciones_bot*/operaciones_bot* (src/trading/ejecucion_bot.py)
```

`ft_scoring.py` RE-EXPORTA `calcular_score_tecnico` desde `src/strategies/scoring.py`
-> los ~11 importadores existentes (bots FT, BT) siguen sin tocarse. La direccion
de dependencias es **scripts -> src** (nunca al reves).

## La masticada: `senales_bot_diaria` (Railway)

1 fila por (ticker, fecha), PK `(ticker, fecha)` (unique index FULL -> ON CONFLICT).
~18 columnas, la unica tabla de mercado US que los bots leen de Railway:

```
ticker, fecha, close, sector                          identidad + precio + grupo
alert_nivel, alert_score                              ML (bot 1)
sma21, sma50, sma200, rsi14, macd, macd_signal, atr14 tecnico (bots 2,3)
pcr_score (0-3), pcr_valido                            opciones decision (bot 3)
pcr_corto, pcr_medio, pcr_largo (A/B/null)            opciones detalle (bot 3)
```

SIN columnas SMC (estructura/choch/bos/patrones): ninguna de las 3 estrategias
elegidas usa estructura.

### El productor: `scripts/push_senales_bot.py`

Corre en LOCAL (Windows), CONEXION DUAL:
- LEE de la DB local (fuente de verdad bajo Plan C): indicadores_tecnicos,
  precios_diarios, activos, alertas_scanner, opciones_snapshot.
- ESCRIBE en Railway (`senales_bot_diaria`), engine dedicado construido desde
  `DATABASE_URL` de `.env.local` (no via get_engine() global).

Auto-inicializa la tabla (`CREATE TABLE IF NOT EXISTS`). Enganchado como paso
final de `ft_run_diario.bat` (ahi la data local ya esta fresca, incluido
opciones que el paso [0] de FT sincroniza de Railway). Standalone:
`scripts/manual/push_senales_bot.bat` re-pushea sin re-correr FT.
`_pcr_vol_por_ventanas(engine, tickers)` replica la logica de PCR_VOL de
`ft_bot_tech_sectorial_options_v2`.

## Los adapters Alpaca (`src/trading/`)

- **senales_adapter.py** (data): lee la masticada y arma los inputs del cerebro
  (`indicadores` = las filas tal cual; `pcr_map`; `ml_senales` filtradas;
  `precios` = close; `sector`). En GH Actions get_engine() -> Railway (no hay
  conexion dual de este lado). `fecha_esperada_hoy()` y `senales_frescas()` =
  guard de frescura.
- **ejecucion_bot.py** (ejecucion): generaliza el viejo `portfolio.py` por bot
  (`BotAlpacaConfig`: account_suffix + tabla_pos + tabla_oper). Lee/escribe estado
  y coloca ordenes. **Import de `alpaca_client` es LAZY** -> el dry-run y la
  decision NO dependen de alpaca-py (solo la ejecucion real).

## Entrypoints y mapeo bot -> cuenta -> tablas

`scripts/alpaca/bot_ml.py | bot_tech_sector.py | bot_options.py`. Cargan
`.env` + `.env.local` (override) -> `DATABASE_URL=Railway` + credenciales Alpaca
(en GH Actions vienen de secrets).

| Bot | cuenta (suffix) | tablas (heredadas del reset) |
|-----|-----------------|------------------------------|
| 1 ML_SCANNER_v1          | `""`  | posiciones_bot / operaciones_bot |
| 2 TECH_SECTOR_v1         | `"_2"`| posiciones_bot_tech / operaciones_bot_tech |
| 3 TECH_SECTOR_OPTIONS_v2 | `"_3"`| posiciones_bot_candle / operaciones_bot_candle |

NOTA: el sufijo de CUENTA (`""`/`_2`/`_3`, en alpaca_client) NO coincide con el
de TABLA (`""`/`_tech`/`_candle`). Los nombres `_tech`/`_candle` son HEREDADOS
de los bots viejos (technical/candle) y se reusan tal cual tras el reset; el
nombre no implica la estrategia que escribe ahi.

## Decisiones clave (no obvias)

1. **Precio de DECISION = `close` de la masticada** (homologable a FT). La
   EJECUCION usa precio vivo de Alpaca (`get_latest_price`), con fallback al
   close. La masticada lleva el close = decision + fallback.
2. **Rutina nocturna MANUAL** (decision del usuario, 4/6/2026): el productor se
   corre dentro de la rutina local nocturna (recovery -> features -> scanner ->
   ft_run_diario -> push). NO se automatiza. El guard de frescura hace que
   saltarse una noche sea SEGURO.
3. **Guard de frescura**: `fecha_esperada_hoy() = prev_trading_day(today)` (el
   bot corre intradia 15:15 UTC sobre el cierre de la rueda anterior, lag 1d).
   Si la masticada no es de esa fecha -> el bot NO opera (en vez de tradear data
   rancia). Escape hatch `--ignore-frescura`.
4. **Orden de prioridad de CIERRES difiere v1 vs v2** (preservado, NO unificado):
   - v1: earnings -> score degradado -> SL -> TP
   - v2: earnings -> SL -> TP -> score+PCR (con RETENCION_OPCIONES si pcr>=1)
   Es una diferencia historica heredada de los bots FT; se reprodujo tal cual al
   compartir el cerebro. Revisar si se quiere homogeneizar a futuro.

## Timing

- NOCHE (ART, post-cierre): rutina local -> el productor sube la masticada del
  cierre a Railway.
- DIA SIGUIENTE 15:15 UTC (GH Actions, L-V): los 3 workflows corren los bots ->
  leen la masticada -> deciden -> ejecutan en Alpaca. Lag 1 dia, igual que FT.

Workflows: `trading_bot.yml` -> bot_ml; `trading_bot_technical.yml` ->
bot_tech_sector; `trading_bot_candle.yml` -> bot_options. Secrets por cuenta:
DATABASE_URL + ALPACA_API_KEY{,_2,_3}/SECRET.

## Como correr / verificar

```
# Dry-run local (lee la masticada REAL de Railway si .env.local apunta ahi; 0 ordenes)
venv/Scripts/python.exe scripts/alpaca/bot_ml.py --dry-run
venv/Scripts/python.exe scripts/alpaca/bot_tech_sector.py --dry-run
venv/Scripts/python.exe scripts/alpaca/bot_options.py --dry-run

# Re-push manual de la masticada (sin re-correr FT)
scripts/manual/push_senales_bot.bat
```

Verificacion en produccion: GitHub Actions (3 workflows en verde) + cuentas
Alpaca paper (posiciones) + Railway `operaciones_bot*`/`posiciones_bot*`.

## Pendiente (Pasos 6-7, gateados a verificar los bots)

- **Paso 6**: DROP de tablas crudas de Railway que ya nadie consume en vivo
  (~330 MB) + retencion de opciones (snapshot 30d / derivadas 90d) + backup
  off-site (pg_dump a OneDrive). REGLA DE ORO: nada se dropea hasta confirmar que
  los bots operan OK con la masticada.
- **Paso 7**: limpiar Streamlit Cloud (confirmado sin uso).

Habilitado por Plan B: con los bots leyendo solo la masticada (chica), las crudas
de Railway dejan de tener consumidores vivos -> se pueden dropear.
