# Plan de Backtesting Historico — Local PostgreSQL
# Creado: 2026-05-04
# Actualizado: 2026-05-04 — estrategias reducidas a 3 (decision de sesion)
# Sesion: analisis completo pre-implementacion

---

## Contexto y motivacion

Se busca evaluar el desempeno historico de las 3 estrategias seleccionadas,
sobre un periodo conocido donde los datos son completos y verificables.
El backtesting no reemplaza el forward-testing en curso: es un complemento
que permite validar la logica de entrada/salida antes de interpretar
resultados en tiempo real.

El motor corre en local PostgreSQL para:
- Evitar latencia de red Railway en cada iteracion de desarrollo
- Aislar completamente del entorno de produccion (Railway + bots Alpaca)
- Iterar parametros libremente sin impacto en el cron diario

---

## Estrategias a backtestear

| # | Instancia | Logica | Script fuente | Tablas fuente |
|---|---|---|---|---|
| 1 | TECH_SECTOR_v1 | Tecnico sectorial, 9 sectores x $11.111 | ft_bot_tech_sectorial.py | indicadores_tecnicos, precios_diarios, activos |
| 2 | COMBO_v1 | Tecnico sectorial + candle score 5d tiebreaker | ft_bot_combo_v1.py | + features_precio_accion, features_market_structure |
| 3 | SMC_v1 | CHoCH/BOS + estructura + trailing SL | ft_bot_smc.py | features_market_structure, features_precio_accion, precios_diarios |

Orden de ejecucion: TECH_SECTOR_v1 primero (mas simple para validar logica),
luego COMBO_v1 (capa incremental), luego SMC_v1 (mas compleja).

TECH_v1 excluida: TECH_SECTOR_v1 es su generalizacion con sector partitioning.
No aporta informacion adicional correrla por separado en esta fase.

Capital inicial por instancia: $100.000

---

## Periodos de backtesting

| Etapa | Periodo | Dias habiles aprox | Proposito |
|---|---|---|---|
| Validacion inicial | 01/06/2025 - 31/12/2025 | ~145 | Verificar logica, detectar bugs |
| Ventana completa | 01/01/2024 - 31/12/2025 | ~504 | Evaluacion definitiva |

---

## Limitaciones conocidas y aceptadas

### Limitacion 1 — Filtro de Earnings deshabilitado

El filtro de earnings (src/indicators/earnings_filter.py) opera llamando a
yfinance en tiempo real para obtener la proxima fecha de earnings de cada ticker.
Para simulacion de fechas pasadas devuelve fechas del año actual (2026), no del
periodo simulado.

**Decision:** el backtesting corre SIN filtro de earnings.

Implicancias:
- Las estrategias podran mantener posiciones abiertas durante earnings historicos
- Esto hace al backtesting LEVEMENTE MAS OPTIMISTA que la realidad (en la practica,
  los earnings generan volatilidad extrema que suele activar el SL de emergencia)
- El efecto neto sobre el resultado final se considera bajo, dado que cada
  estrategia tiene salidas propias por degradacion de score, SL o time stop
- Esta limitacion se documenta en cada reporte de resultados

### Limitacion 2 — Universe de tickers

Local PostgreSQL tiene 125 tickers (universo original, pre-21/4/2026).
Los 74 tickers incorporados en Railway en abril 2026 no estan en local.

**Decision:** backtesting sobre los 125 tickers disponibles localmente.

Implicancias:
- No afecta la validez logica del backtest
- Si el universo ampliado es relevante en el futuro, se puede re-ejecutar
  luego de sincronizar datos desde Railway

### Limitacion 3 — Precio de ejecucion = cierre del dia de la senal

Heredado del diseno original de forward-testing. El precio de compra es el
cierre del mismo dia en que se genera la senal, lo cual asume que la orden
se ejecuta al final del dia. Esto es levemente mas optimista que la realidad
(ejecucion al open del dia siguiente).

**Decision:** aceptado. Es el mismo supuesto que usaran los bots en FT en vivo.
Esto hace la comparacion BT vs FT consistente.

### Limitacion 4 — Comisiones y costos de transaccion

No se modelan comisiones ni spreads. Capital virtual sin restricciones de
margin o shortfall.

**Decision:** aceptado en esta fase. El objetivo es evaluar la logica de
la estrategia, no simular un broker real.

---

## Datos disponibles (local PostgreSQL, verificado 04/05/2026)

| Tabla | Desde | Hasta | Tickers |
|---|---|---|---|
| precios_diarios | 2020-01-02 | 2026-04-28 | 125 |
| indicadores_tecnicos | 2020-10-15 | 2026-04-28 | 125 |
| features_precio_accion | 2020-10-15 | 2026-04-09 | 125 |
| features_market_structure | 2020-01-02 | 2026-04-09 | 125 |

Para Jun-Dic 2025: datos completos en todas las tablas para los 125 tickers.
Para Ene 2024 - Dic 2025: datos completos en todas las tablas para los 125 tickers.

---

## Arquitectura del motor de backtesting

### Principio fundamental

NO modificar ningun script existente (ft_bot_*.py, ft_scoring.py, ft_utils.py).
El motor de backtesting es un script nuevo que REUTILIZA la logica de scoring
pero en modo batch (datos pre-cargados en pandas, sin queries en el loop diario).

### Separacion de resultados

Los resultados del backtesting historico se guardan en tablas con prefijo bt_hist_*,
separadas de las tablas ft_* del forward-testing en vivo. Esto evita contaminar
la serie historica del FT con datos simulados.

| Tabla BT | Equivalente FT | Descripcion |
|---|---|---|
| bt_hist_estrategias | ft_estrategias | Instancias de BT con parametros y capital |
| bt_hist_operaciones | ft_operaciones | Libro de trades historicos |
| bt_hist_metricas_diarias | ft_metricas_diarias | Equity curve dia a dia |
| bt_hist_candidatos | ft_candidatos_diarios | Todos los candidatos evaluados |

### Flujo de ejecucion del runner

```
ft_backtesting_runner.py --estrategia TECH_v1 --desde 2025-06-01 --hasta 2025-12-31

1. INIT
   - Verificar que existen bt_hist_* tables (si no, crear schema)
   - Registrar instancia en bt_hist_estrategias (parametros, capital_inicial, rango)
   - Generar lista de dias habiles en el rango (trading_calendar.py)

2. BULK LOAD (una sola vez, al inicio)
   - precios_diarios      -> DataFrame pd_df      (ticker, fecha, close, open, high, low)
   - indicadores_tecnicos -> DataFrame ind_df     (ticker, fecha, sma21/50/200, rsi14, macd, atr14)
   - features_precio_accion -> DataFrame fpa_df  (ticker, fecha, es_alcista, patrones, vol_*)
   - features_market_structure -> DataFrame fms_df (ticker, fecha, estructura_10, choch_*, bos_*, dist_sl)
   - activos              -> DataFrame act_df     (ticker, sector) [solo para TECH_SECTOR y COMBO]
   Todos indexados por (ticker, fecha) para lookup O(1)

3. LOOP POR DIA (sin queries a DB dentro del loop)
   Para cada fecha en lista_dias_habiles:
     a. CIERRES: para cada posicion abierta, evaluar condiciones de salida
        segun la estrategia (degradacion score, SL, TP, time stop)
        -> actualizar estado en memoria (dict de posiciones)
     b. ENTRADAS: evaluar candidatos del dia, aplicar filtros y ranking
        segun la estrategia (score >= umbral, max_posiciones, capital disponible)
        -> agregar a posiciones en memoria
     c. METRICAS: calcular equity del dia, pnl del dia, posiciones abiertas

4. ESCRITURA FINAL (al terminar el loop)
   - INSERT bt_hist_operaciones (todos los trades cerrados + posiciones abiertas al final)
   - INSERT bt_hist_metricas_diarias (equity curve completa)
   - INSERT bt_hist_candidatos (todos los candidatos evaluados por dia)
   - UPDATE bt_hist_estrategias (capital final, fecha_fin, metricas resumen)
```

---

## Parametros de salida por estrategia (sin earnings)

### SMC_v1
| Condicion | Motivo | Prioridad |
|---|---|---|
| precio <= stop_loss_trailing | TRAILING_SL | P1 (maxima) |
| choch_bear_10 = 1 | CHOCH_BEAR | P2 |
| estructura_10 = -1 | ESTRUCTURA_ROTA | P3 |
| dias_abierta >= 20 | TIME_STOP_20D | P4 (minima) |

Sin take profit. Sin earnings filter. Trailing SL actualizado cada dia.

### TECH_SECTOR_v1
- Entry: score >= 4.0 (SMA/MACD/RSI 3 capas, max 5.5pts), precio > SMA200
- Capital: 9 sectores x $11.111, max 5 posiciones/sector, ~$2.222/trade
- Exit P1: score <= 3.5 (SCORE_DEGRADADO) — primario
- Exit P2: precio <= entrada - 2*ATR14 (STOP_LOSS_ATR) — emergencia
- Exit P3: precio >= entrada + 4*ATR14 (TAKE_PROFIT_ATR) — emergencia
- Sin cross-sector allocation. Sin earnings filter.

### COMBO_v1
Identica a TECH_SECTOR_v1 en capital y logica de salida. Diferencias de entrada:
- Tiebreaker de ranking: candle_score_5d DESC (acumulacion 5 dias)
- Exclusion si candle_score_5d < -3.0 (distribucion activa detectada)
- Sin earnings filter.

---

## Metricas de evaluacion

| Metrica | Descripcion | Tabla |
|---|---|---|
| Retorno total % | (capital_final - capital_inicial) / capital_inicial | bt_hist_estrategias |
| Drawdown maximo % | Max caida desde pico en equity curve | calculada post-loop |
| Win rate % | trades ganadores / total trades | bt_hist_operaciones |
| Profit factor | suma_ganancias / abs(suma_perdidas) | bt_hist_operaciones |
| R/R promedio | retorno_pct_promedio ganadores / abs(perdedores) | bt_hist_operaciones |
| Operaciones totales | COUNT trades en periodo | bt_hist_operaciones |
| Tiempo promedio en posicion | avg(fecha_salida - fecha_entrada) dias habiles | bt_hist_operaciones |
| Sharpe simplificado | retorno_diario_promedio / std_retorno_diario | bt_hist_metricas_diarias |

---

## Etapas de implementacion

### ETAPA A — Schema de tablas bt_hist_* en local

**Objetivo:** crear las 4 tablas de resultados en local PostgreSQL.
**Script:** scripts/backtesting_historico/bt_create_schema.py
**Flags:** --create, --drop (para resetear entre pruebas), --status

Tablas a crear:
- bt_hist_estrategias
- bt_hist_operaciones
- bt_hist_metricas_diarias
- bt_hist_candidatos

No toca Railway. No toca ft_* tables. No toca ningun bot existente.

### ETAPA B — Runner de backtesting historico

**Objetivo:** motor que itera sobre datos historicos y simula las 3 estrategias.
**Script:** scripts/backtesting_historico/ft_backtesting_runner.py
**Flags:**
  --estrategia [TECH_SECTOR_v1 | COMBO_v1 | SMC_v1]
  --desde YYYY-MM-DD
  --hasta YYYY-MM-DD
  --dry-run (sin escritura a DB, solo imprime metricas)
  --verbose (log detallado por dia)
  --reset (borra resultados anteriores de esa instancia y re-corre)

**Modulos internos:**
  - bt_data_loader.py: bulk load de DataFrames desde local PostgreSQL
  - bt_scoring.py: adaptacion de ft_scoring.py para modo batch (sin queries)
  - bt_position_manager.py: estado de posiciones en memoria (dict)
  - bt_metrics.py: calculo de metricas al finalizar

**Estructura de carpetas:**
```
scripts/backtesting_historico/
  bt_create_schema.py
  ft_backtesting_runner.py
  bt_data_loader.py
  bt_scoring.py
  bt_position_manager.py
  bt_metrics.py
```

### ETAPA C — Ejecucion del backtesting (una estrategia a la vez)

Orden recomendado: TECH_SECTOR_v1 primero (mas simple), luego COMBO_v1
(capa incremental), luego SMC_v1 (mas compleja). Validar cada una antes
de correr la siguiente.

**C1 — Validacion inicial (Jun-Dic 2025):**
```
# 1. TECH_SECTOR_v1
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia TECH_SECTOR_v1 --desde 2025-06-01 --hasta 2025-12-31 --dry-run
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia TECH_SECTOR_v1 --desde 2025-06-01 --hasta 2025-12-31
# -> revisar trades generados antes de continuar

# 2. COMBO_v1
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia COMBO_v1 --desde 2025-06-01 --hasta 2025-12-31

# 3. SMC_v1
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia SMC_v1 --desde 2025-06-01 --hasta 2025-12-31
```

**C2 — Ventana completa (Ene 2024 - Dic 2025):**
```
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia TECH_SECTOR_v1 --desde 2024-01-01 --hasta 2025-12-31
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia COMBO_v1 --desde 2024-01-01 --hasta 2025-12-31
python scripts/backtesting_historico/ft_backtesting_runner.py --estrategia SMC_v1 --desde 2024-01-01 --hasta 2025-12-31
```

Atajo via .bat: scripts/backtesting_historico/run_bt_completo.bat

### ETAPA D — Visualizacion de resultados

**D1 — Consultas SQL directas (inmediato):**
Queries en scripts/backtesting_historico/bt_analisis.sql:
- Comparacion de retorno total entre las 3 estrategias
- Equity curves lado a lado
- Top 10 trades ganadores y perdedores por estrategia
- Distribucion de motivos de salida
- Win rate por sector (TECH_SECTOR, COMBO)
- Drawdown maximo por estrategia

**D2 — Tab Streamlit (fase posterior):**
Nueva tab "Backtesting Historico" en la app Streamlit, leyendo bt_hist_*.
Compara las 3 estrategias en graficos superpuestos de equity curve, tabla
de metricas resumen y drill-down de trades por estrategia.

Esta tab se diseña DESPUES de validar que los resultados del runner son correctos.

---

## Orden de ejecucion recomendado

1. [A] Crear schema bt_hist_* local (bt_create_schema.py --create)          [COMPLETADO]
2. [B] Implementar bt_data_loader.py (bulk load correcto es lo mas critico)
3. [B] Implementar bt_scoring.py (adaptar ft_scoring.py a modo batch)
4. [B] Implementar bt_position_manager.py (estado en memoria)
5. [B] Integrar en ft_backtesting_runner.py con --dry-run
6. [C1] TECH_SECTOR_v1: dry-run Jun-Dic 2025, verificar trades
7. [C1] TECH_SECTOR_v1: run real, validar metricas
8. [C1] COMBO_v1: run Jun-Dic 2025
9. [C1] SMC_v1: run Jun-Dic 2025
10. [C2] Las 3 estrategias en ventana 2024-2025
11. [D1] Analizar con queries SQL
10. [D2] Tab Streamlit (opcional, posterior)

---

## Registro de decisiones

| Decision | Alternativa descartada | Razon |
|---|---|---|
| Local PostgreSQL, no Railway | Railway remoto | Latencia, riesgo produccion, iteracion libre |
| Tablas bt_hist_* separadas de ft_* | Reusar ft_* con flag | Evitar contaminar FT en vivo con datos simulados |
| Sin earnings filter | Fetch historico de yfinance | yfinance devuelve fechas 2026 para periodos pasados |
| Precio = cierre del dia de senal | Precio open del dia siguiente | Consistencia con diseno FT; comparacion BT vs FT valida |
| 125 tickers (local) | Sincronizar 199 desde Railway | Datos ya completos; 74 nuevos sin historia pre-2024 |
| Scripts nuevos en backtesting_historico/ | Modificar ft_bot_*.py | Principio: no modificar lo que esta funcionando |
| Bulk load en pandas | Query por dia en el loop | Performance; 504 dias x 125 tickers = ~63k filas, cabe en RAM |
