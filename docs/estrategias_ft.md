# Estrategias Forward-Testing — Especificacion
# Creado: 2026-04-23
# Branch: feature/forward-testing
#
# Convencion de nombres de instancia: {LogicaBase}_v{N}_{variante}
# Ejemplo: TECH_v1_base, SMC_v2_scoreMin2
#
# Nota sobre precio de ejecucion:
#   Todas las estrategias usan precio de cierre del dia de la senal.
#   Esto es consciente y aceptado (ver forward_testing.md).
#   No se usa precio de mercado en tiempo real (sin Alpaca).
#
# ACTUALIZACION 2026-06-04 (Plan B, Tarea 16):
#   La logica de decision de 3 estrategias se MOVIO a un cerebro COMPARTIDO
#   FT <-> Alpaca en `src/strategies/` (scoring/sectorial/ml_scanner, PURO).
#   ft_scoring.py re-exporta calcular_score_tecnico desde ahi (importadores
#   intactos). Los 3 bots Alpaca de produccion (ML_SCANNER_v1, TECH_SECTOR_v1,
#   TECH_SECTOR_OPTIONS_v2) consumen ese mismo cerebro -> deciden identico a su
#   bot FT homonimo. Detalle de la arquitectura de produccion: docs/bots_alpaca.md.
#   Este documento sigue siendo la spec conceptual de las logicas FT.
#
# ACTUALIZACION 2026-09-17 (Tarea 23, Fase 2b):
#   ALTA: ESTRATEGIA 3b (SMC sobre estructura CONFIRMADA), dos instancias
#   FT_SMC_v3_N5 y FT_SMC_v3_N3.
#   BAJA: FT_COMBO_v1 y FT_SMC_v2, discontinuadas con la rueda 2026-09-16
#   (posiciones liquidadas, activa=FALSE, bots fuera de ft_run_diario.bat).
#   El cierre de cada una, con periodo, parametros, metricas y motivos, vive en
#   docs/forward_testing/estrategias/COMBO_v1.md y SMC_v2.md.
#   Los criterios de alta y baja de una estrategia estan en CLAUDE.md,
#   seccion "Alta y baja de estrategias FT".

---

## Parametros globales de riesgo (referencia: src/trading/risk.py)

| Parametro           | Valor default | Descripcion                              |
|---------------------|---------------|------------------------------------------|
| MAX_POSICIONES      | 5             | Posiciones abiertas simultaneas          |
| RIESGO_POR_TRADE    | 15%           | Capital por operacion sobre equity total |
| MAX_EXPOSICION      | 75%           | Exposicion maxima del portafolio         |
| CAPITAL_INICIAL     | $100.000      | Capital virtual por instancia            |

Sizing por operacion (modo "fixed"):
  qty = int(capital_por_trade / precio_entrada)
  capital_por_trade = capital_actual * RIESGO_POR_TRADE

---

## Exit Condicional Transversal (aplica a TODAS las estrategias)

### Filtro Earnings
- Fuente: src/indicators/earnings_filter.py
- Accion CIERRE: cierra posicion el dia ANTERIOR al earnings del ticker
  Motivo salida: "EARNINGS_MANANA"
- Accion BLOQUEO: no abre posicion si el ticker tiene earnings dentro del buffer
- Fail-safe: si no hay datos de earnings, NO cierra (evita cierres silenciosos)
- Este filtro tiene PRIORIDAD MAXIMA sobre cualquier otro criterio de salida o entrada

---

## ESTRATEGIA 1 — ML Scanner (Bot1 equivalent)

**Nombre instancia base**: ML_SCANNER_v1
**Logica**: ml_scanner
**Fuente de datos**: alertas_scanner, precios_diarios

### Parametros de Entrada

| Parametro        | Valor  | Descripcion                                      |
|------------------|--------|--------------------------------------------------|
| nivel_min        | COMPRA_FUERTE | Nivel minimo de alerta del scanner ML     |
| score_ml_min     | 65     | Score ML minimo (0-100)                          |
| max_posiciones   | 5      | Posiciones abiertas simultaneas                  |
| riesgo_por_trade | 15%    | Del capital actual                               |
| modo_dist        | fixed  | Capital fijo por trade                           |
| filtro_mtf       | False  | Filtro multi-timeframe (1W/1M) desactivado       |

Condiciones de entrada (TODAS deben cumplirse):
  1. nivel_alerta = COMPRA_FUERTE
  2. score_ml >= 65
  3. ticker NO tiene posicion abierta
  4. ticker NO bloqueado por earnings
  5. posiciones_abiertas < max_posiciones

Ranking de candidatos: score_ml descendente

### Parametros de Salida

| Tipo           | Parametro      | Valor  | Descripcion                           |
|----------------|----------------|--------|---------------------------------------|
| Exit primario  | nivel_degradado| < COMPRA_FUERTE | Scanner ya no valida la senal  |
| Exit emergencia| stop_loss_pct  | 5%     | SL fijo sobre precio de entrada       |
| Exit emergencia| take_profit_pct| 10%    | TP fijo sobre precio de entrada       |

Calculos:
  stop_loss   = precio_entrada * (1 - 0.05)
  take_profit = precio_entrada * (1 + 0.10)

### Exit Condicional
  - Earnings manana: SI (ver seccion transversal)
  - SL y TP son de emergencia; el exit primario es la degradacion del scanner

### Caracteristicas adicionales
  - Sin time stop definido (el exit primario cubre el tiempo en posicion)
  - Sin trailing SL
  - La logica de entrada y salida usan la MISMA fuente (alertas_scanner)
  - Variantes posibles: cambiar score_ml_min (ej. 70, 75), agregar filtro_mtf=True

---

## ESTRATEGIA 2 — Tecnico SMA/MACD/RSI (Bot2 equivalent)

**Nombre instancia base**: TECH_v1
**Logica**: tecnico
**Fuente de datos**: indicadores_tecnicos, precios_diarios

### Sistema de scoring (max 5.5 pts)

| Capa | Condicion                        | Puntos | Tipo       |
|------|----------------------------------|--------|------------|
| 1    | precio > SMA200                  | -      | OBLIGATORIO (0 si no cumple) |
| 2    | precio > SMA50                   | 2.0    | Tendencia  |
| 2    | precio > SMA21                   | 1.0    | Tendencia  |
| 3    | MACD > Signal AND hist > 0       | 1.5    | Momentum   |
| 3    | RSI entre RSI_MIN y RSI_MAX      | 1.0    | Momentum   |

### Parametros de Entrada

| Parametro        | Valor | Descripcion                               |
|------------------|-------|-------------------------------------------|
| score_entrada    | 4.0   | Score minimo para abrir (sobre 5.5)       |
| rsi_min          | 45.0  | RSI minimo (evita momentum debil)         |
| rsi_max          | 68.0  | RSI maximo (evita sobrecompra)            |
| max_posiciones   | 5     | Posiciones abiertas simultaneas           |
| riesgo_por_trade | 15%   | Del capital actual                        |

Condiciones de entrada (TODAS deben cumplirse):
  1. precio > SMA200 (filtro obligatorio, capa 1)
  2. score >= 4.0
  3. ticker NO tiene posicion abierta
  4. ticker NO fue cerrado en la misma corrida (mismo dia)
  5. ticker NO bloqueado por earnings
  6. posiciones_abiertas < max_posiciones

Ranking de candidatos: score descendente

### Parametros de Salida

| Tipo              | Parametro       | Valor | Descripcion                        |
|-------------------|-----------------|-------|------------------------------------|
| Exit primario     | score_salida    | 3.5   | Score <= 3.5 = indicadores se desalinean |
| Exit emergencia   | atr_mult_sl     | 2.0x  | SL = entrada - 2 * ATR14          |
| Exit emergencia   | atr_mult_tp     | 4.0x  | TP = entrada + 4 * ATR14          |

Garantia matematica de diseno:
  score_salida (3.5) < score_entrada (4.0)
  -> imposible que la misma data genere exit + entry simultaneamente

### Exit Condicional
  - Earnings manana: SI (ver seccion transversal)
  - Misma corrida: ticker cerrado no puede re-entrar el mismo dia (misma data, decision opuesta)

### Caracteristicas adicionales
  - Sin time stop definido
  - Sin trailing SL (SL ATR fijo al momento de entrada)
  - La Capa 1 (SMA200) actua como filtro de regimen de mercado
  - Variantes posibles: score_entrada 4.5, rsi_max 65, cambiar multiplicadores ATR

---

## ESTRATEGIA 3 — Estructura SMC CHoCH/BOS (Bot3 equivalent)

**Nombre instancia base**: SMC_v1
**Logica**: smc_estructura
**Fuente de datos**: features_market_structure, features_precio_accion,
                     indicadores_tecnicos, precios_diarios

### Parametros de Entrada

| Parametro          | Valor | Descripcion                                         |
|--------------------|-------|-----------------------------------------------------|
| lookback_dias      | 12    | Dias calendario para buscar CHoCH/BOS (~10 habiles) |
| score_entrada_min  | 1     | Score minimo de calidad (sobre 3)                   |
| min_sl_dist_pct    | 1.0%  | Distancia minima SL estructural desde close         |
| max_sl_dist_pct    | 8.0%  | Distancia maxima SL estructural desde close         |
| max_posiciones     | 5     | Posiciones abiertas simultaneas                     |
| riesgo_por_trade   | 15%   | Del capital actual                                  |

Condiciones OBLIGATORIAS de entrada (TODAS deben cumplirse):
  1. CHoCH_BULL o BOS_BULL detectado en ultimos 12 dias calendario
  2. estructura_10 >= 0 hoy (estructura no rota al baja)
  3. choch_bear_10 = 0 hoy (sin cambio de caracter bajista)
  4. es_alcista = 1 hoy (vela de cierre > apertura)
  5. dist_sl_10_pct entre 1% y 8% (SL estructural valido)
  6. ticker NO tiene posicion abierta
  7. ticker NO bloqueado por earnings
  8. posiciones_abiertas < max_posiciones

### Scoring de calidad para ranking (0-3 pts)

| Condicion                              | Puntos | Descripcion                   |
|----------------------------------------|--------|-------------------------------|
| tuvo_choch_bull = 1                    | +1     | CHoCH > BOS (senal mas fuerte)|
| vol_spike=1 OR eng_bull=1 OR hammer=1 | +1     | Confirmacion vela/volumen     |
| estructura_10 = +1                     | +1     | Tendencia HH/HL confirmada    |

Ranking de candidatos: score descendente

### Parametros de Salida

| Tipo              | Condicion                     | Motivo registrado    |
|-------------------|-------------------------------|----------------------|
| P1 (maxima prior) | precio_actual <= stop_loss    | TRAILING_SL          |
| P2                | choch_bear_10 = 1             | CHOCH_BEAR           |
| P3                | estructura_10 = -1            | ESTRUCTURA_ROTA      |
| P4                | dias_abierta >= 20            | TIME_STOP_Nd         |

Stop Loss Trailing (solo sube, nunca baja):
  swing_low = close / (1 + dist_sl_10_pct / 100)
  Se actualiza SOLO si nuevo_swing_low > sl_actual

Take Profit: NINGUNO (filosofia de salida estructural pura)

### Exit Condicional
  - Earnings manana: SI (ver seccion transversal, prioridad sobre P1-P4)
  - Time stop: 20 dias calendario (P4, prioridad mas baja dentro de la estrategia)

### Caracteristicas adicionales
  - Sin SL precio-based fijo (removido 22/4/2026 — generaba whipsaw EOD)
  - Filosofia: entra por estructura, sale por estructura
  - La dist_sl_10_pct se recalcula con precio real de cierre al entrar
  - Para FT: verificar margen adicional (+2%) en dist_real al validar con cierre EOD
  - Variantes posibles: lookback_dias 15, score_min 2, max_sl_dist 10%

---

## ESTRATEGIA 3b — SMC sobre estructura CONFIRMADA (alta 17/9/2026)

**Nombres de instancia**: SMC_v3_N5 (id 12) y SMC_v3_N3 (id 13)
**Logica**: `smc_estructura_confirmada`
**Script**: `scripts/forward_testing/ft_bot_smc_v3.py --ventana {5|3}`
**Fuente de datos**: `features_estructura`, `features_velas`, `features_precio_accion`
(solo `vol_spike`), `indicadores_tecnicos`, `precios_diarios`
**Control**: SMC_v1 (ESTRATEGIA 3), que sigue corriendo sin cambios
**Ficha**: docs/forward_testing/estrategias/SMC_v3.md

La MISMA regla de la ESTRATEGIA 3 con otra fuente de estructura. El score se IMPORTA
de `ft_scoring.calcular_score_estructura`: no se reimplementa, para que la diferencia
de resultados sea atribuible a la fuente y a la ventana N.

| Aspecto | ESTRATEGIA 3 (SMC_v1) | ESTRATEGIA 3b (SMC_v3) |
|---|---|---|
| Tabla de estructura | `features_market_structure` | `features_estructura` |
| Swings | ventana centrada; la ultima barra trae swings provisionales | CONFIRMADOS en p+N, inmutables (test de invariancia) |
| Ventana N | 10 | 5 y 3 (una instancia cada una) |
| Patrones de vela | `features_precio_accion` (definiciones flojas) | `features_velas` (clasicas con contexto) |
| `es_alcista` | columna de la tabla | derivada (`close > open`) |
| Ancla del lookback | `CURRENT_DATE` | ultima rueda de DATOS |
| Todo lo demas | -- | identico (score, filtros, trailing SL, salidas, sizing, time stop 20d) |

Parametros: los de la ESTRATEGIA 3, mas `ventana_confirmacion` (5 o 3).

Por que dos instancias: el backtest pre-registrado 2021-09 -> 2026-09
(docs/estructura_velas.md sec. 9.3) dio N=10 confirmado +13,5% (no pasa), N=5 +59,8%
y N=3 +58,2% (pasan). N=5 y N=3 empatan en el total y difieren por anio, y el
pre-registro dice que el N no se elige mirando el backtest: se decide en FT.

---

## Instancias DISCONTINUADAS

Una estrategia dada de baja conserva: la ficha con su cierre, su historia en
`ft_operaciones` / `ft_equity_diaria`, su entrada en `ft_setup_estrategias.ESTRATEGIAS`
con la marca `discontinuada`, el bloque comentado en `ft_run_diario.bat` y el registro
en `ft_cambios`. El codigo del bot NO se borra.

| Instancia | Periodo | Equity final | Motivo (resumen) | Cierre |
|---|---|---|---|---|
| COMBO_v1 (id 5) | 2026-04-28 -> 2026-09-16 | 99.901,60 (-0,10%) | El candle score, su unico aporte sobre TECH_SECTOR_v1, no agrega: backtest 5 anios +19,9% contra +24,4% sin velas; FT -0,09% contra +2,29% del control | [COMBO_v1.md](forward_testing/estrategias/COMBO_v1.md) |
| SMC_v2 (id 7) | 2026-05-04 -> 2026-09-16 | 94.538,65 (-5,46%) | Peor equity de las once; la salida por agotamiento no disparo nunca y sus filtros se apoyan en velas y estructura sin confirmar, medidas sin valor. Sin backtest previo | [SMC_v2.md](forward_testing/estrategias/SMC_v2.md) |

---

## Registro de instancias activas

Estado al 17/9/2026 (11 activas, $100.000 cada una). La fuente de verdad es
`ft_estrategias`; `ft_setup_estrategias.py --status` la imprime.

| id | nombre | logica | inicio | notas |
|----|--------|--------|--------|-------|
| 1  | FT_ML_SCANNER_v1 | ml_scanner | 2026-04-23 | Benchmark Bot1 Alpaca. Control de la v2 |
| 2  | FT_TECH_v1 | tecnico | 2026-04-23 | Benchmark Bot2 Alpaca |
| 3  | FT_SMC_v1 | smc_estructura | 2026-04-23 | Benchmark Bot3 Alpaca. Control de la v3 |
| 4  | FT_TECH_SECTOR_v1 | tecnico_sectorial | 2026-04-25 | Sectorial 9 sectores. Control de COMBO |
| 6  | FT_TECH_SECTOR_v2 | tecnico_sectorial_v2 | 2026-05-05 | Retencion + rotacion intrasectorial |
| 8  | FT_TECH_SECTOR_OPTIONS_v1 | tecnico_sectorial_options_v1 | 2026-05-17 | + PCR_OI |
| 9  | FT_TECH_SECTOR_OPTIONS_v2 | tecnico_sectorial_options_v2 | 2026-05-17 | + PCR_VOL |
| 10 | FT_TECH_SECTOR_OIEXIT_v1 | tecnico_sectorial_oiexit_v1 | 2026-05-23 | Salida por muros de OI |
| 11 | FT_ML_SCANNER_v2 | ml_scanner | 2026-09-14 | Modelo ML v2 en paralelo |
| 12 | FT_SMC_v3_N5 | smc_estructura_confirmada | 2026-09-16 | Estructura confirmada N=5 |
| 13 | FT_SMC_v3_N3 | smc_estructura_confirmada | 2026-09-16 | Estructura confirmada N=3 |

Los ids 5 (COMBO_v1) y 7 (SMC_v2) estan discontinuados, ver arriba. Los ids no se
reutilizan: la historia de una estrategia dada de baja sigue colgada de su id.

Notas:
  - Nuevas variaciones de parametros se agregan como instancias nuevas, en paralelo:
    una version nueva NUNCA corta a la vieja, que queda como control
  - id se asigna al insertar en ft_estrategias (DB)

---

---

## Hoja de ruta — Fases de desarrollo

### FASE 1 — Benchmarks (replicas de bots Alpaca, scripts nuevos)
Proposito: linea base para comparar todo lo que se desarrolle despues.
Scripts en: scripts/forward_testing/

  [x] Especificacion escrita (este documento)
  [ ] Diseno tablas ft_* aprobado
  [ ] ft_ml_scanner_v1.py — replica Bot1
  [ ] ft_tech_v1.py       — replica Bot2
  [ ] ft_smc_v1.py        — replica Bot3

### FASE 2 — Variaciones de parametros
Misma logica, distintos parametros. Objetivo: optimizar configuraciones.

  [ ] FT_TECH_v2  — score_entrada 4.5 (mas selectivo)
  [ ] FT_TECH_v3  — agregar filtro_mtf: tendencia_1w != bajista
  [ ] FT_SMC_v2   — score_min 2 (solo CHoCH + confirmacion, sin BOS puro)
  [ ] FT_ML_v2    — score_ml_min 70 + filtro_mtf activo

### FASE 3 — Estrategias nuevas con Opciones
Estrategias construidas desde cero. Incorporan en forma progresiva:

  Bloque A — Indicadores y Osciladores
    [ ] FT_IND_v1   — combinacion SMA + RSI + MACD con parametros propios

  Bloque B — Soportes y Resistencias
    [ ] FT_SR_v1    — niveles clave de precio como filtro de entrada/salida

  Bloque C — Opciones (disponible ~mayo 2026, cuando haya 30d de datos)
    Datos fuente: opciones_snapshot, opciones_resumen_diario
    Variables a incorporar:
      - PCR (Put/Call Ratio): sentimiento del mercado por ticker
      - Volumen Call vs Put: presion direccional
      - OI (Interes Abierto): niveles con mayor concentracion
      - IV (Volatilidad Implicita): filtro de riesgo / expansion de volatilidad

    [ ] FT_OPT_v1   — filtro de entrada con PCR + volumen opciones
    [ ] FT_COMBO_v1 — logica tecnica + confirmacion estructural + filtro opciones

  Nota: las estrategias de Fase 3 se disenan cuando Fase 1 este corriendo
        y tengamos datos suficientes de opciones para validar los filtros.
