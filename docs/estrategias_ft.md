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

## Registro de instancias activas

| id | nombre           | logica       | capital    | activa | fecha_inicio | notas                  |
|----|------------------|--------------|------------|--------|--------------|------------------------|
| -  | ML_SCANNER_v1    | ml_scanner   | $100.000   | -      | pendiente    | Benchmark Bot1 Alpaca  |
| -  | TECH_v1          | tecnico      | $100.000   | -      | pendiente    | Benchmark Bot2 Alpaca  |
| -  | SMC_v1           | smc_estructura| $100.000  | -      | pendiente    | Benchmark Bot3 Alpaca  |

Notas:
  - Los 3 anteriores son los benchmarks iniciales (replica logica Alpaca en FT)
  - Nuevas variaciones de parametros se agregan como filas adicionales
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
