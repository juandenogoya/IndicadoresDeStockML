# Parametros de interpretacion -- MCP server
# Ultima actualizacion: 2026-05-16

## Proposito

Registro unico de los umbrales y parametros que las tools del MCP usan para
convertir valores crudos (PCR, RSI, OI, etc.) en lecturas categoricas
(alcista/bajista, ATM/OTM, etc.).

Por que existe: estos parametros son decisiones. Algunos son estandar de
industria, otros son defaults tentativos elegidos sin validar. Tenerlos
centralizados permite, en el futuro, replantearlos con otros valores y medir
si los resultados mejoran.

Regla de mantenimiento: si se cambia un parametro en el codigo, se actualiza
este doc. Si se agrega un calculo nuevo con umbrales, se registra aca.

## Estado de validacion (leyenda)

- ESTANDAR  : umbral estandar de la industria (RSI 70/30, ADX 25, etc.)
- PROYECTO  : decision del proyecto, coherente pero no validada con datos
- TENTATIVO : default elegido sin sustento empirico -- candidato a re-tunear

---

## Opciones -- mcp_server/tools/options.py

### Sesgo PCR -- funcion `_sesgo_pcr()`

Convierte un Put/Call Ratio en sesgo. Aplica a `sesgo_pcr_vol`,
`sesgo_pcr_oi` y al `sesgo_actual` de `pcr_por_vencimiento`.

| Condicion           | Sesgo   |
|---------------------|---------|
| PCR < 0.7           | alcista |
| 0.7 <= PCR <= 1.0   | neutro  |
| PCR > 1.0           | bajista |

Nota critica: PCR < 1 = MAS CALLS que puts (no al reves).
Estado: PROYECTO -- el 1.0 es el punto natural; el 0.7 (corte del "neutro")
es una decision.

### IV skew -- constante `_IV_SKEW_NEUTRAL = 0.05`

Convierte el IV skew (`iv_put_avg - iv_call_avg`) en sesgo. Funcion
`_sesgo_iv_skew()`.

| Condicion          | Sesgo   |
|--------------------|---------|
| skew > +0.05       | bajista (puts mas caras = cobertura/miedo) |
| -0.05 <= skew <= +0.05 | neutro |
| skew < -0.05       | alcista (calls mas caras = optimismo) |

Estado: TENTATIVO -- el 0.05 fue elegido sin validar; candidato a re-tunear.

### Moneyness -- funcion `_moneyness_label()`

Clasifica un strike segun `moneyness_pct = (strike/precio_subyacente - 1)*100`,
relativo al precio del subyacente DE HOY (no es prediccion del precio futuro).

| Condicion            | Etiqueta |
|----------------------|----------|
| abs(moneyness) <= 2% | ATM      |
| call: moneyness > 2% | OTM      |
| call: moneyness < -2%| ITM      |
| put: moneyness > 2%  | ITM      |
| put: moneyness < -2% | OTM      |

Estado: PROYECTO -- banda ATM de +-2%.

### Ventanas de vencimiento (Etapa 4 -- a implementar)

Espejo de `scripts/forward_testing/ft_bot_tech_sectorial_options_v1.py`.
El MCP no puede importar de scripts/, asi que estas constantes se replican
en el codigo del MCP. Si cambian en el FT, sincronizar aca.

| Ventana | Dias al vencimiento | Caracter |
|---------|---------------------|----------|
| Corto   | 1 - 14              | weeklies, posicionamiento tactico |
| Medio   | 15 - 45             | mensuales, mayor OI, mas confiable |
| Largo   | 46 - 90             | institucional, medio plazo |

Estado: PROYECTO -- definicion heredada del Forward Testing.

### Liquidez minima por ventana (Etapa 4 -- a implementar)

`MIN_OI_POR_VENTANA = 500`: una ventana necesita al menos 500 de OI total
(call + put) para que su PCR se considere valido. Por debajo, la ventana se
marca `valido: false` ("sin liquidez") y no se le da veredicto de sesgo.

Estado: PROYECTO -- valor heredado del Forward Testing.

### Ancho de zona de soporte/resistencia (Etapa 4 -- a implementar)

`ZONA_OI_PCT = 0.40`: la zona de soporte/resistencia es la corrida CONTIGUA
de strikes alrededor del strike-pico (el de mayor OI) cuyo OI sigue siendo
>= 40% del OI del pico. Mayor % = zona mas angosta; menor % = zona mas ancha.

Estado: TENTATIVO -- el 40% fue elegido sin validar.

---

## Indicadores tecnicos -- mcp_server/tools/overview.py

### Tendencia SMA -- funcion `_tendencia_sma()`

| Condicion                              | Tendencia      |
|----------------------------------------|----------------|
| close > sma21 > sma50 > sma200          | alcista fuerte |
| close > sma50 y close > sma200          | alcista        |
| close < sma21 < sma50 < sma200          | bajista fuerte |
| close < sma50 y close < sma200          | bajista        |
| resto                                   | lateral        |

Estado: PROYECTO.

### Estado RSI14 -- funcion `_rsi_estado()`

| Condicion       | Estado        |
|-----------------|---------------|
| RSI >= 70       | sobrecomprado |
| RSI <= 30       | sobrevendido  |
| 55 <= RSI < 70  | alcista       |
| 30 < RSI <= 45  | bajista       |
| 46 <= RSI <= 54 | neutral       |

Estado: ESTANDAR (70/30) + PROYECTO (cortes 55/45 del alcista/bajista).

### Direccion MACD -- funcion `_macd_direccion()`

macd_hist > 0 -> alcista ; macd_hist <= 0 -> bajista. Estado: ESTANDAR.

### Posicion en Bandas de Bollinger -- funcion `_bb_posicion()`

Posicion normalizada: `pos = (close - bb_lower) / (bb_upper - bb_lower)`.

| Condicion        | Posicion   |
|------------------|------------|
| pos >= 0.8       | alta       |
| 0.6 <= pos < 0.8 | media-alta |
| 0.4 <= pos < 0.6 | media      |
| 0.2 <= pos < 0.4 | media-baja |
| pos < 0.2        | baja       |

Estado: PROYECTO.

### Fuerza ADX -- funcion `_adx_fuerza()`

| Condicion       | Fuerza      |
|-----------------|-------------|
| ADX >= 40       | muy fuerte  |
| 25 <= ADX < 40  | fuerte      |
| 20 <= ADX < 25  | moderada    |
| ADX < 20        | debil       |

ADX mide FUERZA de tendencia, no direccion. Estado: ESTANDAR (25/20).

### Tendencia de velas -- funcion `_tendencia_velas_label()`

Interpreta la columna `tendencia_velas` (int).

| Condicion  | Etiqueta              |
|------------|-----------------------|
| val >= 3   | alcista fuerte        |
| 0 < val < 3| alcista               |
| val == 0   | neutral               |
| -3 < val < 0| bajista              |
| val <= -3  | bajista fuerte        |

Estado: TENTATIVO -- cortes en +-3 elegidos sin validar.

---

## Estructura de mercado -- varias tools

### Etiqueta de estructura -- `estructura_5` / `estructura_10`

Criterio unico, replicado en overview.py, alerts.py y screener.py
(las tools del MCP no comparten helpers entre si por diseño).

| Valor | Etiqueta |
|-------|----------|
| 1     | alcista  |
| 0     | neutral  |
| -1    | bajista  |

Estado: PROYECTO -- convencion de la tabla features_market_structure.

---

## Limites operativos (no son de interpretacion)

| Parametro | Valor | Tool | Que limita |
|-----------|-------|------|------------|
| MAX_LIMIT | 100   | screener.py, alerts.py | Max filas devueltas |
| MAX_DIAS_HISTORIA | 60 | options.py | Max snapshots de historia |
| DEFAULT_DIAS_HISTORIA | 20 | options.py | Snapshots por defecto |

---

## Pendiente de validacion

Los parametros marcados TENTATIVO son los primeros candidatos a re-tunear
con datos reales:

- `_IV_SKEW_NEUTRAL` (0.05) -- banda neutra del IV skew.
- `ZONA_OI_PCT` (0.40) -- ancho de la zona de soporte/resistencia.
- cortes +-3 de `tendencia_velas`.

Metodologia futura: correr el calculo con distintos valores sobre el
historico y medir cual separa mejor las señales que efectivamente
anticiparon movimiento.
