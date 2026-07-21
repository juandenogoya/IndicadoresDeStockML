# Glosario — Forward Testing

**Ultima actualizacion**: 2026-05-05
**Criterio**: toda variable que aparezca en logica de entrada/salida, calculo de metricas
o analisis de resultados debe estar definida aqui antes de usarse en codigo o documentos.

---

## A. Features de Entrada (inputs de estrategias)

### tech_score
**Rango**: 0 a 5.5 puntos
**Fuente**: `src/scoring/rule_based.py → calcular_scoring()`
**Calculo**:
| Capa | Condicion | Puntos | Tipo |
|------|-----------|--------|------|
| 1 | precio > SMA200 | 0 si NO cumple | OBLIGATORIO |
| 2 | precio > SMA50 | +2.0 | Tendencia |
| 2 | precio > SMA21 | +1.0 | Tendencia |
| 3 | MACD hist > 0 Y MACD > Signal | +1.5 | Momentum |
| 3 | RSI entre RSI_MIN (45) y RSI_MAX (68) | +1.0 | Momentum |

Score maximo: 5.5 pts. Si precio < SMA200, score = 0 independientemente del resto.

---

### candle_score_5d
**Rango**: tipicamente -5 a +5 (sin cota fija, depende de 5 velas)
**Fuente**: `scripts/forward_testing/ft_scoring.py → obtener_candle_score_5d()`
**Calculo**: suma de scores individuales de las ultimas 5 velas.
Cada vela contribuye segun: direccion (alcista/bajista), tamano del cuerpo,
posicion del cierre en el rango del dia, y confirmacion de volumen.
**Interpretacion**: positivo = momentum de velas alcista; negativo = bajista o indeciso.

---

### lateral_ratio
**Formula**: `rango_5d_abs / ATR14`
**Donde**: `rango_5d_abs = MAX(close, 5d) - MIN(close, 5d)`
**Interpretacion**:
| Valor | Clasificacion | Significado |
|-------|---|---|
| < 0.5 | Lateral / Acumulacion | Precio moviendose menos que su volatilidad historica |
| 0.5 - 1.0 | Rango estrecho | Movimiento reducido, sin tendencia clara |
| > 1.0 | En movimiento | Precio con desplazamiento significativo vs volatilidad |

Uso principal: detectar si una posicion esta "trabada" o con momentum real.
Un lateral_ratio < 0.5 durante varios dias puede indicar acumulacion (no salir)
o agotamiento (salir). La diferencia la da el candle_score_5d.

---

### rango_5d_pct
**Formula**: `(MAX(close, 5d) - MIN(close, 5d)) / precio_entrada * 100`
**Unidad**: porcentaje
**Diferencia con lateral_ratio**: este es absoluto en %; lateral_ratio es relativo al ATR.
Util para comparar posiciones entre si independientemente del precio del activo.

---

### up_vol_5d
**Rango**: 0 a 5 (entero)
**Formula**: COUNT de dias en los ultimos 5 donde:
- `es_alcista = 1` (cierre > apertura)  Y
- `vol_price_confirm = 1` (volumen confirma el movimiento)
**Interpretacion**: 4-5 = momentum alcista con volumen consistente; 0-1 = debil o sin confirmacion.

---

### vol_price_confirm
**Tipo**: binario (0/1)
**Fuente**: tabla `features_precio_accion`
**Logica**: 1 si el volumen del dia confirma la direccion del precio.
- Vela alcista con volumen por encima del promedio → confirm
- Vela bajista con volumen por encima del promedio → confirm (movimiento bajista respaldado)

---

### vol_price_diverge
**Tipo**: binario (0/1)
**Fuente**: tabla `features_precio_accion`
**Logica**: 1 si hay divergencia precio-volumen (precio sube pero volumen cae, o viceversa).
Señal de debilidad del movimiento actual.

---

### ATR14 (Average True Range 14)
**Fuente**: tabla `indicadores_tecnicos`
**Calculo**: promedio del rango verdadero de los ultimos 14 dias.
Rango verdadero = MAX(high-low, |high-close_ant|, |low-close_ant|)
**Uso**: medida de volatilidad del activo. Base del lateral_ratio y de los SL/TP ATR-based.

---

### es_alcista
**Tipo**: binario (0/1)
**Fuente**: tabla `features_precio_accion`
**Definicion**: 1 si close >= open en esa sesion.

---

### BOS (Break of Structure)
**Tipo**: evento de estructura de mercado
**Fuente**: tabla `features_market_structure`
**BOS Bullish**: precio rompe por encima de un maximo previo significativo (Higher High).
Indica continuacion de tendencia alcista. Menos fuerte que CHoCH.
**BOS Bearish**: precio rompe por debajo de un minimo previo (Lower Low).

---

### CHoCH (Change of Character)
**Tipo**: evento de estructura de mercado — mas relevante que BOS
**Fuente**: tabla `features_market_structure`
**CHoCH Bullish**: precio en tendencia bajista rompe por encima de un maximo previo.
Indica CAMBIO de tendencia (de bajista a alcista). Senal mas fuerte que BOS.
**CHoCH Bearish**: inverso — fin de tendencia alcista.

---

### dist_sl_10_pct
**Fuente**: tabla `features_market_structure`
**Definicion**: distancia porcentual desde el precio de cierre hasta el swing low
estructural calculado con ventana de 10 barras.
Usada en SMC_v1 para calcular el stop loss estructural.

---

### score ML (ml_prob_ganancia)
**Rango**: 0.0 a 1.0 (probabilidad)
**Fuente**: tabla `alertas_scanner`, modelos V3 Random Forest
**Interpretacion**: probabilidad estimada de que el activo suba >3% en los proximos 20 dias.
Umbral en ML_SCANNER_v1: 0.65 (65%).

---

## B. Features de Salida / Rotacion

### retorno_pct
**Formula**: `(precio_cierre_hoy - precio_entrada) / precio_entrada * 100`
**Unidad**: porcentaje
**Nota**: puede ser negativo (perdida). No incluye costos de transaccion (FT es virtual).

---

### dias_abierta
**Calculo**: `len(trading_days_between(fecha_entrada, fecha_actual))`
**Fuente**: `src/utils/trading_calendar.py → trading_days_between()`
**IMPORTANTE**: retorna lista de fechas, se usa `len()` para obtener el entero.
Cuenta unicamente dias habiles NYSE.

---

### retorno_por_dia
**Formula**: `retorno_pct / dias_abierta`
**Uso**: comparar eficiencia entre posiciones con distinto tiempo abierto.
Una posicion con +5% en 2 dias es mas eficiente que +5% en 15 dias.

---

## C. Metricas del Sistema

> **ADVERTENCIA (2026-07-21)**: `capital_actual` / `capital_total` estan valuados
> a **COSTO DE ENTRADA**, no a mercado. Solo se mueven cuando se cierra una
> operacion. **No usarlos para medir riesgo, volatilidad ni drawdown**: la
> serie resultante es una curva de PnL realizado y subestima el riesgo.
> Para eso existe `ft_equity_diaria` (seccion G). Ver docs/forward_testing/METRICAS.md.

### capital_actual
Valor total de la estrategia = cash_disponible + capital_inmovilizado.
Se recalcula al final de cada corrida del bot.
**A costo, no a mercado** (ver advertencia arriba).

### capital_inmovilizado
Suma de `capital_entrada` de todas las operaciones abiertas.
Capital comprometido, no disponible para nuevas posiciones.
Es **cost basis**: no refleja el valor de mercado de las posiciones.

### cash_disponible
Capital liquido disponible para abrir nuevas posiciones.
`cash_disponible = capital_actual - capital_inmovilizado`

### retorno_total
`(capital_actual - capital_inicial) / capital_inicial * 100`
Calculado sobre capital_inicial ($100,000). NO sobre capital invertido.
**Por que total y no invertido**: el capital no desplegado esta inmovilizado en la estrategia
(no disponible para otras estrategias), por lo que el denominador correcto es el total.

### PnL realizado
Ganancia/perdida de operaciones ya cerradas. Acumulado historico.

### PnL no realizado
Ganancia/perdida estimada de posiciones abiertas a precio de mercado actual.
`SUM((precio_actual - precio_entrada) * cantidad)` para cada posicion abierta.

### saldo_estimado
`cash_disponible + valor_mercado_actual_posiciones_abiertas`

---

## D. Conceptos del Sistema FT

### slot
Cupo disponible para una posicion dentro de un sector o estrategia.
En estrategias sectoriales: cada sector tiene 5 slots. Si hay 3 posiciones abiertas,
hay 2 slots libres.

### capital_sector
Capital asignado a cada sector en estrategias sectoriales: $11,111.11
(9 sectores x $11,111.11 = ~$100,000).

### candidato
Activo que cumple los criterios de entrada de una estrategia pero que puede o no
haber abierto posicion (por limite de capital, slots, o ranking).

### oportunidad
Candidato que cumplio criterios de entrada pero NO abrio posicion
(motivo: capital insuficiente, slots llenos, fuera del ranking final).

### SCORE_DEGRADADO
Motivo de salida registrado cuando el score actual del activo cae por debajo
del umbral minimo de salida definido en la estrategia.
En TECH_SECTOR_v1: `SCORE_DEGRADADO_0.0` significa score = 0 (precio < SMA200).

### precio de ejecucion
Todas las estrategias usan el precio de CIERRE del dia de la senal.
Esto es una simplificacion consciente del FT (sin acceso a precios intraday).

### dry_run
Modo de prueba: el bot calcula senales y las muestra pero NO escribe en la DB.
Usar con `--dry-run` para validar logica sin efectos secundarios.

---

## E. Metricas de Observacion (capa analitica — ft_posiciones_diarias)

Estas metricas se calculan diariamente para cada posicion abierta y se almacenan
en `ft_posiciones_diarias`. Sirven para analizar el comportamiento interno de las
posiciones, no son inputs directos de las estrategias (aun).

| Campo | Descripcion |
|---|---|
| tech_score | Score tecnico del activo en la fecha del snapshot |
| candle_score_5d | Score de estructura de velas 5 dias en la fecha |
| lateral_ratio | rango_5d_abs / ATR14 en la fecha |
| rango_5d_pct | Variacion % precio en 5 dias sobre precio entrada |
| up_vol_5d | Dias alcistas con volumen confirmado (0-5) |
| vol_price_confirm | Confirmacion precio-volumen del dia |
| vol_price_diverge | Divergencia precio-volumen del dia |
| dias_abierta | Dias habiles desde la entrada hasta la fecha |
| retorno_pct | Retorno % vs precio de entrada |

---

## F. Sectores (clasificacion Yahoo Finance / yfinance)

Los 9 sectores utilizados en estrategias sectoriales:

| Sector | Descripcion |
|---|---|
| Technology | Semiconductores, software, hardware |
| Consumer Cyclical | Autos, retail, entretenimiento, viajes |
| Financial Services | Bancos, seguros, brokers |
| Industrials | Manufactura, logistica, aeroespacial |
| Basic Materials | Mineria, quimica, litio |
| Healthcare | Farmaceuticas, biotech, equipos medicos |
| Energy | Petroleo, gas, energia renovable |
| Communication Services | Medios, telecomunicaciones, plataformas |
| Consumer Defensive | Alimentos, bebidas, tabaco, higiene |

Nota: Real Estate y Utilities no estan representados en el universo actual (199 tickers).

---

## G. Metricas de Riesgo (capa derivada — ft_equity_diaria)

Diseno completo: [METRICAS.md](METRICAS.md). Definiciones canonicas:

### equity (marcado a mercado)
`cash + SUM(close(ticker, dia) * cantidad)` sobre las posiciones abiertas al
cierre de ese dia. Es el valor **real** de la estrategia ese dia.
Se distingue de `capital_actual` (seccion C), que esta a costo de entrada.

### exposicion_pct
`valor_mercado / equity`. Fraccion del capital efectivamente desplegada.
Separa la habilidad de la estrategia del *cash drag*: dos estrategias con el
mismo retorno pero distinta exposicion no son comparables directamente.

### max drawdown
`max_d( (peak_hasta_d - equity_d) / peak_hasta_d )`. La peor caida desde un
maximo previo. **Solo es calculable con equity a mercado**: sobre la serie a
costo el drawdown de posiciones abiertas es invisible.

### Sharpe
`(mean(r) - rf_d) / std(r) * sqrt(252)`, con `r` = retornos diarios de la equity
y `rf_d` la tasa libre de riesgo diaria (default 4.0% anual).
Mide exceso de retorno por unidad de volatilidad **total**.
**Siempre se reporta con `n` e IC95%** (ver mas abajo).

### Sortino
Igual que Sharpe pero el denominador es la *downside deviation*:
`sqrt(mean(min(r - rf_d, 0)^2))`. Solo castiga la volatilidad a la baja.
**Es el ratio primario del FT**: las estrategias son long-only con stop, o sea
asimetricas por diseno, y Sharpe penalizaria las subidas fuertes que las
estrategias de corrida (OIEXIT_v1 Fase 2) justamente buscan.

### Information Ratio (IR)
`mean(r - r_bench) / std(r - r_bench) * sqrt(252)`. Mide si la **seleccion** de
tickers aporta por encima del benchmark. Responde "¿esto le gana a comprar todo
el universo?", que es la pregunta relevante en una estrategia long-only.

### IC95% del Sharpe (Lo, 2002)
`SE = sqrt((1 + SR_diario^2 / 2) / n) * sqrt(252)`, `IC95% = SR +- 1.96 * SE`.
Con n<40 el intervalo mide varios puntos de Sharpe.
**Regla**: si el IC incluye cero, el ratio se marca **NO CONCLUYENTE** y no
alcanza para justificar un cambio de estrategia.

### expectancy en R
`R` = riesgo inicial de la operacion = `(precio_entrada - stop_loss_inicial) * cantidad`.
Expectancy en R = PnL medio expresado en multiplos de ese riesgo.
**Limitacion**: el trailing pisa `ft_operaciones.stop_loss`, por lo que el SL
inicial no siempre es recuperable en la historia previa (ver METRICAS.md #6).

### profit factor
`SUM(pnl > 0) / abs(SUM(pnl < 0))`. Cuantos dolares gana por cada dolar que
pierde. >1 es rentable; <1 pierde. Insensible al n, a diferencia del Sharpe.

### benchmark equiponderado
Indice sintetico del universo activo (`activos` WHERE activo=TRUE) con peso
igual por ticker, reconstruido desde `precios_diarios` sobre **la misma ventana
de cada estrategia**. Es el "comprar todo" contra el que se mide la seleccion.
