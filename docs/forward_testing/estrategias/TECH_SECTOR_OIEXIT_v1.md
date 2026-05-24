# TECH_SECTOR_OIEXIT_v1 — Documentacion de Estrategia

**Estado**: EN DESARROLLO
**ID en DB**: 10 (pendiente registro en ft_estrategias)
**Estrategia base (entrada)**: [TECH_SECTOR_v1.md](TECH_SECTOR_v1.md)
**Script**: `scripts/forward_testing/ft_bot_tech_sectorial_oiexit_v1.py` (pendiente)
**Logica base**: `tecnico_sectorial_oiexit_v1`

---

## Concepto

Toma la entrada **identica** a TECH_SECTOR_v1 (score tecnico + particion
sectorial, sin opciones) y le cambia **solo la salida**. La salida nueva
reemplaza el SL/TP fijo de 2x/4x ATR por un esquema en dos partes:

1. **SL inicial anclado al posicionamiento de opciones** (put wall de OI), con
   un esquema de proteccion por escalones de R.
2. Al alcanzar el objetivo 3:1, **se libera el techo** y la posicion pasa a
   una fase de "corrida" que la deja acompanar el impulso, saliendo por senales
   de agotamiento (no por un numero fijo definido en la entrada).

La tesis: un TP fijo corta a los ganadores justo cuando mas corren. Si la
perdida ya esta controlada (SL que sube), el valor a capturar esta en **dejar
correr** las tendencias fuertes y salir cuando el mercado da senal real de
agotamiento, no antes.

**Pregunta que responde**:
?Una salida basada en soportes de opciones (OI walls) + corrida con trailing
y senales de agotamiento supera al SL/TP fijo 2x/4x ATR de TECH_SECTOR_v1,
manteniendo exactamente la misma entrada?

**Por que la entrada es identica a v1**:
Para aislar UNA sola variable: la salida. Si la entrada tambien cambiara, no
sabriamos atribuir las diferencias de rendimiento. El dia de la apertura las
posiciones son las mismas que TECH_SECTOR_v1; las curvas de equity divergen
esencialmente por la logica de salida.

> Nota de honestidad: el dia 1 las entradas coinciden con v1, pero apenas las
> salidas difieren, el estado del portfolio diverge (distintos slots/capital
> libre) y las entradas de dias siguientes empiezan a diferir. No es un A/B
> perfecto eterno; es la comparacion mas limpia posible.

---

## Parametros Globales (identicos a TECH_SECTOR_v1)

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado |
| n_sectores | 9 | Sectores activos |
| capital_por_sector | $11,111.11 | capital_total / n_sectores |
| capital_por_posicion | $2,222.22 | capital_por_sector / max_pos_sector |
| max_posiciones_sector | 5 | Maximo de posiciones abiertas por sector |

---

## Logica de Entrada (identica a TECH_SECTOR_v1)

**Sin componente de opciones. Puro tecnico + sectorial.**

Condiciones obligatorias:
1. `tech_score >= 4.0` (scoring SMA/MACD/RSI, igual que v1)
2. `precio > SMA200` (implicito: si no, tech_score = 0)
3. ticker sin posicion abierta en esta estrategia
4. ticker no cerrado en la misma corrida del dia
5. ticker sin earnings proximos
6. sector con slots y capital disponible

**Ranking dentro del sector**: `tech_score DESC` (idem v1).

**Sizing**: `qty = floor($2,222.22 / precio_entrada)`.

La unica diferencia con v1 en la apertura es el calculo del SL inicial (ver
Salida) y que NO se fija un TP fijo de 4x ATR.

---

## Logica de Salida (el corazon de la estrategia)

### SL inicial — put wall de OI, con fallback a ATR

El SL inicial se ancla al **put wall**: el strike con mayor put OI por debajo
del precio, dentro de una zona cercana, si tiene liquidez suficiente.

```
Ventana de vencimiento para el wall: medio (15 - 45 dias)
Zona de busqueda: strikes entre (precio * 0.90) y precio   (hasta -10%)
Soporte (put wall) = strike con MAX(put_OI) en esa zona

Validez por liquidez RELATIVA al ticker:
    put_OI(wall) >= 3 x mediana(put_OI de los strikes de la zona)
    Y put_OI(wall) >= 1.000        (piso de sanidad, descarta tickers finos)

Cota de distancia:
    si el wall queda a menos de 2% del precio -> demasiado pegado
    si no hay wall valido (liquidez o distancia) -> FALLBACK: SL = 2x ATR14
```

**Fundamento**: un put wall liquido es un soporte real (donde los escritores
de puts defienden). Pero la liquidez de opciones NO sigue al tamano de la
empresa: tickers caros como CAT o LLY pueden tener OI por strike ridiculo
(walls de ruido), mientras que defensivas como KO tienen walls nitidos. Por eso
la validez es **relativa al propio ticker** (cluster real vs campo ralo), con
un piso minimo absoluto.

### Definicion de R

```
R = precio_entrada - SL_inicial      (riesgo por accion, fijo, definido en la entrada)
```

Todo el esquema se mide en unidades de R, ancladas a la entrada (no se
recalculan cuando el SL sube).

### Fase 1 — Proteccion (entrada -> +3R)

El SL trepa por escalones a medida que avanza el precio. El TP "virtual" esta
en +3R, pero al tocarse NO cierra: dispara la transicion.

```
precio >= entrada + 1R   ->  SL = precio_entrada       (breakeven, trade "gratis")
precio >= entrada + 2R   ->  SL = entrada + 1R          (1R asegurado)
precio >= entrada + 3R   ->  TRANSICION (no cierra)
Si el SL se toca en esta fase -> cerrar con la ganancia del escalon alcanzado.
```

### Transicion (+3R)

```
Al tocar +3R: se libera el techo. SL piso pasa a entrada + 2R.
(de aca en mas se asegura al menos 2R)
```

### Fase 2 — Corrida (post +3R, sin techo)

Dos mecanismos de salida conviven; cierra el primero que se cumpla.

**Backstop (siempre activo, por si solo):**
```
Chandelier = MAX(cierre desde la entrada) - 2.5 x ATR14
precio_cierre <= Chandelier  ->  cerrar
```
Garantiza que no se devuelva mas de ~2.5x ATR desde el pico.

**Quorum 3 de 4 (salida anticipada por agotamiento):**
```
1. cierre < SMA21
2. candle_score_5d <= -2                 (distribucion / velas bajistas)
3. PCR_VOL mayoria bajista (pcr_score <= 1 de 3 ventanas alcistas)
4. divergencia de volumen:
       dist_max_20d >= -1.0   (precio dentro del 1% del maximo de 20 dias)
       Y vol_ratio_5d < 0.8   (volumen del dia por debajo del promedio de 5d)

Si se cumplen 3 de estas 4 -> cerrar.
```

Si el ticker no tiene liquidez de opciones para calcular el PCR_VOL ese dia,
esa senal NO cuenta (ni a favor ni en contra): el quorum se evalua sobre las
otras 3 (se necesitarian las 3). El backstop Chandelier siempre protege.

### Salida transversal (cualquier fase)

```
P0. Earnings manana -> EARNINGS_MANANA   (prioridad absoluta, igual que todas)
```

### Resumen de prioridades de salida

| Prioridad | Condicion | Motivo |
|---|---|---|
| P0 | Earnings manana | `EARNINGS_MANANA` |
| P1 (Fase 1) | precio <= SL escalonado | `SL_PROTECCION` |
| P2 (Fase 2) | precio <= Chandelier 2.5xATR | `BACKSTOP_CHANDELIER` |
| P3 (Fase 2) | 3 de 4 senales de agotamiento | `AGOTAMIENTO_QUORUM` |

---

## Implementacion — notas de diseno

- **Evaluacion diaria al cierre** (igual que el resto del motor FT).
- **Estado derivado, no almacenado**: cada corrida se recalcula el pico de R
  alcanzado desde `fecha_entrada` via `MAX(close)` de `precios_diarios`. De ahi
  se deriva el escalon de SL y si la posicion esta en Fase 2 (pico >= +3R).
- **`stop_loss` muta dia a dia**: a diferencia de las otras estrategias que lo
  fijan una vez, esta lo reescribe en cada corrida (solo hacia arriba).
- Es la salida mas compleja del sistema (estado derivado, SL movil, multi-senal).
  Es el punto del experimento, asi que la complejidad esta justificada.

---

## Fuentes de datos

| Dato | Tabla / campo |
|---|---|
| tech_score (entrada) | `indicadores_tecnicos` (sma21/50/200, rsi14, macd) |
| put wall (SL inicial) | `opciones_snapshot` (strike, open_interest, tipo, ventana medio) |
| PCR_VOL (salida) | `opciones_snapshot` (volumen por ventana corto/medio/largo) |
| candle_score_5d | `ft_scoring.obtener_candle_score_5d()` |
| SMA21 / ATR14 | `indicadores_tecnicos` |
| dist_max_20d / vol_ratio_5d | `features_precio_accion` |
| pico de R (Chandelier) | `precios_diarios` (MAX(close) desde entrada) |

---

## Tabla comparativa TECH_SECTOR_v1 vs OIEXIT_v1

| Aspecto | TECH_SECTOR_v1 | TECH_SECTOR_OIEXIT_v1 |
|---|---|---|
| Entrada | tech_score >= 4.0, sectorial | **Identica** |
| Opciones en entrada | No | No |
| SL inicial | 2x ATR fijo | **Put wall de OI** (fallback 2x ATR) |
| TP | 4x ATR fijo | **3:1 como interruptor, luego corrida** |
| Proteccion | Ninguna (SL fijo) | **Escalones de R** (BE, +1R) |
| Salida en tendencia | Cierra en TP | **Deja correr** con trailing + senales |
| Salida por agotamiento | score = 0 | **Backstop Chandelier + quorum 3/4** |
| Opciones en salida | No | **Put wall (SL) + PCR_VOL (quorum)** |
| Earnings | Transversal | Transversal |

---

## Parametros de la estrategia — Resumen completo

| Parametro | Valor | Tipo |
|---|---|---|
| tech_score_entrada_min | 4.0 | Entrada |
| wall_ventana_dias | 15 - 45 (medio) | Salida — SL inicial |
| wall_zona_pct | 10 | Salida — zona de busqueda (-10%) |
| wall_liquidez_mult_mediana | 3 | Salida — liquidez relativa |
| wall_liquidez_min_abs | 1.000 | Salida — piso de OI |
| wall_dist_min_pct | 2.0 | Salida — distancia minima del wall |
| sl_fallback_atr_mult | 2.0 | Salida — SL si no hay wall |
| r_breakeven | 1.0 | Salida — Fase 1 (SL a BE) |
| r_lock | 2.0 | Salida — Fase 1 (SL a +1R) |
| r_libera_techo | 3.0 | Salida — transicion |
| r_piso_fase2 | 2.0 | Salida — piso Fase 2 |
| chandelier_atr_mult | 2.5 | Salida — Fase 2 backstop |
| quorum_min | 3 de 4 | Salida — Fase 2 |
| sma_salida | 21 | Salida — quorum senal 1 |
| candle_score_salida_max | -2 | Salida — quorum senal 2 |
| pcr_vol_salida_max | 1 | Salida — quorum senal 3 (mayoria bajista) |
| div_dist_max_20d_min | -1.0 | Salida — quorum senal 4 |
| div_vol_ratio_max | 0.8 | Salida — quorum senal 4 |

---

## Metricas de validacion especificas

Ademas de las estandar (retorno, drawdown, win rate), el objetivo es comparar
contra TECH_SECTOR_v1 (misma entrada):

**1. Efecto del exit vs v1**:
- Retorno total y max drawdown OIEXIT_v1 vs TECH_SECTOR_v1
- ?La curva de equity supera a v1, y a costa de que (mas drawdown? menos trades?)

**2. Uso del put wall vs fallback ATR**:
- % de aperturas donde el SL salio del put wall vs cayo a 2x ATR
- ?Los walls quedaron a distancias razonables o seguido caian al fallback?

**3. Comportamiento de la corrida (Fase 2)**:
- % de trades que alcanzaron +3R (entraron en Fase 2)
- De esos, R promedio capturado al cierre vs el +3R que v1 habria cobrado
- Giveback: R del pico vs R de la salida (cuanto se devuelve por dejar correr)

**4. Distribucion de motivos de salida**:
- SL_PROTECCION / BACKSTOP_CHANDELIER / AGOTAMIENTO_QUORUM / EARNINGS_MANANA
- ?Que mecanismo cierra mas seguido y cual aporta o resta?

---

## Metricas (pendiente — estrategia en desarrollo)

| Metrica | Valor |
|---|---|
| Fecha inicio | pendiente |
| Retorno total | — |
| Max drawdown | — |
| Operaciones totales | — |
| Win rate | — |
| % trades que llegaron a +3R | — |
| Giveback promedio (pico R - salida R) | — |
| % SL por put wall vs ATR fallback | — |

---

## Hipotesis para v2 (TECH_SECTOR_OIEXIT_v2)

A definir tras observar resultados. Posibles direcciones:

- **Trailing del SL por put wall en Fase 2**: en vez de Chandelier ATR, subir
  el SL al nuevo put wall que se forma debajo del precio en ascenso (atado al
  OI de punta a punta).
- **PCR hibrido en el quorum**: combinar PCR_VOL (reactivo) + PCR_OI (estructura).
- **Ajustar el gatillo de liberacion**: probar 2:1 o 4:1 en vez de 3:1.
- **Quorum 2 de 4** (mas agresivo para dejar correr) vs 3 de 4 actual.
- **TP por call wall** como techo opcional en lugar de corrida infinita, para
  tickers donde la resistencia de OI es nitida.
