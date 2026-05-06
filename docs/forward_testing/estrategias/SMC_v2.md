# SMC_v2 — Documentacion de Estrategia

**Estado**: EN DESARROLLO
**ID en DB**: 7
**Version anterior**: [SMC_v1.md](SMC_v1.md)
**Script**: `scripts/forward_testing/ft_bot_smc_v2.py`
**Logica base**: `smc_estructura_v2`

---

## Concepto

Extension de SMC_v1 con dos cambios:

1. **Filtro de entrada por contexto de mercado**: no entrar si tanto el
   momentum de velas como la estructura de rango son desfavorables.
   Filtra BOS/CHoCH en mercados laterales sin confirmacion de momentum.

2. **Salida por agotamiento de senal**: cierra la posicion cuando el mercado
   muestra convergencia de tres senales negativas simultaneas, sin necesidad
   de esperar al time stop de 20 dias (que fue eliminado).

La logica de entrada estructural (CHoCH/BOS, calidad SMC) y los exits
de estructura (CHOCH_BEAR, ESTRUCTURA_ROTA) se mantienen sin cambio.
El trailing SL se mantiene sin cambio.

**Pregunta que responde**:
?Filtrar entradas en mercados laterales y salir por agotamiento de senal
(en lugar de time stop fijo) mejora la calidad de las operaciones de SMC?

**Hipotesis principal**:
Los problemas identificados en v1 — entradas en contextos laterales y
posiciones estancadas durante semanas — se pueden resolver con senales de
mercado en lugar de parametros temporales arbitrarios. Un BOS en mercado
lateral sin momentum de velas es una trampa que v2 evita. Una posicion
estancada con tres senales negativas convergentes deberia cerrarse antes
de que el deterioro sea mayor.

---

## Parametros Globales (sin cambios respecto a v1)

| Parametro | Valor | Descripcion |
|---|---|---|
| capital_total | $100,000.00 | Capital asignado |
| max_posiciones | 5 | Sin restriccion sectorial |
| riesgo_por_trade | 15% | Del capital actual |

---

## Logica de Entrada — CAMBIOS RESPECTO A v1

### Condiciones obligatorias de v1 (sin cambio)

1. CHoCH_BULL o BOS_BULL detectado en los ultimos 12 dias calendario
2. `estructura_10 >= 0` hoy (estructura no rota al baja)
3. `choch_bear_10 = 0` hoy (sin cambio de caracter bajista activo)
4. `es_alcista = 1` hoy (vela de cierre > apertura)
5. `dist_sl_10_pct` entre 1.0% y 8.0% (SL estructural valido)
6. ticker sin posicion abierta
7. ticker sin earnings proximos
8. `posiciones_abiertas < 5`

### Filtro nuevo de contexto de mercado (v2)

**En v1**: no existia. Entraba en cualquier contexto post-BOS/CHoCH.

**En v2**: requerir AL MENOS UNA de las siguientes (logica OR):
```
lateral_ratio   > 1.0    → estructura de precio trending (no lateral)
candle_score_5d > 0      → momentum de velas positivo en los ultimos 5 dias
```

**Logica OR**: basta con que UNA condicion sea verdadera para permitir entrada.
No se usa AND para no sobre-restringir — SMC ya filtra por estructura (CHoCH/BOS),
por lo que agregar AND limitaria excesivamente el numero de entradas.

**Interpretacion**:
Un BOS en mercado trending (lateral_ratio > 1.0) es confiable aunque las velas
recientes sean neutras. Un BOS en mercado lateral pero con momentum de velas
positivo (candle_score_5d > 0) indica que el rompimiento tiene respaldo de precio.
Solo se descarta cuando ambas condiciones fallan: mercado lateral Y velas sin momentum.

**Candidatos rechazados**: registrar en ft_candidatos_diarios con
`entro = FALSE` y `motivo_skip = 'FILTRO_CONTEXTO_SMC'`.

### Scoring de calidad (sin cambio)

| Condicion | Puntos |
|---|---|
| `tuvo_choch_bull = 1` | +1 |
| `vol_spike=1` OR `eng_bull=1` OR `hammer=1` | +1 |
| `estructura_10 = +1` | +1 |

Ranking: `score_calidad DESC`. Minimo: score >= 1.

### Sizing y SL (sin cambio)

```
capital_por_trade = capital_actual * 0.15
qty = floor(capital_por_trade / precio_entrada)
SL  = precio_entrada * (1 - dist_sl_10_pct / 100)
```

---

## Logica de Salida — CAMBIOS RESPECTO A v1

### Exit por earnings (sin cambio, prioridad maxima)
Motivo: `EARNINGS_MANANA`

### Exit trailing SL (sin cambio)
```
nuevo_swing_low = close / (1 + dist_sl_10_pct / 100)
if nuevo_swing_low > sl_actual:
    sl_actual = nuevo_swing_low   # solo sube, nunca baja
```
Motivo: `TRAILING_SL`

### Exit estructural (sin cambio)

| Prioridad | Condicion | Motivo |
|---|---|---|
| P2 | `choch_bear_10 = 1` | CHOCH_BEAR |
| P3 | `estructura_10 = -1` | ESTRUCTURA_ROTA |

### Exit por agotamiento de senal (nuevo en v2)

**En v1**: time stop fijo de 20 dias (eliminado en v2 por ser arbitrario).

**En v2**: cerrar si se cumplen TODAS las siguientes condiciones simultaneamente:
```
up_vol_5d       = 0      → ningun dia alcista con volumen en ultimos 5
candle_score_5d < -2     → momentum de velas deteriorado (umbral = -2)
lateral_ratio   < 0.5    → mercado lateral sin rango definido
```
Motivo registrado: `AGOTAMIENTO_SEÑAL`

**Interpretacion**:
La posicion esta estancada en un mercado que no esta generando presion
compradora (up_vol_5d = 0), las velas de los ultimos 5 dias muestran deterioro
estructural significativo (candle_score < -2, no solo neutral) y el precio
opera en un rango inferior al ATR (lateral_ratio < 0.5). La convergencia de
los tres criterios indica que la tesis de entrada (BOS/CHoCH) no se confirmo
y el capital puede liberarse para mejores oportunidades.

**Por que -2 y no 0**:
Un candle_score_5d entre -2 y 0 puede representar consolidacion normal post-BOS.
El umbral -2 exige deterioro material (mas de 2 velas bajistas netas), evitando
cierres prematuros en fases de consolidacion previas a la ruptura definitiva.

### Tabla comparativa v1 vs v2

| Aspecto | v1 | v2 | Señal usada |
|---|---|---|---|
| Filtro entrada lateral | No existe | OR(lateral_ratio > 1.0, candle_score_5d > 0) | lateral_ratio, candle_score_5d |
| Take profit | Ninguno | Ninguno | Sin cambio |
| Trailing SL | Si (solo sube) | Si (solo sube) | Sin cambio |
| Exit CHOCH_BEAR | Si | Si | Sin cambio |
| Exit ESTRUCTURA_ROTA | Si | Si | Sin cambio |
| Time stop | 20 dias (fijo) | ELIMINADO | — |
| Exit agotamiento | No existe | up_vol=0 AND candle<-2 AND lateral<0.5 | up_vol_5d, candle_score_5d, lateral_ratio |

---

## Parametros de v2 — Resumen

| Parametro | Valor | Tipo |
|---|---|---|
| lookback_dias | 12 | Entrada estructural |
| score_calidad_min | 1 | Entrada calidad |
| min_sl_dist_pct | 1.0% | Entrada SL |
| max_sl_dist_pct | 8.0% | Entrada SL |
| filtro_lateral_ratio_min | 1.0 | Filtro entrada (OR) |
| filtro_candle_score_5d_min | 0 | Filtro entrada (OR) |
| filtro_logica | OR | Filtro entrada |
| agotamiento_up_vol_max | 0 | Salida agotamiento |
| agotamiento_candle_score_max | -2 | Salida agotamiento |
| agotamiento_lateral_ratio_max | 0.5 | Salida agotamiento |
| agotamiento_logica | AND | Salida agotamiento |
| time_stop | ELIMINADO | — |

---

## Metricas (pendiente — estrategia en desarrollo)

Completar una vez activa.

| Metrica | Valor |
|---|---|
| Fecha inicio | pendiente |
| Retorno total | — |
| Max drawdown | — |
| Operaciones totales | — |
| Win rate | — |
| Avg dias abierta | — |
| Entradas filtradas por contexto | — |
| Salidas por AGOTAMIENTO_SEÑAL | — |

**Metrica de validacion especifica para v2**:

1. **Filtro de entrada**: registrar cuantos candidatos rechaza `FILTRO_CONTEXTO_SMC`
   y si, en retrospectiva (retorno_5d/10d), esas entradas habrian sido ganadoras
   o perdedoras. Valida si el filtro discrimina correctamente.

2. **Salida por agotamiento**: registrar cuantas salidas `AGOTAMIENTO_SEÑAL`
   ocurren y cual era el retorno al momento del cierre vs lo que habria pasado
   sin ese exit. Valida si el criterio de agotamiento evita perdidas mayores.

---

## Hipotesis para v3

A definir tras observar resultados de v2. Posibles direcciones:
- Cambiar filtro entrada de OR a AND si hay demasiadas entradas en mercados laterales
- Ajustar umbral candle_score agotamiento de -2 a -3 si cierra demasiado pronto
- Agregar filtro de volumen en entrada (vol_spike o vol_price_confirm) como condicion adicional
- Explorar salida parcial: reducir posicion (no cerrar completo) ante primera senal de agotamiento
