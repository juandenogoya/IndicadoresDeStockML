# SMC_v2 — Documentacion de Estrategia

**Estado**: **DISCONTINUADA el 17/9/2026** (ver [Cierre](#cierre-17092026)).
Corrio del 2026-05-04 al 2026-09-16.
**ID en DB**: 7 (`activa = FALSE`)
**Version anterior**: [SMC_v1.md](SMC_v1.md) (sigue activa, es el control)
**Sucesoras**: [SMC_v3.md](SMC_v3.md) (FT_SMC_v3_N5 / FT_SMC_v3_N3)
**Script**: `scripts/forward_testing/ft_bot_smc_v2.py` (se conserva, fuera de la rutina)
**Logica base**: `smc_estructura_v2`
**Registro del cambio**: `ft_cambios.clave = smc_v2_discontinuada` (id 18)

> Motivo en una linea: la peor equity de las once (-5,46%) y sus tres agregados sobre
> SMC_v1 se apoyan en dos insumos que la revision de la Tarea 23 midio sin valor
> (patrones de vela agregados y estructura sin confirmar). Nunca tuvo backtest.

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

---

## Cierre (17/09/2026)

### Que se estaba probando

Tres cambios sobre SMC_v1, todos en la misma direccion: reemplazar parametros
temporales por "senales de mercado".

1. **Filtro de entrada por contexto** (OR): entrar solo si `lateral_ratio > 1,0`
   (mercado en tendencia) **o** `candle_score_5d > 0` (momentum de velas positivo).
2. **Salida por agotamiento** (AND): cerrar si `up_vol_5d = 0` **y**
   `candle_score_5d < -2` **y** `lateral_ratio < 0,5`.
3. **Time stop eliminado**: SMC_v1 cierra a los 20 dias; v2 no.

Entrada estructural (CHoCH/BOS + estructura + vela alcista), score de calidad,
trailing SL y exits estructurales: identicos a v1. SMC_v1 es el CONTROL exacto.

### Periodo y parametros evaluados

| | |
|---|---|
| Periodo de datos | 2026-05-04 -> 2026-09-16 (94 ruedas de equity) |
| Capital | $100.000, max 5 posiciones, 15% por trade, techo de despliegue 80% |
| Entrada | score >= 1 sobre 3; CHoCH/BOS bull en 12 dias; estructura >= 0; sin CHoCH bear; vela alcista; `dist_sl` entre 1% y 8% |
| Filtro nuevo | `lateral_ratio > 1,0` OR `candle_score_5d > 0` |
| Salida | trailing SL estructural, CHOCH_BEAR, ESTRUCTURA_ROTA, earnings manana |
| Salida nueva | `up_vol_5d = 0` AND `candle_score_5d < -2` AND `lateral_ratio < 0,5` |
| Time stop | ELIMINADO (v1 usa 20 dias) |
| Fuente de estructura y velas | `features_market_structure` + `features_precio_accion` (ventana 10) |

### Resultados en forward testing

| Metrica | SMC_v2 | SMC_v1 (control, mismas 94 ruedas) |
|---|---|---|
| Retorno (equity a mercado) | **-5,46%** | **+2,36%** |
| Max drawdown | 6,68% | 4,68% |
| Volatilidad anualizada | 10,66% | -- |
| Sortino | -2,16 | -- |
| Operaciones cerradas | 32 (27 sin contar la liquidacion) | 35 |
| Aciertos | 37,5% | -- |
| PnL realizado | -5.461,35 USD | -- |
| Expectancy por operacion | -170,67 USD (-1,16%) | -- |
| Profit factor | 0,54 | -- |
| Payoff ratio | 0,91 | -- |
| Retorno medio por operacion | -1,672% (n=27) | +0,083% (n=35) |

**Diferencia por operacion contra el control: -1,756%, IC95 [-4,257%; +0,746%] ->
NO distinguible de cero.** Con 27 operaciones no hay potencia estadistica: el numero
es malo pero el intervalo incluye el cero, y eso hay que decirlo. La baja no se apoya
en ese test (ver motivos).

Salidas (todas las operaciones cerradas):

| Motivo | Ops | PnL USD | Medio |
|---|---|---|---|
| TRAILING_SL | 15 | -9.994 | -4,55% |
| EARNINGS_MANANA | 12 | +3.371 | +1,93% |
| ESTRATEGIA_DISCONTINUADA | 5 | +1.161 | +1,60% |

**La tabla es el diagnostico**: sin time stop, ninguna posicion se cerro por tiempo y
la unica salida propia que actuo fue el trailing SL -- 15 operaciones a -4,55% de
media, -9.994 USD. La salida por agotamiento (el cambio estrella de la v2) **no
disparo ni una sola vez** en 94 ruedas: pedia tres condiciones simultaneas
(`up_vol_5d = 0` AND `candle < -2` AND `lateral < 0,5`) y esa conjuncion practicamente
no ocurre. El resultado neto de los tres cambios fue: quitar la salida que limitaba el
tiempo en posicion y no poner ninguna en su lugar.

### Motivos de la baja

1. **Sus dos insumos nuevos no tienen valor medido.** `candle_score_5d` (patrones de
   vela agregados) y las senales `bos_*_5`/`choch_*_5` que lo componen: ningun patron
   de vela tiene exceso distinguible de cero sobre 148.000 velas, y la historia de
   `features_market_structure` mira 10 ruedas al futuro
   (`docs/estructura_velas.md` sec. 4 y 5.4). Los tres agregados de la v2 se apoyan en
   esos dos insumos.
2. **El unico cambio que si actuo fue quitar el time stop, y actuo en contra.** El
   analisis del TIME_STOP de SMC_v1 (rama `feature/ml-walkforward-tarea20`) ya habia
   mostrado que esa salida no era el problema de la v1. Quitarla dejo las posiciones
   corriendo hasta el trailing SL.
3. **Nunca tuvo backtest.** Se desplego directo a FT. Hoy el motor de backtesting
   existe y esta verificado, asi que una hipotesis como esta se mide antes.
4. **Peor equity de las once** (-5,46%), con el control de la misma logica base en
   +2,36% sobre los mismos dias.
5. **La sucesion ya esta cubierta.** La pregunta de fondo de SMC ("sirve leer la
   estructura para entrar y salir?") se responde ahora con FT_SMC_v3_N5/N3, que usan
   estructura CONFIRMADA y tienen backtest pre-registrado.

### Que NO dice este cierre

- No dice que filtrar por contexto sea mala idea: dice que **estos** filtros, armados
  con **estos** insumos, no mejoraron nada y que su salida nueva nunca se activo.
- No dice que el trailing SL estructural este mal: en SMC_v1, con time stop, la misma
  salida convive con un resultado positivo.
- Con 27 operaciones, nada de esto es concluyente por si mismo; la baja se apoya en
  que los insumos no tienen valor medido (punto 1), no en el resultado.

### Cierre operativo

- Las **5 posiciones abiertas** (BP, CVX, V, BSBR, MCK) se liquidaron al cierre del
  2026-09-16 con `motivo_salida = ESTRATEGIA_DISCONTINUADA` (+1.160,77 USD). Salida
  ARTIFICIAL, etiquetada: la serie limpia termina con las 27 operaciones anteriores
  (PnL -6.622,12 USD).
- `ft_estrategias.activa = FALSE` (id 7).
- Bot retirado de `scripts/manual/ft_run_diario.bat` (bloque del BOT 7 comentado con
  el motivo).
- Entrada de `ft_setup_estrategias.ESTRATEGIAS` marcada `discontinuada`, con sus
  parametros intactos.
- Equity final: **94.538,65 USD (-5,46%)** al 2026-09-16, con 0 posiciones.
- Script de baja: `scripts/oneshot/discontinuar_estrategias_ft.py`.

### Que sobrevive de la v2

La idea de **salir por deterioro y no por reloj** sigue siendo razonable; lo que fallo
fue la implementacion (tres condiciones AND que no ocurren nunca, sobre insumos sin
valor). Si se retoma, medir primero en backtest cuantas veces dispara la regla de
salida antes de creerle algo: una salida que no se activa en 94 ruedas no es una
salida, es un adorno.
