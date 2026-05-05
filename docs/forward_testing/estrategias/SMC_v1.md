# SMC_v1 — Documentacion de Estrategia

**Estado**: ACTIVA
**ID en DB**: 3
**Inicio**: 2026-04-28
**Script**: `scripts/forward_testing/ft_bot_smc.py`
**Logica base**: `smc_estructura`

---

## Concepto

Estrategia basada en Smart Money Concepts (SMC). Entra cuando detecta un cambio
de caracter (CHoCH) o ruptura de estructura (BOS) alcista, validando que la
estructura de mercado actual no contradiga la senal.

**Filosofia**: entra por estructura, sale por estructura.
No usa take profit fijo — la posicion se mantiene mientras la estructura sea alcista.
El stop loss es estructural (swing low calculado desde features_market_structure).

**Pregunta que responde**: ?las senales de cambio de estructura de mercado tienen
valor predictivo real? ?Una entrada en CHoCH/BOS con confirmacion de vela y volumen
genera alfa respecto al benchmark?

---

## Parametros de Entrada

| Parametro | Valor | Descripcion |
|---|---|---|
| lookback_dias | 12 | Dias calendario para buscar CHoCH/BOS (~10 habiles) |
| score_entrada_min | 1 | Score minimo de calidad (0-3) para entrar |
| min_sl_dist_pct | 1.0% | Distancia minima SL estructural desde close |
| max_sl_dist_pct | 8.0% | Distancia maxima SL estructural desde close |
| max_posiciones | 5 | Posiciones abiertas simultaneas (sin restriccion sectorial) |
| riesgo_por_trade | 15% | Del capital actual |

### Condiciones de entrada (TODAS obligatorias)

1. CHoCH_BULL o BOS_BULL detectado en los ultimos 12 dias calendario
2. `estructura_10 >= 0` hoy (estructura no rota al baja en ventana de 10 barras)
3. `choch_bear_10 = 0` hoy (sin cambio de caracter bajista activo)
4. `es_alcista = 1` hoy (vela de cierre > apertura — confirmacion del dia)
5. `dist_sl_10_pct` entre 1.0% y 8.0% (SL estructural valido y no demasiado ajustado)
6. ticker NO tiene posicion abierta
7. ticker NO bloqueado por earnings
8. `posiciones_abiertas < 5`

### Scoring de calidad para ranking (0-3 pts)

| Condicion | Puntos | Razon |
|---|---|---|
| `tuvo_choch_bull = 1` | +1 | CHoCH es senal mas fuerte que BOS (cambio de tendencia vs continuacion) |
| `vol_spike=1` OR `eng_bull=1` OR `hammer=1` | +1 | Confirmacion de vela o volumen |
| `estructura_10 = +1` | +1 | Tendencia HH/HL confirmada (Higher Highs, Higher Lows) |

Ranking: `score_calidad DESC`. Con score_entrada_min = 1, al menos una condicion debe cumplirse.

### Sizing
```
capital_por_trade = capital_actual * 0.15
qty = floor(capital_por_trade / precio_entrada)
```

### SL estructural
```
SL = precio_entrada * (1 - dist_sl_10_pct / 100)
```
La dist_sl_10_pct se recalcula con precio real de cierre al entrar.
El SL evoluciona como trailing (solo sube, nunca baja).

---

## Logica de Salida

Prioridades (P1 = maxima prioridad, ejecuta antes):

| Prioridad | Condicion | Motivo registrado |
|---|---|---|
| P0 | Earnings manana | EARNINGS_MANANA |
| P1 | precio_actual <= stop_loss | TRAILING_SL |
| P2 | choch_bear_10 = 1 | CHOCH_BEAR |
| P3 | estructura_10 = -1 | ESTRUCTURA_ROTA |
| P4 | dias_abierta >= 20 | TIME_STOP_20d |

### Trailing Stop Loss
```
nuevo_swing_low = close / (1 + dist_sl_10_pct / 100)
if nuevo_swing_low > sl_actual:
    sl_actual = nuevo_swing_low   # solo sube
```
El SL sigue al precio hacia arriba pero nunca retrocede.

### Take Profit
NINGUNO. La posicion se mantiene mientras la estructura alcista este intacta.
Esto es intencional — la filosofia SMC es "dejar correr las ganancias por estructura".

---

## Metricas al 2026-05-05

| Metrica | Valor |
|---|---|
| Dias activa | 7 |
| Capital actual | En seguimiento |
| Posiciones abiertas | En seguimiento |
| Operaciones cerradas | En seguimiento |

---

## Caracteristicas Especiales

- **Sin SL precio-based fijo**: removido el 22/04/2026 porque generaba whipsaw EOD.
  El precio de cierre puede tocar el SL intradiariamente pero recuperar al cierre.
  Ahora solo se evalua precio de cierre vs SL.
- **Sin TP fijo**: filosofia de salida estructural pura. El mercado determina la salida.
- **Lookback de 12 dias**: permite capturar senales con algo de retraso sin ser demasiado permisivo.

---

## Problemas Identificados en v1

1. **Entradas en contextos laterales**: un BOS puede ocurrir en un mercado que
   oscila sin tendencia. Si lateral_ratio < 0.5 al momento del BOS, la senal
   puede ser una trampa — el precio rompe pero no tiene momentum para continuar.

2. **Sin confirmacion de momentum de corto plazo**: la estrategia valida la estructura
   (BOS/CHoCH) y la vela del dia, pero no analiza si las ultimas 5 velas tienen
   momentum positivo (candle_score_5d). Un activo puede tener una vela alcista hoy
   pero 4 dias previos bajistas.

3. **Time stop de 20 dias puede ser demasiado largo**: una posicion estancada
   durante 15 dias con lateral_ratio < 0.5 y candle_score negativo deberia cerrarse
   antes del vencimiento del time stop.

4. **Sin filtro de contexto de mercado macro**: entra en cualquier condicion de mercado.
   En mercados con alta volatilidad (VIX alto), las estructuras BOS/CHoCH fallan mas.

---

## Hipotesis para v2

Ver [SMC_v2.md](SMC_v2.md) cuando este disponible.

Lineas de exploracion:
- **Filtro de entrada lateral**: no entrar si `lateral_ratio < 1.0` al momento del BOS/CHoCH.
  Un BOS en un mercado lateral es menos confiable.
- **Confirmacion candle_score_5d**: requerir `candle_score_5d > 0` para entrar
  (momentum de velas positivo en los ultimos 5 dias).
- **Salida anticipada por estancamiento**: si `lateral_ratio < 0.5` durante N dias
  consecutivos Y `candle_score_5d < 0`, cerrar sin esperar al time stop de 20 dias.
- **Reducir time stop**: de 20 a 12-15 dias para liberar capital mas rapido.
