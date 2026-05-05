# TECH_v1 — Documentacion de Estrategia

**Estado**: ACTIVA
**ID en DB**: 2
**Inicio**: 2026-04-28
**Script**: `scripts/forward_testing/ft_bot_tecnico.py`
**Logica base**: `tecnico`

---

## Concepto

Replica del Bot2 de Alpaca. Estrategia rule-based pura basada en indicadores
tecnicos clasicos (SMA, MACD, RSI). Sin ML, sin estructura de mercado.
Sin restriccion sectorial — los 5 slots compiten globalmente entre los 199 tickers.

**Pregunta que responde**: ?los indicadores tecnicos clasicos solos, sin ML ni SMC,
generan alfa? ?Es suficiente con SMA + MACD + RSI para operar sistematicamente?

---

## Sistema de Scoring (max 5.5 pts)

| Capa | Condicion | Puntos | Tipo |
|---|---|---|---|
| 1 | precio > SMA200 | 0 si NO | OBLIGATORIO |
| 2 | precio > SMA50 | +2.0 | Tendencia |
| 2 | precio > SMA21 | +1.0 | Tendencia |
| 3 | MACD hist > 0 Y MACD > Signal | +1.5 | Momentum |
| 3 | RSI entre 45 y 68 | +1.0 | Momentum |

---

## Parametros de Entrada

| Parametro | Valor | Descripcion |
|---|---|---|
| score_entrada_min | 4.0 | Score minimo de entrada |
| rsi_min | 45.0 | RSI minimo |
| rsi_max | 68.0 | RSI maximo (evita sobrecompra) |
| max_posiciones | 5 | Global, sin restriccion sectorial |
| riesgo_por_trade | 15% | Del capital actual |

### Condiciones de entrada (TODAS obligatorias)
1. `precio > SMA200` (Capa 1 obligatoria)
2. `tech_score >= 4.0`
3. ticker sin posicion abierta
4. ticker no cerrado en la misma corrida
5. ticker sin earnings proximos
6. `posiciones_abiertas < 5`

Ranking: `tech_score DESC`

### SL / TP (ATR-based)
```
SL = precio_entrada - (2.0 * ATR14)
TP = precio_entrada + (4.0 * ATR14)
```
Ratio implicito: 1:2 riesgo/beneficio.

---

## Logica de Salida

| Prioridad | Condicion | Motivo |
|---|---|---|
| P0 | Earnings manana | EARNINGS_MANANA |
| P1 | precio <= SL | SL |
| P2 | precio >= TP | TP |
| P3 | tech_score <= 3.5 | SCORE_DEGRADADO |

**Garantia de diseno**: score_salida (3.5) < score_entrada (4.0).
Imposible que la misma data genere exit + entry en el mismo dia.

---

## Metricas al 2026-05-05

| Metrica | Valor |
|---|---|
| Dias activa | 7 |
| Capital actual | ~$97,353 |
| Retorno total | -2.65% |
| Posiciones abiertas | 5 |
| Cash disponible | ~$23,798 |

**Nota**: el retorno negativo en los primeros 7 dias no es necesariamente
indicativo del rendimiento de largo plazo. El benchmark Alpaca Bot2 tiene
historico real que servira como referencia cuando tengamos mas datos de FT.

---

## Notas

- El umbral de salida (3.5) es mas bajo que el de entrada (4.0): brecha de 0.5 pts
  para evitar que la misma data genere apertura y cierre en el mismo dia.
- Sin trailing SL — el SL se fija al momento de entrada y no cambia.
- Variantes: score_entrada = 4.5, rsi_max = 65, multiplicadores ATR distintos.
