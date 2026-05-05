# ML_SCANNER_v1 — Documentacion de Estrategia

**Estado**: ACTIVA
**ID en DB**: 1
**Inicio**: 2026-04-28
**Script**: `scripts/forward_testing/ft_bot_ml_scanner.py`
**Logica base**: `ml_scanner`

---

## Concepto

Replica del Bot1 de Alpaca en el entorno de forward testing.
Usa el pipeline de alertas del scanner ML como fuente de senales.
Solo entra cuando el scanner emite `COMPRA_FUERTE` con probabilidad ML >= 65%.

**Pregunta que responde**: ?el scanner ML tiene valor predictivo real cuando
se opera de forma sistematica? ?Sirve como estrategia standalone?

---

## Parametros de Entrada

| Parametro | Valor | Descripcion |
|---|---|---|
| nivel_min | COMPRA_FUERTE | Nivel minimo de alerta |
| score_ml_min | 0.65 | Probabilidad ML minima (ml_prob_ganancia) |
| max_posiciones | 5 | Sin restriccion sectorial |
| riesgo_por_trade | 15% | Del capital actual |

### Condiciones (TODAS obligatorias)
1. `alert_nivel = COMPRA_FUERTE`
2. `ml_prob_ganancia >= 0.65`
3. ticker sin posicion abierta
4. ticker sin earnings proximos
5. `posiciones_abiertas < 5`

Ranking: `ml_prob_ganancia DESC`

### SL / TP
```
SL = precio_entrada * (1 - 0.05)   # 5% fijo
TP = precio_entrada * (1 + 0.10)   # 10% fijo
```

---

## Logica de Salida

| Prioridad | Condicion | Motivo |
|---|---|---|
| P0 | Earnings manana | EARNINGS_MANANA |
| P1 | precio <= SL | SL |
| P2 | precio >= TP | TP |
| P3 | alert_nivel != COMPRA_FUERTE | SCORE_DEGRADADO |

El exit primario es la degradacion del scanner.
SL/TP son de emergencia — protegen contra movimientos extremos antes de la proxima evaluacion.

---

## Metricas al 2026-05-05

| Metrica | Valor |
|---|---|
| Dias activa | 7 |
| Capital actual | ~$100,036 |
| Retorno total | +0.04% |

---

## Notas

- La fuente de senales (alertas_scanner) se genera una vez por dia por el cron.
- El bot evalua las alertas del dia anterior (ultimo escaneo disponible).
- Variantes posibles: nivel_min = COMPRA, score_ml_min = 0.70, agregar filtro MTF.
