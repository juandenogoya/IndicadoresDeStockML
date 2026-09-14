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
Solo entra cuando el scanner emite `COMPRA_FUERTE` (score compuesto >= 75; ver la
correccion del 13/9/2026 abajo). Desde el 14/9/2026 corre en paralelo
[ML_SCANNER_v2](ML_SCANNER_v2.md), la misma estrategia con el modelo ML v2: esta
queda como control.

**Pregunta que responde**: ?el scanner ML tiene valor predictivo real cuando
se opera de forma sistematica? ?Sirve como estrategia standalone?

---

## Parametros de Entrada

> **Correccion 13/9/2026**: esta seccion decia que la entrada era por
> `ml_prob_ganancia >= 0.65`. El codigo (`ft_bot_ml_scanner.py` +
> `src/strategies/ml_scanner.py`) nunca lo hizo: entra por el **score compuesto**
> del scanner (`alert_score`, 0-100), que suma ML + price action + score tecnico -
> senales bajistas (`src/pipeline/alert_classifier.py`). COMPRA_FUERTE es score >= 75.
> Tampoco las prioridades de salida eran las del codigo. Abajo, lo que corre.

| Parametro | Valor | Descripcion |
|---|---|---|
| nivel_min | COMPRA_FUERTE | Nivel minimo de alerta (score compuesto >= 75) |
| score_min | 65 | `alert_score` minimo (redundante con el nivel) |
| max_posiciones | 5 | Sin restriccion sectorial |
| max_deploy_pct | 80% | Techo de capital desplegado |
| riesgo_por_trade | 15% | Del capital actual |

### Condiciones (TODAS obligatorias)
1. `alert_nivel = COMPRA_FUERTE` en la ultima corrida del scanner
2. `alert_score >= 65`
3. ticker sin posicion abierta
4. ticker sin earnings proximos
5. `posiciones_abiertas < 5` y cash desplegable

Ranking: `alert_score DESC`

### SL / TP
```
SL = precio_entrada * (1 - 0.05)   # 5% fijo
TP = precio_entrada * (1 + 0.10)   # 10% fijo
```

---

## Logica de Salida

| Prioridad | Condicion | Motivo |
|---|---|---|
| P1 | Earnings manana | EARNINGS_MANANA |
| P2 | Sin senal, `alert_nivel != COMPRA_FUERTE` o `alert_score < 65` | SCORE_DEGRADADO_* |
| P3 | precio <= SL | STOP_LOSS |
| P4 | precio >= TP | TAKE_PROFIT |

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
