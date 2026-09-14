# ML_SCANNER_v2 — Documentacion de Estrategia

**Estado**: ACTIVA (desde la rueda 2026-09-14)
**ID en DB**: 11
**Script**: `scripts/forward_testing/ft_bot_ml_scanner_v2.py`
**Logica base**: `ml_scanner` (cerebro compartido `src/strategies/ml_scanner.py`)
**Control**: [ML_SCANNER_v1](ML_SCANNER_v1.md)

---

## Concepto

La MISMA estrategia que ML_SCANNER_v1, con la senal del modelo ML v2. Es la Fase 5
del reentrenamiento (Tarea 20, Etapa 3): en vez de reemplazar el modelo de la v1,
se corre en paralelo y se decide despues de medir.

**Pregunta que responde**: con las mismas reglas de entrada, salida y tamano de
posicion, el modelo v2 (RF calibrado, 196 tickers) elige mejores operaciones que el
modelo V3 de la v1 (123 tickers entrenados)?

**Que cambia y que no**:

| | v1 | v2 |
|---|---|---|
| Modelo ML | V3 (RF, desplegado 10/4/2026) | v2 (RF + calibracion isotonica, 13/9/2026) |
| Probabilidad | `ml_prob_ganancia` | `ml_prob_v2` |
| Cortes de puntos ML | 0,75 / 0,65 / 0,55 / 0,45 / 0,35 | 0,769 / 0,607 / 0,557 / 0,471 / 0,364 |
| Price action, score tecnico, bajistas | iguales | iguales |
| Reglas de entrada / salida / sizing | iguales | iguales |
| Columnas que lee | `alert_nivel` / `alert_score` | `alert_nivel_v2` / `alert_score_v2` |

Los cortes de la v2 dejan arriba la misma fraccion de casos que los de la v1 sobre
las mismas filas (holdout abr-ago 2026). Calibrada, la probabilidad vive en otra
escala: con los cortes de la v1 la v2 daria la mitad de senales fuertes y la
comparacion mediria selectividad, no modelo. Detalle y numeros:
`docs/ml_reentrenamiento.md` sec. 8c.

---

## Parametros de Entrada (identicos a v1)

| Parametro | Valor | Descripcion |
|---|---|---|
| nivel_min | COMPRA_FUERTE | `alert_nivel_v2` (score compuesto v2 >= 75) |
| score_min | 65 | `alert_score_v2` minimo |
| max_posiciones | 5 | Sin restriccion sectorial |
| max_deploy_pct | 80% | Techo de capital desplegado |
| riesgo_por_trade | 15% | Del capital actual |

Condiciones: nivel COMPRA_FUERTE v2 en la ultima corrida del scanner, score >= 65,
sin posicion abierta en el ticker, sin earnings proximos, slot y cash disponibles.
Ranking: `alert_score_v2 DESC`.

SL = entrada x 0,95 | TP = entrada x 1,10.

## Logica de Salida (identica a v1)

| Prioridad | Condicion | Motivo |
|---|---|---|
| P1 | Earnings manana | EARNINGS_MANANA |
| P2 | Sin senal v2, nivel v2 != COMPRA_FUERTE o score v2 < 65 | SCORE_DEGRADADO_* |
| P3 | precio <= SL | STOP_LOSS |
| P4 | precio >= TP | TAKE_PROFIT |

---

## Diferencias operativas con la v1

- **Guard**: si la ultima corrida del scanner no trae la v2 (artefacto
  `models_ml_v2/rf_cal_global.joblib` ausente o que no valida contra su
  metadata), el bot no opera y sale con codigo 1. ft_run_diario lo marca como
  error. Usar la ultima fila con v2 de otra corrida seria decidir con senal vieja.
- **Estado de posiciones**: toma la ultima fila CON v2 de cada ticker, igual que
  la v1 toma la ultima fila que exista.
- **Detalle de la operacion**: guarda `ml_prob_v2` y `ml_modelo_v2`, para atribuir
  despues por que difiere de la v1.

## Como se compara con la v1

- Mismo periodo, mismo capital, mismas reglas: la comparacion de cartera usa las
  metricas de `ft_equity_diaria` contra el grupo de control (METRICAS.md sec. 12).
- Por que difieren (Etapa 3f): seccion "ML v1 vs v2: por que difieren" del reporte
  HTML diario y `scripts/forward_testing/ft_comparar_ml.py`
  (`reportes/ft_comparar_ml.md`). Senales exclusivas de cada una sobre las mismas
  filas de `alertas_scanner`, con retorno real a 5/20 ruedas contra el universo;
  atribucion (tickers fuera del entrenamiento de la v1, nivel que dio la otra
  version); operaciones compartidas vs exclusivas; candidatos que quedaron afuera
  por el tope de 5 posiciones. Metodo: [METRICAS.md sec. 13](../METRICAS.md).
- Decision de reemplazo: en la Etapa 4 (~40 operaciones cerradas u 8 semanas).
- La v2 queda CONGELADA mientras dure la comparacion: reentrenarla seria un corte.

## Registro

- `ft_cambios`: `ml_scanner_v2_lanzamiento` (MODELO, estrategia 11, rueda 14/9) y
  `scanner_v2_en_paralelo` (INFRA, marca sobre la v1).
- JOURNAL 2026-09-14, LANZAMIENTO.
