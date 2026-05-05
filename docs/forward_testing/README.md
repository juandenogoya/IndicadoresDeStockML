# Forward Testing — Base de Conocimiento

**Objetivo**: Identificar que estrategias igualan o mejoran rendimientos de mercado,
comprendiendo todas las variables que determinan cada resultado para poder
explorar y parametrizar de la mejor manera posible.

---

## Indice

| Documento | Contenido |
|---|---|
| [GLOSARIO.md](GLOSARIO.md) | Definiciones canonicas de todos los terminos, features y metricas |
| [JOURNAL.md](JOURNAL.md) | Log cronologico de decisiones, hipotesis y observaciones |
| **Estrategias activas** | |
| [estrategias/ML_SCANNER_v1.md](estrategias/ML_SCANNER_v1.md) | Bot ML — scoring scanner + ML prob |
| [estrategias/TECH_v1.md](estrategias/TECH_v1.md) | Bot Tecnico — SMA/MACD/RSI rule-based |
| [estrategias/SMC_v1.md](estrategias/SMC_v1.md) | Bot SMC — estructura BOS/CHoCH |
| [estrategias/TECH_SECTOR_v1.md](estrategias/TECH_SECTOR_v1.md) | Bot Sectorial — tech score con diversificacion sectorial |
| [estrategias/COMBO_v1.md](estrategias/COMBO_v1.md) | Bot Combo — sectorial + candle score desempate |
| **Templates** | |
| [templates/ESTRATEGIA_TEMPLATE.md](templates/ESTRATEGIA_TEMPLATE.md) | Template para documentar nuevas versiones |

---

## Estado actual (2026-05-05)

| ID | Estrategia | Inicio | Retorno | Notas |
|----|---|---|---|---|
| 1 | ML_SCANNER_v1 | 2026-04-28 | +0.04% | Benchmark Bot1 Alpaca |
| 2 | TECH_v1 | 2026-04-28 | -2.65% | Benchmark Bot2 Alpaca |
| 3 | SMC_v1 | 2026-04-28 | en seguimiento | Benchmark Bot3 Alpaca |
| 4 | TECH_SECTOR_v1 | 2026-05-02 | +0.04% | Sectorial 9 sectores |
| 5 | COMBO_v1 | 2026-04-28 | +0.41% | Sectorial + candle score |

---

## Convencion de nombres

```
{LOGICA}_{VERSION}.md

Ejemplos:
  TECH_SECTOR_v1.md   <- primera version de la logica sectorial tecnica
  TECH_SECTOR_v2.md   <- segunda version (incorpora filtro momentum)
  SMC_v2.md           <- SMC con filtro de confirmacion por candle score
```

Cada version es un archivo independiente. Se mantienen todos — los archivos
de versiones anteriores son el registro historico de por que evolucionamos.

---

## Principio de documentacion

1. **Primero el documento, despues el codigo.**
   Antes de implementar una nueva version, escribir su archivo .md.
   El documento es el diseno; el codigo es la implementacion del diseno.

2. **El JOURNAL captura el por que.**
   El codigo captura el que. Git captura cuando. El JOURNAL captura por que.

3. **Los parametros exactos, no rangos.**
   Documentar el valor que efectivamente corrio, no "entre 4 y 5".

---

## Referencia — Archivos heredados

- `docs/estrategias_ft.md` — especificacion original (abril 2026), supersedida
  por los archivos individuales en este directorio. Se mantiene como referencia.
- `docs/backtesting_local_plan.md` — plan de backtesting historico
- `memory/forward_testing.md` — notas de diseno del sistema (uso en sesiones Claude)
