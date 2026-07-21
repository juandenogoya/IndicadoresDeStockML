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
| [METRICAS.md](METRICAS.md) | Medicion de riesgo y rendimiento: equity a mercado (`ft_equity_diaria`), max drawdown, Sharpe/Sortino con IC, benchmarks |
| **Estrategias activas** | |
| [estrategias/ML_SCANNER_v1.md](estrategias/ML_SCANNER_v1.md) | Bot ML — scoring scanner + ML prob |
| [estrategias/TECH_v1.md](estrategias/TECH_v1.md) | Bot Tecnico — SMA/MACD/RSI rule-based |
| [estrategias/SMC_v1.md](estrategias/SMC_v1.md) | Bot SMC — estructura BOS/CHoCH |
| [estrategias/TECH_SECTOR_v1.md](estrategias/TECH_SECTOR_v1.md) | Bot Sectorial — tech score con diversificacion sectorial |
| [estrategias/COMBO_v1.md](estrategias/COMBO_v1.md) | Bot Combo — sectorial + candle score desempate |
| **En desarrollo** | |
| [estrategias/TECH_SECTOR_v2.md](estrategias/TECH_SECTOR_v2.md) | Sectorial + retencion condicional + rotacion |
| [estrategias/SMC_v2.md](estrategias/SMC_v2.md) | SMC + filtro contexto entrada + salida agotamiento |
| [estrategias/TECH_SECTOR_OPTIONS_v1.md](estrategias/TECH_SECTOR_OPTIONS_v1.md) | Sectorial + confirmacion PCR_OI opciones |
| [estrategias/TECH_SECTOR_OPTIONS_v2.md](estrategias/TECH_SECTOR_OPTIONS_v2.md) | Sectorial + confirmacion PCR_VOL opciones |
| [estrategias/TECH_SECTOR_OIEXIT_v1.md](estrategias/TECH_SECTOR_OIEXIT_v1.md) | Entrada TECH_SECTOR_v1 + salida por OI walls / corrida |
| **Templates** | |
| [templates/ESTRATEGIA_TEMPLATE.md](templates/ESTRATEGIA_TEMPLATE.md) | Template para documentar nuevas versiones |

---

## Estado actual (2026-07-21)

> Forward Testing corre 100% en la DB **local** (Plan C). Los 10 bots se ejecutan
> con `scripts/manual/ft_run_diario.bat`. Railway no recibe escrituras de FT.

> **En curso (rama `feature/ft-metricas-riesgo`)**: capa de metricas de riesgo.
> La equity a costo de `ft_metricas_diarias` se reemplaza, para fines de
> medicion, por `ft_equity_diaria` (marcada a mercado, sin huecos ni fines de
> semana). Ver [METRICAS.md](METRICAS.md).

| ID | Estrategia | Estado | Inicio | Notas |
|----|---|---|---|---|
| 1 | ML_SCANNER_v1 | ACTIVA | 2026-04-28 | Benchmark Bot1 Alpaca |
| 2 | TECH_v1 | ACTIVA | 2026-04-28 | Benchmark Bot2 Alpaca |
| 3 | SMC_v1 | ACTIVA | 2026-04-28 | Benchmark Bot3 Alpaca |
| 4 | TECH_SECTOR_v1 | ACTIVA | 2026-05-02 | Sectorial 9 sectores |
| 5 | COMBO_v1 | ACTIVA | 2026-04-28 | Sectorial + candle score |
| 6 | TECH_SECTOR_v2 | EN DESARROLLO | 2026-05-17 | Retencion + rotacion intrasectorial |
| 7 | SMC_v2 | EN DESARROLLO | 2026-05-17 | Filtro contexto + salida agotamiento |
| 8 | TECH_SECTOR_OPTIONS_v1 | EN DESARROLLO | 2026-05-17 | Sectorial + PCR_OI opciones |
| 9 | TECH_SECTOR_OPTIONS_v2 | EN DESARROLLO | 2026-05-17 | Sectorial + PCR_VOL opciones |
| 10 | TECH_SECTOR_OIEXIT_v1 | EN DESARROLLO | 2026-05-18 | Entrada v1 + salida OI walls/corrida |

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
