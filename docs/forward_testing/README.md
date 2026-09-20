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
| [ANALISIS_SALIDAS.md](ANALISIS_SALIDAS.md) | Analisis de SALIDAS (19/9/2026): que hizo el precio despues de cada salida, contra el universo; anatomia de la salida de TECH_SECTOR_v1 (el score es una regla de si/no; la historia antes del 29/5 es el bug del score 0,0); pre-registro del paso 1 (sin SMA21) |
| [ANALISIS_ENTRADAS.md](ANALISIS_ENTRADAS.md) | Analisis de ENTRADAS de TECH_SECTOR_v1 (19-20/9/2026), espejo del anterior: la salida y el reparto quedan fijos y se varia la entrada. PASO 0 hecho -- el filtro califica a 48,7 candidatos por rueda para 45 lugares y el desempate entre empatados es ALFABETICO tambien en la sectorial; influencia de cada condicion (SMA21 decide mas que el MACD); distancia a la media y RSI en banda son casi incompatibles; los ponderadores SI eligen tickers distintos (Jaccard 0,72 a 0,96). PASO 1 hecho (20/9): se corrieron las **166 reglas booleanas monotonas** enteras (incluidas las 18 que ningun score ponderado alcanza) mas 20 sorteos de control, y **ninguna pasa las cuatro condiciones pre-registradas**. El resultado que manda es la **banda del sorteo**: la misma regla con el desempate sorteado da de +16,3% a +26,3% de retorno, ~10 pp, contra los +0,9 pp de la mejor regla -- el desempate entre candidatos empatados pesa mas que la regla de entrada. La grilla sin ajustar mide exposicion (correlacion +0,61/+0,67) y el orden entre reglas no se sostiene de un periodo al otro (Spearman +0,19) |
| **Estrategias activas** | |
| [estrategias/ML_SCANNER_v1.md](estrategias/ML_SCANNER_v1.md) | Bot ML — scoring scanner + ML prob |
| [estrategias/ML_SCANNER_v2.md](estrategias/ML_SCANNER_v2.md) | Bot ML v1 con el modelo ML v2 calibrado, en paralelo (control: v1) |
| [estrategias/TECH_v1.md](estrategias/TECH_v1.md) | Bot Tecnico — SMA/MACD/RSI rule-based |
| [estrategias/SMC_v1.md](estrategias/SMC_v1.md) | Bot SMC — estructura BOS/CHoCH (control de la v3) |
| [estrategias/SMC_v3.md](estrategias/SMC_v3.md) | Bot SMC sobre estructura CONFIRMADA, N=5 y N=3 (control: v1) |
| [estrategias/TECH_SECTOR_v1.md](estrategias/TECH_SECTOR_v1.md) | Bot Sectorial — tech score con diversificacion sectorial |
| **En desarrollo** | |
| [estrategias/TECH_SECTOR_v2.md](estrategias/TECH_SECTOR_v2.md) | Sectorial + retencion condicional + rotacion |
| [estrategias/TECH_SECTOR_OPTIONS_v1.md](estrategias/TECH_SECTOR_OPTIONS_v1.md) | Sectorial + confirmacion PCR_OI opciones |
| [estrategias/TECH_SECTOR_OPTIONS_v2.md](estrategias/TECH_SECTOR_OPTIONS_v2.md) | Sectorial + confirmacion PCR_VOL opciones |
| [estrategias/TECH_SECTOR_OIEXIT_v1.md](estrategias/TECH_SECTOR_OIEXIT_v1.md) | Entrada TECH_SECTOR_v1 + salida por OI walls / corrida |
| **Discontinuadas** (la ficha cierra con periodo, parametros, metricas y motivos) | |
| [estrategias/COMBO_v1.md](estrategias/COMBO_v1.md) | Sectorial + candle score de desempate. BAJA 17/9/2026: las velas no aportan |
| [estrategias/SMC_v2.md](estrategias/SMC_v2.md) | SMC + filtro de contexto + salida por agotamiento. BAJA 17/9/2026: peor equity, insumos sin valor |
| **Templates** | |
| [templates/ESTRATEGIA_TEMPLATE.md](templates/ESTRATEGIA_TEMPLATE.md) | Template para documentar nuevas versiones |

---

## Estado actual (2026-09-17)

> Forward Testing corre 100% en la DB **local** (Plan C). Los 11 bots ACTIVOS se
> ejecutan con `scripts/manual/ft_run_diario.bat`. Railway no recibe escrituras de FT.

> **17/9/2026**: entran FT_SMC_v3_N5 y FT_SMC_v3_N3 (estructura confirmada); salen
> FT_COMBO_v1 y FT_SMC_v2 (discontinuadas, posiciones liquidadas, `activa = FALSE`).
> Una estrategia dada de baja conserva su ficha, su historia en `ft_operaciones` y
> `ft_equity_diaria`, y su entrada en `ft_setup_estrategias` con la marca
> `discontinuada`. Criterios de baja: CLAUDE.md, "Alta y baja de estrategias FT".

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
| 5 | COMBO_v1 | **DISCONTINUADA 17/9/2026** | 2026-04-28 | Sectorial + candle score. Las velas no aportan: backtest +19,9% vs +24,4% sin velas; FT -0,09% vs +2,29% |
| 6 | TECH_SECTOR_v2 | EN DESARROLLO | 2026-05-17 | Retencion + rotacion intrasectorial |
| 7 | SMC_v2 | **DISCONTINUADA 17/9/2026** | 2026-05-17 | Filtro contexto + salida agotamiento. Peor equity (-5,46%); la salida nueva nunca disparo |
| 8 | TECH_SECTOR_OPTIONS_v1 | EN DESARROLLO | 2026-05-17 | Sectorial + PCR_OI opciones |
| 9 | TECH_SECTOR_OPTIONS_v2 | EN DESARROLLO | 2026-05-17 | Sectorial + PCR_VOL opciones |
| 10 | TECH_SECTOR_OIEXIT_v1 | EN DESARROLLO | 2026-05-18 | Entrada v1 + salida OI walls/corrida |
| 11 | ML_SCANNER_v2 | ACTIVA | 2026-09-14 | ML_SCANNER_v1 con el modelo ML v2 calibrado; control: v1 |
| 12 | SMC_v3_N5 | ACTIVA | 2026-09-16 | Regla de SMC_v1 sobre estructura CONFIRMADA N=5; control: v1 |
| 13 | SMC_v3_N3 | ACTIVA | 2026-09-16 | Idem con N=3. Las dos corren porque el backtest no distingue |

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
