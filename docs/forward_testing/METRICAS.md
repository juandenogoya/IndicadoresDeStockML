# Metricas de Riesgo y Rendimiento — Forward Testing

**Estado**: DISENO (2026-07-21). Documento previo a codificar.
**Alcance**: como se mide el rendimiento y el riesgo de las 10 estrategias FT.
**Rama**: `feature/ft-metricas-riesgo`

---

## 1. Por que existe este documento

Hasta hoy el FT responde "cuanto gano cada estrategia" (`retorno_acumulado_pct`).
No responde "cuanto riesgo tomo para ganarlo", que es la pregunta que decide si
una estrategia sigue viva. Al intentar calcular un Sharpe sobre los datos
existentes aparecieron tres problemas, el tercero de ellos bloqueante.

### Problema 1 — la serie no es diaria (contaminacion de calendario)

`ft_metricas_diarias` tiene 52 fechas entre 2026-04-23 y 2026-07-21:

- **12 marks en dia NO habil**: 8 sabados, 2 domingos, y 2 feriados NYSE
  (2026-06-19 Juneteenth, 2026-07-03 Independence Day observado).
  Producen retornos cero falsos que aplastan el desvio estandar.
- **21 de 61 dias habiles (34%) sin mark**. Cada hueco comprime 3-4 dias de
  mercado en una sola observacion "diaria", inflando el desvio de esa fila.

Causa: la rutina nocturna es MANUAL (decision de diseno, ver CLAUDE.md). El bot
solo escribe la fila del dia en que corre. La serie resultante es de muestreo
irregular, y anualizar eso con raiz(252) como si fuera diaria limpia es invalido.

### Problema 2 — n insuficiente para inferencia

Entre 28 y 39 dias habiles por estrategia. El Sharpe es un cociente y su error
estandar explota en muestras cortas (Lo, 2002):

```
SE(SR_anual) ~ sqrt((1 + SR_diario^2 / 2) / n) * sqrt(252)
```

Con n=39 el IC95% mide aproximadamente +-5 puntos de Sharpe. Nueve de las diez
estrategias tienen intervalo que incluye el cero: son indistinguibles del azar.
Para separar un Sharpe de 0.5 de uno de 1.0 con precision +-0.5 harian falta
~15 anios de datos diarios. **No se resuelve esperando.**

### Problema 3 (BLOQUEANTE) — `capital_total` no esta marcado a mercado

En `ft_utils.registrar_metricas_diarias()` se escribe:

```
capital_total = ft_estrategias.capital_actual
capital_actual = cash_disponible + capital_inmovilizado
capital_inmovilizado = SUM(capital_entrada)   <- COSTO DE ENTRADA
```

`capital_inmovilizado` es **costo historico**, no valor de mercado. Consecuencia
verificada: en el **100% de los dias sin cierres, en las 10 estrategias**, el
capital no se mueve.

```
2026-07-09   101851.91   5 posiciones   0 cierres   delta 0.00
2026-07-10   101851.91   5 posiciones   0 cierres   delta 0.00
2026-07-11   101851.91   5 posiciones   0 cierres   delta 0.00
```

Lo que llamamos "equity curve" es en realidad una **curva de PnL realizado**:
escalonada, plana entre cierres. Implicancias:

1. La volatilidad medida es **la cadencia de salidas**, no la fluctuacion de la
   cartera. Toda metrica de riesgo derivada de ella **subestima el riesgo**.
2. El **max drawdown es invisible**: una estrategia puede estar -15% contra
   posiciones abiertas y la curva se ve plana.
3. **Sesga el ranking a favor de las estrategias lentas**: menos salidas ->
   menos "vol" medida -> mas Sharpe. Parte de la brecha observada entre
   ML_SCANNER_v1 (+3.08) y OIEXIT_v1 (-7.11) es artefacto de cadencia de salida,
   no diferencia de riesgo.

**Conclusion**: no se puede calcular ninguna metrica de riesgo sobre la serie
actual. Hay que reconstruirla primero.

---

## 2. Principio de diseno: la equity curve es una capa DERIVADA

La buena noticia es que **la serie correcta es 100% reconstruible sin datos
nuevos**. Todo lo necesario ya vive en la DB local:

- `ft_operaciones`: fecha/precio/cantidad de entrada y de salida por posicion.
- `precios_diarios`: el close de cada ticker de cada dia habil.

Por lo tanto la equity diaria es una **funcion pura del estado de las
operaciones y de los precios**, recomputable en cualquier momento y hacia atras.
Es el mismo patron que `fundamentales_ratios_q` (capa derivada, recomputable sin
re-fetch) ya establecido en el proyecto.

Esto tiene tres consecuencias de diseno:

1. **No dependemos de haber corrido el bot ese dia.** Los 21 dias habiles sin
   mark se reconstruyen. El caracter manual de la rutina nocturna deja de
   contaminar la medicion.
2. **Es idempotente y auditable.** Si aparece un bug en el calculo, se corrige y
   se recomputa toda la historia. No hay estado que se pierda.
3. **No tocamos los bots.** El motor de metricas lee; no escribe en
   `ft_operaciones` ni cambia la logica de trading.

### Decision: tabla nueva, no mutar la existente

`ft_metricas_diarias` **se mantiene intacta**. Es el log operativo de lo que el
bot vio cuando corrio (lo leen `ft_reporte_html.py` y el MCP). Cambiar la
semantica de `capital_total` en el lugar romperia consumidores y arriesgaria los
bots en produccion.

La capa derivada vive en una tabla nueva: **`ft_equity_diaria`**.

> Regla: **las metricas de riesgo se calculan SOLO desde `ft_equity_diaria`.**
> `ft_metricas_diarias.capital_total` NO es valor de mercado y no debe usarse
> para medir riesgo, volatilidad ni drawdown.

---

## 3. Tabla `ft_equity_diaria`

Una fila por (estrategia, dia habil), desde el inicio de cada estrategia hasta
el ultimo cierre disponible. Sin huecos y sin fines de semana.

```sql
CREATE TABLE ft_equity_diaria (
    estrategia_id      INTEGER NOT NULL REFERENCES ft_estrategias(id),
    fecha              DATE    NOT NULL,

    equity             NUMERIC(14,2) NOT NULL,  -- cash + valor_mercado
    cash               NUMERIC(14,2) NOT NULL,
    valor_mercado      NUMERIC(14,2) NOT NULL,  -- posiciones abiertas a close del dia
    costo_posiciones   NUMERIC(14,2) NOT NULL,  -- cost basis (para comparar)

    n_posiciones       SMALLINT NOT NULL DEFAULT 0,
    exposicion_pct     NUMERIC(7,4),            -- valor_mercado / equity

    pnl_realizado_dia  NUMERIC(14,2) NOT NULL DEFAULT 0,
    pnl_no_realizado   NUMERIC(14,2) NOT NULL DEFAULT 0,
    retorno_dia_pct    NUMERIC(9,6),            -- vs dia habil anterior de la serie
    retorno_acum_pct   NUMERIC(9,4),            -- vs capital_inicial

    precios_stale      SMALLINT NOT NULL DEFAULT 0,  -- tickers marcados con close viejo
    calculado_en       TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (estrategia_id, fecha)
);
```

LOCAL-only (Plan C: FT es local, fuente de verdad).

---

## 4. Reglas de construccion

**Posicion abierta al cierre del dia `d`**:
```
fecha_entrada <= d  AND  (fecha_salida IS NULL OR fecha_salida > d)
```

**Cash**:
```
cash(d) = capital_inicial
        - SUM(capital_entrada)          de ops con fecha_entrada <= d
        + SUM(precio_salida * cantidad) de ops con fecha_salida  <= d
```

**Valor de mercado**:
```
valor_mercado(d) = SUM( close(ticker, d) * cantidad )  sobre posiciones abiertas en d
equity(d)        = cash(d) + valor_mercado(d)
```

**Coherencia en los bordes** (verificada por construccion): el precio de
ejecucion del FT es el close del dia, entonces el dia de entrada
`close(d)*cantidad == capital_entrada` y la equity no salta; el dia de salida la
posicion ya no cuenta y el efectivo entra por `precio_salida`. Sin doble conteo.

**Dias habiles**: la serie se construye SOLO sobre dias habiles NYSE via
`src/utils/trading_calendar.is_trading_day` (regla no negociable #1 del
proyecto). Nunca se escribe una fila en fin de semana o feriado.

**Ticker sin precio ese dia**: se arrastra el ultimo close conocido
(forward-fill) y se incrementa `precios_stale`. Es preferible a interrumpir la
serie; el contador deja el hecho visible en vez de esconderlo.

**Inicio de serie**: primer dia habil >= `ft_estrategias.fecha_inicio`. Antes de
esa fecha la estrategia no existe (no se rellena con capital_inicial plano, para
no inventar dias de retorno cero).

---

## 5. Metricas de serie

Calculadas sobre `retorno_dia_pct` de `ft_equity_diaria`.

| Metrica | Formula | Nota |
|---|---|---|
| Retorno acumulado | `equity_fin / equity_ini - 1` | ya existe, ahora correcto |
| Volatilidad anual | `std(r, ddof=1) * sqrt(252)` | |
| **Max drawdown** | `max_d( (peak_hasta_d - equity_d) / peak_hasta_d )` | **nuevo: hoy es invisible** |
| Duracion del DD | dias habiles entre peak y recuperacion | NULL si no recupero |
| Sharpe | `(mean(r) - rf_d) / std(r) * sqrt(252)` | siempre con IC (seccion 8) |
| Sortino | `(mean(r) - rf_d) / downside_dev * sqrt(252)` | `dd = sqrt(mean(min(r - rf_d, 0)^2))` |
| Exposicion media | `mean(exposicion_pct)` | separa habilidad de cash drag |
| Information Ratio | `mean(r - r_bench) / std(r - r_bench) * sqrt(252)` | vs benchmark, seccion 7 |
| Beta | `cov(r, r_bench) / var(r_bench)` | cuanto es solo mercado |

### Sortino antes que Sharpe

Las 10 estrategias son long-only con stop loss: distribucion **asimetrica** por
diseno (perdidas cortadas, ganancias con cola). El Sharpe castiga la volatilidad
al alza, que es exactamente lo que la Fase 2 de `TECH_SECTOR_OIEXIT_v1` esta
disenada para capturar. **Sortino es el ratio primario; Sharpe se reporta por
convencion y comparabilidad externa.**

### Tasa libre de riesgo

`rf` parametrizable, **default 4.0% anual** (tasa cash 2026).
`rf_diario = (1 + rf_anual)^(1/252) - 1`. No es despreciable: 4% anual son
~0.0155%/dia contra medias diarias del orden de 0.14%. Se reporta siempre que
`rf` se uso.

---

## 6. Metricas de trade

Aca esta el n grande y por lo tanto la senal estadistica real: entre 13 y 584
operaciones cerradas por estrategia, contra 28-39 dias de serie.

| Metrica | Formula |
|---|---|
| n operaciones | cerradas (`fecha_salida IS NOT NULL`) |
| Win rate | `n_ganadoras / n` |
| Expectancy % | `mean(pnl_pct)` |
| Profit factor | `SUM(pnl>0) / abs(SUM(pnl<0))` |
| Payoff ratio | `mean(pnl_pct \| ganadoras) / abs(mean(pnl_pct \| perdedoras))` |
| Duracion media | dias habiles entre entrada y salida |
| Por motivo de salida | las anteriores, agrupadas por `motivo_salida` |

El corte **por `motivo_salida`** es el que conecta con el analisis de
TIME_STOP en SMC_v1 (JOURNAL 2026-07-18): permite comparar la calidad de cada
regla de salida entre versiones.

### Limitacion conocida: expectancy en R

`R = (precio_entrada - stop_loss_inicial) * cantidad` requiere el SL **inicial**.
Pero `ft_utils.actualizar_stop_loss()` hace `UPDATE ... SET stop_loss = :sl`,
o sea el trailing **pisa** el valor original. En las estrategias con trailing el
SL inicial no es recuperable desde la columna.

Plan: verificar en implementacion si `detalle_entrada` (JSONB) lo conserva.
- Si lo conserva -> expectancy en R para todas.
- Si no -> expectancy en R solo donde el SL es fijo, y se agrega columna
  `stop_loss_inicial` a `ft_operaciones` para que quede disponible **hacia
  adelante**. La historia previa no se puede reconstruir en R.

---

## 7. Benchmarks

Una estrategia long-only en un mercado alcista sube por beta, no por habilidad.
Sin benchmark, cualquier retorno positivo se lee como exito.

| Benchmark | Fuente | Que responde |
|---|---|---|
| **Universo equiponderado** | `precios_diarios` + `activos` (activo=TRUE) | ¿la SELECCION de tickers aporta algo sobre comprar todo? |
| **ES=F** (S&P 500) | `futuros_diarios` | referencia de mercado estandar |

`SPY` no sirve: el universo son 200 acciones, sin ETFs (ver
`docs/gestion_universo.md`).

**Regla de ventana**: cada estrategia arranca en fecha distinta (23/4 a 27/5).
El benchmark se recalcula sobre **la misma ventana de cada estrategia**.
Comparar contra un benchmark de ventana fija seria comparar periodos distintos.

Medicion de referencia sobre 2026-04-23 -> 2026-07-21 (dias habiles):

| | Retorno | Sharpe (rf=0) |
|---|---|---|
| ES=F | +4.77% | +1.61 |
| Universo equiponderado | +1.55% | +0.54 |

Con 7 de 10 estrategias en negativo sobre ese mismo periodo, el hallazgo
importante ya es visible sin ningun ratio: **en un mercado que subio, la mayoria
de las estrategias pierde contra comprar todo el universo.**

---

## 8. Regla no negociable: la incertidumbre se reporta siempre

Con n entre 28 y 39, un Sharpe puntual es una cifra sin contenido. Por lo tanto:

> **Todo Sharpe/Sortino se reporta con `n` y su IC95%.**
> Si el IC incluye cero, se marca explicitamente **NO CONCLUYENTE**.

```
SE(SR_anual) = sqrt((1 + SR_diario^2 / 2) / n) * sqrt(252)
IC95%        = SR +- 1.96 * SE
```

Esto no es decoracion estadistica: es lo que evita decidir que
`ML_SCANNER_v1` "es la mejor" cuando su IC95% es [-1.95, +8.11].

**Ningun cambio de estrategia se justifica con un Sharpe cuyo IC incluye cero.**
Las decisiones a este n se toman con metricas de trade (seccion 6) y con
comparacion directa contra benchmark (seccion 7).

---

## 9. Que NO vamos a hacer (y por que)

- **No anualizar retornos (CAGR)** con menos de 1 anio de serie: extrapolar
  3 meses a 12 multiplica el ruido por 4.
- **No rankear estrategias por Sharpe** mientras los IC se solapen. El ranking
  seria de ruido.
- **No incorporar costos de transaccion todavia**: en STANDBY por decision
  explicita (pendiente definir escenario de broker y verificar costos 2026).
  Cuando se retome, entra como ajuste al PnL de `ft_operaciones`, aguas arriba
  de este modulo, y todas las metricas se recomputan solas.
- **No deprecar `ft_metricas_diarias`** en esta iteracion. Se evalua recien
  cuando `ft_equity_diaria` este validada y los consumidores migrados.

---

## 10. Archivos

### Se crean
| Archivo | Rol |
|---|---|
| `scripts/oneshot/create_ft_equity_table.py` | DDL de `ft_equity_diaria` (one-shot) |
| `src/utils/ft_metricas.py` | Modulo **PURO**: ratios sobre una serie de retornos (sin DB, sin config). Testeable e importable por el MCP |
| `scripts/forward_testing/ft_compute_equity.py` | Motor: reconstruye la equity desde `ft_operaciones` + `precios_diarios` y escribe `ft_equity_diaria`. Idempotente, con `--backfill` y `--desde` |

### Se modifican
| Archivo | Cambio |
|---|---|
| `scripts/forward_testing/ft_reporte_html.py` | Bloque de riesgo: max DD, Sortino, Sharpe+IC, exposicion, vs benchmark |
| `scripts/manual/ft_run_diario.bat` | Encadenar `ft_compute_equity.py` antes del reporte |
| `docs/forward_testing/GLOSARIO.md` | Seccion G (metricas de riesgo) + advertencia en seccion C |
| `docs/forward_testing/JOURNAL.md` | Entrada del diagnostico |
| `docs/forward_testing/README.md` | Indice |

### Al cerrar
`CLAUDE.md` (tabla `ft_equity_diaria` en "Tablas DB principales" y script en la
tabla de referencia rapida) y `memory/forward_testing.md`. Se actualizan **al
final**, cuando reflejen lo que efectivamente corre.

---

## 11. Orden de implementacion

1. **Tabla + motor + backfill.** Reconstruir la serie completa de las 10
   estrategias. Validar: sin fines de semana, sin huecos, y el `equity` del
   ultimo dia de una estrategia sin posiciones abiertas debe coincidir con
   `ft_estrategias.capital_actual` (control de cuadre).
2. **Modulo puro de metricas** + metricas de trade (n grande, valor inmediato).
3. **Reporte HTML**: bloque de riesgo con IC y benchmark.
4. **Cierre**: CLAUDE.md, memoria, JOURNAL con los primeros numeros reales.
