# Metricas de Riesgo y Rendimiento — Forward Testing

**Estado**: IMPLEMENTADO (2026-07-21). Tabla, motor y modulo puro en produccion local.
Seccion 12 (medir un cambio: `ft_cambios`, tramos y grupo de control) IMPLEMENTADA 2026-09-13.
**Alcance**: como se mide el rendimiento y el riesgo de las 10 estrategias FT.
**Rama**: `feature/ft-metricas-riesgo`

> **PREREQUISITO RESUELTO — splits (2026-07-21)**: al construir la primera
> equity a mercado, el control de cuadre destapo que `precios_diarios` no se
> re-ajustaba ante splits (KLAC 10:1, CRWD 4:1). Ocho operaciones de FT tenian
> -12.709 USD de perdidas ficticias y 4 estrategias reportaban ~3.9 puntos de
> retorno de menos. Corregido antes de calcular cualquier metrica; detalle en
> JOURNAL 2026-07-21 BUG FIX y en `scripts/manual/splits.py`.
>
> Leccion de diseno: **el control de cuadre valia mas que las metricas**. Fue
> lo que convirtio "los numeros dan raro" en "estos 8 trades estan mal y este
> es el motivo".

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

### Expectancy en R — RESUELTO (2026-07-21)

`R = (precio_entrada - stop_loss_inicial) * cantidad` requiere el SL **inicial**,
y `ft_utils.actualizar_stop_loss()` pisa la columna `stop_loss` con el trailing.
La verificacion sobre las 1.811 operaciones dio mejor resultado del esperado:

- **Solo 3 estrategias usan trailing**: SMC_v1 (3), SMC_v2 (7) y OIEXIT_v1 (10).
  En las otras 7 no hay trailing, asi que `ft_operaciones.stop_loss` **es** el
  inicial.
- **Las 3 con trailing conservan el SL inicial en `detalle_entrada`**:
  `sl_inicial` (junto a `r_value`, `sl_source`, `tp_ref_3r`) en OIEXIT_v1;
  `swing_low` en SMC_v1 y SMC_v2.

Orden de resolucion del SL inicial (implementar en el consumidor):
1. `detalle_entrada.sl_inicial`   (estrategia 10)
2. `detalle_entrada.swing_low`    (estrategias 3, 7)
3. `ft_operaciones.stop_loss`     (resto: sin trailing, es el inicial)

**Conclusion: R es reconstruible para las 10 estrategias con historia completa.**
No hace falta agregar columna ni resignar historia.

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

## 11. Estado de implementacion

| Paso | Estado |
|---|---|
| 1. Tabla + motor + backfill | **HECHO** — 523 filas, 10 estrategias, cuadre exacto |
| 2. Modulo puro + metricas de trade | **HECHO** — `src/utils/ft_metricas.py` |
| 3. Reporte HTML con bloque de riesgo | **HECHO** |
| 4. Encadenar a `ft_run_diario.bat` | **HECHO** (paso final 0, antes del reporte) |
| 5. Detector de splits en el pipeline diario | PENDIENTE (hoy es manual) |

### Reporte HTML

`ft_reporte_html.py` suma tres bloques y cambia la fuente del grafico:

- **Grafico**: pasa a leer `ft_equity_diaria.equity` (a mercado) en vez de
  `ft_metricas_diarias.capital_total` (a costo), y superpone el **universo
  equiponderado** como linea negra punteada, reescalado a 100k. Lo que quede por
  debajo de esa linea no esta aportando por seleccionar.
- **Riesgo**: max DD, volatilidad, Sortino, Sharpe **con IC95% y marca
  `no concl.`**, Information Ratio, beta y exposicion media. Con texto explicito
  de como leerla — el reporte se mira meses despues y el IC sin explicacion se
  ignora.
- **Por operacion**: n, win rate, expectancy, profit factor, payoff, ganancia y
  perdida medias, duracion. Es donde hay senal real a este horizonte.

El benchmark se reindexa a las fechas de **cada** estrategia (arrancan en dias
distintos). La duracion se calcula sobre `fecha_datos`, no sobre la de registro.

### Ventana comparable (constante `VENTANA_COMPARABLE`, `--desde`)

El reporte muestra **dos ventanas**, y la principal NO es la historia completa:

- **Ventana comparable — desde 2026-05-30** (default). Es la fecha del fix del
  bug de `SCORE_DEGRADADO_0.0`: antes de eso la historia de las estrategias 4,
  6, 8 y 9 mide el bug, no la estrategia. Es la unica ventana en la que las 10
  son comparables entre si.
- **Historia completa**, como secundaria: sirve para ver mas de un regimen de
  mercado, pero no para rankear.

Cada encabezado muestra **el retorno del benchmark de esa ventana**, y no por
prolijidad: entre las dos el benchmark **cambia de signo** (+8.26% en la
completa, -4.95% en la comparable). Leer los retornos contra cero en vez de
contra el benchmark lleva a la conclusion opuesta a la correcta — con los datos
del 21/7, ocho de diez estrategias estan en rojo absoluto en la ventana
comparable y las diez le ganan al mercado. Ver JOURNAL 2026-07-21 RESULTADO.

Las tablas se ordenan por **Sortino** (riesgo) y por **profit factor** (trade),
no por el orden de los ids.

**El retorno del bloque de riesgo difiere del resumen comparativo** y esta
aclarado en el propio reporte: el de riesgo sale de la serie de equity, que
termina en el ultimo cierre cargado; el del resumen usa el estado actual de la
estrategia, que ya incluye las operaciones de hoy. Un dia de diferencia es lo
esperado en un sistema asincronico.

### Encadenado

En `ft_run_diario.bat`, `ft_compute_equity.py` corre **despues de los 10 bots y
antes del reporte**. Corre todos los dias aunque no se opere, porque la equity a
mercado cambia por el movimiento del mercado, no por la actividad. Si falla no se
pierde nada: el dato vive en `ft_operaciones` + `precios_diarios` y la corrida
siguiente lo reconstruye entero.

### Control de cuadre (el invariante)

El cash es **puro flujo realizado**, asi que el reconstruido debe coincidir
exacto con `ft_estrategias.cash_disponible`, existan o no posiciones abiertas:

```
cash = capital_inicial - SUM(capital_entrada) + SUM(precio_salida * cantidad)
```

**Se calcula sobre TODAS las operaciones, sin corte de fecha** — no sobre el
ultimo dia de la serie. La serie termina en el ultimo cierre de
`precios_diarios`, pero el sistema es **asincronico** (los bots operan con el
OHLCV del dia habil anterior), asi que normalmente hay operaciones fechadas
DESPUES del ultimo precio disponible. Comparar contra el ultimo dia de la serie
da un falso descuadre — fue el primer error del control.

Tolerancia 0.05 USD: el cash se arma de sumas de `NUMERIC(12,2)` redondeados.

### Fechado asincronico: `fecha_datos` vs `fecha_entrada`

El sistema es **asincronico por diseno**: los bots deciden y ejecutan con el
OHLCV del ultimo cierre disponible, no con el del dia en que corren. No es un
bug — es la convencion de todo el proyecto (indicadores, osciladores, señales).

**Pero el desfase NO es fijo.** Medido sobre las 1.811 operaciones:

| desfase (dias habiles) | entradas | causa |
|---|---|---|
| 0 (mismo dia) | 302 (16.7%) | la recovery ya habia corrido |
| 1 | 1.325 (73.2%) | el caso tipico |
| 2 | 112 (6.2%) | se salteo una noche |
| 5 | 62 (3.4%) | se salteo una semana |
| 6 | 10 (0.6%) | dia de lanzamiento (23/4 uso datos del 15/4) |

Depende de cuan rancia estaba `precios_diarios` cuando corrio el bot, porque la
rutina nocturna es manual.

> **CORRECCION de una version previa de este documento.** Aca se afirmaba que la
> serie de retornos era *identica* bajo cualquier convencion de fechado. Eso
> **solo vale para desfase = 1**. Con desfase 2, 5 o 6, marcar la posicion por
> primera vez el dia del REGISTRO mete varios dias de movimiento de mercado
> como un salto de un solo dia: infla la volatilidad y distorsiona el drawdown.
> Afectaba a ~10% de las operaciones.

**Solucion**: columnas `fecha_datos` y `fecha_datos_salida` en `ft_operaciones`.

- **Hacia adelante** las escribe `ft_utils` solo, via `obtener_fecha_datos()`:
  como el precio que usan los bots ES el ultimo close, la fecha del dato es
  `MAX(fecha)` de ese ticker. **Los 10 bots no necesitaron cambios** — todos
  pasan por `abrir_operacion`/`cerrar_operacion`, que es el unico lugar del
  proyecto que escribe `ft_operaciones`.
- **La historia** se backfilleo con
  `scripts/oneshot/add_fecha_datos_ft_operaciones.py`. El matching directo de
  precio contra el close falla en las salidas por SL/TP (el precio es el nivel
  del stop, no un cierre). Se resolvio con la observacion de que **todas las
  operaciones de una misma corrida comparten la misma fecha de dato**: se agrupa
  por `(estrategia, fecha de registro)`, se juntan los votos de las ops que si
  matchean, y la moda del grupo se aplica a todo el grupo. **100% resuelto**,
  entradas y salidas.

`ft_compute_equity` usa `COALESCE(fecha_datos, fecha_entrada)`: si alguna
quedara sin resolver se degrada a la fecha de registro en vez de perderla.

**`ft_posiciones_diarias` tiene la misma trampa** (agregado 2026-09-12): `fecha` es
el dia en que corrio el bot, pero `precio_cierre` -- y con el retorno, rango, ATR y
scores de la fila -- es el ultimo close disponible. Columna `fecha_datos`:
- Hacia adelante la escribe `ft_utils.registrar_estado_posiciones()` con el
  `MAX(fecha)` del ticker, el mismo criterio que `obtener_fecha_datos()`.
- La historia se backfilleo con `scripts/oneshot/add_fecha_datos_ft_posiciones.py`:
  match EXACTO y UNICO del precio contra el close; si no hay match, o hay dos ruedas
  con el mismo close, la moda de la corrida. Resolvio el 100% de 23.541 filas
  (22.972 por match, 569 por moda). Desfase: 18,8% del mismo dia, 79,6% de la rueda
  anterior. Control contra `ft_operaciones` el dia de entrada y de cierre: 99,3% y
  99,96% coinciden; las 19 diferencias son todas de UNA noche (2026-06-02).
- OJO con la tolerancia: un centavo (la de `add_fecha_datos_ft_operaciones.py`, que
  funciona porque aplica la moda a TODA la corrida) no sirve para matchear filas
  sueltas. En tickers baratos dos ruedas seguidas difieren en un centavo y el match
  elegia la mas reciente (LAC 3,14 el 13/7 y 3,15 el 14/7).

**Efecto medido** (volatilidad anualizada, antes -> despues):

| Estrategia | antes | despues |
|---|---|---|
| TECH_SECTOR_v1 (584 ops) | 19.7% | **16.7%** |
| TECH_SECTOR_OPTIONS_v2 | 12.2% | 11.0% |
| TECH_SECTOR_v2 | 13.2% | 11.7% |
| ML_SCANNER_v1 | 14.9% | 14.5% |

La caida mas grande es la de TECH_SECTOR_v1, que es justamente la de mayor
churn (584 operaciones) y por lo tanto la mas expuesta al artefacto. Coherente.

**Uso obligatorio en cualquier analisis**: para cruzar una operacion con
`precios_diarios`, `indicadores_tecnicos` o cualquier tabla de mercado hay que
usar `fecha_datos`, **no** `fecha_entrada`. Con la fecha de registro se lee el
dia equivocado — silencioso y sistematico.

### Splits: se REGISTRAN, no se corrigen (decision 2026-09-12)

Un split con una posicion abierta rompe la operacion: el precio post-split llega
contra el precio de entrada y la cantidad pre-split, el stop se dispara sobre un
derrumbe que no existio y la salida queda con una perdida ficticia (incidente
KLAC/CRWD, JOURNAL 2026-07-21: -12.709 USD y dos estrategias que cambiaban de
signo). A los bots de Alpaca les pasa lo mismo: guardan precio de entrada y stop
en su propia tabla.

**Decision**: FT y Alpaca son estrategias en paper para EVALUAR. Un split es un
suceso excepcional: la operacion **no se corrige**, el split **se registra** y los
analisis lo tienen en cuenta. El registro es la tabla `splits_aplicados` (LOCAL),
que escribe `scripts/manual/splits.py corregir` al corregir `precios_diarios`.

- La correccion del 21/7 (`fix_ft_ops_split.py`) queda como estaba: esas 8
  operaciones ya llevan `_SPLIT_FIX` en `motivo_salida`.
- `ft_posiciones_diarias` conserva 39 filas contaminadas de KLAC/CRWD (retorno
  -70%..-89%, ATR x9) que NO se corrigen.

**Uso en un analisis**: una operacion quedo expuesta si entro con un dato previo a
la frontera y seguia abierta cuando el sistema ya veia la escala nueva. La
frontera es `fecha_corte_db` (la primera rueda que precios_diarios tuvo en la
escala nueva, que depende de cuando se bajo la data) y NO `execution_date`: KLAC
ejecuto el 12/6, pero la rueda del 11/6 ya llego ajustada y la op 820 cerro con
ella. Sin correccion registrada, la frontera es la ejecucion.

```sql
SELECT o.id, o.estrategia_id, o.ticker, o.motivo_salida,
       s.execution_date AS split_fecha, s.ratio AS split_ratio
FROM ft_operaciones o
JOIN splits_aplicados s
  ON s.ticker = o.ticker
 AND COALESCE(o.fecha_datos, o.fecha_entrada)
       < COALESCE(s.fecha_corte_db, s.execution_date)
 AND (o.fecha_salida IS NULL
      OR COALESCE(o.fecha_datos_salida, o.fecha_salida)
           >= COALESCE(s.fecha_corte_db, s.execution_date))
```

Para `ft_posiciones_diarias` y las tablas de los bots (`operaciones_bot*`, en
Railway), el mismo criterio con sus columnas de entrada y salida.

---

## 12. Medir un cambio: `ft_cambios`, tramos y grupo de control (2026-09-13)

La pregunta es "este cambio, ¿mejoro o empeoro la estrategia?". Mirar la ventana
posterior contra la anterior tiene tres trampas, y las tres ya aparecieron en
este proyecto:

1. **La fecha.** El commit no es el dia en que el cambio entra en las decisiones.
   El sistema es asincronico y la rutina es manual: el fix del score se commiteo
   a las 11.08 del 30/5 y la corrida de las 11.25 decidio con el dato del 29/5.
2. **El mercado.** Antes y despues de un corte son regimenes distintos. En el fix
   del 29/5, el universo subio +6,7% en las 24 ruedas previas y +0,7% en las 72
   posteriores: cualquier estrategia "empeora".
3. **La caja.** Con 55-80% de exposicion, contra un indice al 100% cualquier caida
   parece una mejora (seccion 7 y JOURNAL 2026-07-21 RESULTADO).

### 12.1 Registro `ft_cambios` (LOCAL)

Una fila por cambio: `clave` (UNIQUE), `fecha_efectiva`, `tipo` (BUG_FIX,
PARAMETRO, MODELO, DATOS, MEDICION, REFACTOR, INFRA), `estrategias` (INTEGER[]),
`cambia_decisiones`, `titulo`, `detalle`, `ref`.

- **`fecha_efectiva`** = la primera rueda de DATOS con la que la estrategia decidio
  ya con el cambio (el `fecha_datos` de la primera corrida que lo tuvo).
- **`cambia_decisiones`** = TRUE solo si cambia la LOGICA, los PARAMETROS o el
  MODELO. Esos cortan tramos. Una correccion de datos puntual va en FALSE aunque
  mueva alguna decision: queda como marca visible en el reporte, sin partir la
  historia en pedazos que nunca llegan a la muestra minima.
- Alta: `scripts/forward_testing/ft_cambios.py add` (valida dia habil, estrategias
  y clave; avisa si ya hubo corridas con ese dato y si el cambio se queda sin
  grupo de control). Consulta: `ft_cambios.py list`.
- Carga inicial: `scripts/oneshot/create_ft_cambios.py`, 10 cambios. Cada fecha se
  fecho con **evidencia**: la hora del commit, del archivo o de las filas escritas
  contra `ft_operaciones.creado_en` de cada corrida. Ejemplos: el relleno del 28/8
  se inserto a las 21.47 del 12/9, despues de la corrida de las 19.22, asi que rige
  desde la rueda del 14/9; el put wall de OIEXIT se verifico en las operaciones (la
  corrida del 10/9 abrio 5 con SL por ATR, la del 12/9 ya abrio 2 por put wall).
  Solo uno corta tramos: el fix del score (rueda 29/5, estrategias 4/6/8/9).

### 12.2 Tramos: la convencion de fechas

Un tramo que va de `desde` a `hasta` contiene:

| | Regla | Por que |
|---|---|---|
| Retornos diarios | `desde < fecha <= hasta` | la decision tomada con el dato de `desde` se ejecuta al cierre de ese dia: su primer retorno es el del dia siguiente |
| Operaciones | entrada Y salida con `desde <= fecha_datos < hasta` | una decision con el dato del corte ya es regla nueva |

- Una operacion que abrio antes del corte y cerro despues vivio bajo las dos reglas:
  **no cuenta de ningun lado** (el reporte informa cuantas cruzan).
- Las `_SPLIT_FIX` quedan afuera: su salida es contrafactual.
- Las abiertas no tienen resultado y no cuentan.

### 12.3 Las comparaciones

Por cada estrategia afectada, el tramo de antes va desde su corte previo (o su
inicio) hasta el cambio, y el de despues desde el cambio hasta su corte siguiente
(o hoy). Modulo PURO: `src/utils/ft_tramos.py`.

| Comparacion | Que es | Rol |
|---|---|---|
| **vs control** | diferencia diaria entre la estrategia y la cartera de control (promedio diario de las NO afectadas), antes contra despues. En puntos por mes (x21) | **PRINCIPAL** |
| vs universo | lo mismo contra el universo equiponderado | referencia: incluye el efecto de la caja |
| **expectancy vs control** | pnl_pct medio por operacion, diferencia-en-diferencias contra las operaciones del control en los mismos tramos | la de operaciones |
| expectancy propia | la misma sin control | descriptiva, en gris |

- **Grupo de control**: las estrategias no afectadas **sin un corte propio dentro de
  la ventana** (una que cambio sus reglas ya no mide solo el mercado). Si un cambio
  afecta a todas, no hay control y la comparacion principal queda vacia.
- **IC95** con t de Student y grados de libertad de Welch-Satterthwaite (dos grupos
  en vs control; cuatro en la expectancy).
- **Minimos por lado**: 20 ruedas y 10 operaciones, y en la expectancy tambien las
  del control. Por debajo el veredicto es **INSUFICIENTE y no se muestra el
  numero**. Con 14 ruedas el IC de la diferencia diaria mide decenas de puntos por
  mes: el numero existiria y no diria nada.
- **Veredicto**: MEJORA / EMPEORA solo si el IC95 excluye el cero; si no, NO
  CONCLUYENTE.

**Por que la expectancy necesita control** (el caso que lo motivo): sin control,
TECH_SECTOR_v1 daba EMPEORA por operacion en el fix del 29/5, -1,17 pp con IC
[-1,95; -0,39]. Las operaciones del control cayeron igual en los mismos tramos
(COMBO_v1 +1,41% -> -0,59% por operacion, TECH_v1 +1,74% -> -0,55%). Descontado
eso: +0,50 pp [-1,37; +2,38], NO CONCLUYENTE. Se descarto antes que fuera el churn
del bug: sin las 93 operaciones de duracion cero el crudo sigue en EMPEORA.

**Limites que hay que tener presentes al leer**:
- El control no es un experimento aleatorizado: son otras estrategias, con otra
  logica. Descuenta mercado y estructura, no todo.
- Los retornos diarios se tratan como independientes (autocorrelacion chica en
  carteras diarias; el IC queda algo angosto si no lo fuera).
- Con 4 estrategias x 3 comparaciones, alguna puede salir concluyente por azar (5%
  cada una). Se decide con la comparacion principal, no con la que mas convenga.
- `VENTANA_COMPARABLE` del reporte es `2026-05-30` (`fecha >= 30/5`, primer punto
  el 1/6) y la fecha efectiva del fix es el 29/5: la ventana comparable deja afuera
  el retorno del 1/6, que ya es post-fix. Un dia; no se cambio para no mover los
  numeros ya publicados en el JOURNAL.

### 12.4 Foto de base

`scripts/forward_testing/ft_foto_base.py` -> `reportes/ft_foto_base_<rueda>.{json,md}`.
Por estrategia y en las dos ventanas: retorno, benchmark, max DD, **Sortino con IC95
por bootstrap** (`ft_tramos.ic95_bootstrap`, remuestreo iid, 2000 remuestras,
semilla fija), Sharpe con IC95 de Lo, metricas por operacion, tramo vigente y los
cambios registrados. Nunca pisa una foto existente salvo `--forzar`.

Existe porque el reporte se regenera y la historia se RECOMPUTA: corregido un dato,
la version anterior deja de existir. Primera foto: rueda **2026-09-11**, antes de
la Fase 5. OJO: `reportes/` esta gitignoreado; la respalda OneDrive.

### 12.5 Primer resultado: fix del score=0.0 (rueda 29/5)

| Estrategia | Ruedas antes / despues | vs control (pp/mes) | Expectancy vs control (pp) |
|---|---|---|---|
| TECH_SECTOR_v1 | 24 / 72 | -0,1 [-12,6; +12,3] NO CONCL. | +0,50 [-1,37; +2,38] NO CONCL. |
| TECH_SECTOR_v2 | 18 / 72 | INSUFICIENTE | +1,77 [-0,63; +4,18] NO CONCL. |
| TECH_SECTOR_OPTIONS_v1 | 14 / 72 | INSUFICIENTE | +3,09 [-3,77; +9,95] NO CONCL. |
| TECH_SECTOR_OPTIONS_v2 | 14 / 72 | INSUFICIENTE | -2,84 [-9,29; +3,61] NO CONCL. |

Control: ML_SCANNER_v1, TECH_v1, SMC_v1, COMBO_v1, SMC_v2, OIEXIT_v1. Tres de las
cuatro nacieron menos de 20 ruedas antes del corte: su comparacion diaria **nunca**
va a ser posible. El fix se justifica por correccion (el score era 0 por un bug),
no por resultado.

### 12.6 Procedimiento para un cambio nuevo

1. Antes de desplegar, decidir que estrategias toca y si cambia decisiones.
2. Fecha efectiva = la rueda con la que va a decidir la proxima corrida (la ultima
   rueda cargada cuando corra `ft_run_diario.bat`).
3. `ft_cambios.py add ... --dry-run`, revisar los avisos, y sin `--dry-run`.
4. Entrada en el JOURNAL con la linea `**Registro**: ft_cambios <clave>`.
5. Una estrategia NUEVA en paralelo (ej. ML_SCANNER_v2) no es un corte de la vieja:
   se comparan en el mismo periodo. Si el despliegue toca algo que la vieja usa (el
   scanner), se registra como marca sobre la vieja.

---

## 13. Por que difieren ML_SCANNER_v1 y v2 (Etapa 3f, 2026-09-14)

La cartera y la seccion 12 responden CUAL rinde mas. Con dos estrategias que tienen
las mismas reglas y distinto modelo hace falta ademas saber POR QUE eligen distinto:
sin eso, una diferencia de rendimiento no dice si vino del modelo, de la cobertura de
tickers o del tope de posiciones.

Modulo PURO `src/utils/ft_comparar.py`. Reporte `scripts/forward_testing/ft_comparar_ml.py`
(`reportes/ft_comparar_ml.md`, `--desde`) y seccion "ML v1 vs v2: por que difieren" del
reporte HTML diario, las dos con las mismas tablas (`ft_comparar.tablas`).

### 13.1 Niveles

| Nivel | Fuente | Que responde |
|---|---|---|
| Senales | `alertas_scanner`, las MISMAS filas (columnas v1 y `_v2`) | que da cada modelo y cuanto rindio despues, aunque nadie lo haya operado |
| Atribucion | idem + historia de precios | por que una senal es de una sola |
| Operaciones | `ft_operaciones` (fechas de dato) | que operaron, compartido o exclusivo, y como salieron |
| Oportunidades | `ft_candidatos_diarios` | que dejo afuera el tope de 5 posiciones |
| Cartera | `ft_equity_diaria` | diferencia diaria v2 - v1, pareada |

**Senal** = `COMPRA_FUERTE` con score >= 65, la regla de entrada de las dos. Por rueda:
ambas / solo v1 / solo v2, y Jaccard = ambas / union.

### 13.2 Retorno de una senal

- Rueda de datos = `precio_fecha`, nunca `scan_fecha`. Si hubo dos corridas sobre la
  misma rueda, cuenta la ultima.
- Retorno a N ruedas (5 y 20) = close(D+N) / close(D) - 1, con D+N contado sobre las
  ruedas del MERCADO (dias habiles NYSE con precio), no sobre la serie del ticker, y con
  el close de ese dia exacto. Un hueco en la serie del ticker deja la senal sin retorno
  en vez de correr la ventana.
- **Exceso** = retorno menos el promedio de TODAS las filas de esa rueda (el universo,
  tengan o no la v2) en la misma ventana. Es la medida principal: descuenta el mercado,
  que las senales de una misma rueda comparten.
- Sale de `precios_diarios` y no de `alertas_scanner.retorno_Nd_real`: esas columnas no
  se llenan desde mayo (el Paso 4 no esta en la rutina).
- **Splits**: se lee `precios_diarios` tal como esta. Un split corregido deja la serie
  continua (`splits.py corregir` divide la historia), asi que una ventana que lo cruza es
  valida: el +12,9% de KLAC del 11/6 es movimiento real. Uno SIN corregir parte la serie y
  contamina sus ventanas hasta que se corrige; como todo se recalcula en cada corrida,
  corregirlo arregla la historia sola. El diseno inicial descartaba las ventanas que cruzan
  `splits_aplicados.fecha_corte_db`: se saco al verificar que esas ventanas son continuas y
  que un split sin corregir no figura en ese registro, o sea que la regla no protegia nada
  y tiraba datos buenos.

### 13.3 La comparacion que decide

Las senales compartidas son identicas en las dos: la diferencia de eleccion vive entera
en las exclusivas. La comparacion es el **exceso medio de las exclusivas de la v2 contra
el de las exclusivas de la v1**, con IC95 de Welch (`ft_tramos.comparar_medias`). A FAVOR
DE V2 / A FAVOR DE V1 solo si el intervalo excluye el cero; si no, NO CONCLUYENTE.

**Minimos**: 10 senales **en al menos 5 ruedas distintas** por lado. El minimo de ruedas
no esta en la seccion 12 y es a proposito: el 11/9 la v2 dio 14 senales exclusivas en UNA
rueda. Con solo el minimo de 10 se publicaria un IC como si fueran 14 observaciones
independientes, y son una sola semana de mercado.

### 13.4 Atribucion

Las dos versiones comparten price action, score tecnico y bajistas: la diferencia de score
ES la diferencia de puntos ML. Por grupo:

- **Sin entrenar en la v1**: tickers que no estaban en el entrenamiento de la v1, leidos
  contra la tasa base de las filas (~38%). El conjunto se reproduce por la historia de
  precios (hasta fin de 2021 = los 123 de `features_ml` de la v1): la tabla se reconstruyo
  en la Etapa 3b con 196, y `activos.modelo_asignado` da 125 porque HOOD y LAC tienen modelo
  asignado sin esa historia.
- **Nivel en la otra version**: COMPRA (60-74) es un desacuerdo de borde; NEUTRAL o menos,
  de fondo.
- Probabilidad media de cada modelo, diferencia de score y sectores.

### 13.5 Operaciones, oportunidades y cartera

- **Compartida** = la otra version tuvo el mismo ticker abierto algun dia en comun (las
  abiertas se extienden hasta la ultima rueda). Expectancy, win rate, permanencia en ruedas
  y motivos de salida (`SCORE_DEGRADADO_*` agrupado); `_SPLIT_FIX` afuera. Expectancy
  v2 - v1 con IC95, minimo 10 cerradas por lado.
- Solo cuentan las operaciones con entrada desde la rueda de inicio: las posiciones que la
  v1 ya tenia abiertas quedan afuera de este nivel, aunque si estan en su equity.
- **Oportunidades**: `ft_candidatos_diarios` guarda el dia de REGISTRO; la rueda de datos es
  la de la ultima alerta del ticker hasta ese dia. Sus `retorno_Nd` propios no se usan:
  cuentan dias desde el registro y solo se llenan si el bot corre justo N ruedas despues.
  Exceso de los que entraron contra los que quedaron afuera: si afuera rinde mas, el orden
  por score no elige bien dentro del dia, y con mas senales por dia (la v2) el tope pesa mas.
- **Cartera**: diferencia diaria de retorno v2 - v1 en los dias comunes, en pp/mes con IC95
  de muestra pareada (el mismo mercado cada dia). Minimo 20 ruedas.

### 13.6 Validacion (14/9/2026)

- Retornos contra un calculo independiente en SQL (ruedas por ROW_NUMBER sobre fechas
  distintas): 120 ventanas a 5 y 20 ruedas, diferencia maxima 2e-16.
- Camino completo sobre junio-septiembre con una v2 FALSA en memoria, porque la real
  todavia no tiene filas: las 6 tablas en ASCII y 212 candidatos con su rueda resuelta.
- La v2 real escribe filas desde la rueda del 14/9: todo arranca INSUFICIENTE.

### 13.7 Limites

- El IC trata las senales como independientes; varias del mismo sector y semana lo dejan
  angosto de mas. El exceso y el minimo de ruedas lo atenuan, no lo eliminan.
- El retorno a 20 ruedas tarda 20 ruedas en existir: esa comparacion va un mes atras de
  la de 5.
- Senales y operaciones miden cosas distintas: una senal exclusiva buena que el tope dejo
  afuera no suma a la cartera. La decision de reemplazo es de la Etapa 4, con la cartera.
