# Journal de Decisiones — Forward Testing

**Formato**: fecha | categoria | descripcion | hipotesis | resultado esperado
**Regla**: una entrada por decision relevante. No documentar cambios de formato
o correcciones de bugs obvios — solo decisiones de diseno, parametros y estrategia.

---

## 2026-04

### 2026-04-23 — DISENO
**Creacion del sistema Forward Testing**
Decidimos construir un motor propio de forward testing sobre la DB existente,
en lugar de usar un backtester externo (Backtrader, Zipline, etc.).
**Razon**: necesitamos evaluar estrategias en tiempo real con la misma data que
usa el scanner, no con data historica limpia. El FT es la extension natural del pipeline.
**Estructura definida**: ft_estrategias, ft_operaciones, ft_candidatos_diarios, ft_metricas_diarias.
**Precio de ejecucion**: cierre del dia (consciente y aceptado — sin acceso intraday).
**Ref**: docs/estrategias_ft.md

---

### 2026-04-28 — LANZAMIENTO
**Primera corrida de los 5 bots en produccion**
Estrategias activas desde hoy:
- ML_SCANNER_v1 (id=1): benchmark Bot1 Alpaca
- TECH_v1 (id=2): benchmark Bot2 Alpaca
- SMC_v1 (id=3): benchmark Bot3 Alpaca
- COMBO_v1 (id=5): primera estrategia nueva — sectorial + candle score

Capital inicial: $100,000 cada una.
TECH_SECTOR_v1 (id=4) arranca unos dias despues por ajustes de bugs.

---

## 2026-05

### 2026-05-02 — OBSERVACION
**TECH_SECTOR_v1: cierra y reabre las mismas posiciones**
Observamos que el bot cierra todas las posiciones con motivo SCORE_DEGRADADO_0.0
y las reabre inmediatamente. Esto indica que el scoring es demasiado binario:
cuando precio < SMA200 un solo dia, score = 0 y el bot cierra todo.
**Efecto**: capital girado innecesariamente, costos simulados sin beneficio real.
**Hipotesis**: una posicion con candle_score_5d positivo y lateral_ratio < 0.5
esta en acumulacion — no deberia cerrarse solo por un dia con score = 0.
**Pendiente**: disenar logica de retencion para TECH_SECTOR_v2.

---

### 2026-05-02 — BUG FIX
**Bug: dias_abierta enviado como lista en lugar de entero**
`trading_days_between()` retorna una lista de fechas, no un entero.
Se estaba pasando la lista directamente a la columna smallint de la DB.
**Fix**: usar `len(trading_days_between(entrada, fecha))`.
**Commit**: bc4a6b3

---

### 2026-05-02 — BUG FIX
**Bug: columnas es_alcista/vol_price_confirm en tabla incorrecta**
La query buscaba estas columnas en `features_market_structure`, pero
estan en `features_precio_accion`. Son features de precio-accion, no de estructura.
**Fix**: cambiar tabla en la query batch de registrar_estado_posiciones().
**Commit**: fda1328

---

### 2026-05-05 — DISENO
**Creacion de la capa de observacion diaria**
Agregamos `ft_posiciones_diarias` para trackear el estado de cada posicion
dia a dia: tech_score, candle_score, lateral_ratio, rango_5d, up_vol_5d.
**Objetivo**: tener datos suficientes para analizar patrones antes de modificar
las estrategias. No queremos cambiar parametros sin entender primero como se
comportan las posiciones actuales.
**Ref**: scripts/migrations/add_ft_posiciones_diarias.py

---

### 2026-05-05 — DISENO
**Creacion de retornos contrafactuales en ft_candidatos_diarios**
Para las oportunidades que NO se abrieron, calculamos el retorno que hubieran
tenido a 5, 10 y 20 dias habiles.
**Objetivo**: responder si estamos dejando dinero sobre la mesa al no abrir
ciertas posiciones (por slots llenos o capital insuficiente).

---

### 2026-05-05 — DECISION
**Denominador de retorno: capital total, no capital invertido**
El capital no desplegado en una estrategia esta inmovilizado — no esta disponible
para otras estrategias. Por lo tanto el denominador del retorno debe ser
el capital total asignado ($100,000), no el capital invertido.
Esto penaliza correctamente las estrategias que no despliegan bien su capital.

---

### 2026-05-05 — EXPLORACION INICIADA
**Hipotesis: incorporar Momentum/Lateral/Retorno como features de salida/rotacion**
Las features de la capa de observacion (candle_score_5d, lateral_ratio, up_vol_5d)
podrian mejorar las decisiones de salida y rotacion en:
- TECH_SECTOR_v1: evitar cierres innecesarios de posiciones en acumulacion
- SMC_v1: filtrar entradas en contextos laterales, anticipar salidas estancadas
**Estado**: en diseno. Ver TECH_SECTOR_v2.md y SMC_v2.md cuando esten disponibles.
**Proximos pasos**: definir parametros especificos antes de implementar.

---

### 2026-05-17 — LANZAMIENTO
**Estrategias #8 y #9 — TECH_SECTOR_OPTIONS v1 y v2**
- TECH_SECTOR_OPTIONS_v1 (id=8): sectorial + confirmacion por PCR_OI de opciones.
- TECH_SECTOR_OPTIONS_v2 (id=9): identica pero usa PCR_VOL (volumen diario) en
  vez de PCR_OI (open interest acumulado).
Gate de entrada doble: tech_score >= 4.0 AND pcr_score >= 2/3 ventanas alcistas.
**Razon / Hipotesis**: el posicionamiento en opciones es una fuente de informacion
independiente de la senal tecnica; su convergencia deberia reducir entradas en
falsos breakouts. Correr v1 y v2 en paralelo aisla la variable OI (lento,
acumulado) vs VOL (reactivo, diario) para comparar cual confirma mejor.
**Resultado real**: (pendiente — primera corrida real via ft_run_diario.bat)
**Ref**: estrategias/TECH_SECTOR_OPTIONS_v1.md y v2.md

---

### 2026-05-18 — DECISION
**Forward Testing migra a DB local (Plan C)**
Los bots FT escribian en Railway sin querer: cargaban .env.local con override,
que setea DATABASE_URL=Railway. Plan C define que FT corre 100% en local.
**Que se hizo**: helper ft_env.py fuerza get_engine() a local; los 9 bots y
ft_setup lo usan en lugar de load_dotenv(.env.local). Migracion puntual de las
5 tablas ft_* Railway->local (migrate_ft_railway_to_local.py, schema real +
datos + secuencias). El sync ya no baja forward_testing — local es fuente de
verdad. Railway nunca mas recibe escrituras de FT.
**Efecto esperado**: FT autonomo y local, sin costo de procesamiento en Railway.
**Ref**: commit 8652db5

---

### 2026-05-18 — DISENO
**earnings_calendar — cache de fechas de earnings**
earnings_filter.py consultaba yfinance por cada ticker en cada corrida de cada
bot (~1.500+ llamadas/dia) -> rate limit por IP. Se crea la tabla
earnings_calendar (Railway + local), refrescada semanalmente; earnings_filter
pasa a leerla via get_engine() y ya no llama a yfinance.
**Fuentes evaluadas**: yfinance (throttle por IP a mitad de corrida), FMP (el
plan free solo cubre 54/199 tickers), Nasdaq earnings calendar (cobertura
completa, gratis, indexado por fecha) -> se elige Nasdaq.
**Fail-safe**: ticker sin fecha conocida -> earnings_date NULL -> sin filtro de
earnings, los bots corren igual sin cortarse.
**Efecto esperado**: cero rate limit en los bots; filtro de earnings preciso
sobre fechas confirmadas.
**Ref**: scripts/refresh_earnings_calendar.py | cron Oracle lunes 12:00 UTC

---

### 2026-05-18 — DISENO
**Estrategia #10 — TECH_SECTOR_OIEXIT_v1 (entrada v1, salida nueva)**
Experimento de variable unica: entrada **identica** a TECH_SECTOR_v1 (puro
tecnico + sectorial, sin opciones), cambiando **solo la salida**.
**Salida nueva**: SL inicial anclado al put wall de OI (fallback 2x ATR);
proteccion por escalones de R (BE en +1R, +1R en +2R); al tocar +3R se libera
el techo y la posicion pasa a "corrida" — backstop Chandelier 2.5x ATR +
quorum 3 de 4 (cierre<SMA21 / candle_5d<=-2 / PCR_VOL mayoria bajista /
divergencia de volumen). Earnings transversal.
**Razon / Hipotesis**: un TP fijo corta a los ganadores cuando mas corren. Si
la perdida ya esta acotada por el SL que sube, el valor esta en dejar correr
las tendencias y salir por senal de agotamiento, no por un numero fijo.
**Hallazgos del analisis de OI (NVDA + JPM/KO/CAT/XOM/LLY/MSFT)**:
- El OI lejano son coberturas de cola (inutiles como S/R) -> buscar wall solo
  en zona +-10% del precio.
- El soporte (put wall) es nitido; la resistencia (call wall) es ruidosa y
  suele estar pegada -> SL por wall, TP por R, no forzar ambos por OI.
- La liquidez de opciones NO sigue al tamano de la empresa (CAT/LLY caros pero
  ralos; KO barato pero liquido) -> validez del wall RELATIVA al ticker.
- Walls y ATR suelen coincidir; el valor esta donde divergen (a medir).
**Efecto esperado**: capturar mas de las corridas fuertes que el TP 4x ATR de
v1 cortaba, sin perder el control de la perdida.
**Ref**: estrategias/TECH_SECTOR_OIEXIT_v1.md

---

### 2026-05-30 — BUG FIX
**Score tecnico = 0.0 en evaluacion de salida de 4 bots sectoriales**
La query `obtener_estado_tecnico_tickers()` de TECH_SECTOR_v1, TECH_SECTOR_v2,
TECH_SECTOR_OPTIONS_v1 y TECH_SECTOR_OPTIONS_v2 no devolvia `close` (faltaba
el JOIN con precios_diarios) y, en el caso de v1, devolvia `dist_sma*` en
lugar de `sma*` absolutas. `calcular_score_tecnico` evaluaba Capa 1 con
`close=0 > sma200=0` -> False -> score = 0.0 todos los dias.

**Manifestacion (por que solo se vio en v1)**:
- TECH_SECTOR_v1: cerraba TODAS sus posiciones diariamente con motivo
  `SCORE_DEGRADADO_0.0`. Sin capa intermedia que enmascarara el bug.
- TECH_SECTOR_v2: el `score=0` constante disparaba siempre la capa de
  retencion por candle_5d/up_vol_5d -> cierres etiquetados
  `SCORE_DEGRADADO_SIN_MOMENTUM` que parecian motivados pero no lo estaban.
- OPTIONS_v1/v2: el `score=0` constante disparaba siempre el filtro PCR ->
  retenciones (`RETENCION_OPCIONES`) o cierres (`SCORE_DEGRADADO_OPCIONES`)
  cuya decision era 100% PCR, sin contribucion real del score tecnico.

**Por que COMBO_v1, TECH_v1 y los SMC no estaban afectados**: TECH_v1 reusa
el `indicadores_map` de la entrada (que SI tiene close+SMAs). COMBO_v1 tiene
la query de exit con JOIN a precios_diarios. ML/SMC usan otro sistema de
scoring.

**Fix**: reemplazar la query de exit en los 4 bots con la misma forma que
usa la query de entrada y COMBO_v1 — JOIN precios_diarios para `close` +
SMAs absolutas (no `dist_sma*`).

**Verificado en dry-run (2026-05-30)**:
- v1: "Sin posiciones a cerrar" (vs 20+ cierres `SCORE_DEGRADADO_0.0` antes).
- v2: 1 cierre real (STOP_LOSS_ATR) vs 12 `SIN_MOMENTUM` antes.
- OPTIONS_v1: 1 TAKE_PROFIT + 8 retenciones con scores reales 0.0-3.5
  (los tickers SI tienen score degradado, no porque close=0).
- OPTIONS_v2: 0 cierres + 5 retenciones con scores reales.

**Implicaciones**: las curvas historicas de las 4 estrategias estan
sesgadas por este bug — las decisiones de salida se tomaron con `score=0`
todos los dias durante toda la historia del FT. Desde el fix los
resultados reflejan la logica documentada.

**Commit**: en esta sesion (4 archivos: ft_bot_tech_sectorial.py,
ft_bot_tech_sectorial_v2.py, ft_bot_tech_sectorial_options_v1.py,
ft_bot_tech_sectorial_options_v2.py).

---

### 2026-05-30 — DECISION
**Split de validez 30/5/2026 — NO resetear estrategias tras el bug fix**

Tras el fix del score=0 en la query de salida de v1/v2/OPTIONS v1/v2, surge
la pregunta: ¿reseteamos esas 4 a $100k y empezamos de cero para no
arrastrar curvas contaminadas?

**Decision: opcion A (seguir sin reset).** Ninguna estrategia se resetea.

**Razones**:
1. **Las 6 estrategias no afectadas** (ML_SCANNER, TECH_v1, SMC_v1,
   COMBO_v1, SMC_v2, OIEXIT_v1) tienen ~5 semanas de historia valida desde
   sus respectivas fechas de inicio. Resetearlas seria perder signal real
   sin beneficio alguno.
2. **Las posiciones abiertas en las 4 afectadas son testbed gratuito**
   para la logica de salida corregida. Cerrarlas hoy al precio actual
   pierde ese experimento natural en curso.
3. **El "dia cero" sintetico** (resetear todo hoy) no es mas limpio en
   sentido fuerte: serian aperturas a precios arbitrarios de hoy, no
   necesariamente representativos.

**ALERTA para analisis cuantitativo de rendimientos**:

Para las 4 estrategias afectadas (id 4, 6, 8, 9), los datos historicos
**antes del 2026-05-30** estan sesgados por el bug. Concretamente:

| Estrategia (id) | Sesgo dominante en datos pre-30/5/2026 |
|---|---|
| TECH_SECTOR_v1 (4) | Cerraba **TODAS** las posiciones diariamente con `SCORE_DEGRADADO_0.0`. Churn artificial constante. |
| TECH_SECTOR_v2 (6) | El score=0 ficticio disparaba siempre la capa de retencion -> cierres `SCORE_DEGRADADO_SIN_MOMENTUM` eran artefactos del bug, no del agotamiento real de momentum. |
| TECH_SECTOR_OPTIONS_v1 (8) | El score=0 ficticio dominaba el filtro PCR -> retenciones (`RETENCION_OPCIONES`) y cierres (`SCORE_DEGRADADO_OPCIONES`) decididos 100% por PCR, sin contribucion real del score tecnico. |
| TECH_SECTOR_OPTIONS_v2 (9) | Idem v1. |

**Para evaluar rendimiento "real" de estas 4 a futuro**:
- Filtrar `ft_operaciones` por `estrategia_id IN (4,6,8,9) AND fecha_salida >= '2026-05-30'`.
- O equivalente: filtrar `ft_metricas_diarias` por `estrategia_id IN (4,6,8,9) AND fecha >= '2026-05-30'`.
- Las curvas de equity de esas 4 antes del 30/5 son artefactos de bug, no comportamiento real de la estrategia documentada.

**Para las 6 estrategias no afectadas (id 1, 2, 3, 5, 7, 10)**: historia
valida desde su `fecha_inicio` respectiva, sin filtro adicional.

**Evidencia del impacto del fix (corrida 30/5/2026 vs 29/5/2026)**:
- TECH_SECTOR_v1: 26 cierres por SCORE_DEGRADADO_0.0 -> 3 cierres por SCORE_DEGRADADO_3.0 (score real legitimo, -88%).
- TECH_SECTOR_v2: 14 cierres (10 SIN_MOMENTUM enmascarando bug) -> 3 cierres (2 rot + 1 SL, sin SIN_MOMENTUM).
- OPTIONS v1/v2: menos cambio absoluto porque sus motivos dominantes (SL/TP) eran correctos desde antes.

**Ref**: commit 4394f2d.

---

## 2026-07

### 2026-07-21 — DISENO
**Metricas de riesgo: la equity curve no era una equity curve**

Al evaluar la incorporacion del Ratio de Sharpe a las 10 estrategias, el
diagnostico sobre los datos existentes encontro tres problemas, el tercero
bloqueante.

**1. Contaminacion de calendario.** `ft_metricas_diarias` tenia 12 marks en dias
NO habiles (8 sabados, 2 domingos, Juneteenth 19/6 y el 3/7 observado) y le
faltaban 21 de 61 dias habiles (34%). Consecuencia de que la rutina nocturna es
MANUAL: el bot solo escribe la fila del dia en que corre. Serie de muestreo
irregular -> anualizar con raiz(252) es invalido.

**2. n insuficiente.** 28-39 dias habiles por estrategia. IC95% del Sharpe de
~+-5 puntos (Lo 2002). Nueve de diez estrategias tienen IC que incluye cero:
indistinguibles del azar. Para separar Sharpe 0.5 de 1.0 con precision +-0.5
harian falta ~15 anios. **No se resuelve esperando.**

**3. BLOQUEANTE — `capital_total` no esta marcado a mercado.**
`capital_total = cash + capital_inmovilizado`, y `capital_inmovilizado =
SUM(capital_entrada)` = **costo de entrada**. Verificado: en el **100% de los
dias sin cierres, en las 10 estrategias**, el capital no se mueve.
Lo que llamabamos equity curve es una **curva de PnL realizado**.

**Razon por la que importa**: toda metrica de riesgo sobre esa serie
**subestima el riesgo**, y siempre en la misma direccion.
- La "volatilidad" medida es la cadencia de salidas, no la fluctuacion real.
- El max drawdown es **invisible**: -15% contra posiciones abiertas se ve plano.
- Sesga el ranking a favor de estrategias lentas: menos salidas -> menos vol
  medida -> mas Sharpe. Parte de la brecha ML_SCANNER_v1 (+3.08) vs
  OIEXIT_v1 (-7.11) es artefacto de cadencia, no de riesgo.

**Decision**: la equity diaria pasa a ser una **capa DERIVADA** en tabla nueva
`ft_equity_diaria`, reconstruida desde `ft_operaciones` + `precios_diarios`
(mismo patron que `fundamentales_ratios_q`). Es funcion pura del estado de
operaciones y precios -> recomputable hacia atras, sin datos nuevos, sin
depender de haber corrido el bot ese dia. `ft_metricas_diarias` **no se toca**
(log operativo, la leen el reporte y el MCP).

**Decisiones asociadas**:
- **Sortino primario, Sharpe secundario**: las 10 son long-only con stop, o sea
  distribucion asimetrica por diseno. Sharpe castiga la volatilidad al alza, que
  es justo lo que la Fase 2 de OIEXIT_v1 busca capturar.
- **Todo Sharpe se reporta con n e IC95%**; si el IC incluye cero se marca NO
  CONCLUYENTE. Ningun cambio de estrategia se justifica con un ratio asi.
- **Benchmark por ventana propia de cada estrategia** (arrancan entre el 23/4 y
  el 27/5): universo equiponderado + ES=F. SPY no aplica (universo sin ETFs).
- **Costos de transaccion siguen en STANDBY**; cuando entren, lo hacen aguas
  arriba (PnL de `ft_operaciones`) y las metricas se recomputan solas.

**Hallazgo que no necesita ningun ratio**: sobre 2026-04-23 -> 2026-07-21, ES=F
+4.77% y el universo equiponderado +1.55%, con **7 de 10 estrategias en
negativo**. En un mercado que subio, la mayoria pierde contra comprar todo el
universo. Solo ML_SCANNER_v1 (+7.38%) le gana claramente al benchmark.

**Efecto esperado**: max drawdown real (hoy no existe), riesgo comparable entre
estrategias, y separacion entre habilidad de seleccion y beta de mercado.
**Resultado real**: (completar tras el backfill)
**Ref**: docs/forward_testing/METRICAS.md, rama `feature/ft-metricas-riesgo`

---

## Template de entrada

```
### YYYY-MM-DD — [DISENO | LANZAMIENTO | OBSERVACION | BUG FIX | DECISION | RESULTADO | EXPLORACION]
**Titulo descriptivo**
Descripcion de que ocurrio o que decidimos.
**Razon / Hipotesis**: por que tomamos esta decision.
**Efecto esperado**: que esperamos que cambie.
**Resultado real**: (completar cuando tengamos datos)
**Ref**: archivo o commit relacionado
```
