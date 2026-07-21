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

### 2026-07-21 — BUG FIX
**Splits no aplicados: 12.709 USD de perdidas FICTICIAS en 4 estrategias**

Al construir la equity a mercado, el control de cuadre expuso que
`precios_diarios` **no se re-ajusta hacia atras cuando un ticker hace split**.
El pipeline diario solo trae los dias nuevos (ya ajustados por Yahoo), asi que
la historia previa queda en la escala VIEJA y la serie del ticker queda partida
en dos.

**Dos splits sin aplicar**: KLAC 10:1 el 2026-06-11 (536 filas mal) y CRWD 4:1
el 2026-06-30 (548 filas mal).

**Impacto en FT**: ocho posiciones abiertas cruzaron el split y se valuaron y
cerraron al precio POST-split con la cantidad PRE-split -> perdidas de -72% a
-88% que nunca ocurrieron. Peor: dispararon salidas reales
(`STOP_LOSS_ATR`, `SL_PROTECCION`) por un derrumbe inexistente.

| Estrategia | Antes | Despues | Δ |
|---|---|---|---|
| TECH_SECTOR_v1 (4) | -2.12% | **+1.79%** | +3.91 |
| COMBO_v1 (5) | -2.45% | **+0.88%** | +3.33 |
| TECH_SECTOR_v2 (6) | -5.66% | -1.75% | +3.91 |
| TECH_SECTOR_OIEXIT_v1 (10) | -6.95% | -3.04% | +3.91 |

**Alcance mas alla del FT**: toda SMA/RSI/ATR que cruce la fecha del split
estaba corrupta para esos tickers, y con ella las features de ML.

**Correccion aplicada**:
1. `scripts/manual/splits.py` (nuevo) — deteccion en 2 etapas y correccion.
   Etapa 1: barrido local de variaciones diarias > umbral (barato, sirve de
   alerta temprana). Etapa 2: verificacion contra Yahoo SOLO de los candidatos.
   **La etapa 2 es imprescindible**: un movimiento real y un split se ven
   IDENTICOS en la etapa 1. Confirmo que CAR (-38% y -48% consecutivos el
   22-23/4) y FISV (-44% el 29/10/2025) son movimientos REALES, no splits.
2. Correccion por **divisor**, no re-descargando la serie ajustada.
   **Razon**: `precios_diarios` guarda el close CRUDO tal como lo devolvio
   Yahoo el dia que se bajo, y nunca se re-ajusta por dividendos hacia atras.
   El `Close` de yahooquery viene ajustado por split Y dividendos (KLAC da
   ratio 9.8369 en vez de 10 exacto; CRWD, que no paga dividendos, da 4.0000
   clavado). Sobrescribir habria dejado a esos 2 tickers en una base distinta
   a la de los otros 198 — cambiar un problema por otro mas sutil.
3. Recomputo de indicadores, features y z-scores. Backup en `data/backups/`.
4. `scripts/oneshot/fix_ft_ops_split.py` — corrige las 8 operaciones
   (`precio_entrada /= ratio`, `cantidad *= ratio`; `capital_entrada` no cambia
   porque `P*N == (P/r)*(N*r)`) y ajusta el cash de cada estrategia.

**Limitacion que NO se puede corregir**: las salidas son **contrafacticas**. Con
los precios correctos esos stops jamas se habrian disparado y las posiciones
habrian seguido corriendo. Se arreglo la contabilidad, no la historia. Por eso
las 8 operaciones llevan el sufijo **`_SPLIT_FIX`** en `motivo_salida`: sin la
marca contaminarian el analisis por motivo de salida como si fueran evidencia
sobre la calidad de esas reglas (un `SL_PROTECCION` que produce +19.9% es
imposible). Marcadas, quedan en su propio bucket y se ven.

**Deuda abierta**: el detector todavia es manual. Deberia correr en el pipeline
diario (etapa 1 sobre los ultimos dias + alerta Telegram) para que un split no
vuelva a pasar en silencio.

**Ref**: scripts/manual/splits.py, scripts/oneshot/fix_ft_ops_split.py

---

### 2026-07-21 — RESULTADO
**Primeras metricas de riesgo reales (equity a mercado, datos corregidos)**

`ft_equity_diaria` reconstruida para las 10 estrategias: 523 filas, solo dias
habiles, sin huecos, **cuadre exacto** del cash contra `ft_estrategias` en las
10 (tolerancia 0.05 USD).

| Estrategia | n | ret% | maxDD% | Sortino | Sharpe (IC95%) | IR | exp% |
|---|---|---|---|---|---|---|---|
| ML_SCANNER_v1 | 60 | +6.72 | 3.86 | +2.48 | +1.68 [-2.4,+5.7] | +0.97 | 63 |
| TECH_v1 | 60 | +6.44 | 6.30 | +1.94 | +1.31 [-2.7,+5.4] | +1.08 | 75 |
| SMC_v1 | 60 | +3.28 | 4.68 | +1.34 | +0.90 [-3.2,+5.0] | +0.12 | 64 |
| TECH_SECTOR_v1 | 58 | +2.45 | 3.47 | +0.71 | +0.44 [-3.7,+4.6] | +0.06 | 65 |
| COMBO_v1 | 57 | +1.85 | 3.50 | +0.56 | +0.39 [-3.8,+4.5] | -0.51 | 72 |
| TECH_SECTOR_OPTIONS_v2 | 43 | +1.97 | 4.38 | +0.97 | +0.70 [-4.1,+5.5] | +0.43 | 67 |
| TECH_SECTOR_v2 | 52 | -0.01 | 3.79 | -0.32 | -0.24 [-4.6,+4.1] | -0.94 | 77 |
| TECH_SECTOR_OIEXIT_v1 | 38 | -0.81 | 2.76 | -1.06 | -0.82 [-5.9,+4.3] | +0.54 | 80 |
| TECH_SECTOR_OPTIONS_v1 | 43 | -1.23 | 4.48 | -1.35 | -1.05 [-5.9,+3.8] | -1.40 | 72 |
| SMC_v2 | 52 | -2.78 | 6.46 | -1.71 | -1.38 [-5.7,+3.0] | -1.38 | 66 |

Benchmark equiponderado, misma ventana: **+2.74%, maxDD 5.79%, Sharpe +0.60**.

**Observaciones**:
1. **Los 10 IC95% incluyen cero.** Ningun Sharpe es concluyente, tal como se
   anticipo. La regla de reportar el intervalo hace su trabajo: sin el,
   ML_SCANNER_v1 con +1.68 se leeria como superioridad demostrada.
2. **El max drawdown, visible por primera vez, es el hallazgo positivo**: 8 de
   10 estrategias tienen maxDD MENOR al del benchmark (5.79%), varias por
   bastante (OIEXIT_v1 2.76%, TECH_SECTOR_v1 3.47%). Controlan la caida aunque
   no le ganen al indice en retorno. Esto era invisible con la curva a costo.
3. **Se dio vuelta la lectura anterior**: con los datos corruptos eran 7 de 10
   en negativo; corregido, 6 de 10 en positivo. La conclusion "la mayoria
   pierde contra comprar todo el universo" **era en buena parte el artefacto**.
4. **Las metricas de trade discriminan mejor que los ratios** a este n:
   TECH_SECTOR_v1 hace 584 operaciones con profit factor 1.10 y 36% de acierto
   (churn alto, margen finito); ML_SCANNER_v1 hace 112 con PF 1.41 y 53.6%.
   OIEXIT_v1 tiene PF 0.52 y expectancy -2.12% en 66 trades: eso **no** es
   artefacto de split, es la estrategia.

**Resultado real**: (seguir observando; ningun cambio de estrategia justificado
todavia por la regla de la seccion 8 de METRICAS.md)
**Ref**: scripts/forward_testing/ft_compute_equity.py, src/utils/ft_metricas.py

---

### 2026-07-21 — BUG FIX
**`fecha_entrada` no es la fecha del dato: el desfase asincronico NO es fijo**

Planteado por el usuario al revisar los calculos: los indicadores, osciladores
y estrategias se calculan con **el cierre del dia puntual del dato**, no del dia
en que se registra la operacion. Si eso no queda explicito, los reportes
informan mal.

Al medirlo aparecio algo peor que un desfase constante de un dia:

| desfase (dias habiles) | entradas | causa |
|---|---|---|
| 0 | 302 (16.7%) | la recovery ya habia corrido |
| 1 | 1.325 (73.2%) | el caso tipico |
| 2 | 112 (6.2%) | se salteo una noche |
| 5 | 62 (3.4%) | se salteo una semana |
| 6 | 10 (0.6%) | lanzamiento: el 23/4 se opero con datos del 15/4 |

El precio es "el ultimo cierre que habia en `precios_diarios` cuando corrio el
bot", y cuan viejo sea depende de cuando se corrio la recovery por ultima vez
(rutina manual).

**Por que importa (no es solo etiquetado)**:
1. La equity marcaba la posicion por primera vez el dia del REGISTRO. Con
   desfase 5, eso mete **cinco dias de movimiento como un salto de un solo
   dia** -> infla volatilidad y distorsiona drawdown. ~10% de las operaciones.
2. Cualquier analisis que cruce `fecha_entrada` con `indicadores_tecnicos` de
   esa misma fecha **lee el dia equivocado**. Silencioso y sistematico.

**Nota**: una version previa de METRICAS.md afirmaba que la serie de retornos
era identica bajo cualquier convencion de fechado. Eso vale SOLO para desfase=1;
quedo corregido en el documento.

**Solucion**: columnas `fecha_datos` / `fecha_datos_salida` en `ft_operaciones`.
- Hacia adelante las escribe `ft_utils.obtener_fecha_datos()`: el precio que
  usan los bots ES el ultimo close, asi que la fecha es `MAX(fecha)` del ticker.
  **Los 10 bots no necesitaron cambios**: `ft_utils` es el unico lugar del
  proyecto que escribe `ft_operaciones`.
- La historia se backfilleo al 100%. El matching de precio contra el close falla
  en salidas por SL/TP (el precio es el nivel del stop, no un cierre); se
  resolvio con la observacion de que **todas las operaciones de una misma
  corrida comparten la misma fecha de dato** -> se agrupa por (estrategia,
  fecha de registro), se juntan los votos de las que si matchean y la moda del
  grupo se aplica a todo el grupo.

**Efecto medido** (volatilidad anualizada): TECH_SECTOR_v1 19.7% -> **16.7%**,
OPTIONS_v2 12.2% -> 11.0%, TECH_SECTOR_v2 13.2% -> 11.7%. La caida mayor es la
de TECH_SECTOR_v1, la de mayor churn (584 ops) y por lo tanto la mas expuesta
al artefacto — coherente.

Ademas las series arrancan antes (ML_SCANNER_v1 y TECH_v1 pasan de 60 a 66 dias)
porque la primera operacion es del 15/4, no del 23/4.

**Ref**: scripts/oneshot/add_fecha_datos_ft_operaciones.py, ft_utils.py

---

### 2026-07-21 — AUDITORIA
**Barrido sistematico de datos que corrompen decisiones. 3 hallazgos**

Despues del incidente de splits, se audito el resto de los insumos que usan las
estrategias buscando el mismo tipo de bug: dato malo -> decision mala, en
silencio.

**LIMPIO (verificado, sin accion)**:
- **Alineacion indicadores/precios**: las 4 tablas al mismo dia y **los 200
  tickers con desfase 0** entre su ultimo indicador y su ultimo precio. No
  existe el riesgo de decidir con indicadores de un dia y ejecutar con el
  precio de otro.
- **NULLs en indicadores**: cero en sma21/50/200, rsi14, atr14, macd sobre
  15.000 filas del periodo. Cero ATR=0 (que habria dado SL = precio).
- **Dias sin precio**: ninguno. El camino `if not precio: continue` de
  `evaluar_cierres` (que retendria una posicion sin evaluarla, en silencio)
  nunca se activo por falta de dato.
- **PnL extremos**: los |pnl%| > 30 son movimientos REALES (ARM +50%, MU +48%,
  RKLB +34%), no datos rotos.
- **Duracion 0** (entra y sale con el mismo dato): 108 ops, **104 anteriores al
  30/5 y todas con pnl exactamente 0** -> es el churn del bug de score=0.0 ya
  documentado, no un problema nuevo.

**HALLAZGO 1 — errata propia en el fix de splits**
`fix_ft_ops_split.py` corrigio `precio_entrada` y `cantidad` de las 8 ops pero
**olvido `stop_loss` y `take_profit`**, que quedaron en la escala vieja (ej.
entrada 201.14 con SL 1847.31, o sea 10x por encima del precio de entrada de
una posicion LONG). El PnL estaba bien (no depende del SL) pero el registro era
incoherente y rompia el calculo de expectancy en R. Reparado con
`fix_ft_ops_split_sltp.py` (idempotente); las distancias quedaron en 8-11%,
coherentes con stops por ATR. El script original quedo corregido para futuras
corridas.

**HALLAZGO 2 — el detector de splits era ciego a los splits INVERSOS**
`cmd_detectar` filtraba `chg < 0` porque un split forward divide el precio.
Pero un split **inverso multiplica** el precio y se veria como un salto hacia
arriba. Corregido: ahora se verifican tambien las subidas, priorizando las
caidas en el orden.

**HALLAZGO 3 (el mas grave) — el detector generaba FALSOS POSITIVOS destructivos**
Al ampliar el barrido, el detector declaro "SPLIT CONFIRMADO" para ORCL
(ratio 0.9841) y DELL (0.9744). **No existe un split de 0.98:1.** La logica
declaraba split con solo ver un ratio *constante*, sin exigir que fuera un ratio
*plausible*. Seguir esa recomendacion habria dividido datos sanos por 0.98,
corrompiendolos.

Corregido: `es_split = (not disperso) AND (ratio_limpio is not None)`. Los
ratios constantes pero no-plausibles se reportan aparte como
"DISCREPANCIA (no split) -- NO corregir con divisor".

> **Principio que deja el hallazgo 3**: este script recomienda una accion
> DESTRUCTIVA sobre la fuente de verdad. Un falso positivo cuesta mucho mas que
> un falso negativo, y el criterio de deteccion tiene que estar calibrado en esa
> direccion.

**Sobre la discrepancia ORCL/DELL**: es real pero **no afecta al FT**. ORCL
difiere hasta 2025-04-09 y DELL hasta 2025-07-21, ambas muy anteriores al inicio
del forward testing (2026-04-15). Queda como observacion para revisar en el
contexto de ML (features_ml cubre 2025-03 -> 2026-03 y si las incluye). Origen
probable: ajuste por dividendos o un backfill viejo con otra base.

**Pendiente relacionado (fuera del FT)**: los bots Alpaca tienen el mismo patron
de SL en valor absoluto y **nunca contrastan sus posiciones contra el broker**
(`alpaca_client.get_open_positions()` existe y no la llama nadie). Los splits de
KLAC les pegaron igual: -1877.44 en bot_candle y -2164.89 en bot_tech. Es paper,
prioridad baja, pero queda anotado.

**Ref**: scripts/oneshot/fix_ft_ops_split_sltp.py, scripts/manual/splits.py

---

### 2026-07-21 — RESULTADO
**Ventana comparable post-fix: las 10 estrategias le ganaron al mercado**

Con los datos ya corregidos, se midieron las 10 estrategias sobre la ventana en
que son comparables entre si: **desde el 2026-05-30**, fecha del fix del bug de
`SCORE_DEGRADADO_0.0` (antes de esa fecha, la historia de las estrategias 4, 6,
8 y 9 mide el bug, no la estrategia).

**El dato que da vuelta la lectura**: el benchmark cambia de signo entre ventanas.

| Ventana | Universo equiponderado |
|---|---|
| Completa (abril -> julio) | **+8.26%** |
| **Post-fix (30/5 -> 20/7)** | **-4.95%** |

Todo lo medido sobre la ventana completa estaba dominado por el rally de
abril-mayo. En la ventana limpia **el mercado cayo ~5%**.

**Resultados post-fix (n=34 dias habiles, ordenado por Sortino)**:

| Estrategia | Ret. | vs bench | Max DD | Sortino | IR | Expos. |
|---|---|---|---|---|---|---|
| ML_SCANNER_v1 | **+2.67%** | +7.6 | 3.86% | +1.51 | **+3.39** | 56% |
| SMC_v1 | **+1.30%** | +6.3 | 4.22% | +0.78 | +3.09 | 72% |
| TECH_SECTOR_OIEXIT_v1 | -1.17% | +3.8 | **2.76%** | -1.39 | +3.14 | 82% |
| SMC_v2 | -2.23% | +2.7 | 5.23% | -1.85 | +1.31 | 74% |
| TECH_SECTOR_OPTIONS_v2 | -2.07% | +2.9 | 4.33% | -1.99 | +2.59 | 71% |
| TECH_v1 | -2.99% | +2.0 | 5.31% | -2.04 | +1.10 | 75% |
| TECH_SECTOR_v2 | -2.09% | +2.9 | 3.79% | -2.16 | +2.41 | 77% |
| TECH_SECTOR_OPTIONS_v1 | -1.99% | +3.0 | 4.48% | -2.17 | +2.25 | 73% |
| COMBO_v1 | -2.28% | +2.7 | 3.76% | -2.34 | +1.55 | 73% |
| TECH_SECTOR_v1 | -2.34% | +2.6 | 3.73% | -2.44 | +1.57 | 65% |

**Los 10 Information Ratio son POSITIVOS.** Ocho de diez estan en rojo absoluto
pero todas cayeron menos que el indice, y dos subieron mientras el mercado
bajaba. Sin el benchmark al lado, la conclusion habria sido "se rompieron todas
despues del fix" — exactamente lo contrario de lo que paso.

**Por operacion, desde el 30/5** (ordenado por profit factor):

| Estrategia | n | win% | PF | Expectancy |
|---|---|---|---|---|
| SMC_v1 | 16 | 43.8 | **1.52** | +0.98% |
| ML_SCANNER_v1 | 73 | 52.0 | **1.15** | +0.19% |
| TECH_SECTOR_v2 | 174 | 39.7 | 0.84 | -0.32% |
| COMBO_v1 | 159 | 28.3 | 0.80 | -0.44% |
| TECH_SECTOR_OPTIONS_v2 | 73 | 38.4 | 0.79 | -1.01% |
| TECH_v1 | 22 | 31.8 | 0.75 | -0.81% |
| TECH_SECTOR_OPTIONS_v1 | 69 | 34.8 | 0.74 | -1.19% |
| TECH_SECTOR_v1 | 160 | 23.1 | 0.67 | -0.85% |
| SMC_v2 | 11 | 27.3 | 0.58 | -1.31% |
| TECH_SECTOR_OIEXIT_v1 | 65 | 27.7 | **0.52** | -2.11% |

**Observaciones**:
1. **ML_SCANNER_v1 y SMC_v1 son las unicas con profit factor > 1 en las DOS
   ventanas.** Es la senal mas robusta que hay hasta ahora.
2. **TECH_SECTOR_v1 se degrado fuerte**: profit factor de 1.10 a 0.67 y acierto
   de 36% a 23%. Opera 160 veces en 34 dias para perder en cada una.
3. **El max drawdown es casi identico entre ventanas** en la mayoria (3.86 vs
   3.86, 3.73 vs 3.73, 2.76 vs 2.76): la peor caida de cada estrategia ocurrio
   en este periodo. Es el drawdown de un mercado en baja, que es cuando importa.
4. **OIEXIT_v1 tiene el mejor drawdown (2.76%) y el peor profit factor (0.52)**:
   protege bien el capital pero sus operaciones pierden. Coherente con una
   estrategia de salida conservadora que corta rapido.

**Lo que NO se puede afirmar**:
- Ningun Sharpe es concluyente (n=34, los 10 IC incluyen cero).
- El IR alto de todas es **en parte estructural**: con 56-82% de exposicion, en
  una caida se pierde menos que un indice al 100% simplemente por tener caja.
  No todo el IR es habilidad de seleccion.
- Es **un solo regimen** (mercado en baja). Que ML_SCANNER lidere aca no dice
  nada sobre un mercado alcista: en la ventana completa, que incluye el rally,
  el orden es distinto.

**Implementado**: el reporte HTML ahora muestra las DOS ventanas, con la
comparable como tabla principal y la completa como secundaria, cada una con el
retorno de su benchmark en el encabezado. Parametro `--desde`
(default 2026-05-30, constante `VENTANA_COMPARABLE`).

**Ref**: scripts/forward_testing/ft_reporte_html.py

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
