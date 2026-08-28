# Fuentes de datos fundamentales -- evaluacion (27/8/2026)

Registro de la investigacion sobre de donde traer los estados contables
trimestrales. NO es documentacion de codigo en produccion: es el conocimiento
que costo trabajo obtener y que no se puede derivar leyendo el repo.

ESTADO: investigacion cerrada. **Fase 1 (normalizador) y Fase 2 (ingesta y
persistencia) implementadas** -- ver secciones 11 y 12. La fuente SEC esta
ingestada y andando en PARALELO a yahooquery, sin ningun consumidor todavia.
Pendiente: la capa derivada (multiplos historicos, "caro vs si misma") y
decidir cual fuente sigue y cual se deprecia (seccion 10).

`scripts/oneshot/sec_xbrl_prototipo.py` es el prototipo de la investigacion y
queda como material historico reproducible; el modulo bueno es el de
`src/utils/`.

---

## 1. Por que se hizo

Explorando la idea de construir escenarios de valuacion (dado un precio
objetivo, que PER / EV-EBITDA / P-S implica, y como se compara contra pares y
contra su propia historia) aparecieron dos bloqueos:

1. **Sin profundidad historica no hay "caro vs si misma".** yahooquery sirve
   ~5 trimestres reales de income; el TTM consume 4 -> quedan 2 puntos.
2. **La base actual estaba mas vacia de lo que parecia.** Ver auditoria.

---

## 2. La auditoria que disparo todo

Al 26/8/2026, **97 de 200 tickers no tenian su ultimo balance publicado**
(48% del universo). El ultimo refresh habia corrido el 5/8, en plena
temporada de balances del Q2.

### Como se midio (el metodo, que es reutilizable)

Las dos fuentes de calendario del proyecto NO sirven para esto:

| fuente | estado al 26/8 |
|---|---|
| `earnings_historico` | ultimo anuncio 3/8 -> **22 dias atrasada** |
| `earnings_calendar` | **166 de 200 con fecha NULL** (83%) |

El metodo que si funciono: **proyectar la cadencia propia de cada ticker**.
`earnings_historico` tiene 26 trimestres de fechas de anuncio por ticker, lo
que da el lag mediano entre cierre fiscal y publicacion. Con eso:

```
proximo_Q_esperado = ultimo_Q_con_income + 3 meses (fin de mes)
anuncio_esperado   = proximo_Q_esperado + lag_mediano_del_ticker
falta si            anuncio_esperado < hoy - 5 dias de tolerancia
```

Validacion del metodo en el borde del corte: NVDA vencido 4 dias, ZM 2 dias,
y DELL / SNOW / CRM / CRWD con fecha esperada aun futura (no reportaron).
Los calendarios fiscales desplazados se acomodan solos, sin regla especial.

### Resultado del refresh

Correr el refresh bajo el hueco de **97 -> 30**. Los 30 restantes se parten:

- **7 trabados en yahooquery** (BAC, UNH, ABT, BA, MDLZ, AMT, AMZN): 27 a 42
  dias vencidos. No se arreglan esperando.
- **23 recientes** (<=19 dias): se resuelven solos.

---

## 3. Los tres modos de falla del estado actual

Importante para calibrar cuanto desconfiar: **los valores almacenados no son
falsos, son viejos o faltantes**. No hay ningun caso de ROE mal calculado.

**a) Datos viejos presentados como actuales.** 27 tickers mostraban el Q de
marzo como si fuera el ultimo. Sin marca de vigencia.

**b) Filas "stub" -- parciales, no vacias.** yahooquery crea la fila del
trimestre nuevo con el EPS reportado pero sin el estado de resultados. Mezcla
dato fresco, dato copiado del Q anterior y dato faltante:

| campo (JPM 2026-06-30) | stub | Q anterior | |
|---|---|---|---|
| `eps_ttm` | 23.34 | 20.88 | nuevo |
| `book_value_per_share` | 128.379001 | 128.379001 | copiado |
| `revenue_ttm` / `roe_ttm` | NULL | 186.941M / 16.7% | falta |

Consecuencia practica: `ps_ratio_px` y `ev_ebitda_px` quedan NULL en esos
tickers, porque `compute_multiplos_px` busca los denominadores TTM en la fila
mas reciente, que es justo la que no los tiene.

**c) Contaminacion silenciosa de las medianas de pares.** Un ticker sin ROE
**desaparece del calculo de la mediana de su sector**. El `peer_n` de
Healthcare/USA en ROE da 10, 11 o 12 segun el trimestre, y nada lo indica.
Este es el peor: afecta a tickers que estan perfectos.

### El error de calculo, cuantificado

Un denominador TTM mas viejo que el ultimo trimestre PUBLICADO si produce un
multiplo equivocado. Simulado sobre 146 tickers, PER al precio de hoy con el
TTM del Q anterior vs el vigente:

| | |
|---|---|
| error absoluto mediano | **9.3%** |
| percentil 75 / 90 | 24.1% / 53.4% |
| casos con error >10% | 68 de 146 |

Peores: TWLO 32 vs 357, CVX 19.2 vs 34.7, MU 21.3 vs 44.3, ABBV 74.3 vs 129.5.
El sesgo es sistematico: si las ganancias crecieron, el denominador viejo es
mas chico y **el multiplo sale inflado** (la empresa parece mas cara).

NOTA: precio de hoy / TTM del ultimo balance reportado **es correcto** -- es
la definicion de multiplo trailing. El error es usar un TTM mas viejo que lo
ya publicado.

---

## 4. Las tres fuentes

### yahooquery (actual)

- **Aporta**: capa normalizada sobre 200 tickers heterogeneos (US, ADR, banco,
  no-banco, GAAP, IFRS) con el MISMO esquema. Eso es lo que hace posible todo
  lo cross-ticker (peer set, screener sectorial, medianas, z-scores).
  Ademas sirve precios, opciones y perfiles.
- **Falta**: profundidad (5 Q reales de income), trimestres completos (stubs),
  frescura en los trabados, campos sectoriales de banca
  (`LoansReceivable = None`, `ProvisionForDoubtfulAccounts = None`),
  point-in-time (sirve el dato YA reexpresado).
- **Anomalia detectada**: AAP tiene `eps_q` 0.39 duplicado en 2026-03-31 y
  2026-04-30, y su TTM cuenta el mismo trimestre dos veces
  (0.39+0.39+0.10-0.02 = 0.86). **Es el unico caso en los 200.**

### Alpha Vantage

- **Aporta**: 81 trimestres desde 2006, **cero campos vacios de
  revenue/netIncome, cero fechas duplicadas**. Trimestres completos -> no
  necesita el parche de dos anclas. Mas fresco que yahooquery en los trabados
  (tiene el Q2 de BAC que Yahoo no sirvio en 42 dias).
- **Falta / contras**: normalizacion de un tercero, sin point-in-time, y un
  **quiebre de definicion**: el `totalRevenue` de BAC salta de ~25B a ~47B
  entre 2024-09 y 2024-12 (+85%) y se queda ahi. Es cambio de criterio (bruto
  vs neto de intereses), no negocio. AAPL en el mismo periodo es continuo, asi
  que el quiebre esta acotado a **revenue de financieros**. `netIncome` de BAC
  es continuo.
- **Cuota**: si la key es free tier, 25 llamadas/dia.

### SEC XBRL

- **Aporta**: fuente primaria y oficial, sin intermediario. ~74 trimestres por
  ticker (17 anios). **Point-in-time real y verificado**: el mismo hecho
  aparece repetido desde filings distintos con su `filed`, lo que permite
  reconstruir que se sabia en una fecha dada.
- **Falta**: **147 de 200**. Los 53 que no estan son ADR extranjeros que
  presentan 20-F anual (BABA, TSM, ASML, SAP, TM, NVO, ITUB, VALE...); son
  irrecuperables por esta via. No trae precios ni opciones. **No viene
  normalizado.**

---

## 5. Como presenta la informacion SEC XBRL

No es una tabla de estados contables: es un **repositorio de hechos sueltos**,
agrupados por `taxonomia -> tag -> unidad -> [hechos]`.

```json
{ "start": "2026-04-01", "end": "2026-06-30", "val": 62647000000,
  "accn": "0001018724-26-000026", "fy": 2026, "fp": "Q2",
  "form": "10-Q", "filed": "2026-07-31" }
```

JPM declara **918 tags**, de los cuales 433 tienen dato reciente. No existe
"el estado de resultados": hay 918 conceptos que hay que ensamblar.

### Los 4 problemas estructurales

**1. Sinonimos.** Cada empresa elige su tag. Medido sobre los 200 en CY2026Q2:
`RevenueFromContract...Excluding` cubre 78, `Revenues` 48, `...Including` 3.
**La union de 3 tags cubre 118 de 123** (96%). Los 5 restantes (GS, JPM, MS,
WFC, LAC) son bancos.

**2. Los tags cambian con el tiempo DENTRO de la misma empresa.**

| | tag | ultimo dato |
|---|---|---|
| NVDA | `RevenueFromContractWithCustomer...` | 2022-01-30 |
| UNH | `StockholdersEquity` | 2015-06-30 |
| WMT | `CommonStockSharesOutstanding` | 2012-01-31 |

No dejaron de reportar: cambiaron de tag. **Por eso no sirve un mapa
ticker->tag; hay que resolver el candidato POR PERIODO.** La buena noticia es
que eso es UNA tabla global de sinonimos, no 200 mapeos por empresa.

**3. El Q4 no existe.** El 10-K reporta el anio, no el trimestre suelto:

| ticker | trimestres de 3m faltantes |
|---|---|
| AAPL | sep-24, sep-25 |
| WMT | ene-25, ene-26 |
| JPM | dic-24, dic-25 |

Siempre el Q4 fiscal. Se deriva **Q4 = anual - 9 meses** (los hechos de 9
meses si estan).

**4. Restatements.** Cada trimestre aparece 1 o 2 veces desde filings
distintos. Hay que quedarse con uno (`filed` mas reciente) -- y esa misma
multiplicidad ES la capacidad point-in-time.

### Lo que NO es problema

- La **acumulacion YTD es detectable**: el frame declara `start` y `end`, asi
  que se sabe si un hecho es de 3, 6, 9 o 12 meses.
- La API de **frames** (`/api/xbrl/frames/...`) indexa por trimestre
  CALENDARIO, asi que los de calendario fiscal desplazado no aparecen. Sirve
  para medir cobertura, **no como via de ingesta**. Para ingesta va
  `companyfacts` por empresa (~4 MB c/u).
- SEC corta las descargas con `curl` en loop (conexion TLS nueva por pedido).
  Con `requests.Session` (keep-alive) y ~0.55s entre pedidos: **147 de 147,
  cero fallos, 522 MB**.

---

## 6. El prototipo y sus resultados

`scripts/oneshot/sec_xbrl_prototipo.py`. ~120 lineas. Resuelve los 4
problemas. Sobre 147 tickers USA: **parseo en 8 segundos**, 15.335 filas
trimestre-ticker, 29 conceptos, mediana de **~70 trimestres por ticker**.

### Precision vs yahooquery (758 trimestres desde 2024)

| concepto | comparables | exacto (<0.1%) | <1% | desvios >5% |
|---|---|---|---|---|
| pretax_income | 709 | 99% | 99% | 5 |
| gross_profit | 282 | 99% | 99% | 2 |
| net_income | 708 | 98% | 99% | 4 |
| eps_diluted | 566 | 95% | 97% | 16 |
| revenue | 758 | 89% | 91% | 40 |
| operating_income | 600 | 60% | 66% | 124 |

### Los desvios que NO son errores

**`operating_income` (60%)**: el `OperatingIncome` de yahooquery es
EXACTAMENTE `revenue - cost_of_revenue - opex`, un subtotal que Yahoo calcula.
El de SEC es el que la empresa PRESENTO. Caso INTC Q1-2026:

```
yahoo = +0.934B   SEC = -3.136B   formula de Yahoo = 0.934B
```

SEC dice perdida operativa, Yahoo dice ganancia. **El numero presentado es el
de SEC.** Aca el que se desvia es yahooquery.

**`revenue` en bancos**: cruce de 3 vias sobre AAPL/BAC/JPM/NVDA/UNH/WMT ->
SEC vs AV 67% exacto (los 20 desvios son bancos), SEC vs yahooquery 81%,
operating_income 100% en ambos. **Tres fuentes, tres definiciones de
"ingresos" de un banco.** Verificado caso por caso:

| ticker | tag que reproduce a yahooquery |
|---|---|
| GS, AXP | `RevenuesNetOfInterestExpense` (exacto) |
| BAC | `Revenues` (exacto) |
| JPM, MS | ninguno; el mas cercano queda +8.5% / +7.4% |
| C | **ningun tag publica el trimestre de 3 meses** |

No hay tag correcto: hay una decision a tomar.

---

## 7. Los tres errores de normalizacion encontrados

**Lo mas valioso del ejercicio.** Los tres son del tipo SILENCIOSO: devuelven
un numero plausible de otra cosa. Los introdujo el prototipo y ninguno se
habria detectado sin el cruce contra otra fuente.

| # | error | evidencia | efecto del arreglo |
|---|---|---|---|
| 1 | `ProfitLoss` como sinonimo de `NetIncomeLoss` | incluye minoritarios: FCX +41%, CARR +11%, DIS +7.8% | net_income **74% -> 99%** |
| 2 | `...BeforeIncomeTaxesDomestic` como pretax | es el SEGMENTO domestico, no el total (CRWD, SNOW, PATH, AVAV) | pretax **83% -> 99%** |
| 3 | EPS y acciones del Q4 derivados por resta | son promedios PONDERADOS: no se restan | eps **87% -> 95%** |

`NetIncomeLoss` coincide +0.0% con yahooquery siempre que existe; el problema
aparecia cuando faltaba y el extractor caia al siguiente candidato.

**Regla que sale de esto**: la lista de candidatos debe contener SOLO
sinonimos verdaderos. `ProfitLoss` y `...Domestic` son otros renglones del
estado de resultados, no variantes del mismo.

**Detector**: los tres se manifestaron igual -- el extractor uso **dos tags
distintos dentro de la misma serie de un ticker**. Un normalizador de
produccion debe registrar el tag por dato y marcar los cambios de tag intra-
serie como sospechosos.

**Sobre el Q4 y los promedios ponderados**: la formula correcta para el
numero de acciones del Q4 es `4*FY - 3*9M`, no `FY - 9M`. Mientras no este
implementada, el prototipo ANULA el EPS y las acciones de los Q4 derivados:
es preferible un hueco a un numero equivocado.

---

## 8. Cobertura por concepto (147 tickers USA, dato desde 2025)

| nivel | conceptos | cobertura |
|---|---|---|
| **listos** | revenue, tax_provision, equity, cash, cfo, assets, net_income_common, shares_diluted, eps_diluted, net_income, d_and_a | **95-99%** |
| **usables con huecos** | pretax 93, shares_out 90, current_liabilities 89, goodwill 89, current_assets 89, capex 87, sga 85, intangibles 82, operating_income 79, debt_short 78, debt_long 76 | 76-93% |
| **problematicos** | liabilities 73, interest_expense 71, operating_expense 65, cost_of_revenue 63, inventory 58, **rnd 40, gross_profit 37** | <75% |

`gross_profit` al 37% importa: el margen bruto habria que derivarlo como
`revenue - cost_of_revenue`, y eso solo llega al 63%.

**Hueco conocido**: los conceptos de flujo de caja (cfo, capex, d_and_a)
tienen mediana de 35-40 trimestres contra ~70 del resto, porque el estado de
flujo se reporta ACUMULADO en el anio y el prototipo solo deriva el Q4. Falta
la desacumulacion general (Q2 = H1 - Q1, Q3 = 9M - H1). Es mecanico pero
duplica la logica de derivacion, y es donde entra el FCF.

---

## 9. Comparativa y costos

| | yahooquery | Alpha Vantage | SEC XBRL |
|---|---|---|---|
| cobertura | 200 | por verificar | **147** (USA) |
| trimestres | 5 de income | 81 | ~74 |
| trimestres completos | NO (stubs) | SI | SI (tras derivar Q4) |
| frescura en trabados | NO | SI | SI |
| normalizacion | hecha | hecha | **la haces vos** |
| quiebres de definicion | -- | revenue de bancos, 2024 | mismo tema, pero **elegis el tag** |
| point-in-time | NO | NO | **SI (unico)** |
| campos de banca | inexistentes | inexistentes | existen, sin estandarizar |
| precios / opciones | SI | SI | NO |
| dependencia | tercero | tercero | oficial |
| costo de integracion | ya hecho | bajo | **alto pero acotado** |

### Costo real de SEC

**No esta en el codigo** (el prototipo son ~120 lineas y corre 147 tickers en
8 segundos). Esta en:

1. **Descubrir la lista de sinonimos correcta, concepto por concepto**, y
   validarla contra otra fuente. Se introdujeron 3 errores silenciosos en 29
   conceptos.
2. **La validacion cruzada es permanente, no de una vez.** Un tag mal elegido
   no falla ni avisa.
3. Ingesta: ~522 MB una vez para los 147; incremental por el feed de
   `submissions`.
4. Mapeo de calendario: SEC usa la fecha fiscal exacta (AAPL cierra
   2026-06-27), la convencion del proyecto es fin de mes -> matcheo con
   tolerancia de +-7 dias.

### Pendientes tecnicos si se avanza con SEC

1. Desacumulacion YTD del flujo de caja (Q2 = H1 - Q1, Q3 = 9M - H1).
2. Formula correcta del Q4 para promedios ponderados (`4*FY - 3*9M`).
3. Decision explicita sobre revenue de bancos; C y JPM/MS necesitan
   tratamiento aparte.
4. Detector de cambio de tag dentro de una serie.
5. Completar los conceptos de cobertura <75% o decidir derivarlos.

---

## 10. Decisiones pendientes

Ninguna tomada al cierre de esta etapa.

1. **Fuente unica o convivencia.** Si se elige una fuente historica distinta
   de yahooquery, el punto de HOY tiene que calcularse con la MISMA fuente:
   empalmar historia de una con el valor actual de otra mete el salto de
   fuente justo en el percentil que interesa. Consecuencia: van a convivir dos
   "PER de hoy" ligeramente distintos, y no deberian mostrarse en la misma
   pantalla.
2. **Cuanta historia.** 5 anios cubre un ciclo y evita el problema de cambio
   de regimen del negocio (el PER de NVDA en 2008 no es ancla valida para
   NVDA 2026). 20 anios suena mejor pero no cuesta menos.
3. **Bloque bancario.** Ver `docs/` -- los indicadores tipo BCRA (calidad de
   cartera, previsiones, cobertura, ratio de capital) NO estan en yahooquery
   ni en Alpha Vantage. En SEC estan pero sin estandarizar: de 4 conceptos
   medidos sobre los 18 financieros, la cobertura fue 3, 7, 3 y 6 tickers, con
   conjuntos DISTINTOS en cada uno. Ningun tag cubre ni a los 4 bancos
   grandes; Citigroup no aparece en ninguno. Ademas `Tier1RiskBasedCapital
   ToRiskWeightedAssets` existe como tag pero su ultimo dato en JPM es de
   **2009**. Rinde para ~6 prestamistas reales del universo.

---

## Arreglos aplicados (lo unico que si toco produccion)

`scripts/manual/refresh_fundamentales.bat`, 3 bugs:

1. **Faltaban `compute_multiplos_px` y `compute_sector_valuacion_px`** en la
   cadena. Apenas el refresh agrega un trimestre nuevo, la fila mas reciente
   queda sin `*_px` y el dashboard muestra la valuacion vacia. Se vio en vivo:
   PER cayo a 100 y P/S a 103 hasta correrlos a mano.
   **El orden importa**: `compute()` escribe vs_sector con los multiplos
   FISCALES y `compute_sector_valuacion_px()` despues pisa las 4 de valuacion
   con los `*_px`. Si multiplos_px corre al final, el comparativo sectorial
   queda con precios congelados en la fecha del balance.
2. **`pause` incondicional** -> imposible de automatizar. Ahora:
   `REFRESH_NO_PAUSE=1` lo desactiva de forma determinista, y si no, una
   heuristica sobre `%cmdcmdline%` pausa solo al hacer doble clic. El escape
   explicito existe porque la heuristica sola falla: invocar
   `cmd /c ruta\refresh.bat` tambien matchea el nombre del archivo.
3. **El codigo de salida era siempre 0.** El pipe a `tee` pisa `%ERRORLEVEL%`
   con el de tee. Verificado: python saliendo con 7 se veia como 0, o sea que
   el .bat reportaba "REFRESH COMPLETO" aunque fallara. Ahora usa un marcador
   escrito DENTRO del bloque, antes del pipe.

`scripts/compute_fundamentales_sector.py`: flag `--valuacion-px` para que el
.bat pueda invocar `compute_sector_valuacion_px()`, que no tenia entrypoint.

**Queda como estaba** (decision pendiente, no bug): las capas derivadas corren
aunque el refresh falle. Tiene sentido para un refresh PARCIAL; con un fallo
total es trabajo al pedo. Distinguir uno de otro es una decision de diseno.

---

## 11. Fase 1 -- el normalizador implementado (27/8/2026)

`src/utils/sec_xbrl.py`. Modulo PURO: sin DB, sin red, sin config, solo
stdlib. Tests en `tests/test_sec_xbrl.py` (22 casos sinteticos, sin red ni
DB ni dependencia del cache de 522 MB).

API: `normalizar(companyfacts, hasta_filed=None, desde=None)` devuelve
`{entidad, cik, periodos[], avisos[], meta{}}`. Cada periodo trae
`period_end`, `fiscal_year`, `fiscal_quarter` y los conceptos presentes --
un concepto ausente NO aparece como clave, nunca se rellena con cero.
`hasta_filed` habilita el point-in-time.

### Lo que cambio respecto del prototipo

**Desacumulacion general.** El prototipo solo derivaba el Q4 (FY - 9M). Ahora
desacumula toda la cadena (Q2 = H1 - Q1, Q3 = 9M - H1, Q4 = FY - 9M), que es
como se informan el estado de resultados y el de flujo. Resultado sobre los
147 tickers, mediana de trimestres por concepto:

| concepto | prototipo | modulo |
|---|---|---|
| cfo | 36 | **71** |
| capex | 35 | **68** |
| d_and_a | 40 | **69** |

**El EPS acumulado NO es un promedio ponderado -- es aditivo.** Este fue el
error conceptual mas caro de la fase. El prototipo anulaba todos los Q4 de
EPS por no saber derivarlos. La verificacion en AAPL FY2025 lo zanjo:

```
EPS anual 7.46 | EPS de 9 meses 5.62 | Q4 real 1.85
resta simple        7.46 - 5.62          = 1.84   <- correcto
formula ponderada   4*7.46 - 3*5.62      = 12.98  <- absurdo
```

El EPS de 9 meses es `resultado_9m / acciones_promedio_9m`, que se comporta
como la SUMA de los EPS trimestrales. La algebra de promedio ponderado
(`k*acum_k - (k-1)*acum_(k-1)`) aplica al NUMERO DE ACCIONES, que si es un
promedio. Confundirlos da resultados sin sentido. El modulo trata cada uno
con su metodo y lo documenta en el docstring de `_serie_ponderada`.

**El EPS se calcula sobre el resultado de los accionistas COMUNES.** Usar
`NetIncomeLoss` a secas en el control cruzado hacia fallar todo el sector
financiero, donde los dividendos preferidos ya estan descontados del EPS
publicado. JPM pasaba 36 de 73 trimestres al descarte. Con
`NetIncomeLossAvailableToCommonStockholdersBasic` como preferido: **EPS de
JPM de 55 a 72 trimestres, discordancias de 36 a 2**.

**Solo se restan acumulados ADYACENTES.** Lo encontro un test. Si falta un
tramo intermedio (por ejemplo hay Q1 y FY pero no 9M), la diferencia abarca
dos trimestres y quedaria imputada a uno solo -- un error que no se nota. Y
el `k` de la formula sale de la DURACION del hecho, no de su posicion en la
lista de tramos disponibles.

**El indice de periodos lo definen solo los conceptos de duracion.** El
`EntityCommonStockSharesOutstanding` de dei viene fechado en la PORTADA del
filing, semanas despues del cierre; si definiera periodos generaria una fila
por filing. Los instantaneos se enganchan al trimestre por cercania (45 dias,
y los cierres estan a ~91 dias, asi que no puede agarrar el equivocado).

**El calendario fiscal se deduce y se proyecta.** No se puede asumir por mes:
AAPL cierra a fines de septiembre, WMT el 31 de enero, NVDA a fines de enero.
Se deduce de los hechos anuales y se proyecta un anio hacia adelante, porque
el ejercicio EN CURSO todavia no tiene hecho anual (sale con el 10-K) y sin
eso los trimestres mas recientes -- los que interesan -- quedaban sin
etiquetar. Resultado: **145 de 147 tickers con su ultimo periodo etiquetado
FY/Q**.

### Precision vs yahooquery (trimestres desde 2024)

| concepto | comparables | exacto (<0.1%) | <1% | desvios >5% |
|---|---|---|---|---|
| gross_profit | 282 | 99% | 99% | 2 |
| pretax_income | 709 | 99% | 99% | 5 |
| net_income | 709 | 98% | 99% | 4 |
| **cfo** | **713** | **96%** | **97%** | 18 |
| eps_diluted | 667 | 89% | 94% | 24 |
| revenue | 758 | 87% | 90% | 53 |
| operating_income | 600 | 60% | 66% | 124 |

`cfo` entra nuevo gracias a la desacumulacion, con 96% de exactitud sobre 713
trimestres. `operating_income` sigue en 60% por la divergencia de definicion
ya explicada (seccion 6): ahi el que se desvia es yahooquery.

### Avisos que emite

No son errores: son "esto merece que alguien lo mire".

- `cambio_de_tag`: la serie de un concepto se armo con mas de un tag. Es la
  red contra el error silencioso -- los tres casos que se colaron en la
  investigacion se manifestaron todos asi.
- `ponderado_discordante`: la resta de acumulados no coincide con
  resultado/acciones. El trimestre se descarta.
- `ponderado_implausible`: el numero de acciones derivado se fue de banda
  respecto del acumulado (split o emision grande). Se descarta.

### Pendiente de la Fase 1

Conceptos con cobertura baja que habria que completar o decidir derivar:
`gross_profit` (mediana 6 trimestres -- el margen bruto habria que calcularlo
como revenue - cost_of_revenue), `net_income_common` (10), `inventory` (33),
`operating_expense` (36), `rnd` y `net_interest_income` (0: los tags elegidos
no son los que usan las empresas, hay que descubrirlos).

---

## 12. Fase 2 -- ingesta y persistencia (28/8/2026)

Fuente PARALELA. Las dos conviven; ningun consumidor actual lee las tablas
`fundamentales_sec_*` y `fundamentales_income_q` y companiaa siguen intactas.
Decidir cual sigue y cual se deprecia es un paso posterior.

### Disciplina de separacion (sin repo separado)

Se evaluo convertir la fuente en un proyecto independiente y se decidio que
no, por tres razones: el modulo ya esta desacoplado (importa `collections` y
`datetime`, nada mas), el dato tiene que aterrizar en el MISMO PostgreSQL para
cruzarse con precios y sectores -- asi que un repo aparte seria otro codigo
escribiendo en un esquema compartido, peor acoplamiento que tenerlo junto --
y no existe todavia un segundo consumidor.

En su lugar, la costura queda cortada de antemano:

- `src/utils/sec_xbrl.py` -- normalizador PURO (stdlib).
- `src/data/sec/` -- ingesta. **Regla de una sola direccion: nada de este
  paquete importa del lado de trading** (scoring, bots, scanner, strategies,
  pipeline). Si algun dia hiciera falta, es senal de que algo esta mal ubicado.
- `scripts/refresh_fundamentales_sec.py` -- el UNICO lugar de la fuente que
  toca la DB y el universo.

El dia que aparezca un segundo consumidor, `git subtree split` sobre esas
rutas la separa sin desenredar nada.

### Las 3 tablas

| tabla | contenido |
|---|---|
| `fundamentales_sec_q` | la serie. PK (ticker, period_end). 31 columnas de concepto + fiscal_year/fiscal_quarter + `origen` JSONB + `filed_max` |
| `fundamentales_sec_avisos` | avisos del normalizador, consultables |
| `fundamentales_sec_ingesta` | control de descarga por ticker (habilita el incremental) |

**Una tabla de serie y no cuatro.** yahooquery usa 4 (income/balance/cashflow/
valuation) porque asi viene su API. SEC NO organiza por estado contable:
publica hechos sueltos y el normalizador produce una fila por (ticker,
periodo) con los 31 conceptos juntos. Volver a partirla seria re-imponerle a
SEC la forma de otra fuente, y esa particion es justo donde aparece la
ambiguedad de "a que estado pertenece este concepto".

**Las columnas de concepto se generan desde `src/utils/sec_xbrl.py`** en el
DDL, para que el esquema no pueda desincronizarse del normalizador.

**`origen` JSONB en vez de 62 columnas.** Guarda `{concepto: {tag, derivado}}`:
que tag produjo cada numero y si se derivo por desacumulacion. Es lo que se
mira cuando un valor no cuadra.

**El point-in-time NO se almacena como filas multiples.** La tabla guarda la
vista vigente (ultimo `filed`); el point-in-time se RE-DERIVA corriendo
`normalizar(hasta_filed=...)` sobre el cache en disco. Guardar cada version
duplicaria las filas por una capacidad que todavia no se consume.

### El incremental

`submissions` pesa **164 KB contra 3.8 MB** de companyfacts (23x) y trae el
accession del ultimo 10-Q/10-K. Ese es el disparador: si el accession coincide
con el ya ingestado, no se baja el archivo pesado.

Verificado corriendo dos veces seguidas sobre AAPL/JPM/KO:

```
pasada 1  {'actualizado': 3}   descargado: 16.7 MB
pasada 2  {'sin_cambios': 3}   descargado:  0.0 MB
```

Sobre los 147 sin balances nuevos: ~24 MB en vez de ~522 MB.

### SEC exige User-Agent identificable

Devuelve **403 en todos los endpoints** si el User-Agent no lleva un mail de
contacto. Se configura en el `.env`:

```
SEC_USER_AGENT=tu-mail@dominio.com IndicadoresDeStockML
```

El cliente NO reintenta ante un 403 (no es transitorio, y reintentarlo esconde
la causa detras de un "no disponible"), y el script falla temprano con la
instruccion en vez de reventar a mitad de una corrida.

### Resultado de la ingesta completa

| | |
|---|---|
| tickers | 147 |
| filas | 9.062 |
| cobertura temporal | 2007-09-30 a 2026-08-02 (19 anios) |
| avisos | 1.796 |
| tiempo (desde cache) | 48 s |

Avisos por tipo: `cambio_de_tag` 1.131 (145 tickers), `ponderado_discordante`
579 (102), `ponderado_implausible` 86 (26). Los conceptos con mas cambios de
tag -- donde mirar primero si algo no cuadra -- son `cash` (119 tickers),
`revenue` (114), `d_and_a` (106), `shares_out` (100).

Validacion **leyendo de la tabla** (no del modulo), contra yahooquery, para
probar que el viaje a la DB no pierde nada:

| concepto | n | exacto | <1% |
|---|---|---|---|
| net_income | 709 | 98% | 99% |
| cfo | 713 | 96% | 97% |
| eps_diluted | 668 | 89% | 94% |
| revenue | 758 | 87% | 90% |

Identicos a la validacion a nivel modulo.

### Definicion de terminado (acordada antes de empezar)

Para que la fuente no se desborde, "lista" significa exactamente estas tres
cosas, y las tres estan:

1. los 31 conceptos ingestados para los 147 USA
2. refresh incremental andando (sin re-bajar 522 MB)
3. validado contra yahooquery con los numeros medidos

Todo lo demas -- bloque bancario, paridad total de ratios con yahooquery,
point-in-time consumible, capa derivada -- es una decision aparte, tomada
despues, con un consumidor concreto que lo pida.

### Ventana de retencion y las dos fechas de publicacion (28/8/2026)

**Retencion: 2018-01-01 en adelante** (4.781 filas, 33 trimestres promedio por
ticker, 143 de 147 con 5 anios completos). SEC tiene ~19 anios pero el techo
real de cualquier analisis contra precio es `precios_diarios`, que arranca el
2/1/2020 -- y en 2024 para 60 de los 147 tickers. Guardar 2007 era peso muerto.

El corte NO es 2020 sino 2018 porque **el TTM necesita pista de despegue**: el
primer dia con precio necesita el TTM del ultimo trimestre publico en esa
fecha (Q3-2019 para un calendario normal), y ese TTM se extiende hasta
Q4-2018. Verificado: de los 147, **129 tienen un trimestre publico al 2/1/2020
y los 129 tienen TTM completo de 4 trimestres; cero sin TTM**. Los 18
restantes salieron a bolsa despues o su precio arranca en 2024.

El cache en disco conserva la historia completa: recuperarla es correr con
`--desde` mas viejo, sin salir a la red.

**Dos fechas de publicacion, que responden preguntas distintas:**

| campo | que responde |
|---|---|
| `filed_primero` | desde cuando el trimestre fue PUBLICO -- con esta se arma la serie |
| `filed_ultimo` | de que presentacion viene el valor guardado (procedencia) |

Un trimestre no esta disponible el dia que cierra. Medido: **AAPL publica a
los 34 dias, JPM entre 31 y 44**; mediana estable de 31-35 dias para
2019-2026. Armar la serie con `period_end` adelantaria cada trimestre mas de
un mes -- sesgo sistematico en la direccion que hace ver mejor un backtest.

`filed_primero` es el filed MAS VIEJO del periodo, no el del hecho elegido:
`_elegir` se queda con la ultima reexpresion, cuya fecha puede ser anios
posterior.

**INVARIANTE: un periodo no puede ser publico antes de terminar.** Hay hechos
con fecha de cierre futura en filings anteriores (proyecciones, compromisos)
que daban lag negativo, hasta -309 dias en un caso real. Se descartan al
RECOLECTAR y en las dos ramas (duracion e instantaneos): filtrarlos despues de
tomar el minimo dejaba el periodo sin fecha en vez de con la correcta, y los
instantaneos se enganchan por cercania (+-45 dias) trayendo su propia fecha.
Resultado: 4.781 filas con fecha, cero lags negativos.

CAVEAT: para periodos pre-2015 el lag da ~350 dias, porque su 10-Q original
quedo fuera de la retencion XBRL de SEC y su primera aparicion es como
comparativo anios despues. Irrelevante dentro de la ventana 2018+.
