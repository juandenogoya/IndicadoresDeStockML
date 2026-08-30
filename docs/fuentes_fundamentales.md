# Fuentes de datos fundamentales -- evaluacion (27/8/2026)

Registro de la investigacion sobre de donde traer los estados contables
trimestrales. NO es documentacion de codigo en produccion: es el conocimiento
que costo trabajo obtener y que no se puede derivar leyendo el repo.

ESTADO (29/8/2026): **Fases 1, 2, 3 implementadas + curacion de revenue y
operacion** -- normalizador (sec. 11), ingesta (sec. 12), capa derivada de
multiplos diarios (sec. 13), mapeo curado y detector de mezcla de tags
(sec. 16) y la CALCULADORA de escenarios (sec. 17), que es el primer y unico
consumidor. La fuente SEC corre en PARALELO a yahooquery: ninguna tabla de
`fundamentales_*_q` se toco y ningun flujo de trading la lee.

Reparto de roles decidido: SEC = eje TEMPORAL (historia larga, point-in-time,
multiplos diarios y "caro vs si misma", 147 tickers USA). yahooquery = eje
TRANSVERSAL (los 200 tickers incluidos los ADR sin XBRL trimestral, comparativa
sectorial del ultimo Q). Ninguna reemplaza a la otra: SEC no cubre 53 tickers y
yahooquery no puede hacer historia larga ni point-in-time. El cruce entre las
dos es ademas el control que faltaba.

Pendiente y ABIERTO:
- Los **streamers** (NFLX, WBD): la amortizacion de CONTENIDO no esta en los
  tags de D&A, asi que su EBITDA queda subestimado. Es un hueco de cobertura
  real, sec. 16.7.
- **La cobertura de EV/EBITDA cayo a 50 de 144 tickers** al dejar de calcular
  el EV con deuda parcial (sec. 16.6). Recuperarla exige una fuente de deuda
  de largo plazo que la API de companyfacts no entrega.
- Los 13 tickers de revenue decididos por CRITERIO y no por arbitraje externo
  (sec. 16.2) no tienen confirmacion independiente.
- El objetivo original: escenarios de valuacion implicita y comparativa
  sectorial. La capa de datos ya esta; falta el consumidor.

**Si volves aca con una duda puntual, atajos:**

| pregunta | seccion |
|----------|---------|
| por que los multiplos no usan EPS ni BVPS | 13 (base de split) |
| de donde salen las acciones en circulacion | 13 |
| por que a MA/CAT les faltaba el resultado neto | 14 |
| por que revenue falla en bancos, AMT, PM | 15 |
| ya se probo `Q4 = FY - (Q1+Q2+Q3)`? y `frame`? | 15 (las dos, no sirven) |
| SEC o yahooquery para cada metrica | 15, ultimo bloque |
| como se eligio el tag de cada uno de los 23 | 16.2 |
| que hago si aparece un ticker con mezcla de tags | 16.3 y 16.6 |
| por que se podo `Depreciation` de los sinonimos | 16.5 |
| por que el EV de VZ y T estaba mal | 16.6 |
| se puede comparar el EV/EBITDA de SEC con el de yahooquery | 16.7 (no) |
| para que sirve todo esto, en una sola pantalla | 17 |
| el backtest dio que no predice: sirve igual? | 17.3 (mide otra pregunta) |
| donde estan ROIC y ROTCE | 17.5 (afuera, y por que) |
| por que el crecimiento se mide por trimestre y no por rueda | 17.4, regla 4 |
| sirve sec-api.io / EDGAR como fuente unica | 18.8 (no) |
| por que 56 tickers no tienen EV, y cuanto cuesta arreglarlo | 18.8 (33 gratis) |
| por que companyfacts no tiene el ultimo balance de F o C | 18.3 (atrasa) |
| se puede evitar la desacumulacion del Q4 con otra fuente | 18.6 (no) |
| se puede cubrir RIO/UL/VOD/BABA con SEC | 18.6 (no, es regulatorio) |
| que fuente usar para cada cosa | docs/arquitectura_fuentes.md |

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

---

## 13. Fase 3 -- la capa derivada (29/8/2026)

Objetivo original: **escenarios de valuacion implicita** -- dada una tesis de
precio, que PER / EV-EBITDA / P-S implica y si es plausible. Primero "caro
contra SI MISMA" (su propia historia), despues contra la mediana del sector.
Para eso hace falta una serie DIARIA de multiplos historicos, no una foto por
trimestre: sin distribucion no hay contra que comparar.

Tres modulos PUROS nuevos + dos tablas. Todo LOCAL-only, sin consumidores aun.

| modulo | que hace |
|--------|----------|
| `src/utils/fundamentales_ttm.py` | rolling TTM sobre la serie trimestral SEC + as-of por `filed_primero` |
| `src/utils/sec_acciones.py` | serie POINT-IN-TIME de acciones desde la portada `dei` |
| `src/utils/acciones_series.py` | combina yahooquery + SEC y VALIDA que esten en la misma base de split |

| tabla | contenido |
|-------|-----------|
| `acciones_circulacion` (+ `_validacion`) | acciones por ticker/fecha en base de split ACTUAL, apareable con `precios_diarios` |
| `fundamentales_sec_multiplos_d` | 1 fila por ticker/rueda: multiplos + percentil trailing |

### El descubrimiento que reescribio el diseno: la base de split

Para multiplicar precio por acciones los dos tienen que estar en la MISMA base
de split. Y las dos fuentes estan en bases distintas:

- `precios_diarios` **se corrige retroactivamente** por divisor (`splits.py
  corregir`) -> toda la historia queda en base ACTUAL.
- SEC **re-expresa lo "por accion" hacia atras, pero solo en los comparativos
  de los filings recientes** -> horizonte de reexpresion de ~2 anios. Ninguna
  serie de acciones de SEC esta en una sola base a lo largo de 8 anios.
- yahooquery `OrdinarySharesNumber` **si** re-expresa a base actual (KLAC da
  1.367,5M para 2023, ya post 10:1) -> aparea directo con el precio corregido.

CONSECUENCIA DE DISENO, no negociable: **todos los multiplos se calculan sobre
AGREGADOS**, nunca sobre magnitudes por accion.

    PER = market_cap / net_income_ttm       market_cap = close(D) * shares(D)
    P/B = market_cap / equity
    P/S = market_cap / revenue_ttm
    EV/EBITDA = (market_cap + net_debt) / ebitda_ttm

Los agregados son invariantes al split; `eps_ttm` y BVPS NO, y por eso se
sacaron de la tabla a proposito (ver los DROP COLUMN en el create).

`dei:EntityCommonStockSharesOutstanding` **nunca** se re-expresa (es un hecho
de portada atado a su accession). `us-gaap:CommonStockSharesOutstanding` si se
re-expresa Y ademas mide acciones EMITIDAS en algunos filers (BA:
1.012.261.159 constante contra 754-790M reales) -- no entra ni como respaldo.

### Como se construye la serie de acciones, y por que asi

Yahoo llega a 2022/2023 (4 puntos anuales + 5 trimestrales); SEC llega a 2018
pero en base de su momento. Ninguna alcanza sola. La combinacion se
COMPRUEBA, no se asume: se comparan las dos donde se solapan; si coinciden
dentro del 10%, SEC esta en base actual en ese tramo y se puede extender hacia
atras -- siempre que ademas no tenga saltos, chequeo que **incluye el primer
punto posterior al corte** (un split que caiga justo en el hueco de datos no
se veria mirando solo los previos).

Reglas: **Yahoo manda donde llega** (SEC solo extiende hacia atras);
**ESCALON, nunca interpolacion** (un escalon es dato viejo y etiquetable, una
interpolacion es dato inventado e indistinguible); **la discrepancia avisa, no
corrige**.

Validacion 2021 sobre 200 tickers / 2.379 puntos: **101 arrancan en 2021, 83
en 2022, 16 en 2023+**. Los ratios de rechazo recuperaron los factores de
split exactos sin que se los dijeran: KLAC 10,009 / AVGO 10,003 / NFLX 10,000
/ NVDA 9,984 / NOW 5,012 / CRWD 4,000 / WMT 3,005; mas rechazos por salto
(AMZN x20,03, FTNT x4,91, CSX x2,98, TSLA x3,02) y eventos de capital reales
(RKLB 1,112, BG 1,045, HON 2:1 inverso del spinoff). 74 quedaron "sin
solapamiento" (ADR no-USA + filers de clases multiples) y usan solo Yahoo.

### El error de la granularidad anual, cuantificado

yahooquery sirve UN punto por ano entre 2023 y 2025. Medido contra la serie
trimestral real, el error del escalon: **mediana 0,24% / p75 1,07% / p90 2,74%
/ p99 11,35%**, >5% en el 4,7% de los casos. Cola: BG 33,0%, CAR 29,7%, SPGI
29,1% (fusion con IHS Markit). El error va 1:1 al market cap.

Tiene SIGNO: un conteo viejo queda ALTO (las empresas recompran), medido
+0,29% en media -- sobreestima el market cap y hace ver el multiplo mas barato
de lo que es. Por eso la tabla guarda `shares_dias` (antiguedad del conteo):
sin esa columna el riesgo es invisible en la fila.

### Percentil ESTRICTO

Ventana movil TRAILING de 756 ruedas (~3 anios), y se exige que este llena EN
TIEMPO, no solo que haya `min_obs` observaciones. Son dos cosas distintas: con
250 dias de historia el percentil sale de un solo regimen de tasas y una sola
fase del ciclo del ticker, y despues se lee como "su rango historico". Se
prefiere un NULL honesto a un numero que invita a decidir.
`--percentil-permisivo` afloja la regla.

Resultado: 152.054 filas / 144 tickers / 2021-01-04 a 2026-08-27. El limitante
NO es SEC sino `precios_diarios`: solo 84 tickers llegan a 756 ruedas.

---

## 14. El resultado neto que SEC no tagea (29/8/2026)

Ocho tickers del universo (MA, CAT, SCCO, AVAV, AVGO, F, FCX, AMT) declaran el
resultado bajo `ProfitLoss` y no bajo `NetIncomeLoss`. MA y CAT no tagean
`NetIncomeLoss` desde 2014 y 2011. Como `ProfitLoss` incluye minoritarios esta
prohibido usarlo como sinonimo (seccion 7), y el efecto era que esos ocho
quedaban SIN resultado neto -- y por lo tanto **sin PER** -- sin que nada
avisara. El modo de falla silencioso, otra vez.

La salida no es un sinonimo nuevo sino una IDENTIDAD contable:

    NetIncomeLoss = ProfitLoss - NetIncomeLossAttributableToNoncontrollingInterest

**Condiciones, ninguna relajable** (`_completar_net_income`):

1. Solo rellena HUECOS. Donde hay `NetIncomeLoss`, gana `NetIncomeLoss`.
2. El hecho de minoritarios tiene que existir PARA ESE PERIODO. No se asume
   cero: medido en AVGO, asumir cero da 11% de error. Solo se toma NCI=0
   cuando la empresa NO tiene el tag en ningun lado.
3. Control cruzado contra `...AvailableToCommonStockholders*` (medicion
   independiente del mismo renglon). Si difieren mas de `TOL_NET_INCOME`
   (5%, mas laxa que la de ponderados porque el control descuenta ademas los
   dividendos preferidos), no se emite y queda aviso.

**Resultado**: 4.478 -> 4.682 trimestres con `net_income`; tickers sin
resultado en ningun Q: 5 -> 1 (AU, que es IFRS). PER: 112.574 -> 117.741
filas. Verificado contra el anual publicado por la propia empresa: MA, CAT,
SCCO, F, AMT y FCX reconcilian EXACTO en FY2023, FY2024 y FY2025.

Quedan afuera AVGO y AVAV, con aviso, no adivinados.

### La variante que se RECHAZO por evidencia

Para rescatar AVGO se probo relajar la condicion 2 a "el mismo filing
(accession) no declara minoritarios". **Contradice el control en 256 de 646
casos verificables (40%)**: MS falla 101 de 122, AIG 38 de 69, WBD 63 de 116.
Un filing puede omitir el hecho sin que los minoritarios dejen de existir --
la misma razon por la que companyfacts descarta los conteos de clases
multiples (hechos dimensionales). NO reintentar este camino.

Medicion de respaldo: 0 tickers del cache usan una variante del tag de
minoritarios sin tener tambien el estandar, asi que la ausencia TOTAL del tag
si significa que no hay minoritarios.

---

## 15. Revenue: el problema es SEMANTICO, no aritmetico (29/8/2026)

El control que lo expuso: **la suma de los 4 trimestres tiene que dar el hecho
ANUAL que publico la propia empresa**. Es independiente de la desacumulacion
(el anual viene del 10-K sin tocar) y escala a 147 tickers.

| concepto | anios-ticker | cuadran |
|----------|--------------|---------|
| net_income | 701 | **99,6%** |
| operating_income | 582 | 98,6% |
| cfo | 723 | 97,0% |
| **revenue** | 721 | **87,8%** |

30 tickers fallan en revenue, y la causa es unica: **mezcla de dos tags de
revenue dentro del mismo ejercicio**. Siempre es el Q4 el que cambia de tag,
porque el 10-K etiqueta distinto que los 10-Q y el Q4 sale del anual.

    AXP  Q1-Q3 RFCWCExclTax  +  Q4 InterestAndDividendIncomeOperating
    GS   Q1-Q3 RevenuesNetOfInterestExpense + Q4 IntDivIncome (49,7B vs 59,3B)
    AMT  Q1-Q3 RFCWCExclTax (0,2B!)  +  Q4 Revenues (2,5B, el correcto)

La red de seguridad **funciono**: los 20 peores ya tenian aviso
`cambio_de_tag` en revenue. Nadie lo estaba consumiendo. El aviso tiene recall
alto y precision baja (marca 59 tickers, rotos hay 30).

### Inventario del crudo (lo que SEC realmente da)

Sobre 38.238 hechos de revenue en 147 tickers:

| tramo | % |
|-------|---|
| Q (3 meses discreto) | 55,6% |
| FY | 19,6% |
| H (6 meses) | 12,9% |
| 9M | 11,9% |

Campos al 100%: `start`, `end`, `val`, `accn`, `fy`, `fp`, `form`, `filed`.
`frame` en el 32%. Formularios: 10-Q 23.334, **10-K 13.656** (el 10-K no trae
solo el ano: trae los comparativos).

**El 10-K casi nunca publica el Q4 suelto: 67 de 868 ejercicios, y solo 2
tickers de forma consistente.** El Q4 SIEMPRE hay que derivarlo. Es una
restriccion dura de la fuente.

### Las cuatro salidas que se probaron y no alcanzan

Arbitro externo: el revenue trimestral de yahooquery
(`fundamentales_income_q`, period_type 3M).

| regla | vs yahooquery | reconciliacion anual |
|-------|---------------|----------------------|
| actual (tag por periodo) | 89,7% | 87,8% |
| un tag por ejercicio, por cobertura | 89,2% | 99,0% |
| un tag por ejercicio, mayor anual | **91,6%** | **99,1%** |
| hibrido prioridad + umbral de subtotal | 90,6% | 99,0% |

**Leccion no obvia**: "un tag por ejercicio, por cobertura" sube la
reconciliacion de 87,8% a 99,0% y BAJA la verdad (89,7 -> 89,2). En AMT elige
el tag que aporta 3 trimestres (el subtotal) en vez del correcto: el ano queda
coherente consigo mismo y sigue estando mal. **La reconciliacion anual es un
control de consistencia, no de correccion.**

Corolario, y por eso NO se hace `Q4 = FY - (Q1+Q2+Q3)`: esa resta haria que la
reconciliacion pase POR CONSTRUCCION, destruyendo el unico control
independiente que hay. En AMT daria Q4 = 9,3B (real 2,5B) y el control diria
"cuadra". Convierte un error visible en uno invisible.

Tampoco sirve `frame` (la marca de trimestre calendario de la SEC): **89,0%**,
igual que todo lo demas, porque SEC le asigna frame a CADA tag, subtotal
incluido -- AMT frame=0,23B contra yq=2,56B.

Tampoco sirve quedarse SOLO con lo publicado (sin derivar nada):

| revenue, 3.159 Q 2021+ | share | acierta vs yahooquery |
|------------------------|-------|------------------------|
| publicado directo | 78,1% | **89,0%** |
| derivado | 21,9% | **92,4%** |

| net_income, 3.111 Q | share | acierta |
|---------------------|-------|---------|
| publicado directo | 74,4% | **99,8%** |
| derivado | 25,6% | **96,1%** |

**En revenue los derivados le pegan MEJOR que los publicados.** El error no
esta en la aritmetica sino en los hechos publicados -- AMT de nuevo: Q1-Q3
publicados son el subtotal, Q4 derivado es el total. Quedarse con lo publicado
tira la parte buena y cuesta el 22% de los trimestres.

Y pasarse al ANUAL tampoco lo esquiva: la ambiguedad esta medida SOBRE los
anuales. Ademas costaria el TTM (necesita 4 Q) y dejaria el denominador con
hasta ~14 meses de antiguedad en vez de ~4.

CAVEAT sobre el arbitro: yahooquery tampoco es infalible. En MS el que parece
equivocado es yahooquery (SEC `RevenuesNetOfInterestExpense` 16,79B contra
yq 15,60B).

### El tamano real del problema

Comparando los anuales de todos los tags candidatos por ticker:

    102  un solo tag                       -> sin ambiguedad posible
     20  varios tags pero todos coinciden  -> da igual cual se elija
     23  AMBIGUOS de verdad                <- el problema entero
      2  sin anual de revenue

**122 de 147 (83%) no tienen problema.** Los 23 ambiguos, por brecha entre el
mayor y el menor anual:

| categoria | tickers |
|-----------|---------|
| financieras (bruto por intereses vs neto) | PGR 98%, UPST 98%, MS 84%, GS 80%, AXP 79%, WFC 50%, BAC 47%, C 44% |
| subtotal vs total | RBLX 97%, AMT 93%, BG 76%, BLK 38%, MA 37%, VST 32% |
| impuesto interno (bruto vs neto) | PM 62% |
| brecha chica (linea "otros" menor) | COP 25%, HOG 20%, PFE 15%, GM 11%, LYFT 8%, CVX 5%, OXY 3%, FCX 3% |

### Conclusion y camino propuesto (NO implementado)

"Revenue" no es un renglon en XBRL. Una misma empresa puede tagear a la vez
`RFCWCExcludingAssessedTax`, `Revenues`, `RevenuesNetOfInterestExpense` e
`InterestAndDividendIncomeOperating` -- las cuatro correctamente, las cuatro
con `frame`, las cuatro internamente consistentes. **Ninguna operacion sobre
los numeros elige la buena, porque "cual es el total" es un juicio de
definicion, no una cuenta.** Por eso todas las reglas topan en ~91%.

Propuesta: **mapeo CURADO de 23 filas `ticker -> tag de revenue`**, el mismo
patron que el `profile` banco/no-banco con su override curado
(docs/fundamentales_calculo.md), mas un **detector de ambiguedad automatico**
que avise cuando aparezca un ticker nuevo con dos tags de magnitudes
distintas, para que la curaduria no se desactualice en silencio. La curaduria
se puede sembrar con el arbitraje contra yahooquery (resuelve 112 de 147) pero
la decision final es humana.

### Que significa esto para elegir fuente (seccion 10)

- SEC es SOLIDO en `net_income` (99,6%), `operating_income` (98,6%), `cfo`
  (97,0%) y equity.
- SEC es DEBIL en `revenue`, y por arrastre en **P/S y EV-EBITDA**.

Comparativa SEC vs yahooquery al ultimo dato de cada fuente (144 tickers):

| multiplo | n | mediana | p90 | <=5% |
|----------|---|---------|-----|------|
| P/S | 139 | 0,00% | 3,5% | 90% |
| P/B | 132 | 0,39% | 12,4% | 77% |
| PER | 113 | 1,75% | 14,2% | 78% |
| EV/EBITDA | 74 | 9,04% | 38,7% | 34% |

EV/EBITDA diverge por una razon ya declarada: yahooquery usa
`NormalizedEBITDA`, SEC usa EBIT+D&A sin excluir one-offs.


---

## 16. La curacion del revenue, y el defecto que destapo (29/8/2026)

La seccion 15 cierra con un diagnostico y sin solucion: 23 tickers necesitan
un mapeo curado ticker -> tag, y ningun algoritmo lo resuelve. Esta seccion es
lo que se hizo con eso.

### 16.1 El mecanismo

`normalizar()` acepta `tags_curados={concepto: tag}`. Dos decisiones de diseno
que conviene no revertir por comodidad:

**El tag curado REEMPLAZA la lista de sinonimos, no se antepone.** Si se
antepusiera, los demas quedarian de respaldo y el normalizador volveria a
mezclar exactamente en los trimestres donde el tag elegido falta -- que es el
caso que la curacion viene a resolver. Se prefiere el hueco VISIBLE al numero
mezclado invisible. Es la misma regla que gobierna al resto del modulo.

**Acepta una lista** cuando la empresa migro de taxonomia de verdad y ningun
tag solo cubre la ventana. En ese caso sigue siendo posible mezclar dentro de
un ejercicio, y el aviso de 16.3 lo dice.

El mapeo vive en `src/data/sec/tags_curados.py`, que es un modulo de DATOS: sin
logica, sin DB, sin red. Lo aplica `refresh_fundamentales_sec.py`, que llama a
`tags_curados.para(ticker)` -- `{}` para los 124 tickers sin ambiguedad, con lo
que el comportamiento no cambia para ellos.

### 16.2 Como se decidio cada uno de los 23

`scripts/oneshot/revenue_tags_reporte.py` regenera el diagnostico desde cero:
por ejercicio, cuanto da cada tag candidato, que tags usaron los 4 trimestres,
y cuanto da yahooquery. Derivo los mismos 23 tickers de la investigacion
original, por un camino distinto -- una confirmacion util de que el
diagnostico era correcto.

**Diez por ARBITRAJE** contra yahooquery: el anual de ese tag coincide con la
suma de sus 4 trimestres, que es evidencia de una fuente independiente. AMT,
AXP, BAC, BG, BLK, CVX, FCX, GS, PM, UPST.

**Trece por CRITERIO contable**, porque yahooquery NO tiene los 4 trimestres de
ningun ejercicio para ellos: son sus filas stub con `total_revenue` en NULL,
uno de los defectos que motivaron traer SEC en primer lugar. Las dos fuentes
fallan sobre los mismos tickers, y sobre todo en bancos.

Los criterios, que es lo que hay que revisar si algo huele mal:

- **Bancos y seguros** -> el ingreso NETO de intereses. El bruto cuenta como
  ingreso plata que se paga como costo de fondeo; usarlo infla el denominador
  de un P/S entre 40% y 70% y hace al sector incomparable consigo mismo.
  C, MS, WFC, PGR, MA.
- **Industria y consumo con brazo financiero** -> el tag MAYOR si es el
  titular: la actividad de GM Financial y de HDFS es un segmento operativo, no
  un resultado no operativo. GM, HOG.
- **Energia** -> la linea de ventas, NO "total revenues and other income", que
  suma resultados por participaciones y ventas de activos. Mismo criterio que
  CVX, que si tiene arbitraje. COP, OXY, VST.
- **Resto** -> el total. LYFT, PFE, RBLX.

**Alpha Vantage se probo como tercer arbitro y NO sirve.** Para WFC da 125.397
MM en 2024 y 77.198 MM en 2023: cambia de base a mitad de la serie. Es el mismo
problema semantico, no una fuente de verdad. No reintentarlo.

### 16.3 El detector, que es la parte que envejece bien

Un mapeo curado se pudre en silencio: entra un ticker nuevo al universo, o una
empresa cambia de taxonomia, y nadie se entera. Por eso el aviso
`mezcla_en_ejercicio`: los trimestres de un mismo ejercicio armados con tags
distintos.

Es mas fuerte que `cambio_de_tag` y se lee distinto. Migrar de taxonomia entre
anios es legitimo y deja cada ejercicio internamente consistente. Mezclar
DENTRO de un ejercicio no lo es nunca: los 4 trimestres tienen que medir la
misma magnitud para poder sumarse.

**Pero la mezcla sola no alcanza como senal.** De 126 ejercicios que mezclan
tags, 109 son INOCUOS: los dos tags son sinonimos y publican el mismo anual, asi
que ninguna suma cambia (JPM, GOOG, COST). Avisar de los 126 condenaria el
aviso a que nadie lo lea -- que es literalmente lo que le paso a
`cambio_de_tag`, que ya senalaba a los 20 peores casos de revenue y estuvo
marcado todo el tiempo sin que nadie lo mirara.

Por eso se cruza contra el anual de cada tag y se separan tres casos:

| caso | aviso | que significa |
|---|---|---|
| los anuales DIFIEREN | `mezcla_en_ejercicio` | el defecto real, hay que curar |
| los anuales COINCIDEN | (silencio) | sinonimos redundantes, la suma no cambia |
| un solo tag publica anual | `mezcla_no_verificable` | no se puede comprobar |

### 16.4 Resultado

De 30 tickers que fallaban la reconciliacion en revenue queda **uno**: LNC, y
en el ejercicio 2018 -- el borde de la transicion ASC 606, fuera de la ventana
2021+ de la capa derivada.

El efecto en el TTM no es cosmetico:

| ticker | TTM antes | TTM despues |
|---|---|---|
| AMT | 3.444 | 10.819 |
| AXP | 38.726 | 75.950 |
| BG | 36.816 | 80.547 |
| PGR | 69.283 | 91.055 |
| GM | 172.654 | 185.528 |
| C | 102.003 | 85.225 |
| GS | 73.128 | 66.203 |

Seis quedan igual (BLK, MA, PFE, PM, RBLX, UPST): su mezcla estaba en
ejercicios viejos, asi que la curacion les arregla la historia, no el TTM.

**P/S de SEC ya es consumible.** Los valores resultantes al ultimo cierre:
AMT 7,50 | GM 0,41 | C 2,61 | WFC 2,96 | PGR 1,39 | BG 0,27 | GS 4,58.

### 16.5 El defecto que el detector destapo: d_and_a

Con el criterio de 16.3 aplicado a todos los conceptos, aparece uno que nadie
estaba mirando:

| concepto | ejercicios con mezcla consecuente | tickers |
|---|---|---|
| d_and_a | 476 | 79 |
| cfo | 31 | 10 |
| capex | 22 | 6 |
| net_income_common | 21 | 9 |
| pretax_income | 20 | 7 |
| revenue | 1 | 1 |

`d_and_a` mezcla en 79 de 147 tickers, mas de la mitad del universo. La causa
es distinta a la de revenue y mas simple: **sus sinonimos no son
equivalentes**. La lista es `DepreciationDepletionAndAmortization`,
`DepreciationAmortizationAndAccretionNet`, `DepreciationAndAmortization`,
`Depreciation` -- y `Depreciation` a secas no es lo mismo que D&A. No es
ambiguedad semantica de la empresa; es una lista de sinonimos demasiado
generosa.

Importa porque `d_and_a` alimenta el EBITDA, y eso explica el sintoma que ya
estaba medido en la seccion 12: EV/EBITDA es el multiplo con PEOR acuerdo
contra yahooquery (mediana 9,04%, p90 38,7%, solo 34% dentro del 5%). Parte de
esa divergencia se atribuia a `NormalizedEBITDA`; ahora hay una segunda causa
identificada y del lado de SEC.

RESUELTO. La evidencia decidio y resulto ser las DOS cosas a la vez.

**Podar `Depreciation`.** Sobre 337 ejercicios de las 60 empresas que publican
los dos tags, `Depreciation` es la MEDIANA del 73% de la D&A completa, y en 28
de esas 60 esta por debajo del 70% (SPGI 10%, AMGN 17%, WBD 17%, HL 0%). No es
un sinonimo, es un subconjunto. Sale de la lista. Elimina 429 de las 476
mezclas y cuesta que 17 tickers se queden sin d_and_a -- el precio correcto
segun la regla del modulo.

**Curar los 13 que quedan.** Publican dos tags de D&A completos con valores
distintos: uno es el total y el otro un componente suelto, y cual es cual
cambia por empresa. En AMGN el total es DDA (5.592 vs 887); en MCD es
`DepreciationAndAmortization` (2.199 vs 457); en VLO y WFC es
`...AndAccretionNet`.

Aca la MAGNITUD si sirve de criterio, al reves que en revenue (donde elegia el
bruto en bancos, salida #2 de la seccion 15): D&A no tiene semantica de neteo,
no existe un "D&A bruto" que exceda al titular. Y coincide con un segundo
criterio independiente, la COBERTURA trimestral -- el D&A total esta en el
estado de flujo todos los trimestres y un renglon suplementario no (AMGN 22Q vs
10Q, BLK 11 vs 0, UPS 22 vs 4, WFC 22 vs 0). Los dos criterios coinciden en los
13, y MCD tiene ademas confirmacion de yahooquery.

Mezclas de d_and_a: **476 -> 2**.

### 16.6 El defecto mas grave: el EV se calculaba con deuda PARCIAL

Validar el EV/EBITDA destapo algo peor que la D&A, y del otro lado del
cociente.

`net_debt` tenia la regla "si falta una de las dos deudas, cuenta como 0, la
empresa tagea la que tiene". **La premisa es falsa.** La API de companyfacts
DESCARTA los hechos dimensionados, y varias empresas grandes pasaron a declarar
su deuda de largo plazo solo con dimensiones. Verizon tiene 8 hechos de
`LongTermDebt` y NINGUNO desde 2025; AT&T igual.

No es la lista de sinonimos: 31 de esos 38 tickers publican `LongTermDebt`, que
ya estaba incluida. Es la fuente la que no lo entrega.

El resultado era un EV construido con la deuda corta sola:

| ticker | net_debt calculado | real aproximado |
|---|---|---|
| VZ | 19.479 MM | ~150.000 MM |
| T | NEGATIVO (caja neta) | ~120.000 MM |

52 de 147 tickers sin `debt_long` y 41.820 filas con deuda neta negativa, todo
propagado al EV y al EV/EBITDA sin un solo aviso.

El arreglo es la misma regla que el modulo ya usa para los minoritarios del
resultado neto: `enriquecer()` mira la serie ENTERA y, si la empresa tagea esa
deuda en algun periodo pero no en este, **no asume cero** -- `net_debt` queda en
None y el EV desaparece. Una empresa realmente sin deuda de largo plazo nunca
tagea el concepto, asi que conserva su EV. Filas con deuda neta negativa:
41.820 -> 22.999.

### 16.7 Estado final del EV/EBITDA

Acuerdo contra yahooquery al ultimo punto de cada fuente:

| | n | mediana | dentro del 5% |
|---|---|---|---|
| EV/EBITDA antes | 74 | 9,24% | 34% |
| EV/EBITDA ahora | 48 | 6,38% | 46% |
| P/S (tras curar revenue) | 139 | 0,00% | 94% |
| PER | 113 | 1,75% | 78% |

La cobertura baja de 78 a 50 tickers y ese es el precio buscado: lo que queda
sale de insumos completos.

**Lo que NO se cierra, y hay que leerlo como diferencia de definicion y no como
defecto:** yahooquery usa `NormalizedEBITDA` (excluye one-offs) y SEC usa
EBIT+D&A. Y queda un hueco de cobertura real en los streamers -- la
amortizacion de CONTENIDO no esta en los tags de D&A, por eso NFLX (ratio de
EBITDA 0,42 contra yahooquery) y WBD (0,22) siguen lejos.

**Conclusion para el consumidor:** el EV/EBITDA de SEC es internamente
consistente y sirve para "caro vs si misma", que es el objetivo del proyecto.
NO es intercambiable con el de yahooquery, y no deberia compararse entre
fuentes.

### 16.6 Como se opera esto

```
python scripts/manual/sec_avisos.py --defectos --detalle
python scripts/oneshot/revenue_tags_reporte.py --tickers XXX
```

El primero es el UNICO lector de `fundamentales_sec_avisos`, que hasta esta
fecha no tenia ninguno. Ordena por SEVERIDAD y no por volumen -- DEFECTO /
HUECO / SOSPECHA / info -- porque ordenar por cantidad pone arriba lo
informativo. Un tipo de aviso nuevo cae en "info" hasta que alguien lo
clasifique, para que no se haga pasar por defecto solo. `--alertar` manda a
Telegram unicamente los DEFECTOS: un canal que se usa para lo informativo deja
de leerse.

---

## 17. La calculadora de escenarios (29/8/2026)

Es el consumidor para el que se construyo toda la fuente, y conviene decir
primero lo que **no** es: no predice retornos, no dice si algo va a subir y no
emite una recomendacion. Toma una tesis de precio que ya trae el usuario y la
traduce a lo unico que se puede contestar con datos -- que implica ese precio,
y que tendria que pasar en el negocio para sostenerlo.

Motor puro en `src/utils/valuacion_implicita.py`; CLI de solo lectura en
`scripts/manual/valuacion_implicita.py`.

### 17.1 Las dos direcciones

**DIRECTA** -- se mueve el precio, el negocio queda quieto:

> "Si AAPL valiera 377,50 (+20%), su PER seria 43,00 -- percentil 100 de su
> propia historia: nunca estuvo ahi."

**INVERSA sobre el precio** -- se fija el multiplo, se despeja el precio:

> "Para que su PER vuelva a la mediana de su historia, el precio tendria que
> ser 265."

**INVERSA sobre el negocio** -- se fija el precio Y el multiplo, se despeja el
fundamental. Es la que convierte un numero en un juicio:

> "Para que 377,50 sea un P/S normal para AAPL, las ventas tienen que crecer
> +59,4%. Su mejor anio fue +28,6%."

### 17.2 Por que hace falta la tercera

Un +20% de precio mueve **todos** los multiplos de precio exactamente +20%.
Eso es aritmetica, no informacion: listar cuatro multiplos implicitos es
listar cuatro veces la misma variacion. Lo que informa es contra que se los
compara.

De ahi salen las dos anclas, y ninguna es una prediccion:

1. **El percentil** ubica el multiplo implicito dentro del rango que esa
   empresa efectivamente tuvo. Es toda la razon por la que valia la pena
   construir la historia SEC: sin ella, "PER 43" no se puede juzgar.
2. **El crecimiento requerido contra el historico** traduce el multiplo a una
   exigencia sobre el negocio y la contrasta con lo que la empresa logro
   alguna vez.

Cuando el crecimiento necesario supera el maximo historico, el CLI lo dice
explicitamente y aclara que **no invalida la tesis**: dice que se apoya en algo
que esa empresa todavia no hizo, y que eso merece argumento aparte.

### 17.3 Lo que el backtest midio, y lo que NO

En la seccion anterior se corrio un backtest del percentil por quintiles y dio
que **no ordena el retorno futuro** (spread barato-menos-caro sin signo estable
en ninguno de los cuatro multiplos). Ese resultado esta bien medido y hay que
tenerlo presente, pero contesta **otra pregunta**: si el percentil sirve como
senal de timing autonoma.

Esta herramienta no lo usa asi. Lo usa como **regla de plausibilidad** de una
tesis que el usuario ya trae. Un percentil que no predice puede describir
perfectamente bien donde estuvo un multiplo, que es todo lo que se le pide
aca. Confundir las dos cosas -- y dar el backtest por veredicto sobre la
calculadora -- es un error de lectura, no un hallazgo.

### 17.4 Las reglas del calculo

1. **UNA sola incognita por escenario.** La direccion directa mueve el precio y
   congela los TTM y la deuda neta; la inversa congela el precio y despeja el
   denominador. Un escenario que mueva las dos cosas a la vez tiene dos
   incognitas y una ecuacion: cualquier numero que devuelva es una eleccion
   disfrazada de calculo.

2. **Todo sale de agregados**, nunca de magnitudes por accion. Misma regla que
   gobierna `fundamentales_sec_multiplos_d` (sec. 13): SEC re-expresa lo "por
   accion" ante un split y `precios_diarios` tambien, pero con horizontes
   distintos.

3. **La deuda neta se resta del lado correcto.** En las metricas sobre EV el
   numerador es market cap + deuda neta. `denominador_para(BASE, "ev_ebitda",
   10, 15)` da 170 y no 150, y hay un test que lo fija: usar el market cap ahi
   es el error clasico de esta cuenta. Sin deuda conocida el EV no existe y se
   devuelve None (sec. 16.6).

4. **El crecimiento historico se mide sobre la serie TRIMESTRAL**, deduplicada
   por `period_end`, no sobre la diaria. El TTM es una escalera que solo se
   mueve cuando sale un balance: medirlo en cada rueda multiplica por ~63 las
   observaciones sin agregar una sola, inflando el n y aplastando la
   dispersion. Con la serie trimestral, AAPL tiene 19 observaciones y se dice
   cuantas son.

5. **El crecimiento porcentual exige los dos extremos positivos.** Ir de +100
   a -162 da -262%: aritmetica valida, lectura falsa. Sin esta regla el
   MAXIMO de la distribucion puede salir negativo y se imprime como "su mejor
   anio fue -121%" -- paso con CRWD, cuyo resultado fue negativo toda la
   ventana. Ahora esos pares se descartan y la vista declara cuantas
   observaciones quedaron sobre las posibles (`19/19 obs`, `8/19 obs`), que es
   lo que separa un benchmark solido de uno de dos puntos.

   La **vara es el p90, no el maximo**. El maximo de una serie que salio de una
   base chica es un rebote irrepetible, y usarlo volveria permisiva a la
   herramienta justo donde deberia dudar.

   Lo que NO se puede arreglar en el calculo: si la base es positiva pero
   minuscula, el porcentaje explota sin ser un error (LYFT saliendo de un
   resultado cercano a cero; MU, T y UBER con la mitad de sus trimestres
   descartados). Por eso la vista imprime SIEMPRE los importes absolutos al
   lado del porcentaje: son el respaldo cuando el porcentaje deja de
   significar algo.

6. **El ROE exigido es un TECHO.** Se calcula contra el patrimonio de HOY. Si
   la empresa gana mas y no reparte todo, el patrimonio crece y el ROE que hace
   falta es menor. Sirve para descartar lo imposible, no para proyectar.

7. **Un denominador negativo no produce multiplo.** Un PER con resultado
   negativo no es "barato", es otra categoria.

### 17.5 ROIC y ROTCE: ausentes a proposito

Los dos se pidieron y ninguno esta. No es un pendiente disimulado:

- **ROIC** necesita NOPAT, o sea EBIT y una tasa impositiva efectiva. Ninguno
  de los dos esta en `fundamentales_sec_multiplos_d`.
- **ROTCE** necesita patrimonio TANGIBLE, o sea goodwill e intangibles, que
  tampoco estan en la capa derivada.

Aproximarlos con lo que hay seria inventar un numero con cara de dato. Si se
los quiere, el costo es concreto: agregar `operating_income_ttm`, la tasa
efectiva y los intangibles a la capa derivada, y recomputarla.

Ademas, y esto es lo que hace que su ausencia no rompa nada: **ROE, ROIC y
ROTCE no se mueven con el precio.** Ninguno tiene el precio en su formula. No
existe una columna "ROE si sube 20%"; entran solo como chequeo de plausibilidad
del crecimiento requerido, que es exactamente el papel que cumple el ROE hoy.

### 17.6 Como se lee

```
python scripts/manual/valuacion_implicita.py AAPL --variacion 20
python scripts/manual/valuacion_implicita.py AAPL --precio 250
python scripts/manual/valuacion_implicita.py AAPL --variacion 20 --referencia 75
python scripts/manual/valuacion_implicita.py AAPL --desde 2023-01-01
```

`--referencia` elige que percentil se considera "normal" (default 50). Subirlo
a 75 pregunta "para que este precio sea caro-pero-visto", que suele ser la
comparacion honesta con una empresa en expansion.

`--desde` recorta la historia. Importa mas de lo que parece: buena parte del
rango de multiplos de 2021 viene del **regimen de tasas** y no de la empresa.
Un PER de 20 con la tasa en cero no significa lo mismo que un PER de 20 hoy, y
el percentil no sabe la diferencia. Es la limitacion de fondo de "caro contra
si misma" y esta impresa al pie de cada corrida.

---

## 18. EDGAR / sec-api.io evaluado como fuente (29/8/2026)

Evaluacion de `sec-api.io` (key en `EDGAR_API_KEY` del .env) contra la lista
real de problemas de este proyecto, no en abstracto. La pregunta era si
sirve como FUENTE UNICA. **No sirve como fuente unica, y si sirve como
complemento quirurgico.** Abajo el detalle y, sobre todo, la evidencia.

### 18.1 Que es y que endpoints se probaron

- `POST https://api.sec-api.io` -- buscador de filings (barato, JSON chico).
  Devuelve formType, periodOfReport, filedAt, accessionNo.
- `GET  https://api.sec-api.io/xbrl-to-json?accession-no=...` -- **el que
  importa**. Devuelve UN filing entero (3-5 MB) organizado por estado
  contable (`StatementsOfIncome`, `BalanceSheets`, `StatementsOfCashFlows`)
  MAS cada nota al pie como seccion propia (`DEBT`, `SEGMENTINFORMATION`,
  `REVENUE`...).

Diferencia estructural con `companyfacts`: companyfacts entrega una BOLSA
PLANA de hechos por concepto, sin decir de que estado salieron ni en que
orden. sec-api entrega el estado contable ARMADO.

### 18.2 El hallazgo central: conserva los hechos DIMENSIONADOS

Cada hecho puede traer:

```json
"segment": {"explicitMember": {"dimension": "us-gaap:StatementBusinessSegmentsAxis",
                               "$t": "f:FordCreditMember"}}
```

Esos son exactamente los hechos que la API de companyfacts DESCARTA, y son la
causa raiz del agujero de deuda documentado en 16.6. Verificado en Ford
(10-Q Q2 2026), balance al 2026-06-30:

| concepto | monto (MM) | segmento |
|---|---|---|
| DebtCurrent | 4.381 | Ford ex-Credit |
| DebtCurrent | 46.956 | **Ford Credit** |
| LongTermDebtAndCapitalLeaseObligations | 19.238 | Ford ex-Credit |
| LongTermDebtAndCapitalLeaseObligations | 90.392 | **Ford Credit** |
| | **160.967** | total |

**Ningun hecho de deuda de Ford tiene version consolidada**: los cuatro llevan
segmento. Por eso companyfacts nos daba NULL, y por eso la supresion del EV
(regla de 16.6) era correcta -- el numero que habriamos emitido no habria
estado un poco mal, habria estado mal por 2,6 veces el market cap.

### 18.3 El otro hallazgo: las dos APIs de la SEC se contradicen

`companyfacts` **atrasa**. Verificado en vivo (no era cache viejo nuestro):

| API de la SEC | ultimo 10-Q de Ford |
|---|---|
| `submissions` | Q2, periodo 2026-06-30, **filed 2026-07-28** |
| `companyfacts` | Q1, periodo 2026-03-31, filed 2026-04-30 |

Mas de un mes de atraso sobre un filing que la propia SEC ya lista. NO es un
bug nuestro. Medido sobre el universo, antiguedad del TTM en la ultima rueda:

| antiguedad | tickers |
|---|---|
| <=45 dias (fresco) | 113 (78%) |
| 46-90 (normal entre balances) | 15 (10%) |
| **>90 dias (reporto y no lo tenemos)** | **16 (11%)** |

El peor es **C con 188 dias** (le faltan DOS trimestres). `lag_dias` los
detecta a todos: el problema es visible, no silencioso. Pero es estructural
de companyfacts, y sec-api no lo tiene.

### 18.4 La evaluacion completa, problema por problema

| # | Problema (seccion donde esta) | Lo resuelve? | Evidencia |
|---|---|---|---|
| 1 | Hechos dimensionados descartados (16.6) | **SI** | 18.2 |
| 2 | Atraso de companyfacts | **SI** | 18.3 |
| 3 | DEF 14A / 8-K contaminando (16.8, caso SCHW) | **SI, por construccion** | se pide por accession: elegis el formulario |
| 4 | Point-in-time / reexpresiones | **MEJOR** | cada filing es lo que se publico ese dia; companyfacts mezcla reexpresiones |
| 5 | Ambiguedad de revenue (15, 16.2) | **PARCIAL** | 18.5 |
| 6 | Resultado neto sin taguear (14) | **PARCIAL** | la cara trae la cadena `ProfitLoss` -> minoritarios -> neto: verificable en vez de inferido |
| 7 | Q4 no se publica discreto (11) | **NO** | 18.6 |
| 8 | `Depreciation` subconjunto de D&A (16.5) | **NO** | misma semantica de tags |
| 9 | Cobertura 147 de 200 | **NO** | 18.6 |

### 18.5 Revenue: reduce la curacion, no la elimina

La cara del estado de resultados excluye los tags de notas y segmentos, que
es de donde salia buena parte de la ambiguedad. Citigroup, 10-Q Q2 2026:

```
Revenues                              24.766 MM   <- nuestro tag curado
InterestIncomeExpenseNet              17.125 MM
NoninterestIncome                      7.641 MM     17.125 + 7.641 = 24.766
InterestAndDividendIncomeOperating    37.662 MM     (el bruto: la trampa)
```

Dos conclusiones. **La curacion de C estaba bien**, confirmada por tercera
fuente independiente. Y aparece un control NUEVO que companyfacts no puede
dar: los componentes SUMAN al total, asi que la eleccion del tag es
AUDITABLE en vez de ser un juicio a ciegas.

Pero la cara sigue teniendo cuatro conceptos con cara de "revenue" y hay que
saber cual es el total. Reduce el trabajo de curacion y lo hace verificable;
no lo elimina.

### 18.6 Lo que NINGUNA fuente arregla, porque no es de la fuente

- **Q4.** El 10-K de Ford (FY2025) trae en su estado de resultados **solo
  tres periodos ANUALES** (2023, 2024, 2025). No hay Q4 discreto en ningun
  lado porque las empresas no lo publican. La desacumulacion `Q4 = FY - 9M`
  sigue siendo obligatoria.
- **Cobertura.** RIO, UL, VOD, HMY, BABA, TSM: **ninguno presenta 10-Q**.
  Todos presentan 20-F (anual) + 6-K. Es REGULATORIO, no de la API: los
  emisores privados extranjeros no tienen obligacion de XBRL trimestral. Los
  53 tickers sin fuente SEC seguiran sin ella.
- **Profundidad.** Los 10-K de AAPL figuran desde 1994, pero XBRL arranca
  ~2009 y nuestro cache ya tiene 2007+. No se gana historia.

### 18.7 Limitaciones NUEVAS que introduce

- **Se pierde la serie temporal.** companyfacts da TODA la historia de una
  empresa en UNA llamada de ~4 MB. sec-api da UN filing por llamada de 3-5 MB.
  Reconstruir 147 tickers x ~35 trimestres: **~5.000 llamadas y ~20 GB**, unas
  **40 veces** el volumen para la misma informacion.
- **El armado pasa a ser nuestro.** Cada 10-Q trae el trimestre actual Y los
  comparativos del anio anterior: hay que deduplicar y empalmar filing por
  filing. Trabajo que hoy no hacemos.
- **Dependencia de un tercero pago.** companyfacts es publica y gratis; esto
  es un proveedor con cuota y seria un punto unico de falla que no
  controlamos.

Incremental, en cambio, es trivial: ~600 llamadas al anio (~50/mes).

### 18.8 VEREDICTO, y el descubrimiento que lo relativiza todo

Resuelve 4 de 9 limpio, 2 a medias, deja 3 intactos, y a cambio multiplica
por 40 el volumen y agrega dependencia paga. **No es reemplazo. Es
complemento quirurgico para dos cosas: hechos dimensionados y frescura.**

Y lo mas importante de toda esta investigacion:

> **El mayor agujero que tenemos NO lo arregla ninguna API, porque es
> nuestro.**

Medido: **56 de 144 tickers no tienen EV en la ultima rueda** (39%). De esos:

| causa | tickers | costo de arreglarlo |
|---|---|---|
| tags de deuda que NO estan en nuestra lista de sinonimos | **33** | **gratis, sin API** |
| deuda 100% dimensionada, o sin deuda real | 23 | sec-api, o nada |

Los tags que faltan, con rendimiento medido sobre el cache propio:

```
LongTermDebtAndCapitalLeaseObligations    24 tickers  (T, KO, ORCL, PEP, UPS, HD...)
DebtLongtermAndShorttermCombinedAmount     6 tickers
ConvertibleDebtNoncurrent                  3 tickers
```

Nuestra lista es solo `["LongTermDebtNoncurrent", "LongTermDebt"]`. VZ y T
**si publican deuda consolidada** (158.150 y 136.100 MM) con un tag que no
estabamos mirando. No era la API: eramos nosotros.

**TRAMPA AL APLICARLO, no agregar los tags a la ligera:** choca con el
defecto de `_elegir` documentado en el codigo -- entre sinonimos GANA EL
MENOS PREFERIDO (`pri` grande gana). Agregar los tags al final de la lista
los haria ganarle a los actuales en todos los tickers que hoy funcionan.
Hacerlo bien exige arreglar `_elegir` primero, con su propia medicion.

### 18.9 Costo de esta evaluacion

~8 llamadas a `xbrl-to-json` (3-5 MB c/u) y ~15 al buscador de filings. La
API no expone headers de rate limit; la cuota se mira en la cuenta.

### 18.10 APLICADO (29/8/2026) -- resultado medido

Los tags se agregaron AL FRENTE de las listas de `INSTANTE`, que con el
defecto de `_elegir` es la posicion de FALLBACK (gana el que esta mas al
final). Eso permitio arreglar la deuda **sin tocar `_elegir`**, que se midio
aparte y se descarto para esta tarea: cambia 145 de 147 tickers, llena CERO
huecos y pierde 18 valores (`scripts/oneshot/medir_elegir_pri.py`).

Hizo falta una segunda vuelta. Al agregar solo los tags de `debt_long`, HD,
KO, LOW, TGT y CVS pasaron a tener deuda LARGA pero seguian sin la CORTA, y
la regla de deuda parcial (16.6) les anulaba el EV igual -- que es el
comportamiento correcto, pero mostraba que el problema se habia corrido de
columna. Se repitio la medicion sobre `debt_short` y se agregaron sus
fallbacks.

| | antes | despues |
|---|---|---|
| tickers sin EV en la ultima rueda | 56 de 144 | **29** |
| tickers con EV/EBITDA | 50 | **69** |
| filas de la serie con deuda neta | -- | 77,6% |

Validacion:

- **Consistencia interna: 0 fallas.** `EV = market_cap + net_debt` y
  `EV/EBITDA = EV / ebitda_ttm` se cumplen en las 152.054 filas.
- **Contraste con yahooquery: 80% dentro del 25%, diferencia mediana 5,0%**
  sobre 69 tickers. La cobertura subio 38% (50 -> 69) SIN degradar el
  acuerdo (antes 6,38% de mediana sobre 50).
- Los mayores desacuerdos son los ya documentados y ninguno es nuevo:
  **NFLX 22,93 vs 9,93 y WBD 26,97 vs 7,43** (amortizacion de contenido
  fuera de los tags de D&A, sec. 16.7) y LYFT (EBITDA cerca de cero).

Guardas en `tests/test_sec_xbrl.py`: cuatro tests fijan que el tag preferido
le gana al fallback y que el fallback entra cuando el preferido no esta. **Si
alguien arregla `_elegir` sin dar vuelta las listas, fallan** -- que era el
punto: convertir una mina silenciosa en un error ruidoso.

Quedan 29 sin EV: 13 sin ninguna de las dos deudas (Ford y GM entre ellos,
deuda 100% dimensionada -> solo sec-api.io), 5 sin la larga y 11 sin la
corta, mas los que genuinamente no tienen deuda.
