# Fuentes de datos fundamentales -- evaluacion (27/8/2026)

Registro de la investigacion sobre de donde traer los estados contables
trimestrales. NO es documentacion de codigo en produccion: es el conocimiento
que costo trabajo obtener y que no se puede derivar leyendo el repo.

Nada de lo aca descripto esta implementado salvo los arreglos del
`refresh_fundamentales.bat` (ver "Arreglos aplicados"). El prototipo del
normalizador SEC vive en `scripts/oneshot/sec_xbrl_prototipo.py` y es
material de investigacion, no un script operativo.

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
