# Arquitectura de fuentes fundamentales

> **ESTADO AL 30/8/2026: esta linea esta DORMIDA.** Construida,
> medida y mergeada a `main`, pero con el paso diario APAGADO y sin
> consumidores en uso. El veredicto, lo que quedo sin hacer y cuanto
> valia estan en `docs/fuentes_fundamentales.md`, seccion de cierre.
> Lo de abajo sigue siendo cierto y es lo que hay que leer antes de
> tocar cualquier cosa de fundamentales.

Que fuente sirve para que, por que, y cual manda cuando dos dicen cosas
distintas. Escrito el 29/8/2026, despues de evaluar las cuatro contra
problemas reales (el detalle de cada hallazgo esta en
`docs/fuentes_fundamentales.md`). **Ampliado el 30/8/2026 con Polygon**, que
entro como quinta fuente al resolver las acciones en circulacion (seccion 7).

Existe este documento porque la pregunta "de donde saco este numero" ya se
contesto cuatro veces con cuatro criterios distintos, y porque **mezclar
fuentes en silencio es el error mas caro de este dominio**: no rompe nada,
solo devuelve un numero equivocado que nadie revisa.

---

## 1. Las fuentes, medidas

| fuente | tipo | costo | tickers | profundidad | fortaleza UNICA |
|---|---|---|---|---|---|
| **yahooquery** | libreria | gratis (rate limit por IP) | 196 | 7 Q | cobertura no-USA, acciones en base de split ACTUAL, EBITDA normalizado, deuda total |
| **SEC companyfacts** | API publica | gratis | 147 | **34 Q** | historia profunda + point-in-time real |
| **Alpha Vantage** | API | gratis, 25/dia | 200 | -- | **fecha de anuncio** del balance |
| **EDGAR / sec-api.io** | API | **paga** | 147 (USA) | por filing | hechos **dimensionados** + frescura |
| **Polygon** | API | gratis, 5/min | **200** | por fecha | **splits autoritativos** + conteo TOTAL de multiclase |

Fuentes EVALUADAS Y DESCARTADAS para acciones, para no volver a probarlas:
**FMP** (el plan tope `limit` en ~5: solo 4 trimestres), **Alpha Vantage**
(TSLA oscila 0,8964..1,0970 contra yahoo, y fabrica historia PRE-IPO para
V/TSLA/META), **SimFin** (valida en 5 de 41 tickers; en HSY se desvia 2x
contra el 10-K de la propia empresa). SimFin sigue siendo candidata para
DEUDA, que es otro problema: recupera 18 de los 29 tickers sin EV.

Universo: 200 tickers (`activos`). **53 no tienen ni podran tener fuente
SEC**: son emisores privados extranjeros, presentan 20-F anual y 6-K, sin
obligacion de XBRL trimestral. Eso es regulatorio y ninguna API lo cambia.

---

## 2. El principio que ordena todo

> **Un solo duenio por dato.** Una segunda fuente puede TAPAR UN HUECO, nunca
> pisar al duenio. Y cuando dos tienen que convivir en una misma serie, se
> VALIDA que coincidan antes de mezclarlas, con el veredicto guardado.

El precedente es `src/utils/acciones_series.py`: yahooquery es la base
(unica en base de split actual, apareable con `precios_diarios`), SEC solo
EXTIENDE hacia atras, y solo en los tickers donde se verifico que no cambiaron
de base. El veredicto por ticker vive en `acciones_circulacion_validacion`.
Ese patron se replica, no se reinventa.

Corolario que ya nos costo caro dos veces (D&A parcial, EV con deuda
parcial): **es preferible un hueco visible a un numero mezclado invisible.**

---

## 3. Reparto por EJE, no por metrica

La division no es "PER de aca, P/S de alla". Es por el eje de analisis, y
cada eje tiene una fuente natural porque sus limitaciones son distintas.

### Eje TEMPORAL -- "caro contra si misma" -> **SEC companyfacts**

Historia larga, point-in-time, gratis. 147 tickers USA, mediana 34
trimestres. Es lo unico que permite un percentil trailing de 756 ruedas.

Alimenta: `fundamentales_sec_q` -> `fundamentales_sec_multiplos_d` -> la
calculadora de escenarios (`docs/fuentes_fundamentales.md` sec. 17).

Sus limites, conocidos y medidos: atrasa hasta meses (11% del universo con
TTM >90 dias), descarta hechos dimensionados, y no llega a los 53 no-USA.

### Eje TRANSVERSAL -- "vs sus pares hoy" -> **yahooquery**

196 tickers, incluidos los 53 que SEC no alcanza. Poco fondo (7 Q) pero
ancho, que es exactamente lo que pide una comparacion sectorial de un
snapshot.

Alimenta: `fundamentales_income_q` / `_balance_q` / `_cashflow_q` /
`_valuation_q` -> `fundamentales_ratios_q` -> `fundamentales_ticker_vs_sector`.

Ademas es DUENIO de tres datos que nadie mas da bien:
- **acciones en circulacion** en base de split actual (`acciones_circulacion`);
- **EBITDA normalizado** (SEC solo puede dar EBIT+D&A, sin quitar one-offs);
- **deuda total** consolidada (`fundamentales_balance_q.total_debt`), 199
  tickers -- pero solo desde **2024-09-30**.

### Eje de EVENTOS -- "cuando reporto" -> **Alpha Vantage**

`earnings_historico`: fecha de anuncio por trimestre y si fue pre o post
market. Ningun otro la tiene -- `earnings_calendar` solo trae la proxima y
los fundamentales traen el CIERRE fiscal, no el anuncio.

Ya fue **rechazada como arbitro de revenue** con evidencia (WFC 125.397 en
2024 vs 77.198 en 2023). No reintentarlo.

### Parches puntuales -> **EDGAR / sec-api.io**

NO es fuente. Es bisturi, para las dos unicas cosas donde es superior:
hechos dimensionados y frescura. Backfill completo esta descartado: ~5.000
llamadas y ~20 GB para la misma informacion que companyfacts da en 147
llamadas (`docs/fuentes_fundamentales.md` sec. 18.7).

---

## 4. La deuda neta: la decision abierta, cuantificada

Es el agujero mas grande que tenemos hoy: **56 de 144 tickers sin EV en la
ultima rueda** (39%), y por lo tanto sin EV/EBITDA.

Hay tres candidatos y **no compiten: cubren cosas distintas**.

| opcion | tickers | historia | costo |
|---|---|---|---|
| **A. Arreglar nuestros tags** (SEC) | 33 de 56 | **completa** | gratis |
| **B. Deuda de yahooquery** | ~199 | **solo desde 2024-09** | gratis, ya cargada |
| **C. EDGAR dimensionado** | el residuo (~20) | completa | paga + pesado |

La distincion que decide: **EV de HOY vs EV HISTORICO.**

- El **percentil** ("caro vs si misma") necesita 756 ruedas de historia. La
  opcion B **no puede** sostenerlo: 8 trimestres no alcanzan.
- El **multiplo actual** solo necesita la deuda de hoy. Ahi B alcanza, es
  gratis y ya esta en la DB.

**Orden recomendado:** A primero (mayor rendimiento, costo cero, historia
completa) -> re-medir -> recien ahi decidir si el residuo justifica B para el
dato de hoy, C para su historia, o nada.

**A no es agregar tres strings.** Choca con el defecto de `_elegir`
documentado en `src/utils/sec_xbrl.py`: entre sinonimos gana el MENOS
preferido. Agregar los tags al final los haria ganarle a los actuales en
todos los tickers que hoy funcionan. Va con medicion previa.

---

## 5. Reglas de convivencia

1. **Cada tabla declara su fuente.** `fundamentales_sec_q` tiene `origen`
   JSONB con el tag que produjo cada numero. Toda mezcla nueva lleva rastro
   equivalente.
2. **No se comparan multiplos entre fuentes.** El EV/EBITDA de SEC
   (EBIT+D&A) y el de yahooquery (NormalizedEBITDA) **no son
   intercambiables** -- medido, sec. 16.7. Cada uno sirve contra su propia
   historia.
3. **La reconciliacion anual es control de CONSISTENCIA, no de correccion.**
   Un cambio que la mejora puede estar empeorando la verdad. Ya paso.
4. **Antes de extender una serie con otra fuente, validar y guardar el
   veredicto** (patron `acciones_circulacion_validacion`).
5. **La frescura se declara, no se asume.** `lag_dias` viaja en
   `fundamentales_sec_multiplos_d` y se imprime en toda vista. Fue lo que
   detecto el atraso de Ford.

---

## 6. Que NO tiene duenio todavia

Honestidad sobre los huecos, para que no se descubran de nuevo dentro de
tres meses:

- **EV/EBITDA historico** de los ~23 tickers con deuda dimensionada (F, GM,
  CAT, CVX, DE, MCD, TMUS...). Hoy: sin EV.
- **Los 53 no-USA** no tienen serie temporal profunda y no la van a tener.
  Para ellos solo existe el eje transversal.
- **ROIC y ROTCE** en la capa derivada SEC: faltan NOPAT (tasa efectiva) e
  intangibles (sec. 17.5).
- **Amortizacion de contenido** en streamers (NFLX/WBD): no esta en los tags
  de D&A, asi que su EBITDA queda mal (sec. 16.7).
- **16 tickers con TTM rancio** (>90 dias), C el peor con 188.

---

## 7. Acciones en circulacion: como quedo resuelto (30/8/2026)

El problema medido: **11.491 ruedas (7,6%) sin market cap en 43 tickers**,
hasta el 41% de la historia de uno solo. Sin market cap no hay PER, ni P/B,
ni P/S, ni EV. Quedo en **2.092 ruedas, -81,8%**.

### La cascada de fuentes

`acciones_circulacion` se arma probando candidatos EN ORDEN y quedandose con
el primero que **valida contra yahooquery**. Nunca se elige a ciegas, porque
se probo y no funciona: elegir por preferencia deja a TRIP y LYFT con 4-6
puntos que arrancan en 2025; elegir por cobertura rompe UPST y SNOW, que ya
andaban. El unico que sabe cual sirve es el validador.

| orden | fuente | que es |
|---|---|---|
| base | **yahooquery** | `OrdinarySharesNumber`, base de split ACTUAL, apareable con `precios_diarios` |
| 1 | SEC **portada** | `dei:EntityCommonStockSharesOutstanding`, NO se re-expresa |
| 2 | SEC **balance** | `us-gaap:CommonStockSharesOutstanding`, SI se re-expresa |
| 3 | SEC **promedio diluido** | otra magnitud; ultimo recurso de SEC |
| 4 | **Polygon** | `weighted_shares_outstanding`; unico que cubre multiclase |

Reparto real hoy: yahooquery 1.575 puntos, portada 889 (80 rebasados),
balance 60, promedio 80, Polygon 6.

### Las cuatro cosas que costo aprender

**1. La caida dimensional golpea dos veces.** `companyfacts` descarta todo
hecho con dimensiones. Eso explica la deuda faltante de F *y* que los 21
filers multiclase no tengan portada. En V (Visa) es total: los TRES niveles
vienen vacios. No es que el nivel elegido sea malo -- no hay ninguno.

**2. La base de split se corrige, y el corte es `filed`.** Las fuentes
point-in-time publican la base de SU momento; `precios_diarios` esta en la de
hoy. Hay que rebasar. Pero el corte **no** es la fecha de periodo: un numero
esta en la base vigente cuando se PRESENTO. GOOG declara 658.763.000 acciones
al 2022-03-31 (10-Q de abril) y 13.078.000.000 al 2022-06-30 (10-Q de julio,
ya con el split del 18/7 adentro). Las dos fechas de periodo son previas al
split y solo una esta re-expresada. Cortando por fecha se rebasan las dos,
queda un salto de x19,85 y la serie se rechaza entera.

**3. Polygon mezcla splits con spinoffs en el mismo endpoint.** HON 1,061 es
la escision de Solstice, IBM 1,046 Kyndryl, MMM 1,196 Solventum, DELL 1,973
VMware, GSK 0,8 Haleon. Son ajustes de PRECIO: no mueven el conteo del padre.
Aplicarlos corrompe la serie (HON empeoro de 0,500 a 0,471). **Filtrar por
"ratio plausible" NO alcanza**: BBD 1,1 e ITUB 1,03 son bonificaciones
brasileras que SI cambian el conteo. Como no hay forma confiable de clasificar
la accion corporativa por el ratio, no se clasifica: se prueba la serie con y
sin rebase y gana la que valida. Se rebasaron 12 tickers, todos splits reales
(AMZN AVGO CRWD CSX FTNT GOOG KLAC NFLX NOW NVDA TSLA WMT); HON, IBM, MMM,
DELL y GSK quedaron afuera solos.

**4. `share_class_shares_outstanding` != `weighted_shares_outstanding`.** El
primero es UNA clase. En V da 1.635 MM contra 2.079 MM de yahoo (0,79) --
justo el error que la tabla viene a evitar. El ponderado cruza en 1,0136 y
1,0277 en las dos fechas comunes.

### Tolerancia de base: 12%, medida

`TOL_BASE` separa "deriva ordinaria" de "otra base". Barrido sobre los 200:

| tol | extendidos | |
|---|---|---|
| 0,10 | 135 | |
| **0,12** | **137** | +TRIP (10,2%), +RKLB (11,2%); ninguno perdido |
| 0,15 | 137 | nada nuevo |
| 0,20 | 137 | nada nuevo |

Hay una **banda vacia entre 11,2% y 20%**: los tickers caen o en deriva o muy
lejos, nunca en el medio. Por eso el valor no es un filo de navaja, y por eso
subirlo mas no compra nada. Lo que falta despues de 0,12 no falla por
tolerancia (HON esta 50% afuera, V no tenia con que solapar).

**Riesgo que este numero NO cierra**, dicho explicito: la guarda nunca
protegio contra tomar una CLASE en vez del total. Con 0,10 pasaba una clase
que pesara ~10%; con 0,12, una que pese ~12%. Lo cierra la FUENTE (tags sin
dimension, `weighted_shares`), no la tolerancia.

### Como se verifica que no se rompio nada

La comprobacion no es opcional: la primera version de este arreglo corrompio
AVGO (empalme x9,72) y WMT (x2,99) **en silencio**, y solo aparecio porque se
corrio una pasada de verificacion. El control es contar los saltos >25% entre
puntos consecutivos y mirar de que fuente sale cada lado:

```sql
WITH s AS (SELECT ticker, fecha, shares, fuente,
                  LAG(shares) OVER (PARTITION BY ticker ORDER BY fecha) prev,
                  LAG(fuente) OVER (PARTITION BY ticker ORDER BY fecha) prevfu
           FROM acciones_circulacion)
SELECT ticker, fecha, prevfu, fuente, shares/NULLIF(prev,0) ratio
FROM s WHERE prev IS NOT NULL
  AND (shares/NULLIF(prev,0) > 1.25 OR shares/NULLIF(prev,0) < 0.80);
```

Hoy da **14 saltos, y CERO tocan un punto rebasado**: todos son
yahooquery<->yahooquery o portada<->portada, y son fusiones reales
(AVAV/BlueHalo, PAAS/Yamana, NEM/Newcrest, AA/Alumina, SPGI/IHS Markit,
AMD/Xilinx). Si aparece uno nuevo que toque `_rb`, es un ratio espurio.

### Lo que queda, y por que no es de acciones

| ticker | ruedas | causa |
|---|---|---|
| HON | 465 | separacion real de 2026: el conteo se parte al medio. No es un split |
| SNAP / HSY | 213 c/u | ninguna fuente publica mas atras |
| RKLB | 177 | idem, ya recupero 288 de 465 |
| AI | 107 | idem |
| resto | ~30 c/u | colas anteriores al primer dato disponible |

### Operacion

`scripts/refresh_polygon.py` es REANUDABLE (`polygon_ingesta` marca ticker y
tarea, y distingue "no pedido" de "pedido, sin resultados") y respeta el cupo
con una ventana deslizante de 4/min. Las tres tareas: `--cobertura` (existe el
ticker), `--splits`, `--acciones` (serie por fecha, la mas cara: 1 pedido POR
FECHA). Los 200 tickers existen en Polygon, incluidos los 53 sin SEC -- pero
existir da acciones y splits, NO el denominador fundamental: esos emisores
presentan 20-F anual y eso no lo cambia ninguna API.

Cuidado de campo: el reloj de `requests` **no corre mientras la maquina esta
suspendida**. Una corrida quedo 8 horas congelada en dos tramos, y al despertar
pasaron 4 tickers en el mismo segundo porque la ventana de caudal habia
envejecido. El cliente usa `TIMEOUT = (conectar, leer)` para que el socket
muerto falle rapido, pero eso no evita las horas dormidas: si se deja corriendo
sola, conviene desactivar la suspension.
