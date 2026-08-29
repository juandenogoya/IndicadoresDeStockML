# Arquitectura de fuentes fundamentales

Que fuente sirve para que, por que, y cual manda cuando dos dicen cosas
distintas. Escrito el 29/8/2026, despues de evaluar las cuatro contra
problemas reales (el detalle de cada hallazgo esta en
`docs/fuentes_fundamentales.md`).

Existe este documento porque la pregunta "de donde saco este numero" ya se
contesto cuatro veces con cuatro criterios distintos, y porque **mezclar
fuentes en silencio es el error mas caro de este dominio**: no rompe nada,
solo devuelve un numero equivocado que nadie revisa.

---

## 1. Las cuatro fuentes, medidas

| fuente | tipo | costo | tickers | profundidad | fortaleza UNICA |
|---|---|---|---|---|---|
| **yahooquery** | libreria | gratis (rate limit por IP) | 196 | 7 Q | cobertura no-USA, acciones en base de split ACTUAL, EBITDA normalizado, deuda total |
| **SEC companyfacts** | API publica | gratis | 147 | **34 Q** | historia profunda + point-in-time real |
| **Alpha Vantage** | API | gratis, 25/dia | 200 | -- | **fecha de anuncio** del balance |
| **EDGAR / sec-api.io** | API | **paga** | 147 (USA) | por filing | hechos **dimensionados** + frescura |

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
