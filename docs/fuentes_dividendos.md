# Calendario de dividendos: de donde se saca la fecha de PAGO

> **ESTADO AL 24/10/2026: EXPLORADO Y NO INCORPORADO. LINEA CERRADA.**
> Se midieron las cinco candidatas contra las APIs en vivo y **no se incorporo
> ninguna**: cero tablas, cero scripts, cero pasos de rutina, nada en
> `ft_cambios` (no toca decisiones de FT). El dato de la fecha de PAGO **si se
> puede tener** -- Alpha Vantage lo da, con las cuatro fechas y hasta lo ya
> declarado y no pagado -- y lo que falta no es la fuente sino el motivo: la
> unica razon fuerte para traerlos esta SIN MEDIR (los ~500 ex-dates anuales
> metidos dentro de `precios_diarios`, final de la sec. 10), y mientras no se
> mida, construir la tabla seria para mostrar una fecha linda en el dashboard.
> Lo que reabriria la linea esta en la sec. 11.
>
> Lo medido abajo sigue vigente y es lo que hay que leer antes de volver a
> proponer dividendos. Se escribio asi:

Exploracion del 24/10/2026. **MEDIDO contra las APIs en vivo y SIN NINGUN CAMBIO en el
proyecto**: no se creo ninguna tabla, ningun script, ningun paso de la rutina. La septima
fuente evaluada y la segunda que no es fundamental (ver docs/arquitectura_fuentes.md y
docs/fuentes_ibkr.md).

La pregunta: podemos tener el calendario de fechas de pago de dividendos que informan las
empresas? Se probaron las 4 que pidio el usuario (FMP, Alpha Vantage, SEC, yahooquery) mas
una quinta que el proyecto ya usa para otra cosa (Nasdaq).

## Atajos por pregunta

| Pregunta | Seccion |
|---|---|
| Quien da la fecha de PAGO, y desde cuando | 2 (Alpha Vantage) |
| Por que SEC no sirve y es definitivo | 5 |
| Por que yahooquery miente un dia en TODAS las fechas | 6.2 |
| Por que el `dividendDate` de yahooquery puede estar 3 meses en el pasado | 6.3 |
| Por que Nasdaq, que es gratis y sin cuota, cubre 26 de 199 | 7 |
| Las fuentes se contradicen? | 8 |
| Cuanto cuesta y cuanto tarda un backfill | 9 |
| Que NO se decidio | 10 |
| Que reabriria la linea | 11 |

## 0. Veredicto por servicio

Lo medido en cada uno, en una tabla. El detalle y la evidencia, en la seccion que
indica la ultima columna.

| servicio | fecha de PAGO? | historia | cobertura (de 199) | costo / cuota | el limite que decide | sec. |
|---|---|---|---|---|---|---|
| **Alpha Vantage** `DIVIDENDS` | **SI**, y las otras 3, mas lo DECLARADO y no pagado | fechas desde **2021-2022**, con agujeros ~1 de cada 15; montos hasta 1999 | todos los pedidos respondieron, ADR no-USA incluidos | clave **free 25/dia**, 1/seg, **compartida con el paso `earnings`** | las fechas no existen antes de 2021 y los **montos no estan ajustados por split** | 2 |
| **FMP** `dividends-calendar` | **SI**, las 4, mas `frequency` y `adjDividend` | **solo hacia adelante**: el rango pasado da ACCESS DENIED | **32** (45 simbolos accesibles, con ETFs; usa `GOOGL` por `GOOG`) | incluido, sin costo extra | el **plan**: sin historia por ticker, sin pasado, universo recortado | 3 |
| **SEC XBRL** `companyfacts` | **NO** | n/a | 0 (y de arranque seria 147) | ya descargado | **no hay hechos de tipo fecha**: la API entrega solo numeros. Definitivo | 5 |
| **yahooquery** `calendar_events` | SI, pero **una sola** (la proxima o la ultima) | **ninguna** (`dividend_history` = solo ex-date y monto) | **199**, la unica que cubre todo | 1 pedido para los 199, dentro del lock de yfinance | **se lee mal por defecto**: todas las fechas corridas un dia, y los dos campos se desincronizan | 6 |
| **Nasdaq** `calendar/dividends` | **SI**, las 4, pasado Y futuro, 0 filas sin pago | completa para lo que cubre | **26**, todos NasdaqGS; **ningun NYSE** en 153 dias | **gratis, keyless, SIN CUOTA** | el **alcance**: solo lo listado en Nasdaq | 7 |

Lo que ninguna tabla muestra y es el hallazgo central: **donde dos de estas
fuentes se superponen, coinciden** (sec. 8). No hay desacuerdo que arbitrar, asi
que la idea de usarlas como validador cruzado no sobrevive a la medicion. Lo que
las separa es cobertura e historia.

## 1. El tamano del problema

Universo activo: 199 tickers. Pagan dividendo (ultimo Q con `cash_dividends_paid <> 0` en
`fundamentales_cashflow_q`): **124** -- 104 USA, 11 Europa, 9 Resto. Alguna vez pagaron: 135.

Hoy el proyecto no tiene NADA de dividendos como evento: la unica columna en toda la DB que
los menciona es `fundamentales_cashflow_q.cash_dividends_paid` (el monto agregado del
trimestre, en moneda de reporte). Ninguna fecha, ningun monto por accion, ninguna tabla.

Un dividendo tiene CUATRO fechas y no son intercambiables: declaracion (cuando la empresa lo
anuncia), ex-date (desde cuando comprar ya no da derecho), record (quien figura en el
registro) y pago. La que pidio el usuario es la de PAGO; la que mueve el precio es la EX.

## 2. Alpha Vantage `DIVIDENDS` -- la unica con las cuatro fechas Y historia

Es la mas completa en contenido. Devuelve por ticker una serie con
`ex_dividend_date`, `declaration_date`, `record_date`, `payment_date`, `amount`.

Medido:

- **Trae lo DECLARADO Y NO PAGADO todavia.** MSFT al 24/10/2026: declarado el 2026-09-14,
  ex 2026-11-19, pago **2026-12-10** -- una fecha de pago futura. Eso es exactamente lo que
  se pedia.
- **Cubre los ADR no-USA, con el desfase largo incluido.** VOD: ex 2026-06-05 -> pago
  2026-07-30 (8 semanas; en una accion USA el desfase tipico es de 2 a 3 semanas). HMY
  (Sudafrica): ex 2026-10-09 -> pago 2026-10-19. Esto importa porque los 20 pagadores
  no-USA no tienen filer en SEC y en FMP no entran.
- **Las fechas arrancan en 2021-2022, no antes.** El `amount` y el `ex_dividend_date`
  llegan hasta 1999, pero `declaration_date` / `record_date` / `payment_date` vienen
  `"None"` para todo lo anterior: AAPL las tiene desde 2022-02, MSFT desde 2021-08, VOD
  desde 2020-12, HMY desde 2021-10.
- **Y con agujeros sueltos dentro del tramo bueno**, ~1 de cada 15: AAPL 2023-08-11,
  2021-08-06 y 2020-11-06 vienen sin ninguna de las tres fechas; MSFT 2021-05-19; HMY
  2021-04-15. No es un corte limpio: es queso suizo.
- **Una fila basura**: AAPL 2022-12-23 con `amount = "0.0"` y pago 2023-01-09.

### 2.1 TRAMPA: los montos NO estan ajustados por split

AAPL 2020-08-07 informa `amount = 0.82`; el trimestre siguiente, 0.205. No bajo el
dividendo: el 31/8/2020 AAPL hizo 4:1. Lo mismo en 2014-05-08 (`3.29`, antes del 7:1).
Cada monto esta en la escala de SU dia.

Es **el mismo problema que el proyecto ya resolvio para opciones** (ver CLAUDE.md, "PRECIO
DE REFERENCIA"): `precios_diarios` se corrige hacia atras por divisor, asi que un
dividendo por accion de 2020 pegado crudo contra el precio de hoy da un yield 4 veces mas
grande. La herramienta ya existe: el registro `splits_aplicados` y
`src/utils/precio_referencia.factor_escala`. No hay que inventar nada, hay que acordarse.

FMP, en cambio, entrega las dos columnas (`dividend` y `adjDividend`), y yahooquery entrega
solo la ajustada.

## 3. FMP -- la forma correcta, bloqueada por el plan

`calendar` con `endpoint="dividends-calendar"` devuelve, por rango de fechas y para todo el
mercado: `symbol`, `date` (= ex-date), `recordDate`, `paymentDate`, `declarationDate`,
`dividend`, `adjDividend`, `yield`, `frequency`. Es el mejor esquema de las cinco: tiene el
monto crudo Y el ajustado, y la frecuencia declarada (`Quarterly` / `Semi-Annual`), que es
justo el insumo del guard de cadencia.

Lo que el plan actual NO deja:

- `endpoint="dividends-company"` (la historia por ticker): **ACCESS DENIED**.
- `dividends-calendar` con un rango PASADO (`from_date=2026-08-01`): **ACCESS DENIED**.
  El rango hacia adelante si responde.
- El universo accesible esta recortado. En 4 llamadas con rangos distintos aparecieron **45
  simbolos distintos en total**, y la lista se repite: 32 son de nuestro universo (26% de
  los 124 pagadores) y 13 no (incluye ETFs: SPY, SPYG, VWO).
- Usa `GOOGL` donde nuestro universo tiene `GOOG`.

Calidad de lo que SI entrega: **identico a Alpha Vantage en las cuatro fechas** para AAPL
(ex 2026-08-10, record 08-10, pago 08-13, declaracion 07-30) y MSFT (ex 08-20, pago 09-10,
declaracion 06-10). El `declarationDate` es semanticamente correcto incluso en los casos
raros: WMT 2026-02-19 para un ex-date de agosto (WMT declara el ano entero en febrero) y
NOK 2026-02-27 (aprobacion de la asamblea). Un lector apurado los tomaria por datos viejos
y son correctos.

**Veredicto**: con plan pago seria probablemente la mejor fuente unica. Con el plan de hoy
cubre un tercio de los pagadores y solo hacia adelante.

## 4. Lo que hay en la DB hoy

Nada reutilizable. `cash_dividends_paid` es el flujo agregado del trimestre: sirve para
saber QUIEN paga (seccion 1) y para nada mas.

## 5. SEC XBRL -- NO, y es definitivo

Se barrieron **los 147 `companyfacts` del cache** (`data/sec_cache/*.json`) buscando
cualquier concepto con "ividend" y cualquier unidad que no sea un numero. Resultado: **cero
hechos de tipo fecha**. Lo que hay es todo monto:

| Concepto | En cuantos tickers |
|---|---|
| `CommonStockDividendsPerShareDeclared` | 104 |
| `CommonStockDividendsPerShareCashPaid` | 73 |
| `PaymentsOfDividendsCommonStock` | 70 |
| `PaymentsOfDividends` | 70 |
| `DividendsPayableCurrent` | 34 |
| `DividendsPayableAmountPerShare` | 20 |

Las unicas unidades "raras" en los 147 archivos son `ZAR/shares` (AU), `CAD` (PM) y
`segment` (CVS): monedas y una dimension, ninguna fecha.

Los elementos us-gaap que SI existen en la taxonomia para esto
(`DividendsPayableDateToBePaidDayMonthAndYear`, `...DateOfRecord...`,
`...DateDeclared...`) **no aparecen en ninguno**: la API de `companyfacts` entrega hechos
numericos y descarta los no numericos -- la misma razon por la que descarta los hechos
dimensionados y dejo 52 tickers sin `debt_long` (docs/fuentes_fundamentales.md sec. 16.5).

Asi que SEC contesta "cuanto se declaro por accion en el trimestre" y NUNCA "cuando se
paga". La fecha de pago vive en el texto del 8-K Item 8.01 (el comunicado de prensa), sin
etiquetar: eso es mineria de texto filing por filing, no una fuente. Es la misma conclusion
de docs/arquitectura_fuentes.md: **EDGAR es bisturi, no fuente**. Y de arranque cubriria 147
de 199 (los 53 no-USA no tienen filer).

## 6. yahooquery -- la unica que cubre los 199, y la que mas facil se lee mal

### 6.1 Que da

- `calendar_events` -> `exDividendDate` y `dividendDate` (este ultimo ES la fecha de pago,
  verificado contra Alpha Vantage en HMY y MSFT). Respondieron **199 de 199**: 154 con
  ex-date, 153 con fecha de pago.
- `summary_detail` -> `dividendRate`, `dividendYield`, `payoutRatio`, `exDividendDate`.
- `dividend_history(start=...)` -> **solo ex-date y monto** (ajustado por split). Sin fecha
  de pago, sin record, sin declaracion.

O sea: **no hay historia de fechas de pago en yahooquery**. Solo la proxima (o la ultima).

### 6.2 TRAMPA: todas las fechas vienen corridas un dia

`calendar_events` devuelve `'2026-12-09 21:00:00'` para el pago de MSFT, que Alpha Vantage y
FMP dan el **2026-12-10**. Es el epoch de medianoche UTC renderizado en hora local (UTC-3):
leido literal, **TODAS las fechas quedan un dia antes**. Misma familia que el bug de
timezones mezcladas de `yahooquery_loader` (CLAUDE.md, fix del 18/6/2026). Corregido el
desplazamiento, coincide 8/8 con Nasdaq en ex-date Y en pago.

### 6.3 TRAMPA: un solo valor por ticker, y los dos campos se desincronizan

Es **la misma trampa que `earnings_calendar`** (CLAUDE.md, "Una tabla de EVENTOS se vigila
por CADENCIA"): un valor por ticker que se sobreescribe, sin historia. De los 153 con fecha
de pago, **97 la tienen ya en el pasado**, y no hay manera de distinguir "todavia no
declararon el proximo" de "Yahoo no lo actualizo".

Peor: los dos campos no se actualizan juntos. En **7 de 153 el ex-date es POSTERIOR a la
fecha de "pago"**, que es imposible: AMX, BAC, MOS, NEM, QCOM, SCHW, TTE. El caso QCOM
medido contra Nasdaq:

| | ex-date | pago |
|---|---|---|
| Nasdaq (declarado el 2026-10-08) | 2026-12-03 | 2026-12-17 |
| yahooquery | 2026-12-03 | **2026-09-24** |

El `exDividendDate` ya avanzo al trimestre nuevo y el `dividendDate` quedo en el pago
anterior. Quien lea `dividendDate` como "el proximo pago" se lleva una fecha de **3 meses
atras**, y el par (ex, pago) sale al revez. Esos 7 son los que se pueden PROBAR; los otros
90 son indistinguibles.

## 7. Nasdaq -- gratis, sin cuota, y con la mitad del mercado que no nos sirve

No estaba en la lista, pero el proyecto ya usa esta API para `earnings_calendar`
(`scripts/refresh_earnings_calendar.py`, keyless con headers de browser), asi que valia
probar el endpoint hermano:

    GET https://api.nasdaq.com/api/calendar/dividends?date=YYYY-MM-DD

Da `dividend_Ex_Date`, `payment_Date`, `record_Date`, `announcement_Date`, `dividend_Rate`,
`indicated_Annual_Dividend`. **Sin clave, sin cuota, y funciona tanto para dias PASADOS como
FUTUROS** (a diferencia de FMP). AAPL del 2026-08-10 coincide exacto con Alpha Vantage. De
37 filas de nuestro universo, **0 sin `payment_Date`**.

Y aca el resultado que mata la opcion: barriendo **153 dias habiles** (2026-07-01 ->
2027-01-31, 3.691 filas de todo el mercado), aparecieron **26 de nuestros 199 tickers**.

No es truncamiento ni paginacion: el payload trae solo `asOf` / `headers` / `rows`, sin
`totalrecords` ni parametros de pagina. Es el ALCANCE: los 26 cubiertos son **NasdaqGS los
26**, y los 15 tickers NYSE que se probaron a mano (JPM, KO, JNJ, XOM, CVX, V, MA, PG, HD,
T, VZ, ABBV, GS, LMT, IBM) **no aparecen ni una vez en los 153 dias**. El endpoint de
dividendos sirve solo lo listado en Nasdaq -- y no es una limitacion de la API entera: en el
endpoint de EARNINGS de la misma API si aparecen nombres NYSE (BAC, BLK).

Los 26: AAPL, AMGN, ASML, AVGO, BKR, COST, CSCO, CSX, EBAY, EQIX, ERIC, GOOG, HON, KLAC,
MAR, MDLZ, META, MSFT, MU, NVDA, PEP, PYPL, QCOM, TMUS, TXN, WMT.

## 8. Se contradicen? No. El problema es cobertura, no discrepancia

Esto responde la segunda parte de la pregunta ("si existe en mas de uno, usarlos como
validador"):

| Cruce | Resultado |
|---|---|
| Alpha Vantage vs FMP (AAPL, MSFT) | identicas las 4 fechas |
| Alpha Vantage vs Nasdaq (AAPL) | identicas |
| Nasdaq vs yahooquery (8 tickers con fecha futura en ambos) | **8/8 en ex-date y 8/8 en pago** |

Donde dos fuentes se superponen, coinciden. **Entonces un validador cruzado no agrega casi
nada**: no hay desacuerdo que arbitrar. Lo que falla es distinto en cada una -- cobertura
(Nasdaq 26, FMP 32, SEC 0), profundidad de historia (yahooquery 0, AV desde 2021 con
agujeros) y frescura (yahooquery con 97 fechas pasadas y 7 incoherentes).

La validacion que SI importa no es fuente contra fuente sino **cada fuente contra si misma
en el tiempo**: una fecha de pago declarada no deberia moverse, y si se mueve hay que
saberlo. Eso pide guardar la fecha con la que se la vio, no solo la fecha.

El unico uso real de un segundo origen es como **centinela gratis**: Nasdaq no tiene cuota,
asi que puede chequear todos los dias los 26 Nasdaq-listed sin costo, y yahooquery da un
barrido de los 199 en UN pedido para detectar que alguien declaro algo nuevo (aunque no se
le pueda creer la fecha de pago).

## 9. Costo y tiempo

- **Alpha Vantage: clave free, 25 pedidos/dia** (lo dijo la propia API al cortar la prueba:
  "25 requests per day", 1 por segundo). 124 pagadores = **5 dias** de backfill reanudable,
  igual que `earnings_historico`. Y hay un detalle que no es menor: **es la MISMA clave**
  (`ALPHA_VANTAGE_API_KEY`) que ya consume el paso `earnings` de la rutina todas las noches.
  Agregar dividendos no es gratis en cuota: le saca pedidos al backfill de balances, que
  todavia no termino.
- **FMP**: sin costo extra, pero 32 de 199 y solo hacia adelante. El plan pago es la
  variable que cambia todo el analisis.
- **Nasdaq**: 1 pedido por dia habil, sin clave y sin cuota. Un ano de historia = ~250
  pedidos a 0,4 s = ~2 minutos. Barato y cubre 26.
- **yahooquery**: 1 pedido para los 199 (`calendar_events` batcheado). Entra en el lock de
  yfinance como cualquier otro consumidor.
- **SEC**: ya esta descargado y no tiene el dato.

## 10. Lo que NO se decidio

Nada. No se eligio fuente, no se diseno tabla, no se escribio script, no se toco la rutina.

Lo que cualquier implementacion va a tener que resolver, y conviene que este escrito antes:

1. **Cual de las cuatro fechas es "la fecha de la fila"**. Por la regla del proyecto ("la
   pregunta no es cual es su columna de fecha sino cual de sus fechas dice a que rueda
   pertenece el contenido", CLAUDE.md incidente `scan_fecha`), una fila de dividendo tiene
   tres candidatas y hay que elegir explicito. La que cruza con `precios_diarios` es la
   **ex-date**; la que pidio el usuario es la de **pago**; las dos tienen que estar y
   ninguna puede llamarse `fecha` a secas.
2. **Es una tabla de EVENTOS**: se vigila por CADENCIA propia, no por antiguedad, y
   `src/utils/earnings_cobertura.py` ya tiene el molde exacto (mediana de dias entre
   eventos del propio ticker, margen 15%). Entre un dividendo y el siguiente, "vieja" es lo
   correcto. Ojo que aca hay semestrales reales (VOD, RIO, UL, HMY) -- una cadencia fija
   trimestral los marcaria mal todos los trimestres.
3. **Escala de split en el monto** (sec. 2.1), con `splits_aplicados` como registro.
4. **Guardar la fecha con la que se vio el dato** (point-in-time), si se quiere poder decir
   despues "esto se sabia el dia que el bot decidio".

### Para que serviria (lo que esta exploracion NO midio)

El valor obvio es informativo: el dashboard y las infografias pueden decir "paga el X".

El valor analitico es otro y esta sin medir: **el ex-date produce una caida mecanica del
precio que `precios_diarios` tiene adentro**. El close que guarda el proyecto es crudo y
"nunca se re-ajusta por dividendos" (CLAUDE.md, Splits): entonces 124 pagadores por ~4
eventos al ano son **~500 ex-dates anuales en el universo**, cada uno un escalon que no es
un movimiento de mercado. Las estrategias de FT, los patrones de vela, los swings de
`features_estructura` y los z-scores lo estan leyendo como precio. Cuantos de esos escalones
son grandes en relacion al ATR del ticker, y si alguna senal de FT dispara por ellos, es una
pregunta medible con esta data y no se midio aca. Antes de construir la tabla para mostrar
una fecha linda en el dashboard, vale la pena saber si lo que resuelve es eso.

## 11. Que reabriria la linea

La linea queda CERRADA, no agotada: el dato existe y se sabe donde esta. Tres
cosas distintas la reabren, y conviene no confundirlas porque piden trabajo
distinto:

1. **Que la medicion de los ex-dates de arriba de un impacto real.** Es la
   razon fuerte y la unica que justificaria una tabla con historia, backfill y
   guard de cadencia. Es una medicion de solo lectura: la lista de ex-dates
   pasados se puede armar con `dividend_history` de yahooquery (los 199 en un
   pedido, ex-date y monto ajustado, que es todo lo que hace falta para medir el
   escalon) sin gastar un solo pedido de Alpha Vantage. **Esto se puede medir sin
   decidir ninguna fuente.**
2. **Que FMP pase a plan pago.** Cambia el analisis entero: con la historia por
   ticker y el rango pasado habilitados, FMP pasa a ser la mejor fuente unica
   (monto crudo Y ajustado, `frequency` declarada, todo el mercado por rango) y
   desaparecen de un saque el problema de cuota de Alpha Vantage, el de escala de
   split y el de cobertura de Nasdaq.
3. **Que se quiera el dato solo para mostrar** ("cuando paga este ticker") en el
   dashboard o una infografia. Ese caso NO necesita tabla ni backfill ni cuota:
   alcanza `calendar_events` de yahooquery, un pedido para los 199, **con las dos
   trampas de la sec. 6 corregidas** -- sumar el dia del timezone y descartar la
   fila cuando el ex-date es posterior al pago, que es la unica forma de no
   publicar una fecha de tres meses atras. Es la opcion de menor costo y la que
   mas facil sale mal si se la toma a la ligera.

Lo que NO es motivo para reabrirla: que aparezca otra API con dividendos. Ya
coinciden entre si (sec. 8); una sexta no agrega informacion, agrega otra fuente
que mantener.
