# IBKR como fuente: que da, que no, y un plan para medirlo

> **ESTADO AL 22/10/2026: EVALUADO Y NO INCORPORADO. LINEA DORMIDA.**
> El conector queda disponible como herramienta de CONSULTA (preguntarle un
> precio, una cadena, unos greeks cuando hace falta mirar algo puntual: ahi es
> gratis y util), pero **nada de IBKR entra al pipeline** y ninguna etapa del plan
> de la sec. 6 se va a correr. Los motivos, en orden de peso: la Etapa 3 se
> contesto SIN IBKR y dio cero (`docs/forward_testing/ANALISIS_FILL.md`); la
> Etapa 1 ataca un modo de falla que nunca se observo en este proyecto (todas las
> fallas fueron de dato FALTANTE, ya cubierto por dos guards, nunca de dato
> EQUIVOCADO); la Etapa 2 seria la version rica de una familia de datos cuya
> version base nunca demostro aportar; y el costo real no son los USD 1,50/mes
> sino una sexta fuente que mantener viva en un pipeline con cuatro podredumbres
> silenciosas en el ultimo mes. El plan queda escrito tal cual por si alguna de
> esas premisas cambia.
>
> Lo medido sigue siendo valido y es lo que hay que leer antes de volver a
> proponer IBKR. Lo de abajo se escribio asi:
> El conector de Interactive Brokers quedo online y se probo contra el
> mercado en vivo. No se escribio codigo, no se creo ninguna tabla, no
> se toco ningun consumidor. Lo que sigue es el inventario de lo que
> hay y el plan en etapas para decidir, con evidencia, si vale
> incorporarlo. **Ninguna etapa esta corrida.**

Existe este documento por la misma razon que `docs/arquitectura_fuentes.md`:
la pregunta "de donde saco este numero" se contesta UNA vez y queda escrita.
IBKR es la **sexta** fuente evaluada y la primera de datos de MERCADO (las
cinco anteriores son fundamentales). Los principios de ese doc se reusan tal
cual y no se reinventan: un solo duenio por dato, una segunda fuente TAPA UN
HUECO y nunca pisa al duenio, validar antes de mezclar y guardar el veredicto,
y **es preferible un hueco visible a un numero mezclado invisible**.

| Si la pregunta es... | ir a |
|---|---|
| que devuelve el conector, con numeros reales | sec. 2 |
| por que NO resuelve el techo de opciones de abril/2027 | sec. 3 |
| conector MCP vs TWS API vs Client Portal | sec. 4 |
| por que paper NO es la prioridad | sec. 5 |
| **que cuesta y cuales son los limites de llamadas** | **sec. 9** |
| el plan, etapas y compuertas | sec. 6 |
| que decision tiene reloj y cual puede esperar | sec. 6.0 |
| donde se escriben los resultados | sec. 7 |

---

## 1. Que es y que NO es el conector

El conector es **consultivo**, igual que el MCP server propio (regla #4 del
CLAUDE.md). Sirve para preguntarle cosas al mercado desde una conversacion.
**No sirve como insumo del pipeline**: la rutina nocturna no puede depender de
un conector que vive en la sesion de chat.

Sobre ordenes: la unica herramienta de orden que expone es
`create_order_instruction`, y **no es una orden viva**. Arma una instruccion y
devuelve un deep-link para que una persona la revise y la mande desde IBKR. No
hay `submit_order` en esta instalacion, o sea que el conector esta en modo
revisar-y-aprobar. Queda escrito aparte, como criterio y no como limitacion
tecnica: **la ejecucion de una operacion financiera la hace el usuario**, no un
agente, aunque la herramienta lo permitiera.

La cuenta conectada al 22/10/2026 esta en **USD, saldo cero, sin posiciones y
sin trades**. No esta fondeada.

## 2. Lo medido el 22/10/2026 (contra el mercado en vivo)

### 2.1 Lo que SI devuelve, y es nuevo para el proyecto

| capacidad | medicion concreta | que tenemos hoy |
|---|---|---|
| **greeks por contrato** | AAPL NOV-20-26 345 call: delta 0,5398 / gamma 0,0149 / theta -0,2040 / vega 0,3872 | **nada** |
| **IV por contrato** | midpoint IV 27,36% anual, con flag de validez | **nada** |
| **open interest por contrato** | 11.121 calls en ese strike | lo calculamos del crudo de yahooquery |
| **volumen por contrato** | 1.145 contratos en 223 trades, costo total 12.465 USD | idem |
| **bid/ask con TAMANIOS** | 11,15 x 1447 / 11,40 x 388 | el crudo trae bid/ask, sin profundidad util |
| **barras intradiarias** | 30 seg a 1 mes; **1.078 barras horarias = 2026-03-13 -> 2026-10-22** en UN pedido | **nada** |
| **cadena completa** | 24 vencimientos, de 0DTE (manana) a enero 2029 (LEAPS) | la chain vigente de yahooquery |
| **futuros con expirados** | `include_expired=true` devuelve la escalera historica | `futuros_diarios` de yfinance, sin escalera |
| acciones en tiempo real | AAPL 345,57, estado REALTIME | close diario |
| HV / IV del subyacente | HV30 22,79% anual; IV subyacente 27,29% anual | HV propia en opciones derivadas |
| maximos/minimos 13/26/52s | 348,15 / 242,96 y apertura 52s | lo calculamos |
| cuenta | balances, posiciones, trades, margenes, buying power | N/A |
| extras | screeners guardados, watchlists, alertas del lado IBKR | N/A |

Las **tres primeras filas son la unica novedad real de informacion**. El resto
o ya lo tenemos, o lo podemos derivar, o es un segundo juez del mismo numero
(que tiene valor propio, ver Etapa 1).

### 2.2 Los limites DUROS, medidos

1. **No hay historico DIARIO de opciones, y no es un problema de
   suscripcion: es estructural.** Pedir barras diarias o semanales de ese
   call devuelve `No historical market data available`, y la doc oficial de
   TWS API lo dice como regla: *"End-of-day data is unavailable for options,
   FOPs, warrants, and structured products"*. **No hay nada que comprar que
   lo arregle.** (Corregido el 22/10: el primer borrador de este doc lo
   atribuyo a falta de permisos. Era una conjetura, y es falsa.)
   Lo que **SI** existe es historico INTRADIARIO de opciones: el mismo call
   devuelve 27 barras horarias de los ultimos 3 dias y, pidiendo 4 horas,
   llega hasta el 2026-08-24 (~2 meses). Viene con `delayed: 900`, o sea
   **retrasado 15 minutos**, que es lo que corresponde a OPRA sin suscripcion.
   **Y nunca trae open interest**, en ninguna granularidad.
2. **Tope de ~1.000 puntos por pedido.** `ONE_MIN` + `ONE_MONTH` y
   `ONE_HOUR` + `SIX_MONTHS` las dos rebotan con
   "Combination of period and step will provide more then allowed 1000 data
   points". Pidiendo 1.000 barras horarias entran 1.078 barras (~7,3 meses).
   Para intradiario de un minuto, 1.000 barras son **~2,5 ruedas**: un
   backfill de minuto sobre 199 tickers por este conector no existe.
3. **La barra del dia viene EN CURSO.** Con el mercado abierto, el ultimo dia
   de AAPL trajo 4.495.466 de volumen contra ~18.000.000 de una rueda normal.
   Es **la misma trampa de la barra parcial de yahooquery** que ya nos costo
   un fix (ver CLAUDE.md, "yahooquery_loader -- barra en curso") y la razon de
   la regla #10. Cualquier consumo de IBKR tiene que recortar la barra del dia
   igual que `download_batch`.

## 3. Lo que el conector NO resuelve (y es la conclusion tentadora)

**El techo de opciones sigue intacto.** `opciones_snapshot` arranca el
2026-04-18 y no hay base de 52 semanas hasta ~abril/2027; lo previo es
irrecuperable porque Yahoo solo expone la chain vigente. Es tentador leer
"IBKR tiene historico" y darlo por resuelto. **No lo resuelve, por dos
motivos independientes**, y el segundo sobrevive incluso pagando:

1. **las opciones VENCIDAS no tienen dato, punto.** La doc oficial de TWS
   API: *"Expired options, FOPs, warrants, and structured products have no
   data."* La cadena de una rueda pasada incluye contratos que ya vencieron,
   asi que **no se puede reconstruir la cadena como era**, ni siquiera en
   precio. Esto es definitivo y no depende de pagar nada;
2. el historico de un contrato **vivo** es precio y volumen, no OPEN
   INTEREST (sec. 2.2). Y todo el analisis de opciones del proyecto (muros de
   soporte/resistencia, PCR de OI por plazo) esta construido **sobre el OI**.
   Un historico de precios de opciones no reconstruye un muro de OI.

Corolario, escrito para no volver a discutirlo: **la historia de muros de OI
previa a abril/2026 no se recupera por esta via.** Lo que IBKR puede dar es
una serie NUEVA, hacia adelante, mas rica que la actual (con greeks). Eso es
la Etapa 2 y su reloj arranca el dia que se decida, no antes.

## 4. Las tres vias de acceso, y para que sirve cada una

| via | corre desatendido | profundidad | para que sirve aca |
|---|---|---|---|
| **conector MCP** (lo que hay hoy) | NO | tope ~1.000 puntos/pedido | consulta, exploracion, las mediciones de las Etapas 0, 1 y 3 |
| **TWS API / IB Gateway** (socket) | SI (Gateway es headless, sirve en la VM de Oracle) | completa | la unica via si algo de esto entra al pipeline |
| **Client Portal Web API** (REST+WS) | con friccion (re-login periodico) | completa | descartable aca: el re-login lo vuelve fragil para un cron |

Notas para cuando se llegue a codear, no antes:

- cliente Python: **`ib_async`**, continuacion mantenida de `ib_insync`
  (que quedo sin mantenimiento). **Verificar el estado del paquete al
  momento de usarlo**, no asumirlo por esta linea.
- **pacing de TWS API: 60 pedidos de historico cada 10 minutos, pero SOLO
  para barras de 30 segundos o menos.** La doc oficial aclara que para barras
  de 1 minuto y mas los limites duros **se levantaron**, y queda un throttling
  blando. Es decir: un barrido DIARIO de los 199 tickers no choca con ese
  muro; un barrido de segundos si. (Corregido el 22/10: el primer borrador
  aplicaba el limite de 60/10min a cualquier pedido y concluia "+30 minutos
  para una tabla". Vale para barras de 30 seg o menos, no para las diarias.)
  Los otros limites de esa misma pagina siguen en pie: 50 pedidos de historico
  abiertos a la vez, nada de pedidos identicos dentro de 15 segundos, y no mas
  de 6 pedidos del mismo contrato/exchange/tipo en 2 segundos.
- el lock de yfinance (`src/utils/yfinance_lock.py`) **no cubre a IBKR**: es
  otro proveedor y otro rate limit. Si convivieran, IBKR necesita su propio
  control, no el mismo lock.
- `security_type` y `exchange` son obligatorios y case-sensitive; para futuros
  y opciones de futuros, SMART no alcanza y hay que pasar el exchange nativo.

## 5. Por que paper NO es la prioridad

IBKR da una cuenta paper gratis junto a la real y **funciona con la API igual
que la real** (cambia el puerto). Tecnicamente esta disponible hoy.

El problema no es tecnico. **El proyecto no tiene un problema de ejecucion,
tiene un problema de evidencia**, y esta medido:

- ninguna salida de ninguna estrategia de FT se distingue de salir al azar
  (`ANALISIS_SALIDAS.md`);
- 0 de 166 reglas de entrada monotonas mejoran a TECH_SECTOR_v1, y la banda
  del sorteo del desempate (+16,34% a +26,30%) es 10 veces la ventaja de la
  mejor regla (`ANALISIS_ENTRADAS.md`);
- el ML v3 con features honestas no pasa la compuerta pre-registrada
  (AUC 0,5099);
- los 3 bots Alpaca se apagaron el 13/9/2026 justamente por ser **paper sin
  valor frente a FT**.

Conectar IBKR paper agrega una **cuarta capa de paper** sobre las mismas
decisiones que ya medimos, y no produce una medicion nueva. Por eso la
ejecucion en paper es la Etapa 4 y esta **cerrada con llave** detras de una
compuerta: no se abre sin una estrategia que ya le haya ganado a su control.

Esto no dice "IBKR no sirve". Dice que **su valor esta en los DATOS que hoy no
tenemos, no en la ejecucion**.

---

## 6. EL PLAN

Reglas que valen para todas las etapas:

- **Todo es de solo lectura hasta que una compuerta diga lo contrario.** Ni
  una tabla nueva, ni un consumidor, ni un paso de la rutina.
- **Cada fase se pre-registra ANTES de correr**: pregunta, datos, metricas,
  unidad independiente y **regla de lectura** (que resultado cuenta como
  "pasa"). Mismo formato que `ANALISIS_SALIDAS.md` sec. 6.
- **Cada etapa puede terminar en NO y cerrarse documentada.** Es un resultado
  esperable, no un fracaso: `zonas_precio`, `carteras_perfil` y ML v3
  cerraron asi.
- **Ningun script de estas etapas va a `ft_cambios`**: no los lee ningun bot,
  son analisis (precedente: `ft_backtesting_runner.py`).
- Las mediciones de las Etapas 0, 1 y 3 se pueden hacer **por el conector**,
  sin escribir nada en `src/`.

### 6.0 Que tiene reloj y que puede esperar

| etapa | se puede medir HOY | tiene reloj |
|---|---|---|
| 0 -- viabilidad | si | no |
| 1 -- segundo juez del precio | si | no |
| 3 -- intradiario y slippage | si (7,3 meses de barras horarias) | **la ventana horaria se corre**: lo anterior a ~marzo/2026 se pierde |
| 2 -- greeks e IV | **NO**: no hay historico | **SI. La serie arranca el dia que se decide capturar.** |
| 4 -- paper | no corresponde | no |

**La unica decision con reloj es la Etapa 2.** Todo lo demas puede esperar sin
costo. Si la respuesta a la 2 es "no por ahora", el costo de retomarla en seis
meses es tener seis meses menos de serie.

### ETAPA 0 -- Viabilidad (barata, sin pre-registro: es inventario)

- **Fase 0.1 -- Cobertura del universo.** Resolver el contract_id de los 199
  tickers de `activos` y contar cuantos resuelven con simbolo exacto y
  listado primario US. *Compuerta*: con menos de 190 resueltos, el cruce de
  precios de la Etapa 1 es parcial y **se declara el alcance** antes de
  medir.
- **Fase 0.2 -- Permisos de datos reales. RESUELTA el 22/10/2026** (sec. 9):
  las acciones US llegan en tiempo real GRATIS, las opciones llegan con 15
  minutos de retraso, y el historico diario de opciones no existe por diseno.
  *Compuerta cerrada*: la reconstruccion hacia atras de muros de OI **queda
  muerta y no se intenta** (sec. 3). Queda SOLO abierto si la cuenta esta
  clasificada como profesional, que multiplica los precios por ~20 y lo
  confirma el usuario en su cuenta.
- **Fase 0.3 -- Costo de las suscripciones faltantes. RESUELTA el 22/10/2026**
  (sec. 9): OPRA L1 cuesta **USD 1,50/mes** no-profesional. El costo dejo de
  ser una incognita del costo/beneficio.

### ETAPA 1 -- IBKR como SEGUNDO JUEZ del precio

**La mas barata y la de mayor valor defensivo.** Ataca dos fallas que ya
pasaron y costaron:

- splits sin aplicar: KLAC 10:1 y CRWD 4:1 dejaron **-12.709 USD de perdidas
  FICTICIAS** en 8 operaciones de FT;
- huecos de `precios_diarios`: el 2026-08-28 aparecieron **157 de 200 tickers
  sin close**, y se detecto de rebote por el paso 0 de las derivadas de
  opciones.

Hoy el unico juez del precio es Yahoo. Un segundo juez independiente
convierte las dos cosas en un chequeo directo.

- **Fase 1.1 -- Pre-registro.** Muestra (tickers x ruedas), tolerancia de
  diferencia, y que cuenta como discrepancia material contra que es ruido de
  redondeo o de hora de corte.
- **Fase 1.2 -- Cruce.** Close de IBKR contra `precios_diarios`, recortando
  la barra en curso (sec. 2.2, limite 3).
- **Fase 1.3 -- Lectura.** Clasificar cada divergencia en: split no aplicado,
  hueco nuestro, base distinta, o ruido. El precedente de como se clasifica y
  se guarda el veredicto es `acciones_circulacion_validacion`.
- *Compuerta para que esto entre al proyecto*: que encuentre **al menos una
  discrepancia material que el chequeo actual no detecta**. Si solo confirma
  lo que ya sabemos, el valor es cero y se cierra documentado.

### ETAPA 2 -- Greeks e IV por contrato (la unica con reloj)

**Pregunta**: delta y gamma por strike ordenan los niveles de soporte y
resistencia mejor que el open interest crudo que ya usamos?

Es la unica informacion genuinamente nueva (sec. 2.1) y **no se puede medir
hoy**: sin historico, la serie empieza el dia que se capture. La decision de
hoy no es "cual gana", es **"empezamos a guardar o no"**.

- **Fase 2.1 -- Alcance y costo de captura.** Cuantas llamadas por rueda. Una
  cadena acotada a +/-5 strikes alrededor del dinero son ~11 filas por
  vencimiento por ticker, y cada contrato necesita **su propia llamada de
  snapshot** para traer greeks. Sobre 199 tickers eso no es gratis en tiempo
  ni en pedidos: hay que acotar el alcance (los tickers que marcan las
  estrategias, o un vencimiento por plazo) y **medir el costo real antes de
  prometer una captura diaria**.
- **Fase 2.2 -- Captura EN PARALELO, sin consumidores.** El patron ya validado
  es `features_estructura` / `features_velas`: tabla nueva al lado, nadie la
  lee, ninguna decision cambia. Requiere la via TWS API (sec. 4), no el
  conector.
- **Fase 2.3 -- Medicion, a las ~60 ruedas.** Pre-registro escrito en la Fase
  2.1, antes de tener un solo dato, para que la regla de lectura no se
  acomode al resultado.
- *Compuerta de arranque* (hoy): se captura solo si la Fase 2.1 muestra un
  costo sostenible Y se acepta que el resultado llega en ~3 meses.
- *Compuerta de consumo* (en ~3 meses): los niveles por greeks le ganan a los
  muros de OI con la vara pre-registrada. Sin eso, la tabla queda en paralelo
  o se apaga, como la linea SEC.

### ETAPA 3 -- Intradiario: CERRADA el 22/10/2026 SIN usar IBKR

> **CERRADA. La pregunta se contesto con `precios_diarios` y dio cero.**
> `precios_diarios` tiene la columna `open`, asi que el primer precio obtenible
> (la apertura de la rueda siguiente) estaba disponible para el 99,4% de las
> salidas, sobre la historia COMPLETA y no sobre los 7 meses de la ventana
> horaria. Resultado: diferencia pareada **-0,002 pp, IC95 [-0,190; +0,187]**.
> No hay efecto, asi que **no hay motivo para traer barras intradiarias a buscar
> el mismo numero**. Detalle: `docs/forward_testing/ANALISIS_FILL.md`.
> Lo de abajo es el planteo original, que queda como registro.

#### Planteo original (superado)

**Pregunta**: las salidas de FT se deciden y ejecutan con el close de la
ultima rueda de DATOS disponible. Cuanto cambia el resultado si se ejecutaran
en la apertura siguiente, o a la hora en que la senal aparecio?

Es la palanca que quedo abierta: medimos QUE regla de salida usar (nada se
distingue del azar) pero nunca medimos CUANDO se ejecuta. Y es lo unico que el
intradiario de IBKR permite contestar hoy, con 7,3 meses de barras horarias ya
disponibles.

- **Fase 3.1 -- Pre-registro.** Variantes de momento de ejecucion, metrica
  (diferencia pareada contra la ejecucion actual, operacion por operacion),
  unidad independiente = la rueda (las salidas del mismo dia comparten
  mercado), y la regla de lectura.
- **Fase 3.2 -- Muestra.** Las salidas reales de `ft_operaciones` dentro de la
  ventana horaria disponible, cruzando por **`fecha_datos_salida`** y nunca
  por la fecha de registro (CLAUDE.md, "FT asincronico"). Excluyendo las
  ventanas que `ft_cambios` marca como bug.
- **Fase 3.3 -- Lectura.** Si la diferencia no se distingue del cero, la
  convencion actual **queda confirmada como correcta** y se cierra. Ese es un
  resultado util, no un no-resultado.
- *Compuerta*: una diferencia material y consistente entre tramos. Con IC95
  que incluya el cero, no se cambia nada.

### ETAPA 4 -- Ejecucion en paper (CERRADA con llave)

No se abre sin las dos cosas:

1. una estrategia de FT con ventaja medida **contra su grupo de control**, no
   contra cero ni contra el universo;
2. la Etapa 1 cerrada (si se van a mandar ordenes reales, el precio con el que
   decidimos tiene que estar auditado contra una segunda fuente).

Y aun abierta, la ejecucion la dispara el usuario (sec. 1).

## 7. Donde se escriben el trabajo y los resultados

Siguiendo la convencion ya usada en los analisis de FT:

- **crudo y tablas de cada corrida**:
  `reportes/ibkr/AAAAMMDD_<etiqueta>/` con su `parametros.json`, fuera de git.
- **los numeros que importan**: en este doc, una seccion por fase, con el
  pre-registro ANTES y el resultado DESPUES, incluido el resultado negativo.
- **el estado vivo y la decision**: `memory/AGENDA.md`.
- **si algo llega a produccion**: fila en la tabla de scripts del CLAUDE.md,
  patron critico si deja una leccion, y el reparto de fuentes de
  `docs/arquitectura_fuentes.md` ampliado a datos de mercado (hoy solo cubre
  fundamentales).

## 8. Lo que este plan NO cubre

Honestidad sobre los huecos, para que no se descubran de nuevo en tres meses:

- **La historia de muros de OI anterior a abril/2026.** No se recupera por
  esta via (sec. 3). No hay etapa que lo intente.
- **Backfill intradiario profundo.** Con el tope de ~1.000 puntos, el minuto
  por conector no pasa de ~2,5 ruedas. Por TWS API es posible pero choca con
  el pacing de 60 pedidos / 10 minutos: seria un proyecto propio, no una fase.
- **Datos de opciones de los no-USA.** No medido todavia (Fase 0.1).
- ~~El costo en dinero de las suscripciones.~~ **RESUELTO**, sec. 9.
- **Reemplazar a Yahoo como duenio del precio.** Fuera de alcance a proposito:
  `precios_diarios` es la fuente de verdad del proyecto y la Etapa 1 solo
  propone un AUDITOR, no un reemplazo. Cambiar de duenio significaria
  re-basar toda la historia, y el principio de la sec. 1 lo prohibe sin una
  validacion completa.

---

## 9. Costo y limites de llamadas (medido y verificado el 22/10/2026)

Contestado contra la pagina oficial de precios de IBKR y la doc oficial de
TWS API, no de memoria. **Los precios cambian: el numero que manda es el que
muestra la cuenta del usuario** en Market Data Subscriptions, porque depende
de su clasificacion (ver sec. 9.6).

### 9.1 La API no cuesta nada

**No hay fee de API.** IBKR documenta que la Web API esta disponible para
todos sus clientes sin costo y que no hay cargos adicionales por usar Client
Portal API. Lo mismo vale para TWS API. Lo unico que se paga son los DATOS.

Requisitos: una cuenta abierta y **activada** (no sirve una cuenta a medio
abrir), y **no hay minimo de fondeo** para tener cuenta. El fondeo recien se
necesita para pagar una suscripcion de datos.

### 9.2 El free tier, y es mas generoso de lo que esperaba

Incluido con cualquier cuenta, sin pagar nada:

| incluido gratis | alcance |
|---|---|
| **streaming en tiempo real de TODAS las acciones y ETF listados en US** | fuente Cboe One + IEX, **no consolidado** |
| datos **demorados** | el resto de los productos, donde haya |
| **100 snapshots por mes** | franquicia de USD 1,00/mes en snapshots |

Esto explica una cosa que no cerraba: la cuenta tiene **saldo cero** y aun asi
AAPL volvio con estado `REALTIME`. No es un misterio ni una prueba gratis: el
tiempo real de acciones US viene incluido. Y explica el otro lado: las
opciones volvieron con `delayed: 900` porque OPRA demorado son exactamente 15
minutos.

**Lo "no consolidado" importa y no es un detalle.** Cboe One + IEX no es el
NBBO completo: es un subconjunto del volumen. Para el cierre diario que usa el
proyecto da igual (y de hecho el duenio del precio sigue siendo
`precios_diarios`), pero **para un bid/ask intradiario no es el mismo numero
que ve el mercado**. Si la Etapa 3 llegara a mirar spreads, hay que declararlo.

### 9.3 Lo que habria que pagar, por etapa del plan

Precios **no-profesionales**, mensuales:

| suscripcion | no-pro | pro | se bonifica? |
|---|---|---|---|
| **OPRA Top of Book (L1), opciones US** | **USD 1,50** | USD 32,75 | si, con USD 20 de comisiones |
| US Equity and Options Add-On Streaming Bundle | USD 4,50 | USD 125,00 | **no** |
| US Securities Snapshot and Futures Value Bundle | USD 10,00 base | -- | si, con USD 30 de comisiones |
| Cboe One (L1) | USD 1,00 | USD 5,00 | -- |
| US Futures Value Bundle PLUS | USD 5,00 | -- | **no** |

**Para el plan de la sec. 6 la cuenta es corta:**

- **Etapa 1 (segundo juez del precio): USD 0.** El tiempo real de acciones US
  y el historico diario ya estan incluidos. **No hay que pagar nada para
  correrla.**
- **Etapa 3 (intradiario): USD 0.** Las barras horarias de acciones salen del
  mismo dato incluido.
- **Etapa 2 (greeks e IV): USD 1,50/mes.** Es OPRA L1. Si la cuenta llegara a
  generar USD 20 de comisiones en un mes, ni eso.

O sea que **el costo en dinero no es el argumento contra ninguna etapa.** El
costo real es de tiempo de desarrollo e infraestructura (IB Gateway corriendo
en algun lado), no de suscripcion.

### 9.4 La trampa que NO cayo: el costo de los snapshots tiene techo

Los snapshots por pedido cuestan **USD 0,01** en acciones y ETF US y **USD
0,03** en todo lo demas, con la franquicia de USD 1,00/mes.

Hice la cuenta de la Etapa 2 por esta via y daba para asustarse: 199 tickers x
11 strikes x 2 (call y put) = ~4.400 snapshots por rueda a USD 0,03 son ~USD
130 **por dia**. Pero la letra chica de la pagina oficial lo desarma:

> *"If a client's snapshot quote fees within a month equal the cost of the
> real-time streaming service, they will automatically be upgraded to the
> real-time streaming service for that month."*

**El gasto en snapshots esta topeado en el precio de la suscripcion de
streaming**: al llegar a USD 1,50 de snapshots de opciones, la cuenta pasa
sola a OPRA streaming por ese mes. La cuenta de USD 130/dia no existe. Queda
escrito porque es el tipo de numero que, mal calculado, mata un proyecto en
una reunion.

### 9.5 Los limites NO son cuotas diarias ni mensuales

Esto es lo que mas conviene tener claro, porque la pregunta "cuantas llamadas
por dia" no tiene respuesta: **IBKR no pone cupos diarios ni mensuales.** Pone
limites de VELOCIDAD y de CONCURRENCIA. No hay un contador que se agote.

**TWS API** (doc oficial, la que usariamos si algo entra al pipeline):

| limite | valor |
|---|---|
| pedidos de historico abiertos a la vez | 50 |
| pedidos de historico por 10 minutos | 60, **solo para barras de <= 30 seg** |
| barras de 1 minuto y mas | limite duro **levantado**, queda throttling blando |
| pedidos identicos | no dentro de 15 segundos |
| mismo contrato/exchange/tipo | no mas de 6 en 2 segundos |
| BID_ASK | **cuenta doble** |

**Web API / Client Portal**: 50 pedidos por segundo por sesion autenticada (10
por segundo si se usa el CP Gateway), con topes por endpoint (el de historico
10/seg o 50/min, el de snapshot 10/seg, varios de cuenta 1 cada 5 seg).
Pasarse devuelve HTTP 429 y la IP puede quedar en penalizacion 10 minutos.

**El limite que de verdad aprieta son las LINEAS de market data**: toda cuenta
arranca con **100 lineas concurrentes** de tiempo real, compartidas entre TWS
y la API, y suben con comisiones, con patrimonio o comprando booster packs.
Con 199 tickers en el universo, **un streaming simultaneo de todos no entra en
las 100 lineas**. Para el proyecto no es un problema real porque lo que
necesita son fotos al cierre y no un stream permanente, pero cualquier diseno
que piense en "suscribir el universo" choca con esto primero, antes que con
cualquier otro limite.

### 9.6 Lo unico que queda por confirmar, y lo tiene que ver el usuario

**Profesional vs no-profesional.** La diferencia no es menor: OPRA pasa de USD
1,50 a USD 32,75 y el add-on de streaming de USD 4,50 a USD 125,00. IBKR
define no-profesional como persona fisica que no esta registrada ante la SEC /
CFTC / un organismo o mercado, que no actua como asesor de inversiones en el
sentido de la Investment Advisers Act de 1940, y que no esta empleada en
funciones que exigirian ese registro. **Queda para que lo confirme el usuario
en su cuenta**, porque es una declaracion suya y no algo que se pueda medir
desde aca.

Las dos notas de la pagina oficial que conviene no olvidar: **las
suscripciones no se prorratean** (suscribirse el dia 28 paga el mes entero), y
se puede suscribir data de productos que la cuenta no tiene permiso de operar,
pagando igual.
