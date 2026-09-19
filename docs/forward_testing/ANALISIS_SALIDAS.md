# Analisis de salidas de las estrategias FT

**Estado (19/9/2026)**: etapa EXPLORATORIA hecha (panorama de todas las estrategias y
anatomia de TECH_SECTOR_v1). Paso 1 de TECH_SECTOR_v1 pre-registrado (sec. 6) y CORRIDO
(sec. 7): **ninguna variante pasa**. Sacar la SMA21 de la salida no mejora el promedio y
empeora la cola. Paso 2 (sec. 8): grilla de pesos, 589 reglas, midiendo lo que hace el
precio DESPUES de salir: **0 de 588 reglas son candidatas**; los pesos no eligen el
momento de salir, eligen cuanto se queda la posicion (y con eso la cola). La v1 sigue
como esta.
SMC_v1 (sec. 10), mismo metodo, 96 combinaciones de stop, CHoCH, estructura rota y time
stop: **ninguna se confirma**; CHoCH bajista salio primero 1 vez en 920 (el stop va antes) y
el time stop mas corto (10 dias) es la unica palanca con direccion repetida, sin alcanzar la
vara. **Correccion** (sec. 7.1): `earnings_historico` esta atrasada desde el 21/7/2026; los
numeros del FT de control de TECH_SECTOR_v1 quedan sin respaldo.
**Resultados**: `reportes/analisis_salidas/20260919_exploratorio/` y
`reportes/analisis_salidas/20260919_tech_sector_v1_p1/`,
`reportes/analisis_salidas/20260919_tech_sector_v1_p2/` (log por seccion, CSV por salida
y `parametros.json` con commit, rueda de datos y umbrales). Ese directorio
esta fuera de git como todo `reportes/`: los numeros que importan estan en este doc.
**Reproducir**: `python scripts/forward_testing/ft_analisis_salidas.py --etiqueta <nombre>`
(logica pura en `src/utils/ft_salidas.py`, tests en `tests/test_ft_salidas.py`).

| Si la pregunta es... | Ir a |
|---|---|
| por que mirar las salidas y con que supuesto | 1 |
| como se mide (contra que, en que unidad, que se excluye) | 2 |
| que hizo el precio despues de las salidas, en todas las estrategias | 3 |
| como es de verdad la salida de TECH_SECTOR_v1 | 4 |
| que significa el bug del score 0,0 para la historia | 4.2 |
| por que los pesos del score no importan para la salida | 4.3 |
| que dispara las salidas de TECH_SECTOR_v1 | 4.4 |
| el plan de las 4 preguntas (parametros, pesos, umbral, sector) | 5 |
| el pre-registro del paso 1 (sin SMA21) | 6 |
| el resultado del paso 1: la SMA21 no se saca | 7 |
| el paso 2: grilla de pesos, midiendo la salida en si | 8 |
| que hace el precio despues de la salida actual (FT de control) | 8.2 |
| que dio cada palanca (quitar SMA21/RSI/MACD, mas peso a SMA50/RSI, SMA200 no obligatoria) | 8.2 |
| por que ninguna combinacion de pesos mejora la salida | 8.3 |
| como sale SMC_v1 y por que CHoCH/estructura rota nunca dispararon | 10.1-10.3 |
| el pre-registro de las salidas de SMC_v1 (y el corte por balances) | 10.4 |
| que dieron las salidas de SMC_v1 y el time stop corto | 10.5-10.7 |
| por que el FT de control de TECH_SECTOR_v1 quedo sin respaldo | 7.1 (correccion) |

---

## 1. La idea

Todo lo medido hasta el 17/9/2026 dice que las ENTRADAS no distinguen: el modelo ML
honesto da AUC 0,51 (docs/features_ml.md) y las senales de entrada de SMC, solas y a
plazo fijo, no le ganan al universo (docs/estructura_velas.md sec. 9.2). Lo unico que
paso un backtest fue SMC CON sus salidas (sec. 9.3 de ese doc). Si la entrada vale poco,
la salida es la que arma el resultado: cortar las que no andan, dejar correr las que si.

**Supuesto de trabajo (propuesta del usuario, 19/9/2026)**: la entrada es correcta y no
se toca. La pregunta queda: con la posicion abierta, salir ese dia fue mejor que seguir
adentro? Y despues, si moviendo una palanca de la salida el resultado hubiera sido mejor.

Los tres casos que interesan:
- **salio a tiempo**: despues de salir, el precio siguio peor;
- **salio temprano**: despues de salir, el precio siguio mejor;
- **indiferente / lateral**: despues no paso nada (la salida dio igual). La otra lectura
  de "lateral" -- dias adentro sin avanzar, plata parada -- queda para el paso 1.

---

## 2. Metodo

- **Contra el universo, no contra cero.** El tramo abril-septiembre 2026 fue alcista:
  medido en crudo, casi toda salida parece temprana. Es la misma trampa del label
  absoluto del ML (docs/features_ml.md sec. 9). Exceso = retorno del ticker desde la
  rueda de salida menos el retorno del universo equal-weight en la misma ventana.
- **En unidades de la volatilidad del ticker.** z = exceso / (desvio diario de las
  ultimas 60 ruedas x raiz(h)). Un 3% en una accion tranquila no es lo mismo que en una
  volatil. Mismo criterio que las alertas del dashboard (umbral por z, nunca fijo).
- **Horizontes fijos**: 5, 10 y 20 ruedas. Clasificacion a 10: "a tiempo" z < -0,5;
  "temprano" z > +0,5; "indiferente" en el medio.
- **Referencia**: salir un dia cualquiera al azar (todo el universo, todos los dias del
  mismo tramo): 31% a tiempo / 27% temprano / 43% indiferente.
- **La unidad independiente es el DIA de salida**, no la operacion: TECH_SECTOR_v1 tiene
  300 salidas en 64 ruedas. El IC95 se calcula sobre la media por dia (cada dia pesa uno).
  Las estrategias que comparten logica (TECH_SECTOR v1 / OPTIONS v1 / v2) son una familia.
- **Fechas**: la rueda de salida es `fecha_datos_salida` (el dato con el que decidio el
  bot), no `fecha_salida` (cuando corrio). Ver CLAUDE.md, FT asincronico.
- **Se excluye lo que no mide la regla**:
  - las salidas previas al arreglo `fix_score_cero_salida` (ft_cambios, 29/5/2026) en las
    estrategias 4, 6, 8 y 9: la consulta de salida no traia el close y el score daba 0
    todos los dias (sec. 4.2);
  - `ESTRATEGIA_DISCONTINUADA` (salida artificial de las bajas del 17/9) y `*_SPLIT_FIX`
    (incidente de splits del 21/7).
  Sobre 2.661 salidas cerradas: 576 excluidas por el bug, 38 artificiales, **2.047
  validas** (1.784 con ventana completa a 10 ruedas al 18/9).

**Directorio de resultados**: cada corrida escribe en
`reportes/analisis_salidas/AAAAMMDD_<etiqueta>/`. La fecha es la de la corrida y la
etiqueta dice que se corrio (`exploratorio`, `tech_sector_v1_p1`, ...). No se pisan
corridas de otro dia. `parametros.json` deja el commit y la rueda de datos: con eso cada
numero de este doc se puede rastrear a su corrida.

---

## 3. Panorama de todas las estrategias (exploratorio, 19/9/2026)

Exceso a 10 ruedas con IC95 por dia, y clasificacion. Solo filas con 8+ salidas.

| Estrategia | Salida | Ops | Dias | Exceso 10r [IC95] | Exceso 20r | A tiempo | Temprano | Indif. |
|---|---|---|---|---|---|---|---|---|
| **Referencia: salir al azar** | -- | 18.400 | -- | -- | -- | 31% | 27% | 43% |
| TECH_SECTOR_v1 | score | 238 | 55 | -0,51 [-1,87; +0,84] | -1,55 | 34% | 30% | 36% |
| TECH_SECTOR_v1 | take profit | 18 | 13 | -0,91 [-3,98; +2,15] | -4,40 | 28% | 28% | 44% |
| TECH_SECTOR_v1 | balance | 47 | 18 | -0,98 [-3,84; +1,88] | -1,28 | 47% | 28% | 26% |
| TECH_SECTOR_v2 | rotacion | 205 | 58 | +0,31 [-0,93; +1,55] | -1,17 | 32% | 29% | 39% |
| TECH_SECTOR_v2 | score | 47 | 32 | -0,62 [-2,85; +1,62] | -1,49 | 34% | 23% | 43% |
| TECH_SECTOR_v2 | stop | 41 | 23 | +0,26 [-2,63; +3,15] | -1,49 | 24% | 32% | 44% |
| OPTIONS_v1 | score | 26 | 18 | +2,43 [-0,94; +5,80] | -1,00 | 23% | 50% | 27% |
| OPTIONS_v1 | stop | 50 | 27 | +1,42 [-0,68; +3,52] | -1,57 | 16% | 32% | 52% |
| OPTIONS_v2 | stop | 48 | 29 | +1,32 [-1,06; +3,70] | -0,88 | 19% | 33% | 48% |
| OIEXIT_v1 | stop | 84 | 41 | -0,05 [-2,48; +2,38] | -1,29 | 31% | 30% | 39% |
| ML_SCANNER_v1 | score | 130 | 55 | +0,79 [-0,69; +2,26] | +0,99 | 29% | 31% | 40% |
| SMC_v1 | stop (trailing) | 12 | 10 | +1,35 [-4,26; +6,95] | +1,94 | 25% | 42% | 33% |
| SMC_v1 | time stop | 19 | 17 | +0,61 [-2,57; +3,79] | -0,52 | 32% | 26% | 42% |
| SMC_v2 | stop (trailing) | 13 | 12 | +2,98 [-2,08; +8,04] | +5,76 | 23% | 62% | 15% |
| COMBO_v1 | score | 294 | 65 | -0,02 [-1,22; +1,17] | +0,03 | 32% | 33% | 34% |

(Tabla completa en `panorama.log`.)

**Lectura:**

1. **Ninguna salida, de ninguna estrategia, se distingue de salir al azar.** Todos los
   IC95 incluyen el cero y los porcentajes se parecen a la referencia. En FT, degradar un
   score, rotar o cortar por stop no anticipa que viene peor que el universo.
2. **Los stops asoman como "temprano"** (SMC_v2 62%, SMC_v1 42%, OPTIONS 32-33% contra 27%
   al azar, con exceso positivo despues), coherente con que las senales mas fuertes del ML
   eran de reversion a la media. **No es significativo**: todos los IC incluyen el cero y
   las muestras son chicas. Y un stop se juzga por la cola que corta, no por el promedio.
   Queda como hipotesis.
3. **Las salidas por balance parecen "a tiempo" (~47-50%), pero no es merito de la
   regla.** Son 350 operaciones y **111 eventos unicos** (varias estrategias tenian el
   mismo ticker). Exceso a 10 ruedas de esos eventos: mediana -2,21%. De TODOS los
   balances del universo en el mismo tramo (234): mediana -1,93%. Se parecen: es la
   temporada de balances. Esta salida se juzga por el riesgo que evita (desvio posterior
   ~10-11%), no por la media.
4. **En FT_SMC_v1 las salidas propias de SMC nunca dispararon**: en 41 operaciones
   cerradas no hubo ni una por CHoCH bajista ni por estructura rota. Salio siempre por
   trailing stop, tiempo o balance. "Una regla que nunca dispara no es una regla".

Una primera mirada ese mismo dia, SIN excluir la ventana del bug, daba para TECH_SECTOR_v1
650 salidas por score con +0,67% de exceso. Con la ventana excluida son 238 con -0,51%.
La conclusion (indistinguible del azar) no cambia; el numero si.

---

## 4. TECH_SECTOR_v1: como es de verdad la salida

### 4.1 Parametros (del codigo, no de la ficha)

`scripts/forward_testing/ft_bot_tech_sectorial.py` + `src/strategies/sectorial.py` +
`src/strategies/scoring.py`:

| Parametro | Valor |
|---|---|
| Sectores | 9, presupuesto 11.111 USD c/u, hasta 5 posiciones por sector de ~2.222 USD |
| Entrada | score tecnico >= 4,0 |
| Salida por score | score tecnico <= 3,5 |
| Stop / take profit | entrada -2 x ATR14 / entrada +4 x ATR14 (fijos desde la entrada) |
| Orden de prioridad (v1) | balance manana -> score -> stop -> take profit |

Score tecnico (0 a 5,5):

| Condicion | Puntos |
|---|---|
| precio > SMA200 | obligatoria: si falla, score = 0 |
| precio > SMA50 | 2,0 |
| precio > SMA21 | 1,0 |
| MACD > senal (y histograma > 0, que es lo mismo) | 1,5 |
| RSI entre 45 y 68 | 1,0 |

Como el score va antes que el stop, el stop de 2 x ATR casi nunca llega a disparar (4 de
372 salidas validas). **La salida real de esta estrategia es el score, mas el take profit.**

### 4.2 La historia antes del 29/5 no es la regla (el bug del score 0,0)

Hasta el arreglo `fix_score_cero_salida` (ft_cambios, fecha efectiva 29/5/2026) a la
consulta de salida le faltaba el JOIN con `precios_diarios`: el close llegaba en 0, el
filtro SMA200 fallaba y el score daba 0 todos los dias. Recalculando el score real en
cada salida marcada `SCORE_DEGRADADO_0.0`:

| | Salidas "0.0" | Score real >= 4,0 (no debian salir) | Score real 0 | Ruedas en posicion (mediana) |
|---|---|---|---|---|
| Antes del arreglo | 412 | **357** | 16 | 1 |
| Despues del arreglo | 68 | 0 | 68 | 4 |

El arreglo funciono. Pero **antes del 29/5 la v1 vendia al dia siguiente de comprar** y
esa historia no mide la regla. El bug toco tambien a TECH_SECTOR_v2 y a OPTIONS v1/v2
(churn de `SIN_MOMENTUM` en la v2, decisiones 100% por PCR en OPTIONS).

**La ficha de la v1 lo habia diagnosticado mal**: describia la salida como "score = 0
(precio < SMA200)" y anotaba como problema de la regla que "el bot cerraba las 5
posiciones de Technology y las reabria en la misma corrida". Era el bug. La premisa de
TECH_SECTOR_v2 ("eliminar el exit binario de la v1") salio de ese diagnostico. Las dos
fichas quedaron anotadas.

### 4.3 El score es una regla de si/no: los pesos no mueven la salida

Con los pesos 2 / 1 / 1,5 / 1 los unicos valores posibles del score son
0 / 1 / 1,5 / 2 / 2,5 / 3 / 3,5 / 4 / 4,5 / 5,5. De ahi:

- **Entrar con >= 4,0 equivale a**: precio sobre SMA200, precio sobre SMA50, y al menos
  2 de las otras 3 (SMA21, MACD, RSI en rango). Ninguna combinacion sin SMA50 llega a 4.
- **Salir con <= 3,5 equivale a "dejo de cumplir la entrada".** Ningun score cae entre
  3,5 y 4,0: la distancia entre umbrales que parece haber no existe. Sale en cuanto se da
  vuelta una sola condicion (la que era la segunda de las 2 de 3).
- **Para la salida, los pesos no importan**: que el MACD valga 1,5 o 1,0 no cambia ninguna
  decision. Pesos y umbral juntos solo eligen **que combinaciones de condiciones hacen
  salir**. Muchos juegos de pesos distintos dan exactamente la misma regla, asi que
  "ajustar los ponderadores" es en realidad elegir combinaciones.

Las 17 situaciones posibles (16 combinaciones sobre SMA200 + debajo de SMA200):

| SMA200 SMA50 SMA21 MACD RSI | Score | Entra | Sale hoy | Sale sin_sma21 | Sale sin_macd | Sale sin_rsi |
|---|---|---|---|---|---|---|
| si si si si si | 5,5 | si | - | - | - | - |
| si si . si si | 4,5 | si | - | - | - | - |
| si si si si . | 4,5 | si | - | - | - | - |
| si si si . si | 4,0 | si | - | - | - | - |
| si . si si si | 3,5 | - | SALE | SALE | SALE | SALE |
| si si . si . | 3,5 | - | SALE | **-** | SALE | **-** |
| si si . . si | 3,0 | - | SALE | **-** | **-** | SALE |
| si si si . . | 3,0 | - | SALE | SALE | **-** | **-** |
| si . (cualquier otra) | 0-2,5 | - | SALE | SALE | SALE | SALE |
| si si . . . | 2,0 | - | SALE | SALE | SALE | SALE |
| debajo de SMA200 | 0 | - | SALE | SALE | SALE | SALE |

("sin_X" = X no puede disparar la salida; ver sec. 6.)

### 4.4 Que dispara las salidas validas (29/5 -> 18/9, 64 ruedas)

| Motivo | Salidas | Resultado medio | Ruedas en posicion (mediana) |
|---|---|---|---|
| score | 300 | -2,31% | 5 |
| balance | 47 | +2,40% | 8 |
| take profit | 19 | +15,52% | 11 |
| stop | 4 | -7,88% | 4 |

Condicion que cambio entre la rueda anterior y la de salida, en las 300 por score:

| Gatillo | Salidas | % | Resultado |
|---|---|---|---|
| pierde SMA21 (sola) | 92 | 31% | -1,66% |
| pierde SMA21 + MACD | 37 | 12% | -1,97% |
| sin cambio en la ultima rueda | 29 | 10% | -1,54% |
| pierde SMA50 (sola) | 27 | 9% | -2,21% |
| pierde SMA200 | 22 | 7% | -2,17% |
| pierde SMA50 + SMA21 | 21 | 7% | -3,23% |
| RSI sale por arriba (> 68) | 8 | 3% | **+5,64%** |
| combinaciones de 3 o mas | 64 | 21% | -1,5% a -10% |

Cuantas veces aparece cada condicion, sola o combinada: SMA21 en 194 (65%), SMA50 en 89
(30%), MACD en 71 (24%), SMA200 en 43 (14%), RSI por abajo en 36 (12%), RSI por arriba en
8 (3%).

- **La SMA21 participa en dos de cada tres salidas.** Es la condicion mas rapida (media de
  21 ruedas): la primera candidata a estar generando salidas por ruido.
- **El RSI tiene doble filo.** El tope de 68 esta para no comprar sobrecomprado, pero
  tambien hace salir: 8 veces la estrategia vendio porque la accion se puso demasiado
  fuerte, con +5,6% ganado. Es el caso "salio temprano" en su forma pura.
- **Las 29 "sin cambio en la ultima rueda" son todas atrasos de la operacion**: en las 29
  el bot NO evaluo esa posicion en la rueda anterior (no hay fila en
  `ft_posiciones_diarias`). La condicion se habia perdido antes y la salida llego tarde
  porque la rutina no corrio. Es ruido operativo, no de la regla.

Cuanto cambiaria cada variante, sobre esas 300 salidas: **sin_sma21 no habria salido ese
dia en 153 (51%)**, sin_macd en 162 (54%), sin_rsi en 11 (4%, casi todas las de RSI alto,
resultado medio +4,31%). "No habria salido ese dia" no es "no habria salido": la posicion
sigue hasta que falle la condicion de la variante, o salte el stop, el take profit o el
balance. Lo que pasa despues lo mide el paso 1. Pero muestra que las variantes no son
cosmeticas: cambian la mitad de las salidas. Y que la SMA21 no es la unica que pesa asi:
en la regla de "2 de 3" cualquiera de las tres puede ser la segunda que se pierde.

### 4.5 Que dice y que no dice esto

Dice como es la regla y que la dispara. NO dice todavia si alguna variante es mejor: eso
pide medir que pasa despues con las mismas entradas, y en FT hay 64 ruedas de un solo
tipo de mercado. Elegir una palanca mirando esos meses es sobreajuste hecho a mano (regla
del proyecto, CLAUDE.md "Alta y baja de estrategias FT").

---

## 5. Plan para TECH_SECTOR_v1 (las 4 preguntas del usuario, 19/9/2026)

En este orden, cada una con su pre-registro y sus resultados antes de pasar a la
siguiente:

1. **Parametros: sobran o faltan condiciones?** Para cada condicion, que pasa si no
   puede disparar la salida. Hipotesis principal: la SMA21 trae ruido. Pre-registro en
   la sec. 6.
2. **Pesos** y 3. **umbral** -> una sola pregunta (sec. 4.3): **que combinaciones deben
   hacer salir**. Para cada combinacion, salir vs mantener. De ahi salen pocas reglas
   alternativas concretas (ej. no salir por RSI alto; exigir que la condicion falle 2
   ruedas seguidas, que es la distancia entre umbrales que hoy no existe).
4. **Igual para cada sector o industria?** En FT son ~40 salidas por sector: no alcanza.
   Con la historia larga y la misma prueba del ML sectorial (dispersion entre sectores
   contra la variacion de cada sector entre anios). Por industria no da el tamano (13
   industrias con 5+ tickers). Umbrales por sector multiplican por 9 lo sobreajustable.

**Datos**: FT desde el 29/5 para diagnosticar; para decidir, las entradas que la
estrategia habria hecho entre 2021 y 2026 con la regla actual (motor
`scripts/backtesting_historico/ft_backtesting_runner.py --estrategia TECH_SECTOR_v1`),
fijas, con cada variante de salida re-simulada sobre el camino real del precio.
**Validacion final**: si una variante gana, va como instancia NUEVA en FT en paralelo,
con la v1 como control, registrada en `ft_cambios`. La v1 que esta corriendo no se toca.

---

## 6. PRE-REGISTRO del paso 1 -- parametros (escrito ANTES de correr, 19/9/2026)

**Pregunta**: alguna de las condiciones no obligatorias (SMA21, MACD, RSI) genera
salidas que no convienen? Hipotesis principal (del usuario, 19/9): **la SMA21, por lo
rapido que se cumple y se pierde, trae ruido**.

**Variantes** (la entrada NO cambia en ninguna):

| Variante | Definicion | Equivale a: sale si... |
|---|---|---|
| `actual` | score <= 3,5 | pierde SMA200, o pierde SMA50, o quedan menos de 2 de {SMA21, MACD, RSI} |
| **`sin_sma21`** (principal) | la SMA21 se trata como cumplida al evaluar la salida | pierde SMA200, o pierde SMA50, o se pierden MACD **y** RSI |
| `sin_macd` (comparacion) | idem con el MACD | pierde SMA200, o pierde SMA50, o se pierden SMA21 **y** RSI |
| `sin_rsi` (comparacion) | idem con el RSI | pierde SMA200, o pierde SMA50, o se pierden SMA21 **y** MACD |

"Sin X" quiere decir que **X no puede disparar una salida**, no que se le quiten los
puntos. Quitarle los puntos con el mismo umbral haria que la estrategia salga MAS seguido,
lo contrario de lo que se quiere probar. Implementado en `src/utils/ft_salidas.VARIANTES`,
equivalencias verificadas por test sobre las 32 combinaciones.

**Paso 0 -- fidelidad del motor** (antes de mirar variantes): correr el motor con la
regla actual y comparar sus operaciones contra las de FT en la ventana valida
(29/5 -> 18/9): mismos tickers y fechas de entrada y salida. Si no reproduce, se corrige o
se declara la diferencia antes de seguir.

**Datos**: entradas del motor con la regla ACTUAL, 2021-09-01 -> ultima rueda
disponible, fijas para todas las variantes. Para cada entrada se re-simula la salida
rueda a rueda desde la siguiente, con la prioridad de la v1: balance manana
(`earnings_historico`) -> score (segun la variante) -> stop (close <= entrada - 2 ATR) ->
take profit (close >= entrada + 4 ATR). Precio de salida = close de la rueda que decide.
Posiciones sin salida al final de los datos: se excluyen (censura) y se informan.

**Metricas, por operacion y por variante** (mismas entradas):
- **principal**: exceso de la operacion = retorno entrada -> salida menos el retorno del
  universo equal-weight en la misma ventana; y la **diferencia pareada** variante menos
  actual, operacion por operacion;
- secundarias: exceso por rueda en posicion (uso del capital), ruedas en posicion, % de
  operaciones ganadoras, percentil 5 del retorno (la cola), y la clasificacion a
  tiempo / temprano / indiferente de las salidas que cambian.

**Unidad independiente**: la rueda de entrada (las entradas del mismo dia comparten
mercado). IC95 de la diferencia pareada sobre las medias por rueda.

**Regla de lectura** (que cuenta como "la variante es mejor"), las tres:
1. diferencia pareada media de exceso por operacion > 0 con IC95 que excluye el cero;
2. positiva en al menos 4 de los 6 tramos anuales (2021-09 -> 2021-12, 2022, 2023, 2024,
   2025, 2026);
3. no empeora la cola: el percentil 5 del retorno por operacion no cae mas de 1 punto.

`sin_sma21` es la hipotesis principal y se lee sola. `sin_macd` y `sin_rsi` son
comparaciones: con 3 pruebas, si una secundaria pasa y la principal no, queda como
hipotesis a confirmar, no se adopta directo (regla de comparaciones multiples,
docs/features_ml.md sec. 12).

**Lo que este paso NO mide** (y se declara en el resultado):
- el efecto de cartera: una variante que mantiene mas tiempo ocupa cupos por sector mas
  tiempo y deja entrar menos operaciones nuevas. Con entradas fijas eso no se ve. Si una
  variante pasa, el paso siguiente es correr el motor con la variante completa;
- costos: el backtest no los tiene;
- el desglose por sector se informa pero NO decide (es la pregunta 4).

**Resultados**: `reportes/analisis_salidas/AAAAMMDD_tech_sector_v1_p1/` y la seccion 7
de este doc.

## 7. RESULTADOS -- paso 0 y paso 1 (19/9/2026)

Corrida: `python scripts/forward_testing/ft_analisis_salidas.py --seccion p0p1 --etiqueta
tech_sector_v1_p1` -> `reportes/analisis_salidas/20260919_tech_sector_v1_p1/` (`p0.log`,
`p1.log`, CSV por operacion y variante, `p1_resultado.json`).

**Como se ejecuto, respecto del pre-registro** (nada toca la regla de lectura):
- el motor se llamo EN MEMORIA (`run_tech_sector`): no se escribio nada en `bt_hist_*`;
- el paso 0 se midio en tres partes (el score, las entradas, y la regla actual re-simulada
  sobre las entradas REALES de FT contra sus salidas reales) en vez de comparar solo
  operaciones del motor contra FT. El motor no es una copia de FT (arranca sin
  posiciones, no bloquea entradas por balance, otro tamano de posicion); lo que genera
  todos los resultados es la re-simulacion, y eso es lo que habia que validar;
- "exceso por rueda en posicion": el promedio por operacion lo dominan las operaciones de
  1 rueda y sale distorsionado; se informa como exceso total / ruedas totales.

### 7.1 Paso 0 -- la re-simulacion reproduce a FT

| Chequeo | Resultado |
|---|---|
| score del analisis = el que registro el bot (`ft_posiciones_diarias`, desde 29/5) | 3.777 de 3.814 (99,0%) |
| entradas reales de FT con score >= 4 en su rueda | 365 de 368 |
| entradas del motor contra las de FT, misma ventana | 221 coinciden, 147 solo FT, 178 solo motor (esperable) |
| regla ACTUAL re-simulada sobre las 341 entradas reales cerradas: misma rueda de salida | 253 (74%) |
| ... la re-simulacion sale antes porque el bot no evaluo esa rueda (atraso de la rutina) | 47 |
| ... resto | 41: **36 por balance** (FT sale siempre una rueda ANTES: lee `earnings_calendar` con la fecha del reloj; la re-simulacion usa la fecha real de `earnings_historico`), 4 por datos corregidos despues (relleno de la rueda 28/8, hecho el 14/9) y 1 por escala de split (CRWD) |

**Ninguna diferencia sale de la regla del score.** El balance difiere en el momento, pero
es identico en todas las variantes y no sesga la comparacion pareada. La re-simulacion
sirve para comparar variantes.

**CORRECCION (19/9/2026, encontrada en el paso 0 de SMC_v1, sec. 10.4-10.5).** De esas 36
diferencias "por balance", **27 no son el momento: el anuncio no esta en
`earnings_historico`**. La tabla se carga con cuota (Alpha Vantage) y esta completa hasta el
20/7/2026; de la temporada de julio-agosto tiene 86 de 200 tickers. En esos casos la
re-simulacion atraviesa el balance, y el efecto NO es igual en todas las variantes: las que
salen antes lo esquivan mas seguido. Alcanza a lo re-simulado despues de ~13/7/2026: 208 de
las 341 entradas del FT de control y los ultimos dos meses de la confirmacion del paso 2.
Lo que decidio (el backtest 2021-2025 del paso 1 y la seleccion 2021-2024 del paso 2) no
cambia. **Los numeros del FT de control de las sec. 7.2-7.3 y 8.2-8.3 quedan sin respaldo**
hasta re-correrlos con el corte por balances de la sec. 10.4.

### 7.2 Paso 1 -- ninguna variante pasa

4.955 entradas del motor con la regla actual (2021-10-07 -> 2026-09-18, 195 tickers);
4.902 con salida en las cuatro variantes (53 censuradas en alguna, excluidas).

| Variante | Exceso medio por operacion | Ruedas en posicion (media) | Exceso por rueda-posicion | Ganadoras | p5 del retorno | Salidas por stop / take profit |
|---|---|---|---|---|---|---|
| **actual** | +0,10% | 6,9 | +0,0143% | 34% | -6,37% | 1% / 11% |
| sin_sma21 | +0,04% | 9,5 | +0,0043% | 35% | **-8,07%** | 4% / 15% |
| sin_macd | +0,07% | 10,3 | +0,0069% | 34% | -8,11% | 5% / 18% |
| sin_rsi | +0,10% | 7,3 | +0,0142% | 33% | -6,41% | 1% / 13% |

Regla de lectura pre-registrada (diferencia PAREADA variante menos actual, mismas entradas):

| Variante | Cambia la salida en | Diferencia media [IC95, 1.143 ruedas de entrada] | Anios positivos | Cola (p5) | Resultado |
|---|---|---|---|---|---|
| **sin_sma21** (principal) | 1.729 de 4.902 | -0,023 pp [-0,157; +0,110] -> no | 4/6 -> si | -6,37% -> -8,07% -> **no** | **NO PASA** |
| sin_macd | 2.011 | -0,038 pp [-0,182; +0,106] -> no | 3/6 -> no | -6,37% -> -8,11% -> no | NO PASA |
| sin_rsi | 292 | -0,022 pp [-0,065; +0,021] -> no | 2/6 -> no | -6,37% -> -6,41% -> si | NO PASA |

Por anio, sin_sma21: 2021 -0,09 / 2022 +0,10 / 2023 +0,01 / 2024 +0,13 / 2025 +0,06 /
2026 -0,52 pp.

**Sobre las entradas REALES del FT de control** (diagnostico, 63 ruedas, no decide):

| Variante | Exceso medio | Diferencia pareada contra actual [IC95] |
|---|---|---|
| actual | -0,96% | -- |
| sin_sma21 | -1,54% | **-0,835 pp [-1,313; -0,358]** |
| sin_macd | -1,57% | -0,874 pp [-1,419; -0,329] |
| sin_rsi | -0,99% | -0,052 pp [-0,155; +0,052] |

Por sector (informativo; es la pregunta 4), sin_sma21 menos actual: de Healthcare -0,35
pp a Industrials +0,25 pp.

### 7.3 Lectura

1. **La hipotesis de la SMA21 queda refutada.** Sacarla de la salida no mejora el
   exceso por operacion (diferencia -0,02 pp, intervalo centrado en cero) y empeora la
   cola 1,7 puntos. La estrategia pasa 38% mas dias en posicion (46.359 contra 33.602)
   para ganar menos por dia.
2. **Pasa lo mismo con cualquier condicion.** Quitar SMA21 o MACD hace que la posicion
   dure mas (6,9 -> 9,5-10,3 ruedas), que terminen mas operaciones en el stop (1% -> 4-5%)
   y mas en el take profit (11% -> 15-18%): mas dispersion, el mismo promedio, peor cola.
   **La salida rapida no es ruido que cuesta plata: es control de riesgo sin costo en el
   promedio.** Es coherente con el panorama (sec. 3): despues de salir, el promedio no
   cambia; lo que cambia es la distribucion.
3. **El RSI > 68** (el "salio temprano porque se puso fuerte" de la sec. 4.4) no importa:
   sin_rsi cambia 292 operaciones y la diferencia es cero con un intervalo angosto.
4. **El FT de control apunta al mismo lado, y mas fuerte**: sin SMA21 habria costado
   -0,84 pp por operacion desde el 29/5, con un intervalo que excluye el cero. En el
   backtest, 2026 tambien es el peor anio para la variante (-0,52). Coinciden.
5. **Contexto para lo que sigue**: con la regla actual la estrategia tiene +0,10% de
   exceso por operacion en 5 anios, sin costos (practicamente cero), y -0,96% en FT
   desde el 29/5. Con las entradas fijas, las palancas de la salida movieron la cola,
   no el promedio.

### 7.4 Decision

TECH_SECTOR_v1 sigue como esta. No se abre instancia nueva en FT y no corresponde
registro en `ft_cambios` (no cambia decisiones). Los pasos siguientes (combinaciones y
umbral, sector) y la direccion general se discuten con este resultado a la vista.

---

## 8. PASO 2 -- grilla de pesos: la SALIDA en si (19/9/2026)

### 8.0 Que se mide, en palabras del usuario

Aclaracion del usuario (19/9/2026), despues del paso 1: el objetivo no es el resultado de
la estrategia sino **la salida en si**:
1. como se comporto el precio DESPUES de que el FT dio la orden de salida, tal cual esta
   hoy (el FT de control);
2. mover palancas quitando parametros (sin SMA21, sin RSI, ... y combinaciones);
3. mover los ponderadores (menos peso a la SMA200 y mas al RSI, mas peso a la SMA50, ...);
y en cada caso medir, comparar con el FT de control y sacar conclusiones.

"Definir pesos es arbitrario": en vez de elegir pesos, se prueba una grilla y se mira como
resulta. El paso 1 se leyo por el resultado de la operacion (entrada -> salida); aca la
medicion principal es la de despues de la salida, y las tres variantes del paso 1 entran
en la misma tabla para leerlas tambien con esa vara.

### 8.1 PRE-REGISTRO (escrito ANTES de correr)

**La grilla**
- Pesos de SMA50, SMA21, MACD y RSI, cada uno en {0; 1; 1,5; 2; 3}.
- SMA200: obligatoria (como hoy: si falla, score 0) o con peso en {0; 1; 1,5; 2; 3}.
- Umbral fijo: sale si score <= 3,5.
- 3.750 combinaciones de pesos -> **589 reglas de salida distintas** (cada regla es el
  conjunto de combinaciones de condiciones que hacen salir; varias combinaciones de pesos
  dan la misma). Se mide cada regla una vez y se informa con los pesos que la producen. La
  regla actual aparece 4 veces. 107 reglas mandan salir en algun estado que cumple la
  entrada (saldrian al dia siguiente de comprar si el estado sigue): son parte de la
  grilla y se miden igual.
- Mas las 3 variantes del paso 1 (sin_sma21, sin_macd, sin_rsi) como referencia.
- Una grilla de pesos con umbral fijo mueve la salida en las dos direcciones: pesos mas
  altos hacen salir menos (se mantiene mas); peso 0 en una condicion hace salir MAS (con
  el umbral en 3,5, esa condicion deja de sumar). "Quitar un parametro" tiene asi las dos
  lecturas: peso 0 (no suma, sale mas) y la del paso 1 (no puede disparar, sale menos).

**Lo que NO cambia**: la entrada (score original >= 4,0); balance, stop (-2 ATR) y take
profit (+4 ATR) fijos desde la entrada, con la prioridad de la v1; precio = close.

**Datos**: las mismas 4.955 entradas del motor (regla actual, 2021-10 -> 2026-09), fijas;
la salida se re-simula con cada regla (maquinaria validada en el paso 0, sec. 7.1). Y las
entradas reales del FT de control desde el 29/5.

**Metricas, por regla, pareadas contra la regla actual operacion por operacion** (universo
= indice equal-weight rebalanceado a diario; z en desvios de 60 ruedas del ticker):
- **(a) DESPUES de la salida** (la principal): exceso del ticker contra el universo en las
  10 ruedas siguientes a la salida; clasificacion a tiempo (z < -0,5) / temprano (z > +0,5)
  / lateral. Se informa tambien a 5 y 20. Diferencia pareada = exceso post-salida de la
  regla menos el de la actual en la misma operacion. **Negativa = la regla sale en mejores
  momentos** (despues de irse, la accion anda peor que despues de irse con la actual).
- **(b) EL TRAMO**: diferencia pareada del exceso entrada -> salida. Evita premiar a una
  regla que se ve "a tiempo" despues de salir solo porque se comio la caida adentro.
- **(c) LA COLA**: percentil 5 del retorno por operacion.
- Unidad independiente: la rueda de entrada (IC95 sobre medias por rueda).

**Como se lee** (seleccion y confirmacion en periodos distintos, por las 589 comparaciones):
- **Seleccion** (entradas 2021-10 -> 2024-12): una regla es candidata si (a) media < 0,
  (b) media >= 0 y (c) el p5 no cae mas de 1 punto. De las candidatas se eligen las 3 con
  la diferencia (a) mas negativa.
- **Confirmacion** (entradas 2025-01 -> 2026-09): cada una de las 3 se confirma si en ese
  periodo (a) tiene IC95 enteramente por debajo de cero, (b) media >= 0 y (c) se cumple.
- **FT de control** (entradas reales desde el 29/5): las 3 y la actual; se informa y, si
  apunta al lado contrario, se marca.
- Ademas se informa el mapa completo: cuantas reglas son candidatas en la seleccion, y
  para las elegidas, que combinaciones de condiciones dejan de salir o empiezan a salir
  contra la actual.

**Lo que no mide**: el efecto de cartera (cupos por sector ocupados mas o menos tiempo) y
los costos. Una regla confirmada no se despliega sin pasar por el motor completo y por FT
en paralelo.

### 8.2 Resultados (19/9/2026)

Corrida: `reportes/analisis_salidas/20260919_tech_sector_v1_p2/` (`p2.log`, `p2_grilla.csv`
con las 589 reglas y las metricas de los dos periodos, `p2_resultado.json`).
**Control interno**: la re-simulacion vectorizada de la grilla da la misma rueda de salida
que la maquina validada en el paso 0 en **19.820 de 19.820** casos (4 reglas x 4.955
entradas). Entradas: 2.865 en la seleccion (2021-10 -> 2024-12) y 2.086 en la
confirmacion (2025-01 -> 2026-09).

**Primero: que hace el precio despues de la salida actual (el FT de control)**. Exceso
del ticker contra el universo en las 10 ruedas siguientes a la salida:

| | ops | ruedas en posicion | post10 | a tiempo | temprano | lateral | p5 por operacion |
|---|---:|---:|---:|---:|---:|---:|---:|
| seleccion 2021-24 | 2.865 | 6,8 | -0,20 pp | 32% | 29% | 39% | -5,75% |
| confirmacion 2025-26 | 2.065 | 7,0 | +0,12 pp | 33% | 30% | 37% | -7,00% |
| FT desde el 29/5 (entradas reales) | 341 | 7,1 | -0,58 pp | 36% | 27% | 36% | -6,35% |

El desvio del post10 por operacion es 7,26 pp. Despues de salir, la accion hace lo mismo
que el universo: "a tiempo" y "temprano" casi empatan. Es lo que ya decia el panorama
(sec. 3) para todas las estrategias.

**Resultado pre-registrado: 0 de 588 reglas son candidatas.** 281 reglas bajan el post
(a), 217 no empeoran el tramo (b) y 215 cumplen las dos; ninguna cumple ademas la cola
(c): esas 215 empeoran el percentil 5 entre 1,69 y 3,34 puntos. No hay nada que
confirmar y la regla actual queda.

**Segundo y tercero: las palancas nombradas por el usuario**. Diferencia pareada contra
la regla actual, en pp; post: negativo = sale en mejor momento; tramo (entrada -> salida):
positivo = mejor. `*` = el IC95 por rueda excluye el cero.

| palanca | ruedas | post sel | post conf | post FT | tramo sel | tramo conf | tramo FT | p5 sel / conf / FT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| actual | 6,8 | | | | | | | -5,75 / -7,00 / -6,35 |
| SMA21 peso 0 (quitar) | 3,0 | +0,07 | -0,04 | +0,31 | -0,18 | +0,08 | **+0,69*** | -4,87 / -5,94 / -5,46 |
| RSI peso 0 (quitar) | 4,7 | -0,04 | -0,12 | -0,00 | -0,04 | +0,13 | **+0,48*** | -5,26 / -6,32 / -5,71 |
| MACD peso 0 (quitar) | 4,1 | +0,06 | +0,03 | +0,24 | -0,11 | -0,04 | **+0,47*** | -5,29 / -6,44 / -6,08 |
| SMA50 peso 3 | 10,6 | -0,19 | +0,03 | +0,44 | +0,03 | -0,16 | **-0,91*** | -7,47 / -9,00 / -8,99 |
| RSI peso 2 | 10,0 | **-0,23*** | +0,02 | +0,63 | +0,07 | -0,17 | **-1,00*** | -7,63 / -9,09 / -9,06 |
| SMA200 pesa 1 (no obligatoria) y RSI 2 | 12,2 | -0,10 | -0,02 | +0,61 | +0,02 | -0,14 | **-1,16*** | -8,93 / -9,82 / -9,81 |
| MACD peso 1 (hoy 1,5) | = la regla actual: no cambia ninguna salida | | | | | | | |
| sin_sma21 (paso 1: no puede disparar) | 9,6 | **-0,23*** | +0,02 | +0,39 | +0,07 | -0,18 | **-0,87*** | -7,44 / -8,96 / -8,98 |
| sin_macd (paso 1) | 10,5 | -0,20 | +0,03 | +0,44 | +0,04 | -0,16 | **-0,91*** | -7,46 / -9,00 / -8,99 |
| sin_rsi (paso 1) | 7,3 | +0,07 | -0,01 | +0,00 | -0,04 | +0,00 | -0,08 | -5,76 / -7,06 / -6,35 |

"RSI peso 2" es la misma regla que la mejor de las 588 en la seleccion. Por anio (post /
tramo): 2021 -0,65/-0,34 | 2022 -0,20/+0,08 | 2023 -0,13/+0,02 | 2024 -0,32/+0,14 |
2025 -0,09/+0,06 | 2026 +0,17/-0,48. "SMA21 peso 0", en sentido contrario: 2021
-0,26/-0,60 | 2022 +0,09/-0,02 | 2023 +0,10/-0,25 | 2024 +0,05/-0,19 | 2025 -0,42/-0,02 |
2026 +0,53/+0,20.

**El mapa completo** (complemento descriptivo, NO pre-registrado; no cambia la lectura):
- **La grilla es casi una sola dimension: cuanto tiempo se queda.** De 588 reglas, 516 se
  quedan mas que la actual y solo 72 menos (la actual, "salir si deja de cumplir la
  entrada", ya es de las mas rapidas posibles). Entre reglas, las ruedas en posicion
  correlacionan +0,95 con la caida del p5, -0,42 con la diferencia post y +0,62 con la
  del tramo.
- **El orden de la seleccion no se sostiene en la confirmacion**: la correlacion entre
  reglas de la diferencia post en 2021-24 y en 2025-26 es -0,16 (-0,31 la del tramo).
  Quedarse mas, en promedio: tramo -0,01 en la seleccion y -0,18 en la confirmacion.
- **La conclusion no depende de la condicion de la cola**: ignorandola, las 215 reglas
  que pasan (a) y (b) en la seleccion dan en la confirmacion una diferencia post entre
  -0,10 y +0,21, y ninguna mantiene el tramo >= 0.
- **Escala**: todas las diferencias post de la grilla caen entre -0,23 y +0,22 pp, contra
  un desvio de 7,26 pp por operacion.
- **El umbral queda cubierto**: con los pesos de hoy, los umbrales de 2,0 a 5,0 dan reglas
  que estan dentro de la grilla (solo 1,0 y 1,5 quedan afuera). La pregunta 3 del plan
  (sec. 5) se contesto con esta misma corrida.

### 8.3 Lectura

1. **Ninguna combinacion de pesos saca en mejores momentos.** La mejor de las 588 en
   2021-24 (justo la palanca "mas peso al RSI": -0,23 pp a 10 ruedas, con un IC que apenas
   excluye el cero) es lo que se espera de la mejor de 588 comparaciones por azar: en
   2025-26 da +0,02 y en FT +0,63.
2. **Los pesos no eligen el momento de salir; eligen cuanto tiempo quedarse.** Quedarse
   mas es mas dispersion: la cola empeora siempre (2 a 3 puntos) y el tramo sube o baja
   segun el anio. Salir antes es lo inverso. Es una perilla de riesgo, no de timing.
3. **La senal de salida no tiene informacion sobre lo que viene.** Perder la tendencia
   (medias, MACD, RSI) no anticipa que la accion ande peor que el universo en las 10
   ruedas siguientes: post10 de -0,20 / +0,12 / -0,58 pp con 7,26 pp de desvio.
4. **Lo unico que se repite en los tres periodos**: las tres palancas "peso 0" (salir mas
   rapido) mejoran la cola entre 0,3 y 1,1 puntos en la seleccion, la confirmacion y FT.
   En FT tambien mejoraron el tramo (+0,47 a +0,69 pp, IC que excluye el cero; OJO: sin
   respaldo, faltaban balances en esa re-simulacion, ver la correccion de la sec. 7.1), pero en
   2021-24 lo empeoraban (-0,04 a -0,18) y el desglose por anio muestra que 2026 es el
   anio en que salir rapido pago: es el mercado del tramo, no una propiedad de la regla.
   Y no generaliza a "salir mas rapido": de las 72 reglas mas rapidas, solo 24 mejoran la
   cola en los dos periodos.
5. **RSI peso 0 es la que queda mas cerca** (post ~0 en los tres periodos, cola mejor en
   los tres, tramo -0,04 / +0,13 / +0,48) y no pasa: falla (b) en la seleccion por 0,04.
   Elegirla ahora seria elegir mirando el resultado; queda anotada, no abre nada.

### 8.4 Decision

TECH_SECTOR_v1 sigue como esta. No se abre instancia nueva en FT ni corresponde registro
en `ft_cambios`. Con los pasos 1 y 2 quedan contestadas las preguntas 1 a 3 del plan
(parametros, pesos y umbral) para esta estrategia: **la salida por score no tiene
informacion de timing; su unico efecto es cuanto tiempo se queda la posicion y, con eso,
la cola**. Lo que sigue (la pregunta 4 por sector, y hacia donde ir) se decide con el
usuario con este resultado a la vista.

**Lo que no mide**: el efecto de cartera (cupos ocupados mas o menos tiempo), los costos
(salir mas rapido es operar mas) y las otras salidas de la estrategia (stop -2 ATR, take
profit +4 ATR, balance), que quedaron fijas.

---

## 9. Reglas que deja

1. **Antes de analizar la historia de una estrategia, cruzarla con `ft_cambios`.** La
   v1 tiene 5 semanas que no miden su regla. Un analisis sin cortar esa ventana habria
   estudiado el bug.
2. **Un score ponderado con umbral es una regla de si/no.** Antes de "ajustar pesos",
   enumerar las combinaciones: casi siempre hay menos decisiones distintas de las que
   parece, y varios pesos dan la misma regla.
3. **Comparar la salida contra quedarse adentro y contra el universo**, en unidades de la
   volatilidad del ticker, con el dia como unidad independiente.
4. **Contar el mismo evento una sola vez.** Las salidas por balance eran 350 operaciones y
   111 eventos.
5. **Separar el ruido de la operacion del de la regla**: una salida atrasada porque la
   rutina no corrio no dice nada de la regla (29 de 300 en la v1).
6. **Verificar el diagnostico de una ficha contra los datos antes de construir encima.**
   La v2 se diseno para arreglar un problema de la v1 que era un bug.
7. **Relajar una salida se juzga por la cola, no solo por el promedio.** Sin SMA21 el
   promedio no cambia y el percentil 5 empeora 1,7 puntos: sin la condicion 3 de la
   regla de lectura, la variante habria parecido "neutra".
8. **Validar la re-simulacion contra lo que paso antes de creerle a una comparacion de
   variantes** (paso 0): la misma maquina que genera los resultados tiene que reproducir
   las salidas reales.
9. **Una grilla de pesos se lee con seleccion y confirmacion en periodos distintos.** La
   mejor de 588 reglas en 2021-24 tenia un IC que excluia el cero y en 2025-26 dio
   cero: con cientos de comparaciones, "la mejor" siempre parece buena. Mirar ademas si
   el ORDEN entre reglas se mantiene de un periodo al otro (aca: correlacion -0,16).
10. **Antes de leer una grilla, ver que dimension mueve de verdad.** Aca los pesos solo
   movian cuanto se queda la posicion (correlacion +0,95 con la cola): comparar reglas de
   distinta duracion es comparar riesgo, no timing.
11. **Lo que salio bien en FT puede ser el tramo.** Salir mas rapido mejoro el tramo en
   FT con un IC que excluye el cero y lo empeoraba en 2021-24: el desglose por anio lo
   muestra antes de confundir el mercado de 2026 con una propiedad de la regla. (Y ademas,
   ver la correccion de la sec. 7.1: en FT faltaban balances.)
12. **Si la historia de una tabla mira al futuro, reconstruir lo que el bot veia** rueda por
   rueda y verificarlo contra lo que el bot guardo (sec. 10.5, paso 0) antes de re-simular.
13. **Revisar la cobertura de CADA insumo del periodo, no solo el precio.**
   `earnings_historico` se carga con cuota y estaba dos meses atrasada: la re-simulacion
   atravesaba balances que el bot evita, y no igual en todas las reglas. Se encontro en el
   paso 0 de SMC_v1 y habia pasado inadvertido en el de TECH_SECTOR_v1.
14. **Agrupar las reglas casi iguales antes de elegir "las 3 mejores".** En SMC_v1 las tres
   elegidas eran la misma regla (CHoCH y estructura rota cambian muy pocas operaciones): se eligio
   una sola hipotesis creyendo elegir tres.

---

## 10. SMC_v1 -- anatomia de la salida y PRE-REGISTRO (19/9/2026)

Pedido del usuario: aplicar a SMC_v1 el mismo metodo que a TECH_SECTOR_v1 (sec. 4-8) y
ver como responden las distintas combinaciones de salida contra el FT de control.

### 10.1 Parametros (del codigo, no de la ficha)

`scripts/forward_testing/ft_bot_smc.py` + `ft_scoring.py`. En cada corrida el bot primero
sube el stop y despues evalua, en este orden (la primera que se cumple cierra):

| Prioridad | Salida | Regla |
|---|---|---|
| P0 | balance | anuncio en la rueda siguiente (`earnings_filter`) |
| P1 | TRAILING_SL | close <= stop. Stop inicial = ultimo swing low de 10 barras (`close / (1 + dist_sl_10_pct/100)`); cada rueda sube a ese nivel si es mayor, nunca baja |
| P2 | CHOCH_BEAR | `choch_bear_10 = 1` (el close cruza por debajo del ultimo swing low con estructura alcista) |
| P3 | ESTRUCTURA_ROTA | `estructura_10 = -1` (maximo y minimo descendentes) |
| P4 | TIME_STOP | 20 dias CORRIDOS desde la fecha de registro de la entrada, contados con el reloj (`date.today()`) |

Sin take profit. La salida **no tiene ponderadores**: el score de calidad (0-3) solo ordena
candidatos de entrada. Lo que se puede mover son condiciones (quitar/agregar) y valores
(dias del time stop, que swing low sigue el stop). El motor de backtesting cuenta el time
stop en ruedas (20) y sale del stop al precio del stop: no es FT, por eso aca la salida se
re-simula con las reglas del bot (close, dias corridos).

### 10.2 Que hizo en FT (24/4 -> 14/9/2026)

46 entradas, 41 cerradas. `ft_cambios` no tiene ventanas que invaliden a SMC_v1 (el bug del
score 0,0 no la toco) y ninguna salida es `*_SPLIT_FIX`.

| Salida | Ops | Retorno medio | Mediana | Ruedas en posicion |
|---|---:|---:|---:|---:|
| TIME_STOP | 21 (51%) | +3,37% | +1,12% | 14,6 |
| TRAILING_SL | 12 (29%) | -4,72% | -4,41% | 5,3 |
| balance | 8 (20%) | +0,17% | +0,11% | 7,1 |
| CHOCH_BEAR / ESTRUCTURA_ROTA | 0 | | | |

El stop subio en 12 de las 41. Distancia al swing low al entrar: mediana 4,96% (1,05 a
7,91). En el panorama (sec. 3): despues del stop +1,35 pp a 10 ruedas (42% "temprano") y
despues del time stop +0,61, los dos con IC que incluye el cero. Antecedente: el JOURNAL del
18/7 miro 14 time stops ("cosecha cerca de techos locales") y dejo reevaluar con ~30.

### 10.3 Tres diferencias con TECH_SECTOR_v1 que definen el diseno

1. **El stop es la salida estructural.** Esta puesto en el swing low: un CHoCH bajista es
   que el close pierda ese mismo nivel, y P1 va antes. **Hipotesis** (se verifica en el
   paso 0): por eso CHOCH_BEAR y ESTRUCTURA_ROTA no dispararon nunca. A diferencia de
   TECH_SECTOR (stop 1% y take profit 11% de las salidas, que quedaron fijos), aca el stop
   es el 29% de las salidas y entra como palanca (decision del usuario, 19/9).
2. **La historia no se lee de la tabla.** `features_market_structure` mira 10 ruedas al
   futuro (docs/estructura_velas.md). En vivo el bot lee la ULTIMA fila, que no mira al
   futuro pero tiene swings provisorios. Para re-simular se reconstruye, rueda por rueda,
   lo que el bot veia: el modulo viejo (`_calcular_estructura_n`) sobre los datos hasta
   esa rueda. Verificado en AAPL: con una ventana de 250 barras da lo mismo que con la
   historia completa en 400 de 400 ruedas, y la ultima rueda coincide con la tabla. Usar
   `features_estructura` (swings confirmados) mediria a SMC_v3, no a la v1.
3. **Las entradas son pocas con el tope.** La estrategia tiene 5 cupos (15% del capital
   cada uno); con el tope, el backtest hizo ~330 operaciones en 5 anios. El tope es de la
   estrategia y no se toca: la salida se mide operacion por operacion y el tope solo
   decide cuales senales entraban. La muestra principal son TODAS las senales; las que
   habria tomado la cartera con el tope se informan aparte.

### 10.4 PRE-REGISTRO (escrito ANTES de correr)

**Reconstruccion** (lo que el bot veia cada rueda): estructura N=10 y N=5 del modulo viejo
sobre las ultimas 250 barras hasta la rueda; eventos CHoCH/BOS alcistas en los 12 dias
corridos previos segun esa misma reconstruccion; patrones y volumen de
`features_precio_accion` (lo que lee el bot). Control: la ventana de 250 contra la historia
completa en una muestra de tickers.

**Paso 0 -- fidelidad** (si la reconstruccion no reproduce lo que vio el bot, se frena):
- (a) en las 46 entradas de FT, la reconstruccion contra lo que el bot guardo en
  `detalle_entrada` (distancia al swing low, estructura, evento, vela) y si esa rueda es
  una senal de entrada;
- (b) la regla actual re-simulada sobre las 41 cerradas contra la salida real (rueda y
  motivo), separando los atrasos de la rutina y el reloj del time stop;
- (c) el stop final re-simulado contra el que guardo el bot;
- (d) sobre todas las entradas historicas: motivos de salida de la regla actual, cuantas
  veces CHOCH_BEAR o ESTRUCTURA_ROTA salen primero, y en cuantas salidas por stop el CHoCH
  bajista dispara esa misma rueda.

**Entradas, fijas**: todas las senales de la v1 entre 2021-09-01 y la ultima rueda (la regla
de entrada se IMPORTA de `ft_scoring.calcular_score_estructura`, score >= 1), sin entrar en
la rueda previa a un balance, una posicion por ticker a la vez con la salida actual (puede
volver a entrar la misma rueda en que salio, como el bot). Aparte, la muestra "con tope":
cartera de 5 posiciones que entra por score (desempate alfabetico) con la salida actual.

**La grilla** (el balance queda fijo; precio de salida = close; prioridades del bot):
- Stop: `trail10` (actual) | `trail5` (sigue el swing low de 5 barras, mas ajustado) |
  `fijo` (el de la entrada, no sube) | `sin` stop.
- CHoCH bajista: si / no. Estructura rota: si / no.
- Time stop: 10 | 15 | 20 (actual) | 30 | 45 dias corridos | sin time stop.
- 96 combinaciones; las que dan las mismas salidas en todas las operaciones se agrupan.
- Palancas nombradas (se informan aparte): sin time stop, time stop 10/15/30/45, stop fijo
  (sin trailing), sin stop, trailing de 5 barras, sin salidas estructurales.

**Primero, el control**: con la regla actual, que hizo el precio en las 10 ruedas despues de
cada salida, por tipo de salida (time stop, stop, balance).

**Metricas y lectura: las del paso 2 de TECH_SECTOR (sec. 8.1)**, sin cambios: (a)
diferencia pareada del exceso contra el universo en las 10 ruedas DESPUES de la salida
(principal; negativo = mejor), (b) diferencia pareada del tramo entrada -> salida (>= 0),
(c) el percentil 5 del retorno por operacion no cae mas de 1 punto. Seleccion con las
entradas 2021-09 -> 2024-12 (candidatas: (a) < 0, (b) >= 0, (c); las 3 con (a) mas
negativa); confirmacion 2025-01 -> 2026-09 ((a) con IC95 por rueda enteramente < 0, (b) >= 0,
(c)). FT de control (entradas reales) y muestra con tope: se informan, no deciden. Una
operacion sin salida al final de los datos queda fuera de la comparacion de esa regla.

**Lo que no mide**: el efecto de cartera (cupos ocupados mas o menos tiempo), los costos y
la entrada. FT de SMC_v1 son 41 operaciones: sirve para ver si apunta al mismo lado.

**ENMIENDA despues del paso 0 y ANTES de correr la grilla (19/9/2026): corte por balances.**
El paso 0 encontro que `earnings_historico` esta completa hasta el 20/7/2026 y despues no:
de la temporada de julio-agosto tiene 86 de 200 tickers (la carga es con cuota de Alpha
Vantage y va atrasada). El primer anuncio esperado que falta es ~21/7 (BKR). Re-simular
despues de esa fecha haria que las posiciones "atraviesen" balances que el bot si evita, en
todas las reglas. Por eso la re-simulacion (entradas y salidas) se corta 5 ruedas antes de
la rueda en que el bot habria salido por ese primer balance faltante; las ruedas
posteriores solo se usan para medir que hizo el precio despues de salir. El script calcula
el corte solo (`fin_balances`): con la tabla completa, se vuelve a correr hasta la ultima
rueda. La confirmacion queda 2025-01 -> 2026-07 y el FT de control, las entradas de FT
antes del corte.

### 10.5 Resultados (19/9/2026)

Corrida: `reportes/analisis_salidas/20260919_smc_v1/` (`todas.log`, `grilla.csv`,
`entradas_todas.csv`, `entradas_con_tope.csv`, `p0_resimulacion_vs_ft.csv`). Reconstruccion:
202.460 filas, 200 tickers, 26 minutos, en `reportes/analisis_salidas/cache/`. Control de la
ventana: 250 barras = historia completa en 120 de 120 ruedas (AAPL, JPM, HOOD). Control
interno de la grilla: la salida vectorizada = la de referencia rueda por rueda en **88.320 de
88.320** casos (96 reglas x 920 entradas).

**Paso 0 -- la reconstruccion reproduce al bot**

| Chequeo | Resultado |
|---|---|
| (0) ultima rueda (18/9) reconstruida = `features_market_structure`, lo que lee el bot | 200 de 200 tickers |
| (a) las 46 entradas de FT: reconstruccion = lo que el bot guardo (distancia, estructura, evento, vela, score) | 45 de 46 (SE 19/5 difiere en distancia al swing low y vela) |
| (b) regla actual sobre las 41 cerradas: misma rueda y motivo | 27 |
| ... el balance no esta en `earnings_historico` (ver enmienda de la sec. 10.4) | 6 |
| ... el bot no evaluo esa rueda (rutina) | 3 |
| ... time stop: el bot cuenta con el reloj y la fecha de registro | 2 |
| ... misma rueda, otro motivo (PG: time stop y balance la misma rueda) | 1 |
| ... otra: RIVN (entro y salio por balance con la misma rueda de datos); V (el 8/9 un swing provisorio habria subido el stop; el bot no corrio esa rueda y al dia siguiente ese swing ya no existia) | 2 |
| (c) stop final re-simulado = el del bot (+-0,1%) | 27 de 27 |

V muestra algo propio de esta estrategia: **con swings provisorios, el stop depende de que
dias corre el bot**.

**La historia** (entradas 2021-09-01 -> 2026-07-13, corte por balances): 920 senales en 178
tickers (22 / 99 / 175 / 217 / 278 / 129 por anio); con el tope de 5 posiciones, 503.

| Regla actual | TIME_STOP | TRAILING_SL | balance | ESTRUCTURA_ROTA | CHOCH_BEAR | sin salida |
|---|---:|---:|---:|---:|---:|---:|
| salidas | 380 (41%) | 352 (38%) | 159 (17%) | 19 (2%) | **1** | 9 |

La hipotesis de la sec. 10.3 se confirma: **el CHoCH bajista salio primero 1 vez en 920**. En
55 de las 352 salidas por stop el CHoCH dispara esa misma rueda (el stop va antes); en el
resto el stop, que solo sube, ya esta por encima del ultimo swing low. Sin stop, CHoCH y
estructura rota saldrian en el 12% y el 4%.

**Primero: que hace el precio despues de la salida actual** (10 ruedas, contra el universo):

| Salida | ops sel / conf | retorno de la operacion | post10 sel / conf | a tiempo | temprano |
|---|---|---|---|---|---|
| todas | 513 / 398 | +0,10% / +0,07% | +0,14 / +0,22 pp | 31% / 32% | 32% / 30% |
| time stop | 226 / 154 | +3,34% / +4,09% | +0,18 / -0,05 | 31% / 33% | 33% / 30% |
| stop | 187 / 165 | -4,22% / -4,92% | -0,55 / +0,16 | 31% / 28% | 26% / 25% |
| balance | 87 / 72 | +0,48% / +2,35% | **+1,37 / +1,10** | 31% / 38% | **44% / 43%** |
| estructura rota | 13 / 6 | +3,27% / +6,85% | +0,94 / -2,34 | | |

Desvio del post10 por operacion: 7,53 pp. Despues del time stop y del stop, la accion hace
lo mismo que el universo: ni "cosecha techos" (JOURNAL 18/7, con 14 casos) ni corta
"temprano" (el 42% del panorama de FT eran 12 salidas). Despues de salir por balance las
acciones suben mas que el universo (43-44% "temprano"), sin IC calculado: el balance no es
una palanca de esta grilla y queda como observacion.

**Resultado pre-registrado: ninguna regla se confirma.** En la seleccion 50 de 95 reglas son
candidatas (57 bajan el post, 73 no empeoran el tramo). Las 3 elegidas son, en la practica,
**la misma regla**: time stop de 10 dias con el trailing de 5 barras, con CHoCH y estructura
rota prendidas o apagadas (cambian muy pocas operaciones). Seleccion: post -0,50, tramo +0,32.
Confirmacion: post **-0,31 IC95 [-0,89; +0,27] -> no**; tramo +0,21 -> si; cola -7,43% ->
-6,69% -> si. **NO SE CONFIRMA.** FT de control (20 operaciones antes del corte): post -1,41
IC95 [-6,82; +4,01], no dice nada. Con tope: post -0,36, tramo +0,10, cola -7,21% -> -5,81%.

**Las palancas** (diferencia pareada contra la actual, pp; post: negativo = mejor; tramo:
positivo = mejor; `*` = IC95 por rueda que excluye el cero; "tope" = la muestra de 503 con
el tope de 5 posiciones):

| palanca | ruedas (hoy 9,5) | post sel / conf / tope | tramo sel / conf / tope | p5 sel / conf (hoy -6,52 / -7,43) |
|---|---:|---|---|---|
| time stop 10 dias | 5,9 | **-0,49*** / -0,39 / -0,42 | **+0,31*** / +0,26 / +0,18 | -5,62 / -7,20 |
| time stop 15 dias | 8,0 | -0,15 / -0,01 / -0,01 | +0,06 / +0,14 / +0,01 | -6,23 / -7,30 |
| time stop 30 dias | 12,2 | +0,04 / +0,14 / +0,18 | +0,08 / -0,03 / +0,10 | -6,65 / -7,65 |
| time stop 45 dias | 14,2 | -0,20 / +0,27 / -0,18 | **+0,40*** / -0,11 / +0,34 | -6,65 / -7,67 |
| sin time stop | 15,9 | -0,28 / +0,32 / +0,05 | +0,38 / -0,00 / +0,17 | -6,65 / -7,68 |
| stop fijo (sin trailing) | 9,9 | +0,05 / -0,13 / +0,05 | -0,02 / +0,06 / +0,02 | -6,82 / -7,46 |
| trailing de 5 barras | 8,6 | -0,05 / -0,04 / -0,12 | +0,02 / +0,01 / -0,06 | -5,80 / -7,14 |
| sin stop | 11,6 | +0,29 / -0,28 / -0,07 | **-0,28*** / +0,13 / +0,01 | **-9,31 / -9,18** |
| sin CHoCH ni estructura rota | 9,7 | +0,05 / +0,03* / -0,02 | +0,03 / -0,01 / +0,01 | -6,52 / -7,43 |
| sin stop ni time stop | 24,0 | +0,20 / +0,59 / +0,55 | -0,05 / -0,29 / -0,02 | **-10,66 / -10,98** |

Time stop de 10 dias por anio (post / tramo): 2021 -0,12/-0,19 | 2022 -0,98/+0,83 | 2023
+0,20/-0,32 | 2024 -0,81/+0,59 | 2025 -0,35/+0,44 | 2026 -0,47/-0,15.

**El mapa** (complemento descriptivo, NO pre-registrado): entre reglas, las ruedas en posicion
correlacionan +0,76 con la caida de la cola; el orden de la seleccion se sostiene algo en la
confirmacion (correlacion +0,22 del post y +0,14 del tramo; en TECH_SECTOR era -0,16);
ignorando la cola, 56 reglas pasan (a) y (b) en la seleccion y en la confirmacion 36 de
ellas mantienen el tramo >= 0.

### 10.6 Lectura

1. **Con la vara pre-registrada no hay cambio**: ninguna regla confirma que sale en mejores
   momentos. SMC_v1 queda como esta.
2. **CHoCH bajista y estructura rota casi no existen como salida** (1 y 19 de 920): el
   stop, puesto en el swing low y con prioridad, sale antes. Quitarlas cambia 11 de 513
   operaciones y nada mas.
3. **El stop es control de la cola**: sin stop, el percentil 5 cae ~2,8 puntos en los dos
   periodos. Trailing o fijo casi no difieren (45 operaciones cambian); el trailing de 5
   barras mejora la cola 0,3-0,7 puntos sin mover el resto.
4. **El time stop es la unica palanca con una direccion que se repite**: acortarlo a 10
   dias (~6 ruedas) mejora el post y el tramo en la seleccion con IC que excluye el cero, va
   al mismo lado en la confirmacion (post -0,39 [-0,95; +0,17], tramo +0,26 [-0,16; +0,68])
   y en la muestra con tope, con la cola mejor en todas; el post es negativo en 5 de 6 anios.
   Es consistente con que lo que la senal SMC tenga de bueno dure alrededor de una semana:
   entre las ~7 y las ~14 ruedas esas posiciones andan peor que el universo. Alargarlo o sacarlo
   parece bien en 2021-2024 y se da vuelta en 2025-2026. **No alcanza la vara**: el IC de la
   confirmacion incluye el cero.
5. **Despues de salir**, time stop y stop no se distinguen del universo; el balance si
   (+1,1 a +1,4 pp), como observacion.
6. **El FT de control no informa**: 20 operaciones antes del corte, IC de +-5 pp.

### 10.7 Estado

La regla de SMC_v1 no se toca. No se abre instancia nueva ni corresponde registro en
`ft_cambios`. El time stop corto queda como la hipotesis mejor respaldada de los dos
analisis de salidas, sin confirmar. Si se quisiera probar en FT, conviene saber de antemano
que FT no la va a poder confirmar: una diferencia de ~0,4 pp por operacion con 7,5 pp de
desvio pide del orden de mil operaciones, y SMC_v1 hace ~100 por anio. La decision seria
sobre el backtest. Con `earnings_historico` completa, la confirmacion se extiende a
septiembre con el mismo script (el corte se calcula solo).

**Decidido con el usuario (19/9/2026): no completar ahora las fechas de balance que faltan.**
Se pueden reconstruir sin cuota desde el calendario de Nasdaq por dia (probado: el 29/7
devuelve 284 empresas, PG coincide con la tabla), ~48 consultas del 13/7 al 18/9. Impacto
estimado, marginal: en TECH_SECTOR_v1 la decision salio de la seleccion 2021-2024 (0
candidatas, fechas completas) y solo se corregirian los numeros descriptivos del FT de
control; en SMC_v1 la confirmacion ganaria ~40-45 entradas (+10%) y, para que el time stop de
10 dias se confirme, esas operaciones tendrian que mostrar ~-1,4 pp, 3-4 veces el efecto
observado. Queda pendiente para cuando se quiera decidir el time stop corto. Aparte, conviene
poner al dia `earnings_historico` (carga incremental), que alimenta tambien la vista
"Reaccion a balances" del dashboard.
