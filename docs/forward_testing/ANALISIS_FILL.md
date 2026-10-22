# Realismo del FILL: los precios con los que FT registro sus operaciones, eran alcanzables?

> **ESTADO: CERRADO el 22/10/2026. NO PASA -- la convencion de FT queda
> CONFIRMADA por medicion y no hay nada que arreglar.** La seccion 2 se escribio
> ANTES de mirar un solo resultado y no se toco despues. Resultados en la sec. 3
> (incluida una falla de mi propia regla de lectura, sec. 3.4) y el cierre, con lo
> que NO dice y como se reabriria, en la sec. 4. **Los costos quedaron sin medir
> por decision, no por ser chicos.**

No mide si alguna estrategia tiene ventaja: eso ya esta medido y la respuesta es
que no (`ANALISIS_SALIDAS.md`, `ANALISIS_ENTRADAS.md`). Mide **si lo medido se
puede leer**: toda la historia de FT, las curvas de equity, el Sortino y el Sharpe
con IC95 y la foto de base descansan sobre precios de fill que nadie verifico que
fueran obtenibles.

---

## 1. El problema, medido

FT registra entrada y salida al **close de `fecha_datos`**, la ultima rueda que
`precios_diarios` tenia cuando corrio el bot. Ese close **ya habia ocurrido** en el
momento de decidir. No es una sospecha, es la convencion (CLAUDE.md, "FT
asincronico"): `fecha_datos` es la fecha del OHLCV usado y `fecha_entrada` la de
registro.

Lo nuevo es cuanto. Sobre 3.328 operaciones:

| desfase registro - datos | entradas | salidas |
|---|---|---|
| 0 dias | 9 (0,3%) | 12 (0,4%) |
| **1 dia** | **2.958 (88,9%)** | **2.824 (90,3%)** |
| 2-4 dias | 289 | 240 |
| 8-9 dias | 72 | 53 |

Y la hora real de escritura (`creado_en`, hora local de la maquina) **no es
post-cierre**, que es como esta descrito el flujo:

| franja local | filas | que es en el mercado US |
|---|---|---|
| 09h-11h | 1.598 | **antes de la apertura** |
| 12h-14h | 817 | durante la sesion |
| 15h-18h | 374 | sesion / cierre |
| 19h-23h | 437 | post-cierre |
| 01h-04h | 102 | madrugada |

Eso importa para el diseno: el bot que corrio a las 10h local con el close de ayer
**podia** comprar en la apertura de hoy. El que corrio a las 21h, no: su primera
chance era la apertura de manana.

## 2. PRE-REGISTRO (escrito ANTES de correr)

### Pregunta

Si cada operacion se hubiera ejecutado al **primer precio efectivamente
alcanzable** en vez de al close ya ocurrido, cuanto cambia el resultado de FT?

### Variantes

Las tres usan **las mismas operaciones**: mismos tickers, mismas fechas de
decision, mismo motivo de salida. Solo cambia el precio de fill.

| variante | definicion |
|---|---|
| `actual` (**control**) | `precio_entrada` / `precio_salida` tal como estan registrados = close de `fecha_datos` |
| **`apertura_sig`** (principal) | **apertura** de la primera rueda posterior a `fecha_datos` |
| `cierre_sig` (comparacion) | **close** de la primera rueda posterior a `fecha_datos` |

`apertura_sig` es una **cota OPTIMISTA**: supone que el bot actuo antes de esa
apertura, lo que es cierto para las corridas de la manana local y falso para las de
la noche. Se mide aparte cuantas operaciones cumplen el supuesto (comparando
`creado_en` contra la apertura de esa rueda) y se informa; **si el resultado se
sostiene incluso en la cota optimista, la conclusion no depende del supuesto.**

### Las dos patas se mueven JUNTAS

Se re-valuan **entrada y salida** en la misma variante. Mover solo la salida mediria
un artefacto que no existe: si se compra peor y se vende peor, el efecto se cancela
en parte, y **cuanto se cancela es justamente el resultado**. Una operacion que
pierda cualquiera de las dos patas se excluye y se informa.

### Datos

- `ft_operaciones` con `fecha_salida IS NOT NULL` (3.129 cerradas al 2026-10-21;
  la historia va del 2026-04-24 al 2026-10-21).
- `precios_diarios` para la apertura y el close de la rueda siguiente. Cobertura
  verificada: **3.111 de 3.129 salidas (99,4%)** tienen la rueda siguiente con
  apertura cargada.
- Todas las operaciones son `lado = long`.

### Exclusiones

- `motivo_salida = ESTRATEGIA_DISCONTINUADA` (30): salidas ARTIFICIALES de la baja
  del 17/9, no las decidio ninguna regla.
- `DELISTING` (3): no hay rueda siguiente que valga.
- Las ventanas que `ft_cambios` marca como invalidantes, con el mismo criterio que
  los analisis anteriores.
- Operaciones sin precio en alguna de las dos patas.

Cada exclusion se informa con su conteo.

### Metrica

Se trabaja en **retorno**, no en dolares: `(precio_salida - precio_entrada) /
precio_entrada`. Motivo: si cambia el precio de entrada, la cantidad de acciones
que se podian comprar con el mismo capital tambien cambia, y mezclar eso seria
medir el sizing en vez del fill. **El efecto en dolares queda fuera de alcance y se
declara.**

- **Principal**: diferencia **pareada** por operacion, `variante - actual`.
- Secundarias: retorno medio por operacion de cada variante, suma de retornos por
  estrategia, % de operaciones ganadoras, y el percentil 5 (la cola).

### Unidad independiente

La **rueda de salida**: las salidas del mismo dia comparten mercado. El IC95 se
calcula sobre las medias por rueda (`ft_salidas.ic95_por_dia`, el mismo helper que
los analisis anteriores), con t de Student.

### REGLA DE LECTURA (que cuenta como "el fill es un artefacto material")

Hacen falta las tres:

1. la diferencia pareada media difiere de cero con **IC95 que excluye el cero**;
2. **magnitud**: la diferencia se come **>= 25% del retorno medio por operacion del
   control**, *o* da vuelta el signo del retorno total de al menos una estrategia;
3. **consistencia**: el signo se repite en al menos **4 de los 6 meses** de historia
   (2026-04 a 2026-10).

El punto 2 existe porque con 3.129 operaciones casi cualquier diferencia va a
excluir el cero: la pregunta no es si difiere, es si **cambia una conclusion**. El
umbral de 25% es el que separa "corre las magnitudes" de "no da vuelta nada": todas
las conclusiones vigentes son "indistinguible de cero", y un corrimiento menor no
convierte un resultado indistinguible en otra cosa. Es la misma logica con la que el
analisis de entradas comparo la banda del sorteo (10 pp) contra la ventaja de la
mejor regla (+0,90 pp) en vez de mirar solo el IC.

**Si NO se cumplen las tres**: la convencion queda **confirmada por medicion** y se
cierra. Es un resultado util, no un no-resultado.

### Lo que este analisis NO contesta

- No dice si alguna estrategia tiene ventaja.
- No modela **spread, comisiones ni slippage**: mide el precio que SI se podia
  conseguir, no el costo de conseguirlo. Es otra pregunta, despues.
- No re-simula decisiones: las entradas y salidas son las que fueron. Cambiar el
  precio de fill podria haber cambiado que stop se gatillaba y cuando, y eso NO se
  modela -- seria un backtest nuevo, no una re-valuacion.
- No toca dolares ni cantidad de acciones (ver Metrica).

### Herramientas

- `src/utils/ft_fill.py` -- modulo PURO (sin DB), con los tests de
  `tests/test_ft_fill.py`.
- `scripts/forward_testing/ft_analisis_fill.py` -- solo lee. Salida en
  `reportes/analisis_fill/AAAAMMDD_<etiqueta>/` con su `parametros.json`.
- **No va a `ft_cambios`**: no lo lee ningun bot, es analisis (mismo criterio que
  `ft_backtesting_runner.py`).

---

## 3. RESULTADOS (corrido el 22/10/2026, commit 830a94a)

> **NO PASA. La convencion de FT queda CONFIRMADA por medicion.** El fill
> registrado no es un artefacto que cambie ninguna conclusion. Nada que arreglar.

### 3.1 La muestra

| | |
|---|---|
| operaciones cerradas en la DB | 3.129 |
| menos `ESTRATEGIA_DISCONTINUADA` | -30 |
| menos `DELISTING` | -3 |
| menos la ventana de `fix_score_cero_salida` | **-576** |
| menos sin precio en alguna pata | -15 |
| **entran al analisis** | **2.505 operaciones en 97 ruedas** |

El supuesto de `apertura_sig` (que el bot actuo antes de esa apertura) **se cumple
en 2.500 de 2.505, el 99,2%**, y falla en 5. O sea que la cota optimista casi no es
optimista: el resultado no depende del supuesto. Eso es consecuencia de la hora real
de corrida de la seccion 1 -- las corridas de la manana local son anteriores a la
apertura de Nueva York.

### 3.2 El resultado principal

| | control (`actual`) | `apertura_sig` | `cierre_sig` |
|---|---|---|---|
| retorno medio por operacion | +0,160 % | +0,194 % | +0,139 % |
| **diferencia pareada (rueda como unidad)** | -- | **-0,002 pp** | **-0,161 pp** |
| IC95 de esa diferencia | -- | **[-0,190; +0,187]** | **[-0,455; +0,134]** |
| diferencia por operacion | -- | +0,034 pp | -0,021 pp |
| operaciones ganadoras | 37,8 % | 40,6 % | 44,0 % |
| cola (percentil 5) | -9,12 % | -9,01 % | -9,94 % |

**La diferencia es cero.** -0,002 pp con IC95 que cruza el cero de lado a lado: no
hay efecto que medir. Es exactamente la cancelacion que el diseno anticipaba -- se
compra peor y se vende peor, y el retorno queda igual. **Haber movido las dos patas
juntas es lo que permite verlo**; con solo la salida habria aparecido un efecto que
no existe.

Y el signo, dentro del ruido, va **en contra** de la sospecha: ejecutar en la
apertura siguiente habria dado un retorno marginalmente MEJOR (+0,194 % contra
+0,160 %). La convencion no favorece a FT.

Un detalle descriptivo que si se mueve: la **tasa de ganadoras** sube (37,8% ->
40,6% -> 44,0%) sin que suba el retorno medio. Mas operaciones terminan en positivo
y la distribucion lo compensa por magnitud. No cambia ninguna conclusion, pero
conviene saberlo antes de leer un "% de acierto" como si fuera robusto.

### 3.3 La regla de lectura, condicion por condicion

| condicion pre-registrada | `apertura_sig` | `cierre_sig` |
|---|---|---|
| 1. IC95 excluye el cero | **NO** | **NO** |
| 2. material (>= 25% del control, o da vuelta un signo) | SI (solo por signo) | SI |
| 3. mismo signo en >= 4 de 6 meses | **NO** (3 de 7) | **NO** (3 de 7) |

Hacian falta las tres. Fallan la 1 y la 3 en las dos variantes.

### 3.4 Mi regla de lectura tenia una falla, y la dejo escrita

La condicion 2 se activo por "da vuelta el signo del total de alguna estrategia":
`FT_COMBO_v1` (-48,75 -> +11,62) y `FT_SMC_v1` (+0,45 -> -14,03). **Las dos son
artefactos del criterio, no hallazgos**, y el criterio esta mal especificado:

- **`FT_SMC_v1` tenia un total de +0,45 pp**, o sea indistinguible de cero. Dar
  vuelta el signo de algo que ya vale cero no informa nada. **La condicion deberia
  haber exigido que el total del control fuera distinguible de cero** antes de
  mirarle el signo. No lo exigi.
- **`FT_COMBO_v1` mueve 60 pp, que suena enorme, y es `n` por ruido**: son 452
  operaciones con una diferencia media de +0,100 pp por operacion, IC95
  [-0,185; +0,386], que incluye el cero. **Sumar retornos por estrategia amplifica
  el n y disfraza una diferencia no significativa de efecto dramatico.**

Leccion para cualquier grilla futura de este proyecto: **un total por estrategia no
es una metrica de significancia.** Es media x n. Si se va a usar el signo del total
como criterio, el total tiene que venir con su IC.

### 3.5 Por estrategia (complemento DESCRIPTIVO, no pre-registrado)

Diferencia media por operacion de `apertura_sig`, con IC95 y la rueda como unidad:

| estrategia | ops | ruedas | dif. media | IC95 | excluye 0 |
|---|---|---|---|---|---|
| FT_ML_SCANNER_v1 | 226 | 81 | +0,326 | [+0,020; +0,631] | **SI** |
| FT_SMC_v3_N5 | 5 | 4 | +0,729 | [-3,831; +5,289] | no |
| FT_TECH_SECTOR_OPTIONS_v2 | 216 | 69 | +0,238 | [-0,112; +0,587] | no |
| FT_TECH_SECTOR_OPTIONS_v1 | 216 | 68 | +0,106 | [-0,291; +0,503] | no |
| FT_COMBO_v1 | 452 | 72 | +0,100 | [-0,185; +0,386] | no |
| FT_TECH_SECTOR_v2 | 493 | 84 | +0,020 | [-0,216; +0,257] | no |
| FT_SMC_v2 | 27 | 21 | +0,002 | [-0,530; +0,533] | no |
| FT_SMC_v3_N3 | 5 | 5 | -0,017 | [-1,871; +1,837] | no |
| FT_TECH_SECTOR_v1 | 460 | 82 | -0,025 | [-0,337; +0,287] | no |
| FT_ML_SCANNER_v2 | 71 | 22 | -0,117 | [-0,903; +0,668] | no |
| FT_TECH_v1 | 77 | 51 | -0,287 | [-1,113; +0,539] | no |
| FT_TECH_SECTOR_OIEXIT_v1 | 202 | 69 | -0,343 | [-0,945; +0,259] | no |
| FT_SMC_v1 | 55 | 40 | -0,421 | [-0,941; +0,099] | no |

**12 de 13 incluyen el cero.** El unico que lo excluye, `FT_ML_SCANNER_v1`, lo hace
raspando (el extremo inferior es +0,020) y **con 13 comparaciones un resultado al
95% asi se espera por azar**: 0,05 x 13 = 0,65 falsos positivos esperados. Es el
mismo "contar las comparaciones" de docs/features_ml.md. No se le da estatus de
hallazgo.

El mes de abril es el unico con una diferencia grande (+1,666 pp) y tiene **27
operaciones en 3 ruedas**: la historia arranca el 2026-04-24. Es n chico, no un
tramo.

### 3.6 Que queda cerrado con esto

- **Las conclusiones de FT se leen como estan.** `ANALISIS_SALIDAS.md`,
  `ANALISIS_ENTRADAS.md`, `ft_equity_diaria`, el Sortino y el Sharpe con IC95 y la
  foto de base del 11/9 **no necesitan asterisco por el fill**.
- **La convencion `fecha_datos` queda validada tambien en precio**, no solo como
  convencion de cruce. Ya se sabia que hay que cruzar por ella (CLAUDE.md, "FT
  asincronico"); ahora se sabe que el precio que registra era alcanzable.
- **Cierra tambien la Etapa 3 de `docs/fuentes_ibkr.md`** (el intradiario de IBKR
  para medir el timing de ejecucion): la pregunta se contesto con
  `precios_diarios`, sobre la historia completa, y dio cero. No hay motivo para
  traer barras intradiarias a buscar el mismo efecto.

## 4. CIERRE (22/10/2026, por decision del usuario)

**La investigacion se cierra aca.** No se sigue buscando en esta direccion. El
criterio fue explicito: sin diferencias significativas, no se continua.

### Que dice el cierre

- **El fill registrado de FT era alcanzable.** Diferencia pareada -0,002 pp,
  IC95 [-0,190; +0,187], sobre 2.505 operaciones en 97 ruedas. La convencion
  `fecha_datos` queda validada en precio, no solo como convencion de cruce.
- **Ninguna conclusion previa de FT necesita asterisco por el fill.**
- **La Etapa 3 del plan de IBKR queda cerrada sin usar IBKR** (`docs/fuentes_ibkr.md`).

### Que NO dice el cierre

- **No midio los COSTOS.** Spread, comisiones y slippage quedan **sin medir, por
  decision**. Y conviene no olvidar el orden de magnitud: con un retorno medio de
  +0,160 % por operacion, un costo por operacion del orden del spread **se come el
  resultado entero**. Que no se haya medido no significa que sea chico: significa
  que no se busco.
- **No dice que las estrategias funcionen.** Eso ya esta medido y la respuesta es
  que no (`ANALISIS_SALIDAS.md`, `ANALISIS_ENTRADAS.md`). Este analisis solo dice
  que los numeros con los que se midio eran legitimos.
- **No re-simulo decisiones.** Con otro precio de fill, otro stop podria haber
  gatillado en otro momento. Seria un backtest nuevo, no una re-valuacion.
- **No cubre la ventana de `fix_score_cero_salida`** (576 operaciones excluidas).

### Como se reabriria

Con una de estas, no por iniciativa propia:

1. que aparezca intencion de operar con dinero real una estrategia de FT -- ahi los
   costos dejan de ser una curiosidad y pasan a ser la cuenta;
2. que cambie la convencion de ejecucion de los bots (por ejemplo si pasaran a
   decidir intradia), porque el supuesto medido aca dejaria de aplicar;
3. que FT acumule bastante mas historia Y alguna estrategia muestre una ventaja
   medida contra su control, caso en el que el margen importa de nuevo.

### Lo que queda, aunque la linea este cerrada

- `src/utils/ft_fill.py` + `tests/test_ft_fill.py` (28 tests) y
  `scripts/forward_testing/ft_analisis_fill.py`: **re-correrlo es un comando**, no
  un proyecto. Sale 0 si la convencion sigue confirmada.
- Las dos lecciones de metodo de la sec. 3.4-3.5, que ya viven en CLAUDE.md porque
  valen para cualquier grilla futura de este proyecto, no solo para esta.
- Este documento, con el pre-registro intacto y el resultado negativo escrito.
