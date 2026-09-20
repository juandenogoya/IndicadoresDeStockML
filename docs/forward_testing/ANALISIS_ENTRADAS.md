# Analisis de ENTRADAS -- TECH_SECTOR_v1

Creado: 19/9/2026  -  ampliado 20/9/2026 (seccion 6 y pre-registro del paso 1)
Motor: `scripts/forward_testing/ft_analisis_entradas.py` + `src/utils/ft_entradas.py`
Corrida de referencia: `reportes/analisis_entradas/20260920_paso0/`

---

## 1. La idea

El [analisis de salidas](ANALISIS_SALIDAS.md) cerro con un resultado claro: despues de
salir, la accion hace lo que el universo, en las dos familias de salida que existen en
FT. La contracara es esta: **si la salida no distingue, lo que queda es la entrada**.

Este documento es el espejo del otro. Alla las entradas estaban fijas y se variaba la
salida; aca la salida queda fija -- la de TECH_SECTOR_v1, sin tocar -- y se varia la
entrada. Ademas queda fijo el **reparto** (5 por sector), para no perder la comparacion
contra la estrategia que corre hoy en FT.

Con eso, la entrada es la unica variable.

### Por que TECH_SECTOR_v1 y no TECH_v1

TECH_v1 desempata **por orden alfabetico** y sin particion sectorial no llega a mirar
todo el escenario: su historia no mide su regla de entrada, mide la regla mas el
alfabeto. Es el mismo caracter que el bug del score 0,0 -- un resultado que parece de
la estrategia y es de otra cosa. Sirve como benchmark de que algo se opero, no como
evidencia sobre la entrada.

TECH_SECTOR_v1 reparte en 9 sectores, mira mas tickers y ademas tiene el trabajo previo
hecho: backtest de referencia 2021-2026 (+24,4%), el control vivo en FT y la maquinaria
de re-simulacion ya validada.

### La vara que hay que superar

El backtest pre-registrado de `docs/estructura_velas.md` sec. 9.3 midio TECH_SECTOR_v1
sobre 2021-09 -> 2026-09: **+24,4%** contra un universo equal-weight de **+86,2%**, con
40% de exposicion, y le gana al universo ajustado por exposicion en **2 de 6 anios**.
Con ~960 operaciones por anio y sin costos.

Una entrada nueva no compite contra las otras estrategias de FT: compite contra eso.

---

## 2. Que cubre este documento

El trabajo esta pensado en tres pasos. Al 20/9/2026 estan hechos el 0 y el 1:

| Paso | Que varia | Estado |
|---|---|---|
| **0** | nada -- se cuenta donde cae el universo | **hecho** (secciones 4 a 8) |
| **1** | combinaciones booleanas de las condiciones | **hecho**: pre-registro en la seccion 9, resultados en la 10. **0 de 166 reglas pasan** |
| 2 | ponderadores sobre alertas por zona | pendiente, y ahora con la banda del sorteo como vara (seccion 10.2) |
| 1b / 1c | descomponer la decision; factorial entrada x salida | **disenados y sin correr** (seccion 11) |

El paso 0 **no mide rendimiento**: no calcula retornos, no simula cartera y no compara
reglas. Cuenta. Existe porque un peso sobre un estado que ocurre 50 veces en cinco anios
no es una palanca, y porque la forma de la grilla depende de lo que se cuente aca.

Y porque sigue valiendo la regla del proyecto: medir el retorno forward de una senal de
entrada a plazo fijo es una prueba de PREDICCION y no evalua una estrategia de reglas.
Lo que decide es el backtest con entradas **y** salidas, en el paso 1.

---

## 3. Definiciones

### 3.1 La regla actual, enumerada

Con los pesos de `src/strategies/scoring.py` (SMA50 2,0 | SMA21 1,0 | MACD 1,5 |
RSI 1,0) y umbral 4,0, solo 4 de los 16 estados binarios llegan al umbral:

| SMA50 | SMA21 | MACD | RSI | Score |
|:---:|:---:|:---:|:---:|---:|
| si | si | si | si | 5,5 |
| si | si | si | -- | 4,5 |
| si | -- | si | si | 4,5 |
| si | si | -- | si | 4,0 |

Es decir, la entrada de TECH_SECTOR_v1 es exactamente:

> **close > SMA200  Y  close > SMA50  Y  (al menos 2 de {SMA21, MACD, RSI})**

Dos consecuencias:

1. **SMA50 no "pesa mas": es obligatoria de hecho.** Sin ella el maximo alcanzable es
   1,0 + 1,5 + 1,0 = 3,5 < 4,0. El peso 2,0 no es una ponderacion, es un segundo filtro
   obligatorio disfrazado de puntaje.
2. **El umbral no es un parametro continuo.** Todo su rango produce cinco reglas
   booleanas distintas (>=3,0 | >=3,5 | >=4,0 | >=4,5 | >=5,5), y muchos valores
   intermedios son literalmente la misma regla. Moverlo de 4,0 a 4,5 no es "ser 12% mas
   exigente": es saltar a `SMA200 Y SMA50 Y MACD Y (SMA21 O RSI)`.

`ft_entradas.regla_v1_booleana` escribe eso sin pesos, con una implementacion
independiente, y la seccion de fidelidad lo compara fila por fila contra
`calcular_score_tecnico(...) >= 4,0`.

**La condicion MACD es una sola condicion escrita dos veces.** En `scoring.py`,
`hist` se define localmente como `macd - signal` y la condicion es
`macd > signal and hist > 0`: no hay ningun estado del mundo donde una se cumpla y la
otra no. Es inofensiva -- no cambia ningun resultado -- pero donde parecia haber dos
palancas de momentum hay una sola.

### 3.2 Zonas por distancia (cortes en % fijo)

SMA50 y SMA21 dejan de ser un bit y pasan a zonas de `dist_sma* = (close-sma)/sma*100`:

| Eje | Zonas |
|---|---|
| **SMA50** | `S50_BAJA` < -10 / `S50_MEDIA_NEG` -10 a -5 / `S50_CERCA_NEG` -5 a 0 / `S50_CERCA` 0 a +5 / `S50_MEDIA` +5 a +10 / `S50_ALTA` > +10 |
| **SMA21** | `S21_BAJA` < -6,5 / `S21_MEDIA_NEG` -6,5 a -3,5 / `S21_CERCA_NEG` -3,5 a 0 / `S21_CERCA` 0 a +3,5 / `S21_MEDIA` +3,5 a +6,5 / `S21_ALTA` > +6,5 |
| **MACD** | `MACD_UP` / `MACD_DOWN` (macd vs senal 12/26/9) |
| **RSI** | `RSI_LOW` < 45 / `RSI_IN` [45, 68] / `RSI_HIGH` > 68 |

Los cortes de SMA21 salen de escalar los de SMA50 por raiz(50/21) = 1,543 -- **criterio
declarado antes de mirar la distribucion**, no ajustado a ella. Da 6,49 y 3,25;
redondeados a 6,5 y 3,5.

El RSI se desglosa en tres zonas para **medir** (fallar por debilidad y por fuerza son
lecturas opuestas) y sigue siendo binario para **decidir**, igual que hoy, para mantener
la comparabilidad con el control vivo.

**Convencion de bordes**: intervalos `(a, b]`, abiertos por izquierda. Asi la frontera
en 0 separa exactamente igual que `close > sma`. Con intervalos `[a, b)` una fila con
`close == sma` quedaria del lado positivo y la condicion binaria diria que no.

### 3.3 Unidad de los cortes: porcentaje fijo, y lo que eso implica

Este paso usa **porcentaje fijo, igual para los 200 tickers**. Es deliberado y es la
primera de las tres calibraciones previstas:

1. universal (este paso),
2. por grupo de volatilidad -- que replantea los rangos de cada indicador,
3. por sector.

Lo que hay que vigilar y esta medido mas abajo: un 5% sobre la SMA50 son mas de cuatro
dias de rango en un ticker con ATR% 1,2 y un dia cualquiera en uno con ATR% 5,0. Con
cortes fijos, las acciones volatiles caen sistematicamente en las zonas extremas. Si
esas zonas puntuaran alto, la cartera se llenaria de los tickers mas volatiles de cada
sector sin que nadie lo haya decidido.

---

## 4. Metodo

Periodo **2021-09-01 -> 2026-09-18** (el mismo del backtest pre-registrado), universo
`activos` con `activo = TRUE`, join de `indicadores_tecnicos` con `precios_diarios` por
(ticker, fecha). **178.491 filas, 200 tickers, 1.267 ruedas.**

Las distancias se **recomputan** desde `close` y `sma*` en vez de leer la columna
`dist_sma*`, que esta redondeada a 4 decimales: una fila con el close apenas por encima
de la media puede tener `dist` guardada 0,0000 y caer en la zona negativa mientras la
condicion binaria dice True. El script cuenta cuantas filas clasificarian distinto
(3.290 en SMA50, 4.231 en SMA21) y ademas audita la columna.

Cobertura del universo: 122 tickers en 2021-2024 y 200 desde 2025. Es la misma
limitacion del backtest de referencia y hay que declararla en cualquier lectura por
anios.

---

## 5. Resultados del paso 0 (19/9/2026)

### 5.1 Fidelidad -- la maquinaria reproduce la regla del bot

| Comparacion | Diferencias |
|---|---|
| `score >= 4,0` vs `SMA200 y SMA50 y 2 de 3` | **0** sobre 178.491 filas |
| condiciones derivadas de las ZONAS vs las de `scoring.py` | **0** |
| columna `entra` (vectorizada) vs la escalar | **0** |

Sin esto no se le puede creer nada a lo que sigue: si la maquinaria nueva no reproduce
la regla vieja, cualquier comparacion posterior mide la maquinaria.

### 5.2 El filtro casi no filtra

| | Filas | % |
|---|---|---|
| Total | 178.491 | 100% |
| Pasan el filtro SMA200 | 100.215 | 56,15% |
| **Cumplen la entrada actual** | **59.775** | **33,49% del total / 59,65% de las que pasan** |

Una de cada tres filas del universo califica para entrar. En candidatos por rueda:

| | Valor |
|---|---|
| Candidatos por rueda (media / mediana) | **48,7 / 47** |
| p10 / p90 | 12 / 89 |
| Lugares disponibles (5 x 9 sectores) | **45** |
| Ruedas con mas candidatos que lugares | **52,5%** |
| Candidatos por sector-rueda (media / mediana / p90) | 5,8 / 5 / 12 |
| **Sector-ruedas con mas de 5 candidatos** | **42,8%** |

En cuatro de cada diez sector-ruedas hay mas candidatos que lugares. En esos casos **la
entrada no decide quien entra: decide el ranking**.

### 5.3 Y el ranking tiene tres niveles, con la mitad en el tope

El score de los candidatos solo puede tomar tres valores:

| Score | % de los candidatos |
|---|---|
| 4,0 | 25,84% |
| 4,5 | 25,68% |
| **5,5** | **48,49%** |

Con score maximo (5,5) hay **24,2 candidatos por rueda de media**. Todos empatados.

**Como desempata el bot:** la consulta de candidatos trae `ORDER BY a.sector, i.ticker`
(`ft_bot_tech_sectorial.py:154`), el cerebro hace `append` en ese orden y ordena con
`sort(key=score, reverse=True)` (`src/strategies/sectorial.py:202`). El `sort` de Python
es **estable**: entre empatados conserva el orden previo, que es **alfabetico**.

Es decir: el desempate alfabetico que hace inservible a TECH_v1 **tambien esta en la
sectorial**. Lo que cambia es que reparte en 9 grupos y por eso mira mas tickers, pero
dentro de cada sector el criterio final sigue siendo el alfabeto -- y muerde en el 42,8%
de los sector-ruedas.

Esto es el equivalente, del lado de la entrada, al hallazgo de que ningun score caia
entre 3,5 y 4,0: **el ranking es una palanca de primer orden y hoy no esta definido.**
COMBO_v1 habia nacido justamente para desempatar (candle score) y se midio que ese
desempate puntual no aportaba; lo que nunca se midio es cuanto decide el desempate en
general.

### 5.4 Influencia: cuanto decide cada condicion

Fraccion de filas en las que **voltear esa condicion cambia la decision de entrada**. Es
la medida directa del aporte de cada condicion a una regla booleana: una condicion que
casi nunca es pivotal no decide, tenga el peso que tenga.

| Condicion | Influencia | Peso en el score |
|---|---:|---:|
| SMA200 | **42,52%** | filtro |
| SMA50 | **37,26%** | 2,0 |
| SMA21 | **23,88%** | 1,0 |
| MACD | 16,08% | 1,5 |
| RSI | **9,58%** | 1,0 |

Dos lecturas:

- **SMA21 no es redundante.** Era la candidata a salir por solaparse con SMA50 y con el
  MACD, y resulta la tercera condicion mas influyente: decide en el 23,9% de las filas,
  dos veces y media mas que el RSI. Quitarla no es limpiar una condicion que sobra.
- **El orden de influencia no sigue al de los pesos.** El MACD pesa 1,5 y decide menos
  que SMA21, que pesa 1,0. El RSI pesa 1,0 y decide en menos de una de cada diez filas.

### 5.5 Distribucion de estados

| | Valor |
|---|---|
| Estados observados | **120** de 216 posibles |
| Concentran el 90% de la masa | **30 estados** |
| Concentran el 99% de la masa | 64 estados |
| Con menos de 200 casos | 71 estados (2,76% de la masa) |

El espacio efectivo es de 30 a 64 estados, no 216. Es enumerable y leible.

Los cinco estados mas frecuentes (sobre las filas que pasan el filtro SMA200):

| Estado | Casos | % | Entra |
|---|---:|---:|:---:|
| `S50_ALTA S21_ALTA MACD_UP RSI_HIGH` | 9.298 | 9,28% | **si** |
| `S50_CERCA S21_CERCA MACD_UP RSI_IN` | 8.712 | 8,69% | si |
| `S50_CERCA S21_CERCA_NEG MACD_DOWN RSI_IN` | 7.893 | 7,88% | no |
| `S50_MEDIA S21_CERCA MACD_DOWN RSI_IN` | 5.693 | 5,68% | si |
| `S50_CERCA S21_CERCA MACD_DOWN RSI_IN` | 5.023 | 5,01% | si |

El estado **mas frecuente de todos** es el mas estirado posible con el RSI en
sobrecompra -- y **entra**, porque cumple SMA50, SMA21 y MACD (dos de tres, sin RSI).
La estrategia compra sobrecompra estirada mas seguido que ninguna otra cosa.

### 5.6 Cruces: SMA50 contra SMA21

| | Valor |
|---|---|
| Masa en la diagonal (misma posicion relativa) | 47,56% |
| Concordancia binaria SMA50 / SMA21 | 81,46% |
| Solo SMA21 (sobre la 21, bajo la 50) | 4,92% |
| Solo SMA50 (sobre la 50, bajo la 21) | 13,62% |

Mas de la mitad de la masa cae fuera de la diagonal: las dos distancias no son la misma
informacion. Consistente con la influencia de 23,9% de SMA21.

### 5.7 Cruces: la distancia y el RSI se pelean

Porcentaje de filas con el RSI **dentro** de la banda 45-68, por zona de distancia:

| Zona | % RSI en banda (SMA21) | % RSI en banda (SMA50) |
|---|---:|---:|
| `*_BAJA` | 11,38% | 7,95% |
| `*_MEDIA_NEG` | 22,30% | 19,80% |
| `*_CERCA_NEG` | 74,45% | 54,85% |
| `*_CERCA` | **95,83%** | **94,56%** |
| `*_MEDIA` | 73,84% | 81,03% |
| `*_ALTA` | **41,22%** | **48,92%** |

Confirmado numericamente lo que se sospechaba: **la zona de distancia alta y el RSI en
banda son casi incompatibles**. En `S21_ALTA`, de las 18.026 filas, 10.595 tienen el RSI
por encima de 68 y **ninguna** por debajo de 45.

Consecuencia directa para el paso 2: si se le da peso positivo a la zona de distancia
alta mientras el RSI resta por estar fuera de banda, **el score se contradice a si
mismo** -- premia por un lado lo que castiga por el otro, y el saldo depende de numeros
elegidos casi al azar. Antes de ponderar hay que declarar el **sentido** de cada zona:
si estar muy estirado es bueno (momentum) o malo (riesgo de reversion) es una afirmacion
de AT que se escribe antes de ver resultados.

### 5.8 Por que lado falla el RSI

De las 32.470 filas que pasan el filtro SMA200 y tienen el RSI fuera de banda:

| Motivo | Casos | % |
|---|---:|---:|
| Por debilidad (< 45) | 15.957 | 49,14% |
| Por fuerza (> 68) | 16.513 | 50,86% |

Mitad y mitad. **La mitad de las veces que el RSI quita su punto, lo hace por exceso de
fuerza**, en una estrategia cuyo resto compra fuerza. Es la unica condicion no monotona
del sistema y la unica que castiga el momentum.

---

## 6. Los ponderadores, ?eligen tickers distintos? (20/9/2026)

Es la pregunta que decide si el paso 1 tiene sentido: si todos los juegos de pesos
eligieran los mismos tickers, no habria nada cuyo rendimiento comparar.

**Metodo (PROXY, declarado):** top-5 por (rueda, sector) con el ranking del bot -- score
descendente y, entre empatados, el orden en que vienen los candidatos, que es
alfabetico. Arma el top **de cero cada rueda**; la estrategia real mantiene posiciones y
solo llena lugares vacios, asi que esto **sobreestima** cuanto cambia la operacion.
Responde "hay elecciones distintas", no "cuanto rinden".

**Juegos pre-declarados** (`ft_entradas.JUEGOS_PESOS`), con su criterio de construccion:
la v1; los dos umbrales vecinos con los pesos de la v1 -- los unicos dos que producen
una regla distinta; y subir el peso de cada condicion no obligatoria hasta un valor que
cambie la regla booleana, una por vez. Los seis producen **seis reglas booleanas
distintas** (ningun duplicado), verificado por `reglas_distintas()`.

Base v1: 37.679 selecciones, 29,7 por rueda.

| Juego | Califican | Elegidos | Distintos de v1 | Jaccard | Niveles de score | % en el tope |
|---|---:|---:|---:|---:|---:|---:|
| **v1** (2,0/1,0/1,5/1,0 u4,0) | 59.775 | 37.679 | -- | 1,000 | 3 | 48,5% |
| umbral 3,5 | 64.411 | 39.247 | 1.568 | 0,960 | 4 | 45,0% |
| SMA21 fuerte (1,0 -> 2,5) | 65.704 | 39.467 | 1.910 | 0,952 | 6 | 44,1% |
| MACD fuerte (1,5 -> 3,0) | 66.265 | 40.083 | 3.474 | 0,914 | 4 | 43,7% |
| umbral 4,5 | 44.330 | 31.366 | 6.313 | 0,833 | 2 | 65,4% |
| **RSI fuerte (1,0 -> 3,0)** | 78.492 | 43.798 | **13.271** | **0,720** | 7 | 36,9% |

**Si hay elecciones distintas**: entre 4% y 28% de la seleccion cambia segun el juego.
No es ruido, y tampoco es todo -- coherente con un filtro que califica a un tercio del
universo, la mayor parte de la cartera se mantiene.

Tres lecturas:

1. **Los pesos cambian dos cosas a la vez y hay que separarlas.** "RSI fuerte" hace
   calificar 78.492 filas contra 59.775: con peso 3,0 un candidato con solo SMA50 + RSI
   llega a 5,0 y entra, cuando antes sumaba 3,0 y quedaba afuera. Parte del 28% es
   **laxitud** (entra mas gente) y parte es **reordenamiento** (entra otra gente).
   Medir el rendimiento sin separarlos es comparar exposicion y llamarlo calidad de
   entrada -- la misma trampa que documento el paso 2 del analisis de salidas.
2. **Mover solo el umbral con los pesos fijos da reglas ANIDADAS** (recorta la lista por
   abajo, no reordena). Cambiar los pesos da acceso a otras reglas booleanas, que no son
   anidadas: ahi esta el margen real.
3. **Mas niveles de score = menos alfabeto.** La v1 tiene 3 niveles con 48,5% empatado en
   el tope; "RSI fuerte" tiene 7 con 36,9%. Es un criterio para elegir pesos que **no
   depende del rendimiento**: cuantos mas niveles, menor la fraccion de la decision que
   hoy toma el orden alfabetico. Y al reves, el umbral 4,5 lo empeora (65,4%) aunque sea
   mas exigente.

---

## 7. Hallazgo colateral: indicadores desalineados con `precios_diarios`

La auditoria de `dist_sma*` encontro **25.375 filas (14,2%) con diferencias que el
redondeo no explica**. Al seguirlo aparecio algo mas de fondo.

**Que pasa:** en **51 de 200 tickers**, las columnas `sma21/50/200` de
`indicadores_tecnicos` no corresponden al `close` que hoy tiene `precios_diarios`.
Verificado recomputando la SMA desde los precios: en JPM y MO, **64,7% de las filas**
difieren, desde 2021-03-30 hasta 2025-01-27. AAPL da 0 diferencias.

| | Filas afectadas | Cambia la condicion binaria |
|---|---:|---:|
| `sma21` | 12.321 (6,90%) | 1.711 (0,96%) |
| `sma50` | 12.321 (6,90%) | 1.299 (0,73%) |
| `sma200` | 12.321 (6,90%) | 803 (0,45%) |

Los tickers afectados (JPM, MO, AXP, ACN, AVGO, BBD, DE, ERIC...) son pagadores de
dividendo, y el desfase es multiplicativo y creciente en el tiempo: la firma de una
serie que fue re-descargada o re-ajustada en algun momento sin que los indicadores de
ese tramo se recomputaran. **No afecta a 2026 y casi nada a 2025** (435 filas): el
pipeline actual esta bien; es historia vieja que quedo.

**Alcance real:** cambia menos del 1% de las condiciones binarias, asi que no invalida
nada de lo medido aca -- este paso recomputa las distancias desde `close` y `sma`, y la
fidelidad da 0 diferencias. Pero **`indicadores_tecnicos` es exactamente lo que lee el
backtest historico** (`bt_data_loader.py:156`) mientras ejecuta con los precios de
`precios_diarios`: el backtest de referencia de 5 anios tiene ese ruido adentro.

No se arregla en esta tarea: tocar `indicadores_tecnicos` es una correccion aparte, con
su propia verificacion. Queda anotado y medido.

Es otro caso de la familia conocida: **que una tabla reproduzca el codigo que la escribe
no significa que corresponda a la tabla con la que se la cruza.**

---

## 8. Que deja el paso 0 para el paso 1

1. **El filtro casi no filtra y el ranking decide.** 48,7 candidatos por rueda para 45
   lugares, 42,8% de sector-ruedas con mas candidatos que lugares, 48,5% de los
   candidatos empatados en el score maximo y desempate **alfabetico**. Cualquier grilla
   que varie solo el filtro, dejando el ranking como esta, puede dar "ninguna variante
   mejora" -- y la lectura honesta no seria "el filtro no importa" sino que se lo midio
   con la palanca equivocada fija.
2. **SMA21 se queda.** Es la tercera condicion mas influyente (23,9%), por encima del
   MACD. La duda sobre quitarla queda respondida con datos.
3. **El espacio de estados es chico**: 120 observados, 30 concentran el 90%. Los 71
   estados con menos de 200 casos (2,76% de la masa) no sostienen un ponderador propio y
   conviene fusionarlos antes del paso 2.
4. **Distancia y RSI se pelean**: hay que declarar el sentido de cada zona antes de
   ponderar, no descubrirlo en la grilla.
5. **El umbral no es un dial.** La grilla se expresa en reglas booleanas, no en pesos y
   umbrales: es el mismo espacio, pero legible, y evita que cientos de filas sean seis
   decisiones repetidas.

---

## 9. PRE-REGISTRO del paso 1 (escrito ANTES de correr, 20/9/2026)

### 9.1 Que se fija y que se varia

**Fijo** (no se toca nada de esto): universo `activos` activo=TRUE; periodo
2021-09-01 -> 2026-09-18; reparto 5 por sector sobre los 9 sectores; **la salida de
TECH_SECTOR_v1** (score <= 3,5, SL 2xATR14, TP 4xATR14, earnings manana, todo con sus
prioridades); sizing y capital; y el **desempate alfabetico**, tal como lo hace el bot.

**Varia**: solo la regla booleana de entrada.

### 9.2 El espacio: 166 reglas, enumeradas

Todas las reglas **monotonas** no triviales sobre {SMA50, SMA21, MACD, RSI}, con SMA200
como filtro obligatorio previo. Monotonas = cumplir una condicion mas nunca puede sacar
a un candidato; es el unico supuesto, y es el mismo que hace el score de hoy al usar
pesos no negativos.

| | |
|---|---|
| Reglas monotonas sobre 4 condiciones (Dedekind M(4)) | 168 |
| No triviales (ni "siempre" ni "nunca") | **166** |
| De esas, expresables como score ponderado + umbral | **148** |
| Candidatos por rueda: min / mediana / max | 22,9 / 53,6 / 68,5 |
| Reglas que quedan sin masa suficiente | **ninguna** |

**Por que reglas booleanas y no una grilla de pesos.** Las 166 cubren *todo* lo que
cualquier juego de ponderadores puede expresar, y 18 reglas mas que **ningun score
lineal alcanza**. Una grilla de pesos, ademas, repite la misma decision muchas veces
(seccion 3.1 y 6): en el analisis de salidas, 3.750 juegos de pesos colapsaron en 589
reglas distintas, y la regla actual aparecio 4 veces. Al
enumerar reglas, cada fila de la grilla es una decision distinta por construccion.

La v1 esta en el espacio y es una de las 148 de umbral: **es el control**.

### 9.3 Control de aleatorizacion: la banda del sorteo

Ademas de las 166, se corre la **v1 con desempate por sorteo** con 20 semillas fijas
(0 a 19), todo lo demas identico.

Motivo: el paso 0 midio que el desempate decide en el 42,8% de los sector-ruedas y hoy
lo resuelve el alfabeto. La dispersion entre esas 20 corridas es **el piso de ruido del
experimento**: una diferencia entre reglas menor que esa banda no es atribuible a la
regla, es la loteria del orden. Es un control, no una variante a desplegar.

### 9.4 Metricas, por regla

Retorno total  -  max drawdown  -  operaciones  -  retorno medio por operacion  -  **exposicion
media**  -  retorno por anio  -  retorno/exposicion por anio contra el universo
equal-weight  -  candidatos por rueda  -  exceso por operacion contra el universo de los
mismos dias.

**Exposicion y candidatos por rueda son obligatorios en cada fila**: el tope normaliza
por arriba pero no por abajo, y una regla estricta opera con capital ocioso. Sin esas
dos columnas la grilla premia a las reglas laxas por construccion (seccion 6, lectura 1).

### 9.5 Regla de lectura (fijada antes de ver un solo resultado)

Particion: **seleccion 2021-09-01 -> 2024-12-31**, **confirmacion 2025-01-01 ->
2026-09-18**. La unidad de independencia es el **dia**, no la operacion.

Una regla **pasa** si cumple las cuatro:

1. en **seleccion**, le gana a la v1 en retorno/exposicion **y** su exceso por operacion
   contra la v1 tiene IC95 que excluye el cero;
2. en **confirmacion**, mantiene el signo (no se exige significancia: el tramo es corto);
3. la diferencia contra la v1 **supera la banda del sorteo** de 9.3;
4. le gana al universo ajustado por exposicion en **al menos 4 de los 6 tramos anuales**
   -- el mismo criterio del backtest de referencia.

Se reporta ademas la **correlacion de orden entre seleccion y confirmacion** sobre las
166. Si es baja (como el -0,16 del paso 2 de salidas), ninguna conclusion sobre una
regla individual es fiable, por mas que pase las cuatro condiciones.

### 9.6 Multiplicidad, declarada de antemano

Son **166 comparaciones**. Con IC95 individuales, del orden de **8 reglas van a dar
"significativo" solo por azar**. Por eso la confirmacion en un periodo separado y la
banda del sorteo son parte de la regla de lectura y no un extra.

### 9.7 Lo que el paso 1 NO va a responder

Nada sobre el **reparto** (fijo), nada sobre la **salida** (fija), nada sobre las **zonas
de distancia** (eso es el paso 2), y nada sobre **costos** (el backtest no los incluye;
hay que declarar las operaciones por anio de cada regla junto al retorno).

### 9.8 Motor y condicion de arranque

Re-simulacion de cartera **vectorizada**, propia, como la del paso 2 del analisis de
salidas -- 166 backtests con el runner no es practico.

**Condicion de arranque, sin excepcion**: la re-simulacion tiene que reproducir el
backtest de `ft_backtesting_runner.py` para la v1 en el mismo periodo, operacion por
operacion. Si no reproduce, no se lee ninguna comparacion. Es el mismo paso 0 de
fidelidad que valido el analisis de salidas, y la razon es la misma: la maquina que
genera los resultados tiene que reproducir primero lo que ya paso.

Queda anotado que el backtest de referencia arrastra el ruido de la seccion 7
(indicadores desalineados en 51 tickers hasta enero de 2025), que afecta a todas las
reglas por igual y no se corrige en esta tarea.

---

## 10. RESULTADOS del paso 1 (20/9/2026)

186 corridas: las 166 reglas mas 20 sorteos de control, ~20 s cada una.

### 10.1 Condicion de arranque: cumplida, con una desviacion del pre-registro

La mascara de la v1 contra el filtro original del runner, mismo periodo:

| Metrica | Runner | Mascara v1 | Dif |
|---|---:|---:|---:|
| operaciones | 4.955 | 4.955 | 0,000000 |
| retorno total | +24,2693% | +24,2693% | 0,000000 |
| drawdown maximo | -8,3491% | -8,3491% | 0,000000 |
| exposicion media | 39,7760% | 39,7760% | 0,000000 |
| retorno medio por operacion | +0,3321% | +0,3321% | 0,000000 |

**Desviacion declarada**: la seccion 9.8 pre-registro una re-simulacion vectorizada
propia. No se escribio. Al medir el runner dieron 41 s por corrida, asi que en lugar de
reimplementar la maquina se **parametrizo la existente** con tres argumentos opcionales
(`filtro_entrada`, `orden_candidatos`, `registrar_candidatos`); con los defaults se
comporta igual que antes. La fidelidad deja de ser algo que hay que demostrar y pasa a
ser por construccion -- la tabla de arriba es el control de que la parametrizacion no
movio nada. Es una desviacion hacia el lado seguro, pero es una desviacion.

Al integrarlo aparecio que el runner pedia los indicadores del dia **dos veces** (una
para cierres, otra para entradas). Reusar la llamada bajo la corrida de 41 s a 25 s con
resultados identicos.

**Precision sobre la condicion 1 de 9.5**: el texto decia "exceso por operacion contra
la v1 con IC95" pero fijaba el **dia** como unidad de independencia. Se implemento como
**diferencia diaria pareada del retorno de cartera** (v1 contra la regla, mismos dias),
que es lo que la unidad declarada permite. Queda anotado porque las dos lecturas no dan
el mismo numero.

### 10.2 La banda del sorteo: el resultado principal

La v1 con el desempate sorteado, 20 semillas, todo lo demas identico:

| | |
|---|---:|
| Retorno minimo / maximo | **+16,34% / +26,30%** |
| Mediana / media | +22,46% / +22,12% |
| Desvio | 2,52 pp |
| Amplitud de la banda | **~10 pp** |
| v1 con desempate ALFABETICO | +24,27% (16 de 20 sorteos por debajo) |
| Operaciones / exposicion | 4.899-4.953 / 39,7-39,8% (v1: 4.955 / 39,8%) |

Las 20 corridas usan **la misma regla de entrada, la misma salida, el mismo reparto y
casi las mismas operaciones**. Lo unico que cambia es a cual de dos candidatos empatados
se le da el lugar. Eso solo mueve el retorno de cinco anios en 10 puntos porcentuales.

Traducido a la regla de lectura: una regla necesita superar a la v1 por mas de
**+2,03 pp** para que la diferencia sea atribuible a la regla. **La mejor de las 166 le
gana por +0,90 pp.**

El alfabetico queda en el percentil 80 de su propia banda. No es que ordenar por
abecedario sea bueno: es que con 20 tiros alguno tenia que salir arriba, y este es uno.

### 10.3 El resultado: 0 de 166

| Condicion pre-registrada | Pasan |
|---|---:|
| 1. Gana en seleccion **y** IC95 excluye el cero | **0** de 166 |
| 2. Mantiene el signo en confirmacion | 8 de 166 |
| 3. Supera la banda del sorteo | **0** de 166 |
| 4. Gana al universo ajustado por exposicion en 4 de 6 anios | **0** de 166 |
| **Las cuatro** | **0 de 166** |

La mejor en seleccion (`RSI | SMA50&MACD | SMA50&SMA21`) da una ventaja diaria pareada
de **+0,0020 pp con IC95 [-0,0076 ; +0,0116]**, que contiene al cero con holgura; en
confirmacion se da vuelta a -0,0022 pp. Ninguna regla se acerca a la condicion 1.

**85 de 166 le ganan a la v1 en seleccion. De esas 85, exactamente 1 mantiene el signo
en confirmacion.** Con 166 comparaciones y IC95 individuales se esperaban ~8
significativas por azar (seccion 9.6): no hubo ninguna.

### 10.4 Por que: la grilla mide exposicion, no calidad de entrada

| Correlacion | |
|---|---:|
| Exposicion media vs ventaja en seleccion (2021-24) | +0,608 |
| Exposicion media vs ventaja en confirmacion (2025-26) | +0,667 |
| Ventaja en seleccion vs ventaja en confirmacion (Pearson) | +0,586 |
| **Orden entre las 166, seleccion vs confirmacion (Spearman)** | **+0,187** |

La primera columna del ranking bruto la ordena cuanto tiempo la regla se queda
invertida, en los dos periodos. Una vez que se divide por exposicion, la ventaja
desaparece: la mejor en retorno bruto (`MACD | SMA50&SMA21&RSI`, +25,17%) rinde **0,59
por punto de exposicion contra 0,61 de la v1** -- es decir, queda peor. La v1 sale
**tercera de 166** en retorno por exposicion, y las dos que la superan lo hacen por 0,4 y
0,5 decimas.

Que el Pearson de la ventaja de 0,586 conviva con un Spearman de orden de 0,187 dice
donde esta cada cosa: la correlacion lineal la sostienen las reglas extremas (las muy
estrictas pierden en los dos periodos), mientras que **dentro del pelo del pelo de
candidatos realistas el orden se rearma casi al azar de un periodo al otro**. Es el mismo
+/-0,16 del paso 2 del analisis de salidas. Por la propia regla de lectura de 9.5, eso
basta para que ninguna conclusion sobre una regla individual sea fiable.

### 10.5 Lo que aparecio sin buscarlo: el techo no lo pone la entrada

| Anio | v1 / exposicion | Universo / exposicion |
|---|---:|---:|
| 2021 | +4,42% | -0,11% |
| 2022 | -22,07% | -58,25% |
| 2023 | +16,73% | **+71,29%** |
| 2024 | +20,00% | **+36,38%** |
| 2025 | +18,16% | **+66,27%** |
| 2026 | +10,12% | **+29,98%** |

La v1 le gana al universo equal-weight ajustado por exposicion en **2 de 6 anios**, y los
dos son los de mercado plano o bajista. Ninguna de las 166 reglas lo mejora: **153 dan
2/6 y 13 dan 1/6. Ninguna llega a 3.**

Esto no lo causo la grilla -- es la condicion de la estrategia, que el paso 1 hizo
visible al pedir la columna por anio. Y acota lo que se puede esperar del paso 2: si el
reparto y la salida se dejan fijos, mover la entrada no cambia de que lado de esa tabla
esta la estrategia.

### 10.6 Que deja el paso 1

1. **La entrada booleana esta agotada como palanca.** El espacio se enumero entero, no
   una muestra: las 166 reglas monotonas, incluidas las 18 que ningun score ponderado
   alcanza. No hay una regla mejor esperando en una combinacion que no se probo.
2. **El desempate pesa mas que la regla.** 10 pp de banda contra 0,9 pp de la mejor
   diferencia. Antes de seguir refinando el filtro, la palanca con masa es **como se
   elige entre candidatos empatados** -- que es ranking, no filtro.
3. **El paso 2 se corre igual, pero con la banda del sorteo como vara desde el primer
   dia.** Las zonas de distancia a la media son una hipotesis distinta (ordenan, no
   filtran) y por eso no quedan respondidas por esto. Lo que ya no se puede hacer es
   leer un +0,9 pp como una mejora.
4. **El ajuste por exposicion no es un detalle de presentacion.** Sin el, esta grilla
   habria "encontrado" cuatro reglas que le ganan a la v1, y las cuatro son simplemente
   reglas mas laxas.

Datos completos: `reportes/analisis_entradas/20260920_paso1/` (`resultados.jsonl`,
`series/*.npy`, `lectura.csv`, `parametros.json`).

---

## 11. Lo que queda DISENADO y SIN CORRER (20/9/2026)

Las dos continuaciones se discutieron y se disenaron. **Ninguna se corrio.** Quedan
escritas aca para poder retomarlas sin rehacer el razonamiento, y porque la decision de
no correrlas fue deliberada (seccion 11.3), no un olvido.

### 11.1 Paso 1b -- descomponer la decision, en vez del total de cartera

**La pregunta que el paso 1 no puede contestar.** El total de cartera mezcla *elegir
mejor* con *operar mas*. La correlacion exposicion-ventaja de +0,61 / +0,67 dice que hoy
manda la segunda. Separarlas exige bajar al nivel de la decision.

**Los tres tipos de desacuerdo.** En un (sector, rueda) dado, contra la v1:

| Tipo | Que pasa | Que mide |
|---|---|---|
| **Sustitucion** | la regla pone X donde la v1 puso Y | **seleccion** -- el head-to-head limpio |
| La regla llena un lugar que la v1 dejo vacio | un cupo mas ocupado | exposicion |
| La v1 llena uno que la regla dejo vacio | un cupo menos | exposicion |

Solo el primero mide calidad de eleccion. Mezclar los tres es volver a medir exposicion.

**La coincidencia es el denominador, no el numerador.** Cuando las dos reglas eligen el
mismo ticker el mismo dia el trade es identico: mismo precio de entrada, mismo SL/TP
(salen del precio de entrada y el ATR) y misma salida -- verificado, el bloque de cierres
del runner no mira el estado de la cartera. Su retorno porcentual es el mismo bit a bit y
aporta cero a cualquier diferencia. Sirve para medir cuanto margen habia para diferir.

**La trampa de multiplicidad y como se desarma.** 166 reglas x 9 sectores = **1.494
comparaciones**; con IC95 son ~75 significativas por azar. La salida es agrupar en UNA
pregunta -- sobre todos los eventos de sustitucion, el que eligio el desafiante rindio
mas que el de la v1? -- y recien despues abrir por sector (9 tests declarados). Por regla
se reporta la distribucion, no un IC por cabeza.

**El nulo ya esta medido.** Los 20 sorteos usan **la misma regla de entrada** con el
desempate al azar: discrepan con la v1 en ~43% de los sector-ruedas y ninguno de los dos
lados tiene ventaja, por construccion. Su distribucion head-to-head es la **distribucion
nula empirica** de este test, con la misma estructura de dependencia (los mismos dias de
mercado repetidos entre reglas). No hay que suponer normalidad ni independencia.

**Detalle que cambia la unidad de medida.** El sizing es `budget_sector * POSITION_PCT` y
el budget se consume y se libera segun lo que paso antes en ese sector: dos trades
identicos pueden tener **tamanos distintos en dolares**. La comparacion va en **retorno
porcentual por posicion-dia**, nunca en dolares, o se cuela la exposicion por la ventana.

**Costo**: la grilla guardo metricas agregadas, no las operaciones. Hay que re-correr las
186 guardando los trades -- mismo motor, mismos resultados, ~65 min.

**Lo que NO va a hacer**: resucitar una regla. El veredicto de cartera esta dado. Si el
pooled da cero, cierra la pregunta de la entrada en el nivel donde se toma la decision,
con mucha mas potencia que "el total no se movio"; si da distinto de cero pero chico, las
condiciones si discriminan y lo que las diluye es la estructura de la cartera -- lo que
apunta al reparto y al ranking, no al filtro.

### 11.2 Paso 1c -- factorial entrada x salida

**Por que no es solo combinar dos cosas ya probadas.** El pre-registro del paso 2 de
salidas declara lo que no mide: "el efecto de cartera (cupos por sector ocupados mas o
menos tiempo)". Ese es justo el canal que une las dos grillas:

- la grilla de salidas es **casi una sola dimension: cuanto tiempo se queda la posicion**
  (de 588 reglas, 516 se quedan mas que la actual; las ruedas en posicion correlacionan
  **+0,95** con la caida del p5 y -0,42 con la diferencia post);
- cuanto se queda una posicion = cuanto tarda en liberarse el cupo sectorial;
- cupos liberados = cuantas decisiones de entrada se llegan a tomar = **exposicion**;
- y la exposicion fue **la variable que domino el paso 1**.

La grilla de entradas tuvo la salida fija (rotacion de cupos fija) y la de salidas midio
por operacion, no por cartera. El cruce es el unico lugar donde la rotacion es variable.

**El tamano, y por que no es 166 x 589.** Serian **97.774 corridas = 543 horas**.
Descartado. La reduccion no es arbitraria: el espacio de salidas ya se midio
unidimensional, asi que alcanza con una **escalera de tiempos de tenencia**, que el
propio paso 2 dejo medida:

| Salida | Ruedas en posicion |
|---|---:|
| SMA21 peso 0 | 3,0 |
| MACD peso 0 | 4,1 |
| RSI peso 0 | 4,7 |
| **actual (v1)** | **6,8** |
| sin_rsi | 7,3 |
| sin_sma21 | 9,6 |
| RSI peso 2 | 10,0 |
| SMA50 peso 3 | 10,6 |
| SMA200 pesa 1 y RSI 2 | 12,2 |

**Un peldano mas: el time stop.** Es la unica palanca de salida que mostro senal en todo
el proyecto (SMC_v1: mejora tramo y post en 2021-24 con IC que excluye el cero, mismo
lado en 2025-26 sin alcanzar; hipotesis sin confirmar, ANALISIS_SALIDAS.md sec. 10).
TECH_SECTOR_v1 **no tiene time stop** y sus posiciones duran 8,45 dias. Es control directo
de la dimension del experimento y nunca se probo en esta estrategia.

**Factorial, NO busqueda.** Con ~70 celdas y una banda de ruido de 10 pp, buscar la mejor
celda encuentra ruido garantizado. La pregunta va estructurada -- el mejor tiempo de
tenencia depende de que tan laxa es la entrada? -- con una **prediccion direccional
escrita antes de correr**: si el mecanismo de los cupos es real, **una entrada mas laxa
deberia preferir tenencias mas cortas** (con mas candidatos haciendo cola, el cupo ocupado
cuesta mas). Si la superficie sale plana, los dos analisis eran separables y eso tambien
cierra algo.

**Tamano**: ~7 entradas (las que senale el paso 1b) x ~10 salidas = ~70 celdas, ~25 min;
mas el nulo del sorteo en 3 peldanos (corto / actual / largo), 60 corridas, ~20 min. Todo
se lee en retorno/exposicion y contra la banda del sorteo desde la primera tabla.

**Orden obligatorio**: el 1b va primero, porque es quien elige las 7 entradas. Sin el se
cruzarian entradas que difieren solo en laxitud con salidas que difieren solo en tenencia:
se mediria exposicion dos veces y se la llamaria interaccion.

**Prior honesto**: los dos efectos principales dieron cero (las diferencias post de las
589 salidas caen entre -0,23 y +0,22 pp contra un desvio de 7,26 pp; ninguna de las 166
entradas supero la banda). Una interaccion suele ser mas chica que los efectos
principales, asi que la probabilidad de encontrar algo grande es baja. Lo que lo justifica
es que el canal de los cupos es el unico sin medir y cuesta menos de una hora.

### 11.3 DECISION (20/9/2026): se para aca

**No se despliega nada y no se crea ninguna estrategia nueva.** El analisis de entradas
no produjo una regla candidata: 0 de 166 pasan, y la mejor diferencia (+0,90 pp) esta muy
adentro de la banda del sorteo (~10 pp).

Decision del usuario: **seguir trabajando sobre las estrategias existentes** y dejar los
pasos 1b y 1c disenados, sin correr. Ninguna regla de produccion se toca, asi que **no
corresponde registro en `ft_cambios`**; la parametrizacion de `ft_backtesting_runner.py`
es retrocompatible y ese runner no lo usa ningun bot de FT (solo los scripts de analisis).

Lo que queda para retomar, en orden: **el paso 1b**, que es el que tiene la pregunta mas
limpia; **el paso 1c** detras de el; y, antes que los dos, la palanca que este analisis
dejo senalada y sigue sin tocar -- **el ranking entre candidatos empatados**, que hoy
resuelve el abecedario y que resulto pesar diez veces mas que la regla de entrada.

---

## 12. Reglas que deja

1. **Antes de armar una grilla, contar donde cae la masa.** Un peso sobre un estado que
   ocurre 50 veces en cinco anios no es una palanca.
2. **La influencia mide el aporte de una condicion mejor que la correlacion.** "En que
   fraccion de filas voltearla cambia la decision" contesta directamente lo que la
   correlacion con otra condicion solo sugiere.
3. **Un desempate estable sobre una lista ordenada es una decision.** `sort` estable
   sobre una consulta con `ORDER BY ticker` es un desempate alfabetico, este escrito o
   no. El control honesto de cualquier criterio de desempate es sortear con semilla
   fija: si no le gana al azar, no es una palanca.
4. **Recomputar el insumo en vez de leer la columna derivada**, cuando la columna esta
   redondeada y la decision vive en un borde.
5. **Que una tabla reproduzca su propio codigo no significa que corresponda a la tabla
   con la que se la cruza** (seccion 6).
6. **Antes de comparar variantes, medir el piso de ruido del experimento.** Aleatorizar
   la parte arbitraria de la decision (aca el desempate) con semillas fijas cuesta 20
   corridas y da la vara: sin ella, las 166 reglas se leian como si un +0,9 pp
   significara algo, y la banda es de 10 pp. La vara se mide, no se supone.
7. **Una grilla sin ajuste por exposicion premia a la regla mas laxa.** La correlacion
   entre exposicion media y ventaja dio +0,61 y +0,67 en los dos periodos: sin dividir
   por exposicion el ranking mide cuanto tiempo se queda invertida la regla.
8. **Reusar la maquina validada le gana a reimplementarla, si el tiempo alcanza.** Se
   pre-registro una re-simulacion vectorizada nueva y se termino parametrizando el
   runner existente: la fidelidad pasa a ser por construccion en vez de algo que hay que
   demostrar. Medir cuanto tarda lo que ya existe antes de decidir escribir un motor.
9. **Un Pearson alto con un Spearman bajo sobre el mismo par dice donde esta la senal**:
   la relacion la sostienen los extremos y el orden del pelo del medio -- que es donde
   viven los candidatos realistas -- no se sostiene.
