# Niveles de soporte y resistencia por acumulacion de GIROS (19/11/2026)

> **ESTADO: CONSTRUIDO Y MEDIDO.** El detector anda y reproduce 8 de las 11
> bandas que el usuario habia dibujado a mano sobre CVX y HON. Lo que **no** se
> sostiene es llamarlos "fuertes": una version BARAJADA del mismo ticker produce
> niveles con el mismo conteo. Queda como **DESCRIPTOR** -- de la misma familia
> que el RSI o una media, a los que tampoco se les pide ganarle a un nulo.
>
> Y deja un hallazgo que es mas grande que esta funcion: **`precios_diarios` no
> sirve para esto** porque es un patchwork de dos bases de precio (sec. 2).

| pregunta | seccion |
|---|---|
| que es, y por que no es el POC | 1 |
| por que hace falta una serie NOMINAL | 2 |
| el metodo | 3 |
| reproduce lo que marca el ojo? | 4 |
| los niveles son especiales? | 5 |
| que se puede afirmar y que no | 6 |
| dos errores que costaron numeros mal | 7 |
| como correrlo | 8 |

---

## 1. Que es, y por que no es el POC

Un nivel es una franja de precio donde el precio **DIO VUELTA** varias veces: ahi
se acumulan swings confirmados, de arriba (soporte) o de abajo (resistencia).

Es un objeto **distinto** del perfil de volumen (`docs/perfil_precio.md`), y la
diferencia esta medida: **el ojo marca GIROS, que por definicion son lugares
donde el precio estuvo POCO; el POC mide PERMANENCIA, donde el precio se QUEDO.**
Son opuestos, y por eso ningun contador de permanencia iba a reproducir el
grafico que una persona dibuja. El POC se midio el 19/11/2026 contra el azar y no
se distingue; esto es otra cosa y se mide aparte.

## 2. Por que hace falta una serie NOMINAL (y `precios_diarios` no sirve)

Un nivel psicologico es un precio **nominal**: la gente recuerda "208 dolares",
no "208 ajustado por dividendos".

**`precios_diarios` es un patchwork de dos bases de precio** (medido el
19/11/2026, detalle en la AGENDA):

| ticker | coincide con el nominal desde | ratio 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| CVX | 2026-02-17 | 0,8220 | 0,8585 | 0,8894 | 0,9265 | 0,9686 |
| HON | 2026-06-29 | 0,8642 | 0,8807 | 0,8989 | 0,9189 | 0,9383 |

El ~95% de la historia esta en precios AJUSTADOS POR DIVIDENDOS y los ultimos
meses en nominales; la costura de cada ticker es la fecha de su ultimo backfill.
El pipeline de HOY esta bien (`yahooquery_loader` usa `adj_ohlc=False` y guarda
`sub["close"]`, que es el nominal); lo que esta en otra base es la historia vieja.

**Sobre la serie ajustada el ejercicio no es dificil: es imposible, y ademas
enganoso.** Medido en CVX: los dos clusters de giros mas fuertes (13 y 19 toques,
1,7% de ancho, que se ven como niveles impecables) son **escaleras nominales de
15% de ancho** -- 171 -> 167 -> 164 -> 162 -> 160 -> 158 -> 154 -> 152 -> 149.
El ajuste no solo desparrama niveles reales: **los FABRICA**, porque un precio
nominal que baja al ritmo del dividendo se vuelve una linea horizontal perfecta
al ajustar.

Por eso el detector lee `precios_nominales`, una tabla aparte que **nadie del
pipeline toca** (`scripts/manual/cargar_precios_nominales.py`). El nominal sale
de la MISMA llamada de yahooquery que ya hace el proyecto, sin fuente nueva ni
cuota: con `adj_ohlc=False`, open/high/low/close son nominales y `adjclose` viene
aparte. Verificado contra Alpha Vantage: CVX 2022-12-30, yq `close` 179,49 =
crudo de AV; yq `adjclose` 152,67 = ajustado de AV; nuestro guardado, 156,70, no
es ninguno de los dos.

Si algun dia se arregla `precios_diarios`, el detector cambia de fuente y la
tabla se borra.

## 3. El metodo

| paso | que hace | por que asi |
|---|---|---|
| **giros** | `estructura.swings_en_barra` con N=5, marcados en SU barra | son los giros confirmados, sin mirar al futuro: un swing en `i` recien se sabe en `i+n`, y `giros_confirmados(hasta=D)` respeta eso |
| **grilla** | geometrica, paso 0,25% | un bin en dolares es enorme para Ford a 11 y ridiculo para LMT a 590 |
| **banda** | +-1% alrededor del centro | ~2% de ancho, que es lo que miden las bandas dibujadas a ojo en los graficos de referencia |
| **separacion** | 15 ruedas entre dos toques del mismo nivel | una lateralizacion corta es UN episodio visto de cerca, no cinco toques |
| **seleccion** | golosa: el centro con mas toques, y se excluye su entorno | hay VARIOS niveles a la vez; no se busca el maximo sino todos los picos |
| **re-centrado** | el nivel se recentra en la media geometrica de sus toques | si no, parte del precio lo fabrica donde cayo la grilla |
| **de-duplicacion** | dos niveles no pueden quedar a menos de 2 anchos de banda, **medida en LOG** | mas cerca que eso el ojo ve una zona y no dos. El log, por la sec. 7 |

**El orden lo encabeza la PERSISTENCIA** (anios calendario distintos con toque) y
despues el conteo. Cuatro toques en un trimestre son un solo episodio; cinco
repartidos en dos anios atravesaron varios regimenes. Es lo que distingue las
bandas que el ojo acepta, y sale de mirar los graficos de referencia: las que el
usuario dibujo abarcan de 2022 a 2026.

Cada nivel se publica con **cuatro numeros, no uno**: toques, reparto H/L
(polaridad), anios distintos y antiguedad del ultimo toque. Una franja con giros
de los dos signos es un **cambio de polaridad**, y es la regla y no la excepcion.

## 4. Reproduce lo que marca el ojo? SI: 8 de 11 bandas

Contra las bandas que el usuario dibujo en TradingView, sin ajustar ningun
parametro para que coincidieran:

**CVX** (dibujadas: ~206, ~198, ~167, ~155, ~134)

| nivel | toques | H/L | anios | periodo |
|---|---|---|---|---|
| **167,92** | 8 | 6/2 | **5** | 2022-08 -> 2026-01 |
| **148,91** | 13 | 2/11 | 4 | 2022-08 -> 2025-11 |
| **198,79** | 3 | 2/1 | 1 | 2026-05 -> 2026-08 |
| **131,70** | 3 | 0/3 | 2 | soporte puro |

**HON** (dibujadas: ~248, ~220, ~208, ~189, ~182, ~175)

| nivel | toques | H/L | anios | periodo |
|---|---|---|---|---|
| **203,90** | 17 | 9/8 | **6** | 2021-03 -> 2026-10 |
| **216,93** | 13 | 7/6 | 5 | 2021-03 -> 2026-10 |
| **187,46** | 9 | 3/6 | 3 | 2022-04 -> 2024-04 |
| **251,00** | 3 | 3/0 | 1 | resistencia pura, 2026 |

Aparecen ademas HON 196,00 (15 toques, 6 anios) y 230,49 (10 toques, 4 anios),
que el usuario no habia marcado y que por conteo y persistencia son dificiles de
descartar.

**Este es el resultado que justifica la funcion**: hace en un segundo, y para los
199, lo que una persona hace a mano con el mouse sobre un grafico.

## 5. Los niveles son especiales? NO, y el test tiene poder

`--nulo N` corre la misma deteccion sobre N series PERMUTADAS del mismo ticker
(los mismos retornos en otro orden: mismo inicio, misma distribucion, mismo n).

**HON, 50 permutaciones**: el mejor nivel de una serie barajada llega a **18
toques (p95) y 25 (max)**, con **6 anios (p95 y max)**. El mejor nivel REAL tiene
17 toques y 6 anios. **Ninguno de los niveles reales supera al azar.**

Y el test **puede** fallar, que es la regla del proyecto para cualquier
diagnostico que da cero:

| serie | real | nulo (p95 / max) | veredicto |
|---|---|---|---|
| random walk puro | 11 toques / 3 anios | 15/5 y 19/6 | no se distingue (correcto) |
| rebota SIEMPRE en 100 y 130 | 26 toques / 6 anios | 22/6 y 23/6 | **SUPERA** (correcto) |

O sea: cuando hay un nivel de verdad, el test lo encuentra. En HON no lo hay.

La explicacion es geometrica y no hace falta mas: ~170 giros repartidos sobre
~180 franjas se amontonan por palomar, y un random walk con la misma volatilidad
revisita las mismas zonas.

## 6. Que se puede afirmar y que no

| afirmacion | vale? |
|---|---|
| "el precio dio vuelta 17 veces en 203,90 a lo largo de 6 anios" | **SI**, es un hecho de la serie |
| "esta franja fue soporte y despues resistencia" | **SI**, sale del reparto H/L |
| "203,90 es un precio psicologico FUERTE" | **NO**: una version barajada concentra igual |
| "ordenemos los tickers por calidad de sus niveles" | **NO**, por lo mismo |
| "el precio va a reaccionar cuando vuelva ahi" | **NO**, y ademas esta medido que no (`memory/zonas_precio_descartado.md`) |

El detector es un **descriptor**: resume fielmente donde giro el precio. No se lo
publica con la palabra "fuerte" ni se lo usa para decidir.

## 7. Dos errores que costaron numeros mal (y como se evitan)

**`features_estructura` marca la barra de CONFIRMACION, no la del giro.** Su
`is_sh_N` sale de `_desplazar(sh_barra, n)`, y el precio del swing esta `n` barras
antes (`precio_sh = _desplazar(h, n)`). Leer `high` en la fila donde `is_sh_N=1`
devuelve un precio corrido hasta 5 ruedas. Las primeras exploraciones de esta
linea lo hicieron y dieron niveles equivocados. **El modulo usa
`swings_en_barra` directo**, que marca el swing en su propia barra, y maneja la
confirmacion aparte.

**`abs(a/b - 1)` no es simetrico.** Con 156,09 y 162,45 da 3,92% en un sentido y
4,08% en el otro, asi que la de-duplicacion aceptaba o rechazaba el par **segun
cual se hubiera elegido primero**. Sobre una grilla geometrica la distancia es el
LOG: `abs(log(a/b))`. Lo encontro un test, no una lectura.

## 8. Como correrlo

```
python scripts/manual/cargar_precios_nominales.py --crear          # una vez, ~291k filas
python scripts/manual/explorar_niveles.py HON
python scripts/manual/explorar_niveles.py CVX --nulo 50 --tol 1.5
```

`cargar_precios_nominales.py` toma el lock de yfinance (el rate limit es por IP) y
corta en la ultima rueda de `precios_diarios`, que ya cerro, para no traer la
barra en curso. Es idempotente (UPSERT) y LOCAL-only.

Modulo puro: `src/utils/niveles_precio.py` (15 tests, con invariancia e
implementacion sintetica de referencia). Tabla: `precios_nominales` (ticker,
fecha, OHLCV), 291.384 filas / 199 tickers / 2021-01-04 -> 2026-11-18.

## 9. Lo que queda sin hacer

- **Ningun consumidor.** No lo lee el dashboard, ni Telegram, ni el MCP. Antes de
  enchufarlo hay que decidir como se presenta sin la palabra "fuerte".
- **La tolerancia es fija en +-1%.** El documento que origino esta linea proponia
  0,5xATR, que se adapta a la volatilidad del ticker. No se probo; el 1% salio de
  medir las bandas dibujadas a mano, que es poca evidencia.
- **El nulo no tiene umbral operativo.** Dice que ningun nivel de HON supera al
  azar, pero no se barrio el universo: puede haber tickers donde si. Es una
  corrida de `--nulo` por ticker y nadie la hizo.
- **La serie nominal se carga a mano.** No es un paso de la rutina y se va a
  quedar vieja. Mientras sea exploratorio, esta bien asi.
