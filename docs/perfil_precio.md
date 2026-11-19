# Perfil de precio: POC, area de valor y la prueba contra el azar

> **ESTADO (19/11/2026): MEDIDO. NO PASA.** Las
> secciones 1 a 3 describen lo que ya existe. La seccion 4 es el pre-registro,
> escrito ANTES de correr nada y antes de que existiera ningun numero; se deja
> tal cual, con su unica correccion fechada y explicada. La seccion 5 es el
> resultado: **el exceso del POC es cero en las cuatro celdas primarias**, y lo
> que parecia una ventaja era un confundidor que quedo MEDIDO (sec. 5.2). El POC
> queda como DESCRIPTOR.

| pregunta | seccion |
|---|---|
| que es el POC y por que los parametros son los que son | 1 |
| que NO se puede hacer, ya medido | 2 |
| que hay construido y quien lo usa | 3 |
| la prueba contra el azar: diseno y regla de lectura | 4 |
| el resultado | 5 |

---

## 1. Que es, y por que ningun parametro pide tuneo

El precio no se repite nunca exacto, asi que no se puede contar "cuantas veces
estuvo en 335,78". Se corta el eje de precios en franjas y se reparte el volumen
de cada rueda sobre las franjas que toco su rango H-L. El **POC** es la franja
con mas volumen de la ventana; el **area de valor** son las franjas que acumulan
el 70% del volumen alrededor del POC.

Los tres parametros se midieron el 18/11/2026 y ninguno quedo a ojo:

- **paso 0,5% geometrico**: el POC se mueve medio punto con el ancho del bin
  cambiando 20 veces (JPM, ventana 200: 336,29 con bins de 0,1% contra 335,78 con
  bins de 2%). Geometrico y no en dolares porque un bin de 1 USD es enorme para
  Ford a 11 y ridiculo para LMT a 590.
- **volumen y no tiempo**: mismo POC en 11 de 15 casos medidos.
- **grilla ANCLADA** en potencias de 1,005 desde 1,0, igual para todos los tickers
  y para siempre. Con una grilla armada sobre el minimo y el maximo de la ventana,
  los bordes se corren cada dia y parte del movimiento del POC lo fabrica la
  grilla.

**El ancho del area de valor es el error bar del POC, no un adorno.** CAT al
17/11/2026 da POC 811,50 en la ventana de 52 con un area de 40,5% de ancho: el
perfil es plano y el 811,50 es el maximo del ruido. Nunca publicar un POC sin su
ancho al lado.

En los 199 tickers, el area mediana es 9,9% (ventana 21), 14,4% (52) y 24,5%
(200), con area angosta (<10%) en el 52% / 25% / 4% de los casos. **A 200 ruedas
casi nadie esta en equilibrio**: en diez meses una accion tiende, y una tendencia
no tiene POC.

### Una propiedad de la grilla que importa para la seccion 4

Como los bins son geometricos y anclados, **un tramo de n franjas mide siempre el
mismo porcentaje, este donde este**: (1,005^n - 1). Eso permite construir bandas
de control con exactamente el mismo ancho que la del POC, sin que el ancho pueda
explicar ninguna diferencia.

---

## 2. Lo que NO se puede hacer, y esta medido

**Contar cierres exactos repetidos no sirve** (18/11/2026). Encuentra niveles
segun lo BARATO que sea el ticker, no segun su estructura: AVGO tiene 1.429
cierres distintos en 1.438 ruedas y CERO niveles con 3 toques; Ford tiene 54.
Ningun ticker del universo llega a 7. Y las repeticiones que hay son las del
AZAR: permutando los retornos diarios 500 veces, el real cae dentro de la banda
del nulo en los 4 tickers probados.

**Las lineas dibujadas a ojo tampoco son zonas densas.** Las 6 de AVGO caen por
DEBAJO de un precio al azar del mismo rango en toda tolerancia. El ojo marca
GIROS, que por definicion son lugares donde el precio estuvo POCO; contar
volumen mide PERMANENCIA. Son objetos opuestos.

**La tasa de rebote no predice** (6/9/2026, `memory/zonas_precio_descartado.md`).
Medido fuera de muestra sobre 200 tickers y 5 anios: el 85% de rechazo historico
cae a 56% (resistencia) y 71% (soporte); contra una banda cualquiera del ticker
da -1,4 / +3,5 pp; Spearman entre tasa pasada y futura -0,10 / +0,06. Los
controles dentro de muestra (+24 pp) estaban cocinados, porque median sobre las
mismas visitas que habian elegido la zona.

De esa medicion quedo una frase que es la que motiva la seccion 4: **lo unico
real es la UBICACION de donde se congrego el precio, no el porcentaje.** El POC
es exactamente esa ubicacion, y nunca se lo comparo contra el azar.

---

## 3. Que hay construido

| | |
|---|---|
| `src/utils/perfil_precio.py` | modulo PURO (numpy+pandas). `calcular`, `calcular_ventanas`, `calcular_rodante`, `ubicacion`. 26 tests con invariancia e implementacion de referencia |
| `scripts/manual/explorar_perfil_precio.py` | exploracion, SOLO LEE (`--detalle` / `--perfil TICKER` / `--csv`) |
| consumidores | **NINGUNO**. Sin tabla, sin paso de rutina, fuera de `ft_cambios` |

`calcular_rodante` cuesta **0,24 s/ticker** en la ventana 52 (medido 19/11/2026
sobre JPM, 1.730 ruedas), o sea ~50 s para los 199.

**El POC no es soporte ni resistencia por si mismo**: lo es respecto del precio
de hoy. Por eso `ubicacion()` lo DERIVA en cada lectura y no se guarda como
columna -- el dia que el precio cruce el nivel, una etiqueta almacenada quedaria
al reves.

---

## 4. PRE-REGISTRO: el POC contra el azar (escrito el 19/11/2026, antes de correr)

### 4.1 La pregunta

Cuando el precio entra en el area de valor, **hace algo distinguible de lo que
hace al entrar en una banda cualquiera del mismo ticker?**

No se pregunta si el POC es rentable. Se pregunta si es distinto del azar, que es
la pregunta que la seccion 2 dejo abierta y la unica que todavia no tiene
respuesta.

### 4.2 El evento

Primera rueda `t` en que el cierre entra en `[val, vah]` habiendo estado afuera
las **5 ruedas previas**, con la **direccion de llegada** registrada.

- El perfil usado es el de la rueda **t-1**. `calcular_rodante` incluye la rueda
  t en su ventana, asi que usar el de t meteria el dia del evento en el nivel que
  se supone que lo predijo. Con el de t-1 no hay informacion futura, por
  construccion.
- La banda de referencia es UNA sola, la de t-1, y contra ella se evalua tambien
  si el precio estuvo afuera en t-5..t-1. Todas esas ruedas son anteriores a t-1,
  asi que no se mira el futuro; y el evento queda definido como "el precio se
  acerco a ESTA banda desde afuera" en vez de contra bandas que se mueven.
- Direccion: `desde_arriba` si `close[t-1] > vah`, `desde_abajo` si
  `close[t-1] < val`.

### 4.3 Los dos controles

Ambos sobre la MISMA serie real, con el MISMO procedimiento de deteccion y la
MISMA metrica. Solo se mueve la banda. **20 semillas cada uno.**

| control | la banda es | que hipotesis niega |
|---|---|---|
| **A -- visitado** | un tramo de la misma cantidad de franjas, en una posicion al azar entre las franjas que el precio visito en la ventana | "cualquier banda da lo mismo" |
| **B -- denso** | idem, pero la posicion se sortea entre las franjas del **cuartil superior de volumen de las ELEGIBLES**, o sea el lugar mas denso que NO es el area de valor | "cualquier banda MUY VISITADA da lo mismo" |

> **CORRECCION del 19/11/2026, hecha ANTES de correr la medicion y sin ningun
> resultado a la vista.** La primera version calculaba el cuartil superior sobre
> TODAS las franjas del perfil, y asi el control B quedaba vacio **por
> construccion**: el area de valor es por definicion donde esta el 70% del
> volumen, asi que el cuartil superior del perfil completo cae casi entero
> adentro del area -- que es justo lo que el control excluye. En la prueba corta
> con 3 tickers dio 0 eventos y percentiles NaN. El umbral pasa a calcularse
> sobre las franjas CANDIDATAS. No es un ajuste de umbral mirando resultados: el
> control no estaba midiendo nada.

**B es el control que decide.** A solo, el POC lo gana facil por ser un lugar
donde el precio estuvo mucho, y eso no es una conclusion: seria medir que el area
de valor esta donde esta el volumen, que es su definicion.

Las bandas de control tienen **exactamente** el mismo ancho que la del POC (misma
cantidad de franjas sobre una grilla geometrica, ver sec. 1), y no se permite que
se solapen con el area de valor. Si en una rueda no hay posicion valida, esa
rueda se saltea para esa semilla y se informa cuantas veces paso.

### 4.4 La metrica

**Exceso de retorno a h = 10 ruedas contra el universo equal-weight de la misma
rueda**, medido desde `close[t]`.

- Media **por evento**, con **IC95 y el DIA como unidad de independencia**
  (`src/utils/ft_salidas.ic95_por_dia`, que ya existe y ya se uso en los analisis
  de salidas y de entradas de FT). Dos eventos del mismo dia comparten mercado y
  no son observaciones independientes.
- **Nunca un total.** Un total es media x n y no es una metrica de significancia
  (leccion de `docs/forward_testing/ANALISIS_FILL.md` sec. 3.4).
- Signo esperado: `desde_arriba` (el area como soporte) -> exceso **positivo**;
  `desde_abajo` (resistencia) -> exceso **negativo**.

### 4.5 Configuraciones PRIMARIAS: dos, y las dos se declaran ahora

Ventana **52**, horizonte **h = 10**, area **angosta (`ancho_va_pct < 10`)**, y:

1. **SOPORTE**: llegada `desde_arriba`, se espera exceso > 0.
2. **RESISTENCIA**: llegada `desde_abajo`, se espera exceso < 0.

Se evaluan **por separado**, cada una con su propia regla de lectura. El
resultado se informa como las dos pasan / pasa una / no pasa ninguna, y si pasa
una sola se dice explicitamente cual y que la otra no.

Por que la ventana 52: a 21 el perfil se mueve demasiado y a 200 solo el 4% de
los casos tiene area angosta (sec. 1). Por que el umbral de 10%: es el corte de
"area angosta" ya usado en la medicion del 18/11 y deja el 25% de los
ticker-ruedas en esta ventana. **Los dos numeros quedan fijados aca y no se
mueven despues de ver resultados.**

### 4.6 Particion

**Seleccion 2021-2024 / confirmacion 2025-2026**, el mismo corte de los analisis
de entradas y salidas de FT.

Universo pre-registrado: **solo los tickers con al menos 300 ruedas antes del
2025-01-01**. El universo tiene cohortes de alta distintas y sin este filtro la
ventana de seleccion quedaria dominada por unos pocos tickers mientras la de
confirmacion los tiene a todos, que es otra variable cambiando entre los dos
periodos.

### 4.7 REGLA DE LECTURA: pasa si las tres, para esa direccion

1. el exceso medio del POC queda **fuera de la banda p5-p95** de las 20 semillas
   del control **B**, en la direccion esperada;
2. el **IC95 por dia del exceso del POC excluye el cero**;
3. y las dos cosas valen **en seleccion Y en confirmacion**.

Si falla cualquiera de las tres, esa direccion **NO PASA**. Si no pasa ninguna de
las dos direcciones, la familia se cierra con la medicion escrita y el POC queda
como lo que es hoy: un descriptor.

### 4.8 Secundarios: declarados ahora, exploratorios por definicion

Ventana 21, horizonte h = 5, llegada con area **ancha** (`>= 10`), y una banda
**simetrica fija** de POC x [0,95; 1,05] en lugar del area de valor.

Con cerca de una decena de comparaciones, que alguna "pase" por azar es
esperable: **ningun secundario puede sostener una conclusion por si solo**. Estan
para dos cosas:

- el contraste **area angosta contra area ancha** es el chequeo de cordura: si el
  POC "funciona" igual con area ancha, no es el POC el que esta trabajando;
- la **banda simetrica** separa "estar cerca del POC" de "estar dentro del area
  de valor", que son dos objetos distintos y el area no es simetrica alrededor
  del POC.

### 4.9 Lo que esta prueba NO contesta

- **No mide rentabilidad.** No hay costos, no hay reglas de salida y no hay
  sizing. Un exceso distinto de cero no es una estrategia.
- **No dice nada de la ventana de 200**, donde el perfil es plano.
- **No reabre la tasa de rebote**, que ya esta medida y cerrada (sec. 2).
- **No convierte al POC en insumo de decisiones.** Si pasara, lo que habilita es
  disenar una regla y medirla contra un control, no consumirlo directamente.

### 4.10 Que se va a construir

| archivo | que |
|---|---|
| `src/utils/poc_eventos.py` | modulo PURO: deteccion de eventos, bandas de control, exceso contra el universo |
| `tests/test_poc_eventos.py` | invariancia e implementacion de referencia |
| `scripts/manual/medir_poc_vs_azar.py` | la corrida, SOLO LEE; resultados a `reportes/poc_vs_azar/AAAAMMDD/` |

No toca ningun bot, ninguna tabla ni ningun paso de la rutina, y **no va a
`ft_cambios`** (no cambia decisiones de ninguna estrategia de FT).

---

## 5. Resultado (19/11/2026): NO PASA ninguna de las dos direcciones

Corrida: `scripts/manual/medir_poc_vs_azar.py --etiqueta prereg`, salida en
`reportes/poc_vs_azar/20261119_prereg/`. **124 tickers** (los que cumplen el
filtro de 300 ruedas pre-2025 de la sec. 4.6) y 100.427 filas de evento entre
todas las fuentes y ventanas.

### 5.1 Las dos primarias

Ventana 52, h = 10, area angosta. Exceso en puntos porcentuales contra el
universo equal-weight, IC95 con el dia como unidad.

| direccion | periodo | n (dias) | POC | IC95 | control B (p5 / p50 / p95) | POC cae |
|---|---|---|---|---|---|---|
| **soporte** (desde arriba) | seleccion | 264 (200) | **+0,091** | [-0,531; +0,714] | -1,501 / -1,170 / -0,821 | arriba |
| | confirmacion | 129 (105) | **+0,099** | [-0,944; +1,143] | -1,056 / -0,599 / -0,365 | arriba |
| **resistencia** (desde abajo) | seleccion | 258 (181) | **-0,405** | [-1,274; +0,465] | -0,554 / -0,243 / -0,022 | adentro |
| | confirmacion | 102 (85) | **-0,274** | [-1,347; +0,800] | -0,191 / +0,150 / +0,450 | abajo |

**Las cuatro celdas tienen el cero adentro del IC95.** Entrar al area de valor,
desde arriba o desde abajo, va seguido de que el ticker haga lo que hace el
universo. Por la regla de la sec. 4.7 **ninguna de las dos direcciones pasa**:
falla la condicion 2 en las cuatro, y la resistencia falla ademas la 1 en
seleccion.

### 5.2 Lo que SI aparecio, y por que no es evidencia

En **soporte**, la condicion 1 se cumple en los dos periodos: el POC queda por
encima de la banda del control B, consistentemente. Pero la ventaja **no viene de
que el POC suba**: viene de que los controles BAJAN (-1,17 y -0,60 de mediana)
mientras el POC da ~0.

Eso tiene una explicacion mas simple que "el area de valor sostiene al precio", y
esta MEDIDA (diagnostico post-hoc, exceso de las 10 ruedas PREVIAS al evento,
mismo corte: soporte / ventana 52 / area angosta):

| fuente | n | exceso previo medio | mediana |
|---|---|---|---|
| **POC** | 397 | **-0,69 pp** | -0,86 |
| control A | 6.138 | **-3,70 pp** | -3,88 |
| control B | 6.552 | **-3,70 pp** | -3,88 |

**Los eventos del control llegan de una caida tres puntos mayor.** No son la
misma situacion en otro precio: son OTRA situacion. Para alcanzar desde arriba
una banda densa que NO es el area de valor, el precio tiene que haber caido
bastante mas -- la ubicacion y el movimiento previo estan acoplados. El -1,2 pp
del control es el arrastre de esos tickers, no una virtud del POC.

> **LECCION DE METODO, y vale para cualquier prueba futura de soportes y
> resistencias:** una banda de control "del mismo ancho en otra ubicacion" NO es
> un control limpio para un evento DIRECCIONAL. La ubicacion determina cuanto
> tuvo que moverse el precio para llegar, asi que el control cambia dos cosas a
> la vez. Un control honesto tendria que aparear tambien el movimiento previo
> (por ejemplo, sortear la banda entre las que estan a una distancia parecida del
> precio). El placebo de lineas al azar que propone la literatura -- y el sec.
> 5.2.5 del documento que origino esta medicion -- tiene exactamente este
> agujero.

### 5.3 El chequeo de cordura, y los secundarios

El contraste **area angosta contra area ancha** no muestra ninguna diferencia
coherente: con area ancha el exceso tambien es cero (soporte, ventana 52, h 10:
-0,331 en seleccion y -0,215 en confirmacion, los dos con el cero adentro). Si el
POC estuviera trabajando, el filtro de ancho -- que es su error bar -- tendria que
separar algo. No separa nada.

De los secundarios, **ninguno sostiene nada**: la unica celda con IC95 que excluye
el cero (ventana 52, h 10, area ancha, desde abajo, confirmacion: -1,382
[-2,595; -0,169]) tiene el signo opuesto en seleccion (+0,372), que es el patron
clasico del ruido. Con cerca de una decena de comparaciones eso era esperable y
estaba declarado en la sec. 4.8.

### 5.4 Veredicto

**El POC y el area de valor quedan como DESCRIPTORES.** Dicen donde se congrego
el precio -- eso es cierto y es util para mirar un grafico o para una ficha --
pero **no anticipan nada** sobre lo que hace el precio cuando vuelve ahi.

Esto cierra la ultima pregunta abierta de la familia "soportes y resistencias
estaticos" en este proyecto:

1. la **tasa de rebote** no predice (6/9/2026, sec. 2);
2. **contar cierres exactos** mide la escala del ticker y sus repeticiones son
   las del azar (18/11/2026, sec. 2);
3. la **ubicacion por volumen** -- lo unico que habia quedado en pie -- tampoco
   se distingue del azar (19/11/2026, esta seccion).

### 5.5 Como se reabriria

Con una de estas, no por iniciativa propia:

- un **control apareado por movimiento previo**, que es el agujero que dejo
  medido la sec. 5.2. Es la unica critica metodologica seria a esta corrida;
- una **hipotesis distinta del rebote**: por ejemplo que el area de valor prediga
  VOLATILIDAD o rango futuro en vez de direccion, que es una pregunta que nadie
  midio todavia;
- el perfil construido con **barras intradiarias** en vez del reparto uniforme
  sobre el H-L. Hoy no se puede: `precios_intradia_30m` captura 11 de 13 barras y
  el 57-70% del volumen, y lo que falta es el tramo final de la rueda (sec. 1).

Lo que **no** reabre nada: cambiar el ancho del bin, la cobertura del area, el
horizonte o la ventana. Son los parametros que ya se midieron y no mueven el
resultado.
