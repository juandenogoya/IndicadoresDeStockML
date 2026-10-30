# Consumo de Railway: que cuesta y por que (29/10/2026)

> **ESTADO: MEDIDO Y CERRADO SIN CAMBIOS.** Nada se toco: ni la retencion, ni el
> `VACUUM FULL`, ni la cadencia del sync, ni un script. El consumo esta SANO y
> ninguna de las lineas de trabajo de septiembre/octubre lo movio. Se sigue como
> se venia. Lo que este doc deja es la **linea de base medida** y la explicacion
> de por que la factura es lo que es, para no volver a diagnosticarlo desde cero
> ni "optimizar" la palanca equivocada. La sec. 8 dice que lo reabriria.

| pregunta | seccion |
|---|---|
| de donde sale el numero de la factura | 2 |
| esta creciendo la DB? funciona la retencion? | 3 |
| aumento por algo que hicimos? | 4 |
| que es lo unico que sube | 5 |
| la operacion desproporcionada (y por que NO se toco) | 6 |
| cuanto cuesta la cuenta entera | 7 |
| como re-medirlo | 9 |

---

## 1. Que se midio, con que, y que NO se toco

Todo fue **solo lectura**: metricas de la API de Railway (hasta 7 dias de
ventana), consultas a la DB de Railway (`pg_class`, `pg_stat_database`,
`pg_ls_waldir`, conteos por fecha), los logs de la rutina en
`logs/rutina/AAAAMMDD_HHMM/01_sync.log` y la tabla local `rutina_corridas`.

Cero escrituras en Railway, cero cambios en el repo mas alla de este documento.

---

## 2. La factura: el 95% es RAM, no disco

Renglones de la factura del proyecto (ciclo en curso al 29/10/2026) y la tarifa
que sale de dividir **costo / unidades**:

| renglon | unidades | costo USD | tarifa implicita |
|---|---|---|---|
| **RAM** | 5.321,98 GB-min | **1,2319** | 10 USD /GB/mes |
| Volumen | 7.936,21 GB-min | 0,0276 | 0,15 USD /GB/mes |
| Backup de volumen | 6.217,45 GB-min | 0,0216 | 0,15 USD /GB/mes |
| Egress | 0,34 GB | 0,0169 | 0,05 USD /GB |
| CPU | 4,08 vCPU-min | 0,0019 | 20 USD /vCPU/mes |
| **TOTAL** | | **1,30** | |

**La RAM es el 94,8% de la factura. Todo lo demas junto son 7 centavos.**

Eso contradice el modelo mental que quedo despues del incidente del 20/7 (donde
el problema SI era disco). **Hoy el disco cuesta 10 centavos al mes.** Recortar
retencion para "ahorrar" ataca la linea equivocada.

### 2.1 El largo del ciclo, derivado

Las unidades son acumuladas por minuto. Dividiendo por los promedios medidos:

- volumen: 7.936,21 GB-min / 0,641 GB = **12.381 min**
- RAM: 5.321,98 GB-min / 0,430 GB = **12.377 min**

Las dos dan lo mismo: **~12.380 minutos = 8,6 dias**, o sea el ciclo arranco
alrededor del **20/10/2026**. Proyectado a 30 dias: **~4,50 USD/mes** para este
proyecto.

---

## 3. El estado de los datos: la retencion funciona

### 3.1 Contra el incidente

| | 20/7/2026 (Railway detenido) | 29/10/2026 |
|---|---|---|
| `opciones_snapshot` | 923 MB | **227 MB** |
| DB total | 938 MB | **248 MB** |
| estado | detenido por limite de consumo | normal |

### 3.2 Inventario de la DB (248 MB)

| tabla | tamano | nota |
|---|---|---|
| `opciones_snapshot` | 227 MB | 92% de la DB. 0 filas muertas |
| `opciones_ar_gregas` | 5.680 kB | crece, AR, sin retencion |
| `senales_bot_diaria` | 2.144 kB | **congelada en 2026-09-11** (bots Alpaca apagados) |
| `alertas_scanner_ar` | 1.232 kB | crece, AR |
| `pcr_ar_diario` | 536 kB | crece, AR |
| el resto (16 tablas) | < 1 MB c/u | residuos de Plan C |

Las tablas AR crecen sin retencion desde abril/2026 y en 6 meses juntan **7,4 MB
entre las tres**: a ese ritmo no son un problema de costo ni cerca.

### 3.3 La rotacion esta en regimen

`opciones_snapshot` tiene **11 fechas (2026-10-15 -> 2026-10-29), 199 tickers
cada una, ~105k filas/dia = 20,6 MB/dia**. Y es **una sola captura por dia**: el
`created_at` de todas las fechas cae entre las **23:00:2x y las 23:15 UTC**, que
es el intento 1 del cron de Oracle. Los reintentos de 02:00 / 04:00 (GH) / 06:00
no duplican nada.

El ciclo diario completo, medido el 29/10:

1. la purga borra el dia 11 (`2026-10-14`, 92.064 filas): **224 MB -> 205 MB**;
2. a las 23:00 UTC entra la captura del dia: **205 MB -> 227 MB**.

Es un buffer rotativo estable. No hay deriva.

### 3.4 El egress esta explicado 1 a 1

**0,34 GB** en el ciclo, sobre **8 corridas del sync** (20, 21, 22, 23, 24, 27,
28 y 29/10) = **42,5 MB por corrida**. El sync re-baja `fecha_snapshot >= max
local`, que el 29/10 fueron **211.871 filas** a ~200 bytes = **~42 MB**.

Cierra exacto. Eso prueba algo mas fuerte que el costo: **nada mas esta leyendo
Railway**. Si algun script o consumidor estuviera apuntando a Railway por error
(la trampa de `DATABASE_URL` / `.env.local`), aparecia aca.

> El sync re-baja ~2 dias para insertar 1 (211.871 candidatas -> 106.185 nuevas,
> 50% de desperdicio). Son ~0,03 USD/mes. Esta medido y se deja como esta.

---

## 4. No aumento por nada que hicimos

| evidencia | medicion |
|---|---|
| cadencia del sync | **31 corridas en 60 dias habiles, sin un solo hueco** > 3 dias, 50-157 s cada una (`rutina_corridas`) |
| capturas | 1 por dia habil, 199 tickers, ~105k filas, misma ventana horaria |
| egress | explicado 1 a 1 por el sync (sec. 3.4): ningun lector nuevo |
| commits de los ultimos 60 dias | carteras por perfil, cierre por grupo, ANALISIS_FILL, IBKR, dividendos, status de balances: **todos LOCAL o contra APIs externas**. Ninguno escribe en Railway |
| desde el 13/9 | Railway recibe **menos**: `push_senales_bot` salio de la rutina y `senales_bot_diaria` quedo congelada |

**Conclusion: el consumo no subio por el trabajo del proyecto.** Si algo, bajo.

---

## 5. Lo unico que sube: la RAM del contenedor

Es el unico numero con deriva, y resulta ser justo el que paga el 95% de la
factura:

| ventana | min | max | ahora |
|---|---|---|---|
| 7 dias | 0,197 GB | 0,668 GB | 0,668 GB (**= el maximo**) |
| 48 horas | 0,262 GB | 0,668 GB | 0,668 GB |

Postgres **no reinicia desde el 2026-08-23** (67 dias). La metrica de memoria de
Railway es el RSS del contenedor **incluyendo page cache**: el cache se llena con
lo que la DB toca y no se devuelve hasta que hay presion. Eso es el serrucho del
grafico, y como se factura por GB-minuto, **esa deriva ES la trayectoria de la
factura**.

Parametros del server (PostgreSQL 18.6): `shared_buffers` 128MB, `work_mem` 4MB,
`maintenance_work_mem` 64MB, `effective_cache_size` 4GB, `max_connections` 500.
O sea: los 0,67 GB **no** son Postgres configurado grande; son cache.

---

## 6. La operacion desproporcionada: el `VACUUM FULL` diario

`retencion_opciones_railway.py` hace `DELETE` del dia 11 y despues `VACUUM FULL`.
Los numeros:

| | |
|---|---|
| lo que reescribe | **205 MB** (la tabla entera), ~11 s |
| lo que recupera | **~20 MB** (1 dia de 10) |
| lo que vale ese disco | **0,003 USD/mes** |
| WAL que genera | **~205 MB/dia** (la reescritura va a WAL) |
| WAL en disco hoy | **224 MB en 14 archivos = 33% del volumen de 685 MB** |

**En un buffer rotativo no hace falta devolver el disco: las paginas que libera
el DELETE del dia 11 las reusa el INSERT del dia siguiente.** El `VACUUM FULL`
compra ~20 MB de disco pagando una reescritura completa todos los dias. El
docstring del script dice, con razon, que "DELETE solo no devuelve el disco" --
lo que no estaba medido es que **en este caso no hace falta devolverlo**.

### 6.1 Lo que NO se puede probar con lo que hay

**No esta demostrado que sacarlo baje la linea de RAM**, que es el 95% de la
factura. La metrica de memoria de Railway es RSS del contenedor con page cache y
la API devuelve **agregados, no la serie temporal**: no hay forma de atar los
picos de memoria a las 09:23 de la purga. Es una **hipotesis**, no un hallazgo.

Dato **sugerente y no concluyente**: el Postgres de *idoneidad*, mismo plan y
205 MB en disco pero **sin escrituras diarias**, se queda plano en 0,0757-0,0767
GB de RAM en 7 dias. El de IndicadoresML, con 685 MB y rotacion diaria, oscila
entre 0,197 y 0,668. No es un control limpio (uno esta ocioso y el otro no), asi
que sirve para formular la hipotesis, no para confirmarla.

### 6.2 La prueba barata, DISENADA Y NO CORRIDA

Si alguna vez se quiere contestar: pasar el `VACUUM FULL` de diario a **semanal**
durante 2 semanas y comparar el **promedio** de RAM contra estas 2 semanas, no el
pico. Reversible en una linea. **Pre-registrar antes la regla de lectura** (que
diferencia cuenta como "mejora") -- con una sola serie y sin control, la
tentacion de leer el serrucho a favor es alta.

**No se corrio, por decision: el ahorro potencial es del orden de 1 USD/mes.**

---

## 7. La cuenta entera: este proyecto es la mitad

Proyeccion mensual con las tarifas de la sec. 2 (promedios de 7 dias; la fila de
IndicadoresML esta ademas validada contra la factura real):

| proyecto | servicios | ~USD/mes |
|---|---|---|
| **IndicadoresML** | Postgres | **4,54** |
| RRHH Head Hunter | Postgres + API Service + ProyectoRRHH | 3,12 |
| idoneidad | Postgres + idoneidadCNV | 1,36 |
| | **total cuenta** | **~9,0** |

Los 5 servicios de los otros dos proyectos estan prendidos 24/7 con CPU
practicamente en cero y memoria plana: **es costo de estar encendidos, no de
usarse**. `ProyectoRRHH` solo (0,18 GB de RAM) son ~1,9 USD/mes, el 40% de lo que
cuesta todo IndicadoresML.

**Si el objetivo alguna vez es gastar menos, ahi hay mas plata que en cualquier
cosa que se le haga al pipeline de este proyecto.** Queda anotado y fuera de
alcance: son proyectos ajenos a este repo.

---

## 8. CIERRE (29/10/2026, por decision del usuario)

**No se hace nada. Se continua como se esta.**

### Que dice el cierre

- El consumo de Railway esta **sano y en regimen**: 248 MB de DB, retencion
  funcionando, una captura por dia, egress explicado 1 a 1 por el sync.
- **Nada del trabajo de septiembre/octubre lo aumento**, y desde el 13/9 Railway
  recibe menos que antes.
- La factura la manda la **RAM (95%)**, no el disco. Esa es la linea de base.

### Que NO dice el cierre

- **No dice que el `VACUUM FULL` diario sea necesario.** Es desproporcionado
  (reescribe 205 MB para recuperar 20 MB) y se deja por decision de no tocar lo
  que anda, no porque este justificado. Si alguien lo mira de nuevo, que lea la
  sec. 6 antes de defenderlo o de sacarlo.
- **No mide si sacarlo bajaria la RAM.** La prueba esta disenada y sin correr
  (sec. 6.2).
- **No dice que la retencion de 10 dias sea optima.** Dice que bajarla ahorraria
  ~1 USD/mes y recortaria a la mitad el colchon de un dato IRRECUPERABLE que
  depende de un sync manual. **Se decidio no tocarla, y el motivo es el riesgo,
  no el costo.**
- **No cubre los otros dos proyectos** mas alla de cuanto cuestan (sec. 7).

### Como se reabriria

Con una de estas, no por iniciativa propia:

1. que la factura **se vaya del presupuesto** (hoy ~9 USD/mes la cuenta entera,
   ~4,5 este proyecto) o que Railway cambie las tarifas;
2. que vuelva a **detenerse el servicio** por limite, como el 20/7;
3. que entre a Railway un **consumidor nuevo** (reactivar los bots Alpaca y
   `push_senales_bot`, o mover algo del pipeline a la nube);
4. que la **RAM siga subiendo** sin estabilizarse -- el chequeo esta en la
   sec. 9 y lleva un minuto.

### Lo que queda, aunque la linea este cerrada

- La **linea de base medida** de este documento, para comparar contra ella sin
  volver a diagnosticar desde cero.
- La leccion que se va a CLAUDE.md: **en Railway paga la RAM, no el disco**, y
  por eso la intuicion heredada del incidente del 20/7 lleva a la palanca
  equivocada.
- La hipotesis del `VACUUM FULL` escrita con su prueba, por si alguna vez hay
  motivo para correrla.

---

## 9. Como re-medirlo (un minuto)

1. **Factura**: dashboard de Railway, "Cost by Service". Dividir costo/unidades
   de cada renglon reproduce las tarifas de la sec. 2; dividir las unidades por
   el promedio da el largo del ciclo.
2. **RAM**: la metrica `MEMORY_USAGE_GB` del servicio Postgres a 7 dias. Si el
   promedio se despego de **0,43 GB**, cambio algo.
3. **Datos**: el log del dia en `logs/rutina/AAAAMMDD_HHMM/01_sync.log` ya trae
   el tamano antes y despues de la purga, las filas borradas y las bajadas. Si
   "Dias en Railway" pasa de 11 o el tamano final de 205 MB, la purga no corrio.
4. **Lectores inesperados**: comparar el egress del ciclo contra
   (corridas del sync x ~42 MB). Si sobra, alguien mas esta leyendo Railway.
