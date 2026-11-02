# Snapshot intradiario: barras de 30 min en dos cortes de la rueda (30/10/2026)

Captura del precio a mitad de rueda y aviso por Telegram de que vale la pena ir
a mirar un grafico ANTES del cierre, sobre los tickers que la rutina de la noche
ya marco. No es para operar en el momento: la latencia no es critica.

- Estado: **ETAPAS 1 y 2 DESPLEGADAS EN PRUEBA el 2026-11-02**, en la rama
  `main-y9tr06`, sin mergear. El cron de la VM captura solo desde esa fecha
  (sec. 7, puesta en marcha HECHA). Etapa 1 = captura (sec. 1-8): los 4
  supuestos de la sec. 5 quedaron VERIFICADOS, pero su compuerta necesita 2-3
  semanas de cron y todavia no se puede evaluar. Etapa 2 = alerta de la lista
  de vigilancia (sec. 9): criterios 1, 2 y 3 MEDIDOS y pasan; el 4 dice que la
  alerta ADELANTA al mensaje de la noche; el 5 (juicio del usuario) sigue
  abierto. Resultados en la sec. 9.4.
- No modifica nada de lo que ya funciona: todos los archivos son nuevos; la
  etapa 2 solo IMPORTA funciones de la foto y del cierre.
- Modulos puros: `src/utils/sesion_nyse.py` (horario y cortes),
  `src/utils/intradia.py` (normalizacion, agregados, cobertura, spool, DDL) y
  `src/utils/vigilancia_intradia.py` (lista, eventos, novedades, mensaje, DDL).
  Tests: `tests/test_sesion_nyse.py`, `tests/test_intradia.py`,
  `tests/test_vigilancia_intradia.py`.
- Captura + alerta: `scripts/intradia/snapshot_intradia.py` (VM de Oracle <-> Railway).
- Lista de vigilancia: `scripts/intradia/push_vigilancia.py` (LOCAL -> local + Railway).
- Despues de la rutina: `scripts/manual/intradia_post_rutina.bat` (lista + sync).
- Sync a local + purga: `scripts/manual/sync_intradia_railway_to_local.py` (+ `.bat`).
- Compuertas: `scripts/intradia/evaluar_etapa1.py`, `evaluar_etapa2.py` y
  `auditar_etapa2.py` (LOCAL, solo leen).
- Tablas: `scripts/oneshot/create_precios_intradia_table.py --target ambos`.
- Cron del clon de prueba: `scripts/intradia/oracle_crontab_intradia.txt`.

## 1. Decisiones (30/10/2026)

| Tema | Decision | Por que |
|---|---|---|
| Momento | `apertura_90` = apertura + 90 min (11:00 ET); `cierre_60` = cierre - 60 min (15:00 ET, 12:00 ET en media rueda) | Una hora FIJA de Argentina se corre contra la sesion: ARG no cambia de hora y NY si (ver sec. 2) |
| Fuente | yahooquery, barras de 30 min | Ya esta en la VM (es el engine de opciones), sin key, los 199 en un lote |
| Que se captura | Las BARRAS hasta el corte, no la cotizacion del instante | El corte queda exacto aunque el cron arranque tarde, y el MISMO metodo sirve para captura, reconstruccion y medicion historica |
| Que se guarda | Las barras en crudo, todo el universo | Los indicadores no estan decididos: guardar un resumen decidiria hoy que se pierde (ej. el volumen de CADA barra hace falta para comparar "volumen a las 11:00" contra lo normal de esa hora) |
| Donde corre | VM de Oracle | Las alertas van a Telegram y la PC no esta prendida a esas horas |
| Donde escribe | Railway (buffer, ~30 ruedas) -> local (historia) | Mismo esquema que el crudo de opciones. **Primera excepcion al Plan C** ("Railway = SOLO opciones_snapshot"), aceptada: la factura de Railway la manda la RAM, no el disco (docs/consumo_railway.md), y esto son ~2.200 filas por rueda |
| Reintento | Uno por corte | Un corte perdido se RECONSTRUYE: Yahoo guarda barras de 30 min ~60 dias (a diferencia de la chain de opciones, que es irrecuperable) |
| Medicion de concordancia (la foto del corte vs el cierre) | Etapa 2 | Necesita una SENAL definida, y los indicadores son la etapa 2. Con barras se puede bajar la historia cuando haga falta |

Fuentes descartadas para la captura en vivo:

- **yfinance**: mismo backend y mismo limite por IP que yahooquery -> no es
  respaldo (si Yahoo falla, fallan los dos).
- **IBKR**: decision del 22/10/2026 (nada de IBKR entra al pipeline,
  docs/fuentes_ibkr.md) y el conector de claude.ai no se llama desde un cron.
  Sirve para CONSULTA: verificado el 30/10 que sus barras de 30 min estan
  alineadas a :00/:30 desde las 9:30 ET (existen las que cierran a las 11:00 y
  15:00 ET) y que devuelve la barra en curso, igual que Yahoo.
- **Alpha Vantage**: la key gratuita da 25 pedidos/dia (la captura necesita
  ~400) y compite con el paso `earnings` de la rutina.

## 2. Horarios

Argentina es UTC-3 todo el anio; NY es UTC-4 de marzo a noviembre y UTC-5 en
invierno.

| Periodo | apertura_90 | cierre_60 |
|---|---|---|
| Horario de verano EE.UU. (mar-nov) | 15:00 UTC / 12:00 ARG | 19:00 UTC / 16:00 ARG |
| Invierno | 16:00 UTC / 13:00 ARG | 20:00 UTC / 17:00 ARG |
| Media rueda (cierra 13:00 ET; caen en invierno) | 16:00 UTC / 13:00 ARG | 17:00 UTC / 14:00 ARG |

**Medias ruedas cargadas** (`CIERRES_ANTICIPADOS`): 24/12/2025, 27/11/2026,
24/12/2026 y 26/11/2027. Confirmadas contra el comunicado oficial de NYSE del
23/12/2025 ("2026, 2027 and 2028 Holiday and Early Closings Calendar"). Un
calendario de terceros lista medias ruedas que NYSE no tiene (18/6, 4/9 y
25/11/2026). En 2027 no hay media rueda de Nochebuena: el 24/12/2027 es feriado.

**Cobertura**: `trading_calendar` tiene feriados hasta 2027. Fuera de eso
`sesion_nyse.cortes()` devuelve `{}` y el snapshot avisa una vez por dia por
Telegram: sin el guard, un feriado de 2028 pasaria por dia habil sin avisar.
Al cargar 2028 hay que agregar tambien sus medias ruedas.

**Disparo**: el cron no sabe de "relativo a la sesion". Dispara a los minutos 5
y 20 de las 15, 16, 17, 19 y 20 UTC, de lunes a viernes, y el script decide
(`cortes_vencidos`): un corte se captura entre 3 y 45 min despues de su hora.
Los 3 min dejan cerrar la barra que termina en el corte; los 45 cubren el :05 y
el :20 de la misma hora y no el de la hora siguiente (que es del otro horario).
El test `test_el_cron_cubre_cada_corte_2026_2027_con_intento_y_reintento`
verifica sobre cada rueda de 2026-2027 que cada corte cae en exactamente 2
disparos, y otro test que el crontab dice lo mismo que las constantes.

## 3. Tabla `precios_intradia_30m` (Railway + local)

PK `(ticker, corte, inicio_barra)`; indice `(fecha, corte)`.

| Columna | Que es |
|---|---|
| `fecha` | rueda de NY de la barra (NO la fecha de la captura ni la de UTC) |
| `corte` | la captura que guardo la fila |
| `inicio_barra` | inicio de la barra de 30 min (TIMESTAMPTZ) |
| `open/high/low/close/volumen` | OHLCV de la barra |
| `hora_corte` | instante del corte: la ultima barra TERMINA ahi |
| `capturado_en` | reloj de la corrida (NO es fecha de datos; mide la demora) |

**Una fila por CAPTURA, no por barra**: la barra de las 9:30 queda dos veces,
como la vio `apertura_90` y como la vio `cierre_60`. La fila es un registro
POINT-IN-TIME de lo que el sistema sabia en cada corte (si Yahoo revisa una
barra, el `cierre_60` no pisa lo que vio la apertura), y la diferencia entre las
dos copias MIDE esas revisiones. Costo: 14 filas por ticker y rueda en vez de 11.

El OHLCV acumulado hasta el corte NO se guarda: sale de
`intradia.agregar_corte()` (open de la primera barra, close de la ultima, max,
min, suma de volumen).

Barras esperadas por ticker: 3 en `apertura_90`, 11 en `cierre_60`, 5 en el
`cierre_60` de una media rueda (`intradia.barras_esperadas`).

## 4. Que hace una captura

1. Reinyecta spools pendientes de un disparo anterior con la DB caida.
2. Si el corte ya esta en la DB (>= 90% del universo con la barra que termina
   en el corte), sale. Si la DB no responde, captura igual.
3. Lock de Yahoo, descarga, normaliza: descarta pre-market, otras ruedas, barras
   sin close y la barra EN CURSO (solo quedan las que terminan en el corte o antes).
4. Escribe el spool (`data/intradia_spool/`) ANTES de la DB.
5. Upsert en Railway; si confirmo, borra el spool. Si no: aviso "DB inaccesible,
   dato a salvo en disco" y el proximo disparo lo reinyecta solo.
6. Cobertura < 90% -> aviso con los tickers que faltan.

Salida: 0 OK o nada que hacer / 2 aviso / 1 error. El error manda sobre el aviso.

**Cobertura** = tickers con la barra que TERMINA en el corte. Un ticker con las
barras anteriores pero sin esa no cuenta: su precio del corte seria de media
hora o mas atras.

**Reconstruir un corte perdido** (dentro de la ventana de ~60 dias de Yahoo):

```
python scripts/intradia/snapshot_intradia.py --fecha AAAA-MM-DD --corte cierre_60
```

Un corte reconstruido cuenta como FALLA de la captura en vivo en la evaluacion
(su demora es de horas o dias): la etapa 1 mide si el cron captura solo.

## 5. Supuestos a verificar en la primera corrida real

Desde la sesion en la nube donde se programo no se llega a Yahoo (el proxy
rechaza la conexion), asi que la forma real de la respuesta no se vio. Lo
cubren los tests con las variantes posibles, pero hay que mirarlo:

1. **Zona horaria del indice**: aware en hora de NY (lo esperado) o naive. Naive
   se interpreta como hora de NY. El `--dry-run` imprime el indice crudo y las
   barras normalizadas: la de las 9:30 ET tiene que quedar a las 13:30 UTC en
   verano y 14:30 UTC en invierno.
2. **start/end en intradiario**: si Yahoo no los respeta, el script cae a
   `period` (5d o 60d) y filtra la rueda.
3. **Ventana de ~60 dias de las barras de 30 min**: es el limite conocido de
   Yahoo, no medido aca.
4. **Volumen**: si la suma de las barras se parece al volumen diario. La
   evaluacion lo informa (ratio volumen al corte / diario).

### 5.1 Verificados el 2026-11-02 (los cuatro)

`--dry-run --fecha 2026-10-30 --corte cierre_60` desde la VM: **199/199 de
cobertura, 2.189 filas = 199 x 11 exacto, 5 segundos**.

1. **Zona horaria: CONFIRMADO el caso esperado.** El indice vino *aware* en hora
   de NY (`Timestamp('2026-10-30 09:30:00-0400', tz='America/New_York')`) y la
   barra de las 9:30 ET quedo en **13:30 UTC**, que es lo que corresponde en
   horario de verano. No hizo falta la rama de naive.
2. **start/end: CONFIRMADO, Yahoo los respeto.** No se uso el fallback a
   `period`. Volvio solo la rueda pedida y **sin pre-market**: la primera barra
   es la de las 9:30, no las de las 4:00 / 8:00 ET.
3. **Ventana de ~60 dias: CONFIRMADA y es real.** `--reconstruir 40` trajo
   entera la rueda del **2026-09-04**, a 59 dias corridos: **1 dia de margen**.
   Al dia siguiente esa punta ya no se podia pedir. Si alguna vez hay que
   reconstruir, la fecha importa.
4. **Volumen: coherente.** AA en la rueda 2026-10-30, acumulado al `cierre_60`
   contra `precios_diarios`: open **39,03 = 39,03** exacto (la primera barra es
   la apertura real de la sesion), high **39,77 = 39,77**, low 38,60 dentro del
   rango diario (38,59: el minimo del dia llego en la ultima hora), close 38,68
   vs 39,19 (difiere, falta la ultima hora, es lo esperado) y volumen
   **1.810.700 / 3.565.300 = 0,508**, por debajo del 1,02 que la compuerta
   marca como incoherente.

## 6. Compuerta de la etapa 1 (pre-registrada el 30/10/2026)

`evaluar_etapa1.py` sobre 2-3 semanas de captura. PASA si se cumplen las cuatro:

1. cobertura >= 90% en TODOS los cortes capturados;
2. cero cortes faltantes en el periodo (se listan; la explicacion se escribe
   pero el criterio no se relaja);
3. demora mediana <= 10 min y maxima <= 30 min;
4. high/low acumulado del corte fuera del rango diario de `precios_diarios` en
   <= 1% de los ticker-cortes (tolerancia 0,1%).

Informativo, sin umbral: revisiones de Yahoo entre las dos copias de una barra,
open de la primera barra vs open diario, volumen al corte / diario (un ratio >
1,02 es incoherente y se cuenta). Probado contra datos sinteticos que cada
criterio PUEDE dar NO PASA.

**La ventana arranca el 2026-11-02**, primer disparo del cron (16:05 UTC). Las
40 ruedas reconstruidas ese mismo dia **NO cuentan** para esta compuerta, por
diseno: su `capturado_en` es de horas o dias despues del corte y lo que se mide
aca es si el cron captura solo. Antes de ~2026-11-20 `evaluar_etapa1.py` no
tiene con que contestar.

## 7. Puesta en marcha (HECHA el 2026-11-02)

> **ESTADO: EJECUTADA.** Tablas en local y en Railway, clon de prueba en la VM
> en `main-y9tr06`, `--dry-run` verificado (sec. 5.1), 40 ruedas reconstruidas,
> cron agregado (10 -> 11 jobs, backup en `~/crontab_backup_20261102_1512.txt`,
> diff = 1 linea agregada y 0 quitadas) y la historia bajada a local. Lo que
> sigue son los comandos tal como quedaron, con **una correccion respecto de lo
> que decia este doc** (el `.env.local`, abajo).

**CORRECCION (2026-11-02): en la VM no existe `.env.local`.** Este doc decia
copiar `.env` y `.env.local` desde el clon de produccion, pero produccion solo
tiene `.env` (ahi Railway ES el destino normal, asi que su `DATABASE_URL` ya
apunta a Railway). Consecuencias:

- `snapshot_intradia.py` anda igual: carga `.env` y `.env.local` **solo si
  existe**, y despues `get_engine()` cae en Railway por `DATABASE_URL`.
- `create_precios_intradia_table.py --target railway` **NO corre en la VM**:
  tiene hardcodeado leer el DSN de `.env.local` y aborta. Las tablas de Railway
  se crean **desde Windows**, donde `.env.local` si lo tiene. Se verifico
  despues, desde la VM, que ve esas mismas 5 tablas en la instancia que tiene
  `opciones_snapshot`.

**En la VM**, un segundo clon, porque los jobs de produccion hacen
`git pull origin main` y nunca verian la rama:

```
cd /home/ubuntu
git clone <url del repo> IndicadoresDeStockML-intradia
cd IndicadoresDeStockML-intradia
git checkout main-y9tr06
cp ../IndicadoresDeStockML/.env .        # en la VM NO hay .env.local (ver arriba)
source ../IndicadoresDeStockML/venv/bin/activate

# 1. las tablas NO se crean desde aca: --target railway lee el DSN de .env.local,
#    que en la VM no existe. Las 5 de Railway Y las de local se crean desde
#    Windows, con --target ambos (ver el bloque de Windows mas abajo).

# 2. prueba sin escribir, sobre la ultima rueda cerrada (ver sec. 5)
python3 scripts/intradia/snapshot_intradia.py --dry-run --fecha AAAA-MM-DD --corte cierre_60

# 3. historia para la evaluacion de la etapa 2: las ultimas 40 ruedas, sin alertas
python3 scripts/intradia/snapshot_intradia.py --reconstruir 40
python3 scripts/intradia/snapshot_intradia.py --status

# 4. cron: AGREGAR, nunca `crontab archivo` (borraria los jobs de produccion)
crontab -l > ~/crontab_backup_$(date +%Y%m%d_%H%M).txt
( crontab -l ; grep '^[0-9]' scripts/intradia/oracle_crontab_intradia.txt ) | crontab -
crontab -l
```

La linea usa `source`: depende del `SHELL=/bin/bash` que ya esta al principio
del crontab de produccion. El lock de Yahoo vive en `logs/` de cada clon, asi
que los dos clones no se ven; no se pisan por horario (opciones US corre a las
23:00, 02:00 y 06:00 UTC; opciones AR y scanner AR usan IOL).

**En Windows**, una vez (las DOS puntas, no solo local: ver el paso 1 de arriba):

```
python scripts\oneshot\create_precios_intradia_table.py --target ambos
```

y despues, todos los dias, `scripts\manual\intradia_post_rutina.bat` al terminar la
rutina (sube la lista de vigilancia de la rueda y corre el sync).

## 8. Al mergear (no antes)

- Pasar la linea del cron a `scripts/oracle_crontab.txt` con `origin main` y la
  ruta del clon de produccion; borrar el clon de prueba y su linea.
- `push_vigilancia.py` y el sync como pasos de `rutina_diaria`, despues de
  `cierre` (politica INFORMAR: nada depende de la alerta).
- CLAUDE.md: arquitectura (excepcion al Plan C), tablas y scripts clave.

## 9. Etapa 2: alerta de la lista de vigilancia (decidida el 30/10/2026)

**Idea.** La rutina de la rueda D ya dice que tickers vigilar: la foto
(docs/telegram_foto.md) y el cierre por sector (docs/cierre_grupos.md). En cada
corte de la rueda siguiente, la alerta informa que paso HASTA ESE MOMENTO con
esos tickers respecto de los niveles por los que se los vigila. No recorre el
universo ni inventa senales. Es ESTADO, no pronostico.

### 9.1 La lista de vigilancia de la rueda D

`push_vigilancia.py`, en LOCAL, despues de la rutina (que hoy corre antes de la
apertura: la lista de D llega antes del corte de las 11:00 ET de D+1).

| Origen | Que entra |
|---|---|
| Foto | 2+ familias de FT + top 3 por familia (tope 15), de la misma corrida de FT |
| Cierre: cruces de D | muro OI, swing N5, SMA50/200, arriba y abajo |
| Cierre: vigilar | a menos de 1% de un muro OI o de un swing N5; cambio de veredicto |
| ~~Balances en 5 ruedas~~ | **no entran** (decision 30/10/2026) |

Las definiciones se IMPORTAN de `telegram_foto.py` / `telegram_cierre.py`: la
lista dice lo mismo que los mensajes de la noche.

Niveles de cada ticker, todos congelados con los datos de D:

| Nivel | Fuente |
|---|---|
| soporte / resistencia OI (ventana corta) | `opciones_pcr_plazo_diario` de D |
| swing low / high N5 | `features_estructura` de D: la tabla guarda la distancia y no el precio; sale exacto del close, `nivel = close / (1 + dist/100)` (columna DOUBLE) |
| SMA50 / SMA200 | `indicadores_tecnicos` de D |
| **el nivel cruzado en D** | muro OI o swing de **D-1**, el que se cruzo. El muro de D se recalcula con el close de D y queda debajo del precio por construccion (misma razon por la que el cierre usa "el muro de la rueda anterior"). En las SMA el cruce se marca en la propia fila |

Cada nivel guarda de que lado quedo el close de D (`lado_d`).

**Tablas** (local + Railway, dual-write): `vigilancia_listas` (cabecera: una
fila por rueda, aunque la lista este vacia; sin ella Oracle diria "lista no
subida" cuando no habia nada que vigilar), `vigilancia_tickers` (close de D,
desvio, motivos) y `vigilancia_niveles`. Un push de D reemplaza la lista de D.

### 9.2 Eventos y mensaje (pre-registrados)

| Evento | Definicion |
|---|---|
| cruce | el precio del corte quedo del otro lado del nivel que el close de D |
| revierte | idem sobre un nivel cruzado EN D: deshace el cruce de ayer (ej. "perdio el soporte de OI" y vuelve arriba) |
| movimiento | \|z\| >= 2,0, z = ln(precio del corte / close de D) / desvio de los retornos log diarios de las 252 ruedas hasta D (con menos de 126 no se informa). Mismo umbral en los dos cortes: a las 11:00 es exigente (media rueda contra el desvio de un dia). Limitacion declarada; si se cambia, es un cambio registrado |

Precio del corte = close de la barra que termina en el corte; un ticker sin esa
barra no evalua y el mensaje lo cuenta ("sin dato del corte").

- **11:00**: todos los eventos.
- **15:00**: solo lo NUEVO respecto de las 11:00, y lo que SE DESHIZO (se
  describe como estaba a las 11:00). Si el mensaje de las 11:00 no salio con la
  misma lista (no habia lista, o fallo), a las 15:00 se informa todo.
- Sin eventos: una linea "N vigilados, sin novedades" (el silencio no se
  confunde con "no corrio": leccion del cero falso del cierre, 14/10/2026).
- Sin la lista de la rueda habil anterior: lo dice y no evalua. Si la lista
  llega tarde, el reintento del corte la usa.
- Un mensaje por corte, registrado en `vigilancia_envios` (no se repite en el
  reintento). Si no entra en 4.000 caracteres se achican las listas, no se
  parte el HTML.
- Distancia al nivel con 2 decimales (con 1, un cruce por 0,04% se leia
  "-0,0%", como le paso al cierre).

### 9.3 Criterios (pre-registrados el 30/10/2026, antes de correr nada)

`evaluar_etapa2.py` (2 a 4) y `auditar_etapa2.py` (1), sobre las ruedas con
barras en local: las reconstruidas (`--reconstruir 40` en la VM) con listas
reconstruidas (`push_vigilancia.py --desde`), y despues las reales.

| # | Criterio | Pasa si |
|---|---|---|
| 1 | Correcto | un SQL independiente reproduce el 100% de los eventos de cada corte |
| 2 | Dispara con sentido | por TIPO, la mediana de lineas por corte esta entre 1 y 15. Un tipo que falla sale del mensaje; si fallan los tres, se duerme |
| 3 | Cabe | un mensaje por corte (por construccion; se informa cuantos achicaron listas) |
| 4 | Informativo | de los eventos de cada corte, cuantos seguian al cierre de la rueda. Dice si la alerta adelanta al mensaje de la noche o es ruido. NO mide rendimiento |
| 5 | Sirve | juicio del usuario despues de 5-10 ruedas reales. Se escribe aca con la fecha |

Riesgo conocido de la regla 2, escrito antes de ver datos: `revierte` es un
evento raro por construccion (necesita un cruce en D), y su mediana puede dar 0.
Si pasa, la regla lo saca del mensaje; no se mueve el umbral despues de verlo.

**Aproximaciones de la reconstruccion** (solo para la evaluacion historica, no
en vivo): la corrida de FT de la rueda D es la ULTIMA entre D y la habil
siguiente (`ft_candidatos_diarios.fecha` es la de corrida, no la de datos); la
foto solo ve las estrategias activas HOY; las barras reconstruidas tienen
`capturado_en` de hoy (no son point-in-time).

Probado de punta a punta contra una base desechable y un yahooquery falso: los
tres tipos de evento, las novedades y lo deshecho a las 15:00, el "sin lista",
la lista que llega tarde, el no repetir el envio, y que el auditor PUEDE fallar
(con el motor sesgado 0,5% da diferencias en 2 de 6 cortes).

### 9.4 Resultados medidos (2026-11-02)

Sobre **78 cortes de 39 ruedas** (2026-09-04 -> 2026-10-30): barras
reconstruidas en la VM y listas reconstruidas con `push_vigilancia.py --desde
2026-09-03 --hasta 2026-10-29` (39 listas, origen `reconstruida`) mas el `push`
real del 2026-10-30. **Tamano de la lista: mediana 77 tickers, min 52, max
110.** La seccion 9.3 no se toco: esto va aparte.

**Criterio 1 (correcto): PASA al 100%.** 78 cortes, **3.193 eventos** del motor,
todos reproducidos por el SQL independiente de `auditar_etapa2.py`.

**Criterio 2 (dispara con sentido): PASA en los tres tipos** sobre el agregado
de los dos cortes -- cruce 12,5 | revierte 6,0 | movimiento 1,0, todos dentro
de 1 a 15. **Pero el agregado esconde una asimetria que hay que declarar**:

| tipo | apertura_90 | cierre_60 |
|---|---|---|
| cruce | **mediana 22,0** (9 a 49) | 7,0 (1 a 26) |
| revierte | **14,0** (3 a 35) | 3,0 (0 a 11) |
| movimiento | 1,0 (0 a 7) | 1,0 (0 a 12) |
| deshechos | -- | 7,0 (2 a 16) |

A las 11:00 el mensaje lleva ~22 lineas de cruce mas ~14 de revierte: **~36
lineas**. Pasa porque el `cierre_60`, que es tranquilo, tira la mediana del
agregado para abajo. El pre-registro dice "la mediana de lineas **por corte**" y
es **ambiguo** entre agregar los dos cortes o medir cada tipo de corte por
separado; el script usa el agregado. **El umbral no se movio despues de ver los
datos** (CLAUDE.md: los parametros elegidos mirando resultados son sobreajuste
hecho a mano): queda escrita la ambiguedad, no corregida hacia atras. Si alguna
vez se decide medir por tipo de corte, `cruce` y `revierte` de `apertura_90` NO
pasan, y eso es una decision, no un ajuste.

**El riesgo pre-registrado de `revierte` NO se cumplio.** La sec. 9.3 avisaba
que podia dar mediana 0 por ser un evento raro; dio **6,0**. La preocupacion
estaba al reves: es de los que mas dispara.

**Criterio 3 (cabe): PASA.** Largo maximo **2.851 caracteres** contra el tope de
4.000 y **0 cortes** tuvieron que achicar listas.

**Criterio 4 (informativo): la alerta ADELANTA, no es ruido.**

| corte | eventos | siguen al cierre |
|---|---|---|
| apertura_90 | cruce 862 / revierte 563 / movimiento 83 | 79% / 79% / 78% |
| cierre_60 | cruce 1004 / revierte 566 / movimiento 115 | 88% / 91% / 81% |

Las dos lecturas, porque las dos son ciertas. **A favor**: 8 de cada 10 eventos
de las 11:00 y 9 de cada 10 de las 15:00 seguian vigentes al cierre, asi que no
son parpadeos intradiarios. **En contra**: que el 88% de lo que avisa a las
15:00 siga vigente al cierre significa que casi todo eso **va a aparecer esa
misma noche en el mensaje del cierre**; el valor marginal son las ~4 horas de
anticipacion, no informacion nueva. Eso es exactamente lo que tiene que juzgar
el criterio 5.

**Criterio 5: ABIERTO.** Pendiente del juicio del usuario tras 5-10 ruedas
reales. Se escribe aca con la fecha.

**2 cortes de los 80 quedaron sin evaluar, y el motivo NO es de esta linea**:
falta la lista del **2026-09-22**, porque esa rueda tiene 60 de 199 tickers en
`precios_diarios` (es el hueco que `chequeo_rutina.py` viene informando: "139
tickers", avisa y no frena) y 0 en `veredictos_universo_diario`.
`push_vigilancia.py` **se nego a armar la lista** sobre una rueda incompleta en
vez de emitir una plausible pero mal. Los 2 cortes afectados son los del
2026-09-23.

**Observacion operativa**: el dual-write de la lista a Railway tardo **~5
minutos** para 111 tickers y 602 niveles (local: segundos). Como
`intradia_post_rutina.bat` va a correr esto todos los dias, conviene tenerlo
presente; no se toco nada.
