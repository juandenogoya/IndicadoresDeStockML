# Snapshot intradiario: barras de 30 min en dos cortes de la rueda (30/10/2026)

Captura del precio a mitad de rueda para, mas adelante, avisar por Telegram que
vale la pena ir a mirar un grafico ANTES del cierre. No es para operar en el
momento: la latencia no es critica.

- Estado: **ETAPA 1 en prueba en la rama `main-y9tr06`**, sin mergear. Solo
  captura y guarda; Telegram avisa UNICAMENTE fallas de la captura.
- No modifica nada de lo que ya funciona: todos los archivos son nuevos.
- Modulos puros: `src/utils/sesion_nyse.py` (horario y cortes) y
  `src/utils/intradia.py` (normalizacion, agregados, cobertura, spool, DDL).
  Tests: `tests/test_sesion_nyse.py`, `tests/test_intradia.py`.
- Captura: `scripts/intradia/snapshot_intradia.py` (VM de Oracle -> Railway).
- Sync a local + purga: `scripts/manual/sync_intradia_railway_to_local.py` (+ `.bat`).
- Compuerta de la etapa 1: `scripts/intradia/evaluar_etapa1.py` (LOCAL, solo lee).
- Tabla: `scripts/oneshot/create_precios_intradia_table.py --target ambos`.
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

## 7. Puesta en marcha en la VM (pendiente, manual)

Un segundo clon, porque los jobs de produccion hacen `git pull origin main` y
nunca verian la rama:

```
cd /home/ubuntu
git clone <url del repo> IndicadoresDeStockML-intradia
cd IndicadoresDeStockML-intradia
git checkout main-y9tr06
cp ../IndicadoresDeStockML/.env ../IndicadoresDeStockML/.env.local .
source ../IndicadoresDeStockML/venv/bin/activate

# 1. tabla en Railway (la de local se crea desde Windows con --target local)
python3 scripts/oneshot/create_precios_intradia_table.py --target railway

# 2. prueba sin escribir, sobre la ultima rueda cerrada (ver sec. 5)
python3 scripts/intradia/snapshot_intradia.py --dry-run --fecha AAAA-MM-DD --corte cierre_60

# 3. una captura real reconstruida, y el status
python3 scripts/intradia/snapshot_intradia.py --fecha AAAA-MM-DD --corte cierre_60
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

En Windows: `create_precios_intradia_table.py --target local` una vez, y
despues `sync_intradia_railway_to_local.bat` a mano durante la prueba.

## 8. Al mergear (no antes)

- Pasar la linea del cron a `scripts/oracle_crontab.txt` con `origin main` y la
  ruta del clon de produccion; borrar el clon de prueba y su linea.
- Decidir si el sync entra como paso de `rutina_diaria`.
- CLAUDE.md: arquitectura (excepcion al Plan C), tabla y scripts clave.
