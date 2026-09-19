# Reaccion a balances -- vista del dashboard + earnings_historico

Analiza el comportamiento REAL del precio y el volumen en una ventana SIMETRICA
(N ruedas ANTES y N DESDE el dia 0) alrededor de cada balance de un ticker
(desde 2020). Sin estimaciones ni sorpresa de analistas: solo el hecho duro
(cuando reporto) cruzado con como reacciono el precio. Decision del usuario:
interesa el impacto observado, no lo que el consenso esperaba.

Es un analisis DESCRIPTIVO / de psicologia de mercado, NO predictivo: no busca
adivinar la direccion del proximo gap (depende de la sorpresa, que no usamos).
Lo que caracteriza es la MAGNITUD tipica (riesgo-evento), el estilo de reaccion
(gapea y continua vs revierte) y el comportamiento pre-balance (run-up, carga de
volumen) -- propiedades relativamente persistentes por ticker. El overlay de
varios trimestres muestra la DISPERSION de resultados (eso es el hallazgo), no
un unico camino.

## El problema que resuelve: no teniamos la fecha de anuncio historica

Para dibujar la ventana post-balance hace falta la FECHA DE ANUNCIO por
trimestre. Ninguna tabla previa la daba:
- `earnings_calendar`: solo la PROXIMA fecha (1 fila/ticker).
- `fundamentales_*_q`: `fiscal_period_end` = CIERRE del trimestre, NO el anuncio.
  Hay ~2-6 semanas de diferencia (ej. AAPL cierra 31/3 y reporta ~30/4). Usar
  `fiscal_period_end` + 7 ruedas miraria el medio del trimestre siguiente,
  semanas antes de que el balance salga -> ventana sistematicamente mal, error
  silencioso. Se descarto de plano.

## Fuente: Alpha Vantage funcion EARNINGS

Una sola llamada por ticker devuelve toda la historia trimestral (AAPL: hasta
1996). Campos que usamos:
- `fiscalDateEnding` -> `fiscal_period_end` (empata con `fundamentales_*_q`).
- `reportedDate`     -> `announcement_date` (la fecha real del anuncio).
- `reportTime`       -> `report_time` ('pre-market' / 'post-market').
Trae ademas EPS/estimados/sorpresa, que NO se guardan (fuera de alcance).

Elegida sobre las otras keys disponibles (FMP/MarketStack/Nasdaq) porque da la
historia completa + el pre/post-market en 1 call. Verificado: pre/post-market se
reparte ~50/50 en el universo; AAPL siempre post-market (correcto).

### Restriccion de cuota (key FREE)
25 llamadas/dia, 5/min. El backfill de ~200 tickers NO entra en una corrida ->
`refresh_earnings_historico.py` es REANUDABLE y cuota-aware:
- Trae hasta `--max-calls` por corrida (default 20, margen bajo 25).
- Pausa 13s entre llamadas (>= 12s => <= 5/min).
- AV senaliza el tope con 'Note'/'Information' -> corte limpio, se reanuda al dia
  siguiente. Backfill inicial: ~10 corridas (1/dia).

## Tabla `earnings_historico` (LOCAL-only)

```
ticker             VARCHAR
fiscal_period_end  DATE          -- cierre del Q (JOIN con fundamentales_*_q)
announcement_date  DATE          -- fecha REAL del anuncio
report_time        VARCHAR       -- 'pre-market' | 'post-market' | NULL
fetched_at         TIMESTAMP
PK (ticker, fiscal_period_end)
```
LOCAL-only por la misma logica que fundamentales: dato historico recuperable de
una API, no necesita Railway (Plan C). Como AV devuelve historia completa por
llamada, cada fetch REEMPLAZA la del ticker (upsert idempotente).

## Regla del DIA 0 (clave para no correr la ventana)

El dia 0 es la primera rueda en que el mercado pudo reaccionar:
- `post-market`         -> la rueda habil SIGUIENTE al anuncio (cuando reporto,
  el mercado ya estaba cerrado).
- `pre-market` / NULL   -> la rueda del propio anuncio (o la siguiente si ese
  dia no operó).
Se resuelve contra las filas REALES de `precios_diarios` (que ya son dias
habiles): el dia 0 es el primer close cuya fecha cumple la condicion. No depende
del calendario NYSE. La base del % es el close de la rueda ANTERIOR al dia 0
(ultimo precio "limpio"), asi el gap de apertura queda medido dentro de la
ventana.

## Tres casos, un mecanismo

`refresh_earnings_historico.py` cubre todo con "conseguir 1 llamada por ticker
que lo necesita":
1. Backfill inicial -> tickers SIN filas.
2. Ticker nuevo     -> idem (al alta no tiene filas). Ademas `universo.py add`
   dispara un `--ticker X` directo (paso 6b); si la cuota estaba agotada, el
   batch lo levanta despues. Doble red, se auto-cura.
3. Incremental      -> tickers que DEBEN un balance segun su propia cadencia de
   anuncios -> apendicea el Q nuevo. Ver "Como se detecta el atraso".

Los tres comparten la MISMA cola (primero los que no tienen nada, despues del
mas atrasado al menos): la unica diferencia entre ellos es cuantos quedan.

## Como se detecta el atraso: por CADENCIA PROPIA (19/9/2026)

Modulo puro `src/utils/earnings_cobertura.py`, FUENTE UNICA del script que
puebla la tabla y de la vista del dashboard (para que no haya dos definiciones
de "esta al dia").

**Una tabla de EVENTOS no se vigila por antiguedad absoluta.** Entre temporadas
de balances, "el ultimo anuncio es de hace 6 semanas" es lo correcto, no un
atraso. Lo que si es un hecho del ticker es su CADENCIA: la mediana de dias
entre sus anuncios. Si paso mas que su cadencia por un margen (`MARGEN = 1.15`,
~14 dias sobre 91), debe un balance que no tenemos. Mediana y no promedio: un
cambio de cierre fiscal deja un intervalo raro que el promedio se lleva puesto.
Con cadencia fija de 91 dias, un semestral (HMY reporta cada 96, otros cada 182)
daria falso positivo todos los trimestres.

Medido el 19/9/2026 sobre el universo: cadencia mediana 91 dias, los 200
trimestrales, **108 de 200 tickers debian un balance** con la tabla frenada en
el 3/8.

## Ventana y filtros de la vista (dashboard/earnings_reaccion.py)

La vista se arma con `construir_series(ticker, anios, trimestres, n_ruedas)` y
muestra TRES paneles superpuestos por trimestre, con una linea vertical en el
dia 0 (separa pre de post):

1. **Precio (USD)** -- el cierre REAL en dolares (sin normalizar). Cada trimestre
   en su banda de precio; muestra el movimiento en crudo.
2. **Precio (%)** -- variacion acumulada vs el cierre de la rueda PREVIA al dia 0
   (ese punto es el 0%: "el ultimo precio limpio antes de la reaccion"). Es un
   nivel contra UNA referencia fija, NO el retorno dia-a-dia. Normaliza para
   comparar trimestres en la misma escala. El salto de offset -1 a 0 = gap de
   reaccion.
3. **Volumen** -- MULTIPLO (no %) del promedio de `VOL_BASE_N=50` ruedas ANTES de
   la ventana pre (referencia fija por evento). "1.0" = volumen normal previo;
   la linea horizontal en 1.0 ES esa media de 50 ruedas. Mas estable/legible que
   la direccion del precio.

Controles (sidebar): selector de ticker; **filtro de anios** (multi-seleccion,
por cierre fiscal); **toggle de trimestres Q1-Q4** (`st.pills` multi) -- ambos
combinan con AND, ej. Q1 en varios anios = estacionalidad de la reaccion; y
slider **N de ruedas por lado (1 a 10, default 7)**. La ventana es -N..N-1: el
dia 0 CUENTA como la primera rueda post. Ventanas truncadas (balance viejo sin N
ruedas antes, o reciente sin N despues) se muestran con lo que exista.

## Por que se atraso: el incremental era CIEGO (diagnostico 19/9/2026)

No es que no se corriera. **Corriendolo todas las noches tampoco habria traido
nada**, y eso es lo que hay que recordar de este episodio.

La deteccion preguntaba `earnings_calendar.earnings_date <= CURRENT_DATE AND >
nuestra ultima announcement_date`. Pero `earnings_calendar` guarda **solo la
PROXIMA fecha** de cada ticker y la refresca Oracle una vez por semana: el dia
que la empresa reporta, el refresh siguiente reemplaza esa fecha por la del
trimestre siguiente y el ticker **no vuelve a aparecer como desactualizado
nunca**. La unica ventana para detectarlo era el hueco entre el anuncio y el
proximo refresh del calendario.

Sintoma exacto al 19/9/2026, antes del arreglo:

```
Con historia    : 200 tickers (5225 filas, 2020-01-14 -> 2026-08-03)
Sin historia    : 0  []
Desactualizados : 0  []          <- con 108 de 200 debiendo un balance
```

`earnings_calendar` tenia las 200 filas con fechas FUTURAS (24/9 -> 6/11,
refrescada el 14/9): cero candidatos, siempre. Un guard que nunca falla y
devuelve un numero plausible -- la misma familia que `scan_fecha`,
`features_sector` y `fecha_datos` (ver CLAUDE.md, patrones criticos).

**Que NO arregla esto**: los meses ya perdidos se recuperan igual (Alpha Vantage
devuelve la historia completa por llamada), pero mientras estuvo atrasada,
cualquier analisis que necesitara EXCLUIR los dias de balance corrio con un
filtro parcial. Paso dos veces: limito la medicion de alertas de la Tarea 22 (el
panel marco 4% de ticker-dias contra ~11% esperable) y obligo a cortar la
re-simulacion de salidas de SMC_v1 en el 13/7 (ANALISIS_SALIDAS.md sec. 10.4).

### Nasdaq no reemplaza a Alpha Vantage para esto

El calendario de Nasdaq (`api.nasdaq.com/api/calendar/earnings?date=...`, sin
key, el mismo que usa `refresh_earnings_calendar.py`) tambien sirve para dias
PASADOS y no tiene cuota: 3 dias probados el 19/9/2026 devolvieron 284, 532 y 4
empresas. Tentador, pero **no alcanza como fuente**:

- `fiscalQuarterEnding` viene como MES ("Jun/2026"), no como fecha -> no empata
  con la PK `fiscal_period_end` sin un mapeo contra `fundamentales_*_q`.
- `time` viene **siempre** `'time-not-supplied'` en dias pasados -> se pierde
  `report_time`, y con el la regla del dia 0 para ~la mitad del universo
  (post-market = la rueda SIGUIENTE). Sin eso la ventana queda corrida un dia,
  en silencio, justo en el evento que se quiere medir.

Sirve como DISPARADOR gratis (quien reporto y cuando), no como fuente. Hoy no se
usa: la cadencia propia responde lo mismo sin salir a la red.

## Que corre solo, desde el 19/9/2026

`rutina_diaria.bat` tiene un paso nuevo, **`earnings`, el ultimo de todos**, con
politica INFORMAR (nunca frena la rutina):

- Va ultimo porque nada lo espera: `earnings_historico` **no es insumo de
  ninguna decision** (el filtro de balances de los bots lee `earnings_calendar`),
  alimenta al dashboard y a los analisis.
- Trae hasta 20 tickers por noche (tope de la key free) con 13s de pausa: ~4,5
  minutos. Se corta limpio si Alpha Vantage avisa que se acabo la cuota.
- Con ~4-5 balances por dia en temporada, 20 llamadas por noche sobran para
  quedar al dia sola. Un atraso de 108 se vacia en ~6 noches.
- Rehacerlo a mano: `scripts/manual/refresh_earnings_historico.bat`.

Primera corrida (19/9/2026): 20 tickers, 534 filas, la tabla paso de 2026-08-03
a 2026-08-27; quedaron 88 en cola.

## Backfill inicial via Oracle (transito por Railway) -- TEMPORAL

> ESTADO 4/8/2026: backfill COMPLETO (200/200 tickers, 5225 filas, cierre
> 2019-Q4..2026-Q2). Cron de Oracle **apagado** y **sync a local hecho** (local es
> ahora la fuente de verdad). Falta SOLO el paso 3: `DROP TABLE earnings_historico`
> en Railway (el usuario lo hara manual). De aca en mas: incremental en Windows
> (target local, default).

Para no gotear ~10 dias a mano en Windows, el backfill inicial corre en Oracle
(siempre-on) escribiendo a Railway, y se baja a local UNA vez al terminar:

```
1. Oracle cron diario 07:00 UTC: refresh_earnings_historico.py --target railway
   --backfill  (crea la tabla en Railway si no existe; ~20 tickers/dia)
2. NO se sincroniza durante la fase: Railway acumula sola ~10 dias.
3. Cuando --target railway --status dice 200/200:
   a. sync final:  sync_railway_to_local.py --tabla earnings_historico
      (merge idempotente ON CONFLICT DO NOTHING; preserva lo que ya hay en local)
   b. APAGAR el cron (quitar la linea en Oracle + scripts/oracle_crontab.txt)
   c. DROP TABLE earnings_historico en Railway (era transito; local es la verdad)
4. De ahi en mas: incremental en Windows (target local, default).
```

Notas: mismo limite AV (25/dia por key) -> Oracle NO acelera, solo desatiende.
Durante la fase, NO correr el backfill tambien en Windows con la misma key (se
reparten los 25 y se desperdicia cuota). La AV key vive en el .env de Oracle.

## Operacion

- Backfill inicial: en Oracle contra Railway (ver seccion anterior). Manual
  equivalente: `refresh_earnings_historico.py --backfill [--max-calls N]`.
- Ver que falta: `refresh_earnings_historico.py [--target railway] --status`.
- Incremental: corre SOLO como ultimo paso de `rutina_diaria.bat` (paso
  `earnings`). A mano: `scripts/manual/refresh_earnings_historico.bat`, o
  `refresh_earnings_historico.py` sin flags. Ticker puntual: `--ticker X` (lo
  usa universo.py add).
- Vista: dashboard -> "Reaccion a balances" (`dashboard/earnings_reaccion.py`).
  Selector de ticker + filtro de anios + toggle de trimestres Q1-Q4 + slider de
  ruedas por lado (1-10). Tres paneles (precio USD, precio %, volumen x prom 50),
  ventana pre+post superpuesta por trimestre, dia 0 marcado. Detalle en la
  seccion "Ventana y filtros de la vista".

## Estado medido el 15/9/2026 (lo que disparo el arreglo)

Aparecio de costado midiendo alertas (Tarea 22): 114 de 200 tickers sin ningun
`announcement_date` posterior al 2026-07-01, y agosto con 4 registros. La causa
de fondo, encontrada el 19/9, esta arriba ("el incremental era CIEGO"). Queda
como referencia de como se ve el problema desde afuera: **la vista se ve igual
de completa con la tabla atrasada** -- un trimestre que falta no se distingue de
un trimestre que no existe. Por eso ahora la vista muestra la cobertura y avisa
cuando al ticker elegido le falta un balance (`dashboard/earnings_reaccion.py`,
`_aviso_cobertura`).
