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
3. Incremental      -> tickers cuya proxima fecha (`earnings_calendar`) ya paso
   respecto de la ultima `announcement_date` -> apendicea el Q nuevo.

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
- Incremental (post-earnings season, Windows->local): `refresh_earnings_historico.py`
  sin flags. Ticker puntual: `--ticker X` (lo usa universo.py add).
- Vista: dashboard -> "Reaccion a balances" (`dashboard/earnings_reaccion.py`).
  Selector de ticker + filtro de anios + toggle de trimestres Q1-Q4 + slider de
  ruedas por lado (1-10). Tres paneles (precio USD, precio %, volumen x prom 50),
  ventana pre+post superpuesta por trimestre, dia 0 marcado. Detalle en la
  seccion "Ventana y filtros de la vista".
