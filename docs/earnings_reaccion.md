# Reaccion a balances -- vista del dashboard + earnings_historico

Analiza el comportamiento REAL del precio y el volumen en las ruedas posteriores
a cada balance de un ticker (desde 2020). Sin estimaciones ni sorpresa de
analistas: solo el hecho duro (cuando reporto) cruzado con como reacciono el
precio. Decision del usuario: interesa el impacto observado, no lo que el
consenso esperaba.

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

## Backfill inicial via Oracle (transito por Railway) -- TEMPORAL

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

- Backfill / tanda diaria: `scripts/manual/refresh_earnings_historico.bat`
  (o `refresh_earnings_historico.py --backfill [--max-calls N]`).
- Ver que falta: `refresh_earnings_historico.py --status`.
- Incremental (post-earnings season): `refresh_earnings_historico.py` sin flags.
- Vista: dashboard -> "Reaccion a balances" (`dashboard/earnings_reaccion.py`).
  Selector de ticker + cuantos trimestres comparar; panel de precio (% acumulado
  desde el cierre previo) y de volumen (multiplo del promedio de 20 ruedas
  previas), superpuestos por trimestre, dia 0 marcado.
