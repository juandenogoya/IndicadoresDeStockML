# Gestion de universo -- alta / baja de tickers

Estado: EN PRODUCCION desde 18/6/2026 (Tarea 14, merge 533cf26). Solo ACCIONES.
Estado operativo vivo: memory/gestion_universo.md.

Como incorporar o quitar tickers del universo de seguimiento, y por que cada
decision es como es. El SCRIPT es lo facil; lo importante (y lo que documenta
este archivo) son las implicancias.

## 1. Fuente UNICA del universo: la tabla `activos`

Antes el universo se servia de DOS lugares desincronizados:
- la tabla `activos` (respeta `activo=TRUE`) -> la leian recovery, cron, etc.
- la lista HARDCODEADA `config.ALL_TICKERS` -> la leian el snapshot de opciones,
  el scanner, refresh fundamentales/pais.

Problema: dar de alta/baja en la tabla NO se reflejaba en los procesos que leian
config. Solucion (18/6): `src/data/universo.py`.

- `get_universo(solo_activos=True)` -> lista de tickers desde `activos`
  (la DB a la que apunte get_engine: local o Railway; `activos` vive en AMBAS,
  sincronizada). `get_universo_sectores()` -> {ticker: sector}.
- Fallback de seguridad: si `activos` no responde / esta vacia -> cae a
  `config.ALL_TICKERS` con un WARN. El switch nunca deja al pipeline sin universo.
- Consumidores migrados (leen `get_universo`): `33_opciones_snapshot`,
  `17_scanner_alertas`, `refresh_fundamentales`, `refresh_ticker_pais`,
  `cron_diario` (precios + paso_scanner).
- `config.ALL_TICKERS` NO se elimina: lo usan scripts legacy ML/BT y queda como
  fallback.

Consecuencia clave: **modificar `activos` se refleja en TODO el pipeline vivo**.
El snapshot de opciones (Oracle -> Railway) captura altas/bajas automaticamente,
porque lee `activos` de Railway.

## 2. El insight del backfill: el ticker nace CALIENTE

Al incorporar, lo PRIMERO es backfillear 2 anios de OHLCV. Como casi todo deriva
de la historia de PRECIOS, se calcula retroactivamente -> el ticker nace listo,
sin esperas de "warmup":

| Deriva de PRECIOS (backfilleable -> dia 1) | Deriva de SNAPSHOTS de OPCIONES (NO backfilleable) |
|---|---|
| indicadores (SMA200/ATR/MACD/RSI), features (SMC/PA), z-scores de ACCIONES, multiplos *_px, fundamentales | z-scores de OPCIONES (vol_z, pcr_z, iv_z): necesitan ~60d de snapshots acumulados |

Yahoo solo sirve la chain de opciones VIGENTE (irrecuperable) -> los z-scores de
opciones son el unico residuo con espera real. PCR, muros (S/R) y OI son dia 1
(salen del snapshot del dia). Profundidad 2 anios elegida: cubre SMA200, la
ventana de 60d de z de acciones, y deja base para backtesting futuro.

## 3. El script: `scripts/manual/universo.py` (+ .bat)

`add TICKER --sector S [--industry I] [--nombre N] [--anios 2] [--no-features]
[--no-fundamentales] [--dry-run]`:
1. Descarga 2a OHLCV (yahooquery) -> `precios_diarios` (LOCAL).
2. UPSERT en `activos` (sector/industry, `activo=TRUE`, `modelo_asignado=global_rf`)
   -> **DUAL-WRITE local + Railway**.
3. Indicadores (`procesar_indicadores_ticker`) + features (precio_accion +
   market_structure, BULK -> recomputa todos, idempotente).
4. Z-scores de acciones (`backfill_zscore_tickers`, desde el inicio de la historia).
5. Cadena fundamental: refresh_fundamentales -> ratios -> ticker_pais ->
   comparativa sectorial + `compute_multiplos_px` + `compute_sector_valuacion_px`
   (los *_px quedan dia 1).
6. Log en `universo_cambios`.

`remove TICKER --motivo "..." [--force] [--dry-run]`:
- **Guard**: si hay posiciones FT abiertas (`ft_operaciones`, LOCAL) -> ABORTA
  salvo `--force`. Las posiciones Alpaca paper se reportan (no bloquean: son de
  prueba; lo critico son las estrategias FT locales).
- **SOFT delete**: `activos.activo=FALSE` -> DUAL-WRITE local + Railway. Conserva
  TODA la historia. (Hard delete descartado: rompe analisis pasados y backtests.)
- Log en `universo_cambios`.

`list`: universo actual (activos / bajas) + ultimos cambios.

CANONICO: el `--sector` debe ser el string EXACTO del sistema (ej. "Financial
Services", no "Finanzas/Brokers") -- los bots sectoriales filtran por sector
exacto. Yahoo (assetProfile) es la referencia de sector/industry.

### Dual-write de `activos` (local + Railway)
El snapshot corre Oracle -> Railway, asi que `activos` debe quedar IGUAL en ambas.
El `add`/`remove` escriben en local (get_engine local) y en Railway (engine
dedicado desde `.env.local`, patron de `push_senales_bot._engine_railway`). Sin
el dual-write, el snapshot no veria el alta/baja hasta sincronizar.

## 4. Tabla `universo_cambios` (local + Railway)

1 fila por evento: `ticker, accion ('ALTA'|'BAJA'), fecha, sector, motivo,
detalle JSONB`. Log de auditoria para reproducibilidad point-in-time ("este
sector tenia N tickers en tal fecha"). La crea
`scripts/oneshot/create_universo_cambios_table.py`.

## 5. Point-in-time: ya satisfecho por construccion

Los agregados sectoriales de opciones (`calcular_pcr_sector_plazo`,
`calcular_zscore_opciones_sector`) se computan POR FECHA, agregando los tickers
que TIENEN DATO ese dia (filas en la tabla por-ticker), agrupados por el sector
de `activos`:

```
FROM opciones_pcr_plazo_diario p JOIN activos a ON p.ticker=a.ticker
WHERE p.fecha = :f GROUP BY a.sector, ...
```

Por eso el alta/baja NO reescribe la historia:
- ALTA: el ticker nuevo no tiene datos de opciones en fechas pasadas -> no aparece
  en los agregados historicos. Hacia adelante si entra.
- BAJA: soft delete conserva los datos pasados -> su contribucion historica al
  sector se mantiene. Hacia adelante deja de tener datos -> sale solo.

El z-score sectorial (ventana 60d hacia atras) re-agrega fechas pasadas pero
DRIVEN POR PRESENCIA DE DATO, no por la lista actual -> agregar/quitar de
`activos` no altera quien contribuyo a una fecha pasada (hecho historico
inmutable). `universo_cambios` da el log de auditoria.

**Salvedad (la unica):** el JOIN toma el sector ACTUAL de `activos`. Si se CAMBIA
el sector de un ticker existente (reclasificacion), eso SI regruparia su historia
bajo el nuevo sector. Por eso el `--sector` se setea una vez y se cambia rara vez
y de forma deliberada. Para alta/baja normal no aplica (el ticker nace con su
sector y no se reclasifica).

## 6. Aristas e implicancias (resumen)

| Arista | ALTA | BAJA |
|---|---|---|
| Datos & indicadores | backfill 2a -> caliente dia 1 | deja de bajar; historia queda |
| Opciones (PCR/muros/OI) | dia 1 (proximo snapshot, ya lo ve por activos) | libera la ventana del cron |
| Z-scores acciones | backfilleable -> dia 1 | se congelan; historia queda |
| Z-scores opciones | ~60d (no backfilleable) -> "sin historia" mientras acumula | se congelan |
| Sectorial | entra al agregado hacia adelante (point-in-time) | recomposicion hacia adelante |
| Bots / estrategias | candidato dia 1 (ATR/SMA validos) | guard: cerrar posiciones FT 1ro |
| ML scanner | usa global_rf (como los BT) -> caveat menor | sin efecto |
| Fundamental | balances + ratios + vs-sector dia 1 | historia queda |
| Backtests viejos | quedan "vs universo del momento" (asterisco, no se invalidan) | idem |

## 7. Fuera de alcance / a evaluar

- **ETFs**: tema a evaluar (no pendiente). Un ETF no tiene "sector" como una
  accion (SPY != tech). Si se incorporan: categoria aparte, fuera de los agregados
  sectoriales; los ETF sectoriales (XLK/XLF...) podrian ser benchmark del sector.
- **Z-scores de opciones ~60d**: residuo inherente (no backfilleable). Mostrar
  "sin historia suficiente (faltan X dias)" mientras acumula.
- **Primera alta real**: HOOD (Robinhood, Financial Services / Capital Markets),
  18/6/2026 -> universo 199->200. Validado end-to-end (precios/indicadores/
  features/z-scores/fundamentales/vs-sector, local y Railway).

## 8. Primera baja real: WBD, deslistado por compra (14/10/2026)

Paramount Skydance compro Warner Bros. Discovery en efectivo (31 USD por accion
mas un devengo diario). Ultima rueda real 2026-10-05; la barra del 2026-10-06 es
SINTETICA (O=H=L=C=30,95, volumen 0). Universo 200 -> 199.

Como se detecto: el Paso 1 de la rutina quedo PARCIAL 6 corridas seguidas con un
solo ticker pendiente ("WBD, ultimo dato 2026-10-06"). Un ticker que deja de
cotizar NO falla: devuelve vacio y queda como "pendiente" para siempre.

Lo que un deslistado deja y la baja tiene que resolver:
- **Posiciones FT que nunca cierran solas.** Los bots deciden con el ultimo cierre
  disponible del ticker (`fecha_datos`), asi que seguian viendo 30,95 todos los dias:
  ni stop ni salida, marcadas a mercado a ese precio sin fin. El guard de `remove`
  aborto con 3 abiertas (TECH_SECTOR_v1, v2 y OIEXIT_v1). Se cerraron al ULTIMO CLOSE
  de `precios_diarios` (misma convencion que ESTRATEGIA_DISCONTINUADA; el pago del
  deal difiere ~0,2%) con `motivo_salida = DELISTING` (salida ARTIFICIAL, excluible),
  registro `ft_cambios.wbd_delisting_cierre` y `ft_compute_equity.py --rebuild`.
  Recien despues, `remove` sin `--force`.
- **Opciones "zombie".** Yahoo siguio sirviendo una cadena de WBD despues del cierre
  (OI cayendo de 1,8 M a 0,4 M), con el precio congelado. Desde el 7/10 esas filas
  entraban a `opciones_pcr_plazo_diario` con `precio_fuente='snapshot'` y al PCR
  sectorial de Communication Services. La baja lo corta (el snapshot lee `activos`).
- `config.ALL_TICKERS`: se quito a mano (el fallback de ultimo nivel lo reincorporaria).

**La historia se CONSERVA** (30.111 filas en 41 tablas locales, ninguna FK hacia
`activos`: la DB dejaria borrar sin error). Motivos para no hacer hard delete:
descuadra el cash y la equity de FT; mete SESGO DE SUPERVIVENCIA en todo backtest o
grilla futura (correr solo sobre las empresas que sobrevivieron); deja sin
reproducir el entrenamiento ML v2, los `bt_hist_*` y las comparaciones del scanner;
los agregados sectoriales ya escritos de los pares darian otro numero al
recalcularse; y el crudo de opciones es irrecuperable. El espacio que se liberaria
es despreciable.

Receta para el proximo deslistado: (1) confirmar el evento y la ultima rueda real;
(2) `universo.py remove X --dry-run` para ver el guard; (3) cerrar las posiciones FT
abiertas al ultimo close con motivo DELISTING + `ft_cambios add` (tipo DATOS) +
`ft_compute_equity.py --rebuild`; (4) `remove` real; (5) sacarlo de
`config.ALL_TICKERS` y `universo.py cache`.
