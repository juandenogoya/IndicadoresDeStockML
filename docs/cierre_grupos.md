# Cierre por grupo (29/9/2026) -- EN EVALUACION

Mensaje de Telegram con el cierre de la rueda leido POR SECTOR: que grupo quedo
mas fuerte y mas debil, que tickers cruzaron un nivel y que conviene vigilar en
la proxima rueda. Idea tomada del widget `narrative_compare` de un tablero de
TradingView Remix (ver "Origen" al final), sin LLM.

- Motor puro: `src/utils/cierre_grupos.py` (stdlib, tests en `tests/test_cierre_grupos.py`)
- Script: `scripts/manual/telegram_cierre.py` (LOCAL-only; `--dry-run`, `--rueda`, `--evaluar N`)
- Doble clic: `scripts/manual/telegram_cierre.bat` (muestra la vista previa y
  pregunta antes de enviar; con parametros los pasa directo al script)
- Auditor: `scripts/manual/auditar_cierre_grupos.py` (recalcula por SQL, criterio 1)
- Rama: `feature/cierre-grupos`. **NO esta en la rutina diaria**: se suma solo si
  pasa los criterios de abajo. Si no pasa, se documenta el cierre y se duerme
  (codigo y doc en main, sin conectar, como la linea SEC XBRL).

## Que NO es

Es ESTADO, no pronostico. Mismo criterio que `docs/telegram_foto.md`: nada de lo
medido en el proyecto respalda leer un cruce de nivel como senal (salidas FT
indistinguibles del azar, senales SMC de entrada sin ventaja, alertas inusuales
sin direccion a 5 ruedas). El mensaje no usa "oportunidad" ni "deberia". **No se
mide** si "perdio el soporte" anticipa algo: no es la pregunta.

## Definiciones (fijadas antes de ver resultados)

**Rueda D** = la ultima fecha de `precios_diarios` (o `--rueda`). **Rueda
anterior** = la fecha previa de `precios_diarios` en el universo, NO un `shift`
por ticker (con un hueco, el shift por ticker compararia contra una rueda vieja).
Un ticker sin fila en D queda fuera y se informa cuantos.

**Grupos** = `activos.sector`. Solo entran al RANKING los sectores con n >= 5
(mismo umbral que el comparativo fundamental); Real Estate (3) y Utilities (1)
se muestran marcados "n chico" y no compiten.

Por grupo:

| Dato | Definicion |
|---|---|
| var dia | MEDIANA de close_D / close_anterior - 1 (mediana y no promedio: un ticker con balance arrastra el promedio) |
| var 5r | MEDIANA de close_D / close de 5 ruedas antes - 1 (ruedas del universo) |
| suben / bajan | tickers con var dia > 0 / < 0 |
| sobre SMA50 | % de tickers con close_D > sma50_D |
| veredictos | ALCISTA / NEUTRAL / BAJISTA de `veredictos_universo_diario` en D |

Mas fuerte / mas debil = mayor / menor mediana de var dia entre los sectores
rankeables, informada contra la mediana del universo.

**Cruces de nivel.** Tres tipos, cada uno por separado:

| Tipo | Arriba | Abajo | Fuente |
|---|---|---|---|
| Muro OI | close_D > resistencia de la rueda ANTERIOR | close_D < soporte de la rueda ANTERIOR | `opciones_pcr_plazo_diario`, ventana `corto` |
| Swing N5 | `bos_bull_5` o `choch_bull_5` en D | `bos_bear_5` o `choch_bear_5` en D | `features_estructura` (swings confirmados, sin futuro) |
| SMA | close cruza la SMA50 o la SMA200 (lado en D-1 distinto del lado en D) | idem | `indicadores_tecnicos` |

- El muro se toma de la rueda ANTERIOR porque el de D se calcula con el close de
  D (el soporte esta debajo del precio por construccion: comparar contra el de D
  nunca daria cruce). Si falta la fila del muro en la rueda anterior (hay huecos
  irrecuperables en opciones), ese ticker no informa cruce de muro ese dia; si
  falta para todo el universo, el mensaje lo dice.
- El swing usa las columnas que ya existen: el nivel es el ultimo swing
  confirmado conocido en la rueda anterior (ver `estructura.py`).
- SMA: misma regla de "lado" que la foto (`foto_ticker._lado`), para que los dos
  mensajes no digan cosas distintas del mismo ticker.

**Vigilar en la proxima rueda** (hechos, sin direccion):
- a menos de **1,0 %** de un nivel en D: muro OI (`soporte_dist_pct` /
  `resistencia_dist_pct` de D) o swing N5 (`dist_sl_5_pct` / `dist_sh_5_pct` de D);
- balance en las proximas **5** ruedas NYSE (`earnings_historico.announcement_date`
  union `earnings_calendar.earnings_date`; el historico permite reconstruir ruedas
  pasadas, el calendario cubre lo que el historico todavia no tiene);
- cambio de veredicto contra la rueda anterior.

Cada lista muestra hasta **8** tickers e informa el total.

## Criterios pre-registrados (29/9/2026, ANTES de correr nada)

Evaluacion sobre las **20 ruedas** de `precios_diarios` que terminan en
2026-09-28. Limitacion declarada: `veredictos_universo_diario` arranca el
2026-09-01 (14 ruedas) y `opciones_pcr_plazo_diario` tiene huecos y dias con 167
de 200 tickers; eso se informa, no se rellena.

| # | Criterio | Pasa si |
|---|---|---|
| 1 | Correccion | El auditor (SQL independiente, sin el codigo del script) reproduce el 100% de: n, var dia, var 5r, suben/bajan y % sobre SMA50 por sector, y el conjunto de tickers de cada tipo de cruce, en las 20 ruedas (tolerancia 1e-6) |
| 2 | La regla dispara con sentido | Por TIPO de cruce, la mediana diaria de tickers con cruce (arriba + abajo) esta entre 2 y 30. Un tipo que falla sale del mensaje; si fallan los 3, se duerme |
| 3 | Informa algo nuevo | Ningun sector es el "mas fuerte" en mas del 80% de las ruedas, y lo mismo para el "mas debil". Si no, el ranking es composicion del universo y no el cierre del dia |
| 4 | Cabe | En cada una de las 20 ruedas: como mucho 2 mensajes de hasta 4000 caracteres |
| 5 | Sirve | Juicio del usuario despues de leerlo 5-10 ruedas reales. Se escribe aca con la fecha |

Falla 1, 2 (los tres tipos) o 3, o el 5 da "no lo uso" -> seccion de cierre y
se duerme. Ningun umbral de esta seccion se mueve despues de ver resultados; si
algo se cambia, se escribe como cambio con su motivo.

## Resultados (29/9/2026, 20 ruedas: 2026-08-28 -> 2026-09-28)

| # | Resultado | Veredicto |
|---|---|---|
| 1 | Auditor: 220 sector-ruedas x 6 campos y 59 conjuntos de cruces, coincide el 100%. Se comprobo que el auditor PUEDE fallar: corriendo la variacion en 0,001 pp da 116 diferencias; cambiando `>` por `>=` en el lado de la SMA, 3 | PASA |
| 2 | Mediana diaria de tickers con cruce: OI 17 (19 ruedas; el 16/9 sin muros de la rueda anterior), SWING 14, SMA 20. Rangos diarios: OI 8-41, SWING 9-27, SMA 13-33 | PASA los 3 |
| 3 | Mas fuerte: el mas repetido es Energia, 30% de las ruedas (6 de 20; 7 sectores distintos fueron el mas fuerte). Mas debil: Mat. basicos, 40% (8 de 20) | PASA |
| 4 | 1 mensaje por rueda en las 20; el mas largo 2.208 caracteres | PASA |
| 5 | Pendiente: lectura del usuario en 5-10 ruedas reales | -- |

Comandos: `telegram_cierre.py --evaluar 20` (detalle y los 20 mensajes en
`reportes/cierre_grupos/20260929_eval_20/`, fuera de git) y
`auditar_cierre_grupos.py`.

**Lo que aparecio al evaluar:**
- **Hueco en `precios_diarios` el 2026-09-22** (dia habil): 60 de 200 tickers,
  igual en `indicadores_tecnicos`. El criterio de "rueda del universo" (>= la
  mitad de los tickers) lo saltea, asi que el 23/9 "dia" compara contra el 21/9.
  Se agrego un aviso al mensaje (`ruedas_salteadas`) para que no pase
  desapercibido. El hueco en si es un problema de datos aparte (CASO E de
  docs/checklist_recovery_manual.md), no de este mensaje.
- Salvedad del swing en ese hueco: `bos_*_5` de `features_estructura` se calcula
  sobre la serie de CADA ticker, asi que para los 60 tickers que si tienen el 22/9
  el nivel es el del 22 y para el resto el del 21. SMA y muros usan la rueda
  anterior del universo para todos. Solo pasa cuando hay hueco.
- Cambio de formato despues de la primera corrida (no toca ningun umbral): la
  distancia de "cerca de un nivel" se muestra con 2 decimales, porque con 1 un
  0,99% se leia "1,0%" dentro de una lista "a menos de 1,0%".

**Lo que NO dice esta evaluacion:** que el mensaje sirva para decidir algo. Los
criterios 1-4 solo dicen que es correcto, que sus reglas disparan en una
cantidad razonable y que el ranking cambia de un dia al otro. Eso lo decide el 5.

## Origen

Tablero `tvremix-dashboards-2026-09-29.json` (TradingView Remix, 8 cripto): de
sus 14 widgets se tomo solo la idea del resumen de cierre por grupo. Su
`agent_log` narraba acciones (MSFT, NVDA, LMT...) en un tablero de cripto sin
que nada lo marcara: por eso aca el texto sale armado desde Python y no de un LLM.
