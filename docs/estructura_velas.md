# Velas y estructura de mercado (SMC) -- revision medida, leakage y plan

Estado (17/9/2026): Fases 1, 2 y 2b HECHAS (sin commit). El backtest completo de las
estrategias de lectura de estructura esta corrido y pre-registrado (seccion 9.3): la
regla de SMC con N=10 dependia de ver los swings antes de tiempo, y con swings
confirmados rapido (N=5 / N=3) pasa. Con eso se decidio (seccion 9.4): alta de
FT_SMC_v3_N5 y FT_SMC_v3_N3, baja de FT_COMBO_v1 y de FT_SMC_v2. La Fase 3a (ML v3)
tambien esta corrida: con features honestas el modelo NO pasa la compuerta
pre-registrada (AUC 0,51 en 6 folds contra 0,62 con la tabla que mira al futuro), asi
que NO hay v3 y el lockbox quedo sin abrir (seccion 9.6). Particion de datos decidida
(seccion 10). Auditoria de las 3 tablas contra el OHLCV (seccion 12, 17/9/2026):
`features_velas` y `features_estructura` son correctas al 100%; los patrones de
`features_precio_accion` estan mal definidos y dos columnas inventan valores en las
costuras de cada backfill. Estado vivo: AGENDA.md.

Este documento existe porque nada de lo que sigue se deriva leyendo el codigo: el
codigo "anda", no falla, y devuelve numeros plausibles. Lo que cambia es lo que esos
numeros significan en la historia guardada frente a lo que se sabia cada dia.

| Pregunta | Seccion |
|---|---|
| Que esta roto y que tan grave es | 1, 4 |
| Los modelos ML v1/v2 aprendieron algo real? | 4.4 |
| Los resultados de FT son validos? | 4.5 |
| Las velas (envolvente, martillo, doji) estan bien? sirven? | 5 |
| El semanal (dashboard, Telegram) | 6 |
| Que cambia si se corrige, por consumidor | 8 |
| El plan, los archivos y las compuertas | 9 |
| El backtest de SMC/COMBO con entradas y salidas | 9.3 |
| Que se decidio con ese resultado (altas y bajas de FT) | 9.4 |
| El ML v3: pre-registro y resultado (no pasa) | 9.5, 9.6 |
| Sirve entrenar por sector o industria? (no) | 9.7, 9.8 |
| Como particionar los datos para la v3 | 10 |
| Reglas para cualquier feature nueva | 11 |
| Las 3 tablas estan bien calculadas contra el OHLCV? | 12 |

---

## 1. Resumen

1. **Critico -- la historia de `features_market_structure` mira 10 ruedas al futuro.**
   El swing se detecta con una ventana centrada y se registra en su propia barra, no
   el dia en que se confirma. Los modelos ML v1 (V3) y v2 se entrenaron y validaron
   sobre esa historia. Con las features guardadas discriminan con AUC 0,65; con las
   que se sabian cada dia, 0,52 (0,50 = azar). Pasa tambien en tramos que la v1 no vio.
2. **Lo que opera hoy no mira el futuro.** El scanner y los bots FT usan la ultima
   fila, calculada con datos hasta ese dia. Los resultados de FT son honestos. Lo
   contaminado: entrenamiento, validacion offline (walk-forward de la Tarea 20,
   compuerta de la v2), backtests de SMC/COMBO y la historia que muestran dashboard/MCP.
3. **Las velas estan mal definidas**: el 73% de las envolventes marcadas no envuelve y
   martillo/estrella no miran la tendencia previa. Pero ni la definicion actual ni la
   clasica anticipan retorno: corregirlas mejora lo que se describe, no el rendimiento.
4. **Semanal**: usa la misma funcion de swings (mismo problema en la ultima semana) y
   decide que semana esta completa por el reloj. No hay patrones de vela semanales en
   ningun flujo vivo.

---

## 2. Que se calcula y quien lo usa

| Modulo | Tabla | Calculo |
|---|---|---|
| `src/indicators/market_structure.py` | `features_market_structure` | 24 features (N=5 y N=10): is_sh/is_sl, estructura, dist_sh/dist_sl, dias_sh/dias_sl, impulso, BOS/CHoCH bull/bear |
| `src/indicators/precio_accion.py` | `features_precio_accion` | 32 features: anatomia de vela, 8 patrones, rolling, volumen |
| `src/indicators/market_structure_1w.py` | `features_market_structure_1w` (CONGELADA 2026-04-02) | copia del diario sobre barras semanales; hoy se usa AL VUELO |
| `src/indicators/precio_accion_1w.py` | `features_precio_accion_1w` (CONGELADA) | copia del diario; ningun flujo vivo lo usa |
| `src/indicators/estructura.py` (17/9) | `features_estructura` | las mismas 24 columnas con swings CONFIRMADOS; invariante |
| `src/indicators/velas.py` (17/9) | `features_velas` | 11 patrones con definicion clasica y contexto; invariante |

El Paso 2 (`cron_diario --step features`) recalcula AMBAS tablas diarias sobre TODA la
historia y hace upsert de todas las filas, todos los dias.

| Consumidor | Que lee | Fuente |
|---|---|---|
| Modelos ML v1 (V3) y v2 | las 24 de estructura (entrenamiento: tabla; vivo: calculo sobre 500 barras) | `trainer_v3`, `walkforward_ml.cargar_dataset`, `feature_calculator` |
| Score del scanner | EV1-EV4 (hasta +15) con estructura_10, dias_sl_10, dist_sl_10_pct, BOS/CHoCH, envolvente, martillo; bajistas BOS -12, CHoCH -12, estructura -8 | `signal_engine`, `alert_classifier` |
| FT_SMC_v1 / v2 | CHoCH/BOS bull en 12 dias, estructura_10, choch_bear_10, trailing SL desde dist_sl_10_pct | `ft_scoring.py` (tabla) |
| FT_COMBO_v1 (y filtro de SMC_v2) | `candle_score_5d`: patrones de 5 dias + BOS/CHoCH _5 | `ft_scoring.obtener_candle_score_5d` (tabla) |
| Dashboard / veredictos | estructura diaria (tabla) y semanal (al vuelo); patrones en la frase | `sintesis_data`, `dashboard_sintesis`, `clasificacion_tecnica` |
| Telegram | patrones (Msg 2) y tendencia_1w (al vuelo) | `telegram_notifier`, `mtf_context` |
| MCP | get_price_action, get_market_structure, screen_tickers, get_ticker_overview | `mcp_server/db/queries.py` (tabla) |
| Backtest historico | SMC_v1 y COMBO_v1 | `scripts/backtesting_historico/bt_data_loader.py` (tabla) |

No hay NINGUN test de `market_structure.py` ni de `precio_accion.py` (solo tests de
formato del MCP).

---

## 3. Como se midio

Tres mediciones de solo lectura sobre la DB local, 17/9/2026. Los scripts se incorporan
al repo en la Fase 1 (seccion 9) para poder re-correrlas en la compuerta de la Fase 2.

| Medicion | Muestra | Metodo |
|---|---|---|
| Estructura diaria | 60 tickers al azar (semilla 20260917) de los 122 con >= 1.390 ruedas; 780 ruedas c/u (2023-08-08 -> 2026-09-16); 46.800 filas, 45.600 maduras | Para cada rueda t: `_calcular_ticker` sobre las 500 barras hasta t (exactamente lo que ve el scanner), ultima fila = "lo que se sabia ese dia". Se compara con la fila guardada de t. |
| Velas | 148.132 velas, 200 tickers, desde 2023-06-01 | `features_precio_accion` contra OHLC de `precios_diarios` (vela previa por LAG) |
| Estructura semanal | 18.853 semanas, 200 tickers | Ultima semana cerrada calculada con historia hasta esa semana vs la misma semana con 10+ semanas posteriores |
| Impacto ML | 44.663 filas de `features_ml` con label de los mismos 60 tickers | Mismas filas y mismas 29 features no estructurales; solo cambian las 24 de estructura (guardada vs lo que se sabia). Modelo v1 = `models_v3/global/champion.joblib`; v2 = artefacto validado por sha256 |

"Exceso" = retorno forward menos el promedio del mismo dia (de la muestra o del
universo). Media por dia y IC95 sobre los dias (evita contar 200 tickers del mismo dia
como observaciones independientes). `retorno_20d` de `features_ml` esta en PORCENTAJE.

Limites: la v1 en vivo usa modelo sectorial para algunos tickers (se midio solo el
global); la muestra de estructura son tickers con historia larga (el universo "viejo").

---

## 4. Hallazgo 1 -- los swings se fechan antes de poder conocerse (CRITICO)

### 4.1 Mecanismo

`market_structure.py`, `_calcular_estructura_n`:

```python
roll_max = h_s.rolling(2 * n + 1, center=True, min_periods=n + 1).max()
is_sh = (h_s == roll_max).values
```

- `center=True`: la barra p es swing high si es la mas alta de las N anteriores Y de
  las N SIGUIENTES. Eso recien se sabe en p+N, pero `_secuencia_pivots_lookup` lo
  registra en p. Desde la fila p, `last_sh`, `dias_sh`, `dist_sh`, `estructura` y los
  cruces de BOS/CHoCH usan un nivel que ese dia no existia.
- `min_periods=n+1`: en las ultimas N barras la ventana esta incompleta y el swing se
  marca igual. En la ultima barra es swing si es la mas alta de las ultimas 11. Es
  provisional: al dia siguiente puede dejar de serlo. El docstring dice que esas barras
  "quedan sin confirmacion (NaN)" y es falso.
- El Paso 2 recalcula toda la tabla cada dia -> la historia siempre queda reescrita con
  lo que paso despues, y la fila de una fecha cambia de un dia al otro.

La misma linea esta en `market_structure_1w.py`.

### 4.2 Evidencia: la historia guardada predice el futuro

| Fila con... | Guardada: exceso 5 / 20 ruedas | Lo que se sabia ese dia |
|---|---|---|
| is_sh_10 = 1 | -4,72% [-5,02; -4,41] / -6,71% [-7,29; -6,13] | -0,07% / -0,30% |
| is_sl_10 = 1 | +4,16% [+3,82; +4,51] / +5,22% [+4,44; +5,99] | +0,14% / +0,63% |
| dias_sh_10 <= 2 | -3,55% / -5,11% | -0,02% / -0,29% |
| dias_sl_10 <= 2 | +3,23% / +4,00% | +0,11% / +0,35% |

Un swing high "guardado" es, por construccion, un maximo del que el precio despues cayo.

### 4.3 Cuanto difiere

| Columna (N=10) | Filas que difieren | Detalle |
|---|---|---|
| estructura_10 | 30,1% | |
| dias_sh_10 | 34,1% | ==0 en 3,2% de las filas guardadas y 19,9% en vivo; mediana 16 vs 5 |
| dias_sl_10 | 26,1% | ==0 en 3,3% vs 15,9%; mediana 15 vs 7 |
| dist_sl_10_pct | 25,9% | (> 0,01 pp) |
| is_sh_10 | 16,7% | 1.467 confirmados vs 9.064 marcados en su dia |
| is_sl_10 | 12,6% | 1.501 vs 7.254 |
| bos_bull_10 | 0,4% | 717 guardados / 571 vistos en su dia / 564 en ambos |
| bos_bear_10 | 0,2% | 559 / 477 / 472 |
| choch_bull_10 | 0,1% | 207 / 166 / 165 |
| choch_bear_10 | 0,1% | 215 / 177 / 177 |

N=5 da lo mismo (estructura_5 30,7%, dias_sh_5 33,4%). Un ~21% de los BOS/CHoCH de la
historia nunca aparecio en su fecha: aparece dias despues, cuando se confirma el swing.

### 4.4 Efecto en los modelos ML

AUC sobre las mismas filas (0,50 = azar):

| Tramo | Filas | v1 guardada | v1 real | v2 guardada | v2 real |
|---|---|---|---|---|---|
| 2023-08 -> 2025-01-27 (fuera del entrenamiento de la v1) | 21.771 | 0,655 | 0,520 | 0,650 | 0,518 |
| 2025-01-28 -> 2026-04-10 (la v1 entreno aca) | 17.818 | 0,635 | 0,516 | 0,657 | 0,521 |
| 2026-04-13 -> 2026-08 (fuera del entrenamiento de la v1) | 5.074 | 0,645 | 0,516 | 0,667 | 0,525 |

Decil alto de probabilidad (v1):

| Tramo | Base: acierto / ret 20d | Guardada | Real |
|---|---|---|---|
| 2023-08 -> 2025-01 | 51,2% / +1,67% | 76,3% / +6,48% | 55,7% / +2,38% |
| 2025-01 -> 2026-04 | 48,4% / +1,24% | 68,4% / +5,74% | 53,4% / +2,18% |
| 2026-04 -> 2026-08 | 47,1% / +2,06% | 68,2% / +6,83% | 56,0% / +3,84% |

- Queda una ventaja real, chica: +4 a +9 pp de acierto en el decil alto.
- Probabilidad >= 0,65 de la v1: 4% de las filas con la guardada, 18,5% con la real.
- Pasando de guardada a real en la misma fila, la probabilidad de la v1 se mueve 0,128
  de mediana (p90 0,317) y el 64,8% de las filas cambia de tramo de puntos ML.
- Correlacion con retorno_20d: dias_sh_10 +0,140 guardada / +0,028 real; dias_sl_10
  -0,128 / -0,035; is_sh_10 -0,115 / -0,015; is_sl_10 +0,100 / +0,028.

Explica lo anotado en ml_reentrenamiento.md sec. 2.3: el 38% de la importancia en
`dias_sh_10` y `dias_sl_10` son justo las columnas que traen el futuro. El modelo
aprendio "swing high hace 0 dias = techo confirmado"; en vivo eso significa "hoy hizo
maximo de 11 ruedas" y lo lee como techo.

### 4.5 Que NO esta afectado

- Las decisiones en vivo: el scanner calcula sobre las barras hasta la rueda y los bots
  FT leen la ultima fila de la tabla, que no tiene barras posteriores. Los resultados
  de FT (FT_ML_SCANNER_v1 +$8.554 en 159 operaciones al 17/9) son la unica medicion
  honesta del ML. Llevan la salvedad de ml_reentrenamiento.md sec. 8b.
- Las 29 features no estructurales de `features_ml`: no se auditaron con el test de
  invariancia (seccion 11). Es condicion de la Fase 3.

### 4.6 Consecuencias para documentos previos

- ml_reentrenamiento.md sec. 8 (walk-forward, "edge REAL, FINO y consistente",
  compuertas, label) y sec. 8c (holdout y compuerta de la v2, cortes equivalentes,
  comparacion de tramo): numeros inflados. Addendum en su seccion 10.
- La Etapa 4 (v1 vs v2 en FT) sigue siendo una medicion valida de lo que operan, pero
  los dos modelos comparten el mismo defecto de entrenamiento.
- Los backtests historicos de SMC_v1 y COMBO_v1 (bt_hist) usaron la historia guardada.

---

## 5. Hallazgos de velas (`precio_accion.py`)

### 5.1 Envolvente: no verifica que envuelva

```python
df["patron_engulfing_bull"] = ((body_abs > prev_body) & (es_alc == 1) & (prev_alc == 0))
```

Solo compara tamano de cuerpo y color.

- Envolvente alcista marcada en el 12,69% de las velas (16.838; 1 de cada 8).
- Cumple la definicion clasica (abre <= cierre previo y cierra >= apertura previa, vela
  previa roja): 27,4%.
- Abre POR ENCIMA del cierre previo (gap, no envuelve): 52,6%. Cierra por debajo de la
  apertura previa: 19,0%.
- Vela previa plana (close == open, no es roja) contada como roja: 3,1%.
- Envolvente bajista: 16.015 marcadas, 32,1% clasica.

### 5.2 Martillo y estrella fugaz: forma incompleta y sin contexto

```python
patron_hammer        = (lower_s > 0.60) & (body_ratio < 0.30) & (es_alc == 1)
patron_shooting_star = (upper_s > 0.60) & (body_ratio < 0.30) & (es_alc == 0)
```

- Sin tope de sombra opuesta: 30,4% de los martillos tiene sombra superior <= 10% del
  rango (forma clasica); 17,7% la tiene > 25% (spinning top).
- Exige color: excluye 5.181 martillos rojos (mas que los 4.991 verdes que cuenta) y
  5.090 estrellas verdes.
- Sin contexto de tendencia: el 46,5% de los "martillos" esta en el tercio superior del
  rango de 20 ruedas (es un hanging man, lectura bajista); solo 46,1% viene de caida.
  El 34,5% de las estrellas esta en el tercio inferior (inverted hammer).
- Doji solapado: 2.384 de 7.017 doji tambien salen como martillo o estrella.

### 5.3 Otros

- Frecuencias: doji 5,29%, martillo 3,76%, estrella 4,00%, marubozu 6,54% (sin
  direccion), inside bar 11,43%, outside bar 8,69%.
- close == open cuenta como vela bajista (`es_alcista = close > open`): 0,86%.

### 5.4 Valor predictivo

Exceso vs universo del dia, media por dia, IC95:

| Patron | 5 ruedas | 20 ruedas |
|---|---|---|
| Envolvente alcista actual (n=16.838) | +0,01% [-0,12; +0,15] | -0,12% [-0,42; +0,18] |
| Envolvente alcista clasica (n=4.609) | -0,02% [-0,25; +0,21] | -0,10% [-0,58; +0,37] |
| Envolvente alcista clasica tras caida (n=2.531) | -0,13% [-0,42; +0,15] | -0,35% [-0,93; +0,22] |
| Envolvente bajista actual (n=16.015) | +0,07% [-0,05; +0,19] | +0,21% [-0,04; +0,45] |
| Envolvente bajista clasica tras suba (n=3.135) | +0,15% [-0,14; +0,44] | +0,06% [-0,65; +0,77] |
| Martillo actual (n=4.991) | -0,05% [-0,29; +0,19] | -0,00% [-0,48; +0,47] |
| Martillo clasico, cualquier color, tras caida (n=1.508) | -0,04% [-0,42; +0,35] | -0,05% [-0,82; +0,72] |
| Hanging man (misma forma) tras suba (n=1.736) | -0,17% [-0,54; +0,20] | -0,30% [-1,04; +0,45] |
| Estrella fugaz actual (n=5.310) | +0,13% [-0,09; +0,34] | -0,48% [-0,89; -0,08] |
| Estrella fugaz clasica tras suba (n=1.812) | +0,03% [-0,29; +0,35] | +0,35% [-0,44; +1,13] |
| Doji (n=7.017) | +0,02% [-0,14; +0,18] | +0,23% [-0,12; +0,57] |

Ninguno se distingue de cero salvo la estrella actual a 20 ruedas, esperable por azar
entre 22 pruebas. Con lo que se sabia cada dia, BOS y CHoCH tampoco muestran ventaja
(bos_bull_10: -0,61% [-1,11; -0,10] a 5 ruedas; choch_bull_10: +0,29% [-0,52; +1,10]).

Conclusion: corregir las velas es un tema de DESCRIBIR bien (dashboard, MCP, Telegram,
infografias), no de rendimiento. Ninguna definicion debe venderse como senal.

---

## 6. Semanal

- `mtf_context` (tendencia_1w del Telegram) y `sintesis_data._semanal_bundle` (SMC
  semanal del dashboard) resamplean W-FRI y llaman `_calcular_ticker_1w`: misma linea
  de swings. La estructura_10 de la ultima semana cerrada difiere de la confirmada el
  27,6% de las veces:

  | Se ve esa semana \ queda confirmada | -1 | 0 | +1 |
  |---|---|---|---|
  | -1 | 69,0% | 30,5% | 0,5% |
  | 0 | 17,0% | 69,1% | 13,9% |
  | +1 | 0,0% | 22,9% | 77,1% |

  is_sh_10 semanal: 4.226 marcados en su semana vs 593 confirmados. La etiqueta cambia
  semana a semana el 5,3% de las veces (confirmada: 2,8%).
- La semana en curso se excluye por RELOJ (`src/data/resample_weekly.py`,
  `pd.Timestamp.today()`), no por dato. Si la rutina corre el viernes a la noche con el
  dato del viernes (13 de ~70 ruedas desde mayo), la semana recien cerrada queda afuera
  y el semanal atrasa una semana. Mismo patron en la copia de `src/utils/weekly_tf.py`.
- No hay patrones de vela semanales vivos: `precio_accion_1w.py` y su tabla estan
  congelados y nadie los consume fuera de `scripts/legacy_1w/`.

---

## 7. Deuda y detalles menores

- Diario y semanal duplicados linea por linea; cero tests.
- Con estructura = 0, una ruptura cuenta como BOS en las dos direcciones; no se exige
  alternancia swing high / swing low; highs iguales generan dos swings.
- `ft_scoring.obtener_features_hoy` busca CHoCH/BOS con `CURRENT_DATE - 12 dias`
  (reloj), no desde la rueda de datos.
- El trailing SL de SMC reconstruye el swing low como `close / (1 + dist_sl/100)`: da
  el nivel exacto pero lo esconde, y con swings provisionales puede subir el stop sobre
  un swing que despues desaparece.

---

## 8. Impacto de corregir, por consumidor

| Consumidor | Que cambia |
|---|---|
| Modelos ML v1/v2 | Hay que REENTRENAR con swings confirmados. La validacion offline va a bajar (la de hoy esta inflada). No conviene cambiar el calculo en vivo de v1/v2 sin reentrenar: aprendieron el significado viejo. |
| Score del scanner | Cambian niveles de alerta de v1 y v2 por igual -> sin grupo de control; va a `ft_cambios`. |
| FT_SMC_v1 / v2 | Entradas mas tardias y sin eventos que aparecen despues; sus backtests de diseno no valen. Al 17/9: v1 +$2.188 (41 ops), v2 -$6.622 (27 ops). |
| FT_COMBO_v1 | Envolventes de ~12,7% a ~3,5% de las velas; `candle_score_5d` se mueve mucho menos. Al 17/9: -$1.048 (452 ops). |
| Dashboard, veredictos, Telegram, MCP, infografias | Etiquetas correctas, menos giros falsos. No toca trading. |
| Backtest historico SMC/COMBO | Se puede re-correr con historia honesta. |

---

## 9. Plan de implementacion

Restricciones:
- La v2 esta congelada por la Etapa 4. Nada que cambie las ENTRADAS EN VIVO de v1/v2 o
  del score del scanner antes de cerrarla. Lo nuevo va EN PARALELO.
- Todo cambio que toque decisiones de FT se registra en `ft_cambios` antes de desplegar.
- Acordado con el usuario (17/9/2026): el FT sigue corriendo (es la medicion honesta),
  no se invierte mas en v1/v2 y la Fase 3 no espera al cierre de la Etapa 4.

### 9.1 Hecho (17/9/2026)

**Fase 1** -- `src/indicators/estructura.py` y `src/indicators/velas.py` (puros),
`tests/test_estructura.py` y `tests/test_velas.py` (33 tests; el de invariancia sobre el
modulo viejo falla en 640 de 4.320 celdas, sobre el nuevo en 0),
`scripts/ml/medir_leakage_estructura.py` (secciones A-G) y
`scripts/ml/medir_valor_velas.py`. Docstrings de market_structure.py y _1w corregidos.

**Fase 2** --
- `scripts/compute_estructura_velas.py` -> tablas `features_estructura` y
  `features_velas` (PK ticker, fecha; tipos generados desde las listas de columnas de los
  modulos). Carga inicial 223.330 filas cada una en 35 s. Verificado: lo guardado es
  igual celda por celda al modulo. Incremental: escribe desde la primera rueda que falta
  por ticker con 10 ruedas de solape (6 s); un ticker sin filas se escribe entero.
- Paso 2c en `cron_diario` (tambien en el modo legacy). Si falla avisa y el Paso 2
  sigue: todavia no es insumo. `estado_pipeline` las registra como no criticas.
- `splits.py corregir` las recalcula completas (unico caso en que la historia cambia).
- Semanal: `mtf_context` y `dashboard/sintesis_data` usan `estructura.py`; la semana
  completa se decide por dato (`weekly_tf.excluir_semana_incompleta`, que tambien usa
  `resample_weekly`). Con swings confirmados hace falta mas historia semanal: con 540
  dias el Telegram coincidia con la historia completa en 121/200 tickers, con 730 en
  180/200, con 1.095 en 200/200 -> `mtf_context._DIAS_HISTORIA = 1100`.
- NO cambiado (Fase 4): estructura diaria del dashboard, score del scanner, FT, MCP.
  `volatilidad_mtf.py` (perfiles) sigue excluyendo la semana por reloj: fuera de alcance.

Auditoria estatica de las 29 features no estructurales (technical, scoring,
sector_features, feature_store, feature_calculator): el unico `shift(-n)` es el label.

### 9.2 Compuerta de la Fase 2: valor con historia sin futuro

`medir_leakage_estructura.py --solo FG` (universo, 2021-06-01 -> 2026-08-18, 205.990
filas). Exceso vs universo del dia, media por dia, IC95:

| Evento (swings confirmados) | n | 5 ruedas | 20 ruedas |
|---|---|---|---|
| bos_bull_10 | 3.789 | -0,02% [-0,25; +0,20] | -0,26% [-0,71; +0,20] |
| choch_bull_10 | 1.380 | +0,07% [-0,38; +0,52] | +0,07% [-0,68; +0,82] |
| bos_bear_10 | 2.967 | +0,08% [-0,19; +0,34] | -0,11% [-0,67; +0,45] |
| choch_bear_10 | 1.415 | -0,26% [-0,62; +0,10] | -0,48% [-1,18; +0,22] |
| estructura_10 = +1 | 68.498 | -0,14% [-0,21; -0,07] | -0,38% [-0,51; -0,25] |
| estructura_10 = -1 | 56.311 | +0,07% [+0,01; +0,14] | +0,17% [+0,05; +0,30] |
| bos_bull_5 | 5.508 | +0,00% [-0,19; +0,20] | -0,36% [-0,78; +0,07] |
| choch_bull_5 | 2.090 | -0,41% [-0,72; -0,10] | -0,45% [-1,05; +0,14] |
| choch_bear_5 | 2.091 | -0,00% [-0,30; +0,29] | -0,58% [-1,11; -0,05] |

Regla de entrada de FT_SMC (condiciones obligatorias de
`ft_scoring.calcular_score_estructura`: CHoCH/BOS alcista en 9 ruedas, estructura_10 >= 0,
sin CHoCH bajista, vela alcista, dist_sl_10 entre 1% y 8%):

| Historia | Senales | Por rueda | 5 ruedas | 20 ruedas |
|---|---|---|---|---|
| Guardada (vieja) | 2.878 | 2,2 | +0,16% [-0,04; +0,37] | -0,24% [-0,62; +0,14] |
| Sin futuro (nueva) | 2.434 | 1,9 | -0,23% [-0,45; -0,02] | -0,39% [-0,82; +0,05] |

Lectura:
- Ningun evento alcista de estructura tiene ventaja; la estructura alcista rinde por
  DEBAJO del universo (reversion a la media a 20 ruedas, no continuacion).
- La regla de FT_SMC no tiene ventaja ni con la historia vieja (su backtest de diseno
  estaba inflado en los eventos, no en el resultado a 20 ruedas) y con la nueva queda
  por debajo del universo a 5 ruedas.
- LIMITE DE ESTA MEDICION (corregido 17/9/2026, observacion del usuario): SMC y COMBO
  no buscan pronosticar con la senal sola; leen la estructura para decidir entradas Y
  salidas (trailing SL bajo el ultimo swing low, CHoCH bajista, estructura rota). Esto
  midio solo la entrada a plazo fijo y en exceso sobre el universo, no la estrategia
  como opera (salidas, retorno absoluto). No alcanza para decidir apagar FT_SMC.
- Lo que si dice: la entrada sola no elige acciones mejores que el resto. Si la
  estrategia gana, tiene que ser por las salidas, y eso solo pasa si una vez que el
  precio va a favor tiende a seguir (tendencia).
- COMPUERTA CORRECTA (pendiente): el backtest historico de SMC_v1 y COMBO_v1
  (`scripts/backtesting_historico/`, que ya simula trailing SL, CHoCH bajista, estructura
  rota y time stop) corrido con la historia vieja y con features_estructura/velas, en
  retorno absoluto y contra el universo. Al 17/9 en FT: SMC_v1 +$2.188 en 41 ops,
  SMC_v2 -$6.622 en 27, COMBO_v1 -$1.048 en 452.

### 9.3 Backtest de las estrategias de lectura de estructura (PRE-REGISTRO, 17/9/2026)

Fijado ANTES de correr. Motor: `scripts/backtesting_historico/ft_backtesting_runner.py`
(`--historia vieja|nueva`, `--ventana N`). Verificado: la variante original reproduce el
backtest guardado de SMC_v1 2025-06-01 -> 2025-12-31 (+8,80%, 48 ops).

- Periodo unico: 2021-09-01 -> 2026-09-16 (SMA200 valida en todo el universo de entonces).
- Variantes, todas se reportan:
  - SMC_v1: historia vieja N=10 | nueva N=10 | nueva N=5 | nueva N=3
  - COMBO_v1: vieja | nueva
  - TECH_SECTOR_v1 (sin estructura ni velas): referencia de COMBO
- Metricas: retorno total, drawdown maximo, operaciones, retorno medio por operacion,
  retorno por anio y contra el universo equal-weight, exposicion media.
- Lectura:
  - vieja vs nueva N=10 = cuanto del resultado venia de ver swings antes de tiempo;
  - una variante "funciona" si gana plata en el total Y le gana al universo ajustado
    por exposicion (retorno / exposicion media) en al menos 4 de los 6 tramos anuales;
  - N=5 y N=3 solo informan si confirmar antes cambia la conclusion. No se elige el
    mejor N para desplegar: cualquier cambio de regla se valida en FT.

**Resultados** (dry-run, log en reportes/estructura_velas/bt_estrategias_20260917.log;
universo equal-weight +86,21% en el periodo):

| Variante | Retorno | Max DD | Ops | Ret/op | Expo | Anios que le gana al universo (ajustado por expo) | Veredicto |
|---|---|---|---|---|---|---|---|
| SMC_v1 vieja N=10 (como se diseno) | +61,13% | -7,47% | 328 | +1,40% | 41% | 4/6 | pasa, pero con historia que mira al futuro |
| SMC_v1 nueva N=10 | +13,51% | -6,66% | 341 | +0,39% | 40% | 2/6 | NO pasa |
| SMC_v1 nueva N=5 | +59,76% | -10,29% | 446 | +1,17% | 46% | 4/6 | pasa |
| SMC_v1 nueva N=3 | +58,22% | -9,42% | 521 | +0,94% | 48% | 4/6 | pasa |
| COMBO_v1 vieja | +22,28% | -8,29% | 4.808 | +0,33% | 40% | 2/6 | NO pasa |
| COMBO_v1 nueva | +19,87% | -8,15% | 4.856 | +0,30% | 40% | 2/6 | NO pasa |
| TECH_SECTOR_v1 (referencia) | +24,43% | -8,35% | 4.945 | +0,33% | 40% | 2/6 | NO pasa |

Retorno anual / exposicion (universo: 2021 -0,01 | 2022 -17,66 | 2023 +28,50 | 2024 +16,11 |
2025 +32,75 | 2026 +14,18):

| Variante | 2021* | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|
| SMC vieja N=10 | +19,45 | +22,56 | +13,08 | +36,38 | +20,68 | +15,16 |
| SMC nueva N=10 | -3,36 | -4,78 | +6,04 | +16,28 | +12,00 | +2,10 |
| SMC nueva N=5 | -7,97 | +0,91 | +25,03 | +31,23 | +42,75 | +15,57 |
| SMC nueva N=3 | +8,57 | +23,39 | +37,15 | -1,51 | +35,23 | -0,10 |
| COMBO vieja | +4,55 | -19,58 | +12,44 | +19,98 | +17,12 | +9,58 |
| COMBO nueva | +4,46 | -20,88 | +16,73 | +19,40 | +15,29 | +4,82 |
| TECH_SECTOR | +4,42 | -22,07 | +16,73 | +20,00 | +18,16 | +10,38 |

(*) tramos parciales: 2021 desde septiembre, 2026 hasta el 16/9.

Lectura:
1. **SMC tal como se diseno (N=10) dependia de ver swings antes de tiempo**: +61% con la
   historia vieja, +13,5% con la misma regla y lo que se sabia cada dia. No pasa.
2. **Confirmar antes cambia la conclusion**: con N=5 y N=3 la estrategia gana ~+58-60% y
   pasa la regla. Las dos coinciden en el total pero no en los anios (N=3 flojo en 2024 y
   2026, N=5 en 2021): es consistente con una lectura de estructura que sirve si reacciona
   rapido, pero no alcanza para elegir N. Segun lo pre-registrado, no se despliega sin FT.
3. **COMBO**: las velas no agregan. COMBO vieja +22,3%, nueva +19,9%, TECH_SECTOR sin velas
   +24,4%. La estructura pesa poco en COMBO: la historia vieja casi no lo movia.
4. Ninguna incluye costos. COMBO/TECH_SECTOR hacen ~960 operaciones por anio (SMC ~70-100):
   con costos quedan peor.
5. FT_SMC_v1 en vivo no es ninguna de estas: lee la ultima fila del modulo viejo, con
   swings provisionales (se marcan apenas el precio se aleja, pueden desaparecer). Esta
   mas cerca de una confirmacion rapida que del backtest con N=10 confirmado.

### 9.4 Fase 2b -- que se hizo con el resultado del backtest (17/9/2026)

Decidido con el usuario despues de leer 9.3. Tres decisiones.

**(a) SMC: alta de dos estrategias FT con estructura confirmada.**
`FT_SMC_v3_N5` (id 12) y `FT_SMC_v3_N3` (id 13), $100.000 cada una, desde la rueda
2026-09-16. La MISMA regla de FT_SMC_v1 -- el score se importa de `ft_scoring`, no se
reimplementa -- leyendo `features_estructura` + `features_velas` via
`scripts/forward_testing/ft_scoring_estructura.py`. Las dos ventanas corren en
paralelo porque el backtest no distingue entre ellas y el pre-registro dice que el N
no se elige mirando el backtest. FT_SMC_v1 queda como control, sin tocar.
Ficha: `docs/forward_testing/estrategias/SMC_v3.md`.
Registro: `ft_cambios.smc_v3_estructura_confirmada` (PARAMETRO, 12 y 13, rueda 16/9).

Para eso se agrego N=3 a la tabla: `estructura.VENTANAS_TABLA = (3, 5, 10)` (36
columnas de estructura; `VENTANAS` sigue en (5, 10), que es el set con paridad de
nombres con `FEATURE_COLS_MS`). `compute_estructura_velas.py --crear` ahora agrega las
columnas que falten con `ADD COLUMN IF NOT EXISTS`: una ventana nueva no obliga a
rehacer la tabla. Recarga completa: 223.330 filas en 51 s, verificada celda por celda
contra el modulo (0 diferencias en AAPL/KLAC/EQIX) y con el test de invariancia de N=3.

**(b) COMBO: baja.** `FT_COMBO_v1` discontinuada con la rueda 2026-09-16. Su unico
aporte sobre TECH_SECTOR_v1 es el `candle_score_5d`, y no aporta ni en 5 anios de
backtest (+19,9% contra +24,4% sin velas) ni en los 98 dias de FT (equity -0,09%
contra +2,29% del control; por operacion -0,123%, IC95 [-0,919%; +0,674%], no
distinguible de cero). Debajo, la seccion 5.4: ningun patron de vela tiene exceso
distinguible de cero. 25 posiciones liquidadas al cierre del 16/9 (+950 USD),
`activa = FALSE`, bot fuera de la rutina, codigo conservado.
Cierre: `docs/forward_testing/estrategias/COMBO_v1.md`.
Registro: `ft_cambios.combo_v1_discontinuada` (INFRA, estrategia 5, rueda 16/9).

**(c) SMC_v2: baja.** Peor equity de las once (-5,46% contra +2,36% de SMC_v1 en los
mismos dias). Sus tres agregados se apoyan en velas agregadas y estructura sin
confirmar; la salida por agotamiento (tres condiciones AND) no disparo ni una vez en
94 ruedas, y sin time stop la unica salida propia que actuo fue el trailing SL: 15
operaciones a -4,55% de media. Nunca tuvo backtest. 5 posiciones liquidadas (+1.161
USD). Cierre: `docs/forward_testing/estrategias/SMC_v2.md`.
Registro: `ft_cambios.smc_v2_discontinuada` (INFRA, estrategia 7, rueda 16/9).

Lo que NO se toco: FT_SMC_v1 y el resto de las estrategias, el score del scanner, los
modelos v1/v2, la estructura diaria del dashboard, el MCP. Sigue valiendo la
restriccion de la Etapa 4.

### 9.5 Fase 3a -- ML v3: pre-registro (escrito ANTES de correr, 17/9/2026)

**Paso 0 hecho -- auditoria de invariancia de las otras 29 features.**
`scripts/ml/auditar_invariancia_features.py` (12 tickers x 8 cortes, log en
`reportes/ml_v3/auditoria_invariancia_20260917.log`). Mide dos cosas distintas:
- **invariancia**: `calcular(datos[:t+1]).iloc[-1]` vs `calcular(datos).iloc[t]`. Las
  18 features locales (indicadores + scoring; las 11 sectoriales son z-scores
  transversales y no se recalculan por ticker) dan **diferencia maxima 0,00e+00**:
  ninguna mira al futuro. El unico `shift(-n)` del pipeline sigue siendo el label.
- **skew de ventana**: el dataset sale de `indicadores_tecnicos` calculada sobre la
  historia COMPLETA; el scanner en vivo recalcula desde las ULTIMAS 500 barras
  (`data_manager.preparar_ticker` -> `cargar_precios_db(ultimas_n=500)`). Diferencia
  maxima **0,0000**: con 500 barras de arranque los indicadores recursivos
  (RSI/ATR/ADX/MACD) ya convergieron. No hay train/serve skew por ventana.

**Dataset**: `features_ml` JOIN `features_estructura` (swings confirmados), filas con
label. 1.463 ruedas, 175.425 filas, 196 tickers, 2020-10-15 -> 2026-08-13.
Loader: `walkforward_ml.cargar_dataset(estructura="nueva")`; `"vieja"` sigue siendo el
default para que la v2 se pueda reproducir.

**Particion (fechas CONGELADAS aca y en la metadata del modelo)**, segun la decision
de la seccion 10:

| Tramo | Fechas | Ruedas | Filas | Base |
|---|---|---|---|---|
| Desarrollo (80%) | 2020-10-15 -> 2025-06-12 | 1.170 | 117.998 | 48,65% |
| Embargo | 2025-06-13 -> 2025-07-14 | 20 | -- | -- |
| **Lockbox (20%)** | **2025-07-15 -> 2026-08-13** | **273** | **53.507** | **50,37%** |

El lockbox se mira UNA sola vez, con la configuracion ya congelada.

**Walk-forward purgado dentro del desarrollo**: holdout 126 ruedas, embargo 20, train
expansivo, primer holdout desde la rueda 400 -> **6 folds**, el primero
2022-05-18 -> 2022-11-15 y el ultimo 2024-11-19 -> 2025-05-22.

**Configuracion congelada** (decision de la Tarea 20, no se reabre): RF global
(300 arboles, profundidad 8, `min_samples_leaf` adaptativo, `class_weight=balanced`) +
calibracion isotonica cv=3, label absoluto (`retorno_20d > +1%`). Sin concurso de
algoritmos.

**Dos brazos**: A = 53 features (29 + 24 de estructura confirmada);
B = 29 features (ablacion, sin estructura).

**Compuertas** (fijadas antes de ver resultados):
1. **Walk-forward**: AUC media >= 0,54 y AUC > 0,52 en al menos **5 de los 6** folds.
   (Es mas estricto que el "7 de 10" que se menciono cuando el numero de folds todavia
   no estaba calculado: con 6 folds cada uno pesa mas.)
2. **Ablacion**: el brazo A entra solo si su AUC media supera a la de B por >= 0,005 Y
   le gana en al menos 4 de los 6 folds. Si no, **el modelo v3 es el de 29 features** y
   las 24 de estructura quedan afuera: con la historia honesta no aportan.
3. **Lockbox** (una corrida, brazo y config ya elegidos): AUC >= 0,53; tasa de acierto
   del decil alto >= base + 5 pp; exceso de retorno a 20 ruedas del decil alto positivo
   con IC95 que excluya el cero.
4. Si no pasa 1 o 3: **no hay v3 desplegable** y queda documentado por que. No se
   reabren los parametros para hacerla pasar.

**Comparacion con la v2**: su AUC de holdout (0,641) NO es comparable -- se midio con
las features que miran al futuro. La referencia honesta es la que se mide aca.

**Despues de pasar**: reentrenar con desarrollo + lockbox para el artefacto
(`models_ml_v3/`, fuera de git; `metadata.json` en git con sha256, orden de features y
las fechas de corte), calcular cortes equivalentes a los de v1/v2 y recien entonces la
Fase 3b (despliegue en paralelo como FT_ML_SCANNER_v3, registrado en `ft_cambios`).

### 9.6 Fase 3a -- RESULTADO: la v3 NO PASA (17/9/2026)

`scripts/ml/entrenar_ml_v3.py --solo-wf`
(log `reportes/ml_v3/wf_v3_20260917.log`, folds en `reportes/ml_v3/wf_folds_nueva.csv`).

AUC por fold, los dos brazos, con estructura CONFIRMADA:

| Fold | Holdout | n train | Base | 53 features | 29 features (ablacion) |
|---|---|---|---|---|---|
| 1 | 2022-05-18 -> 2022-11-15 | 16.300 | 45,5% | 0,5506 | 0,5245 |
| 2 | 2022-11-16 -> 2023-05-18 | 31.563 | 47,2% | 0,5654 | 0,5585 |
| 3 | 2023-05-19 -> 2023-11-16 | 46.935 | 47,9% | 0,5221 | 0,5212 |
| 4 | 2023-11-17 -> 2024-05-20 | 62.307 | 54,8% | 0,5054 | 0,5013 |
| 5 | 2024-05-21 -> 2024-11-18 | 77.679 | 51,3% | **0,4683** | **0,4899** |
| 6 | 2024-11-19 -> 2025-05-22 | 93.118 | 48,7% | **0,4682** | **0,4638** |
| | **media** | | | **0,5133** +/- 0,0408 | **0,5099** +/- 0,0326 |

- **Compuerta 1: NO PASA.** AUC media 0,5099 (brazo elegido) contra el minimo de 0,54;
  3 de 6 folds por encima de 0,52 contra el minimo de 5. Los dos folds mas recientes,
  que son los que tienen mas datos de entrenamiento, quedan **por debajo de 0,50**: el
  modelo ordena al reves.
- **Compuerta 2 (ablacion): las 24 features de estructura NO entran.** Delta de AUC
  media +0,0035, debajo del minimo de +0,005 (aunque le gana en 5 de 6 folds). Con la
  historia honesta, la estructura de mercado aporta ~3 milesimas de AUC.
- **El lockbox NO se abrio.** Es la regla del pre-registro: si la compuerta 1 no pasa,
  no se gasta la unica evaluacion del ultimo 20%. Sigue intacto para un intento futuro
  con otra hipotesis, no con otros umbrales de esta.

**Control de diagnostico** (`--estructura vieja`, log
`reportes/ml_v3/wf_control_vieja_20260917.log`): los MISMOS 6 folds, el mismo codigo,
la misma config, cambiando solo la tabla de estructura por la que mira al futuro:

| Brazo | AUC media | Folds AUC > 0,52 | Lift decil | Exceso 20d |
|---|---|---|---|---|
| 53 features, estructura VIEJA (mira al futuro) | **0,6189** +/- 0,0158 | 6/6 | 1,35 | +4,01 |
| 53 features, estructura CONFIRMADA | 0,5133 +/- 0,0408 | 3/6 | 1,05 | +0,66 |
| 29 features (sin estructura) | 0,5099 +/- 0,0326 | 3/6 | 1,07 | +0,87 |

El brazo de 29 features da exactamente el mismo numero en las dos corridas (0,5099),
como tiene que ser: esas features no cambian. Eso valida el montaje -- el 0,51 no es un
bug del walk-forward.

**Conclusion.** Los ~11 puntos de AUC que tenia el modelo salian del look-ahead, no de
la estructura de mercado. Con features que solo usan lo que se sabia cada dia, la
configuracion congelada de la Tarea 20 no tiene ventaja medible sobre 6 ventanas
independientes: **no hay modelo v3 para desplegar y la Fase 3b no se abre.**

Lo que esto NO dice:
- no dice que las features de estructura esten mal calculadas (pasan invariancia);
  dice que, bien calculadas, casi no informan sobre el retorno a 20 ruedas;
- no dice que el problema sea la cantidad de datos: los folds con mas entrenamiento son
  los peores;
- no dice que las 29 features restantes esten contaminadas: la auditoria del paso 0 da
  invariancia exacta y skew de ventana cero;
- no invalida `FT_ML_SCANNER_v1/v2` como experimento EN VIVO: los bots leen la ultima
  fila, sin futuro. Su resultado en FT es la medicion honesta y sigue corriendo. Lo que
  queda sin respaldo es el "AUC 0,65" con el que se justificaron.

Que queda abierto (para pre-registrar aparte, no para retocar esta corrida):
1. el label: `retorno_20d > +1%` absoluto depende del regimen (sec. 2.4 de
   ml_reentrenamiento.md). Un label relativo al universo del dia es otra hipotesis;
2. el horizonte: 20 ruedas puede ser largo para features diarias;
3. la pregunta de fondo: si la ventaja no esta en predecir el retorno a plazo fijo, el
   camino son las estrategias de REGLAS con salidas (que es lo que si paso el backtest,
   sec. 9.3), no un clasificador mejor.

---

### 9.7 Es distinto por sector? (PRE-REGISTRO del paso 1, 17/9/2026)

Pregunta del usuario despues de leer 9.6: todo lo medido es el modelo GLOBAL; entrenar
por sector o por industria puede dar otra cosa.

**Por que la pregunta es legitima y no estaba cerrada.** `docs/ml_reentrenamiento.md`
sec. 2.5 declara los modelos sectoriales "rechazados por evidencia" con un head-to-head
donde el global le gana al sectorial en 6 de 6 sectores. **Ese head-to-head se midio con
`features_market_structure`**, y el sesgo no es neutral: con mas filas agrupadas, el
modelo global explota MEJOR la fuga que un sectorial chico, asi que la comparacion
favorecia al global por una razon que no era su habilidad. Queda reabierta. Ademas la
sec. 2.6 midio que la confiabilidad de la probabilidad del global VARIA por sector
(spread +7,3 en Financial Services, ~0 en Healthcare, -2,3 invertido en Industrials),
que es justo la hipotesis de heterogeneidad -- aunque con una sola ventana, n chico y
las features viejas.

**Por industria NO se prueba.** 72 industrias en el universo: solo 13 tienen 5 o mas
tickers y 31 tienen UNO. La mas grande (Auto Manufacturers) tiene 11. No hay muestra
por industria, y 72 modelos sobre los mismos datos garantizan que dos o tres "funcionen"
por azar. Por sector si: 9 sectores con filas etiquetadas, de 11.480 (Healthcare, 16
tickers) a 34.714 (Technology, 42). Real Estate (3 tickers) y Utilities (1) no tienen
filas en el dataset.

**Paso 1 -- screen de heterogeneidad (antes de entrenar nada por sector).** Sobre las
predicciones FUERA DE MUESTRA del global honesto en los 6 folds de 9.6
(`reportes/ml_v3/wf_oos_nueva.parquet`, las dos ramas de features), calcular por sector:
AUC por fold, AUC media entre folds con IC95, lift del decil alto y exceso a 20 ruedas.
La unidad de independencia es el FOLD (no la fila): dentro de un fold las filas comparten
el mismo mercado, asi que el IC95 se calcula sobre las 6 mediciones por fold.

**Regla de lectura, fijada ANTES de mirar** (si no se cumple, se frena y se documenta):
un sector califica para el paso 2 si, en cualquiera de las dos ramas, cumple las TRES:
1. AUC media entre folds >= 0,54;
2. IC95 de esa media excluye 0,50;
3. AUC > 0,50 en al menos 5 de los 6 folds.

Con 9 sectores x 2 ramas = 18 comparaciones, la exigencia de consistencia entre folds
(condicion 3) es el control de falsos positivos: un sector que "gana" en un fold y
pierde en los otros es ruido, no heterogeneidad.

**Paso 2 (solo si algun sector califica)**: modelos sectoriales sobre los MISMOS 6 folds,
head-to-head contra el global evaluado en las mismas filas de ese sector. Un sector gana
si supera al global-en-sector por >= 0,01 de AUC en al menos 5 de los 6 folds Y su AUC
media es >= 0,54. El lockbox sigue sin abrirse.

### 9.8 Paso 1 -- RESULTADO: no hay heterogeneidad sectorial. El paso 2 NO se corre

`scripts/ml/screen_sectorial_v3.py` sobre 193.068 predicciones fuera de muestra
(96.534 filas x 2 ramas, 6 folds; log `reportes/ml_v3/screen_sectorial_20260917.log`,
detalle en `reportes/ml_v3/screen_sectorial.json`).

Rama de 29 features (la que eligio la ablacion); el IC95 es sobre las 6 mediciones por
fold, que es la unidad de independencia:

| Sector | Filas | AUC media | Desvio entre folds | IC95 | Folds > 0,50 | Lift decil | Exceso 20d |
|---|---|---|---|---|---|---|---|
| Basic Materials | 8.115 | 0,5446 | 0,066 | [0,476; 0,614] | 5/6 | 1,25 | +3,10 |
| Energy | 7.146 | 0,5435 | 0,096 | [0,443; 0,644] | 5/6 | 1,14 | +0,98 |
| Consumer Defensive | 8.487 | 0,5293 | 0,029 | [0,498; 0,560] | 5/6 | 1,21 | +1,67 |
| Communication Services | 7.089 | 0,5222 | 0,058 | [0,462; 0,583] | 4/6 | 1,27 | +3,08 |
| Technology | 19.204 | 0,5194 | 0,059 | [0,458; 0,581] | 4/6 | 1,10 | +1,70 |
| Consumer Cyclical | 17.787 | 0,5122 | 0,018 | [0,493; 0,532] | 4/6 | 1,13 | +1,46 |
| Financial Services | 13.259 | 0,5065 | 0,074 | [0,429; 0,585] | 3/6 | 0,98 | +0,39 |
| Healthcare | 5.805 | 0,4926 | 0,069 | [0,420; 0,565] | 3/6 | 1,03 | -1,14 |
| Industrials | 9.642 | 0,4761 | 0,053 | [0,420; 0,532] | **1/6** | 1,07 | +0,24 |
| GLOBAL | 96.534 | 0,5099 | 0,033 | [0,476; 0,544] | 4/6 | 1,07 | +0,87 |

Con las 53 features (rama con estructura) el cuadro es el mismo: mejor Basic Materials
0,5445, peores Industrials 0,4900 y Communication Services 0,4890.

**Ningun sector califica.** Los tres candidatos (Basic Materials, Energy, Consumer
Defensive) tienen AUC media arriba de 0,54 o cerca, pero **su IC95 incluye 0,50**: con 6
mediciones y un desvio de 0,03-0,10 entre ventanas, no se puede afirmar que discriminen.
Y el que mejor AUC media tiene (Basic Materials, 8.115 filas) es de los sectores mas
chicos, donde el intervalo es mas ancho.

**La prueba de heterogeneidad tambien da negativa**: la dispersion de AUC ENTRE sectores
es 0,0225, y la dispersion DENTRO de cada sector entre folds es 0,0581 -- casi el triple.
Es decir: lo que parece "un sector que anda mejor" es sobre todo QUE VENTANA DE 6 MESES
te toco mirar, no el sector. Si la heterogeneidad sectorial fuera real, la relacion
tendria que ser la inversa.

Un detalle que refuerza la lectura: la sec. 2.6 de `ml_reentrenamiento.md` habia medido
el mejor spread en **Financial Services** (+7,3) sobre una sola ventana con las features
viejas. Aca Financial Services da AUC 0,5065 con lift 0,98 (por debajo de 1: el decil
alto acierta MENOS que la base). El unico que se repite es Industrials, malo en las dos
mediciones.

**Decision (regla del pre-registro): se frena. El paso 2 no se corre.** No hay premisa
que sostenga entrenar 9 modelos sectoriales: no habria senal que capturar y cada modelo
tendria entre 5.800 y 19.200 filas contra las 96.500 del global. Por industria ya estaba
descartado por tamano (13 industrias con 5+ tickers, 31 con uno solo).

Lo que esto cierra y lo que no:
- **Cierra** la pregunta "el problema es que el modelo es global". No lo es: el global no
  anda porque las features no informan sobre el retorno a 20 ruedas, y partir los datos
  no crea informacion.
- **Confirma por la razon correcta** el rechazo de los modelos sectoriales de
  `ml_reentrenamiento.md` sec. 2.5, cuyo head-to-head se habia medido con las features
  contaminadas (y con un sesgo que favorecia al global).
- **No cierra** la pregunta de la HIPOTESIS: label relativo al universo del dia en vez de
  absoluto, horizonte mas corto que 20 ruedas, o dejar la prediccion y quedarse con las
  reglas con salidas (sec. 9.3). El lockbox sigue intacto para eso.

---

### Fase 1 -- modulos puros y tests (no cambia ninguna decision)
- Swings CONFIRMADOS: el swing de la barra p se conoce en p+N con la ventana completa;
  la fecha del evento es p+N. Una sola funcion para diario y semanal.
- Velas con definicion clasica y contexto (umbrales iniciales, se fijan con casos):
  - envolvente: vela previa de color opuesto y no plana; cuerpo actual envuelve al
    previo (abre <= cierre previo y cierra >= apertura previa, alguna estricta);
  - martillo / hanging man: sombra inferior >= 2x cuerpo, cuerpo <= 30% del rango,
    sombra superior <= 10%, cualquier color; martillo si viene de caida, hanging man
    si viene de suba;
  - estrella fugaz / inverted hammer: espejo;
  - doji: cuerpo <= 5% del rango, con precedencia explicita frente a martillo/estrella;
  - marubozu con direccion.
- Test de invariancia: `calcular(datos[:t+1]).iloc[-1] == calcular(datos).iloc[t]` para
  todo t, y casos armados a mano por patron.
- Scripts de medicion al repo (reproducen las secciones 4-6).

### Fase 2 -- historia honesta en paralelo + semanal (no cambia decisiones de FT)
- Tabla nueva calculada por el Paso 2 al lado de la vieja (que no se toca).
- Semanal del dashboard y Telegram con el modulo nuevo; semana completa por
  `trading_calendar` (ultima rueda habil de la semana presente en los datos).
- COMPUERTA: re-medir con historia honesta si algun evento SMC o patron tiene ventaja.
  CORREGIDO el 17/9/2026: esa compuerta medía la senal de ENTRADA sola a plazo fijo, que
  es una prueba de prediccion y no evalua una estrategia que decide entradas Y salidas.
  La compuerta real es el backtest completo de la seccion 9.3, y su resultado NO fue
  apagar SMC: con swings confirmados rapido (N=5 / N=3) la regla pasa. Ver 9.4.

### Fase 3 -- modelo ML v3 con features honestas
- `features_ml` + tabla nueva; auditar las 53 features con el test de invariancia.
- Particion y compuerta segun seccion 10, fijadas ANTES de ver resultados.
- Despliegue en paralelo como FT_ML_SCANNER_v3, registrado en `ft_cambios`.

### Fase 4 -- migrar consumidores (despues de la Etapa 4)
- Score del scanner, SMC y COMBO (como versiones nuevas en paralelo), dashboard, MCP,
  Telegram. Re-correr backtests SMC/COMBO. Retirar los modulos `_1w` duplicados.

---

## 10. Particion de datos para la v3 (DECIDIDO 17/9/2026)

**Decision:** walk-forward purgado (holdout 126 ruedas, embargo 20) sobre el primer 80%
de las ruedas con label para todo lo que se decide (features, calibracion, cortes,
compuerta) + el ultimo 20% como lockbox, con embargo de 20 ruedas antes, evaluado UNA
sola vez con la compuerta fijada antes de correrlo + reentrenar con todo para desplegar
en paralelo (FT es el examen final). Las fechas de corte se fijan en la metadata del
modelo y en este doc antes de correr, no en `features_ml.segmento`. Con la historia al
17/9/2026 el lockbox seria 2025-07-15 -> 2026-08-13; las fechas definitivas se fijan
cuando se reconstruya el dataset (Fase 3).

Discusion que llevo a la decision. Propuesta inicial del usuario: Training 40% / Test
20% / Validation 20% / Backtesting 20%.

Como esta hoy: `feature_store.agregar_segmento` parte 70/15/15 POR TICKER (por posicion
dentro de la historia de cada ticker). Como los tickers tienen historias de distinto
largo, los segmentos se pisan en fechas: TRAIN 2020-10-15 -> 2026-04-06, TEST
2024-12-02 -> 2026-06-24, BACKTEST 2025-10-22 -> 2026-08-13. Filas de TRAIN de un ticker
son del mismo dia que filas de TEST de otro, y no hay embargo para el label de 20 ruedas.
El walk-forward y la v2 no usan esa columna (cortan por fecha), pero V3 si.

Simulacion 40/20/20/20 por fecha sobre `features_ml` con label (1.463 ruedas, 175.425
filas), con embargo de 20 ruedas entre tramos:

| Tramo | Fechas | Ruedas | Filas | Tickers | Base | Ret 20d medio |
|---|---|---|---|---|---|---|
| Training | 2020-10-15 -> 2023-02-10 | 585 | 41.201 | 122 | 44,5% | -0,12% |
| Test | 2023-03-14 -> 2024-04-11 | 272 | 33.184 | 122 | 51,6% | +1,90% |
| Validation | 2024-05-10 -> 2025-06-12 | 273 | 38.733 | 196 | 50,6% | +1,80% |
| Backtesting | 2025-07-15 -> 2026-08-13 | 273 | 53.507 | 196 | 50,4% | +2,13% |

Observaciones (para decidir):
- Cortar por FECHA con embargo es lo correcto y corrige el problema de hoy.
- El 20% final intocable (backtesting / lockbox) es la parte mas valiosa: una ventana
  que no se mira hasta congelar la configuracion, evaluada una sola vez.
- Training 40% = 2020-10 -> 2023-02: solo 122 tickers (los 74 incorporados despues no
  tienen historia ahi) y dominado por el bear 2022 (base 44,5%).
- Cada 20% es ~1 anio = un regimen. Un solo corte mide ese anio, no la habilidad
  (sec. 2.4 de ml_reentrenamiento.md).
- Con la configuracion congelada (RF global, sin concurso de algoritmos) hay poco que
  ajustar: los roles de Test y Validation se superponen.

Por eso se eligio el walk-forward en el 80% + lockbox (arriba) en lugar de cuatro
tramos fijos.

---

## 11. Reglas que deja

1. **Toda feature pasa el test de invariancia**: calcularla con datos hasta t y leer la
   ultima fila tiene que dar lo mismo que calcularla con toda la historia y leer la fila
   t. Si no, la historia no es lo que se sabia ese dia y no sirve para entrenar,
   validar ni backtestear.
2. **Un evento se fecha el dia en que se conoce**, no el dia al que se refiere.
3. **Una tabla que se recalcula entera cada dia no es un registro**: si la fila de una
   fecha puede cambiar manana, no es lo que se vio ese dia.
4. **Una senal que en la historia "anda demasiado bien" es sospechosa antes que buena**:
   -4,7% a 5 ruedas para un swing high es informacion del futuro, no un patron.
5. Misma familia que `scan_fecha` / `fecha_datos` / `features_sector`: una lectura que
   nunca falla y devuelve un numero plausible.

---

## 12. Auditoria de las 3 tablas contra el OHLCV (17/9/2026)

Pregunta: `features_velas`, `features_estructura` y `features_precio_accion`, estan bien
calculadas a partir de `precios_diarios`? Son dos preguntas distintas y se miden por
separado:

1. **La tabla reproduce el codigo?** Recomputar con el modulo desde el OHLCV y comparar
   contra lo guardado. Detecta datos rancios, escala de split vieja y bugs de
   persistencia.
2. **El codigo calcula lo correcto?** Reimplementar cada definicion A MANO desde el OHLCV,
   sin importar el modulo, y aplicarla sobre las filas que la tabla marco. Ningun test
   unitario sobre datos sinteticos contesta esto.

Reproducible: `python scripts/manual/auditar_features_tablas.py` (sale 0 si velas y
estructura pasan todo y las 3 tablas reproducen; los defectos de precio_accion se
informan como CONOCIDOS). Correrlo despues de un backfill, de `splits.py corregir` o de
tocar `velas.py` / `estructura.py` / `precio_accion.py`.

### 12.1 Reproducibilidad: las 3 tablas estan bien persistidas

12 tickers (10 al azar + KLAC y CRWD, que tuvieron split corregido), todas las columnas:

| Tabla | Filas comparadas | Resultado |
|---|---|---|
| `features_velas` | 12.774 | reproduce exacto |
| `features_estructura` (N=3, 5 y 10) | 12.774 | reproduce exacto |
| `features_precio_accion` | 10.335 | reproduce exacto (tolerancia de NUMERIC(x,4)) |

Sin datos rancios, sin escala de split vieja, sin errores de upsert. Lo guardado es
exactamente lo que calcula el codigo: la pregunta pasa a ser si el codigo es correcto.

### 12.2 `features_velas`: correcta

Implementacion independiente de la definicion del docstring de `velas.py`, 40 tickers:

| Chequeo | Cumple |
|---|---|
| `patron_engulfing_bull` envuelve de verdad el cuerpo previo | 1.661 / 1.661 (100%) |
| `patron_engulfing_bear` envuelve de verdad | 1.815 / 1.815 (100%) |
| `patron_hammer` cumple la forma (cuerpo <= 30%, sombra inf >= 2x cuerpo, sombra sup <= 10%) | 603 / 603 (100%) |
| `patron_hammer` viene despues de una CAIDA | 603 / 603 (100%) |
| `patron_hanging_man` viene despues de una SUBA | 652 / 652 (100%) |
| `patron_marubozu_bull` es alcista | 1.573 / 1.573 (100%) |
| Una sola etiqueta de forma por vela | 46.641 / 46.641 (100%) |

### 12.3 `features_estructura`: correcta

25 tickers:

| Chequeo | Cumple |
|---|---|
| INVARIANCIA numerica: la fila de t no cambia al llegar barras nuevas | 150 / 150 (100%) |
| `is_sh_5` marca un swing high REAL (max de las N previas y >= las N siguientes) | 1.618 / 1.618 (100%) |
| `is_sl_5` marca un swing low REAL | 1.623 / 1.623 (100%) |
| `dias_sh_5` entre 5 y 252 (nunca menor que N) | 27.103 / 27.103 (100%) |
| `bos_bull` y `choch_bull` mutuamente excluyentes (idem bear) | 27.525 / 27.525 (100%) |
| N=3 backfilleado en toda la tabla | 223.330 / 223.330 (100%) |

El script repite los chequeos para N=3, 5 y 10: todos 100%.

Frecuencia de swings confirmados sobre toda la tabla: N=3 9,53% SH / 9,55% SL; N=5
5,90% / 5,99%; N=10 3,07% / 3,10%. Queda en ~2/3 del teorico uniforme (1/(2N+1) =
14,29% / 9,09% / 4,76%) en las tres ventanas, y es correcto: en tendencia los extremos
se acumulan en el borde de la ventana y fallan el test del lado que falta. La simetria
casi perfecta entre highs y lows es buena senal.

### 12.4 `features_precio_accion`: los patrones estan mal, el resto esta bien

Los mismos chequeos de 12.2 sobre las columnas de patrones de esta tabla (40 tickers).
Cuantifica lo que la sec. 5 habia encontrado por lectura:

| Chequeo | Cumple | Lectura |
|---|---|---|
| `patron_engulfing_bull` envuelve | 1.404 / 4.904 (**28,6%**) | 71,4% de falsos positivos: el codigo solo compara TAMANO de cuerpo (`body_abs > prev_body`), nunca si envuelve |
| `patron_engulfing_bear` envuelve | 1.504 / 4.617 (**32,6%**) | idem |
| `patron_hammer` con sombra superior <= 10% | 495 / 1.514 (**32,7%**) | 67% no tienen forma de martillo: no hay limite a la sombra opuesta |
| `patron_hammer` tras una caida | 727 / 1.514 (**48,0%**) | 52% son hanging man (misma forma, arriba del rango, lectura opuesta) |
| `patron_doji` sin doble etiqueta | 1.295 / 1.995 (**64,9%**) | 35% es doji y martillo/estrella a la vez |
| `patron_marubozu` distingue direccion | 0 / 2.568 (**0%**) | la columna no tiene signo |

Y dos defectos de otra clase -- **valores inventados**: un numero confiado donde no hay
dato para calcularlo.

- **`tendencia_velas`** se calcula como `velas_alcistas_5d.fillna(0) * 2 - 5`. Donde
  `velas_alcistas_5d` es NULL (no hay 5 barras), escribe **-5**, el maximo bajista
  posible, en vez de NULL.
- **`rango_expansion`** es `(rango_pct > 1.5 * rango_ma10).astype(int)`: cuando la media
  de 10 dias es NaN, la comparacion da False y se guarda **0**.

Son 1.092 filas de 183.385 (0,595%), 783 dentro del dataset ML. Lo interesante es DONDE
caen:

```
  2021-12-10   103 tickers      2025-02-10    75 tickers
  2021-12-13   103 tickers      2025-03-04    73 tickers
  2021-12-14   103 tickers      2025-02-13    72 tickers
  2020-10-15    19 tickers      ...
  -> 32 fechas distintas, 200 tickers
```

No es el arranque de la serie de cada ticker: son **las costuras de cada backfill**. Las
ventanas rolling se reinician en el borde del lote que se cargo, y los tres grupos
(19 / 103 / 73-75 tickers) son exactamente las tres cohortes de alta del universo. El
defecto se va a repetir en cada backfill futuro.

Ademas, **`tendencia_velas` es exactamente `2 * velas_alcistas_5d - 5` en el 100,00% de
las filas** con dato: una transformacion afin, cero informacion nueva. Y `pos_rango_20d`
mide la posicion dentro del rango de CLOSES mientras `dist_max_20d` mide contra el
maximo de HIGHS: de 26.488 filas con `pos_rango_20d = 1` ("en el techo"), el 62,3% tiene
el close mas de 0,5% debajo del maximo real. No es un bug, pero las dos columnas dicen
"maximo de 20 dias" y no son el mismo maximo.

**Lo que esta BIEN calculado** (reproduce y la formula es la documentada): `body_pct`,
`body_ratio`, `upper/lower_shadow_pct`, `es_alcista`, `gap_apertura_pct`,
`rango_diario_pct`, `rango_rel_atr`, `clv`, `body_pct_ma5`, `velas_alcistas_5d/10d`,
`dist_max_20d`, `dist_min_20d`, `pos_rango_20d`, `vol_ratio_5d`, `vol_spike`,
`up_vol_5d`, `ad_flow`, `chaikin_mf_20`, `vol_price_confirm`, `vol_price_diverge`,
`inside_bar`, `outside_bar`.

### 12.5 Quien lee hoy los patrones mal definidos

| Consumidor | Que lee | Estado |
|---|---|---|
| FT_SMC_v1 (`ft_scoring.calcular_score_estructura`) | `patron_engulfing_bull`, `patron_hammer` | A PROPOSITO: es el control de FT_SMC_v3, que lee `features_velas` |
| Scanner (`feature_calculator`, `17_scanner_alertas.py`) y su Telegram | `patron_hammer`, `patron_engulfing_bull` | pendiente, Fase 4 |
| Dashboard (`sintesis_data.py`, `dashboard_sintesis.py`) | envolventes, hammer, shooting star | pendiente, Fase 4 |
| MCP (`screener`, `overview`, `queries.py`) | los 6 patrones + `tendencia_velas` + `rango_expansion` | pendiente, Fase 4 |
| FT_COMBO_v1 | score de velas | DISCONTINUADA (17/9/2026) |

El modelo ML NO lee ninguna columna de `features_precio_accion`. Los valores inventados
caen en filas historicas (costuras de backfill), no en la ultima fila que muestran el MCP
y el dashboard.

### 12.6 Que se hace con esto

- **Patrones**: salen de `features_velas`. No se reparan en `precio_accion.py`: el
  reemplazo ya existe y esta auditado. La migracion de consumidores es la Fase 4.
- **Flujo de volumen y microestructura** (`ad_flow`, `chaikin_mf_20`, `up_vol_5d`,
  `vol_ratio_5d`, `clv`, `gap_apertura_pct`, `rango_rel_atr`): estan bien y son la
  familia sin usar con cobertura completa que propone docs/features_ml.md sec. 11.
- **`tendencia_velas`**: no usar (afin de `velas_alcistas_5d`, y con -5 inventado).
- **`rango_expansion`**: no usar hasta corregir el 0 inventado.
- **Pendiente, sin hacer**: corregir el `fillna(0)` / `astype(int)` de `precio_accion.py`
  para que dejen NULL donde no hay dato. Hasta entonces, el proximo backfill vuelve a
  inventar esos valores en su costura.

### 12.7 Regla que deja

**"La tabla reproduce el codigo" no es "la tabla esta bien".** La reproducibilidad
detecta errores de persistencia; la definicion se audita contra el OHLCV con una
implementacion independiente. Y un `fillna(0)` o un `astype(int)` sobre una comparacion
con NaN no falla: escribe un numero plausible donde no hay dato. Misma familia que
`scan_fecha`, `fecha_datos` y `features_sector`.
