# Features del ML -- inventario, que aporta y que datos hay para construir nuevas

**Estado**: medido el 17/9/2026 sobre el dataset del walk-forward de la Fase 3a.
**Alcance**: solo MEDICION e inventario. No se creo ninguna feature nueva ni se
modifico ningun modelo con este analisis.
**Relacionados**: [ml_reentrenamiento.md](ml_reentrenamiento.md) (diagnostico y fases),
[estructura_velas.md](estructura_velas.md) (leakage y auditoria de las tablas de
features), [fuentes_fundamentales.md](fuentes_fundamentales.md) (la fuente SEC).

Atajos por pregunta:

| Si la pregunta es... | Ir a |
|---|---|
| que features usa el modelo y como se calcula cada una | sec. 2 |
| cuales aportan y cuales no | sec. 3 |
| cuales son redundantes entre si | sec. 4 |
| que tablas hay en la DB y cuales sirven de insumo | sec. 5 |
| cuanta historia hay de verdad (training/test/validation) | sec. 6 y 7 |
| por que los folds no son comparables | sec. 8 |
| por que el label actual es el problema principal | sec. 9 |
| sirve el PER / la valuacion fundamental como feature | sec. 10 |
| que se propone para modelos nuevos | sec. 11 |
| reglas que deja este analisis | sec. 12 |

Reproducir las mediciones: `python scripts/ml/analizar_features_ml.py` (la ablacion de la sec. 3
aparte, con `--seccion ablacion`, ~20 min). La auditoria de las tablas de features:
`python scripts/manual/auditar_features_tablas.py`.

---

## 1. Resumen

El modelo usa 53 features. Se midio el aporte de cada familia quitandola del set
completo y recorriendo los mismos 6 folds purgados: **ninguna familia que describa al
ticker aporta nada medible**. La unica que mueve la aguja son las 5 variables de
contexto sectorial (+0,0126 de AUC), que son identicas para todos los tickers de un
sector en una fecha: no sirven para elegir cual comprar, son un termometro de regimen.
Sacar los 4 indicadores tecnicos base MEJORA el AUC (0,5152 vs 0,5133).

La causa no es la falta de features: es el LABEL. `retorno_20d > +1%` es absoluto, y su
base rate se mueve de 0,302 a 0,694 segun el trimestre. La pregunta que se le esta
haciendo al modelo la contesta el regimen del mercado, no la accion.

El inventario de la DB (71 tablas) muestra tres fuentes sin usar con cobertura completa
(flujo de volumen, sorpresa de volumen, eventos de balance) y que el techo de historia
no lo pone ninguna tabla de features sino `precios_diarios`: hay 19 tickers con 5,7
anios, 122 con 5,6 y 78 con menos de 1,5.

La familia de valuacion fundamental se midio entera: el PER y sus variantes dan moneda. El
unico indicio (percentil transversal de EV/EBITDA, 6/6 folds) va en la direccion CONTRARIA
al valor -- lo caro subio mas -- y se mide sobre un 34% de las filas sesgado por sector.
Detalle en sec. 10.

---

## 2. Las 53 features del modelo

Fuente de la lista: `src/ml/trainer.FEATURE_COLS` (29) +
`src/indicators/market_structure.FEATURE_COLS_MS` (24). El dataset se arma en
`scripts/ml/walkforward_ml.cargar_dataset`: `features_ml` JOIN la tabla de estructura,
`WHERE label_binario IS NOT NULL`.

### 2.1 Las 29 no estructurales

| # | Feature | Tabla origen | Descripcion | Calculo |
|---|---|---|---|---|
| 1 | `rsi14` | `indicadores_tecnicos` | fuerza relativa 14 ruedas (0-100) | `RSIIndicator(close, 14)` de `ta` |
| 2 | `macd_hist` | `indicadores_tecnicos` | histograma MACD (escala de precio) | `MACD(12,26,9).macd_diff()` |
| 3 | `adx` | `indicadores_tecnicos` | fuerza de tendencia (0-100) | `ADXIndicator(high,low,close,14)` |
| 4 | `vol_relativo` | `indicadores_tecnicos` | volumen contra su promedio | `volume / volume.rolling(20).mean()` |
| 5 | `dist_sma21` | `indicadores_tecnicos` | distancia a la SMA21 en % | `(close - sma21)/sma21*100` |
| 6 | `dist_sma50` | `indicadores_tecnicos` | idem SMA50 | `(close - sma50)/sma50*100` |
| 7 | `dist_sma200` | `indicadores_tecnicos` | idem SMA200 | `(close - sma200)/sma200*100` |
| 8 | `bb_posicion` | CONSTRUIDA | posicion dentro de Bollinger (0-1) | `((close-bb_lower)/(bb_upper-bb_lower)).clip(0,1)`; 0,5 si la banda es degenerada |
| 9 | `atr14_pct` | CONSTRUIDA | volatilidad como % del precio | `atr14/close*100` |
| 10 | `momentum_pct` | CONSTRUIDA | retorno ~10 ruedas en % | `momentum/(close-momentum)*100` |
| 11 | `score_ponderado` | `scoring_tecnico` | score de las reglas (0-1) | suma ponderada de las 6 condiciones |
| 12 | `condiciones_ok` | `scoring_tecnico` | cuantas condiciones se cumplen (0-6) | `sum(cond_*)` |
| 13-18 | `cond_rsi`, `cond_macd`, `cond_sma21`, `cond_sma50`, `cond_sma200`, `cond_momentum` | `scoring_tecnico` | banderas 0/1 de cada regla | umbrales fijos sobre las columnas 1-7 |
| 19 | `z_rsi_sector` | `features_sector` | RSI del ticker contra sus pares, en desvios | `(rsi - media_sector)/desvio_sector` en la fecha |
| 20 | `z_retorno_1d_sector` | `features_sector` | idem retorno de 1 dia | idem |
| 21 | `z_retorno_5d_sector` | `features_sector` | idem retorno de 5 dias | idem |
| 22 | `z_vol_sector` | `features_sector` | idem volumen relativo | idem |
| 23 | `z_dist_sma50_sector` | `features_sector` | idem distancia a la SMA50 | idem |
| 24 | `z_adx_sector` | `features_sector` | idem ADX | idem |
| 25 | `pct_long_sector` | `features_sector` | % de tickers del sector en senal larga | breadth del sector esa fecha |
| 26 | `rank_retorno_sector` | `features_sector` | puesto del ticker por retorno en su sector | rank percentil |
| 27 | `rsi_sector_avg` | `features_sector` | RSI promedio del sector | promedio de la fecha |
| 28 | `adx_sector_avg` | `features_sector` | ADX promedio del sector | promedio de la fecha |
| 29 | `retorno_1d_sector_avg` | `features_sector` | retorno de 1 dia promedio del sector | promedio de la fecha |

Las 19-24 son z-scores CONTRA EL SECTOR; las 25-29 son PROPIEDADES DEL SECTOR: para
todos los tickers de un sector en una fecha valen lo mismo. Esa distincion es la que
explica el resultado de la sec. 3.

Los 4 tickers sin contexto sectorial (Real Estate n=3, Utilities n=1) reciben las 11 en
NaN. Ver CLAUDE.md, patrones criticos.

### 2.2 Las 24 de estructura

Doce columnas por cada N en (5, 10), en `features_estructura` (honesta) o
`features_market_structure` (contaminada; NO usar para entrenar):

`is_sh_N`, `is_sl_N` (se confirmo hoy un swing), `estructura_N` (+1 HH/HL, -1 LH/LL,
0 indefinida), `dist_sh_N_pct`, `dist_sl_N_pct` (distancia al ultimo swing confirmado),
`dias_sh_N`, `dias_sl_N` (antiguedad, tope 252), `impulso_N_pct`, `bos_bull_N`,
`bos_bear_N`, `choch_bull_N`, `choch_bear_N`.

Definiciones exactas: `src/indicators/estructura.py`, docstring del modulo.

### 2.3 El label

`label_binario = 1 si retorno_20d > +1%` (`UMBRAL_NEUTRO = 0.01`). Es la unica variable
con `shift(-n)` en todo `src/`: `src/ml/feature_store.py:125`. Es correcto que lo tenga
(es el target) y es el unico lugar donde debe estar.

---

## 3. Que aporta cada familia (ablacion, 17/9/2026)

Metodo: quitar la familia entera del set de 53 y volver a correr los MISMOS 6 folds
purgados. Se mide en contexto y no de a una: dos features correlacionadas pueden parecer
utiles por separado y no sumar juntas (`rsi14` y `dist_sma21` tienen rho 0,92).

| Set | Feats | AUC media | AUC por fold |
|---|---|---|---|
| TODAS (base) | 53 | 0,5133 | 0,551 0,565 0,522 0,505 0,468 0,468 |
| sin indicadores | 49 | 0,5152 | 0,555 0,575 0,522 0,503 0,469 0,466 |
| sin dist_smas | 50 | 0,5128 | 0,545 0,570 0,521 0,508 0,465 0,467 |
| sin engineered | 50 | 0,5141 | 0,546 0,570 0,520 0,508 0,473 0,467 |
| sin scoring | 45 | 0,5132 | 0,550 0,558 0,523 0,508 0,469 0,470 |
| sin sector_z | 47 | 0,5096 | 0,546 0,565 0,521 0,496 0,470 0,460 |
| sin sector_ctx | 48 | **0,5007** | 0,509 0,544 0,513 0,489 0,472 0,477 |
| sin estructura | 29 | 0,5099 | 0,525 0,558 0,521 0,501 0,490 0,464 |
| sin flags_estruct | 41 | 0,5134 | 0,549 0,571 0,523 0,508 0,466 0,464 |

Aporte = AUC base menos AUC sin la familia (positivo = aporta):

| Familia | Feats | Aporte |
|---|---|---|
| `sector_ctx` (pct_long, rank_retorno, rsi_avg, adx_avg, retorno_1d_avg) | 5 | **+0,0126** |
| `sector_z` (los 6 z-scores) | 6 | +0,0037 |
| `estructura` (las 24) | 24 | +0,0035 |
| `dist_smas` | 3 | +0,0006 |
| `scoring` (las 8) | 8 | +0,0002 |
| `flags_estruct` (los 8 is_/bos_/choch_) | 8 | -0,0000 |
| `engineered` (bb_posicion, atr14_pct, momentum_pct) | 3 | -0,0007 |
| `indicadores` (rsi14, macd_hist, adx, vol_relativo) | 4 | **-0,0018** |

**Lectura.** Las 48 features que describen al TICKER aportan cero entre todas. Lo unico
que mueve el AUC son las 5 que describen al SECTOR -- que valen lo mismo para todos los
tickers de ese sector y por lo tanto no pueden ordenar nada dentro de el. Quitarlas deja
el modelo en 0,5007, la moneda exacta.

Dos avisos de honestidad sobre ese +0,0126: **no es consistente** (la base gana en 4 de
6 folds y pierde en 2) y es del tamano del ruido entre folds, que va de 0,468 a 0,565.
No dice "sector_ctx es buena": dice "es la unica que no es exactamente nada".

Y confirma dos decisiones previas con numero: las 24 de estructura aportan +0,0035
contra los +0,005 que pedia la compuerta pre-registrada (bien rechazadas, sec. 9.6 de
estructura_velas.md), y las 8 del scoring aportan +0,0002, que es lo que se espera de
variables determinadas por otras que ya estan en el set.

---

## 4. Redundancia

Medido sobre el dataset (`--seccion redundancia`):

- **Las 8 del scoring son umbrales o sumas sobre otras features del set.** Exactas en el
  100% de las filas: `cond_macd = (macd_hist > 0)`, `cond_momentum = (momentum_pct > 0)`,
  `condiciones_ok = suma de las 6 cond_*`. Casi exactas: `cond_sma21/50/200` coinciden con
  el signo de `dist_sma21/50/200` en el 98,3% / 98,5% / 99,1% de las filas.
  `score_ponderado` y `condiciones_ok`: Spearman 0,995. Aporte medido del grupo entero:
  +0,0002.
- **La posicion de corto plazo del precio se mide 4-5 veces**: `dist_sma21` correlaciona
  0,931 con `momentum_pct`, 0,921 con `rsi14` y 0,907 con `bb_posicion`. En total, 32
  pares del set tienen |Spearman| >= 0,80.
- Fuera del modelo, en `features_precio_accion`, `tendencia_velas` es exactamente
  `2*velas_alcistas_5d - 5` en el 100,00% de las filas: transformacion afin, cero
  informacion nueva (estructura_velas.md sec. 12).

---

## 5. Que hay en la DB (71 tablas, LOCAL)

| Rol | Tablas | Sirve de insumo? |
|---|---|---|
| Materia prima | `precios_diarios`, `precios_semanales`, `futuros_diarios` | si (indirecta) |
| Derivadas diarias x ticker | `indicadores_tecnicos`, `scoring_tecnico`, `features_precio_accion`, `features_estructura`, `features_velas`, `features_sector`, `ticker_zscore_diario` | **si, el nucleo** |
| Contaminada | `features_market_structure` (+`_1w`) | NO (mira 10 ruedas al futuro) |
| Contexto de mercado x fecha | `features_regimen_macro`, `indicadores_tecnicos_futuros` | parcial (sec. 7) |
| Fundamental profundo | `fundamentales_sec_q`, `fundamentales_sec_multiplos_d`, `fundamentales_sec_acciones`, `acciones_circulacion` | parcial (sec. 10) |
| Fundamental corto | `fundamentales_income/balance/cashflow_q`, `ratios_q`, `valuation_q`, `ticker_vs_sector` | NO (ventana rodante de 8 Q) |
| Eventos | `earnings_historico`, `earnings_calendar`, `splits_aplicados`, `universo_cambios`, `polygon_splits` | **si, familia ausente** |
| Opciones | `opciones_snapshot` (4,35 GB), `_resumen_diario`, `_pcr_plazo_diario`, `_zscore_diario`, `_sector_*` | NO hasta ~abril/2027 |
| Salidas del sistema | `alertas_scanner`, `ft_*` (8), `bt_hist_*` (4), `operaciones_bt*`, `resultados_*` (5), `veredictos_universo_diario`, `perfiles_ticker`, `modelos_produccion` | NO (circular) |
| Catalogo / logs | `activos`, `ticker_pais`, `*_ingesta`, `*_avisos`, `rutina_corridas`, `llm_uso_tokens`, `log_ejecuciones` | NO |

Escala: `opciones_snapshot` son 9,2 M de filas y 4,35 GB -- el 87% del disco de la base
para la tabla con menos historia.

**Por que las salidas del sistema no son insumo**: `alertas_scanner` contiene la
probabilidad del propio modelo (`ml_prob_ganancia`, `ml_prob_v2`). Entrenar con eso es
realimentacion, no informacion. Lo mismo con `ft_*` y `bt_*`: son el resultado de decidir
con las features, no un dato del mercado.

**Por que las fundamentales de yahooquery no sirven para entrenar**: `fundamentales_*_q`
guarda los ULTIMOS 8 trimestres por ticker (arrancan en 2024-08-31; `ratios_q` en
2024-12-31). Es una ventana rodante: no hay historia. La unica fuente fundamental con
profundidad es SEC (`fundamentales_sec_q`, 2018+, 147 tickers).
`fundamentales_valuation_q` tiene 164 period_end desde 2012 y los 200 tickers, **pero no
tiene fecha de publicacion**: sin `filed` no hay point-in-time y no se puede usar para
entrenar sin riesgo de mirar al futuro.

---

## 6. La historia real: el techo lo pone `precios_diarios`

```
  arranque de la serie de cada ticker:
    2020-01   + 19 tickers   (acumulado  19)
    2021-01   +103           (acumulado 122)
    2021-10   +  1           (acumulado 123)
    2023-10   +  1           (acumulado 124)
    2024-04   + 76           (acumulado 200)
```

Mediana 1394 ruedas por ticker; minimo 568, maximo 1685. **No hay 200 tickers con 5
anios: hay 122 con 5,6 anios y 78 con ano y medio.** Todo lo demas se deriva de esto --
`features_ml` no es flaca en 2020-2021 por un problema del dataset, sino porque no
existe el precio.

Dataset con label (`features_ml` JOIN `features_estructura`): **175.425 filas, 1.463
ruedas, 196 tickers, 2020-10-15 -> 2026-08-13**. Reconstruirlo con los 200 tickers
recuperaria solo 1,3% de filas: el limite no es el dataset.

| Anio | Filas | Ruedas | Tickers por rueda |
|---|---|---|---|
| 2020 | 1.026 | 54 | 19 |
| 2021 | 6.320 | 252 | 25 |
| 2022 | 30.439 | 251 | 121 |
| 2023 | 30.500 | 250 | 122 |
| 2024 | 30.860 | 252 | 122 |
| 2025 | 46.096 | 250 | 184 |
| 2026 | 30.184 | 154 | 196 |

**La eleccion del universo es explicita y hay que tomarla ANTES de partir los datos:**

| Opcion | Universo | Ventana | Filas aprox. | Costo |
|---|---|---|---|---|
| **A** | 122 tickers | 2021-01 -> 2026-08 (5,6 anios) | ~150k | los 78 tickers nuevos quedan afuera del train |
| B | 200 tickers | 2025-01 -> 2026-08 (1,5 anios) | ~80k | 2-3 folds, un solo regimen |
| C | 19 tickers | 2020-01 -> 2026-08 | ~30k | seccion transversal inservible |

Recomendada: **A, excluyendo explicitamente los 78 tickers tardios de todos los folds**,
en vez de dejarlos aparecer solo en el ultimo (que es lo que pasa hoy, sec. 8).

---

## 7. Cobertura de cada fuente sobre las filas del dataset

| Fuente | % filas con dato | Desde | Veredicto |
|---|---|---|---|
| `features_precio_accion` | **100,0%** | 2020-10-15 | usable completa |
| `features_velas` | **100,0%** | 2020-10-15 | usable completa |
| `earnings_historico` (balance anterior) | **100,0%** | 2020-01-14 | usable completa |
| `ticker_zscore_diario` | **98,7%** | 2021-04-12 | usable |
| `earnings_historico` (balance proximo) | 95,0% | -- | usable ACOTADA (ver abajo) |
| `fundamentales_sec_multiplos_d` | 69,7% fila / 58% PER / 69% P/S / 35% EV-EBITDA / 24% percentil | 2021-01-04 | parcial, riesgosa (sec. 10) |
| `features_regimen_macro` | 52,8% | 2024-02-01 | inservible COMO ESTA |
| `opciones_zscore_diario` | **8,5%** | 2026-04-21 | inservible para entrenar |
| `alertas_scanner` | 7,5% | 2026-02-23 | inservible + circular |

Dos aclaraciones sobre lo que parece:

- `opciones_pcr_plazo_diario` da 20,6% en un LEFT JOIN, pero es artificio: tiene 3 filas
  por (ticker, fecha), una por plazo. La cobertura real es la de
  `opciones_zscore_diario`, 8,5%. No es una decision de diseno: no existe la historia
  (techo 2026-04-18).
- `features_regimen_macro` cubre poco y esta congelada en 2026-05-22, **pero es
  recomputable**: de sus 12 futuros, ES/NQ/YM/RTY tienen 1.689 ruedas desde 2020-01-02
  (100% del dataset) y los otros 8 (GC, SI, UB, TN, XAE, XAU, ZL, ZM) arrancan el
  2025-05-09 con 342. El bloque de los 4 indices se puede reconstruir entero; el de
  metales, bonos y agro, no.

**`dias_hasta_balance` es look-ahead suave y hay que acotarlo**: la fecha del proximo
balance se conoce ~3-4 semanas antes, no 47 dias antes. Usar la fecha cruda como feature
le da al modelo un dato que ese dia no estaba publicado. La version legitima es acotada
(o binaria: "hay balance en <= 5 ruedas"). `dias_desde_balance` no tiene ese problema.

---

## 8. Los 6 folds no son comparables entre si

| Fold | Filas | Tickers | Ventana | Base rate | AUC |
|---|---|---|---|---|---|
| 1 | 15.283 | 122 | 2022-05 -> 2022-11 | 0,455 | 0,5506 |
| 2 | 15.372 | 122 | 2022-11 -> 2023-05 | 0,472 | 0,5654 |
| 3 | 15.372 | 122 | 2023-05 -> 2023-11 | 0,479 | 0,5221 |
| 4 | 15.372 | 122 | 2023-11 -> 2024-05 | 0,548 | 0,5054 |
| 5 | 15.459 | 123 | 2024-05 -> 2024-11 | 0,513 | 0,4683 |
| 6 | 19.676 | **196** | 2024-11 -> 2025-05 | 0,487 | 0,4682 |

El fold 6 evalua sobre 196 tickers de los cuales 74-76 casi no estuvieron en el train
(entraron en 2024-04 y necesitan 200 ruedas de calentamiento para la SMA200). No es el
mismo experimento que los otros cinco. El lockbox registrado (2025-07-15 -> 2026-08-13)
tiene el mismo problema: su universo no es el del train.

**Regla que deja**: pre-registrar el UNIVERSO ademas de las fechas.

---

## 9. El label absoluto depende del regimen (medido)

Base rate = fraccion de filas con `retorno_20d > +1%`, por trimestre, sobre las
predicciones fuera de muestra del set de desarrollo:

| Trimestre | Base absoluto | Base relativo |
|---|---|---|
| 2022Q2 | 0,334 | 0,496 |
| 2022Q3 | 0,402 | 0,496 |
| 2022Q4 | 0,586 | 0,500 |
| 2023Q1 | 0,465 | 0,500 |
| 2023Q2 | 0,550 | 0,500 |
| 2023Q3 | **0,302** | 0,500 |
| 2023Q4 | **0,638** | 0,500 |
| 2024Q1 | 0,550 | 0,500 |
| 2024Q2 | 0,498 | 0,500 |
| 2024Q3 | 0,550 | 0,497 |
| 2024Q4 | 0,422 | 0,496 |
| 2025Q1 | 0,359 | 0,497 |
| 2025Q2 | **0,694** | 0,500 |
| **desvio entre trimestres** | **0,1204** | **0,0019** |

("relativo" = `retorno_20d` por encima de la MEDIANA del universo de esa rueda.)

**La respuesta correcta al label cambia del 30% al 69% segun el trimestre.** El label
absoluto no pregunta "es esta accion mejor que las otras", pregunta "va a subir el
mercado". Con eso, cualquier feature correlacionada con "el mercado viene subiendo"
parece predictiva en unas ventanas y anti-predictiva en otras -- que es exactamente el
comportamiento de `sector_ctx` en la sec. 3.

### 9.1 El label relativo no rescata al modelo actual (diagnostico)

La MISMA probabilidad del modelo v3, juzgada con los dos labels sobre las mismas filas:

| Fold | AUC vs absoluto | AUC vs relativo |
|---|---|---|
| 1 | 0,5506 | 0,5205 |
| 2 | 0,5654 | 0,5217 |
| 3 | 0,5221 | 0,4970 |
| 4 | 0,5054 | 0,5038 |
| 5 | 0,4683 | 0,4690 |
| 6 | 0,4682 | 0,4922 |
| **media** | **0,5133** IC95 [0,4706; 0,5561], 4/6 | **0,5007** IC95 [0,4801; 0,5214], 3/6 |

El modelo fue ENTRENADO con el label absoluto: esto NO es un modelo nuevo, es la misma
prediccion medida con otra pregunta. Dice dos cosas:

1. **Casi todo lo que parecia senal en los folds 1 y 2 era el regimen**, no la capacidad
   de ordenar tickers. Al sacar la componente de regimen, queda 0,5007.
2. **El label relativo mide con la mitad del ruido**: el rango entre folds se comprime de
   0,468-0,565 a 0,469-0,522, y el ancho del IC95 cae de +-0,043 a +-0,021.

O sea: el label relativo no crea senal, pero es la unica forma de MEDIR con precision
suficiente para decidir algo. Un modelo ENTRENADO con label relativo es una hipotesis
distinta y todavia no se probo.

---

## 10. La familia VALOR (el PER y parientes) -- medida entera

### 10.1 La tabla ya existe y esta bien construida

`fundamentales_sec_multiplos_d`: 152.054 filas, 144 tickers, desde 2021-01-04. Es
exactamente "precio diario x balances": `precios_diarios` sobre `fundamentales_sec_q`.

Point-in-time verificado:

```
  filed_primero <= fecha (correcto)   : 151.809 (99,84%)
  filed_primero >  fecha (MIRA FUTURO): 0
  period_end   >  fecha               : 0
  lag balance -> rueda: min 0 / medio 46 / max 188 dias
```

Cero filas usando un balance publicado despues de la fecha.

### 10.2 La cobertura es 58% y el hueco NO es aleatorio

Sobre las 175.425 filas del dataset:

| | Filas | % |
|---|---|---|
| **con PER** | 101.706 | **58,0%** |
| ticker sin fuente SEC (los no-USA) | 48.630 | 27,7% |
| tiene SEC pero no hay fila ese dia | 4.489 | 2,6% |
| hay fila pero la empresa da PERDIDA (PER indefinido) | 17.299 | 9,9% |
| hay fila, gana plata, y aun asi falta | 3.301 | 1,9% |

Por sector va de **25,7%** (Basic Materials) a **85,9%** (Consumer Defensive), pasando
por Energy 37,1%, Consumer Cyclical 48,0%, Technology 53,4%, Communication Services
62,2%, Financial Services 68,4%, Industrials 72,3%, Healthcare 73,1%.

**La ausencia del PER es casi un identificador de region y de sector.** Un Random Forest
parte por la ausencia y la usa como proxy: seria darle "es una empresa USA rentable de
consumo defensivo" disfrazado de variable de valuacion.

Eso tiene arreglo: el **earnings yield** (`net_income_ttm / market_cap`) esta definido
tambien con perdida (da negativo), recupera el 9,9% y sube la cobertura a 67,1%. Es la
construccion correcta; el PER no lo es.

### 10.3 Senal univariada -- 6 folds purgados, mismas filas que el modelo

| Variable | Cob. | AUC media | IC95 (por fold) | Folds >0,50 |
|---|---|---|---|---|
| PER crudo | 55,5% | 0,4971 | [0,4729; 0,5213] | 2/6 |
| 1/PER | 55,5% | 0,5029 | [0,4787; 0,5271] | 4/6 |
| earnings yield | 67,1% | 0,5003 | [0,4751; 0,5255] | 3/6 |
| earnings yield, percentil transversal | 67,1% | 0,4928 | [0,4668; 0,5188] | 3/6 |
| percentil del PER vs su propia historia | 20,9% | 0,4907 | [0,4363; 0,5451] | 1/3 |
| P/S | 67,9% | 0,4988 | [0,4737; 0,5239] | 1/6 |
| sales yield | 67,9% | 0,5012 | [0,4761; 0,5263] | 5/6 |
| P/S, percentil transversal | 67,9% | 0,5078 | [0,4848; 0,5308] | 4/6 |
| FCF yield | 60,8% | 0,4971 | [0,4729; 0,5213] | 4/6 |
| FCF yield, percentil transversal | 60,8% | 0,4887 | [0,4644; 0,5130] | 2/6 |
| EV/EBITDA | 33,9% | 0,5110 | [0,4942; 0,5279] | 5/6 |
| EV/EBITDA, percentil transversal | 33,9% | 0,5220 | [0,5068; 0,5373] | 6/6 (ver 10.4) |

(IC95 con t de Student de n-1 grados sobre las mediciones por fold; el percentil propio
del PER solo tiene 3 folds con muestra.)

**Salvo EV/EBITDA transversal (sec. 10.4), ninguna tiene IC95 que excluya 0,50.** El PER
no tiene direccion: el retorno a 20 ruedas del decil mas caro menos el mas barato cambia
de signo fold a fold (-2,86 / +5,54 / +1,54 / -1,92 / -5,04 / +0,85 puntos). El earnings
yield transversal, idem: IC de Spearman +0,039, -0,074, -0,057, +0,013, +0,024, -0,042;
en el fold 2 el decil CARO rindio +6,84% y el BARATO -1,11%.

**Por que, estructuralmente** (mas util que el numero): el PER se mueve todos los dias
solo por el precio; el denominador cambia 4 veces al ano. Dentro de un trimestre es una
transformacion monotona del precio, que el modelo ya tiene via `dist_sma*`. La dimension
genuinamente nueva es la TRANSVERSAL -- caro o barato contra los pares de hoy -- y eso
da 0,4928. Ademas el valor como factor funciona en horizontes de anios, no de 20 ruedas.

### 10.4 Lo unico que asomo: EV/EBITDA transversal -- y no es valor

| Variable | Label | Cob. | AUC | IC95 | Folds >0,50 |
|---|---|---|---|---|---|
| **EV/EBITDA percentil transversal** | absoluto | 33,9% | **0,5220** | **[0,5068; 0,5373]** | **6/6** |
| **EV/EBITDA percentil transversal** | relativo | 33,9% | **0,5247** | [0,4973; 0,5521] | **6/6** |
| EV/EBITDA crudo | absoluto | 33,9% | 0,5110 | [0,4942; 0,5279] | 5/6 |
| EV/EBITDA crudo | relativo | 33,9% | 0,5214 | [0,4935; 0,5492] | 5/6 |
| P/S percentil transversal | relativo | 67,9% | 0,5075 | [0,4755; 0,5396] | 3/6 |
| resto de la familia | relativo | -- | 0,484-0,505 | incluye 0,50 | 2-3/6 |

Es la unica variable de toda la medicion con 6/6 folds, y contra el label absoluto su
IC95 excluye 0,50. Tres razones para NO leerlo como un hallazgo:

1. **La direccion es la contraria al valor.** AUC > 0,50 con el percentil de EV/EBITDA
   quiere decir que lo CARO subio mas. Retorno a 20 ruedas del decil mas caro menos el mas
   barato, por fold: -0,04 / +5,78 / +1,93 / +2,69 / -3,59 / +5,56 puntos. Es "lo caro
   siguio subiendo" en el tramo medido (2022-05 -> 2025-05), no una prima por comprar
   barato: un efecto de estilo de ese tramo, no un factor que se pueda suponer estable.
2. **Se mide sobre un 34% sesgado por sector.** El EV exige deuda completa (CLAUDE.md,
   tabla `fundamentales_sec_q`): hay EV/EBITDA en el 83% de las filas de Consumer
   Defensive, 62% de Industrials y 54% de Communication Services, pero en el 12% de
   Financial Services, 13% de Energy y 8% de Basic Materials. El "percentil transversal"
   compara contra un universo distinto en cada sector.
3. **Comparaciones multiples.** La seccion `valor` del script hace 24 comparaciones (12
   variables x 2 labels; ese dia se miraron mas). Bajo azar puro, la probabilidad de que
   al menos una saque 6/6 es 31% (`1 - (1 - 1/64)^24`), y la de que alguna tenga un IC95
   que excluya 0,50 llega a ~70% si fueran independientes (no lo son del todo: los dos
   labels y varias variables estan correlacionados). Encontrar una asi es lo que se
   espera del ruido.

Queda como **hipotesis a pre-registrar**, formulada con su direccion real ("un multiplo
EV/EBITDA alto dentro del universo del dia predice mejor retorno relativo"), con su
propia compuerta y su propio lockbox. No como "valor", y no como feature a incorporar.

### 10.5 Veredicto sobre el PER

No hace falta construir una tabla de PER: existe, esta bien calculada y es point-in-time.
Agregar el PER a los modelos **como esta planteada la pregunta hoy, no aporta**.

Lo que si vale, en este orden: usar **earnings yield y no PER** (definido con perdida,
67% de cobertura), medirlo siempre como **percentil transversal** y no crudo, y dejar el
**percentil transversal de EV/EBITDA como hipotesis pre-registrada con su direccion real**
(caro > barato), con su propia compuerta y su propio lockbox.

Esta medicion dice "la valuacion no predice si una accion supera +1% en 20 ruedas". NO
dice que el valor sea inutil: dice que es inutil para ESTA pregunta y ESTE horizonte.

---

## 11. Seleccion propuesta para modelos nuevos

Ordenada por valor esperado. **Nada de esto esta implementado.**

### Prioridad 1 -- no es una feature, es el label (gratis, 100% de cobertura)

Label relativo al universo del dia (mediana o decil superior). Base rate constante por
construccion, inmune al regimen, y la medicion pasa a tener la mitad del ruido (sec.
9.1). No requiere un solo dato nuevo.

Efecto colateral que hay que declarar: con label relativo, **todo el bloque de regimen
deja de servir por construccion** (es identico para todos los tickers de una fecha, no
ordena nada). Y ese bloque era el unico que aportaba. No es una perdida: es dejar de
cobrar por adivinar el mercado.

### Prioridad 2 -- features puras, cobertura completa, familias ausentes

| # | Variables | Tabla | Cob. | Por que |
|---|---|---|---|---|
| 1 | `vol_zscore`, `percentil_vol`, `retorno_zscore` | `ticker_zscore_diario` | 98,7% | sorpresa/atencion: hoy no hay ninguna medida de "esto es anomalo para ESTE ticker" |
| 2 | `ad_flow`, `chaikin_mf_20`, `up_vol_5d`, `vol_ratio_5d` | `features_precio_accion` | 100% | flujo de volumen: familia entera ausente |
| 3 | `clv`, `gap_apertura_pct`, `rango_rel_atr` | `features_precio_accion` | 100% | microestructura del dia, no otra media movil |
| 4 | `dias_desde_balance` | de `earnings_historico` | 100% | eventos; con horizonte 20 ruedas casi siempre cae un balance adentro |

Las columnas de PATRONES de `features_precio_accion` NO entran: sus definiciones estan
mal (estructura_velas.md sec. 12). Los patrones salen de `features_velas`.

### Prioridad 3 -- feature engineering sobre lo que ya tenemos

| # | Construccion | Insumo | Por que |
|---|---|---|---|
| 5 | percentil de cada feature DENTRO del universo de la rueda | `indicadores_tecnicos` | convierte "nivel absoluto" en "posicion relativa", que es la decision real (elegir 5 de 200). No es lo mismo que los z-scores sectoriales: esos comparan contra 1-3 pares en Real Estate/Utilities; contra el universo el n siempre es 122-196 |
| 6 | `dias_hasta_balance` ACOTADO (o binaria "<= 5 ruedas") | `earnings_historico` | la fecha del proximo balance se conoce ~3-4 semanas antes: cruda es look-ahead suave |
| 7 | bloque de regimen recomputado de ES/NQ/YM/RTY | `futuros_diarios` | 100% de cobertura contra el 52,8% de `features_regimen_macro`. **Solo tiene sentido si NO se pasa a label relativo** |
| 8 | `earnings_yield` + percentil transversal de `ev_ebitda` | `fundamentales_sec_multiplos_d` | unica familia de valuacion. Riesgos declarados: 33-39% de NaN y el RF parte por la ausencia (proxy oculto de region y sector), y el unico indicio va en contra del valor (sec. 10.4) |

### Lo que NO se propone, y por que

- **Mas indicadores sobre la misma serie de precio** (Stochastic, CCI, Williams %R, ROC,
  Ichimoku): los 4 actuales aportan **-0,0018**. Agregar mas de la misma familia es
  repetir un experimento que ya salio negativo.
- **Las 8 del scoring**: +0,0002 y determinadas por otras. Candidatas a SALIR.
- **`features_market_structure`**: mira al futuro.
- **Todas las de opciones**: 8,5% de cobertura. No es diseno, es que no existe la
  historia. Reevaluable ~abril/2027.
- **`fundamentales_ratios_q` / `income_q` / `balance_q` / `cashflow_q` /
  `ticker_vs_sector`**: ventana rodante de 8 trimestres, sin historia.
- **`fundamentales_valuation_q`**: sin fecha de publicacion, no hay point-in-time.
- **`alertas_scanner`, `ft_*`, `bt_*`, `veredictos_*`, `perfiles_ticker`**: salidas del
  sistema; `alertas_scanner` ademas contiene la probabilidad del propio modelo.

---

## 12. Reglas que deja

1. **Medir el aporte de una feature EN CONTEXTO, no de a una.** El AUC univariado de una
   variable correlacionada con otra ya presente es enganoso. La ablacion por grupo sobre
   los mismos folds es la medicion correcta.
2. **Una variable que vale lo mismo para todos los tickers de una fecha no puede ordenar
   tickers.** Si aporta AUC, esta explotando el regimen, y el regimen no se predice.
   Revisar a que nivel varia una feature antes de celebrar su aporte.
3. **El base rate del label se reporta por tramo.** Si se mueve, el AUC entre tramos no
   es comparable y el modelo esta prediciendo el tramo.
4. **Pre-registrar el UNIVERSO ademas de las fechas.** Un fold con 196 tickers y otro con
   122 no son el mismo experimento.
5. **Cobertura parcial: mirar si el hueco es informativo.** Si la ausencia correlaciona
   con sector, region o estado de la empresa, el arbol la usa como proxy. Reportar la
   descomposicion del NaN, no solo el porcentaje.
6. **Contar las comparaciones.** Con 24 pruebas, un 6/6 folds aparece por azar el ~31% de
   las veces. Lo que sobrevive a un screen es una hipotesis para pre-registrar, no un
   hallazgo.
7. **Una transformacion afin de otra columna no es una feature.** Verificar antes de
   agregar (`tendencia_velas` = `2*velas_alcistas_5d - 5`, 100,00% de las filas).
8. **Mirar la DIRECCION y el retorno por decil, no solo el AUC.** Un AUC de 0,52 dice que
   hay orden, no cual ni por que. EV/EBITDA "aporta" al reves de la teoria que lo
   justificaria (sec. 10.4): sin mirar el decil se habria incorporado como "valor".
