# Reentrenamiento ML -- diagnostico, decisiones y esquema (Tarea 20)

Estado: FASES 0-4 EJECUTADAS (2026-07-02). PAUSADO antes de Fase 5 (despliegue),
por decision del usuario. Config final decidida (ver seccion 8). El modelo V3
actual sigue en produccion sin cambios.
Estado operativo vivo: AGENDA.md Tarea 20. Resultados crudos: reportes/ml_walkforward/.

**OJO (17/9/2026): los numeros de las secciones 8 y 8c estan INFLADOS.** Las 24
features de market structure de `features_market_structure` miran 10 ruedas al
futuro. Con lo que se sabia cada dia, v1 y v2 dan AUC ~0,52. Ver seccion 10 y
docs/estructura_velas.md.

**Y la conclusion es mas fuerte que "los numeros no valen" (seccion 10.1):** el
intento de v3 con features honestas ya se corrio con particion y compuertas
pre-registradas y NO pasa -- AUC media 0,5099 en 6 folds purgados, contra 0,6189 con la
tabla que mira al futuro en los MISMOS folds. La ventaja que este documento midio no
existe con datos honestos. Antes de volver a entrenar hay que cambiar la hipotesis
(label relativo, otro horizonte), no los umbrales.

**Por que da 0,51 (seccion 10.2, medido 17/9/2026):** ninguna familia de features que
describa al ticker aporta; lo unico que mueve el AUC es el contexto sectorial, que es
regimen. El label absoluto tiene base rate de 0,30 a 0,69 segun el trimestre. Inventario
completo de features, fuentes de la DB y la familia de valuacion:
[features_ml.md](features_ml.md).

Este documento captura el analisis del ML actual, las decisiones de diseno
(que se hace y por que) y el esquema de 5 fases para reentrenar. Es conocimiento
NO derivable del codigo: performance en vivo, rechazo empirico de los modelos
sectoriales, concentracion de features, dependencia de regimen del label.

---

## 1. Arquitectura ML actual (lo que hay)

Pipeline de 3 niveles, versionado V1/V2/V3 (src/ml/trainer*.py):
- Nivel 1: modelo GLOBAL (todos los sectores juntos).
- Nivel 2: modelos SECTORIALES (uno por sector de SECTORES_ML).
- Nivel 3: challenger -- por cada sector compara global-vs-sector en el TEST del
  sector y despliega el ganador (`deployed.joblib` + tabla `modelos_produccion`).

Algoritmos: RF + XGB + LGBM (todos arboles). Elige el mejor por f1 de la clase
GANANCIA (label_binario = retorno 20d > 0) en TEST.

Versiones:
- V1: 29 features (indicadores + scoring de regla + z-scores sectoriales).
- V2: +30 features de anatomia de vela (no mejoro a V1).
- V3: 29 V1 + 24 de market structure (swings HH/HL/LH/LL, BOS, CHoCH) = 53.
  Es lo DESPLEGADO. Entrenado 2026-04-10.

## 2. Diagnostico (datos reales, 2026-07-02)

### 2.1 El modelo esta viejo y cubre menos que el universo
- Entrenado 10/04 sobre features hasta 12/03/2026 -> ~3,5 meses stale.
- `features_ml` tiene 123 tickers; el universo son 200. 77 tickers (incl. HOOD y
  todos los sumados) NO estan en el training. `modelo_asignado` en `activos`:
  75 vacios, 60 'global', resto por sector, 1 'global_rf' (HOOD) -- inconsistente.

### 2.2 Pero el modelo desplegado FUNCIONA en vivo (no hay que tirarlo)
Medido sobre `alertas_scanner` verificadas (retorno_20d_real, ~1069 filas,
feb-abr 2026):
- Senal COMPRA_FUERTE (la unica que opera la estrategia): +3,94% a 20d,
  59,5% aciertos, prob media 0,80.
- Discriminacion real en extremos: bucket prob 0,6-0,8 -> +4,39%/54,8%;
  bucket mas bajo -> -4,46%/24,5%.
- FT_ML_SCANNER_v1: +$3.726, 50% win, 86 trades -- una de solo 3 estrategias FT
  positivas (de 10), la mas robusta. Y opera con los modelos viejos.
- LIFT en regimen dificil: en BACKTEST (tasa base GANANCIA 27%) el global tiene
  precision_1 0,589 -> lift 2,17x. Aporta MAS valor relativo cuando el mercado
  esta feo (en TEST, base 55%, lift 1,12x).

### 2.3 El modelo se apoya casi todo en dos contadores de timing (bandera roja)
Importancias del global V3 desplegado:
- MS = 0,76 del total; V1 = 0,24. 13 de las top-15 son MS.
- Las 2 top: `dias_sh_10` (0,213) + `dias_sl_10` (0,163) = ~38% de TODA la
  importancia. Son "dias desde el ultimo swing high/low de ventana 10" =
  contadores de recencia/timing. Concentracion + fragilidad de regimen.
- Colateral tranquilizador: las features de la REGLA (score_ponderado,
  condiciones_ok) NO aparecen en el top 20 -> el ML no es un re-derivado del
  scanner de reglas. Pero se fue al extremo opuesto (timing de swing).

### 2.4 El label es muy dependiente del regimen -> evaluar por sector mide "deriva"
Tasa base GANANCIA (label 20d>0) por sector x segmento salta brutalmente:
Energy BACKTEST 95%, Basic Materials TEST 70% -> BT 11%, Consumer Defensive
TEST 80% -> BT 15%. Las ventanas TEST/BACKTEST son de ~2 meses -> un f1 de un
solo split por sector mide si el sector subio en esa ventana, no habilidad.
Raiz: el LABEL de retorno ABSOLUTO hace que el modelo persiga la deriva del
sector. Un label RELATIVO al sector (batir la mediana del sector a 20d) o
triple-barrier lo haria robusto al regimen.
**MEDIDO el 17/9/2026 (features_ml.md sec. 9):** base rate del label absoluto por
trimestre entre 0,302 y 0,694 (desvio 0,1204); el relativo al universo del dia, 0,496-0,500
(desvio 0,0019).

### 2.5 Los modelos sectoriales estan rechazados por evidencia

**OJO (17/9/2026): el head-to-head de abajo se midio con `features_market_structure`,
que mira 10 ruedas al futuro, y el sesgo NO era neutral** -- con mas filas agrupadas el
global explota mejor la fuga que un sectorial chico, asi que la comparacion lo
favorecia por algo que no era su habilidad. La conclusion se RE-MIDIO con features
honestas (docs/estructura_velas.md sec. 9.7-9.8) y **sobrevive, ahora por la razon
correcta**: sobre 96.534 predicciones fuera de muestra en 6 folds purgados, ningun
sector discrimina (mejor: Basic Materials AUC 0,5446 con IC95 [0,476; 0,614], que
incluye 0,50; peor: Industrials 0,4761 con 1 de 6 folds por encima de 0,50), y la
dispersion de AUC ENTRE sectores (0,0225) es un tercio de la dispersion DENTRO de cada
sector entre ventanas (0,0581): lo que parece un sector bueno es la ventana que te toco
mirar. Por INDUSTRIA no se prueba: de 72 industrias, 13 tienen 5 o mas tickers y 31
tienen uno solo.

El challenger V3 desplego el modelo GLOBAL en los 6 sectores (tipo='global' en
`modelos_produccion`). Ningun sectorial gano. Tendencia: V1 Financials sectorial
ganaba, V2 tambien, V3 CERO sectoriales. A mas datos pooled, el global domina.
Head-to-head V3 (global-en-sector vs sector-propio, f1 TEST):

| Sector | Global-en-sector | Sector-propio | Gana |
|---|---|---|---|
| Technology | 0,584 | 0,455 | Global +0,129 |
| Consumer Discretionary | 0,669 | 0,609 | Global +0,060 |
| Financial Services | 0,672 | 0,616 | Global +0,057 |
| Consumer Staples | 0,564 | 0,523 | Global +0,041 |
| Financials | 0,609 | 0,594 | Global +0,014 |
| Industrials | 0,586 | 0,578 | Global +0,008 |

El contexto sectorial YA esta inyectado en el global via features (z_rsi_sector,
adx_sector_avg, rank_retorno_sector -- aparecen en el top de importancias). Esa
es la forma correcta de dar "conciencia de sector", no un modelo por sector.

### 2.6 La discriminacion en vivo VARIA por sector (base del ponderador)
Retorno 20d con prob alta (>=0,55) vs baja, por sector (verificadas en vivo):

| Sector | n | prob-alta | prob-baja | spread |
|---|---|---|---|---|
| Financial Services | 131 | +2,08 | -5,26 | +7,3 OK |
| Consumer Cyclical | 174 | +2,19 | -2,90 | +5,1 OK |
| Consumer Defensive | 74 | +4,11 | -0,87 | +5,0 OK |
| Healthcare | 67 | +1,60 | +1,49 | ~0 nulo |
| Technology | 209 | +11,17 | +10,14 | ~1 nulo (todo subio) |
| Industrials | 105 | -1,11 | +1,15 | -2,3 invertido |

La lectura NO es "hacer modelos sectoriales" (pierden), sino que la
CONFIABILIDAD de la probabilidad del global varia por sector -> ponderar/umbral
por sector. CAVEAT: sale de 1 sola ventana, sectores con n<20 -> hay que validar
con walk-forward antes de fijar un peso.

**VALIDADO Y NEGATIVO (17/9/2026)**: se hizo ese walk-forward con features honestas
(6 folds, 96.534 filas fuera de muestra; docs/estructura_velas.md sec. 9.8) y el orden
de esta tabla NO se reproduce. Financial Services, el mejor spread de aca (+7,3), da
AUC 0,5065 y lift 0,98 -- el decil alto acierta MENOS que la base. El unico que se
repite es Industrials, malo en las dos mediciones. Conclusion: no hay peso sectorial
que fijar, porque la variacion por sector es ruido temporal.

### 2.7 Bug de taxonomia (arreglar antes de reentrenar)
- SECTORES_ML (config) = {Financials, Consumer Staples, Consumer Discretionary,
  Technology, Financial Services, Industrials}.
- `features_ml.sector` (hoy) = taxonomia Yahoo {Technology, Consumer Cyclical,
  Financial Services, Industrials, Consumer Defensive, Basic Materials, Energy,
  Communication Services, Healthcare}.
- "Financials", "Consumer Staples", "Consumer Discretionary" YA NO existen en
  `features_ml.sector` -> reentrenar V3 hoy: 3 de 6 scopes sectoriales entrenan
  sobre 0 filas. Unificar a Yahoo (o retirar scopes sectoriales) antes de tocar
  el training.

## 3. Datos disponibles (viabilidad del walk-forward)

| Tabla | Cobertura | Nota |
|---|---|---|
| precios_diarios | 2020-01 -> 2026-07, 1632 dias, 200 t | 5,5 anios crudos |
| indicadores_tecnicos | 2020-10 -> 2026-07, 200 t | full-history |
| features_market_structure | 2020-01 -> 2026-07, 200 t | full-history |
| features_ml (training) | 2025-01 -> 2026-03, 123 t, 284 dias | EL CUELLO |

Profundidad por ticker: 19 arrancan 2020, 104 en 2021, 76 en 2024, 1 en 2023 ->
123 con >=2 anios, 77 nuevos (~1,5-2,5 a). Los INGREDIENTES (precios/indicadores/
market structure) ya existen full-history para los 200: reconstruir `features_ml`
sobre historia larga es JOIN + labeling + z-scores sectoriales point-in-time,
SIN re-fetch de Yahoo.

Viabilidad: con el features_ml actual (284 dias, ~1,5 regimenes) NO alcanza para
walk-forward. Reconstruyendo 2021-2026 (~1250 dias) -> 6-8 folds purgados, cada
sector medido en 6-8 regimenes -> ahi si se puede juzgar estabilidad de spreads.

## 4. Decisiones de diseno

1. **Motor de produccion = RF-global (el ganador).** No reabrir el concurso de
   algoritmos como camino principal: RF/XGB/LGBM empatan (~0,61 f1), cambiar de
   algoritmo es la palanca de MENOR valor. RF es lo desplegado y lo positivo en
   vivo.
2. **Modelos sectoriales = descartados** (2.5). No se reentrenan. "Todos los
   modelos" colapsa a UNO: RF-global.
3. **Lineal (elastic-net logistico) y calibracion = instrumentos, no
   competidores.** El lineal es un CONTROL ("vale la complejidad del RF?"): si
   empata al RF en lift OOS -> alerta de overfitting. La calibracion (isotonica)
   es una CAPA sobre el RF ganador, obligatoria porque los pesos sectoriales y el
   umbral COMPRA_FUERTE dependen de que la probabilidad signifique algo (hoy el
   bucket alto rinde peor que el medio -> mala calibracion).
4. **Ensamblado:** evaluar soft-voting de los 3 arboles calibrados vs el
   "pick-best" actual (que tira 2 de 3). NO deep learning (muestra y S/N no lo
   justifican).
5. **Validacion = walk-forward purgado, NO un split unico.** El label 20d solapa
   muestras a <20 dias -> purga + embargo (~20 dias) entre bloques, obligatorio.
   Un holdout unico mide un regimen (2.4), no sirve para juzgar el ponderador.
6. **Ponderador sectorial: se VALIDA, no se hardcodea.** Un peso se acepta solo
   si su spread es estable en signo en >=(N-1) de N folds y con magnitud
   material; si no, peso 1,0.
7. **Label: probar binario-absoluto (actual) vs relativo-al-sector** en el mismo
   walk-forward y comparar estabilidad de tasa base + lift OOS. Si el relativo
   estabiliza, buena parte de la necesidad del ponderador se diluye sola.

## 5. Esquema de ejecucion (5 fases con compuertas)

**Fase 0 -- Prerequisitos (sin ML)**
- Fijar taxonomia sectorial canonica (Yahoo) en config + features_ml; retirar
  scopes legacy muertos.
- Verificar que el pipeline de features/labels corre end-to-end sobre 200
  (bajo Plan C esta semi-congelado).
- Congelar rango de rebuild (propuesta 2021-07 -> 2026-06) y regla point-in-time
  de entrada de tickers.

**Fase 1 -- Reconstruccion de features_ml (prerequisito real)**
- Rebuild 2021-2026 para los 200 (ingredientes full-history -> JOIN + labeling +
  z-scores sectoriales point-in-time). Sin re-fetch.
- Preparar DOS versiones de label (binario-absoluto y relativo-al-sector).
- Entregable: features_ml 200t x ~5 anios con purga+embargo 20d listos.

**Fase 2 -- Walk-forward liviano (Alt A) sobre RF-global**
- UN modelo: RF-global, 6-8 folds purgados. Controles en los mismos folds:
  elastic-net logistico + RF calibrado.
- Metrica: precision en decil alto (COMPRA_FUERTE), lift sobre base, calidad de
  calibracion. NO f1 global.
- COMPUERTA 1: el RF calibrado supera al lineal de forma estable? la calibracion
  arregla el bucket alto? Si el lineal empata -> replantear features.

**Fase 3 -- Determinacion de pesos sectoriales**
- Sobre las predicciones OOS de los folds, medir spread por sector en cada fold.
- Aceptacion: signo estable en >=(N-1) de N folds Y magnitud material. Resto ->
  peso 1,0.
- COMPUERTA 2: hay >=3-4 sectores con edge estable? Si no -> el ponderador es un
  retoque menor o se descarta; el valor queda en label+calibracion.

**Fase 4 -- Seleccion de label + config final**
- Comparar binario-absoluto vs relativo-al-sector (estabilidad + lift OOS).
- Congelar: label, RF calibrado, pesos sectoriales validados, umbral
  COMPRA_FUERTE.

**Fase 5 -- Despliegue + cadencia**
- Reentrenar el modelo final sobre todo el historial y desplegar.
- Documentar cadencia de reentrenamiento (mensual/trimestral, manual bajo Plan C)
  y homologar con FT via el cerebro compartido (src/strategies/ml_scanner).

## 6. Alternativas de walk-forward evaluadas (para referencia)

- **Alt A (elegida como 1er paso):** congelar el modelo, rodar solo los pesos.
  Entrena 1 modelo, predicciones OOS, spreads en ventanas rodantes. Barato,
  aisla la pregunta "el ponderador merece existir?".
- **Alt B:** walk-forward anclado (train expansivo, reentrena por fold). El paso
  "serio" si Alt A confirma senal. Mide el sistema completo OOS.
- **Alt C:** ventana deslizante (train fijo ~18m). Responde de paso la cadencia
  de reentrenamiento.
- **Alt D:** purged K-fold combinatorio con embargo (Lopez de Prado). Lo mas
  riguroso; refinamiento posterior si hace falta blindar el resultado.

## 7. Fuera de alcance / expectativas honestas

- El sample en vivo es chico (sectores con n<20). Es posible que el walk-forward
  diga que los spreads sectoriales NO son estables -> conclusion valida:
  "peso 1,0 para todos", el valor queda en label+calibracion. NO es un fracaso.
- Deep learning: descartado por tamano de muestra y relacion senal/ruido.

## 8. RESULTADOS DE EJECUCION (Fases 0-4, 2026-07-02)

Scripts: scripts/ml/walkforward_ml.py (Fase 2-3), scripts/ml/fase4_label_compare.py
(Fase 4). Artefactos crudos (gitignored): reportes/ml_walkforward/{fold_metrics.csv,
oos_predictions.parquet, fase4_label_compare.csv, run.log, fase4_run.log}.

### Fase 1 -- rebuild features_ml
Cuello real = features_sector (123t/desde-2025) y scoring_tecnico (124t/hasta-09/04),
NO los precios (200t/full). Cadena de recompute local (sin Yahoo): 03_calcular_scoring
(get_universo, 200t) -> 05 (features_sector, WHERE sector NOT IN Real Estate/Utilities
-> 9 sectores) -> 06 (features_ml). Resultado: features_ml 196t (200 - 4 de RE/Utilities),
169.511 filas, 2020-10 -> 2026-07, GANANCIA 48,6/52,1/47,3 (balanceado, antes 52/55/27).
Se crearon unique index faltantes en local (features_sector, features_ml).
NOTA (efecto en shared module): la exclusion RE/Utilities vive en src/indicators/
sector_features.py -> aplica tambien a futuras corridas del pipeline diario (4 tickers
sin z-scores sectoriales). Politica sostenida (N>=5), no un side effect accidental.

### Fase 2 -- walk-forward purgado (9 folds, holdout 126d, embargo 20d, start 2021-06)
COMPUERTA 1 = PASA. RF-global tiene edge OOS ESTABLE: LIFT@decil 1.34 en 9/9 folds,
ret@decil +5,22%, mayor lift en el bear 2022. Motor elegido = RF CALIBRADO (isotonica):
el RF crudo esta sobre-confiado en el extremo (top-bin obs 0,62 @ prob 0,92) y la
calibracion lo arregla (top-bin 0,92). La no-monotonia del MEDIO persiste (problema de
señal, no de calibracion -> se opera solo la cola alta). FLAG: el elastic-net lineal
casi empata al RF (AUC 0,594 vs 0,599) -> el MODELO no es la palanca; el RF solo gana
robustez en estres (fold 2 crash: rf ret -0,83 vs en -5,52).

### Fase 3 -- pesos sectoriales
COMPUERTA 2 = SIN PONDERADOR (peso=1,0 para todos). La evidencia EN VIVO (1 ventana)
que motivo el ponderador (sectores invertidos/nulos) era RUIDO: el walk-forward (9
folds) muestra que TODOS los sectores discriminan POSITIVO (+2,08 a +5,11), 8/9 estables.
La gradacion es en parte artefacto de VOLATILIDAD (spread en retorno crudo). Fittear 9
pesos sobre 9 folds = overfitting de señal ruidosa. Global +2,92 en el medio.

### Significancia del edge (respuesta a "es casi una moneda?")
En clasificacion GLOBAL si es señal debil (AUC 0,60). En la COLA que se opera, NO es
moneda y es robusto: decil alto 64,5% acierto vs 48,9% base (+15,7pp, t=11,35, 9/9),
excess vs mercado +3,75%/20d (t=10,44, 9/9), bootstrap (permuto prob) p~0,0000. Aguanta
el crash 2022 (excess +4,26% con mercado -2,68%). Es un edge REAL, FINO y consistente.

### Fase 4 -- label absoluto vs relativo-al-sector
El label NO mueve la aguja en PnL. excess_mkt ABS +4,00 (t=10,88, 9/9) vs REL +4,08
(t=7,61, 9/9) = EMPATE; ABS mas consistente. REL solo mejora su propio objetivo
(beat_sec 0,576 vs 0,561). CONFIRMA: ~+4% excess decil alto / AUC 0,60 es el TECHO de
las 53 features. La unica palanca restante = features NUEVAS (otro proyecto).
DECISION: label ABSOLUTO. Razon de fondo (no inercia): para long-only auto-gatea riesgo
(menos señales en mercado malo); el relativo compraria "el mejor de los que caen" en un
crash. Ademas empata en PnL y es mas consistente.

### CONFIG FINAL CONGELADA (para Fase 5 cuando se retome)
Motor = RF-global (construir_modelo('rf'), 53 features V3) + CalibratedClassifierCV
isotonica + label ABSOLUTO (ret_20d > +1%) + SIN ponderador sectorial. Entrenar sobre
features_ml 196t/2020-2026. Edge esperado: decil alto ~64% acierto, +4% excess/20d.

### Fase 5 -- PENDIENTE (decision del usuario: consolidar y pausar)
Valor del despliegue (NO es mejor edge, es): (1) cobertura 123 -> 196 tickers (73 mas
con ML propio); (2) probabilidades calibradas -> umbral COMPRA_FUERTE confiable. Toca
PRODUCCION (scanner que alimenta alertas_scanner + bot ML FT/Alpaca); al calibrar cambia
la distribucion de prob -> RECALIBRAR el umbral COMPRA_FUERTE. Requiere listar archivos
+ aprobacion antes de codear. Homologar con FT via src/strategies/ml_scanner.

## 8b. Insumo del scanner en vivo: features sectoriales congeladas (13/9/2026)

Encontrado al preparar la Fase 5. `feature_calculator._obtener_zscore_sectorial`
tomaba la ULTIMA fila de `features_sector` sin mirar la fecha, y esa tabla no la
actualizaba ningun paso diario: solo el script legacy 05, a mano (24/2, 30/3,
9-10/4 y 2/7 segun `log_ejecuciones`). El modelo desplegado recibio 11 de sus 53
features (z-scores y promedios del sector) con semanas de antiguedad durante toda
la vida de FT_ML_SCANNER_v1: del 23/4 al 2/7 con datos del 9/4, y desde el 2/7
con los del 1/7. Las otras 42 se calculan en vivo y estaban al dia.

Medido sobre la rueda 2026-09-11 (solo lectura, mismo codigo y mismo dato salvo
esas 11 columnas):
- correlacion ~0 entre el valor congelado y el de la rueda en los 6 z-scores;
- `ml_prob_ganancia` se mueve 0,065 en promedio (p90 0,18, max 0,37);
- 57 de 200 tickers cambian de nivel; COMPRA_FUERTE pasa de 7 a 9 con 3 en comun.

Consecuencias para este documento:
- La performance EN VIVO de la seccion 2.2 (y la de FT_ML_SCANNER_v1) midio el
  modelo con esta falla, no el modelo funcionando bien. Es valida como registro
  de lo que opero; no dice si con el dato fresco habria rendido mas o menos.
- El walk-forward (secciones 5-8) NO esta afectado: entrena y evalua sobre
  `features_ml`, que trae los z-scores de cada rueda.
- Arreglo (Etapa 3a): el Paso 2 recalcula `scoring_tecnico` y `features_sector`
  de las ultimas 10 ruedas, el scanner lee la fila de la rueda de la barra (NaN
  si falta) y `estado_pipeline` vigila la tabla. Registrado en `ft_cambios`.

## 8c. Fase 5 -- modelo v2 entrenado (Etapa 3c, 13/9/2026)

Script: `scripts/ml/entrenar_ml_v2.py`. Modulo: `src/ml/ml_v2.py`. Artefacto:
`models_ml_v2/rf_cal_global.joblib` (4,1 MB, FUERA de git) + `metadata.json` (EN
git: sha256, datos, holdout, compuerta, cortes y versiones). Config = la congelada
en la seccion 8.

Datos: `features_ml` reconstruida en la Etapa 3b (KLAC/CRWD ya en escala correcta),
175.425 filas con label, 196 tickers, 2020-10-15 -> 2026-08-13, base 0,493.

**Holdout** (train hasta 2026-01-13, embargo 20, evalua 2026-02-12 -> 2026-08-13,
24.696 filas, base 0,482, retorno 20d medio +2,01%):

| AUC | lift@decil | Acierto decil alto | Retorno 20d decil alto | Brier (constante) |
|---|---|---|---|---|
| 0,641 | 1,51 | 72,8% | +8,30% | 0,2345 (0,2497) |

Calibracion por decil: el alto da 0,68 de probabilidad y 0,73 observado; el medio
sigue algo sobreconfiado (d6: 0,50 contra 0,44). La compuerta (umbrales fijados en el
script antes de correrlo) pasa los 5 criterios.

**Cortes equivalentes** (16.856 filas posteriores al despliegue de la v1,
2026-04-13 -> 2026-08-13; mismas filas para las dos):

| Corte v1 | Filas arriba | Corte v2 | Acierto v1 | Acierto v2 |
|---|---|---|---|---|
| 0,75 | 2,1% | 0,769 | 72,1% | 80,5% |
| 0,65 | 8,1% | 0,607 | 64,1% | 73,1% |
| 0,55 | 29,7% | 0,557 | 60,6% | 62,8% |
| 0,45 | 61,2% | 0,471 | 55,9% | 56,7% |
| 0,35 | 88,9% | 0,364 | 51,9% | 52,3% |

Mismo tramo, informativo: v1 AUC 0,606 / lift 1,28 / retorno decil +5,24%; v2 AUC
0,631 / lift 1,45 / +8,26%.

Lo que NO dice:
- Es UNA ventana de seis meses con mercado alcista, no el walk-forward de 9 folds.
  Que la v2 supere a la v1 aca es consistente con mas cobertura (196 contra 123
  tickers entrenados) y calibracion, pero lo decide la Etapa 4 en forward testing.
- La v1 recibe aca las features sectoriales de cada rueda (vienen de `features_ml`);
  en vivo recibio las congeladas (seccion 8b). Por eso su fraccion >= 0,65 da 8,1%
  aca y 13,8% en `alertas_scanner`.
- Los cortes salen del modelo del holdout y se aplican al modelo final. Hay que
  comparar en vivo la cantidad de COMPRA_FUERTE de v1 y v2.

Servicio (Etapa 3d): el Paso 3 (`cron_diario --step scanner`) carga la v2 una vez,
validada contra la metadata, y por ticker calcula su probabilidad y un score
compuesto con las MISMAS senales de price action, score tecnico y bajistas que la
v1; solo cambian la probabilidad y los cortes (`alert_classifier.clasificar_alerta`,
parametro `cortes_ml`). Escribe `ml_prob_v2`, `ml_modelo_v2`, `alert_score_v2` y
`alert_nivel_v2` en la misma fila de `alertas_scanner`. Si el artefacto no carga, esas
columnas quedan NULL y la v1 corre sin cambios: es el control del experimento.

Cadencia: la v2 queda CONGELADA durante la Etapa 4 (reentrenarla seria un corte de
tramo). El RF tiene semilla fija: si el artefacto se pierde, regenerarlo con los
mismos datos y comparar su sha256 con la metadata.

## 9. Addendum -- validez predictiva del PCR (previo a sumar features de opciones)

Antes de invertir en features de opciones para intentar subir el techo (unica
palanca real segun seccion 8), se testeo si el PCR (put/call ratio) PREDICE el
retorno futuro. Script: scripts/ml/pcr_predictive_validity.py (reutilizable).

Metodo: Information Coefficient (Spearman cross-seccional por fecha) del PCR en t
vs retorno a 5/10/20 dias, por ventana (corto/medio/largo). Dos medidas:
- IC(nivel): rankea por el PCR crudo -> mezcla rasgos ESTATICOS del ticker.
- IC(dinamico): rankea por PCR - baseline del propio ticker (la DESVIACION = el
  verdadero "cambio de sentimiento", la señal tradeable).

RESULTADO (48 dias, 2026-04-18 -> 2026-07-01, UN regimen alcista):
- IC(nivel) FUERTE y consistente: pcr_oi ventana medio 10-20d IC +0.15/+0.17,
  breadth ~90%, CONTRARIAN (PCR alto -> retorno alto). pcr_oi >> pcr_vol
  (posicionamiento/OI predice mejor que flujo/volumen). Spread quintil sup-inf
  +3 a +6% a 20d.
- PERO IC(dinamico) se DERRUMBA a ~0 (incluso negativo en corto: -0.09).

VEREDICTO: la aparente señal NO es "el sentimiento institucional se cumplio". Es
un ARTEFACTO ESTATICO cross-seccional de un regimen (nombres que estructuralmente
tienen PCR alto rindieron mejor en este mercado alcista). El CAMBIO de
posicionamiento -- lo que realmente seria sentimiento -- no predice. Mismo patron
que los spreads sectoriales "en vivo" (ruido de una ventana).

DECISION: PREMATURO sumar PCR como feature del ML (inyectaria un factor fragil de
un solo regimen -- justo la trampa a evitar). LIMITE DURO: opciones_snapshot
arranca 2026-04-18 (snapshot nuevo, sin mas historia local). RE-CORRER el test con
2+ regimenes de historia (crece dia a dia, ~6-12 meses). Mientras tanto las
opciones se usan donde NO requieren validacion predictiva: muros como S/R,
expected move, tablero de sintesis (descriptivo/riesgo).

META-APRENDIZAJE (cierra el loop): tanto el techo del ML (seccion 8) como el
bloqueo de las features de opciones vienen de LO MISMO -- falta de informacion
across regimenes. Hoy no hay con que subir el edge predictivo; el valor esta en
el ensamble de señales finas + usos descriptivos, y en acumular datos con paciencia.

## 10. Addendum -- leakage en las features de market structure (17/9/2026)

Detalle, metodo y plan: docs/estructura_velas.md (secciones 4 y 10).

`market_structure._calcular_estructura_n` detecta swings con una ventana centrada
(`rolling(2N+1, center=True)`) y los registra en su propia barra, cuando recien se
conocen N barras despues. Las 24 features de estructura de `features_market_structure`
(la tabla que JOINean `trainer_v3`, `walkforward_ml.cargar_dataset` y
`entrenar_ml_v2`) usan informacion de las 10 ruedas siguientes. En vivo el scanner
calcula sobre las barras hasta la rueda: ahi no hay futuro.

Medido sobre 44.663 filas (60 tickers, 2023-08 -> 2026-08), mismas filas y mismas 29
features no estructurales, cambiando solo las 24 de estructura:

| Tramo | v1 guardada | v1 real | v2 guardada | v2 real |
|---|---|---|---|---|
| 2023-08 -> 2025-01 (fuera del entrenamiento v1) | 0,655 | 0,520 | 0,650 | 0,518 |
| 2025-01 -> 2026-04 (v1 entreno aca) | 0,635 | 0,516 | 0,657 | 0,521 |
| 2026-04 -> 2026-08 (fuera del entrenamiento v1) | 0,645 | 0,516 | 0,667 | 0,525 |

Lo que invalida de este documento:
- Sec. 2.3: la concentracion en `dias_sh_10` / `dias_sl_10` no es "fragilidad de
  timing": son las columnas que traen el futuro (swing high hace 0 dias = techo que
  despues cayo).
- Sec. 8, Fases 2-4: el edge del walk-forward (AUC 0,60, decil alto 64,5%, t=11,35,
  "techo de las 53 features"), la eleccion de motor, el rechazo del ponderador y la
  comparacion de labels se midieron con esas features. Las conclusiones pueden
  sostenerse o no; los numeros no valen.
- Sec. 8c: el holdout de la v2 (AUC 0,641, decil 72,8%), su compuerta, los cortes
  equivalentes y la comparacion v1/v2 del mismo tramo.

Lo que sigue valiendo: la performance EN VIVO (sec. 2.2, con la salvedad de 8b) y el
forward testing, porque en vivo las features no tienen futuro. Con las features reales
queda una ventaja chica: decil alto de la v1 55,7% de acierto contra 51,2% de base.

Consecuencia: ningun reentrenamiento ni validacion sobre `features_market_structure`
hasta tener la historia con swings confirmados (docs/estructura_velas.md, Fase 2).
El modelo v3 (Fase 3) audita las 53 features con el test de invariancia y fija la
particion por fecha con embargo antes de correr (hoy `features_ml.segmento` parte por
ticker y los segmentos se pisan en fechas; lo usa `trainer_v3`).

### 10.1 El intento de v3 CORRIDO y su resultado (17/9/2026)

Hecho: `scripts/ml/entrenar_ml_v3.py` con la particion y las compuertas
pre-registradas en docs/estructura_velas.md sec. 9.5, sobre `features_ml` JOIN
`features_estructura` (swings confirmados, invariantes). Resultado completo en la
seccion 9.6 de ese doc.

| Brazo (6 folds purgados en el 80% de desarrollo) | AUC media | Folds > 0,52 |
|---|---|---|
| 53 features, estructura que mira al futuro (control) | 0,6189 | 6/6 |
| 53 features, estructura CONFIRMADA | 0,5133 | 3/6 |
| 29 features, sin estructura (ablacion) | 0,5099 | 3/6 |

**No pasa la compuerta** (pedia AUC media >= 0,54 y > 0,52 en 5 de 6 folds). Los dos
folds mas recientes, los de mas datos, caen por debajo de 0,50. El lockbox
(2025-07-15 -> 2026-08-13) quedo SIN ABRIR, disponible para una hipotesis nueva.

Que agrega esto a la seccion 10: no es solo que los NUMEROS de las secciones 2.3, 8 y
8c no valgan; es que **la ventaja que median no existe cuando las features son
honestas**. El motor (RF global), el label absoluto y las 53 features, que es la
configuracion que este documento congelo, dan AUC ~0,51 sobre 6 ventanas
independientes. Antes de volver a entrenar hay que cambiar la HIPOTESIS, no los
umbrales: label relativo al universo del dia en vez de absoluto (sec. 2.4), horizonte
mas corto que 20 ruedas, o abandonar la prediccion a plazo fijo y quedarse con las
estrategias de reglas con salidas, que es lo unico que paso su backtest
(docs/estructura_velas.md sec. 9.3).

Las features NO estructurales quedaron auditadas numericamente en el paso 0
(`scripts/ml/auditar_invariancia_features.py`): invariancia exacta (diferencia maxima
0,00e+00 en 12 tickers x 8 cortes) y **skew de ventana cero** -- el dataset sale de la
historia completa y el scanner recalcula con las ultimas 500 barras, y los indicadores
recursivos ya convergieron. No hay train/serve skew por ventana.

### 10.2 Por que da 0,51: que aporta, que no, y el label (17/9/2026)

Detalle completo, tablas y reglas: [features_ml.md](features_ml.md). Reproducible con
`scripts/ml/analizar_features_ml.py`. Lo que cambia respecto de este documento:

**1. Ninguna familia que describa al ticker aporta (ablacion, mismos 6 folds).** Quitando
cada familia del set de 53: indicadores -0,0018 (sin ellos MEJORA), engineered -0,0007,
flags de estructura -0,0000, scoring +0,0002, distancias a SMAs +0,0006, estructura
+0,0035, z-scores sectoriales +0,0037. La unica que mueve algo son las 5 de CONTEXTO
sectorial (+0,0126; sin ellas el modelo da 0,5007) -- y esas valen lo mismo para todos
los tickers de un sector en una fecha: son regimen, no seleccion. Ni siquiera son
consistentes: la base les gana en 4 de 6 folds.

Esto corrige la lectura de la sec. 2.3 ("el modelo se apoya en dos contadores de
timing"): con features honestas no se apoya en nada que distinga a un ticker de otro.

**2. El label mide el regimen.** Base rate del label absoluto por trimestre: 0,302 (2023Q3)
a 0,694 (2025Q2). Juzgando la MISMA probabilidad con un label relativo (retorno por
encima de la mediana del universo de la rueda), el AUC cae de 0,5133 a 0,5007 y el rango
entre folds se comprime de 0,468-0,565 a 0,469-0,522: lo que parecia senal en los folds
1-2 era regimen. El label relativo no crea senal, pero mide con la mitad del ruido
(IC95 +-0,021 contra +-0,043). Un modelo ENTRENADO con label relativo todavia no se
probo.

**3. Los folds no son comparables.** Folds 1-5 con 122 tickers, fold 6 con 196: los 74-76
incorporados en 2024-04 aparecen en el ultimo holdout casi sin haber estado en el train.
El lockbox tiene el mismo problema. El techo lo pone `precios_diarios`: 19 tickers
desde 2020-01, 122 desde 2021-01, 200 recien desde 2024-04. Hay que pre-registrar el
UNIVERSO ademas de las fechas.

**4. La valuacion fundamental no aporta a esta pregunta.** `fundamentales_sec_multiplos_d`
es point-in-time correcto (0 filas con un balance publicado despues de la fecha), pero el
PER cubre el 58% de las filas y su ausencia es un proxy de region y sector (27,7% son
tickers sin SEC, 9,9% empresas con perdida). PER, earnings yield, P/S y FCF yield, crudos
o como percentil transversal: AUC 0,49-0,51, IC95 incluye 0,50. El unico indicio
(percentil transversal de EV/EBITDA, 6/6 folds) va en contra del valor -- lo caro subio
mas -- sobre un 34% de filas sesgado por sector, y con 24 comparaciones un 6/6 aparece
por azar el ~31% de las veces.

**Implicancia para este documento.** La "config final congelada" de la sec. 8 (RF global,
53 features, label absoluto) queda sin respaldo como punto de partida. Un modelo nuevo
empieza por el label (relativo al universo del dia) y el universo (pre-registrado), y
recien despues por las features: las familias sin usar con cobertura completa son flujo
de volumen, sorpresa de volumen y eventos de balance (features_ml.md sec. 11).
