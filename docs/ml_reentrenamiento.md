# Reentrenamiento ML -- diagnostico, decisiones y esquema (Tarea 20)

Estado: FASES 0-4 EJECUTADAS (2026-07-02). PAUSADO antes de Fase 5 (despliegue),
por decision del usuario. Config final decidida (ver seccion 8). El modelo V3
actual sigue en produccion sin cambios.
Estado operativo vivo: AGENDA.md Tarea 20. Resultados crudos: reportes/ml_walkforward/.

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

### 2.5 Los modelos sectoriales estan rechazados por evidencia
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
