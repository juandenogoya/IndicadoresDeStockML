# CARTERAS_PERFIL -- Documentacion de Estrategia (pre-registro)

**Estado**: **CERRADA (15/10/2026), sin FT.** Decision del usuario: lo analizado no
es mejor que tirar una moneda. Resumen abajo; el detalle de cada etapa sigue despues.
**Instancias previstas**: 4 (`FT_CARTERA_CONSERVADORA`, `FT_CARTERA_MODERADA`,
`FT_CARTERA_ARRIESGADA`, `FT_CARTERA_ESPECULATIVA`). Ids a asignar en el alta.
**Logica base**: seleccion por familias de las estrategias FT activas + salidas
mecanicas por perfil de riesgo.
**Insumos**: candidatos de las estrategias FT (`ft_candidatos_diarios`), perfil de
riesgo del ticker ([docs/perfiles_carteras.md](../../perfiles_carteras.md)), close
de `precios_diarios`, filtro de earnings transversal.
**Rama**: `feature/carteras-perfil`

> Todo lo que esta en las secciones "Definiciones", "Control", "Backtest" y
> "Regla de lectura" quedo fijado ANTES de correr el backtest. Ningun parametro
> se mueve despues de ver resultados; si algo cambia, se escribe como cambio, con
> fecha y motivo, en "Cambios al pre-registro".

---

## RESUMEN DE CIERRE (15/10/2026)

### 1. Hipotesis

**Tomar los tickers que marcan las estrategias de FT (top 3 por familia), repartirlos
en 4 carteras segun su perfil de riesgo y darle a cada una salidas acordes a su
volatilidad da mejores resultados que elegir al azar dentro del mismo perfil.**
Despues, como la seleccion no paso, una segunda pregunta (diagnostico): **el
problema esta en la salida?** -- se variaron solo las salidas con la seleccion al
azar fija.

### 2. Condiciones de las carteras (iguales en las 4, salvo SL/TP)

| | Condicion |
|---|---|
| Elegibilidad | Perfil de riesgo vigente del ticker (recalculado mes a mes) + estar en el top 3 de una familia: **Tecnica** (score >= 4,0, maximo 1 por sector) o **SMC v3** (estructura alcista confirmada). ML solo en FT, no en el backtest. Empates por sorteo, nunca por alfabeto |
| Apertura | Lugar libre (maximo 6), 15% del capital actual, al cierre de la rueda; no si ya esta en cartera, si salio esa rueda o si tiene balance hoy o en la siguiente. Prioridad: mas familias > mejor puesto > sorteo |
| Cierre (V0) | Sobre el cierre: balance (rueda anterior) > stop loss > take profit. Sin time stop |
| $100.000 por cartera | Conservadora SL 7% / TP 12% · Moderada 7% / 15% · Arriesgada 15% / 30% · Especulativa 20% / 50% |

Backtest 2021-09-01 -> 2026-09-30 (1.274 ruedas), 20 semillas por lado, sin costos.
Control: la misma cartera eligiendo al azar dentro del perfil. Validaciones previas:
perfil recalculado 200/200 contra el snapshot guardado; elegibles SMC 58/58 contra
lo que registro FT.

### 3. Resultados (medianas de 20 semillas)

**Etapa 1 -- seleccion contra azar (salida V0):**

| Cartera | Seleccion | Azar | Mantener el perfil | Lectura |
|---|---|---|---|---|
| Conservadora | +32,6% | +28,9% | +64% | no pasa (3/6 anios) |
| Moderada | +41,3% | +60,3% | +76% | no pasa (3/6) |
| Arriesgada | +86,5% | +76,8% | +89% | no pasa (3/6) |
| Especulativa | +125,5% | +61,7% | +60% | pasa al minimo (4/6), lo que el azar da ~1 de cada 3 veces |

**Etapa 2 -- salidas contra V0, seleccion al azar fija** (V1 sin salida por balance,
V2 sin SL/TP, V3 sin TP, V4 mantener 60 ruedas sin balance ni SL/TP):

| Cartera | V0 | V1 | V2 | V3 | V4 | Lectura |
|---|---|---|---|---|---|---|
| Conservadora | +28,9% | +38,4% | +28,2% | +27,6% | +43,6% | ninguna mejora (V4 frenada por la cola por operacion) |
| Arriesgada | +76,8% | +63,4% | +69,3% | +74,9% | +63,9% | ninguna mejora; V0 es la mejor |
| Especulativa | +61,7% | +35,9% | +99,4% | +70,3% | +78,0% | ninguna mejora (V2 frenada por la cola por operacion) |

### 4. Conclusiones

1. **La seleccion por familias no es mejor que el azar.** En 3 de 4 carteras no se
   distingue; la cuarta pasa por un margen que el azar da seguido. Coincide con todo
   lo medido antes en el proyecto (reglas tecnicas, ML, senales SMC: ninguna elige
   mejores acciones que el universo).
2. **En los perfiles tranquilos, no hacer nada gana.** Conservadora y Moderada:
   mantener todo el perfil le gana a toda mecanica probada.
3. **El SL y el TP no son el problema; la salida por balance depende del perfil.**
   Sacarlos casi no cambia nada. Salir antes del balance cuesta en la Conservadora y
   protege en Arriesgada y Especulativa.
4. **El stop protege la operacion y le cuesta a la cartera.** Las dos mejores
   variantes del estudio (Conservadora V4, Especulativa V2) mejoran el retorno sin
   empeorar el drawdown de la cartera, y caen solo por el criterio de cola por
   operacion fijado antes. Queda como hipotesis sin confirmar; no se aprueba
   cambiando el criterio despues de ver el resultado.
5. **Decision: se cierra, sin FT.** Los costos no cambiarian la conclusion: lo
   analizado no supera al azar.

**Lo que queda util para otros trabajos:** el motor (`src/utils/carteras_perfil.py`,
18 tests) y el script (`bt_carteras_perfil.py`, con `--validar-perfil`,
`--validar-smc` y `--salidas`); el hallazgo de que `volatilidad_mtf` ancla la barra
en curso al reloj (no point-in-time); el comportamiento de `earnings_filter` con
fechas de balance vencidas (tarea aparte); y la vara de comparar siempre contra el
azar con semillas y contra mantener el perfil.

**Como se reabriria:** con una hipotesis NUEVA y su pre-registro, probada sobre
datos que no se usaron aca (FT hacia adelante); no ajustando esta.

---

## Pregunta

**La seleccion por familias de las estrategias FT, repartida por perfil de riesgo,
elige mejores tickers que el azar dentro del mismo perfil, con las mismas salidas?**

Las salidas son MECANICAS (porcentaje sobre el precio de entrada + earnings) y no
dependen de ninguna senal: lo unico que se pone a prueba es la SELECCION. Si la
seleccion no aporta, una cartera elegida al azar dentro del perfil rinde lo mismo.

Origen: la "foto al cierre" (docs/telegram_foto.md) ya arma el top 3 por familia
todos los dias como informacion. Esta estrategia lo convierte en una regla de
compra, separa los tickers por perfil y les pone salidas acordes a su volatilidad.

---

## Definiciones

### Seleccion (cada rueda) -- por la REGLA DE ENTRADA de cada familia

La seleccion NO lee `ft_candidatos_diarios`. Esa tabla solo registra tickers que la
estrategia todavia NO tiene en cartera (medido: 0 candidatos ya abiertos), asi que
heredarla ataria las carteras a las posiciones de otras 10 estrategias y haria el
backtest irreproducible. Cada rueda se aplica la regla de entrada de cada familia a
TODO el universo, sin estado:

| Familia | Califica si | Ranking | Top |
|---|---|---|---|
| **Tecnica** | `calcular_score_tecnico` >= 4,0 (`src/strategies/scoring.py`: close > SMA200 obligatorio; SMA50 2,0 / SMA21 1,0 / MACD 1,5 / RSI 45-68 1,0) | score; empates por SORTEO | 3, **como maximo 1 por sector** |
| **SMC** | regla de entrada de FT_SMC_v3 sobre `features_estructura` (N5 **o** N3): CHoCH/BOS bull confirmado en el lookback, estructura >= 0, sin CHoCH bear, vela alcista, distancia al swing low entre 1% y 8%. Se reusa `ft_scoring.calcular_score_estructura` via `ft_scoring_estructura`, no se reimplementa | score de calidad 0-3 (el maximo entre N5 y N3); empates por SORTEO | 3 |
| **ML** | `alertas_scanner` en COMPRA_FUERTE con score >= minimo de FT_ML_SCANNER (v1 o v2) | score (el maximo entre v1 y v2) | 3 |

- **TECH y TECH_SECTOR son UNA familia** ("Tecnica"): las dos califican con el mismo
  score >= 4,0; en FT solo difieren en el manejo de cartera (cupo por sector,
  rotacion). Con dos familias, todo ticker tecnico sumaria "2 familias" de prioridad
  sin dos opiniones distintas detras. El "maximo 1 por sector" toma la idea de
  diversificacion de TECH_SECTOR (decision del usuario, 15/10/2026).
- **Desempate por SORTEO, nunca por alfabeto.** En FT el alfabeto decide:
  FT_TECH_v1 compro 77 tickers A-L y **0 M-Z** en 5 meses con candidatos parejos
  (2.680 / 2.642); FT_TECH_SECTOR_v1 548 / 339. Medido 2021-09 -> 2026-10 (1.284
  ruedas): mediana de **22 tickers empatados en el score maximo 5,5** por rueda; en
  el **90%** de las ruedas el top 3 tecnico lo decide el desempate. Por eso la
  seleccion se corre con 20 semillas (ver Control).
- Tickers unicos: un ticker en el top de 2 familias es UN candidato (2 familias
  para la prioridad).
- Cada ticker va a la cartera de SU perfil vigente en esa rueda.

**ML entra como una familia mas (decision 14/10/2026, opcion A)**. Ver
"Limitaciones": el backtest no puede reconstruir ML.

### Cartera

| Parametro | Valor |
|---|---|
| Capital inicial | $100.000 por cartera |
| Tamano de cada compra | 15% del capital ACTUAL (equity marcada a mercado), acciones enteras: `qty = int(0,15 * equity / close)` |
| Maximo de posiciones | 6 |
| Lugar libre | Hasta no vender, no entra otro ticker |
| Cash insuficiente | Si el cash no alcanza para una compra completa, no entra (no se compra fraccion) |
| Tope de exposicion | El de la estrategia: 6 x 15% = 90%. NO aplica el MAX_EXPOSICION 75% generico de FT |
| Ticker ya en cartera | Se ignora (no se promedia) |
| Precio de entrada | Close de la rueda de la senal (convencion FT: decide y ejecuta con el ultimo cierre) |

**Prioridad** cuando hay mas candidatos del perfil que lugares libres:
1. el que esta en MAS familias;
2. el de mejor puesto dentro de su familia (1ro, 2do, 3ro; si esta en varias, el
   mejor). En Tecnica y SMC el puesto ya salio del sorteo cuando hay empate;
3. empate: sorteo con la misma semilla de la corrida (nunca el alfabeto).

### Salidas (todas sobre el CLOSE)

| Cartera | Stop loss | Take profit | SL en ATRs* | TP en ATRs* |
|---|---|---|---|---|
| Conservadora | -7% | +12% | 2,9 | 5,0 |
| Moderada | -7% | +15% | 2,4 | 5,2 |
| Arriesgada | -15% | +30% | 4,0 | 7,9 |
| Especulativa | -20% | +50% | 3,4 | 8,4 |

\* ATR% diario mediano del perfil en el snapshot del 2026-08-06 (Conservadora 2,40%,
Moderada 2,89%, Arriesgada 3,79%, Especulativa 5,93%). Informativo.

- SL: close <= entrada x (1 - SL). TP: close >= entrada x (1 + TP). Se ejecuta al
  close de esa rueda (un gap puede llevar la perdida mas alla del SL: se registra el
  precio real, no el teorico).
- **Earnings (transversal, prioridad maxima)**: la regla de todas las estrategias FT
  (`src/indicators/earnings_filter.py`): sale la rueda anterior al balance
  (`EARNINGS_MANANA`) y no entra si el balance cae dentro del buffer
  (`DIAS_BUFFER_ENTRADA`, 1 rueda habil). Sin dato de earnings no cierra (fail-safe).
- **NO se usan las salidas propias de cada familia** (nivel ML degradado, score <=
  3,5, trailing/CHoCH/estructura de SMC, muros de OI). La seleccion entra por las
  familias; la salida es solo de la cartera.
- **Sin time stop.** Se MIDE el tiempo en posicion (mediana, p90, posiciones que
  quedan abiertas al final) y se informa; no se corta.
- **Reentrada**: un ticker vendido puede volver a entrar desde la rueda SIGUIENTE si
  vuelve a ser candidato. La misma rueda no.
- **Cambio de perfil** (snapshot mensual): la posicion abierta se queda en su cartera
  con el SL/TP con que entro. El perfil nuevo solo rige para compras nuevas.

Decision sobre Moderada (14/10/2026): SL 7%, igual que Conservadora, aunque en ATRs
queda el stop mas ajustado de las cuatro (2,4). Se eligio sabiendo eso y antes de
ver ningun resultado.

---

## Control

0. **La SELECCION tambien corre con 20 semillas** (las de sus sorteos de empate).
   Como el top 3 tecnico lo decide el sorteo el 90% de las ruedas, una sola corrida
   puede mostrar una "diferencia" que es la semilla. Se compara BANDA contra BANDA.
1. **Azar dentro del perfil (principal).** La MISMA cartera -- mismo capital, tamano,
   maximo de 6, mismas salidas y earnings -- pero cada compra elige al AZAR entre los
   tickers del perfil que pasan el filtro de earnings y no estan en cartera, con la
   misma CANTIDAD de compras que la seleccion en esa rueda. 20 semillas fijas.
   Mide el piso de ruido (la leccion de ANALISIS_ENTRADAS.md: la vara se mide, no se
   supone).
2. **Universo equal-weight del perfil.** Retorno del perfil comprado y mantenido, y
   ajustado por exposicion media (retorno / exposicion), como en los backtests SMC.

Mismo control en el backtest y en FT.

---

## Backtest

| Item | Valor |
|---|---|
| Periodo | 2021-09 -> 2026-09 (el mismo de SMC y de entradas) |
| Tramos anuales | 6 (el ultimo parcial; se informa su largo) |
| Familias reconstruibles | Tecnica y SMC (v3, N5 y N3). ML no |
| Seleccion en backtest | Top 3 por familia de esas 2 -> hasta 6 tickers por rueda |
| Perfil | Recalculado MES A MES con el motor puro (`perfil_metricas` + `perfil_riesgo`) sobre el precio disponible a esa fecha. `perfiles_ticker` tiene un solo snapshot (2026-08-06) y no sirve para el pasado |
| Earnings | `earnings_historico` (fechas de anuncio). Se corta donde la tabla deja de estar completa (`--status`), como en el analisis de salidas SMC |
| Costos | Sin costos. Se informa la cantidad de operaciones por anio y por cartera |
| Maquina | Reusar `ft_backtesting_runner.py` donde se pueda (fidelidad por construccion) |

Metricas por cartera y por tramo: retorno total, retorno / exposicion media, max
drawdown, Sortino, operaciones, % aciertos, expectancy, profit factor, motivos de
salida (SL / TP / EARNINGS / abierta al final), tiempo en posicion (mediana y p90),
lugares ocupados promedio y ruedas con la cartera llena.

---

## Regla de lectura (por cartera)

**PASA** si las dos:
1. la MEDIANA de las 20 corridas de la seleccion gana plata en el total del
   periodo, **y**
2. la MEDIANA de la seleccion le gana a la MEDIANA de las 20 carteras al azar en
   **al menos 4 de los 6 tramos anuales**.

Se lee cada cartera por separado: pueden pasar unas y no otras. Una que no pasa no
va a FT. Ademas se informa (no decide): la banda completa (min, p25, mediana, p75,
max) de las dos, cuanto se solapan, y la comparacion contra el universo
equal-weight del perfil.

---

## Medicion previa (14/10/2026, solo lectura)

OJO: esta medicion se hizo con la definicion ORIGINAL (top 3 de 4 familias leido de
`ft_candidatos_diarios`), antes del cambio del 15/10. Sirve como orden de magnitud
del flujo; con la regla de entrada el universo elegible es mas ancho (mediana de 46
tickers con score tecnico >= 4 por rueda), asi que el flujo no puede ser menor.

Sobre las 97 corridas de FT (2026-04-28 -> 2026-10-14), estrategias activas, perfil
del snapshot 2026-08-06 para todo el periodo:

| Cartera | Con ML: mediana por rueda | Con ML: ruedas en 0 | Sin ML: mediana | Sin ML: ruedas en 0 |
|---|---|---|---|---|
| Conservadora | 2 | 4% | 2 | 15% |
| Moderada | 3 | 2% | 2 | 10% |
| Arriesgada | 2 | 13% | 2 | 16% |
| Especulativa | 1 | 28% | 1 | 35% |

Ninguna cartera queda sin flujo de candidatos: con 6 lugares y "no entra hasta
vender", el ritmo lo van a marcar las SALIDAS. Universo del snapshot: Moderada 57,
Arriesgada 53, Conservadora 46, Especulativa 44.

---

## Limitaciones (escritas antes de correr)

- **El backtest NO mide lo mismo que FT.** El backtest valida Tecnica + SMC v3. FT
  corre con las 3 familias (Tecnica, SMC, ML):
  - **ML no entra al backtest**: sus modelos se entrenaron con esos anios (y con la
    tabla de estructura que mira al futuro); su "eleccion" en 2023 conoceria el
    resultado. Su aporte solo se mide hacia adelante, en FT, y **no se va a poder
    separar del resto** (opcion A elegida a sabiendas; la B -- 4 carteras con ML y 4
    sin -- quedo descartada por simplicidad).
  - Los filtros de opciones de TECH_SECTOR OPTIONS / OIEXIT no entran en ningun
    lado: la familia Tecnica es solo el score (opciones existe desde 2026-04-18).
  - **SMC solo v3**: SMC_v1 lee `features_market_structure`, que mira al futuro.
- El perfil del backtest se recalcula; el de FT sale del snapshot mensual vigente.
- La seleccion usa las REGLAS de entrada compartidas (`src/strategies/scoring.py`,
  `ft_scoring`, el scanner ML): un cambio en esas reglas cambia estas carteras. Va
  a `ft_cambios` (afecta a las 4). Los cambios de posiciones de otras estrategias
  NO las afectan.
- Sin costos.

---

## Cambios al pre-registro

Todos ANTES de correr el backtest (ningun resultado visto):

| Fecha | Cambio | Motivo |
|---|---|---|
| 15/10/2026 | La seleccion sale de la REGLA DE ENTRADA de cada familia, no de `ft_candidatos_diarios` | Esa tabla excluye lo que cada estrategia ya tiene: ataba las carteras a otras 10 carteras y no se podia reconstruir |
| 15/10/2026 | TECH + TECH_SECTOR = una familia "Tecnica", top 3 con maximo 1 por sector. Quedan 3 familias (hasta 9 tickers en FT, 6 en backtest) | Misma regla de entrada; el "1 por sector" toma la diversificacion de TECH_SECTOR |
| 15/10/2026 | La seleccion corre con 20 semillas y la regla de lectura compara MEDIANA contra MEDIANA | El top 3 tecnico lo decide el sorteo el 90% de las ruedas |

---

## Validaciones previas al backtest (15/10/2026)

Motor: `src/utils/carteras_perfil.py` (13 tests). Script:
`scripts/backtesting_historico/bt_carteras_perfil.py --validar-perfil | --validar-smc`.

**1. Perfil point-in-time: 200/200**, diferencia de score 0,000000 contra el
snapshot 2026-08-06 de `perfiles_ticker`. Dos cosas que hubo que resolver para
llegar ahi:
- `volatilidad_mtf.resample_ohlc` excluye la "barra en curso" con el RELOJ
  (`pd.Timestamp.today()`), no con la fecha de los datos. Recalculando hoy el perfil
  del 6/8, la barra parcial de agosto entraba y 7 tickers cambiaban de caja (CVS y
  EOG ~10 puntos de score). En produccion no es un error (el perfil se calcula el
  mismo dia), pero no es point-in-time: el backtest ancla la barra en curso a la
  fecha de la rueda (`atr_pit`). Misma familia que `scan_fecha`.
- El universo del backtest son TODOS los tickers de `activos`, tambien los dados de
  baja: con solo los vivos hoy faltaba WBD (baja 14/10/2026) y cambiaba un
  percentil (199/200). Usar los vivos hoy es sesgo de supervivencia.
- El benchmark de la beta (futuros ES) hoy arranca en 2020-01-02, no en 2025-05-09
  como dice el docstring de compute_perfiles_carteras: la beta existe en todo el
  periodo.

**2. Elegibles SMC contra FT_SMC_v3_N5/N3 (corridas desde 2026-10-01): los 58
candidatos que registro FT se reconstruyen todos (0 faltan).** De 14 elegibles
reconstruidos que FT no registro, 13 se explican porque la estrategia ya lo tenia en
cartera o tenia balance cerca, y el restante (WFC, corrida 15/10) porque
`earnings_calendar` seguia mostrando el balance del 13/10 -- ya ocurrido -- y
`earnings_filter` toma una fecha pasada como balance inminente (bloquea hasta el
refresh semanal). El backtest usa las fechas reales de `earnings_historico`: es una
diferencia DECLARADA con FT (en FT, despues de cada balance el ticker queda
bloqueado unos dias). Para fechar cada corrida hubo que deducir su rueda de datos
del precio registrado (FT asincronico: la corrida del dia f decide con el ultimo
cierre que habia, en general el de f-1).

## Resultados del backtest (15/10/2026)

2021-09-01 -> 2026-09-30, 1.274 ruedas, 20 semillas por lado, sin costos, sin ML.
`earnings_historico` completa en todo el periodo (sin corte). Detalle:
`reportes/carteras_perfil/20261015_1142/`.

| Cartera | Lectura | Tramos ganados al azar | Retorno total sel (mediana, p25-p75) | Azar (mediana, p25-p75) | P(azar >= sel) |
|---|---|---|---|---|---|
| Conservadora 7/12 | NO PASA | 3/6 | +32,6% (22,8 / 39,8) | +28,9% (19,5 / 33,7) | 42% |
| Moderada 7/15 | NO PASA | 3/6 | +41,3% (37,2 / 57,9) | +60,3% (46,0 / 73,5) | 77% |
| Arriesgada 15/30 | NO PASA | 3/6 | +86,5% (55,4 / 102,6) | +76,8% (60,5 / 90,9) | 45% |
| Especulativa 20/50 | **PASA** | **4/6** | +125,5% (83,7 / 139,9) | +61,7% (19,6 / 74,7) | 28% |

Por tramo (mediana seleccion / mediana azar / universo equal-weight del perfil):

| Anio | Conservadora | Moderada | Arriesgada | Especulativa |
|---|---|---|---|---|
| 2021* | 4,1 / 6,7 / 7,0 | **4,7** / 2,7 / 4,9 | -1,9 / -0,9 / -0,2 | **-4,2** / -4,6 / -13,8 |
| 2022 | **-2,9** / -3,8 / -0,1 | -19,1 / -8,6 / -12,3 | **-18,3** / -21,6 / -20,5 | **-27,8** / -35,4 / -39,9 |
| 2023 | **5,2** / 0,4 / 7,5 | 12,4 / 15,1 / 15,9 | **49,7** / 43,5 / 38,6 | 31,3 / 41,7 / 52,5 |
| 2024 | 9,3 / 9,9 / 14,1 | **22,2** / 18,3 / 19,4 | **17,7** / 11,5 / 19,3 | **1,6** / -2,3 / 5,9 |
| 2025 | 7,1 / 7,9 / 12,3 | **25,1** / 20,6 / 25,2 | 24,5 / 25,8 / 27,9 | **95,4** / 59,9 / 71,0 |
| 2026* | **6,5** / 5,7 / 11,6 | -1,8 / 7,3 / 10,2 | 2,6 / 10,6 / 12,6 | 12,1 / 14,7 / 11,9 |

\* parciales (desde septiembre / hasta el 30 de septiembre). En negrita, tramo ganado
al azar.

Operacion (mediana de la seleccion): 44 a 64 operaciones por anio por cartera;
exposicion media 75-86%; 26-37 dias en posicion (p90 73-83); cartera llena el 85-89%
de las ruedas (Especulativa 63%).

### Que dice

1. **Por la regla pre-registrada, pasa una de cuatro: Especulativa** (4/6 tramos, el
   minimo exigido; mediana +125,5% contra +61,7%; tambien mejor drawdown, Sortino y
   profit factor que el azar). Las otras tres no van a FT.
2. **Una de cuatro es lo que se espera por azar.** Con 6 tramos, ganarle a la
   mediana del azar en 4 o mas por casualidad tiene probabilidad 22/64 = 34% por
   cartera; con 4 carteras, ~81% de que al menos una lo logre (cota gruesa: las
   carteras comparten mercado y no son independientes). En las bandas: el azar le
   gana a la seleccion de Especulativa en el 28% de los pares de semillas, y su
   ventaja sale casi toda de 2025 (+95% contra +60%). Pasa, pero es evidencia
   debil: FT es lo que decide.
3. **Las salidas que manda el BALANCE, no el SL/TP del perfil.** De ~220-320
   operaciones por cartera, 167-185 salen por `EARNINGS_MANANA`; stop loss 42-106 y
   take profit 12-37. Sin time stop, la posicion se queda hasta el proximo balance
   (~un trimestre): la diferencia entre carteras la pone sobre todo QUE perfil de
   tickers tienen, no sus SL/TP. Es un hallazgo de diseno, no un parametro a mover
   ahora (cualquier cambio es un pre-registro nuevo).
4. **El universo equal-weight del perfil le gana a la seleccion y al azar en la
   mayoria de los tramos** (sin ajustar por exposicion: el EW esta 100% invertido y
   las carteras 75-86%, y el EW no sale por balances).

### Que NO dice

- Que la seleccion de Especulativa funcione: dice que paso una vara que una de
  cuatro carteras pasa por azar con alta probabilidad.
- Nada sobre ML: no esta en el backtest.
- Nada con costos (44-64 operaciones por anio por cartera).

---

## Cierre (15/10/2026)

**Decision del usuario: no va a FT por ahora.** "Respecto al rendimiento de nuestra
eleccion, no es mejor que el azar". Especulativa paso la vara formal, pero por un
margen que el azar da con alta probabilidad (ver "Que dice", punto 2). Queda en
pausa para pensar DONDE esta el problema: en que ticker se elige, en el momento de
entrada o en la salida.

**Descomposicion con los numeros del mismo backtest** (retorno acumulado
aproximado, encadenando los tramos de la tabla; medianas):

| Cartera | Universo EW del perfil | Azar (mismas salidas) | Seleccion | Mecanica de cartera = azar - EW | Seleccion = sel - azar |
|---|---|---|---|---|---|
| Conservadora | ~+64% | +28,9% | +32,6% | **~-35 pp** | +4 pp |
| Moderada | ~+76% | +60,3% | +41,3% | ~-16 pp | -19 pp |
| Arriesgada | ~+89% | +76,8% | +86,5% | ~-12 pp | +10 pp |
| Especulativa | ~+60% | +61,7% | +125,5% | ~+2 pp | +64 pp |

"Mecanica de cartera" mezcla tres cosas que este backtest NO separa: la exposicion
(75-86% invertido contra 100% del EW), las salidas por balance (60-70% de las
operaciones) y los SL/TP. Separarlas es la pregunta siguiente.

Como reabrir: el motor (`src/utils/carteras_perfil.py`) y el script
(`bt_carteras_perfil.py`) quedan como estan; cualquier variante nueva (otra
seleccion, otras salidas) es un PRE-REGISTRO NUEVO, no un ajuste de este.

---

## Diagnostico de SALIDAS de la Conservadora (pre-registro 15/10/2026)

**Pregunta:** que pieza de la mecanica de salida se come los ~35 pp que la cartera
Conservadora pierde contra mantener todo el perfil? Es un DIAGNOSTICO, no una
estrategia: lo que salga es punto de partida de una cartera nueva con su propio
pre-registro.

**Fijo en todas las variantes:** la ENTRADA de la cartera Conservadora tal cual
(15% del capital actual, 6 lugares, no entra hasta vender, no compra con balance en
la rueda o la siguiente) y la SELECCION AL AZAR dentro del perfil, 20 semillas --
las MISMAS listas de candidatos por rueda y semilla en todas las variantes, asi la
diferencia es de la salida y se compara por pares de semillas. Mismo periodo que el
backtest (2021-09-01 -> 2026-09-30), sin costos.

| Variante | Balance | SL | TP | Time stop |
|---|---|---|---|---|
| V0 (control, actual) | sale | -7% | +12% | no |
| V1 | NO sale | -7% | +12% | no |
| V2 | sale | no | no | no |
| V3 | sale | -7% | no | no |
| V4 | NO sale | no | no | 60 ruedas |

Todo sobre el close. El time stop cuenta ruedas del backtest desde la de entrada.

**Metricas:** retorno total y por tramo anual (mediana de 20 semillas), banda
(min/p25/mediana/p75/max), retorno / exposicion, max drawdown, Sortino, peor
operacion y p5 de las operaciones (%), operaciones por anio, motivos, dias en
posicion; referencia: mantener el perfil (~+64%).

**Regla de lectura** (por variante, contra V0): MEJORA si las tres:
1. la mediana del retorno total supera a la de V0;
2. supera a V0 en al menos 4 de los 6 tramos anuales (medianas);
3. la cola no empeora mas de lo tolerado: max drawdown mediano a lo sumo 5 pp peor
   que V0 Y p5 de las operaciones a lo sumo 2 pp peor que V0 (relajar una salida se
   juzga por la cola: ANALISIS_SALIDAS.md, paso 1).

Se informa ademas P(variante >= V0) sobre las 20 semillas PAREADAS. Son 4
variantes: alguna puede cumplir por azar (34% por variante para el criterio 2); se
lee con eso presente y se escribe.

### Resultados del diagnostico (15/10/2026)

2021-09-01 -> 2026-09-30, 20 semillas, seleccion al azar (mismas listas en todas),
sin costos. Detalle: `reportes/carteras_perfil/20261015_1503_salidas_conservadora/`.
**Control del montaje: V0 da +28,93%, igual al azar del backtest (+28,9%).**
Referencia: mantener todo el perfil Conservadora **+64,2%**.

| (medianas) | V0 actual | V1 sin balance | V2 sin SL/TP | V3 sin TP | V4 time stop 60 |
|---|---|---|---|---|---|
| Retorno total | +28,9% | +38,4% | +28,2% | +27,6% | **+43,6%** |
| Banda p25 / p75 | 20 / 34 | 28 / 52 | 20 / 46 | 16 / 40 | 35 / 62 |
| Max drawdown | 15,8% | 17,2% | 16,6% | 17,4% | 16,3% |
| Sortino | 0,21 | 0,37 | 0,20 | 0,19 | 0,45 |
| Peor operacion | -13,5% | -14,1% | -22,2% | -13,5% | -21,7% |
| p5 operaciones | -8,7% | -9,7% | -9,6% | -8,5% | **-14,1%** |
| Operaciones | 285 | 136 | 209 | 257 | 122 |
| Dias en posicion (mediana) | 30 | 56 | 63 | 34 | 87 |
| Tramos ganados a V0 | -- | 3/6 | 3/6 | 2/6 | **6/6** |
| P(variante >= V0), pareado | -- | 65% | 60% | 40% | 85% |
| **Lectura** | control | no mejora | no mejora | no mejora | **no mejora (cola)** |

Por tramo (V0 / V1 / V2 / V3 / V4 / EW): 2021 6,7/6,2/6,6/7,1/7,3/7,0 · 2022
-3,8/-5,5/-1,9/-5,0/-0,8/-0,1 · 2023 0,4/4,3/3,6/3,1/2,6/7,5 · 2024
9,9/16,0/9,0/9,8/15,2/14,1 · 2025 7,9/8,0/8,8/7,0/11,5/12,3 · 2026
5,7/4,8/4,6/4,3/6,5/11,6.

**Que dice:**
1. **Por la regla pre-registrada ninguna variante mejora.**
2. **El SL y el TP no son el problema**: sacarlos (V2) o sacar solo el TP (V3) deja
   la cartera igual que V0.
3. **La salida por balance cuesta**: sin ella (V1) +38% contra +29%, pero solo 3/6
   tramos: parte del problema, no todo.
4. **V4 (mantener 60 ruedas, sin balance, sin SL/TP) es la que mas se acerca al
   perfil**: +43,6%, le gana a V0 en los 6 tramos (por azar, 1 en 64) y en el 85%
   de los pares de semillas; drawdown de cartera casi igual (16,3% contra 15,8%) y
   Sortino del doble. **No pasa por la cola de las OPERACIONES**: p5 -14,1% contra
   -8,7% (tolerancia 2 pp). Sin stop, alguna operacion individual cae mas; a nivel
   cartera el riesgo no empeoro. La regla se fijo antes y no se mueve: queda
   escrito que el criterio que la frena es el de la operacion, no el de la cartera.
5. **Aun la mejor queda ~20 pp debajo de mantener el perfil** (+43,6% contra
   +64,2%). Lo que falta no es la salida: es tener 6 nombres en vez del perfil
   entero y ~12% en cash.

**Que NO dice:** nada sobre la seleccion (aca es al azar); nada con costos (V0 285
operaciones, V4 122); nada sobre las otras carteras.

---

## Conservadora y Moderada: CERRADAS (15/10/2026)

Decision del usuario. Conclusion, en terminos de lo medido: **para estos perfiles,
mantener todo el perfil le gana a toda mecanica probada y la seleccion no aporta**
(Conservadora: perfil +64%, mejor salida +44%, seleccion +4 pp = ruido; Moderada:
perfil ~+76%, azar +60%, seleccion -19 pp). Son acciones de baja volatilidad y poca
dispersion entre si: hay poco que ganar eligiendo y su retorno viene de quedarse
invertido; cada regla que agrega rotacion cuesta. NO dice que no exista un algoritmo
conservador (rebalanceo, filtros de tendencia, control de riesgo): dice que estas
reglas de seleccion y salida no sirven en estos perfiles. No se sigue con ellas.

---

## Diagnostico de SALIDAS de Arriesgada y Especulativa (pre-registro 15/10/2026)

Mismo diseno que el de la Conservadora, para que sea comparable:

- **Principal (decide):** seleccion AL AZAR dentro del perfil, 20 semillas, mismas
  listas que el backtest; misma entrada; solo cambia la salida.
- Variantes, con los SL/TP de CADA cartera (Arriesgada 15/30, Especulativa 20/50):

| Variante | Balance | SL | TP | Time stop |
|---|---|---|---|---|
| V0 (control) | sale | el de la cartera | el de la cartera | no |
| V1 | NO sale | el de la cartera | el de la cartera | no |
| V2 | sale | no | no | no |
| V3 | sale | el de la cartera | no | no |
| V4 | NO sale | no | no | 60 ruedas |

- **Misma regla de lectura** (mejora = total > V0 Y >= 4/6 tramos Y max drawdown
  mediano <= V0 + 5 pp Y p5 de operaciones >= V0 - 2 pp).
- **Control del montaje:** V0 tiene que reproducir el azar del backtest (Arriesgada
  +76,8%, Especulativa +61,7%).
- **Comparaciones multiples, declarado:** son 8 (4 variantes x 2 carteras). Por azar,
  el criterio de tramos (>= 4/6) se cumple con 34% por comparacion: una "mejora"
  aislada vale poco. Ademas se eligio mirar estos dos perfiles DESPUES de ver que
  eran los que mejor daban: es otra razon para exigir consistencia (mismo signo en
  las dos carteras y en la mayoria de los tramos) antes de creerle algo.
- **Informativo (no decide):** las mismas 5 variantes con la SELECCION en vez del
  azar. V0 tiene que reproducir el backtest (Arriesgada +86,5%, Especulativa
  +125,5%). Sirve para ver si seleccion y salida suman o se pisan.

### Resultados del diagnostico de Arriesgada y Especulativa (15/10/2026)

Detalle: `reportes/carteras_perfil/20261015_1516_salidas/`. Controles del montaje:
V0 reproduce el backtest en los 6 casos (azar 76,77 / 61,70; seleccion 86,54 /
125,50; y la Conservadora re-corrida da 28,93 / 32,62: el codigo generalizado no
cambio nada).

**Principal (seleccion al azar). Retorno total mediano; entre parentesis tramos
ganados a V0 / P(variante >= V0) pareado:**

| | V0 actual | V1 sin balance | V2 sin SL/TP | V3 sin TP | V4 time stop 60 | Mantener perfil |
|---|---|---|---|---|---|---|
| Arriesgada 15/30 | **+76,8%** | +63,4% (4/6, 45%) | +69,3% (3/6, 70%) | +74,9% (2/6, 50%) | +63,9% (3/6, 50%) | +89,0% |
| Especulativa 20/50 | +61,7% | +35,9% (1/6, 30%) | **+99,4% (6/6, 75%)** | +70,3% (3/6, 70%) | +78,0% (2/6, 55%) | +60,0% |

Riesgo de las que mas dan, contra V0:
- Especulativa V2: max drawdown de cartera 41,7% contra 46,7% (MEJOR), Sortino 0,70
  contra 0,49; pero p5 de operaciones -29,3% contra -23,0% y peor operacion -50,3%.

**Lectura pre-registrada: ninguna variante mejora en ninguna de las dos carteras.**
Especulativa V2 cumple total y 6/6 tramos y la frena SOLO la cola de las
operaciones (6,3 pp peor; tolerancia 2). No hay consistencia entre carteras: en
Arriesgada V2 queda debajo de V0.

**Informativo (con la seleccion):** Especulativa V0 +125,5% / V2 +144,4% (4/6, 75%,
cola falla) / V3 +108,1% / V4 +93,6% / V1 +47,2%. Arriesgada V0 +86,5% / V2 +88,3%
(5/6, 60%, cola falla) / V1 +75,0% / V3 +68,5% / V4 +75,7%.

**Que dice:**
1. **Salir por balance PROTEGE en los perfiles volatiles** (V1 empeora en las dos:
   Especulativa -26 pp, Arriesgada -13 pp). En la Conservadora pasaba lo contrario.
   El efecto del balance depende de cuanto se mueve el ticker: tiene sentido.
2. **En Arriesgada la salida actual ya es la mejor de las cinco**, y aun asi queda
   ~12 pp debajo de mantener el perfil.
3. **En Especulativa, sacar el SL/TP y dejar solo la salida por balance (V2) es la
   mejor variante en las DOS corridas** (azar y seleccion), con menor drawdown de
   cartera. Lo frena el criterio de cola por OPERACION, igual que a V4 en la
   Conservadora: en las dos mejores variantes del diagnostico, el stop protege la
   operacion individual y cuesta a nivel cartera.
4. Especulativa V2 al azar (+99%) le gana a mantener el perfil (+60%): en este perfil
   la mecanica de cartera (6 nombres + salir antes del balance) SUMA.

**Que NO dice:** el criterio de cola se fijo antes y no se mueve. Cambiarlo ahora
por uno de cartera y "aprobar" V2 seria elegir la regla mirando el resultado, sobre
datos 2021-2026 que ya se usaron para todo este analisis. Lo que queda es una
HIPOTESIS ("en Especulativa, sin SL/TP y con salida por balance") que solo datos
NUEVOS pueden confirmar. Sin costos.
