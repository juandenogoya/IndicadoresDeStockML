# Perfiles de carteras -- segmentacion del universo por comportamiento

Documento de DISENO (previo a codear). Consolida el debate conceptual de la
rama `feature/perfiles-carteras`. NO hay implementacion todavia: esto fija el
marco, las decisiones tomadas y las cuestiones abiertas antes de listar
archivos y construir.

Fecha: 3/8/2026. Rama: `feature/perfiles-carteras`.

---

## 1. Objetivo

Identificar y segmentar los ~200 tickers del universo en cuatro perfiles de
riesgo/comportamiento, pensados para **carteras de mediano a largo plazo con
enfoque de swing trading**:

- **Conservadora**
- **Moderada**
- **Arriesgada** (o Agresiva)
- **Especulativa**

El fin ultimo es habilitar armado de carteras por perfil y, en una segunda
etapa, rotaciones (ver Proyecto 2).

## 2. Que ES el perfil (y que NO es)

Decision de fondo, ya cerrada en el debate:

- El perfil es una propiedad **del instrumento**, no del momento de trade.
  "Este ticker, por naturaleza, se comporta como agresivo". Es una etiqueta
  ESTABLE (cambia despacio), no una senal que parpadea rueda a rueda.
- NO es "perfil de la operacion de hoy". KO nunca garantiza una operacion
  conservadora: puede estar en un momento pesimo. Ese seria otro proyecto.
- NO es una senal de compra/venta. Es una clasificacion de caracter/riesgo.

## 3. Anclas (ground truth / conjunto de validacion)

El usuario aporto ejemplos que actuan como **etiquetas verdaderas**. El modelo
cuantitativo se calibra para reproducirlas; si no las reproduce, esta mal.

| Perfil        | Anclas                                  |
|---------------|-----------------------------------------|
| Conservadora  | KO, WMT, PG, JNJ, MCD                    |
| Moderada      | JPM, BAC, CAT, DE, LMT, RTX             |
| Arriesgada    | Tech (large caps tecnologicas)          |
| Especulativa  | Semiconductores, relacionadas a cripto  |

Insight clave: este ordenamiento es, en la practica, un **ranking de beta /
sensibilidad al ciclo**. Staples (beta < 1) -> financieras/industriales/defensa
(beta ~ 1) -> tech (beta > 1) -> semis/cripto (beta >> 1).

Como el mapeo sector -> perfil es casi trivial, el VALOR del sistema no es
etiquetar por sector, sino:

1. Detectar **excepciones**: un staple que este semestre se comporta como
   agresivo, o una tech que esta dormida como moderada.
2. **Rankear dentro de cada caja**: de las moderadas, cual esta mas caliente.
3. Detectar **drift temporal**: cuando un ticker migra de caja, ahi hay senal.

## 4. Enfoque metodologico

**Top-down con confirmacion cuantitativa** (fork resuelto en el debate):

- El **sector es el esqueleto/prior**: da el perfil base (staple -> conservadora).
- El **comportamiento cuantitativo confirma o corrige**: ajusta dentro de la caja
  y marca las excepciones donde el ticker se despega de su sector.

Corte de las cajas: **hibrido**.

- Umbrales absolutos anclados con sentido de mercado (beta, ATR%, drawdown).
- Percentil dentro del universo para desempatar y para el ranking intra-caja.
- Caveat honesto: el universo es todo US large/mid de calidad; una
  "especulativa" aca puede ser normal en el mercado amplio. Las etiquetas son
  "dentro de este universo". Hacerlo explicito.

Filosofia de combinacion: el perfil **no es un promedio de numeros**, es un
**conjunto de senales que tienen que estar de acuerdo**. Donde no coinciden,
ahi esta la excepcion interesante.

## 5. Los ejes de comportamiento

Las 6 variables que se plantearon originalmente (variacion de precio,
volatilidad, estructura de 200 ruedas, volumen, PCR, sesgo de OI) NO son 6
dimensiones independientes: varias miden lo mismo (riesgo de precio) y se
contarian doble. Se colapsan en pocos ejes reales:

### 5.1. Beta (columna vertebral)

Sensibilidad al mercado (contra SPY/futuros, que ya tenemos). Un solo numero
reproduce ~70-80% del ordenamiento de las anclas. Es el eje primario.

### 5.2. Volatilidad ATR% multi-timeframe

Motor de volatilidad = **ATR% = (ATR / close) * 100**, calculado por timeframe.
Origen de la idea: indicador Pine "Volatilidad Multi-TF" del usuario
(request.security del ATR% sobre 1H/4H/D/W/M).

Por que ATR% y no desvio de retornos:
- ATR es basado en rango (true range), captura el recorrido real de la barra.
- Normalizado en % -> comparable entre tickers de precio muy distinto
  (KO ~60 vs NVDA ~900). Ideal para rankear el universo.

Timeframes y parametros (de default, CONFIGURABLES):

| TF        | Periodo ATR | Lookback real aprox |
|-----------|-------------|---------------------|
| Diario    | 14          | ~3 semanas          |
| Semanal   | 8           | ~2 meses            |
| Mensual   | 6           | ~6 meses            |
| Quincenal | (a definir) | opcional, entre W y M |

Decisiones:
- **Intradia (1H, 4H) DESCARTADO**: no tenemos barras intradia (precios_diarios
  es diario) y para swing de mediano/largo la vol de 1H es ruido irrelevante.
- Lookback **progresivo** (14/8/6): cada TF mas largo mira mas atras (3 sem / 2
  meses / 6 meses). Es una propiedad deseada: cubre corto/medio/largo natural.
  Alternativa "lookback parejo" (misma ventana en los 3) descartada por ahora;
  se reservaria para medir el caracter con precision de laboratorio.
- El **caracter** (tiende vs revierte) sale GRATIS de comparar las vols de los
  distintos TF: si la vol crece mucho al alargar el TF -> persistencia/tendencia
  (agresivo); si se aplana -> mean-reverting, mas tranquilo al horizonte (aunque
  se vea nervioso en el diario). No es un calculo aparte.
- Caveat: 8 semanas y 6 meses son lookbacks cortos -> responsivos (bueno para
  detectar drift) pero el mensual con ~6 barras es algo salton (un earnings feo
  lo mueve). Aceptable porque es UNA senal entre varias, no sola.

Reuso: `ATR14` diario ya existe en `indicadores_tecnicos`; el resample semanal
ya existe en `src/utils/weekly_tf.py` (W-FRI). Falta agregar resample mensual
(y quincenal). El eje de vol arranca casi construido.

### 5.3. Drawdown / lado malo

Profundidad y frecuencia de las caidas. La volatilidad es simetrica (trata
igual subir que bajar); el drawdown mide "cuanto me puede doler", que es mas
honesto para un perfil de riesgo. Los semis no son solo mas volatiles: tienen
caidas mas hondas. Eje asimetrico, independiente de la vol.

### 5.4. Matices (no ejes primarios)

- **Estructura de las 200 ruedas / gaps**: tendencia vs rango, cantidad de
  BOS/CHoCH, gaps de earnings. Puede subir una categoria a un ticker de vol
  moderada pero comportamiento erratico. Fuentes posibles:
  `features_market_structure`, `features_precio_accion`.
- **Volumen / liquidez**: se CAE como eje primario. El universo es todo liquido
  (NVDA es de lo mas liquido del mundo). Para este universo la liquidez es casi
  constante y no discrimina. Se podria retener como filtro de borde, no como eje.

## 6. Overlay de opciones (PCR + sesgo de Open Interest)

Decision cerrada:

- Las opciones tienen historia **solo desde abril 2026 (~4 meses)**. Demasiado
  poco para un perfil estructural o estacional.
- PCR y sesgo de OI van **SOLO como overlay del estado actual**, dinamico, NUNCA
  como parte de "que es" el ticker ni de su historia.
- Cobertura **parcial**: no todos los tickers tienen cadena liquida. Los que no,
  quedan sin overlay, con flag honesto (mismo espiritu que `peer_basis='none'`
  en `fundamentales_ticker_vs_sector`).
- Rol: "esta especulativa, hoy con sesgo bajista fuerte en opciones". Se muestra
  aparte del perfil, no lo modifica.

## 7. Salidas de valor esperadas

1. Etiqueta de perfil por ticker (las 4 cajas).
2. Ranking dentro de cada caja (quien esta mas caliente).
3. Excepciones: tickers que se despegan del perfil tipico de su sector.
4. Drift temporal: migraciones de caja en el tiempo.
5. Overlay de opciones (estado actual) donde haya cobertura.

## 8. Proyecto 2 (SEPARADO): estacionalidad y rotacion

No mezclar con el perfilado. El perfil es la capa ESTABLE (que es cada ticker);
la rotacion es la capa TACTICA encima (cuando y hacia donde moverse). El
perfilado bien hecho es requisito de esto.

- **Estacionalidad**: con ~5 anios de precios, por ticker es flaco (~5 muestras
  por mes = casi ruido). Tiene sentido **a nivel sector** (mas senal), no ticker.
  La estacionalidad seria necesitaria 15-20 anios.
- **Rotacion dentro del sector**: de la que ya corrio a la que quedo atras ->
  fuerza relativa + correlacion.
- **Grupos por comportamiento (cross-industria)**: clustering por correlacion
  ignorando la etiqueta GICS -> puede juntar tickers de sectores distintos que
  se mueven igual. Justamente lo que habilita rotar cross-industria. Este es el
  motivo por el que el enfoque general es "sector como prior, no como carcel".

## 9. Datos disponibles (inventario)

- `precios_diarios`: OHLCV, ~2020/2021 -> hoy (~5 anios). Fuente del perfilado.
- `indicadores_tecnicos`: ATR14 diario ya calculado.
- `src/utils/weekly_tf.py`: resample semanal W-FRI (modulo puro, reutilizable).
- Futuros / indices: base para calcular beta.
- `activos`: sector/industry del universo (prior top-down).
- `features_market_structure`, `features_precio_accion`: estructura/SMC/gaps.
- Opciones (RAILWAY, desde abril 2026): overlay de estado actual, cobertura
  parcial.

## 10. Decisiones tomadas (resumen)

1. Perfil = del instrumento, estable. No del trade, no senal.
2. Anclas del usuario = conjunto de validacion; el modelo debe reproducirlas.
3. Enfoque top-down (sector = prior) + confirmacion cuantitativa.
4. Corte hibrido: umbrales absolutos + percentil intra-universo. Etiquetas
   "dentro de este universo", explicito.
5. Ejes: beta (backbone) + ATR% multi-TF + drawdown. Estructura/gaps como
   matiz. Liquidez descartada como eje (universo homogeneo en liquidez).
6. ATR% multi-TF: D/W/M (+ quincenal opcional), periodos 14/8/6 CONFIGURABLES,
   lookback progresivo, intradia descartado. Caracter tiende-vs-revierte sale
   de comparar TFs.
7. Opciones (PCR + OI): overlay dinamico de estado actual, cobertura parcial con
   flag. Fuera del perfil estructural (solo 4 meses de historia).
8. Estacionalidad + rotacion = Proyecto 2, separado, encima del perfilado.
   Estacionalidad a nivel sector (no ticker) por historia corta.

## 11. Cuestiones abiertas (a decidir antes de construir)

- Periodo del ATR quincenal (o si se incluye).
- Metrica exacta de drawdown (max DD historico, DD promedio, frecuencia de DD >
  umbral, etc.). Reuso posible: `src/utils/ft_metricas.py` (modulo puro de
  metricas de riesgo, ya tiene max DD).
- Ventana de beta (2 anios? 1 anio? multi-ventana como la vol?) y contra que
  indice (SPY vs futuros ES).
- Ventanas de vol solapadas (rolling) vs no solapadas (calendario) -- afecta el
  numero de muestras y la autocorrelacion.
- Umbrales absolutos concretos por eje, calibrados contra las anclas.
- Formato de salida: tabla nueva (perfiles_ticker?), vista en el dashboard,
  ambos. Point-in-time si alguna vez lo consumen los bots/FT.
- Frecuencia de recomputo del perfil (mensual? semanal?) dado que es estable.
- UI: landing de la vista general como scatter (beta vs ATR%) vs tabla grande
  rankeable. Y "Por cartera" como subtabs vs dropdown selector.

## 12. Presentacion (UI en el dashboard)

Entrada propia en la barra lateral del dashboard: **"Carteras"** (coherente con
las vistas actuales: Informe / Radar / Financiero / Consultas IA).

Decision de diseno: NO estructurar solo como 4 subtabs (una por perfil). Eso
aisla justo los insights transversales que dan valor (excepciones, drift, ver el
gradiente). Se espeja el patron ya resuelto en la vista **Financiero** (modos
"Por ticker" vs "Screener sectorial"): un modo de detalle + un modo transversal.

Tres vistas dentro de "Carteras":

1. **Mapa / Vista general** (landing por default). El universo entero de una:
   scatter beta vs ATR% coloreado por caja (o tabla grande rankeable), para VER
   el gradiente, donde caen los cortes y que excepciones saltan.
2. **Por cartera** (aca van las 4: subtabs o dropdown selector). Cada perfil:
   tabla de sus tickers rankeados intra-caja, con las columnas que explican el
   porque (beta, ATR% D/W/M, max drawdown), un flag de **consistencia** (los ejes
   coinciden o es caso de borde) y el chip de opciones donde haya cobertura. Que
   muestre el porque, no solo la lista.
3. **Excepciones / Validacion**. Tickers que se despegan de su sector + chequeo
   de que las **anclas** cayeron donde deben (KO conservadora, NVDA especulativa).
   Al principio vale oro: es como se confia en que el modelo no delira.

Matiz de nombre: "Carteras" sugiere cartera construida (pesos, sizing). v1 es la
**segmentacion** (universo clasificado por perfil), NO una cartera ponderada. La
construccion real (elegir N + ponderar) seria un paso posterior.

## 13. Politica de trabajo

Antes de codear cualquier archivo en `src/` o `scripts/`: listar los archivos a
crear/modificar y pedir aprobacion (regla del proyecto). Este documento es solo
el marco conceptual; la implementacion se decide en pasos posteriores.
