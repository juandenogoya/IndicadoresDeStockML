"""
catalogo_tecnico.py
Catalogo de lo que el proyecto CALCULA y de como estan programadas las
estrategias. Modulo PURO de DATOS (solo stdlib: sin DB, sin pandas, sin red).

PARA QUE EXISTE
    Para poder experimentar con parametros y scores hace falta primero saber
    que hay: que mide cada columna, con que umbral, de que tabla sale y cual
    tiene un defecto conocido. Eso no se deduce leyendo una tabla de la DB.
    Lo consume `scripts/manual/exportar_catalogo.py`, que arma el .xlsx.

REGLA DE ORO: ESTE ARCHIVO NO CALCULA NADA
    Cada fila es una DESCRIPCION de codigo que vive en otro lado, y la columna
    `origen` dice donde. Si un umbral cambia, cambia ALLA y se corrige aca; el
    valor escrito aca no alimenta ningun calculo. Los parametros de las
    estrategias NO estan en este archivo a proposito: los lee el exportador de
    `ft_estrategias`, que es lo que de verdad corrio.

QUE NO ESTA (por decision, 23/11/2026)
    Fundamentales (PER/ROE/margenes/FCF), perfiles de riesgo y las 53 features
    del ML en detalle: ninguna estrategia FT las consume hoy. Inventario de las
    features del ML: docs/features_ml.md. Fundamentales:
    docs/fundamentales_calculo.md.
"""

from typing import Dict, NamedTuple, Tuple


class Indicador(NamedTuple):
    """Una fila de la hoja 'Indicadores'."""
    sigla: str
    nombre: str
    funcion: str
    origen: str


class Logica(NamedTuple):
    """La explicacion conceptual de una familia de estrategias (hoja 2)."""
    titulo: str
    pregunta: str      # para que existe la estrategia: que se quiso contestar
    concepto: str
    entrada: str
    salida: str
    sizing: str
    insumos: str
    estado: str


class FilaScore(NamedTuple):
    """Una condicion de un score, con lo que suma y donde se evalua."""
    capa: str
    condicion: str
    puntos: str
    nota: str


# ============================================================================
# HOJA 1 -- INDICADORES
# ============================================================================
# Los parametros de calculo (periodos) viven en src/utils/config.py:
#   SMA_PERIODS=[21,50,200] RSI_PERIOD=14 MACD=(12,26,9) ATR_PERIOD=14
#   BB_PERIOD=20 BB_STD=2 ADX_PERIOD=14 VOL_MA_PERIOD=20

_MEDIAS = (
    Indicador("SMA21", "Media Movil Simple de 21 ruedas",
              "Tendencia corta. El score tecnico suma 1,0 punto si close > SMA21.",
              "indicadores_tecnicos / src/indicators/technical.py"),
    Indicador("SMA50", "Media Movil Simple de 50 ruedas",
              "Tendencia media. Es la condicion que MAS pesa en el score tecnico: "
              "2,0 de 5,5 puntos.",
              "indicadores_tecnicos / src/indicators/technical.py"),
    Indicador("SMA200", "Media Movil Simple de 200 ruedas",
              "Tendencia larga. FILTRO OBLIGATORIO del score tecnico: si close <= "
              "SMA200 el score es 0 y no hay entrada posible, sin importar el resto.",
              "indicadores_tecnicos / src/indicators/technical.py"),
    Indicador("dist_sma21 / 50 / 200", "Distancia porcentual del cierre a cada SMA",
              "(close - SMA) / SMA * 100. Positivo = precio sobre la media. Mide "
              "cuan estirado esta el precio; ninguna estrategia la usa todavia.",
              "indicadores_tecnicos / src/indicators/technical.py"),
    Indicador("Tendencia SMA", "Alineacion de las tres medias",
              "Alcista si close esta sobre las medias y SMA21>=SMA50>=SMA200; "
              "Bajista si es el espejo; cualquier mezcla = Lateral. Es una lectura "
              "para el tablero, no entra en ningun score.",
              "src/utils/clasificacion_tecnica.py"),
)

_OSCILADORES = (
    Indicador("RSI14", "Relative Strength Index de 14 ruedas",
              "Oscilador 0-100 de fuerza del movimiento. OJO: el proyecto usa DOS "
              "umbrales distintos y opuestos. Clasificacion/tablero: <35 Sobreventa, "
              ">65 Sobrecompra. Score tecnico de FT: suma 1,0 punto si esta ENTRE "
              "45 y 68 (zona de tendencia sana, sin sobrecompra). Score ponderado "
              "viejo: suma si <35. Ver la hoja Scores.",
              "indicadores_tecnicos / config.RSI_PERIOD"),
    Indicador("MACD", "Moving Average Convergence Divergence (12, 26, 9)",
              "Diferencia entre la EMA de 12 y la de 26 ruedas. Mide momentum.",
              "indicadores_tecnicos / src/indicators/technical.py"),
    Indicador("macd_signal", "Linea de senal del MACD (EMA 9 del MACD)",
              "Referencia de cruce. MACD > signal = Compra por convencion del proyecto.",
              "indicadores_tecnicos"),
    Indicador("macd_hist", "Histograma del MACD (MACD - signal)",
              "El score tecnico exige las DOS cosas: MACD > signal Y hist > 0 "
              "(+1,5 puntos). Son redundantes por definicion; queda asi para no "
              "cambiar el comportamiento historico.",
              "indicadores_tecnicos"),
    Indicador("ADX", "Average Directional Index de 14 ruedas",
              "Fuerza de la tendencia SIN direccion. Umbrales de Wilder: >=25 "
              "Fuerte, 20-25 Moderada, <20 Debil. Se calcula y se muestra; "
              "NINGUNA estrategia lo usa. Candidato limpio para experimentar.",
              "indicadores_tecnicos / clasificacion_tecnica.py"),
    Indicador("momentum", "Momentum de 14 ruedas",
              "close[t] - close[t-14], en dolares (no normalizado, asi que no es "
              "comparable entre tickers de precio distinto). Solo lo usa el score "
              "ponderado viejo.",
              "indicadores_tecnicos"),
    Indicador("RSI / MACD semanal", "Los mismos dos, en barras semanales",
              "Se calculan AL VUELO desde precios_diarios (resample W-FRI), no "
              "salen de una tabla: indicadores_tecnicos_1w esta CONGELADA desde "
              "2026-04-02. Los usan el dashboard y el MCP, ninguna estrategia.",
              "src/utils/weekly_tf.py"),
)

_VOLATILIDAD = (
    Indicador("ATR14", "Average True Range de 14 ruedas",
              "Rango verdadero medio: la unidad de volatilidad del proyecto. Es la "
              "base de TODOS los stops de las sectoriales: SL = entrada - 2xATR, "
              "TP = entrada + 4xATR. Cambiar esos multiplos es la palanca mas "
              "directa que hay sobre la cola de resultados.",
              "indicadores_tecnicos / config.ATR_PERIOD"),
    Indicador("BB_upper / middle / lower", "Bandas de Bollinger (20 ruedas, 2 desvios)",
              "Envolvente de volatilidad alrededor de la SMA20. Se calculan y se "
              "guardan; NINGUNA estrategia las usa.",
              "indicadores_tecnicos / config.BB_PERIOD, BB_STD"),
    Indicador("rango_diario_pct", "Rango de la vela sobre el cierre",
              "(high - low) / close * 100.",
              "features_precio_accion / src/indicators/precio_accion.py"),
    Indicador("rango_rel_atr", "Rango de la vela en unidades de ATR",
              "(high - low) / ATR14. Dice si la vela de hoy es grande para ESTE "
              "ticker. Familia de microestructura: medida como correcta en la "
              "auditoria del 17/9/2026 y sin usar.",
              "features_precio_accion"),
    Indicador("rango_expansion", "Flag de rango mayor al de la vela previa",
              "DEFECTUOSA: una comparacion con NaN pasada a int escribe 0 donde no "
              "hay dato, asi que INVENTA valores en las 32 costuras de backfill. "
              "No usar sin corregir (docs/estructura_velas.md sec. 12).",
              "features_precio_accion"),
)

_VOLUMEN = (
    Indicador("OBV", "On Balance Volume",
              "Volumen acumulado con el signo del dia. Serie acumulada: su NIVEL no "
              "es comparable entre tickers ni entre tramos de backfill. Se calcula "
              "y no se usa.",
              "indicadores_tecnicos"),
    Indicador("vol_relativo", "Volumen relativo a su media de 20 ruedas",
              "volumen / media(volumen, 20). >1 = mas volumen que lo normal.",
              "indicadores_tecnicos / config.VOL_MA_PERIOD"),
    Indicador("vol_spike", "Flag de pico de volumen",
              "Confirma entradas: suma al score SMC (+1 junto con los patrones) y "
              "es condicion obligatoria de EV3.",
              "features_precio_accion"),
    Indicador("up_vol_5d", "Fraccion del volumen de 5 ruedas en velas alcistas",
              "0 a 1. >0,55 vale como confirmacion de volumen en EV1. Tambien es "
              "una de las 3 condiciones de retencion de TECH_SECTOR_v2.",
              "features_precio_accion"),
    Indicador("vol_ratio_5d", "Volumen de 5 ruedas sobre su referencia",
              "Familia de flujo de volumen: medida como correcta y SIN USAR.",
              "features_precio_accion"),
    Indicador("vol_price_confirm", "Flag de volumen que confirma al precio",
              "Precio y volumen en el mismo sentido. Condicion obligatoria de EV2.",
              "features_precio_accion"),
    Indicador("vol_price_diverge", "Flag de divergencia precio-volumen",
              "Precio sube con volumen que cae. Es uno de los 4 votos del quorum "
              "de salida de OIEXIT_v1 (con dist_max_20d >= -1% y vol_ratio <= 0,8).",
              "features_precio_accion"),
    Indicador("ad_flow", "Flujo de acumulacion / distribucion",
              "Familia de flujo de volumen: correcta y sin usar. La propone "
              "docs/features_ml.md para modelos nuevos.",
              "features_precio_accion"),
    Indicador("chaikin_mf_20", "Chaikin Money Flow de 20 ruedas",
              "Flujo de dinero de Chaikin. Correcta y sin usar.",
              "features_precio_accion"),
    Indicador("clv", "Close Location Value",
              "Donde cerro la vela dentro de su rango: +1 en el maximo, -1 en el "
              "minimo. Microestructura correcta y sin usar.",
              "features_precio_accion"),
)

_PRICE_ACTION = (
    Indicador("es_alcista", "Flag de vela alcista",
              "close > open. Condicion OBLIGATORIA de entrada de SMC (v1 y v3) y "
              "de EV4.",
              "features_precio_accion"),
    Indicador("body_pct / body_ratio", "Tamano del cuerpo de la vela",
              "Cuerpo como fraccion del rango, y contra el cuerpo tipico.",
              "features_precio_accion"),
    Indicador("upper_shadow_pct / lower_shadow_pct", "Sombras de la vela",
              "Fraccion del rango que ocupa cada sombra. Son los insumos de los "
              "patrones de martillo y estrella.",
              "features_precio_accion"),
    Indicador("gap_apertura_pct", "Hueco de apertura",
              "(open - close previo) / close previo. Microestructura, sin usar.",
              "features_precio_accion"),
    Indicador("dist_max_20d / dist_min_20d", "Distancia al maximo y minimo de 20 ruedas",
              "En %. dist_max_20d >= -1% (pegado al maximo) es uno de los 4 votos "
              "del quorum de salida de OIEXIT_v1.",
              "features_precio_accion"),
    Indicador("pos_rango_20d", "Posicion dentro del rango de 20 ruedas",
              "0 = en el minimo, 1 = en el maximo.",
              "features_precio_accion"),
    Indicador("inside_bar / outside_bar", "Vela contenida / envolvente en rango",
              "inside: high < high previo Y low > low previo. outside: el espejo. "
              "Definicion identica en las dos tablas de velas.",
              "features_precio_accion y features_velas"),
    Indicador("velas_alcistas_5d / 10d", "Cuenta de velas alcistas en la ventana",
              "0 a 5 y 0 a 10.",
              "features_precio_accion"),
    Indicador("tendencia_velas", "Score de direccion de las ultimas 5 velas",
              "INUTIL Y DEFECTUOSA: es exactamente 2*velas_alcistas_5d - 5, o sea "
              "CERO informacion nueva, y el fillna(0) escribe -5 (el maximo "
              "bajista) donde no hay 5 barras. No usar.",
              "features_precio_accion"),
    Indicador("candle_score_5d", "Score de velas de 5 ruedas",
              "Desempate de velas que uso COMBO_v1 (dado de baja: no aporta). "
              "Sigue vivo como voto de salida en TECH_SECTOR_v2 (<0) y en el "
              "quorum de OIEXIT_v1 (<=-2).",
              "src/strategies/ y los bots sectoriales"),
)

_VELAS_VIEJAS = (
    Indicador("patron_engulfing_bull / bear", "Envolvente alcista / bajista (version vieja)",
              "MAL DEFINIDA: solo compara TAMANO de cuerpo y color, no que un "
              "cuerpo envuelva al otro. Medido: el 71-73% de las marcadas no "
              "envuelve. La lee todavia FT_SMC_v1 (a proposito: es el control) y "
              "la condicion EV1.",
              "features_precio_accion"),
    Indicador("patron_hammer", "Martillo (version vieja)",
              "MAL DEFINIDO: exige color, no limita la sombra opuesta y no mira la "
              "tendencia previa. El 46,5%-52% de los marcados esta arriba del "
              "rango: son hanging man, que es lo contrario.",
              "features_precio_accion"),
    Indicador("patron_shooting_star", "Estrella fugaz (version vieja)",
              "Mismo defecto que el martillo, en espejo.",
              "features_precio_accion"),
    Indicador("patron_doji", "Doji (version vieja)",
              "Cuerpo muy chico. Sin contexto de tendencia.",
              "features_precio_accion"),
    Indicador("patron_marubozu", "Marubozu (version vieja)",
              "MAL DEFINIDO: no distingue direccion (no dice si es alcista o "
              "bajista), lo que lo hace inservible como senal.",
              "features_precio_accion"),
)

_VELAS_NUEVAS = (
    Indicador("patron_hammer / hanging_man", "Martillo / Hombre colgado (clasicos)",
              "MISMA FORMA, contexto OPUESTO: cuerpo <=30% del rango, sombra "
              "inferior >=2x el cuerpo, sombra superior <=10%. Si viene de una "
              "CAIDA (5 ruedas) es martillo; si viene de una SUBA es hanging man. "
              "Esa separacion es lo que la version vieja no hacia.",
              "features_velas / src/indicators/velas.py"),
    Indicador("patron_shooting_star / inverted_hammer", "Estrella fugaz / Martillo invertido",
              "El espejo del par anterior: sombra superior larga, inferior <=10%. "
              "Tras suba = estrella fugaz; tras caida = martillo invertido.",
              "features_velas"),
    Indicador("patron_engulfing_bull / bear", "Envolvente que SI envuelve",
              "Exige que el cuerpo actual contenga al previo (open <= close previo "
              "y close >= open previo, con una desigualdad estricta) y que los "
              "colores sean opuestos. Sin requisito de contexto.",
              "features_velas"),
    Indicador("patron_marubozu_bull / bear", "Marubozu con direccion",
              "Cuerpo >=85% del rango, separado por color. Corrige el defecto de "
              "la version vieja.",
              "features_velas"),
    Indicador("patron_doji", "Doji (clasico)",
              "Cuerpo <=5% del rango Y que no sea ninguno de los 4 patrones de "
              "sombra: una sola etiqueta por forma (un doji libelula tras una "
              "caida se etiqueta martillo, porque la sombra larga es la informacion).",
              "features_velas"),
)

_ESTRUCTURA = (
    Indicador("is_sh_N / is_sl_N", "Swing high / swing low confirmado (N = 3, 5, 10)",
              "TRAMPA DOCUMENTADA: la fila marca la barra de CONFIRMACION, no la "
              "del swing. El precio del swing esta N barras ANTES (high[t-N]). "
              "Leer high[t] donde is_sh_N=1 da un precio equivocado.",
              "features_estructura / src/indicators/estructura.py"),
    Indicador("estructura_N", "Estado de la estructura",
              "+1 alcista (maximos y minimos ascendentes), -1 bajista, 0 neutral. "
              "Condicion obligatoria de SMC (>=0, no rota a la baja) y de EV1/EV4 "
              "(=1).",
              "features_estructura"),
    Indicador("dist_sh_N_pct / dist_sl_N_pct", "Distancia al ultimo swing high / low",
              "En %. dist_sl es el INSUMO DEL STOP de SMC: el SL se pone en el "
              "swing low, y la operacion se descarta si la distancia cae fuera del "
              "rango 1%-8% (muy cerca = ruido, muy lejos = riesgo que no cabe).",
              "features_estructura"),
    Indicador("dias_sh_N / dias_sl_N", "Antiguedad del ultimo swing, en ruedas",
              "EV1 exige dias_sl <= 15 y EV4 exige <= 10: un swing viejo ya no es "
              "referencia. Van como float (NaN si no hay swing).",
              "features_estructura"),
    Indicador("impulso_N_pct", "Tamano del ultimo tramo de impulso",
              "En %. Se calcula y no se usa.",
              "features_estructura"),
    Indicador("bos_bull_N / bos_bear_N", "Break of Structure (continuacion)",
              "Ruptura del ultimo swing en el sentido en que ya venia la "
              "estructura. bos_bull es una de las dos puertas de entrada de SMC "
              "y la condicion central de EV2.",
              "features_estructura"),
    Indicador("choch_bull_N / choch_bear_N", "Change of Character (giro)",
              "La misma ruptura pero CONTRA la estructura previa: el giro. "
              "choch_bull abre entrada en SMC y en EV3; choch_bear la veta ese dia "
              "y resta 12 puntos en el score del scanner.",
              "features_estructura"),
    Indicador("features_market_structure.*", "Las mismas columnas, version VIEJA",
              "PROHIBIDA para entrenar, validar o backtestear: detecta swings con "
              "ventana centrada (rolling 2N+1) y los escribe en SU barra, cuando "
              "se conocen N barras despues. Costo medido: AUC 0,65 guardada contra "
              "0,52 real. En vivo no enganaba (se lee la ultima fila), pero la "
              "HISTORIA mira 10 ruedas al futuro.",
              "features_market_structure / market_structure.py"),
)

_OPCIONES = (
    Indicador("PCR_VOL / PCR_OI", "Put Call Ratio por volumen / por open interest",
              "puts / calls. Convencion del proyecto: <1,0 Alcista, >=1,0 Bajista. "
              "Se calcula por VENTANA de vencimiento: corto 1-14 dias, medio 15-45, "
              "largo 46-90.",
              "opciones_pcr_plazo_diario / src/utils/opciones_plazo.py"),
    Indicador("pcr_score", "Cuantas de las 3 ventanas estan alcistas",
              "0 a 3. Es el gate 2 de OPTIONS_v1 (OI) y v2 (VOL): entra si >=2 de "
              "3. Una ventana sin 500 de OI (o de volumen) no vota: queda invalida.",
              "opciones_pcr_plazo_diario"),
    Indicador("Muro de OI (put / call wall)", "Strike con mayor open interest combinado",
              "Soporte si esta debajo del precio, resistencia si esta arriba. "
              "OIEXIT_v1 pone su STOP en el put wall de la ventana media. Zona "
              "dinamica por expected move: +/- min(1,5 x EM, 18%).",
              "opciones_pcr_plazo_diario / opciones_plazo.py"),
    Indicador("EM", "Expected Move",
              "precio x IV_ATM x raiz(DTE_ponderado / 365): cuanto espera moverse "
              "el mercado hasta el vencimiento. Define el ancho de la zona donde "
              "se busca el muro.",
              "src/utils/opciones_plazo.py"),
    Indicador("IV_ATM", "Volatilidad implicita at-the-money",
              "Promedio de la IV de los contratos a +/-5% del precio.",
              "opciones_snapshot -> opciones_plazo.py"),
    Indicador("Fuerza del muro", "Concentracion del OI en el strike del muro",
              "% del OI de ese lado que esta en el muro. Reemplazo del criterio "
              "viejo '3x la mediana', incompatible con OI combinado.",
              "opciones_pcr_plazo_diario"),
    Indicador("Precio de referencia", "Cierre que se usa para calcular en local",
              "NO es precio_subyacente de la captura: manda el close de "
              "precios_diarios, y la captura solo tapa el hueco. Para una cadena "
              "vieja se lleva a la escala de SU dia con los splits posteriores.",
              "src/utils/precio_referencia.py"),
)

_ZSCORES = (
    Indicador("z-score de ticker", "Desvios respecto de su propia historia (60 ruedas)",
              "Normaliza precio y volumen contra el propio ticker. REGLA MEDIDA "
              "del 15/9/2026: cualquier alerta de comportamiento inusual se "
              "define por z-score, NUNCA por multiplo fijo.",
              "ticker_zscore_diario / src/utils/zscore_pipeline.py"),
    Indicador("z-score de opciones", "Lo mismo para PCR y volumen de opciones",
              "Ventana 60 ruedas. TECHO DURO: la historia de opciones arranca el "
              "2026-04-18, asi que no hay base de 52 semanas hasta ~abril/2027 y "
              "toda ventana queda dentro de UN regimen.",
              "opciones_zscore_diario"),
    Indicador("z-score sectorial", "El ticker contra SUS PARES del sector",
              "Son 11 de las 53 features del ML. Con n<=3 tickers no miden nada: "
              "Real Estate (3) y Utilities (1) los reciben NaN y el ticker se "
              "publica con la marca 'Sin contexto sectorial'.",
              "features_sector / opciones_sector_zscore_diario"),
)

_NIVELES = (
    Indicador("POC", "Point of Control del perfil de precio",
              "El bin de precio con MAS volumen de la ventana: donde el precio "
              "paso mas tiempo negociando. Ventanas 21, 52 y 200 ruedas; bins "
              "geometricos de 0,5% ancladas en 1,0 (un tramo de n bins mide "
              "siempre el mismo %). MEDIDO Y NO PASA contra el azar: es "
              "DESCRIPTOR, no senal (docs/perfil_precio.md sec. 5).",
              "perfil_precio_diario / src/utils/perfil_precio.py"),
    Indicador("VAL / VAH", "Bordes del area de valor",
              "La zona que concentra el 70% del volumen de la ventana.",
              "perfil_precio_diario"),
    Indicador("ancho_va_pct", "Ancho del area de valor, en %",
              "Es el ERROR BAR del POC: con area ancha el POC no ubica nada. A 200 "
              "ruedas solo el 3% de los tickers tiene area angosta.",
              "perfil_precio_diario"),
    Indicador("Nivel de S/R por giros", "Precio donde el precio giro varias veces",
              "Acumulacion de GIROS confirmados (no de permanencia, que es lo que "
              "mide el POC): banda de +-1%, minimo 3 toques separados por 15 "
              "ruedas, orden por PERSISTENCIA (anios distintos) y no por conteo. "
              "Reproduce 8 de 11 bandas dibujadas a mano, pero 'fuerte' no se "
              "sostiene contra una serie barajada. DESCRIPTOR.",
              "src/utils/niveles_precio.py (sin tabla: no tiene consumidores)"),
    Indicador("Precios NOMINALES", "OHLCV sin ajustar por dividendos",
              "Tabla aparte porque precios_diarios es un PATCHWORK: la historia "
              "esta ajustada por dividendos y los ultimos meses son nominales, con "
              "la costura en una fecha DISTINTA por ticker. Comparar NIVELES de "
              "precio entre anios sobre precios_diarios fabrica niveles que no "
              "existen (una escalera nominal de 15% se ve como linea horizontal).",
              "precios_nominales / cargar_precios_nominales.py"),
)

_ML = (
    Indicador("ml_prob_ganancia", "Probabilidad del modelo ML (v1/V3 global)",
              "P(retorno_20d > +1%) segun el Random Forest global de 53 features. "
              "Aporta hasta +30 o -15 puntos al score del scanner por tramos de "
              "probabilidad (0,75 / 0,65 / 0,55 / 0,45 / 0,35).",
              "alertas_scanner.ml_prob_ganancia"),
    Indicador("ml_prob_v2", "La misma probabilidad, modelo v2",
              "RF calibrado (isotonica) sobre 196 tickers, en PARALELO en la misma "
              "fila desde el 13/9/2026. La probabilidad calibrada vive en otra "
              "escala: usa sus propios cortes equivalentes.",
              "alertas_scanner.ml_prob_v2 / src/ml/ml_v2.py"),
    Indicador("alert_score", "Score compuesto del scanner (0-100)",
              "Base 50 + ML + price action + score rule-based - senales bajistas. "
              "Ver la hoja Scores.",
              "alertas_scanner / src/pipeline/alert_classifier.py"),
    Indicador("alert_nivel", "Nivel de alerta",
              "COMPRA_FUERTE >=75, COMPRA >=60, NEUTRAL >=40, VENTA >=25, "
              "VENTA_FUERTE <25. COMPRA_FUERTE es el unico nivel con el que abren "
              "las estrategias ML.",
              "alertas_scanner"),
    Indicador("score_ponderado", "Score rule-based ponderado (0 a 1)",
              "6 condiciones binarias por pesos que suman 1,0. Entra al score del "
              "scanner como un componente de hasta 10 puntos. OJO: su condicion "
              "de RSI es rsi<35 (sobreventa), al REVES del score tecnico de FT, "
              "que premia 45-68.",
              "scoring_tecnico / src/scoring/rule_based.py"),
    Indicador("EV1 a EV4", "Las 4 condiciones de entrada de price action",
              "Confluencias de estructura + patron + volumen. Cuantas se cumplen "
              "define el aporte de price action al score del scanner (3 o mas = "
              "+15 puntos). Ver la hoja Scores.",
              "src/backtesting/strategies_pa.py"),
    Indicador("alert_score_v2 / alert_nivel_v2", "El score y el nivel con el modelo v2",
              "Misma formula y mismas senales de price action: lo unico que cambia "
              "es la probabilidad y sus cortes. Los lee FT_ML_SCANNER_v2.",
              "alertas_scanner"),
)

INDICADORES: Tuple[Indicador, ...] = (
    _MEDIAS + _OSCILADORES + _VOLATILIDAD + _VOLUMEN + _PRICE_ACTION
    + _VELAS_VIEJAS + _VELAS_NUEVAS + _ESTRUCTURA + _OPCIONES
    + _ZSCORES + _NIVELES + _ML
)

# Titulo de grupo -> cuantas filas ocupa, para escribir los separadores.
GRUPOS: Tuple[Tuple[str, int], ...] = (
    ("MEDIAS Y TENDENCIA", len(_MEDIAS)),
    ("OSCILADORES Y MOMENTUM", len(_OSCILADORES)),
    ("VOLATILIDAD Y RANGO", len(_VOLATILIDAD)),
    ("VOLUMEN Y FLUJO", len(_VOLUMEN)),
    ("PRICE ACTION (forma de la vela y posicion)", len(_PRICE_ACTION)),
    ("PATRONES DE VELA -- version VIEJA (defectuosa, todavia en uso)", len(_VELAS_VIEJAS)),
    ("PATRONES DE VELA -- version CLASICA (correcta, en paralelo)", len(_VELAS_NUEVAS)),
    ("ESTRUCTURA DE MERCADO (SMC)", len(_ESTRUCTURA)),
    ("OPCIONES", len(_OPCIONES)),
    ("Z-SCORES (normalizacion)", len(_ZSCORES)),
    ("NIVELES Y PERFIL DE PRECIO", len(_NIVELES)),
    ("ML Y SCORES DEL SCANNER", len(_ML)),
)


# ============================================================================
# HOJA 2 -- ESTRATEGIAS: la explicacion conceptual, por LOGICA
# ============================================================================
# La clave es ft_estrategias.logica. Varias instancias comparten logica
# (FT_ML_SCANNER_v1 y v2, FT_SMC_v3_N5 y N3): lo que cambia son los parametros.

LOGICAS: Dict[str, Logica] = {
    "tecnico": Logica(
        titulo="Tecnico puro (no sectorial)",
        pregunta="Un score de medias + momentum, aplicado al universo entero y sin "
                 "reparto por sector, elige mejor que el azar?",
        concepto="Compra tendencia establecida: precio sobre sus tres medias, MACD "
                 "a favor y RSI en zona sana (ni debil ni sobrecomprado). Es la "
                 "forma mas simple de seguimiento de tendencia del proyecto y el "
                 "control natural de las sectoriales: misma regla, sin particion.",
        entrada="score tecnico >= 4,0 de 5,5. El filtro de SMA200 es obligatorio: "
                "sin el, score 0.",
        salida="score <= 3,5 (que es exactamente dejar de cumplir la entrada: "
               "ningun score cae entre 3,5 y 4,0), stop a 2xATR, objetivo a 4xATR, "
               "o balance al dia siguiente.",
        sizing="15% del capital por operacion, maximo 5 posiciones.",
        insumos="indicadores_tecnicos + close de precios_diarios.",
        estado="Activa. Replica del Bot 2 de Alpaca (apagado el 13/9/2026).",
    ),
    "tecnico_sectorial": Logica(
        titulo="Tecnico sectorial",
        pregunta="Repartir el capital en partes iguales por sector y competir solo "
                 "dentro de cada sector, mejora la seleccion del mismo score?",
        concepto="El mismo score tecnico, pero con el capital PARTICIONADO: 9 "
                 "sectores con un presupuesto fijo cada uno. Un ticker no compite "
                 "contra todo el universo sino contra sus pares, asi que la "
                 "diversificacion es por construccion y no una regla aparte.",
        entrada="score >= 4,0, hasta 5 posiciones por sector al 20% del "
                "presupuesto sectorial. Empate: lo resuelve el ALFABETO, y eso "
                "decide el 42,8% de los sector-ruedas (ver la hoja Metodo).",
        salida="Prioridad: balance manana -> score <= 3,5 -> stop 2xATR -> "
               "objetivo 4xATR. El score pesa MAS que el stop, al contrario de "
               "las variantes con opciones.",
        sizing="Presupuesto por sector = capital / 9; 20% de ese presupuesto por "
               "posicion.",
        insumos="indicadores_tecnicos + sector de la tabla activos.",
        estado="Activa. Es la estrategia de referencia del proyecto: casi todas "
               "las demas son esta con una pata cambiada.",
    ),
    "tecnico_sectorial_v2": Logica(
        titulo="Tecnico sectorial con retencion y rotacion",
        pregunta="La salida de la v1 es demasiado rapida? Y sirve rotar a un "
                 "candidato mejor cuando el sector esta lleno?",
        concepto="Dos cambios sobre la v1, los dos sobre la SALIDA y la ocupacion: "
                 "(1) no cierra por score degradado si la vela o el volumen "
                 "todavia acompanan, (2) si el sector esta lleno y aparece un "
                 "candidato con un punto mas de score, rota.",
        entrada="Igual que la v1: score >= 4,0.",
        salida="Cierra solo si score = 0 (perdio la SMA200) Y la retencion no "
               "aplica. Retiene mientras candle_score > 0 O up_vol_5d > 2.",
        sizing="Igual que la v1.",
        insumos="Los de la v1 + candle_score_5d y up_vol_5d.",
        estado="Activa. Mide exactamente el costo de la salida rapida de la v1.",
    ),
    "tecnico_sectorial_options_v1": Logica(
        titulo="Tecnico sectorial + confirmacion por OPEN INTEREST",
        pregunta="El posicionamiento en opciones (PCR por open interest) confirma "
                 "o contradice la senal tecnica?",
        concepto="Doble puerta: primero el tecnico, despues las opciones. El PCR "
                 "se mira por PLAZO (corto / medio / largo) y hace falta mayoria: "
                 "2 de las 3 ventanas alcistas. El open interest es posicion "
                 "ACUMULADA, o sea mas lento y mas estructural que el volumen.",
        entrada="Gate 1: tecnico >= 4,0. Gate 2: pcr_score >= 2 de 3. Una ventana "
                "con menos de 500 de OI no vota.",
        salida="Orden distinto a la v1: balance -> stop -> objetivo -> (score y "
               "PCR juntos). Cierra si score <= 3,5 Y pcr_score = 0; RETIENE si "
               "score <= 3,5 pero pcr_score >= 1.",
        sizing="Igual que la sectorial v1.",
        insumos="Los de la v1 + opciones_pcr_plazo_diario (PCR_OI).",
        estado="Activa. Su par v2 aisla OI contra VOL: la unica diferencia entre "
               "las dos es esa.",
    ),
    "tecnico_sectorial_options_v2": Logica(
        titulo="Tecnico sectorial + confirmacion por VOLUMEN de opciones",
        pregunta="Y si en vez del open interest acumulado se mira el volumen "
                 "diario de opciones, que es flujo del dia?",
        concepto="Identica a OPTIONS_v1 salvo UNA cosa: el PCR se calcula sobre "
                 "volumen diario en vez de open interest. Ese aislamiento es el "
                 "punto: cualquier diferencia de resultado es atribuible al dato, "
                 "no a la regla.",
        entrada="Igual que v1, con PCR_VOL y minimo 500 de volumen por ventana.",
        salida="Igual que v1.",
        sizing="Igual que la sectorial v1.",
        insumos="Los de la v1 + opciones_pcr_plazo_diario (PCR_VOL).",
        estado="Activa. Es la que replicaba el Bot 3 de Alpaca.",
    ),
    "tecnico_sectorial_oiexit_v1": Logica(
        titulo="Tecnico sectorial con salida por muro de open interest",
        pregunta="Un stop puesto donde el mercado de opciones tiene su muro de "
                 "puts protege mejor que un stop de volatilidad?",
        concepto="La ENTRADA es identica a la sectorial v1 (no toca opciones): "
                 "todo el cambio esta en la SALIDA. El stop se pone en el put "
                 "wall, con proteccion escalonada por R (multiplos del riesgo "
                 "inicial) y un backstop de Chandelier mas un quorum de 4 votos.",
        entrada="score >= 4,0, igual que la v1. Sin gate de opciones.",
        salida="SL = put wall de la ventana media (zona -10%, liquidez relativa; "
               "fallback 2xATR). Breakeven en +1R, asegura +1R en +2R, +3R libera "
               "el techo. Fase 2 desde +2R: Chandelier 2,5xATR y quorum de 3 de 4 "
               "(cierre < SMA21 / candle_5d <= -2 / PCR_VOL mayoria bajista / "
               "divergencia de volumen).",
        sizing="Igual que la sectorial v1.",
        insumos="Los de la v1 + muros de OI + candle_score + vol_price_diverge.",
        estado="Activa. Es la estrategia con mas parametros del proyecto (23): la "
               "mas expuesta a sobreajuste si se tocan mirando resultados.",
    ),
    "ml_scanner": Logica(
        titulo="Scanner ML",
        pregunta="La probabilidad del modelo, convertida en un nivel de alerta, "
                 "selecciona acciones que suben mas que el universo?",
        concepto="La unica estrategia que NO decide con reglas: delega en el score "
                 "compuesto del scanner (base 50 + ML + price action + rule-based "
                 "- bajistas) y solo opera el nivel maximo. Los stops son "
                 "PORCENTUALES fijos, no por volatilidad: es la unica asi.",
        entrada="alert_nivel = COMPRA_FUERTE y alert_score >= 65 (el nivel ya "
                "exige >= 75, asi que el 65 no recorta nada: el nivel manda). "
                "Top-N por score que entren en cash.",
        salida="Prioridad: balance manana -> score degradado (sin senal, o nivel "
               "distinto de COMPRA_FUERTE, o score < 65) -> stop -5% -> "
               "objetivo +10%.",
        sizing="min(capital x 15%, cash) / precio, maximo 5 posiciones. La v2 "
               "agrega techo de despliegue del 80%.",
        insumos="alertas_scanner (la v1 lee alert_nivel/alert_score; la v2 las "
                "columnas _v2 del mismo registro).",
        estado="Activa, dos instancias en paralelo (v1 como control, v2 con el "
               "modelo nuevo). OJO: el 'AUC 0,65' del modelo era look-ahead; en "
               "vivo el resultado de FT es la medicion honesta.",
    ),
    "smc_estructura": Logica(
        titulo="SMC sobre estructura (version original)",
        pregunta="Un giro o una ruptura de estructura, con el stop en el swing, da "
                 "una relacion riesgo-beneficio mejor que un stop de volatilidad?",
        concepto="Smart Money Concepts: no mira medias sino la secuencia de "
                 "maximos y minimos. Entra cuando la estructura se rompe al alza "
                 "(BOS) o gira (CHoCH) y pone el stop donde la idea se "
                 "invalidaria: el ultimo swing low. NO TIENE OBJETIVO: deja correr "
                 "con trailing stop estructural.",
        entrada="Las 5 obligatorias: CHoCH bull O BOS bull en 12 dias; "
                "estructura_10 >= 0; sin choch_bear hoy; vela alcista; distancia "
                "al swing low entre 1% y 8%. El score 0-3 solo RANKEA la calidad.",
        salida="Trailing stop al swing low (se sube, nunca baja), time stop de 20 "
               "dias, balance. Sin take profit.",
        sizing="15% del capital por operacion, maximo 5 posiciones.",
        insumos="features_market_structure (la tabla cuya HISTORIA mira al futuro) "
                "+ patrones viejos de features_precio_accion.",
        estado="Activa COMO CONTROL. Se deja a proposito sobre los insumos viejos: "
               "es contra ella que se mide la v3.",
    ),
    "smc_estructura_v2": Logica(
        titulo="SMC con filtro de contexto y salida por agotamiento",
        pregunta="Filtrar la entrada por contexto y salir por agotamiento en vez "
                 "de por tiempo, mejora la v1?",
        concepto="Tres filtros nuevos sobre la v1 (dos en entrada por OR, uno de "
                 "salida por AND) y el time stop eliminado.",
        entrada="La de la v1 + (lateral_ratio > 1,0 O candle_score > 0).",
        salida="Trailing estructural + agotamiento: up_vol = 0 Y candle < -2 Y "
               "lateral < 0,5. Sin time stop.",
        sizing="Igual que la v1.",
        insumos="Los de la v1 + velas y lateralidad.",
        estado="DADA DE BAJA el 17/9/2026: la peor equity de las 11 (-5,46%, "
               "-6.622 USD en 27 operaciones) y sus 3 filtros se apoyan en velas y "
               "estructura sin confirmar. La salida por agotamiento NO DISPARO NI "
               "UNA VEZ en 94 ruedas: una regla que nunca dispara no es una regla.",
    ),
    "combo_tech_candle": Logica(
        titulo="Tecnico sectorial + velas como desempate",
        pregunta="Los patrones de vela aportan algo como criterio de desempate "
                 "sobre el score tecnico?",
        concepto="La sectorial v1 con un segundo criterio de orden: a igual score "
                 "tecnico, gana el que tenga mejor score de velas de 5 ruedas. Y "
                 "excluye los que estan en distribucion activa.",
        entrada="score >= 4,0 Y candle_score_5d >= -3. Orden: tecnico DESC, "
                "despues velas DESC.",
        salida="Igual que la sectorial v1.",
        sizing="Igual que la sectorial v1.",
        insumos="Los de la v1 + candle_score_5d.",
        estado="DADA DE BAJA el 17/9/2026: las velas no aportan. Backtest "
               "2021-09 a 2026-09 COMBO +22,3% contra +24,4% de la sectorial SIN "
               "velas, y en vivo -0,10% contra 0,00%. Coherente con la medicion de "
               "fondo: ningun patron de vela anticipa retorno.",
    ),
    "smc_estructura_confirmada": Logica(
        titulo="SMC sobre estructura CONFIRMADA",
        pregunta="La misma regla de SMC, alimentada con swings que no miran al "
                 "futuro, sigue funcionando?",
        concepto="Identica a SMC_v1 en la REGLA (el score se importa, no se "
                 "reimplementa) y distinta en el INSUMO: lee features_estructura y "
                 "features_velas, donde el swing existe N barras despues de su "
                 "barra. Es la unica version de SMC cuya historia es reproducible.",
        entrada="La misma de la v1, con swings confirmados a N ruedas. El lookback "
                "se ancla a la ultima rueda de DATOS, no al reloj.",
        salida="La misma de la v1: trailing estructural, time stop 20 dias, sin TP.",
        sizing="Igual que la v1.",
        insumos="features_estructura + features_velas (las tablas sin futuro).",
        estado="Activa, DOS instancias en paralelo (N=5 y N=3). Las dos porque el "
               "backtest no las distingue (+59,8% y +58,2%) y un parametro no se "
               "elige mirando resultados: decide la operacion real. Con N=10 "
               "confirmado da +13,5%, o sea que la ventaja del original dependia "
               "del futuro.",
    ),
}


# ============================================================================
# HOJA SCORES -- como se arma cada numero que decide
# ============================================================================

SCORE_TECNICO: Tuple[FilaScore, ...] = (
    FilaScore("Capa 1 -- filtro", "close > SMA200", "obligatoria",
              "Si NO se cumple, el score es 0 y no hay entrada posible. No suma "
              "puntos: habilita."),
    FilaScore("Capa 2 -- tendencia", "close > SMA50", "+2,0",
              "La condicion que mas pesa."),
    FilaScore("Capa 2 -- tendencia", "close > SMA21", "+1,0", ""),
    FilaScore("Capa 3 -- momentum", "MACD > signal Y macd_hist > 0", "+1,5",
              "Las dos condiciones son la misma cosa (hist = MACD - signal): "
              "redundancia preservada para no cambiar el historico."),
    FilaScore("Capa 3 -- momentum", "RSI14 entre 45 y 68", "+1,0",
              "Zona de tendencia sana. Es el umbral OPUESTO al del score "
              "ponderado viejo, que premia RSI < 35."),
    FilaScore("TOTAL", "Maximo alcanzable", "5,5",
              "Entrada >= 4,0 / salida <= 3,5. NINGUN score cae entre 3,5 y 4,0: "
              "el umbral de salida es exactamente 'dejo de cumplir la entrada', y "
              "por eso mover los PESOS no mueve ninguna salida (medido: 589 reglas, "
              "0 mejoran)."),
)

SCORE_SMC: Tuple[FilaScore, ...] = (
    FilaScore("Obligatoria 1", "CHoCH bull O BOS bull en el lookback (12 dias)",
              "obligatoria", "Es la puerta: sin ruptura ni giro no hay setup."),
    FilaScore("Obligatoria 2", "estructura_10 >= 0", "obligatoria",
              "La estructura no esta rota a la baja."),
    FilaScore("Obligatoria 3", "choch_bear_10 = 0", "obligatoria",
              "No hubo giro bajista hoy."),
    FilaScore("Obligatoria 4", "es_alcista = 1", "obligatoria",
              "La vela de hoy cierra sobre su apertura."),
    FilaScore("Obligatoria 5", "distancia al swing low entre 1% y 8%", "obligatoria",
              "Define DONDE va el stop. Muy cerca es ruido; muy lejos es un "
              "riesgo que no cabe en la posicion."),
    FilaScore("Calidad", "CHoCH bull detectado", "+1",
              "El giro vale mas que la continuacion."),
    FilaScore("Calidad", "vol_spike O engulfing bull O hammer", "+1",
              "Confirmacion de vela o volumen. OJO: usa los patrones VIEJOS, que "
              "estan mal definidos."),
    FilaScore("Calidad", "estructura_10 = +1", "+1", "Estructura claramente alcista."),
    FilaScore("TOTAL", "Maximo alcanzable", "3",
              "El score NO es un umbral: si las 5 obligatorias se cumplen hay "
              "entrada. Los 0-3 puntos solo ORDENAN los candidatos. Score -1 = "
              "no cumple alguna obligatoria."),
)

SCORE_SCANNER: Tuple[FilaScore, ...] = (
    FilaScore("Base", "Punto de partida", "50",
              "Sin ninguna senal, el score es 50 = NEUTRAL."),
    FilaScore("ML", "P(ganancia) >= 0,75", "+30", "ML fuerte alcista."),
    FilaScore("ML", "P >= 0,65", "+22", ""),
    FilaScore("ML", "P >= 0,55", "+14", ""),
    FilaScore("ML", "P >= 0,45", "+5", "ML neutro."),
    FilaScore("ML", "P >= 0,35", "-5", ""),
    FilaScore("ML", "P < 0,35", "-15",
              "El ML es el componente que mas mueve el score: 45 puntos de rango."),
    FilaScore("Price action", "3 o mas condiciones EV cumplidas", "+15",
              "Alta confluencia."),
    FilaScore("Price action", "2 condiciones EV", "+10", ""),
    FilaScore("Price action", "1 condicion EV", "+5", ""),
    FilaScore("Price action", "ninguna", "0", ""),
    FilaScore("Rule-based", "score_ponderado >= 0,75", "+10", ""),
    FilaScore("Rule-based", "score_ponderado >= 0,60", "+7", ""),
    FilaScore("Rule-based", "score_ponderado >= 0,45", "+3", ""),
    FilaScore("Rule-based", "score_ponderado >= 0,30", "-2", ""),
    FilaScore("Rule-based", "score_ponderado < 0,30", "-5", ""),
    FilaScore("Bajistas", "BOS bajista en 10 ruedas", "-12", ""),
    FilaScore("Bajistas", "CHoCH bajista en 10 ruedas", "-12", ""),
    FilaScore("Bajistas", "estructura_10 bajista", "-8",
              "Las tres penalizaciones se ACUMULAN: hasta -32."),
    FilaScore("NIVELES", "COMPRA_FUERTE >= 75 | COMPRA >= 60 | NEUTRAL >= 40 | "
                         "VENTA >= 25 | VENTA_FUERTE < 25", "corte",
              "Las estrategias ML solo operan COMPRA_FUERTE. Para llegar a 75 "
              "desde la base 50 hace falta practicamente el ML fuerte: el modelo "
              "es, de hecho, el que decide."),
)

SCORE_PONDERADO: Tuple[FilaScore, ...] = (
    FilaScore("Condicion", "RSI14 < 35 (sobreventa)", "0,20",
              "ATENCION: premia el RSI BAJO, al reves del score tecnico de FT, que "
              "premia 45-68. Son dos hipotesis opuestas (rebote contra tendencia) "
              "conviviendo en el mismo pipeline."),
    FilaScore("Condicion", "macd_hist > 0", "0,20", ""),
    FilaScore("Condicion", "close > SMA21", "0,10", ""),
    FilaScore("Condicion", "close > SMA50", "0,15", ""),
    FilaScore("Condicion", "close > SMA200", "0,20",
              "Aca la SMA200 es UN PESO MAS, no un filtro obligatorio como en el "
              "score tecnico."),
    FilaScore("Condicion", "momentum > 0", "0,15", ""),
    FilaScore("TOTAL", "Los pesos suman 1,0", "1,00",
              "Senal LONG >= 0,60, cierre <= 0,30 (config.py). Hoy su unico "
              "consumo real es aportar hasta 10 puntos al score del scanner."),
)

SCORE_PCR: Tuple[FilaScore, ...] = (
    FilaScore("Ventana corta", "PCR de vencimientos a 1-14 dias < 1,0", "+1",
              "Alcista."),
    FilaScore("Ventana media", "PCR de vencimientos a 15-45 dias < 1,0", "+1", ""),
    FilaScore("Ventana larga", "PCR de vencimientos a 46-90 dias < 1,0", "+1", ""),
    FilaScore("Validez", "OI (o volumen) total de la ventana >= 500", "requisito",
              "Una ventana sin liquidez NO VOTA: queda invalida y no suma."),
    FilaScore("TOTAL", "Maximo alcanzable", "3",
              "Entrada de OPTIONS v1/v2 con >= 2 de 3. Cierre si = 0 junto con "
              "score tecnico degradado; retencion si >= 1."),
)

SCORE_EV: Tuple[FilaScore, ...] = (
    FilaScore("EV1", "estructura_10 = 1 Y dias_sl_10 <= 15 Y (engulfing bull O "
                     "hammer) Y (vol_spike O up_vol_5d > 0,55)", "1 condicion",
              "Retroceso a swing reciente con patron y volumen."),
    FilaScore("EV2", "bos_bull_10 = 1 Y vol_price_confirm = 1 Y estructura_10 >= 0",
              "1 condicion", "Ruptura de continuacion confirmada por volumen."),
    FilaScore("EV3", "choch_bull_10 = 1 Y vol_spike = 1 Y dist_sl_10_pct > -4,0",
              "1 condicion", "Giro con pico de volumen."),
    FilaScore("EV4", "estructura_10 = 1 Y dist_sl_10_pct entre 0 y 4 Y es_alcista = 1 "
                     "Y dias_sl_10 <= 10", "1 condicion",
              "Pegado al swing low, con vela alcista."),
    FilaScore("USO", "Cuantas se cumplen", "0 a 4",
              "No son estrategias que operen: alimentan el componente de price "
              "action del score del scanner. Cualquier feature que falte o sea "
              "NaN hace que la condicion de False."),
)

SCORES: Dict[str, Tuple[FilaScore, ...]] = {
    "Score tecnico (SMA/MACD/RSI, max 5,5)": SCORE_TECNICO,
    "Score estructural SMC (5 obligatorias + calidad 0-3)": SCORE_SMC,
    "Score del scanner (0-100)": SCORE_SCANNER,
    "Score ponderado rule-based (0-1)": SCORE_PONDERADO,
    "pcr_score de opciones (0-3)": SCORE_PCR,
    "Condiciones EV1-EV4 de price action": SCORE_EV,
}


# ============================================================================
# PARAMETROS -- que significa cada clave de ft_estrategias.parametros
# ============================================================================
# El VALOR de cada parametro NO esta aca: lo lee el exportador de la DB, que es
# lo que de verdad corrio. Aca vive solo el significado.

PARAMETROS: Dict[str, str] = {
    # -- Scores y umbrales de decision --
    "score_entrada": "Score tecnico minimo para abrir (de 5,5).",
    "score_salida": "Score tecnico que dispara el cierre. 3,5 = dejo de cumplir "
                    "la entrada.",
    "score_maximo": "Tope teorico del score. Referencia, no decide nada.",
    "score_entrada_min": "Score minimo de entrada (SMC: las obligatorias ya "
                         "decidieron; esto solo pide calidad >= N).",
    "score_min": "Score minimo del scanner (0-100) para abrir.",
    "nivel_min": "Nivel de alerta minimo. COMPRA_FUERTE ya exige score >= 75.",
    "rsi_min": "Piso de la banda de RSI que suma punto (45).",
    "rsi_max": "Techo de la banda de RSI que suma punto (68): evita comprar "
               "sobrecompra.",
    "tech_score_entrada_min": "Gate 1 de las variantes con opciones: score tecnico.",
    "tech_score_salida_max": "Score tecnico por debajo del cual se evalua cerrar, "
                             "junto con el PCR.",
    "rotacion_delta_score": "Ventaja de score que tiene que tener un candidato "
                            "nuevo para desplazar a una posicion abierta en un "
                            "sector lleno.",

    # -- Stops y objetivos --
    "atr_mult_sl": "Multiplo de ATR para el stop loss (2,0 = entrada - 2xATR).",
    "atr_mult_tp": "Multiplo de ATR para el objetivo (4,0 = riesgo-beneficio 1:2).",
    "sl_pct": "Stop loss PORCENTUAL fijo, no por volatilidad (0,05 = -5%).",
    "tp_pct": "Take profit porcentual fijo (0,10 = +10%).",
    "min_sl_dist_pct": "Distancia MINIMA al swing low para aceptar el setup: mas "
                       "cerca es ruido de mercado.",
    "max_sl_dist_pct": "Distancia MAXIMA: mas lejos, el riesgo no cabe en la "
                       "posicion.",
    "sl_fallback_atr_mult": "Multiplo de ATR del stop cuando no hay muro de OI "
                            "utilizable.",
    "chandelier_atr_mult": "Multiplo de ATR del backstop Chandelier (desde el "
                           "maximo alcanzado).",
    "dias_max_pos": "Time stop: ruedas maximas en posicion.",
    "lookback_dias": "Ventana en dias CALENDARIO para buscar el CHoCH/BOS "
                     "(12 ~ 10 habiles).",
    "lookback_ancla": "A que fecha se ancla el lookback. Tiene que ser la ultima "
                      "rueda de DATOS, no el reloj.",

    # -- Proteccion por R (OIEXIT) --
    "r_breakeven": "En cuantas R de ganancia el stop se mueve al precio de entrada.",
    "r_lock": "En cuantas R se asegura +1R.",
    "r_libera": "En cuantas R se libera el techo y se deja correr.",
    "r_piso_fase2": "R desde la cual se activa la fase 2 (Chandelier + quorum).",
    "quorum_min": "Cuantos de los 4 votos de deterioro hacen falta para cerrar.",
    "sma_salida": "Que SMA se usa como voto de salida (cierre por debajo).",
    "candle_salida_max": "Score de velas por debajo del cual el voto de velas "
                         "cuenta como deterioro.",
    "pcr_vol_salida_max": "Cuantas ventanas alcistas como maximo para que el voto "
                          "de opciones cuente como bajista.",
    "div_dist_max20_min": "Distancia al maximo de 20 ruedas desde la que se "
                          "considera que el precio esta extendido.",
    "div_vol_ratio_max": "Ratio de volumen por debajo del cual la suba se lee "
                         "como divergente.",

    # -- Opciones --
    "pcr_score_entrada_min": "Ventanas alcistas minimas (de 3) para habilitar "
                             "entrada.",
    "pcr_score_salida_max": "Ventanas alcistas maximas para cerrar junto con score "
                            "degradado.",
    "pcr_score_retencion_min": "Ventanas alcistas minimas para RETENER a pesar del "
                               "score degradado.",
    "pcr_oi_umbral_alcista": "PCR por open interest por debajo del cual la ventana "
                             "es alcista (1,0).",
    "pcr_vol_umbral_alcista": "Lo mismo sobre volumen diario.",
    "min_oi_por_ventana": "Open interest minimo para que la ventana vote.",
    "min_vol_por_ventana": "Volumen minimo para que la ventana vote.",
    "ventana_corto_min": "Dias al vencimiento: piso de la ventana corta.",
    "ventana_corto_max": "Dias al vencimiento: techo de la ventana corta.",
    "ventana_medio_min": "Piso de la ventana media.",
    "ventana_medio_max": "Techo de la ventana media.",
    "ventana_largo_min": "Piso de la ventana larga.",
    "ventana_largo_max": "Techo de la ventana larga.",
    "wall_ventana_min": "Ventana de vencimientos donde se busca el muro (piso).",
    "wall_ventana_max": "Ventana de vencimientos donde se busca el muro (techo).",
    "wall_zona_pct": "Que tan lejos del precio se busca el muro.",
    "wall_liq_mult_mediana": "Criterio viejo de validez del muro (multiplo de la "
                             "mediana de OI).",
    "wall_liq_min_abs": "Piso de open interest absoluto para que el muro valga.",
    "wall_dist_min_pct": "Distancia minima del muro al precio: evita poner el stop "
                         "en el strike at-the-money.",

    # -- Capital y ocupacion --
    "max_posiciones": "Posiciones abiertas simultaneas como maximo.",
    "max_pos_sector": "Posiciones maximas por sector.",
    "position_pct": "Fraccion del presupuesto SECTORIAL por posicion.",
    "riesgo_por_trade": "Fraccion del capital por operacion (0,15 = 15%). No es "
                        "riesgo en el sentido de perdida maxima: es tamano.",
    "max_deploy_pct": "Techo de capital desplegado a la vez.",
    "sectores": "Los 9 sectores entre los que se reparte el capital.",
    "n_sectores": "Cuantos son: el presupuesto sectorial es capital / n_sectores.",

    # -- Velas, estructura y contexto --
    "candle_score_min": "Score de velas minimo para no excluir al candidato.",
    "candle_ventana_d": "Ruedas de la ventana del score de velas.",
    "retencion_candle_min": "Score de velas desde el cual se retiene a pesar del "
                            "score tecnico.",
    "retencion_up_vol_min": "up_vol_5d desde el cual se retiene.",
    "retencion_logica": "Como se combinan las condiciones de retencion (OR = "
                        "alcanza una).",
    "filtro_lateral_ratio_min": "Lateralidad minima del contexto para aceptar la "
                                "entrada.",
    "filtro_candle_score_min": "Score de velas minimo del contexto de entrada.",
    "filtro_logica": "Como se combinan los filtros de entrada.",
    "agotamiento_up_vol_max": "up_vol maximo para considerar agotamiento.",
    "agotamiento_candle_max": "Score de velas maximo para considerar agotamiento.",
    "agotamiento_lateral_max": "Lateralidad maxima para considerar agotamiento.",
    "agotamiento_logica": "Como se combinan (AND = tienen que darse todas, y por "
                          "eso no disparo nunca).",
    "ventana_confirmacion": "N de ruedas con las que se confirma un swing. Cuanto "
                            "mas chico, mas rapido reacciona y mas opera.",
    "fuente_estructura": "Que tabla de estructura se lee. La diferencia entre "
                         "mirar o no al futuro.",
    "fuente_velas": "Que tabla de patrones de vela se lee.",

    # -- Modelo --
    "senal": "Que columnas de alertas_scanner se leen.",
    "modelo": "Artefacto del modelo con el que se calculo la senal.",
}


# ============================================================================
# HOJAS 3..N -- que indicador entra en cada logica, y con que parametro
# ============================================================================
# Una fila por INSUMO de la decision. `params` son las claves de
# ft_estrategias.parametros que gobiernan ese insumo; el exportador les pone al
# lado el valor que tiene la instancia en la DB. Las filas con params=() son
# insumos sin parametro configurable (el dato entra o no entra).


class FilaUso(NamedTuple):
    """Un insumo de una estrategia, con los parametros que lo gobiernan."""
    indicador: str
    params: Tuple[str, ...]
    rol: str


# -- Bloques reutilizables (las sectoriales comparten casi todo) --------------

_SCORE_TECNICO_FILAS: Tuple[FilaUso, ...] = (
    FilaUso("SMA200", ("score_entrada",),
            "FILTRO OBLIGATORIO. close <= SMA200 -> score 0 -> no hay entrada "
            "posible por mas que todo lo demas este a favor."),
    FilaUso("SMA50", (),
            "+2,0 puntos si close > SMA50. Es la condicion de mayor peso del "
            "score."),
    FilaUso("SMA21", (),
            "+1,0 punto si close > SMA21."),
    FilaUso("MACD + macd_signal + macd_hist", (),
            "+1,5 puntos si MACD > signal Y hist > 0 (la segunda condicion es "
            "redundante con la primera)."),
    FilaUso("RSI14", ("rsi_min", "rsi_max"),
            "+1,0 punto si el RSI cae DENTRO de la banda. Es un filtro de "
            "sobrecompra, no de sobreventa: arriba de rsi_max no suma."),
    FilaUso("Score tecnico (resultado)", ("score_entrada", "score_salida",
                                          "score_maximo"),
            "La suma de las 4 condiciones sobre el maximo. Entre score_salida y "
            "score_entrada no cae NINGUN score posible: el umbral de salida es "
            "dejar de cumplir la entrada."),
)

_STOPS_ATR: Tuple[FilaUso, ...] = (
    FilaUso("ATR14", ("atr_mult_sl", "atr_mult_tp"),
            "Unidad de volatilidad de los stops: SL = entrada - mult x ATR, "
            "TP = entrada + mult x ATR. Con 2 y 4 la relacion es 1:2."),
)

_EARNINGS: Tuple[FilaUso, ...] = (
    FilaUso("earnings_calendar", (),
            "Cierre con PRIORIDAD ABSOLUTA si el balance es manana: ninguna "
            "estrategia se queda sobre un balance. Sin parametro: es si/no."),
)

_SECTOR: Tuple[FilaUso, ...] = (
    FilaUso("activos.sector", ("sectores", "n_sectores", "max_pos_sector",
                               "position_pct"),
            "Particiona el capital: presupuesto = capital / n_sectores, y cada "
            "posicion toma position_pct de ESE presupuesto. El ticker compite "
            "solo contra sus pares."),
)

_CAPITAL_PCT: Tuple[FilaUso, ...] = (
    FilaUso("(capital, no es un indicador)", ("max_posiciones", "riesgo_por_trade",
                                              "max_deploy_pct"),
            "Tamano y ocupacion. riesgo_por_trade es TAMANO de la posicion, no "
            "perdida maxima: con 15% y stop de 5% la perdida por operacion es "
            "~0,75% del capital."),
)

_SMC_FILAS: Tuple[FilaUso, ...] = (
    FilaUso("choch_bull_N", ("lookback_dias",),
            "PUERTA DE ENTRADA (giro). Se busca en la ventana del lookback y "
            "suma +1 de calidad: el giro vale mas que la continuacion."),
    FilaUso("bos_bull_N", ("lookback_dias",),
            "La otra puerta (continuacion). Alcanza con una de las dos."),
    FilaUso("estructura_N", (),
            "OBLIGATORIA: >= 0, la estructura no esta rota a la baja. Si es "
            "exactamente +1 suma +1 de calidad."),
    FilaUso("choch_bear_N", (),
            "VETO: un giro bajista hoy cancela la entrada."),
    FilaUso("es_alcista", (),
            "OBLIGATORIA: la vela de hoy cierra sobre su apertura."),
    FilaUso("dist_sl_N_pct", ("min_sl_dist_pct", "max_sl_dist_pct"),
            "DEFINE EL STOP: el SL va al swing low. Fuera de la banda el setup "
            "se descarta -- muy cerca es ruido, muy lejos no cabe en la "
            "posicion. Es el parametro que mas filtra operaciones."),
    FilaUso("vol_spike / patron_engulfing_bull / patron_hammer", (),
            "+1 de calidad si cualquiera esta presente. OJO: en v1 y v2 son los "
            "patrones VIEJOS, que estan mal definidos."),
    FilaUso("Swing low (trailing)", (),
            "La salida: el stop se MUEVE al nuevo swing low y nunca baja. No hay "
            "take profit: la estrategia deja correr."),
    FilaUso("Score de calidad (resultado)", ("score_entrada_min", "score_maximo"),
            "0 a 3. NO es un umbral de entrada (las 5 obligatorias ya "
            "decidieron): solo ORDENA los candidatos cuando sobran."),
)


def _pcr_filas(umbral: str, minimo: str, dato: str) -> Tuple[FilaUso, ...]:
    """Las filas del gate de opciones (identicas en v1/OI y v2/VOL)."""
    return (
        FilaUso(dato + " ventana corta", ("ventana_corto_min", "ventana_corto_max",
                                          umbral, minimo),
                "Vencimientos mas cercanos. Vale +1 al pcr_score si esta debajo "
                "del umbral; sin liquidez minima NO VOTA."),
        FilaUso(dato + " ventana media", ("ventana_medio_min", "ventana_medio_max",
                                          umbral, minimo),
                "Vencimientos intermedios: +1 al pcr_score."),
        FilaUso(dato + " ventana larga", ("ventana_largo_min", "ventana_largo_max",
                                          umbral, minimo),
                "Vencimientos lejanos: +1 al pcr_score."),
        FilaUso("pcr_score (resultado)", ("pcr_score_entrada_min",
                                          "pcr_score_salida_max",
                                          "pcr_score_retencion_min"),
                "0 a 3. GATE 2 de entrada; en la salida decide junto con el "
                "score tecnico: cierra si llego a 0, RETIENE si queda >= 1."),
    )


USO: Dict[str, Tuple[FilaUso, ...]] = {
    "tecnico": (
        _SCORE_TECNICO_FILAS + _STOPS_ATR + _EARNINGS + _CAPITAL_PCT
    ),
    "tecnico_sectorial": (
        _SCORE_TECNICO_FILAS + _STOPS_ATR + _EARNINGS + _SECTOR + (
            FilaUso("(desempate, no es un indicador)", (),
                    "A igual score ordena el ALFABETO (sort estable sobre ORDER "
                    "BY sector, ticker) y eso decide el 42,8% de los "
                    "sector-ruedas. Es un parametro OCULTO: ver la hoja Metodo."),
        )
    ),
    "tecnico_sectorial_v2": (
        _SCORE_TECNICO_FILAS + _STOPS_ATR + _EARNINGS + _SECTOR + (
            FilaUso("candle_score_5d", ("retencion_candle_min", "retencion_logica"),
                    "RETENCION: no cierra por score degradado si las velas "
                    "todavia acompanan."),
            FilaUso("up_vol_5d", ("retencion_up_vol_min", "retencion_logica"),
                    "La otra condicion de retencion. Con logica OR alcanza una "
                    "de las dos para quedarse."),
            FilaUso("Score tecnico de los candidatos", ("rotacion_delta_score",),
                    "ROTACION: en un sector lleno, un candidato con esta ventaja "
                    "de score desplaza a una posicion abierta."),
        )
    ),
    "tecnico_sectorial_options_v1": (
        _SCORE_TECNICO_FILAS[:-1] + (
            FilaUso("Score tecnico (resultado)", ("tech_score_entrada_min",
                                                  "tech_score_salida_max"),
                    "GATE 1. Pasarlo no alcanza: despues decide el PCR."),
        ) + _pcr_filas("pcr_oi_umbral_alcista", "min_oi_por_ventana",
                       "PCR por OPEN INTEREST")
        + _STOPS_ATR + _EARNINGS + _SECTOR + (
            FilaUso("(orden de cierre)", (),
                    "A diferencia de la sectorial v1, el stop y el objetivo se "
                    "evaluan ANTES del score: balance -> SL -> TP -> score+PCR."),
        )
    ),
    "tecnico_sectorial_options_v2": (
        _SCORE_TECNICO_FILAS[:-1] + (
            FilaUso("Score tecnico (resultado)", ("tech_score_entrada_min",
                                                  "tech_score_salida_max"),
                    "GATE 1, identico a OPTIONS_v1."),
        ) + _pcr_filas("pcr_vol_umbral_alcista", "min_vol_por_ventana",
                       "PCR por VOLUMEN diario")
        + _STOPS_ATR + _EARNINGS + _SECTOR + (
            FilaUso("(diferencia con OPTIONS_v1)", (),
                    "UNA sola: el PCR se calcula sobre volumen diario (flujo del "
                    "dia) en vez de open interest (posicion acumulada). Todo lo "
                    "demas es igual, para que la diferencia sea atribuible al "
                    "dato."),
        )
    ),
    "tecnico_sectorial_oiexit_v1": (
        _SCORE_TECNICO_FILAS + (
            FilaUso("Muro de OI (put wall)", ("wall_ventana_min", "wall_ventana_max",
                                              "wall_zona_pct", "wall_liq_min_abs",
                                              "wall_liq_mult_mediana",
                                              "wall_dist_min_pct"),
                    "EL STOP: se pone donde el mercado de opciones tiene su muro "
                    "de puts, no a una distancia de volatilidad."),
            FilaUso("ATR14 (fallback)", ("sl_fallback_atr_mult",),
                    "Si no hay muro utilizable, el stop vuelve a ser de "
                    "volatilidad."),
            FilaUso("R (multiplo del riesgo inicial)", ("r_breakeven", "r_lock",
                                                        "r_libera", "r_piso_fase2"),
                    "Proteccion escalonada: breakeven, asegurar +1R, liberar el "
                    "techo, y desde r_piso_fase2 se activa la fase 2."),
            FilaUso("ATR14 (Chandelier)", ("chandelier_atr_mult",),
                    "Backstop desde el maximo alcanzado, ya en fase 2."),
            FilaUso("SMA21 (voto)", ("sma_salida",),
                    "Voto 1 del quorum: cierre por debajo de la media."),
            FilaUso("candle_score_5d (voto)", ("candle_salida_max",),
                    "Voto 2: velas deterioradas."),
            FilaUso("PCR_VOL (voto)", ("pcr_vol_salida_max",),
                    "Voto 3: mayoria de ventanas bajistas."),
            FilaUso("vol_price_diverge + dist_max_20d (voto)",
                    ("div_dist_max20_min", "div_vol_ratio_max"),
                    "Voto 4: precio extendido subiendo con volumen que cae."),
            FilaUso("Quorum (resultado)", ("quorum_min",),
                    "Cuantos votos hacen falta para cerrar. Con 3 de 4, un solo "
                    "deterioro no saca la posicion."),
        ) + _EARNINGS + _SECTOR
    ),
    "ml_scanner": (
        FilaUso("ml_prob_ganancia", (),
                "El insumo que decide: P(retorno_20d > +1%). Aporta de +30 a -15 "
                "al score del scanner. La estrategia NO la lee directo: le llega "
                "ya convertida en nivel."),
        FilaUso("alert_nivel", ("nivel_min", "senal"),
                "ENTRADA. COMPRA_FUERTE exige score >= 75 por si mismo."),
        FilaUso("alert_score", ("score_min", "senal"),
                "Orden de los candidatos (top-N por score) y segunda condicion. "
                "OJO: score_min=65 NO RECORTA NADA, porque el nivel ya pide 75."),
        FilaUso("Modelo", ("modelo",),
                "Que artefacto produjo la probabilidad. Es lo UNICO que "
                "diferencia a la v2 de la v1."),
        FilaUso("Precio de entrada", ("sl_pct", "tp_pct"),
                "Stops PORCENTUALES fijos sobre el precio de entrada, no por "
                "volatilidad: es la unica estrategia asi. Un -5% es muy distinto "
                "en un ticker tranquilo que en uno volatil."),
        FilaUso("Score degradado", ("nivel_min", "score_min"),
                "SALIDA: si el ticker deja de aparecer, baja de nivel o cae del "
                "score. Es la salida que mas cierra."),
    ) + _EARNINGS + _CAPITAL_PCT,
    "smc_estructura": _SMC_FILAS + (
        FilaUso("Fuente de los datos", (),
                "features_market_structure + patrones viejos. En VIVO no engana "
                "(lee la ultima fila), pero su historia mira 10 ruedas al futuro: "
                "por eso esta instancia queda como CONTROL y no se backtestea."),
        FilaUso("Time stop", ("dias_max_pos",),
                "Cierra por tiempo. Medido: bajarlo de 20 a 10 mejora en 2021-24 "
                "con IC que excluye el cero y va al mismo lado en 2025-26 sin "
                "alcanzar. HIPOTESIS SIN CONFIRMAR."),
    ) + _EARNINGS + _CAPITAL_PCT,
    "smc_estructura_v2": _SMC_FILAS + (
        FilaUso("lateral_ratio / candle_score (filtro)",
                ("filtro_lateral_ratio_min", "filtro_candle_score_min",
                 "filtro_logica"),
                "FILTRO DE CONTEXTO en la entrada, con logica OR."),
        FilaUso("up_vol / candle_score / lateral_ratio (agotamiento)",
                ("agotamiento_up_vol_max", "agotamiento_candle_max",
                 "agotamiento_lateral_max", "agotamiento_logica"),
                "SALIDA POR AGOTAMIENTO con logica AND: las tres juntas. NO "
                "DISPARO NI UNA VEZ en 94 ruedas -- con AND de tres condiciones "
                "extremas, la regla era inalcanzable."),
        FilaUso("Time stop", (),
                "ELIMINADO en esta version. Sin time stop y sin que el "
                "agotamiento disparara, la unica salida real era el trailing."),
    ) + _EARNINGS + _CAPITAL_PCT,
    "smc_estructura_confirmada": _SMC_FILAS + (
        FilaUso("Fuente de los datos", ("fuente_estructura", "fuente_velas"),
                "LA DIFERENCIA con SMC_v1: tablas sin informacion futura, con "
                "patrones de vela clasicos. La regla es la MISMA (el score se "
                "importa, no se reimplementa)."),
        FilaUso("Ventana de confirmacion", ("ventana_confirmacion",),
                "N ruedas para que un swing exista. Mas chico = reacciona antes "
                "y opera mas. N=5 y N=3 corren en PARALELO porque el backtest no "
                "las distingue (+59,8% y +58,2%); con N=10 cae a +13,5%."),
        FilaUso("Ancla del lookback", ("lookback_ancla",),
                "El lookback cuelga de la ultima rueda de DATOS, no del reloj: "
                "con la rutina atrasada, anclar al reloj leeria una ventana "
                "corrida."),
        FilaUso("Time stop", ("dias_max_pos",),
                "Igual que la v1: 20 ruedas."),
    ) + _EARNINGS + _CAPITAL_PCT,
    "combo_tech_candle": (
        _SCORE_TECNICO_FILAS + (
            FilaUso("candle_score_5d", ("candle_score_min", "candle_ventana_d"),
                    "DESEMPATE: a igual score tecnico gana el de mejores velas, y "
                    "quedan excluidos los que estan debajo del minimo "
                    "(distribucion activa). ESTO ES LO QUE SE MIDIO Y NO APORTO."),
        ) + _STOPS_ATR + _EARNINGS + _SECTOR
    ),
}


# ============================================================================
# HOJA METODO -- las reglas MEDIDAS para experimentar sin engancharse
# ============================================================================
# Cada una salio de una medicion de este proyecto, no de un libro. Son las que
# evitan que un cambio de parametro parezca una mejora cuando no lo es.

METODO: Tuple[Tuple[str, str], ...] = (
    ("Medir el PISO DE RUIDO antes de comparar",
     "En la sectorial el desempate entre candidatos empatados decide el 42,8% de "
     "los sector-ruedas y lo resuelve el ALFABETO. Correr LA MISMA regla con ese "
     "desempate sorteado, con 20 semillas, da de +16,34% a +26,30%: una banda de "
     "~10 puntos. La mejor de 166 reglas de entrada le gana a la actual por 0,90 "
     "puntos, o sea MENOS que el ruido. Antes de festejar una mejora, "
     "aleatorizar la parte arbitraria y ver cuanto se mueve sola."),
    ("Ajustar por EXPOSICION o la grilla premia a la mas laxa",
     "Correlacion entre exposicion media y ventaja: +0,61 y +0,67. Una regla que "
     "esta comprada mas tiempo gana mas en un mercado alcista sin elegir mejor. "
     "El retorno se divide por la exposicion media; sin eso gana siempre la que "
     "compra todo."),
    ("Pre-registrar la REGLA DE LECTURA antes de correr",
     "Periodo, variantes, metrica y que resultado cuenta como PASA, por escrito "
     "y antes. Despues de ver los numeros, cualquier umbral se puede justificar."),
    ("Seleccion y confirmacion SEPARADAS",
     "Con cientos de combinaciones algo siempre destaca. La grilla de 589 reglas "
     "de salida se leyo en 2021-24 y se confirmo en 2025-26: la mejor del primer "
     "tramo dio CERO en el segundo, y el orden entre reglas correlaciona -0,16 "
     "entre periodos."),
    ("Enumerar reglas monotonas, no barrer pesos",
     "Sobre 4 condiciones binarias hay 166 reglas monotonas: cubren todo lo que "
     "cualquier juego de ponderadores puede expresar, y 18 mas. Una grilla de "
     "3.750 juegos de pesos colapsa a 589 decisiones distintas: se repite la "
     "misma prueba cientos de veces y se paga en comparaciones multiples."),
    ("Un TOTAL no es una metrica de significancia",
     "Es media x n. COMBO_v1 movia 60 puntos al cambiar el fill, y era 452 "
     "operaciones x +0,100 pp con IC95 que incluye el cero. Todo total va con su "
     "intervalo, y la unidad de independencia es el DIA, no la operacion."),
    ("Contar ACTIVACIONES antes de creerle a una regla",
     "La salida por agotamiento de SMC_v2 no disparo ni una vez en 94 ruedas. "
     "Una regla que nunca dispara no es una regla, y su presencia en la ficha "
     "hace creer que la estrategia hace algo que no hace."),
    ("Los parametros SI tienen sesgo de sobreajuste",
     "Las reglas fijas no, pero un parametro elegido mirando resultados es "
     "sobreajuste hecho a mano. Corolario del proyecto: si dos valores empatan "
     "en el backtest (N=5 y N=3), NO se elige el mejor -- van los dos a FT y "
     "decide la operacion real."),
    ("Toda version nueva va EN PARALELO, la vieja queda como control",
     "No se reemplaza una logica en su lugar: se crea una instancia nueva con su "
     "id y las dos corren los mismos dias. Sin control de la MISMA familia no "
     "hay conclusion: comparar contra otra logica mide el mercado del tramo."),
    ("Registrar el cambio en ft_cambios ANTES de desplegarlo",
     "Sin registro, el cambio no se puede medir despues. La fecha efectiva se "
     "razona como fecha_datos (la rueda con la que va a decidir la proxima "
     "corrida), no como la del commit."),
    ("Con la potencia de FT, casi nada es distinguible de cero",
     "Con 27 o 450 operaciones el IC95 contra el control casi siempre incluye el "
     "cero. Eso se escribe; las decisiones se apoyan en el backtest de anios y "
     "en los insumos, no en el resultado de FT."),
    ("Los COSTOS estan SIN MEDIR, a proposito",
     "Spread, comisiones y slippage no se midieron. Con +0,160% de retorno medio "
     "por operacion, un costo del orden del spread se come el resultado entero. "
     "Cualquier parametro que aumente la frecuencia de operacion empeora eso, y "
     "la grilla no lo ve."),
)


# ============================================================================
# LO MAS IMPORTANTE PARA EXPERIMENTAR: DONDE SE CAMBIA UN PARAMETRO
# ============================================================================
# VERIFICADO el 23/11/2026 leyendo los 12 bots: `ft_estrategias.parametros` es
# un REGISTRO, no configuracion. Ningun bot lee de ahi sus umbrales:
# `ft_utils.cargar_estrategia` trae la columna, pero cada bot decide con
# CONSTANTES de modulo (ft_bot_tech_sectorial.py: SCORE_ENTRADA = 4.0 ...).
# Cambiar un valor en la tabla NO cambia ninguna decision; cambiarlo en el bot
# SI, y entonces la tabla queda desalineada. El exportador compara las dos
# cosas y avisa.

ADVERTENCIA_PARAMETROS = (
    "LOS PARAMETROS DE LA TABLA SON UN REGISTRO, NO CONFIGURACION. Ningun bot "
    "lee ft_estrategias.parametros para decidir: cada uno tiene sus umbrales "
    "como CONSTANTES de modulo. Cambiar un numero en la tabla no cambia nada; "
    "para experimentar hay que editar el .py del bot. Y ahi hay dos "
    "obligaciones del proyecto: (1) registrar el cambio en `ft_cambios` ANTES "
    "de desplegarlo, con la fecha de la rueda de DATOS con la que va a decidir "
    "-- sin registro el cambio no se puede medir despues; (2) no reemplazar la "
    "version vieja, crear una INSTANCIA NUEVA en paralelo y dejar la vieja como "
    "control."
)

# Que bot implementa cada instancia (clave: ft_estrategias.nombre).
# FT_SMC_v3_N5 y N3 comparten archivo: la ventana entra por --ventana.
BOTS = {
    "FT_TECH_v1": "scripts/forward_testing/ft_bot_tecnico.py",
    "FT_TECH_SECTOR_v1": "scripts/forward_testing/ft_bot_tech_sectorial.py",
    "FT_TECH_SECTOR_v2": "scripts/forward_testing/ft_bot_tech_sectorial_v2.py",
    "FT_TECH_SECTOR_OPTIONS_v1":
        "scripts/forward_testing/ft_bot_tech_sectorial_options_v1.py",
    "FT_TECH_SECTOR_OPTIONS_v2":
        "scripts/forward_testing/ft_bot_tech_sectorial_options_v2.py",
    "FT_TECH_SECTOR_OIEXIT_v1":
        "scripts/forward_testing/ft_bot_tech_sectorial_oiexit_v1.py",
    "FT_COMBO_v1": "scripts/forward_testing/ft_bot_combo_v1.py (comentado en "
                   "ft_run_diario.bat: dado de baja)",
    "FT_ML_SCANNER_v1": "scripts/forward_testing/ft_bot_ml_scanner.py",
    "FT_ML_SCANNER_v2": "scripts/forward_testing/ft_bot_ml_scanner_v2.py",
    "FT_SMC_v1": "scripts/forward_testing/ft_bot_smc.py",
    "FT_SMC_v2": "scripts/forward_testing/ft_bot_smc_v2.py (comentado en "
                 "ft_run_diario.bat: dado de baja)",
    "FT_SMC_v3_N5": "scripts/forward_testing/ft_bot_smc_v3.py --ventana 5",
    "FT_SMC_v3_N3": "scripts/forward_testing/ft_bot_smc_v3.py --ventana 3",
}

# Donde vive la LOGICA compartida: editar aca toca a VARIAS estrategias a la vez.
CEREBRO = {
    "tecnico": "src/strategies/scoring.py (calcular_score_tecnico): el score y "
               "la banda de RSI los comparten TODAS las que puntuan tecnico, "
               "FT y Alpaca.",
    "tecnico_sectorial": "src/strategies/scoring.py (score) + "
                         "src/strategies/sectorial.py (entradas, cierres y "
                         "prioridades). Lo comparten las 5 sectoriales.",
    "tecnico_sectorial_v2": "src/strategies/scoring.py + "
                            "src/strategies/sectorial.py.",
    "tecnico_sectorial_options_v1": "src/strategies/scoring.py + "
                                    "src/strategies/sectorial.py "
                                    "(usar_opciones=True) + "
                                    "src/utils/opciones_plazo.py (PCR y muros).",
    "tecnico_sectorial_options_v2": "Idem OPTIONS_v1: el mismo cerebro con "
                                    "PCR_VOL.",
    "tecnico_sectorial_oiexit_v1": "src/strategies/scoring.py (entrada) + la "
                                   "salida propia en el bot + "
                                   "src/utils/opciones_plazo.py (muro).",
    "ml_scanner": "src/strategies/ml_scanner.py (ConfigML, entradas y cierres). "
                  "El score y el nivel los produce antes "
                  "src/pipeline/alert_classifier.py.",
    "smc_estructura": "scripts/forward_testing/ft_scoring.py "
                      "(calcular_score_estructura y los limites del SL).",
    "smc_estructura_v2": "scripts/forward_testing/ft_scoring.py + los filtros "
                         "propios del bot.",
    "smc_estructura_confirmada": "scripts/forward_testing/ft_scoring_estructura.py "
                                 "(loader) + el score importado de ft_scoring.py: "
                                 "la regla NO se reimplementa.",
    "combo_tech_candle": "src/strategies/scoring.py + el desempate de velas en "
                         "el bot.",
}

# Claves que una instancia NO declara, con el motivo. Evita que la planilla
# diga "no definido" cuando el umbral SI existe, solo que vive en otro lado.
NO_PARAMETRIZADO = {
    "rsi_min": "NO es por estrategia: es la constante RSI_MIN = 45,0 de "
               "src/strategies/scoring.py, compartida por TODAS las que puntuan "
               "tecnico. Cambiarla las toca a todas a la vez, y entonces ninguna "
               "queda como control.",
    "rsi_max": "Idem: constante RSI_MAX = 68,0 de src/strategies/scoring.py.",
    "score_maximo": "Tope del score, no un umbral: SCORE_MAXIMO_TECH = 5,5 en "
                    "src/strategies/scoring.py (SMC usa SCORE_MAXIMO_SMC = 3 en "
                    "ft_scoring.py). Es un dato derivado de los puntos, no se "
                    "configura.",
    "score_salida": "Esta instancia NO sale por score degradado a secas: tiene "
                    "su propia regla de salida (retencion en TECH_SECTOR_v2, "
                    "muro y quorum en OIEXIT_v1). El default del cerebro "
                    "compartido es 3,5.",
    "score_entrada": "Esta instancia lo declara con otro nombre: "
                     "tech_score_entrada_min, porque el tecnico es el GATE 1 de "
                     "dos. Mismo numero, mismo rol.",
    "max_deploy_pct": "No declarado aca. Las sectoriales no tienen techo global: "
                      "las limita el presupuesto por sector. En las ML el default "
                      "de ConfigML es 0,80 y solo la v2 lo deja escrito.",
    "senal": "No declarado: la v1 lee las columnas BASE de alertas_scanner "
             "(alert_nivel / alert_score). Lo declara la v2 para dejar explicito "
             "que ella lee las columnas _v2.",
    "modelo": "No declarado: la v1 usa el modelo global V3 del scanner. Lo "
              "declara la v2 porque el artefacto es LO UNICO que las diferencia.",
}
