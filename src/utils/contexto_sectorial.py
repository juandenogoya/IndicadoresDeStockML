"""
src/utils/contexto_sectorial.py
Que tickers quedan SIN features sectoriales, y con que etiqueta se los marca.

PURO: stdlib solamente. Sin DB, sin config, sin Streamlit. Lo importan el
productor de los features (src/indicators/sector_features.py), el scanner, el
notificador de Telegram y el MCP server -- por eso no puede arrastrar
dependencias pesadas (regla 1 del MCP: solo funciones puras de src/utils/ y
src/indicators/).

QUE PROBLEMA RESUELVE
    Los features sectoriales son z-scores del ticker CONTRA SUS PARES. Con un
    sector de 1 o 3 miembros ese z-score no mide nada: con n=1 el ticker es su
    propio promedio y da 0 por construccion; con n=3 un solo par moviendose
    define la escala. Por eso Real Estate (3) y Utilities (1) quedaron afuera
    del calculo (Tarea 20).

    La consecuencia estaba invisible: esos 4 tickers igual reciben score de ML,
    pero 11 de las 53 features del modelo llegan en NaN.

LA DISTINCION QUE IMPORTA
    Son dos cosas distintas y solo una estaba resuelta:

      1. QUE MODELO se usa   -> el global, y esta bien: con 1 y 3 tickers no
                                hay con que entrenar un modelo sectorial.
      2. QUE DATOS recibe    -> incompletos. El modelo global fue ENTRENADO con
                                esas 11 features presentes; al inferir sobre
                                EQIX llegan vacias.

    Elegir el modelo global resuelve (1) y no toca (2). De ahi la marca: la
    probabilidad se sigue publicando, pero dice sobre que base se calculo.

POR QUE LA LISTA VIVE ACA Y NO EN EL SQL
    El productor (sector_features.py) excluye estos sectores en su WHERE. Si la
    marca los enumerara por su cuenta, el dia que un sector crezca y entre al
    calculo las dos listas se desincronizarian y la marca seguiria apareciendo
    sobre tickers que ya tienen contexto -- o peor, dejaria de aparecer sobre
    los que no. Una sola definicion, dos lectores.

COMO SE DESACTIVA
    Sola: cuando un sector deje de estar en SECTORES_SIN_FEATURES (porque se
    incorporaron tickers y ya hay pares con que comparar), el productor empieza
    a calcularle features y la marca deja de emitirse. No hay backfill: la
    marca se DERIVA del sector en cada lectura, no se almacena.
"""

from typing import Optional, Sequence

# Sectores excluidos del calculo de features sectoriales por tamano de muestra.
# Al 2/9/2026: Real Estate = AMT, EQIX, PLD (3) | Utilities = VST (1).
SECTORES_SIN_FEATURES: Sequence[str] = ("Real Estate", "Utilities")

# Umbral que motivo la exclusion, para no dejarlo como numero magico en la
# discusion: con n <= 3 el z-score contra pares es ruido.
MIN_PARES = 3

ETIQUETA = "Sin contexto sectorial"
ETIQUETA_CORTA = "s/ctx"

# Explicacion de una linea, para tooltips y salidas del MCP.
MOTIVO = ("el sector tiene muy pocos tickers para comparar: 11 de las 53 "
          "features del modelo llegan vacias")

# Las 11 features sectoriales que quedan en NaN. Es la lista canonica: la
# importa src/pipeline/feature_calculator.py para leerlas de features_sector.
FEATURES_SECTORIALES: Sequence[str] = (
    "z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
    "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector",
    "pct_long_sector", "rank_retorno_sector",
    "rsi_sector_avg", "adx_sector_avg", "retorno_1d_sector_avg",
)


def sin_contexto_sectorial(sector: Optional[str]) -> bool:
    """True si un ticker de ese sector NO tiene features sectoriales.

    `sector` None o vacio tambien da True, y no es un caso de borde defensivo:
    el WHERE del productor es `sector NOT IN (...)`, y en SQL NULL NOT IN (...)
    evalua a NULL, asi que esa fila queda fuera del calculo igual que las
    excluidas a proposito. Devolver False aca haria que un ticker sin sector
    apareciera SIN marca justamente cuando tampoco tiene contexto.
    """
    if not sector:
        return True
    return sector in SECTORES_SIN_FEATURES


def marca(sector: Optional[str], corta: bool = False) -> str:
    """Etiqueta lista para concatenar, o '' si el ticker tiene contexto.

    Devolver cadena vacia (y no None) deja que el llamador la sume sin
    condicionales: f"{nivel}{marca(sector)}" no rompe en el caso normal, que
    es 196 de 200 tickers.
    """
    if not sin_contexto_sectorial(sector):
        return ""
    return f" [{ETIQUETA_CORTA}]" if corta else f" [{ETIQUETA}]"
