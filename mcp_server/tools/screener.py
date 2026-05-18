"""
mcp_server/tools/screener.py
Screener multi-criterio sobre los 199 tickers del universo.

Tool unica: screen_tickers(...)

Responde preguntas del tipo:
  "Que tickers tienen RSI > 60 y estructura alcista?"
  "Dame los 5 con mayor alert_score que tengan CHoCH alcista y vol_spike."
  "Tickers del sector Technology con MACD alcista y sobre la SMA200."

Diseño (opcion B):
  - SQL hardcodeado en queries.py con 17 parametros opcionales.
  - Cada filtro usa el patron ($N::tipo IS NULL OR condicion) para que
    NULL = sin filtro. El LLM solo pasa valores tipados, nunca SQL libre.
  - Todos los LEFT JOINs anclan en el ultimo dato disponible por ticker.
  - Columnas booleanas de detalle (choch_*, bos_*, patron_*) se usan
    solo para filtrar en SQL; en Python se sintetizan en señal_smc y
    patron_activo para un resultado limpio.

Validaciones en Python (antes de tocar la DB):
  - Valores de string contra sets cerrados de valores validos.
  - limit clampeado a [1, MAX_LIMIT].
"""

from datetime import date
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import SQL_SCREEN_TICKERS

SCREENER_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

MAX_LIMIT = 100

# Valores validos para parametros de string
_VALID_MACD_DIR   = {"alcista", "bajista"}
_VALID_SEÑAL_SMC  = {"choch_bull", "choch_bear", "bos_bull", "bos_bear"}
_VALID_PATRON     = {
    "engulfing_bull", "engulfing_bear", "hammer",
    "shooting_star", "marubozu", "doji",
}
_VALID_ALERT_NIVEL = {
    "COMPRA_FUERTE", "COMPRA", "NEUTRO", "VENTA", "VENTA_FUERTE",
}
_VALID_ORDENAR_POR = {"alert_score", "rsi14", "adx", "vol_relativo"}


# ── Helpers de conversion y calculo ──────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Convierte asyncpg.Record (o dict) a dict JSON-serializable."""
    result = {}
    for key, val in row.items():
        if isinstance(val, Decimal):
            result[key] = float(val)
        elif isinstance(val, date):
            result[key] = val.isoformat()
        else:
            result[key] = val
    return result


_SEÑAL_SMC_MAP = [
    ("choch_bull_5", "CHoCH alcista"),
    ("choch_bear_5", "CHoCH bajista"),
    ("bos_bull_5",   "BOS alcista"),
    ("bos_bear_5",   "BOS bajista"),
]

_PATRON_MAP = [
    ("patron_engulfing_bull", "engulfing_bull"),
    ("patron_engulfing_bear", "engulfing_bear"),
    ("patron_hammer",         "hammer"),
    ("patron_shooting_star",  "shooting_star"),
    ("patron_marubozu",       "marubozu"),
    ("patron_doji",           "doji"),
]

_DETALLE_COLS = [
    "choch_bull_5", "choch_bear_5", "bos_bull_5", "bos_bear_5",
    "patron_engulfing_bull", "patron_engulfing_bear", "patron_hammer",
    "patron_shooting_star", "patron_marubozu", "patron_doji",
    "sma200",  # usado solo para filtro sobre_sma200, no aporta al resultado
]


def _resultado_row(row) -> dict:
    """
    Convierte una fila del screener a dict limpio.

    Agrega señal_smc y patron_activo como campos sintetizados.
    Elimina columnas booleanas de detalle ya sintetizadas.
    Renombra modelo_asignado -> modelo_ml.
    """
    r = _row_to_dict(row)

    # Sintetizar señal SMC
    r["señal_smc"] = next(
        (label for col, label in _SEÑAL_SMC_MAP if r.get(col)), None
    )

    # Sintetizar patron activo
    r["patron_activo"] = next(
        (label for col, label in _PATRON_MAP if r.get(col)), None
    )

    # Renombrar modelo_asignado
    r["modelo_ml"] = r.pop("modelo_asignado", None)

    # Eliminar columnas de detalle ya sintetizadas
    for col in _DETALLE_COLS:
        r.pop(col, None)

    return r


# ── Tool publica ──────────────────────────────────────────────────────────────

async def screen_tickers(
    # ── Perfil ────────────────────────────────────────────────
    solo_ml_activos: bool = False,
    sector: str = None,
    # ── Tecnicos ──────────────────────────────────────────────
    rsi_min: float = None,
    rsi_max: float = None,
    adx_min: float = None,
    vol_relativo_min: float = None,
    sobre_sma200: bool = None,
    macd_direccion: str = None,
    # ── Estructura SMC ────────────────────────────────────────
    estructura_5: int = None,
    señal_smc: str = None,
    # ── Price Action ──────────────────────────────────────────
    es_alcista: bool = None,
    vol_spike: bool = None,
    patron: str = None,
    # ── ML ────────────────────────────────────────────────────
    alert_nivel: str = None,
    alert_score_min: float = None,
    # ── Resultado ─────────────────────────────────────────────
    ordenar_por: str = "alert_score",
    limit: int = 20,
) -> dict:
    """
    Screener multi-criterio: filtra los 199 tickers segun condiciones tecnicas,
    de estructura, price action y ML. Ancla en el ultimo dato disponible.

    Cada parametro es opcional (None = sin filtro). Se pueden combinar
    libremente. Sin parametros devuelve los primeros N tickers ordenados
    por alert_score.

    Filtros disponibles:
      solo_ml_activos   : True = solo los 22 tickers con modelo RF activo
      sector            : ej. "Technology" (case-insensitive)
      rsi_min / rsi_max : rango de RSI14, ej. rsi_min=55 rsi_max=70
      adx_min           : fuerza minima de tendencia, ej. 25
      vol_relativo_min  : volumen minimo relativo a media, ej. 1.5
      sobre_sma200      : True = precio sobre SMA200, False = por debajo
      macd_direccion    : "alcista" (hist>0) | "bajista" (hist<=0)
      estructura_5      : 1=alcista | 0=neutral | -1=bajista (ventana 5)
      señal_smc         : "choch_bull" | "choch_bear" | "bos_bull" | "bos_bear"
      es_alcista        : True = vela del dia alcista
      vol_spike         : True = spike de volumen detectado
      patron            : "engulfing_bull"|"engulfing_bear"|"hammer"|
                          "shooting_star"|"marubozu"|"doji"
      alert_nivel       : "COMPRA_FUERTE"|"COMPRA"|"NEUTRO"|"VENTA"|"VENTA_FUERTE"
      alert_score_min   : score ML minimo (0-100), ej. 65

    Ordenamiento (ordenar_por): "alert_score" (default)|"rsi14"|"adx"|"vol_relativo"
    limit: max resultados (default 20, max 100)

    Returns:
        {fecha_analisis, filtros_aplicados, total_encontrados, tickers: [...]}
        En caso de parametros invalidos o error: {"error": "..."}

    Ejemplo de uso:
        screen_tickers(rsi_min=60, estructura_5=1, alert_nivel="COMPRA_FUERTE", limit=10)
        -> tickers alcistas con RSI alto y señal ML de compra fuerte
    """
    # ── Validaciones ──────────────────────────────────────────────────────────
    errores = []

    if macd_direccion is not None and macd_direccion not in _VALID_MACD_DIR:
        errores.append(
            f"macd_direccion invalido: '{macd_direccion}'. "
            f"Valores validos: {sorted(_VALID_MACD_DIR)}"
        )
    if señal_smc is not None and señal_smc not in _VALID_SEÑAL_SMC:
        errores.append(
            f"señal_smc invalido: '{señal_smc}'. "
            f"Valores validos: {sorted(_VALID_SEÑAL_SMC)}"
        )
    if patron is not None and patron not in _VALID_PATRON:
        errores.append(
            f"patron invalido: '{patron}'. "
            f"Valores validos: {sorted(_VALID_PATRON)}"
        )
    if alert_nivel is not None and alert_nivel not in _VALID_ALERT_NIVEL:
        errores.append(
            f"alert_nivel invalido: '{alert_nivel}'. "
            f"Valores validos: {sorted(_VALID_ALERT_NIVEL)}"
        )
    if ordenar_por not in _VALID_ORDENAR_POR:
        errores.append(
            f"ordenar_por invalido: '{ordenar_por}'. "
            f"Valores validos: {sorted(_VALID_ORDENAR_POR)}"
        )
    if estructura_5 is not None and estructura_5 not in {-1, 0, 1}:
        errores.append(
            f"estructura_5 invalido: {estructura_5}. Valores validos: -1, 0, 1"
        )

    if errores:
        return {"error": " | ".join(errores)}

    # Clampear limit
    limit = max(1, min(int(limit), MAX_LIMIT))

    # ── Filtros aplicados (solo los no-None) para la respuesta ────────────────
    filtros_aplicados = {k: v for k, v in {
        "solo_ml_activos":  solo_ml_activos or None,
        "sector":           sector,
        "rsi_min":          rsi_min,
        "rsi_max":          rsi_max,
        "adx_min":          adx_min,
        "vol_relativo_min": vol_relativo_min,
        "sobre_sma200":     sobre_sma200,
        "macd_direccion":   macd_direccion,
        "estructura_5":     estructura_5,
        "señal_smc":        señal_smc,
        "es_alcista":       es_alcista,
        "vol_spike":        vol_spike,
        "patron":           patron,
        "alert_nivel":      alert_nivel,
        "alert_score_min":  alert_score_min,
    }.items() if v is not None}

    # ── Consulta ──────────────────────────────────────────────────────────────
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                SQL_SCREEN_TICKERS,
                # $1 .. $17
                bool(solo_ml_activos),   # $1  siempre bool, nunca NULL
                sector,                  # $2
                rsi_min,                 # $3
                rsi_max,                 # $4
                adx_min,                 # $5
                vol_relativo_min,        # $6
                sobre_sma200,            # $7
                macd_direccion,          # $8
                estructura_5,            # $9
                señal_smc,               # $10
                es_alcista,              # $11
                vol_spike,               # $12
                patron,                  # $13
                alert_nivel,             # $14
                alert_score_min,         # $15
                ordenar_por,             # $16
                limit,                   # $17
            )

        tickers = [_resultado_row(r) for r in rows]

        return {
            "fecha_analisis":    date.today().isoformat(),
            "filtros_aplicados": filtros_aplicados,
            "total_encontrados": len(tickers),
            "ordenado_por":      ordenar_por,
            "limit":             limit,
            "tickers":           tickers,
        }

    except Exception as exc:
        return {"error": f"Error en el screener: {exc}"}
