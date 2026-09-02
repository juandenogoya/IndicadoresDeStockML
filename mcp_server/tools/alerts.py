"""
mcp_server/tools/alerts.py
Historial de alertas del scanner ML.

Tool unica: get_ml_alert_history(...)

Responde preguntas del tipo:
  "Que alertas COMPRA_FUERTE hubo esta semana?"
  "Cuales son las ultimas alertas de AAPL?"
  "Que tickers tuvieron alerta con score > 70 verificada y cuanto subieron?"

Columnas clave de alertas_scanner:
  scan_fecha      : cuando corrio el scanner (TIMESTAMP, filtrado como DATE)
  precio_fecha    : dia de cierre sobre el que se calcula la alerta
  alert_score     : score ML 0-100
  alert_nivel     : COMPRA_FUERTE | COMPRA | NEUTRO | VENTA | VENTA_FUERTE
  alert_detalle   : resumen legible generado por el scanner
  ml_prob_ganancia: probabilidad ML de ganancia (0.0-1.0)
  condiciones_pa  : suma de EV1+EV2+EV3+EV4 (0-4), condiciones PA cumplidas
  estructura_10   : contexto market structure (1=alcista,0=neutral,-1=bajista)
  verificado      : TRUE si ya se cargaron retornos reales post-alerta
  retorno_Nd_real : retorno % real a 1d/5d/20d post-alerta (null si no verificado)

Convencion de columnas (critica):
  - Fecha: scan_fecha (NO "fecha_alerta", ese nombre no existe).
  - Nivel: alert_nivel (NO "senal"; valores: COMPRA_FUERTE, COMPRA, NEUTRO,
    VENTA, VENTA_FUERTE).
  - Score ML: alert_score (0-100). Distincion vs score_ponderado (rule-based).
"""

from datetime import date
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import SQL_ML_ALERT_HISTORY
from src.utils.contexto_sectorial import MOTIVO, sin_contexto_sectorial

ALERTS_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

MAX_LIMIT = 100

_VALID_ALERT_NIVEL = {
    "COMPRA_FUERTE", "COMPRA", "NEUTRO", "VENTA", "VENTA_FUERTE",
}

_ESTRUCTURA_LABEL = {1: "alcista", 0: "neutral", -1: "bajista"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Convierte asyncpg.Record a dict JSON-serializable."""
    result = {}
    for key, val in row.items():
        if isinstance(val, Decimal):
            result[key] = float(val)
        elif isinstance(val, date):
            result[key] = val.isoformat()
        else:
            result[key] = val
    return result


def _clean_alert_row(row) -> dict:
    """
    Convierte fila de alertas_scanner a dict limpio para el LLM.

    - estructura_10 (int) -> texto legible
    - verificado -> bool explicito
    - retornos NULL se mantienen como None (el LLM los omite o aclara)
    """
    r = _row_to_dict(row)

    est = r.get("estructura_10")
    r["estructura_10"] = (
        _ESTRUCTURA_LABEL.get(int(est), "sin datos")
        if est is not None else "sin datos"
    )

    r["verificado"] = bool(r.get("verificado"))

    # Campo DERIVADO del sector, no almacenado: aplica retroactivamente a toda
    # la historia de alertas_scanner. Va como texto y no como booleano crudo
    # porque el LLM vuelca los 0/1 sin interpretarlos; asi puede repetir el
    # motivo tal cual al usuario.
    r["contexto_sectorial"] = (
        f"AUSENTE -- {MOTIVO}" if sin_contexto_sectorial(r.get("sector"))
        else "completo"
    )

    return r


# ── Tool publica ──────────────────────────────────────────────────────────────

async def get_ml_alert_history(
    ticker: str = None,
    desde: str = None,
    hasta: str = None,
    alert_nivel: str = None,
    alert_score_min: float = None,
    solo_verificados: bool = False,
    limit: int = 20,
) -> dict:
    """
    Historial de alertas del scanner ML sobre los 200 tickers.

    Cada fila representa una alerta generada por el scanner en una fecha dada,
    con el score ML, nivel, condiciones tecnicas y (si ya paso tiempo suficiente)
    los retornos reales post-alerta para validar la señal.

    Parametros (todos opcionales):
      ticker           : ej. "AAPL" -- filtra por ticker especifico
      desde            : fecha inicio "YYYY-MM-DD" (filtra por scan_fecha)
      hasta            : fecha fin "YYYY-MM-DD" (filtra por scan_fecha)
      alert_nivel      : "COMPRA_FUERTE"|"COMPRA"|"NEUTRO"|"VENTA"|"VENTA_FUERTE"
      alert_score_min  : score minimo, ej. 70.0
      solo_verificados : True = solo alertas con retornos reales ya cargados
                         (util para evaluar performance historica del modelo)
      limit            : max resultados (default 20, max 100)

    Campos clave en la respuesta por alerta:
      fecha_scan       : dia en que corrio el scanner
      precio_fecha     : dia de cierre sobre el que se calculo la alerta
      alert_score      : 0-100 (mayor = señal mas fuerte)
      alert_nivel      : nivel categorico de la alerta
      alert_detalle    : descripcion legible generada por el scanner
      ml_prob_ganancia : probabilidad de ganancia segun modelo RF (0.0-1.0)
      condiciones_pa   : condiciones price action cumplidas (0-4)
      estructura_10    : contexto market structure al momento de la alerta
      contexto_sectorial: "completo" | "AUSENTE -- <motivo>". AUSENTE significa
                         que el ticker pertenece a un sector demasiado chico
                         para comparar contra pares (Real Estate, Utilities),
                         asi que 11 de las 53 features del modelo llegaron
                         vacias. El modelo usado igual es el correcto (global);
                         lo que esta incompleto es la entrada. Si aparece
                         AUSENTE, decirlo al reportar la alerta.
      verificado       : True si ya se cargaron retornos reales
      retorno_1d_real  : retorno % real al dia siguiente (null si no verificado)
      retorno_5d_real  : retorno % real a 5 dias (null si no verificado)
      retorno_20d_real : retorno % real a 20 dias (null si no verificado)

    Ejemplo de uso:
      get_ml_alert_history(alert_nivel="COMPRA_FUERTE", alert_score_min=75, limit=10)
      -> ultimas 10 alertas de compra fuerte con score alto

      get_ml_alert_history(ticker="AAPL", solo_verificados=True, limit=20)
      -> historial verificado de AAPL para evaluar performance del modelo
    """
    # ── Validaciones ──────────────────────────────────────────────────────────
    errores = []

    if alert_nivel is not None and alert_nivel not in _VALID_ALERT_NIVEL:
        errores.append(
            f"alert_nivel invalido: '{alert_nivel}'. "
            f"Valores validos: {sorted(_VALID_ALERT_NIVEL)}"
        )

    desde_date = None
    hasta_date = None
    if desde is not None:
        try:
            desde_date = date.fromisoformat(desde)
        except ValueError:
            errores.append(f"desde invalido: '{desde}'. Formato esperado: YYYY-MM-DD")
    if hasta is not None:
        try:
            hasta_date = date.fromisoformat(hasta)
        except ValueError:
            errores.append(f"hasta invalido: '{hasta}'. Formato esperado: YYYY-MM-DD")

    if errores:
        return {"error": " | ".join(errores)}

    limit = max(1, min(int(limit), MAX_LIMIT))

    # ── Filtros aplicados (solo los activos) ──────────────────────────────────
    filtros_aplicados = {k: v for k, v in {
        "ticker":           ticker,
        "desde":            desde,
        "hasta":            hasta,
        "alert_nivel":      alert_nivel,
        "alert_score_min":  alert_score_min,
        "solo_verificados": solo_verificados or None,
    }.items() if v is not None}

    # ── Consulta ──────────────────────────────────────────────────────────────
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                SQL_ML_ALERT_HISTORY,
                ticker,                  # $1
                desde_date,              # $2
                hasta_date,              # $3
                alert_nivel,             # $4
                alert_score_min,         # $5
                bool(solo_verificados),  # $6
                limit,                   # $7
            )

        alertas = [_clean_alert_row(r) for r in rows]

        return {
            "fecha_consulta":    date.today().isoformat(),
            "filtros_aplicados": filtros_aplicados,
            "total_encontrados": len(alertas),
            "limit":             limit,
            "alertas":           alertas,
        }

    except Exception as exc:
        return {"error": f"Error consultando historial de alertas: {exc}"}
