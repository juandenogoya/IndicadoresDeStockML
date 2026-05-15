"""
mcp_server/tools/options.py
Tool de analisis de opciones: sentimiento de mercado e inferencia de
posicionamiento institucional.

Tool unica: get_options_analysis(ticker, dias_historia=20)

Retorna un envelope con 3 secciones:
  tendencia_diaria     -- PCR e IV diarios (de opciones_resumen_diario + zscore)
  pcr_por_vencimiento  -- balance call/put por cada vencimiento vivo, dia a dia
  acumulacion_oi       -- top 10 calls + top 10 puts con mayor crecimiento de OI

La seccion 3 responde: "a donde fue el dinero nuevo en el periodo?"
La seccion 2 responde: "como evoluciono el sentimiento por vencimiento?"
La seccion 1 responde: "cual es la tendencia macro del mercado de opciones?"

El LLM sintetiza la conclusion a partir de los tres insumos.

Reglas de diseño:
  - Solo contratos vivos (vencimiento > fecha_snapshot mas reciente).
  - dias_historia: cuantos snapshots diarios tomar (default 20, max 60).
  - Sin parametros de fecha explicitos: ancla en el snapshot mas reciente
    disponible y mira N dias hacia atras.
  - IV skew = iv_put_avg - iv_call_avg. Positivo = mercado paga mas por
    proteccion bajista. Negativo = optimismo.
  - moneyness_pct = (strike/precio_subyacente - 1) * 100.
    Positivo = strike sobre precio (OTM call / ITM put).
    Negativo = strike bajo precio (ITM call / OTM put).
"""

from datetime import date
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import (
    SQL_OPTIONS_ACUMULACION_OI,
    SQL_OPTIONS_DATE_RANGE,
    SQL_OPTIONS_PCR_POR_VENCIMIENTO,
    SQL_OPTIONS_TENDENCIA,
)

MAX_DIAS_HISTORIA = 60
DEFAULT_DIAS_HISTORIA = 20

OPTIONS_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


# ── Helpers de conversion ─────────────────────────────────────────────────────

def _cvt(val):
    """Convierte tipos asyncpg a JSON-serializable."""
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, date):
        return val.isoformat()
    return val


def _moneyness_label(pct: float | None, tipo: str) -> str:
    """
    Clasifica el strike relativo al precio del subyacente.

    Para calls:  strike > precio = OTM  |  strike < precio = ITM
    Para puts:   strike > precio = ITM  |  strike < precio = OTM
    ATM: dentro de +-2% del precio.
    """
    if pct is None:
        return "N/A"
    if abs(pct) <= 2.0:
        return "ATM"
    if tipo == "call":
        return "OTM" if pct > 0 else "ITM"
    else:  # put
        return "ITM" if pct > 0 else "OTM"


def _tend_row(row) -> dict:
    """Convierte una fila de tendencia_diaria y agrega iv_skew calculado."""
    r = {k: _cvt(v) for k, v in row.items()}
    iv_put  = r.get("iv_put_avg")
    iv_call = r.get("iv_call_avg")
    if iv_put is not None and iv_call is not None:
        r["iv_skew"] = round(iv_put - iv_call, 4)
    else:
        r["iv_skew"] = None
    return r


def _acum_row(row) -> dict:
    """Convierte una fila de acumulacion_oi y agrega moneyness_label."""
    r = {k: _cvt(v) for k, v in row.items()}
    r["moneyness_label"] = _moneyness_label(
        r.get("moneyness_pct"), r.get("tipo", "")
    )
    return r


def _build_pcr_por_vencimiento(rows) -> list[dict]:
    """
    Agrupa las filas (vencimiento x fecha_snapshot) en lista anidada:
    [{vencimiento, dias_a_venc, serie: [{fecha, call_vol, put_vol, pcr_vol,
                                         call_oi, put_oi, pcr_oi}]}]
    Orden: vencimiento ASC (el mas proximo primero).
    PCR calculado en Python para manejar division por cero limpiamente.
    """
    venc_map: dict[str, dict] = {}

    for row in rows:
        venc_date = row["vencimiento"]
        venc_key  = venc_date.isoformat() if isinstance(venc_date, date) else str(venc_date)

        fecha_snap = row["fecha_snapshot"]
        fecha_str  = fecha_snap.isoformat() if isinstance(fecha_snap, date) else str(fecha_snap)

        if venc_key not in venc_map:
            venc_map[venc_key] = {
                "vencimiento": venc_key,
                "dias_a_venc": int(row["dias_a_venc"]) if row["dias_a_venc"] is not None else None,
                "serie": [],
            }

        call_vol = int(row["call_vol"] or 0)
        put_vol  = int(row["put_vol"]  or 0)
        call_oi  = int(row["call_oi"]  or 0)
        put_oi   = int(row["put_oi"]   or 0)

        venc_map[venc_key]["serie"].append({
            "fecha":    fecha_str,
            "call_vol": call_vol,
            "put_vol":  put_vol,
            "pcr_vol":  round(put_vol / call_vol, 4) if call_vol > 0 else None,
            "call_oi":  call_oi,
            "put_oi":   put_oi,
            "pcr_oi":   round(put_oi  / call_oi,  4) if call_oi  > 0 else None,
        })

    return sorted(venc_map.values(), key=lambda x: x["vencimiento"])


# ── Tool publica ──────────────────────────────────────────────────────────────

async def get_options_analysis(
    ticker: str,
    dias_historia: int = DEFAULT_DIAS_HISTORIA,
) -> dict:
    """
    Analiza el posicionamiento del mercado de opciones para un ticker.

    Responde: donde esta pensando ir el precio segun los inversores?

    Ancla en el snapshot mas reciente disponible en la DB y mira
    dias_historia snapshots hacia atras (solo dias con datos cargados).
    Solo incluye contratos vivos (vencimiento posterior al ultimo snapshot).

    Retorna tres secciones:

    tendencia_diaria: PCR vol/OI, IV call/put, IV skew y z-scores por dia.
      - pcr_vol > 1 indica mas actividad en puts (presion bajista).
      - iv_skew > 0 indica que el mercado paga mas por puts (miedo).
      - vol_total_zscore alto indica actividad inusual de opciones.

    pcr_por_vencimiento: balance call/put para cada vencimiento vivo,
      dia a dia. Muestra si el sentimiento para un vencimiento especifico
      esta cambiando de direccion. Orden: vencimiento mas cercano primero.

    acumulacion_oi: top 10 calls y top 10 puts donde mas crecio el OI
      durante el periodo. Identifica donde fue el dinero nuevo:
      - calls OTM acumulando = apuesta alcista (espera subida de precio)
      - puts ATM acumulando  = cobertura bajista (espera caida o volatilidad)

    Args:
        ticker:        Simbolo del activo (ej: "AAPL", "SPY"). Case-sensitive.
        dias_historia: Snapshots diarios a incluir (default 20, max 60).
                       20 dias ~ 1 mes de historia de opciones.

    Returns:
        {ticker, fecha_snapshot, cutoff_fecha, precio_subyacente,
         dias_historia, tendencia_diaria, pcr_por_vencimiento,
         acumulacion_oi: {top_calls_por_delta_oi, top_puts_por_delta_oi}}
        En caso de error o sin datos: {"error": "<descripcion>"}.
    """
    dias_historia = max(1, min(int(dias_historia), MAX_DIAS_HISTORIA))

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:

            # 1. Rango de fechas disponibles para el ticker
            date_row = await conn.fetchrow(
                SQL_OPTIONS_DATE_RANGE, ticker, dias_historia
            )

            if date_row is None or date_row["max_fecha"] is None:
                return {
                    "error": (
                        f"No hay datos de opciones para '{ticker}'. "
                        "Verificar con list_tickers() que el ticker existe."
                    )
                }

            max_fecha    = date_row["max_fecha"]
            cutoff_fecha = date_row["cutoff_fecha"]

            # 2. Tendencia diaria macro (resumen + zscore JOIN)
            tend_rows = await conn.fetch(
                SQL_OPTIONS_TENDENCIA, ticker, cutoff_fecha
            )

            # 3. PCR por vencimiento, serie temporal
            pcr_rows = await conn.fetch(
                SQL_OPTIONS_PCR_POR_VENCIMIENTO, ticker, cutoff_fecha, max_fecha
            )

            # 4. Acumulacion de OI — top 10 calls + top 10 puts
            acum_rows = await conn.fetch(
                SQL_OPTIONS_ACUMULACION_OI, ticker, cutoff_fecha
            )

        # Precio subyacente del snapshot mas reciente (primer row = mas reciente)
        precio_sub = None
        if tend_rows:
            raw = tend_rows[0]["precio_sub"]
            precio_sub = float(raw) if raw is not None else None

        top_calls = [_acum_row(r) for r in acum_rows if r["tipo"] == "call"]
        top_puts  = [_acum_row(r) for r in acum_rows if r["tipo"] == "put"]

        return {
            "ticker":              ticker,
            "fecha_snapshot":      max_fecha.isoformat(),
            "cutoff_fecha":        cutoff_fecha.isoformat(),
            "precio_subyacente":   precio_sub,
            "dias_historia":       dias_historia,
            "tendencia_diaria":    [_tend_row(r) for r in tend_rows],
            "pcr_por_vencimiento": _build_pcr_por_vencimiento(pcr_rows),
            "acumulacion_oi": {
                "top_calls_por_delta_oi": top_calls,
                "top_puts_por_delta_oi":  top_puts,
            },
        }

    except Exception as exc:
        return {"error": f"Error al analizar opciones para '{ticker}': {exc}"}
