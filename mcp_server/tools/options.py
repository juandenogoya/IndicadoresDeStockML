"""
mcp_server/tools/options.py
Tool de analisis de opciones: sentimiento de mercado e inferencia de
posicionamiento institucional.

Tool unica: get_options_analysis(ticker, dias_historia=20)

Retorna un envelope con 3 secciones:
  tendencia_diaria     -- {actual, serie}: ultimo dia completo + serie diaria
                          recortada (PCR vol/OI, IV skew, z-scores por dia).
                          Conserva la trayectoria; campos redundantes fuera.
  pcr_por_vencimiento  -- resumen computado por vencimiento vivo: PCR OI
                          inicio/actual, delta de OI, sesgo y tendencia
                          (NO devuelve la serie diaria cruda -- ahorro de tokens)
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


def _iv_skew(iv_put, iv_call):
    """IV skew = iv_put - iv_call. None si falta alguno."""
    if iv_put is not None and iv_call is not None:
        return round(iv_put - iv_call, 4)
    return None


# Banda neutra del IV skew. Default ajustable: skews dentro de +-0.05 se
# consideran neutros. Fuera de esa banda el signo define el sesgo.
_IV_SKEW_NEUTRAL = 0.05


def _sesgo_iv_skew(skew) -> str:
    """
    Clasifica el IV skew (iv_put - iv_call) en sesgo legible.

      skew >  +0.05 -> bajista  (puts mas caras = cobertura / miedo)
      skew <  -0.05 -> alcista  (calls mas caras = demanda / optimismo)
      entre medio   -> neutro

    El signo es contraintuitivo: skew POSITIVO = sentimiento defensivo.
    El umbral _IV_SKEW_NEUTRAL es un default a ajustar con la experiencia.
    """
    if skew is None:
        return "sin datos"
    if skew > _IV_SKEW_NEUTRAL:
        return "bajista"
    if skew < -_IV_SKEW_NEUTRAL:
        return "alcista"
    return "neutro"


def _tend_serie_row(row) -> dict:
    """
    Fila de la serie diaria con campos recortados.

    Conserva la trayectoria (un punto por dia) pero elimina campos
    redundantes: call_vol/put_vol/call_oi/put_oi (ya resumidos en pcr_*),
    iv_call_avg/iv_put_avg (resumidos en iv_skew), precio_sub (top-level),
    vol_relativo/percentil_vol (redundantes con vol_total_zscore).

    Cada PCR lleva su sesgo computado (alcista/neutro/bajista) al lado: el
    LLM interpreta mal el PCR crudo (PCR < 1 = mas calls, no mas puts).
    """
    pcr_vol = _cvt(row["pcr_vol"])
    pcr_oi  = _cvt(row["pcr_oi"])
    skew    = _iv_skew(_cvt(row["iv_put_avg"]), _cvt(row["iv_call_avg"]))
    return {
        "fecha":            _cvt(row["fecha"]),
        "pcr_vol":          pcr_vol,
        "sesgo_pcr_vol":    _sesgo_pcr(pcr_vol),
        "pcr_oi":           pcr_oi,
        "sesgo_pcr_oi":     _sesgo_pcr(pcr_oi),
        "iv_skew":          skew,
        "sesgo_iv_skew":    _sesgo_iv_skew(skew),
        "vol_total_zscore": _cvt(row["vol_total_zscore"]),
        "pcr_vol_zscore":   _cvt(row["pcr_vol_zscore"]),
        "iv_zscore":        _cvt(row["iv_zscore"]),
    }


def _tend_actual(row) -> dict:
    """
    Ultimo dia completo: incluye los niveles de IV (call/put) y n_contratos,
    que son utiles para el estado actual pero no para cada dia historico.

    Cada PCR lleva su sesgo computado al lado (ver _tend_serie_row).
    """
    pcr_vol = _cvt(row["pcr_vol"])
    pcr_oi  = _cvt(row["pcr_oi"])
    skew    = _iv_skew(_cvt(row["iv_put_avg"]), _cvt(row["iv_call_avg"]))
    return {
        "fecha":            _cvt(row["fecha"]),
        "pcr_vol":          pcr_vol,
        "sesgo_pcr_vol":    _sesgo_pcr(pcr_vol),
        "pcr_oi":           pcr_oi,
        "sesgo_pcr_oi":     _sesgo_pcr(pcr_oi),
        "iv_call_avg":      _cvt(row["iv_call_avg"]),
        "iv_put_avg":       _cvt(row["iv_put_avg"]),
        "iv_skew":          skew,
        "sesgo_iv_skew":    _sesgo_iv_skew(skew),
        "n_contratos":      _cvt(row["n_contratos"]),
        "vol_total_zscore": _cvt(row["vol_total_zscore"]),
        "pcr_vol_zscore":   _cvt(row["pcr_vol_zscore"]),
        "iv_zscore":        _cvt(row["iv_zscore"]),
    }


def _acum_row(row) -> dict:
    """
    Convierte una fila de acumulacion_oi a dict limpio.

    Omite campos redundantes para ahorrar tokens sin perder informacion:
      - precio_subyacente: constante, ya esta en el top-level del envelope.
      - oi_inicio: derivable (oi_fin - delta_oi); se mantienen oi_fin y delta_oi.
    Agrega moneyness_label sintetizado.
    """
    return {
        "vencimiento":     _cvt(row["vencimiento"]),
        "tipo":            row["tipo"],
        "strike":          _cvt(row["strike"]),
        "oi_fin":          _cvt(row["oi_fin"]),
        "delta_oi":        _cvt(row["delta_oi"]),
        "iv_actual":       _cvt(row["iv_actual"]),
        "moneyness_pct":   _cvt(row["moneyness_pct"]),
        "moneyness_label": _moneyness_label(
            _cvt(row["moneyness_pct"]), row["tipo"]
        ),
    }


def _pcr(put: int, call: int) -> float | None:
    """PCR = put/call. None si call es 0 (division indefinida)."""
    return round(put / call, 4) if call and call > 0 else None


def _sesgo_pcr(pcr: float | None) -> str:
    """
    Clasifica un PCR segun los umbrales documentados del proyecto:
      < 0.7  -> alcista   (dominan calls)
      0.7-1.0 -> neutro
      > 1.0  -> bajista   (dominan puts)
    """
    if pcr is None:
        return "sin datos"
    if pcr < 0.7:
        return "alcista"
    if pcr <= 1.0:
        return "neutro"
    return "bajista"


def _resumen_pcr_por_vencimiento(rows) -> list[dict]:
    """
    Resume cada vencimiento vivo en metricas computadas, SIN devolver la
    serie diaria cruda (ahorro de tokens ~85% en esta seccion).

    Por cada vencimiento:
      - pcr_oi_inicio / pcr_oi_actual : PCR de open interest al inicio y al
        final de la ventana consultada (inicio = snapshot mas antiguo)
      - pcr_vol_promedio              : PCR de volumen promediado en la ventana
      - call_oi / put_oi inicio, actual, delta : acumulacion de OI
      - sesgo_actual                  : alcista|neutro|bajista (PCR OI actual)
      - tendencia                     : "estable en X" o "X -> Y" si el sesgo
                                        cambio entre inicio y actual

    Las filas llegan ordenadas (vencimiento ASC, fecha_snapshot DESC), asi que
    dentro de cada vencimiento el primer dia es el mas reciente.
    Orden de salida: vencimiento ASC (el mas proximo primero).
    """
    venc_map: dict[str, dict] = {}

    for row in rows:
        venc_date = row["vencimiento"]
        venc_key  = venc_date.isoformat() if isinstance(venc_date, date) else str(venc_date)

        fecha_snap = row["fecha_snapshot"]
        fecha_str  = fecha_snap.isoformat() if isinstance(fecha_snap, date) else str(fecha_snap)

        if venc_key not in venc_map:
            venc_map[venc_key] = {
                "dias_a_venc": int(row["dias_a_venc"]) if row["dias_a_venc"] is not None else None,
                "dias": [],
            }

        venc_map[venc_key]["dias"].append({
            "fecha":    fecha_str,
            "call_vol": int(row["call_vol"] or 0),
            "put_vol":  int(row["put_vol"]  or 0),
            "call_oi":  int(row["call_oi"]  or 0),
            "put_oi":   int(row["put_oi"]   or 0),
        })

    resumen = []
    for venc_key, info in venc_map.items():
        dias   = info["dias"]
        actual = dias[0]    # mas reciente (SQL ordena fecha_snapshot DESC)
        inicio = dias[-1]   # mas antiguo de la ventana

        pcr_oi_actual = _pcr(actual["put_oi"], actual["call_oi"])
        pcr_oi_inicio = _pcr(inicio["put_oi"], inicio["call_oi"])

        pcr_vols = [
            p for p in (_pcr(d["put_vol"], d["call_vol"]) for d in dias)
            if p is not None
        ]
        pcr_vol_prom = round(sum(pcr_vols) / len(pcr_vols), 4) if pcr_vols else None

        sesgo_actual = _sesgo_pcr(pcr_oi_actual)
        sesgo_inicio = _sesgo_pcr(pcr_oi_inicio)
        if sesgo_actual == sesgo_inicio:
            tendencia = f"estable en {sesgo_actual}"
        else:
            tendencia = f"{sesgo_inicio} -> {sesgo_actual}"

        resumen.append({
            "vencimiento":      venc_key,
            "dias_a_venc":      info["dias_a_venc"],
            "dias_con_data":    len(dias),
            "pcr_oi_inicio":    pcr_oi_inicio,
            "pcr_oi_actual":    pcr_oi_actual,
            "pcr_vol_promedio": pcr_vol_prom,
            "call_oi_inicio":   inicio["call_oi"],
            "call_oi_actual":   actual["call_oi"],
            "delta_call_oi":    actual["call_oi"] - inicio["call_oi"],
            "put_oi_inicio":    inicio["put_oi"],
            "put_oi_actual":    actual["put_oi"],
            "delta_put_oi":     actual["put_oi"] - inicio["put_oi"],
            "sesgo_actual":     sesgo_actual,
            "tendencia":        tendencia,
        })

    return sorted(resumen, key=lambda x: x["vencimiento"])


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

    tendencia_diaria: {actual, serie}.
      - actual: ultimo dia completo (PCR vol/OI con su sesgo, IV call/put,
        IV skew, n_contratos, z-scores).
      - serie: un punto por dia con campos recortados (pcr_vol, pcr_oi,
        cada uno con sesgo, iv_skew, z-scores) -- preserva la trayectoria.
      Cada PCR trae su sesgo computado (sesgo_pcr_vol, sesgo_pcr_oi):
      alcista si PCR < 0.7, neutro 0.7-1.0, bajista > 1.0. IMPORTANTE:
      PCR < 1 = MAS CALLS que puts (no al reves). iv_skew trae sesgo_iv_skew:
      bajista si skew > 0 (puts mas caras = miedo), alcista si skew < 0
      (calls mas caras). vol_total_zscore alto = actividad inusual.

    pcr_por_vencimiento: resumen computado por cada vencimiento vivo. En vez
      de la serie diaria cruda devuelve PCR OI inicio/actual, delta de OI
      (call y put), PCR vol promedio, sesgo actual y tendencia (si el sesgo
      cambio en la ventana). Orden: vencimiento mas cercano primero.

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

        # tendencia_diaria: ultimo dia completo (actual) + serie recortada
        if tend_rows:
            tendencia_diaria = {
                "actual": _tend_actual(tend_rows[0]),
                "serie":  [_tend_serie_row(r) for r in tend_rows],
            }
        else:
            tendencia_diaria = {"actual": None, "serie": []}

        return {
            "ticker":              ticker,
            "fecha_snapshot":      max_fecha.isoformat(),
            "cutoff_fecha":        cutoff_fecha.isoformat(),
            "precio_subyacente":   precio_sub,
            "dias_historia":       dias_historia,
            "tendencia_diaria":    tendencia_diaria,
            "pcr_por_vencimiento": _resumen_pcr_por_vencimiento(pcr_rows),
            "acumulacion_oi": {
                "top_calls_por_delta_oi": top_calls,
                "top_puts_por_delta_oi":  top_puts,
            },
        }

    except Exception as exc:
        return {"error": f"Error al analizar opciones para '{ticker}': {exc}"}
