"""
mcp_server/tools/overview.py
Tool de vision general de un ticker: snapshot completo del estado actual.

Tool unica: get_ticker_overview(ticker)

Retorna hasta 7 secciones:
  perfil           -- sector, industry, modelo ML, activo
  precio           -- OHLCV ultimo dia + variaciones 1d/5d/20d
  tecnicos         -- indicadores del ultimo dia + señales categoricas derivadas
  price_action     -- patron de vela y features PA del ultimo dia
  market_structure -- estructura SMC + señal de cambio reciente (BOS/CHoCH)
  volumen          -- z-score de volumen del ultimo dia
  opciones         -- snapshot de opciones del ultimo dia (si disponible)

Cada seccion es independiente: error en una no rompe las demas.
Señales categoricas (tendencia_sma, rsi_estado, etc.) se calculan en Python:
  - Son logica de presentacion, no features ML.
  - Permiten que el LLM sintetice sin necesidad de umbrales en SQL.

Reglas de diseño:
  - Un solo parametro: ticker. Sin fechas (ancla en el ultimo dato disponible).
  - Perfil es la unica seccion requerida: si el ticker no existe en activos,
    retorna {"error": ...} en lugar del envelope completo.
  - Si una seccion falla (sin datos o error de DB), retorna
    {"disponible": False, "razon": ...} o {"error": ...} en esa clave.
"""

from datetime import date
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import (
    SQL_OVERVIEW_MARKET_STRUCTURE,
    SQL_OVERVIEW_OPCIONES_RESUMEN,
    SQL_OVERVIEW_OPCIONES_TOP_OI,
    SQL_OVERVIEW_PERFIL,
    SQL_OVERVIEW_PRECIO,
    SQL_OVERVIEW_PRICE_ACTION,
    SQL_OVERVIEW_TECNICOS,
    SQL_OVERVIEW_VOLUMEN_ZSCORE,
)

OVERVIEW_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


# ── Conversion ────────────────────────────────────────────────────────────────

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


# ── Señales categoricas ───────────────────────────────────────────────────────

def _tendencia_sma(close, sma21, sma50, sma200) -> str:
    """
    Clasifica la tendencia segun la posicion del precio respecto a las SMAs.

    alcista fuerte : precio > SMA21 > SMA50 > SMA200  (alineacion perfecta)
    alcista        : precio sobre SMA50 y SMA200
    bajista fuerte : precio < SMA21 < SMA50 < SMA200
    bajista        : precio bajo SMA50 y SMA200
    lateral        : configuracion mixta
    """
    if any(v is None for v in [close, sma21, sma50, sma200]):
        return "sin datos"
    if close > sma21 > sma50 > sma200:
        return "alcista fuerte"
    if close > sma50 and close > sma200:
        return "alcista"
    if close < sma21 < sma50 < sma200:
        return "bajista fuerte"
    if close < sma50 and close < sma200:
        return "bajista"
    return "lateral"


def _rsi_estado(rsi) -> str:
    """
    Clasifica el estado del RSI14.

    sobrecomprado: >= 70  |  sobrevendido: <= 30
    alcista: 55-69        |  bajista: 31-45
    neutral: 46-54
    """
    if rsi is None:
        return "sin datos"
    if rsi >= 70:
        return "sobrecomprado"
    if rsi <= 30:
        return "sobrevendido"
    if rsi >= 55:
        return "alcista"
    if rsi <= 45:
        return "bajista"
    return "neutral"


def _macd_direccion(macd_hist) -> str:
    """Direccion del MACD segun el histograma (positivo=alcista)."""
    if macd_hist is None:
        return "sin datos"
    return "alcista" if macd_hist > 0 else "bajista"


def _bb_posicion(close, bb_upper, bb_lower) -> str:
    """
    Posicion del precio dentro de las Bandas de Bollinger.

    Normaliza la posicion: 0.0 = banda inferior, 1.0 = banda superior.
    alta: >= 0.8  |  media-alta: 0.6-0.8  |  media: 0.4-0.6
    media-baja: 0.2-0.4  |  baja: < 0.2
    """
    if any(v is None for v in [close, bb_upper, bb_lower]):
        return "sin datos"
    rango = bb_upper - bb_lower
    if rango <= 0:
        return "sin datos"
    pos = (close - bb_lower) / rango
    if pos >= 0.8:
        return "alta"
    if pos >= 0.6:
        return "media-alta"
    if pos >= 0.4:
        return "media"
    if pos >= 0.2:
        return "media-baja"
    return "baja"


def _adx_fuerza(adx) -> str:
    """
    Fuerza de la tendencia segun ADX.

    muy fuerte: >= 40  |  fuerte: 25-39  |  moderada: 20-24  |  debil: < 20
    """
    if adx is None:
        return "sin datos"
    if adx >= 40:
        return "muy fuerte"
    if adx >= 25:
        return "fuerte"
    if adx >= 20:
        return "moderada"
    return "debil"


_PATRON_PRIORITY = [
    ("patron_engulfing_bull", "engulfing_bull"),
    ("patron_engulfing_bear", "engulfing_bear"),
    ("patron_hammer",         "hammer"),
    ("patron_shooting_star",  "shooting_star"),
    ("patron_marubozu",       "marubozu"),
    ("patron_doji",           "doji"),
]


def _patron_activo(pa_dict: dict) -> str | None:
    """
    Devuelve el patron de vela mas relevante activo, en orden de prioridad.
    Retorna None si no hay ningun patron activo.
    """
    for col, label in _PATRON_PRIORITY:
        if pa_dict.get(col):
            return label
    return None


_MS_SIGNAL_PRIORITY = [
    ("choch_bull_5",  "CHoCH alcista (ventana 5)"),
    ("choch_bear_5",  "CHoCH bajista (ventana 5)"),
    ("bos_bull_5",    "BOS alcista (ventana 5)"),
    ("bos_bear_5",    "BOS bajista (ventana 5)"),
    ("choch_bull_10", "CHoCH alcista (ventana 10)"),
    ("choch_bear_10", "CHoCH bajista (ventana 10)"),
    ("bos_bull_10",   "BOS alcista (ventana 10)"),
    ("bos_bear_10",   "BOS bajista (ventana 10)"),
]


def _senal_reciente_ms(ms_dict: dict | None) -> str | None:
    """
    Retorna la señal de market structure mas relevante del dia.
    Prioridad: CHoCH > BOS, ventana 5 > ventana 10.
    Retorna None si no hay señal activa.
    """
    if ms_dict is None:
        return None
    for col, label in _MS_SIGNAL_PRIORITY:
        if ms_dict.get(col):
            return label
    return None


def _variaciones(rows: list[dict]) -> tuple:
    """
    Calcula variaciones 1d, 5d y 20d a partir de una lista de filas de
    precios ordenadas DESC por fecha (mas reciente primero).

    Usa adj_close si close no esta disponible.
    Retorna (v1d, v5d, v20d) donde cada valor puede ser float o None.
    """
    if not rows:
        return None, None, None

    close_hoy = rows[0].get("close") or rows[0].get("adj_close")
    if close_hoy is None or float(close_hoy) == 0:
        return None, None, None

    def pct(idx: int):
        if len(rows) <= idx:
            return None
        prev = rows[idx].get("close") or rows[idx].get("adj_close")
        if prev and float(prev) != 0:
            return round((float(close_hoy) / float(prev) - 1) * 100, 2)
        return None

    return pct(1), pct(5), pct(20)


def _tendencia_velas_label(val) -> str:
    """Convierte tendencia_velas (int) a texto descriptivo."""
    if val is None:
        return "sin datos"
    v = int(val)
    if v >= 3:
        return f"alcista fuerte ({v} velas)"
    if v > 0:
        return f"alcista ({v} velas)"
    if v <= -3:
        return f"bajista fuerte ({abs(v)} velas)"
    if v < 0:
        return f"bajista ({abs(v)} velas)"
    return "neutral"


def _estructura_label(val) -> str:
    """Convierte estructura_5 / estructura_10 (int) a texto."""
    if val is None:
        return "sin datos"
    return {1: "alcista", 0: "neutral", -1: "bajista"}.get(int(val), str(val))


def _moneyness_label(pct: float | None, tipo: str) -> str:
    """
    Clasifica el strike relativo al precio del subyacente.
    ATM: dentro de +-2%. Call OTM = strike > precio. Put OTM = strike < precio.
    """
    if pct is None:
        return "N/A"
    if abs(pct) <= 2.0:
        return "ATM"
    if tipo == "call":
        return "OTM" if pct > 0 else "ITM"
    return "ITM" if pct > 0 else "OTM"


def _top_oi_row(row) -> dict:
    """Convierte fila de top OI y agrega moneyness_label."""
    r = _row_to_dict(row)
    r["moneyness_label"] = _moneyness_label(r.get("moneyness_pct"), r.get("tipo", ""))
    return r


# ── Tool publica ──────────────────────────────────────────────────────────────

async def get_ticker_overview(ticker: str) -> dict:
    """
    Snapshot completo del estado actual de un ticker.

    Consolida en una sola llamada: precio, indicadores tecnicos, price action,
    market structure, volumen z-score y un resumen de opciones.

    Diseñado para responder preguntas como:
      "Cual es el estado tecnico actual de AAPL?"
      "Dame el panorama completo de SPY."
      "Como esta posicionado el mercado en META?"

    Señales derivadas en tecnicos.señales:
      tendencia_sma  : alcista fuerte / alcista / lateral / bajista / bajista fuerte
      rsi_estado     : sobrecomprado / alcista / neutral / bajista / sobrevendido
      macd_direccion : alcista / bajista
      bb_posicion    : alta / media-alta / media / media-baja / baja
      adx_fuerza     : muy fuerte / fuerte / moderada / debil

    Args:
        ticker: Simbolo del activo (ej: "AAPL", "SPY"). Case-sensitive.

    Returns:
        {ticker, fecha_analisis, perfil, precio, tecnicos, price_action,
         market_structure, volumen, opciones}
        Cada seccion puede contener {"disponible": False, "razon": ...} si no
        hay datos, o {"error": ...} si fallo la query.
        Si el ticker no existe en activos: {"error": "<descripcion>"}.
    """
    result: dict = {
        "ticker":         ticker,
        "fecha_analisis": date.today().isoformat(),
    }

    try:
        pool = await get_pool()
    except Exception as exc:
        return {**result, "error": f"Error de conexion para '{ticker}': {exc}"}

    try:
        async with pool.acquire() as conn:

            # ── 1. Perfil (requerido) ──────────────────────────────────────────
            perfil_row = await conn.fetchrow(SQL_OVERVIEW_PERFIL, ticker)
            if perfil_row is None:
                return {
                    **result,
                    "error": (
                        f"Ticker '{ticker}' no encontrado en la base de datos. "
                        "Verificar con list_tickers()."
                    ),
                }
            result["perfil"] = {
                "sector":    perfil_row["sector"],
                "industry":  perfil_row["industry"],
                "modelo_ml": perfil_row["modelo_asignado"],
                "activo":    perfil_row["activo"],
            }

            # ── 2. Precio (ultimos 21 dias) ────────────────────────────────────
            precio_rows: list[dict] = []
            try:
                precio_raw = await conn.fetch(SQL_OVERVIEW_PRECIO, ticker)
                if precio_raw:
                    precio_rows = [_row_to_dict(r) for r in precio_raw]
                    v1d, v5d, v20d = _variaciones(precio_rows)
                    hoy = precio_rows[0]
                    result["precio"] = {
                        "fecha":             hoy.get("fecha"),
                        "open":              hoy.get("open"),
                        "high":              hoy.get("high"),
                        "low":               hoy.get("low"),
                        "close":             hoy.get("close") or hoy.get("adj_close"),
                        "volume":            hoy.get("volume"),
                        "variacion_1d_pct":  v1d,
                        "variacion_5d_pct":  v5d,
                        "variacion_20d_pct": v20d,
                    }
                else:
                    result["precio"] = {
                        "disponible": False,
                        "razon": "sin datos de precio",
                    }
            except Exception as exc:
                result["precio"] = {"error": str(exc)}

            # close para señales tecnicas (None si precio no disponible)
            close_actual = result.get("precio", {}).get("close")

            # ── 3. Indicadores tecnicos (ultimo dia) ───────────────────────────
            try:
                tec_raw = await conn.fetchrow(SQL_OVERVIEW_TECNICOS, ticker)
                if tec_raw:
                    t = _row_to_dict(tec_raw)
                    result["tecnicos"] = {
                        "fecha":        t.get("fecha"),
                        "sma21":        t.get("sma21"),
                        "sma50":        t.get("sma50"),
                        "sma200":       t.get("sma200"),
                        "dist_sma21":   t.get("dist_sma21"),
                        "dist_sma50":   t.get("dist_sma50"),
                        "dist_sma200":  t.get("dist_sma200"),
                        "rsi14":        t.get("rsi14"),
                        "momentum":     t.get("momentum"),
                        "macd":         t.get("macd"),
                        "macd_signal":  t.get("macd_signal"),
                        "macd_hist":    t.get("macd_hist"),
                        "atr14":        t.get("atr14"),
                        "bb_upper":     t.get("bb_upper"),
                        "bb_middle":    t.get("bb_middle"),
                        "bb_lower":     t.get("bb_lower"),
                        "obv":          t.get("obv"),
                        "vol_relativo": t.get("vol_relativo"),
                        "adx":          t.get("adx"),
                        "señales": {
                            "tendencia_sma":  _tendencia_sma(
                                close_actual,
                                t.get("sma21"), t.get("sma50"), t.get("sma200"),
                            ),
                            "rsi_estado":     _rsi_estado(t.get("rsi14")),
                            "macd_direccion": _macd_direccion(t.get("macd_hist")),
                            "bb_posicion":    _bb_posicion(
                                close_actual, t.get("bb_upper"), t.get("bb_lower"),
                            ),
                            "adx_fuerza":     _adx_fuerza(t.get("adx")),
                        },
                    }
                else:
                    result["tecnicos"] = {
                        "disponible": False,
                        "razon": "sin datos tecnicos",
                    }
            except Exception as exc:
                result["tecnicos"] = {"error": str(exc)}

            # ── 4. Price action (ultimo dia) ───────────────────────────────────
            try:
                pa_raw = await conn.fetchrow(SQL_OVERVIEW_PRICE_ACTION, ticker)
                if pa_raw:
                    p = _row_to_dict(pa_raw)
                    result["price_action"] = {
                        "fecha":             p.get("fecha"),
                        "es_alcista":        bool(p.get("es_alcista")),
                        "body_pct":          p.get("body_pct"),
                        "body_ratio":        p.get("body_ratio"),
                        "gap_apertura_pct":  p.get("gap_apertura_pct"),
                        "rango_diario_pct":  p.get("rango_diario_pct"),
                        "rango_rel_atr":     p.get("rango_rel_atr"),
                        "clv":               p.get("clv"),
                        "tendencia_velas":   _tendencia_velas_label(p.get("tendencia_velas")),
                        "inside_bar":        bool(p.get("inside_bar")),
                        "outside_bar":       bool(p.get("outside_bar")),
                        "vol_spike":         bool(p.get("vol_spike")),
                        "vol_price_confirm": bool(p.get("vol_price_confirm")),
                        "vol_price_diverge": bool(p.get("vol_price_diverge")),
                        # patron_* sintetizados en un solo campo
                        "patron_activo":     _patron_activo(p),
                    }
                else:
                    result["price_action"] = {
                        "disponible": False,
                        "razon": "sin datos de price action",
                    }
            except Exception as exc:
                result["price_action"] = {"error": str(exc)}

            # ── 5. Market structure (ultimo dia) ───────────────────────────────
            try:
                ms_raw = await conn.fetchrow(SQL_OVERVIEW_MARKET_STRUCTURE, ticker)
                if ms_raw:
                    m = _row_to_dict(ms_raw)
                    result["market_structure"] = {
                        "fecha":           m.get("fecha"),
                        # Ventana 5
                        "estructura_5":    _estructura_label(m.get("estructura_5")),
                        "dist_sh_5_pct":   m.get("dist_sh_5_pct"),
                        "dist_sl_5_pct":   m.get("dist_sl_5_pct"),
                        "impulso_5_pct":   m.get("impulso_5_pct"),
                        # Ventana 10
                        "estructura_10":   _estructura_label(m.get("estructura_10")),
                        "dist_sh_10_pct":  m.get("dist_sh_10_pct"),
                        "dist_sl_10_pct":  m.get("dist_sl_10_pct"),
                        "impulso_10_pct":  m.get("impulso_10_pct"),
                        # Señal sintetizada (BOS/CHoCH mas relevante del dia)
                        "señal_reciente":  _senal_reciente_ms(m),
                    }
                else:
                    result["market_structure"] = {
                        "disponible": False,
                        "razon": "sin datos de market structure",
                    }
            except Exception as exc:
                result["market_structure"] = {"error": str(exc)}

            # ── 6. Volumen z-score (ultimo dia) ───────────────────────────────
            try:
                vol_raw = await conn.fetchrow(SQL_OVERVIEW_VOLUMEN_ZSCORE, ticker)
                if vol_raw:
                    v = _row_to_dict(vol_raw)
                    # Eliminar columnas de metadatos innecesarias para el LLM
                    for col in ("id", "ticker", "created_at", "updated_at"):
                        v.pop(col, None)
                    result["volumen"] = v
                else:
                    result["volumen"] = {
                        "disponible": False,
                        "razon": "sin datos de z-score de volumen",
                    }
            except Exception as exc:
                result["volumen"] = {"error": str(exc)}

            # ── 7. Opciones (ultimo snapshot disponible) ───────────────────────
            try:
                opc_raw = await conn.fetchrow(SQL_OVERVIEW_OPCIONES_RESUMEN, ticker)
                if opc_raw:
                    o = _row_to_dict(opc_raw)
                    iv_put  = o.get("iv_put_avg")
                    iv_call = o.get("iv_call_avg")
                    iv_skew = (
                        round(iv_put - iv_call, 4)
                        if iv_put is not None and iv_call is not None
                        else None
                    )
                    top_oi_raw = await conn.fetch(SQL_OVERVIEW_OPCIONES_TOP_OI, ticker)
                    top_calls = [
                        _top_oi_row(r) for r in top_oi_raw if r["tipo"] == "call"
                    ][:3]
                    top_puts = [
                        _top_oi_row(r) for r in top_oi_raw if r["tipo"] == "put"
                    ][:3]
                    result["opciones"] = {
                        "disponible":   True,
                        "fecha":        o.get("fecha"),
                        "pcr_vol":      o.get("pcr_vol"),
                        "pcr_oi":       o.get("pcr_oi"),
                        "iv_call_avg":  o.get("iv_call_avg"),
                        "iv_put_avg":   o.get("iv_put_avg"),
                        "iv_skew":      iv_skew,
                        "precio_sub":   o.get("precio_sub"),
                        "n_contratos":  o.get("n_contratos"),
                        "top_calls_oi": top_calls,
                        "top_puts_oi":  top_puts,
                    }
                else:
                    result["opciones"] = {
                        "disponible": False,
                        "razon": "sin datos de opciones",
                    }
            except Exception as exc:
                result["opciones"] = {
                    "disponible": False,
                    "razon": f"error consultando opciones: {exc}",
                }

    except Exception as exc:
        result["error_critico"] = f"Error de conexion: {exc}"

    return result
