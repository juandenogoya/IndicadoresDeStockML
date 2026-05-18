"""
mcp_server/tools/stocks.py
Tools de datos historicos de acciones: precios, indicadores tecnicos,
price action y market structure.

Todas las tools son SOLO LECTURA (readOnlyHint=True).

Convencion de retorno (envelope):
    {
        "ticker":           str,
        "desde_servido":    str,   # YYYYMMDD
        "hasta_servido":    str,   # YYYYMMDD
        "dias_solicitados": int,
        "dias_servidos":    int,
        "warning":          str | None,
        "data":             list[dict]
    }

En caso de error (fecha invalida, DB no disponible):
    {"error": "<descripcion>"}

Limite de periodo: MAX_DAYS = 360 dias. Si el rango solicitado lo supera,
se recorta desde `desde` hasta `desde + 360 dias` y se informa en `warning`.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal

from mcp_server.db.pool import get_pool
from mcp_server.db.queries import (
    SQL_MARKET_STRUCTURE,
    SQL_PRICE_ACTION,
    SQL_PRICE_HISTORY,
    SQL_TECHNICAL_INDICATORS,
)

MAX_DAYS = 360

STOCKS_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> date:
    """
    Convierte string YYYYMMDD a date.
    Lanza ValueError si el formato es incorrecto o la fecha no existe.
    """
    return datetime.strptime(s, "%Y%m%d").date()


def _row_to_dict(row) -> dict:
    """
    Convierte asyncpg.Record a dict JSON-serializable.
    - Decimal  -> float
    - date     -> str ISO (YYYY-MM-DD)
    - El resto pasa directo (int, str, None).

    Usa row.items() en lugar de dict(row) para compatibilidad con mocks de test
    y con la interfaz de mapping de asyncpg.Record.
    """
    result = {}
    for key, val in row.items():
        if isinstance(val, Decimal):
            result[key] = float(val)
        elif isinstance(val, date):
            result[key] = val.isoformat()
        else:
            result[key] = val
    return result


def _clamp_range(desde: date, hasta: date) -> tuple[date, int, int, str | None]:
    """
    Aplica el limite de MAX_DAYS al rango [desde, hasta].

    Retorna (hasta_servido, dias_solicitados, dias_servidos, warning).
    Si el rango es valido, warning=None y hasta_servido == hasta.
    """
    dias_solicitados = (hasta - desde).days
    if dias_solicitados > MAX_DAYS:
        hasta_servido = desde + timedelta(days=MAX_DAYS)
        warning = (
            f"Periodo solicitado ({dias_solicitados} dias) excede el maximo "
            f"de {MAX_DAYS} dias. Se sirven los primeros {MAX_DAYS} dias: "
            f"desde {desde.strftime('%Y%m%d')} hasta "
            f"{hasta_servido.strftime('%Y%m%d')}."
        )
        return hasta_servido, dias_solicitados, MAX_DAYS, warning
    return hasta, dias_solicitados, dias_solicitados, None


def _build_envelope(
    ticker: str,
    desde: date,
    hasta_servido: date,
    dias_solicitados: int,
    dias_servidos: int,
    warning: str | None,
    rows: list[dict],
) -> dict:
    return {
        "ticker": ticker,
        "desde_servido": desde.strftime("%Y%m%d"),
        "hasta_servido": hasta_servido.strftime("%Y%m%d"),
        "dias_solicitados": dias_solicitados,
        "dias_servidos": dias_servidos,
        "warning": warning,
        "data": rows,
    }


async def _fetch_stock_data(sql: str, ticker: str, desde: str, hasta: str) -> dict:
    """
    Logica compartida para las 4 tools:
    1. Parsea y valida fechas.
    2. Aplica clampeo de MAX_DAYS.
    3. Ejecuta la query.
    4. Retorna envelope.
    """
    # --- validacion de fechas ---
    try:
        desde_date = _parse_date(desde)
    except ValueError:
        return {"error": f"Formato de fecha invalido para 'desde': '{desde}'. Usar YYYYMMDD."}

    try:
        hasta_date = _parse_date(hasta)
    except ValueError:
        return {"error": f"Formato de fecha invalido para 'hasta': '{hasta}'. Usar YYYYMMDD."}

    if hasta_date < desde_date:
        return {
            "error": (
                f"'hasta' ({hasta}) no puede ser anterior a 'desde' ({desde})."
            )
        }

    # --- clampeo de periodo ---
    hasta_servido, dias_solicitados, dias_servidos, warning = _clamp_range(
        desde_date, hasta_date
    )

    # --- consulta DB ---
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, ticker, desde_date, hasta_servido)
        data = [_row_to_dict(r) for r in rows]
    except Exception as exc:
        return {"error": f"Error al consultar datos para '{ticker}': {exc}"}

    return _build_envelope(
        ticker, desde_date, hasta_servido,
        dias_solicitados, dias_servidos, warning, data,
    )


# ── Tools publicas ────────────────────────────────────────────────────────────

async def get_price_history(ticker: str, desde: str, hasta: str) -> dict:
    """
    Devuelve el historico de precios OHLCV de un ticker en el rango indicado.

    Columnas retornadas: fecha, open, high, low, close, volume, adj_close.
    Periodo maximo: 360 dias. Si el rango excede ese limite, se recorta
    y se indica en el campo 'warning' del envelope de respuesta.

    Args:
        ticker: Simbolo del activo (ej: "AAPL", "MSFT"). Case-sensitive.
        desde:  Fecha inicio en formato YYYYMMDD (ej: "20250101"). Requerido.
        hasta:  Fecha fin   en formato YYYYMMDD (ej: "20250331"). Requerido.

    Returns:
        Envelope con campos: ticker, desde_servido, hasta_servido,
        dias_solicitados, dias_servidos, warning, data (list de OHLCV).
        En caso de error: {"error": "<descripcion>"}.
    """
    return await _fetch_stock_data(SQL_PRICE_HISTORY, ticker, desde, hasta)


async def get_technical_indicators(ticker: str, desde: str, hasta: str) -> dict:
    """
    Devuelve los indicadores tecnicos calculados para un ticker en el rango indicado.

    Columnas retornadas: sma21/50/200, dist_sma*, rsi14, momentum,
    macd/signal/hist, atr14, bb_upper/middle/lower, obv, vol_relativo, adx.
    Periodo maximo: 360 dias.

    Args:
        ticker: Simbolo del activo (ej: "AAPL"). Case-sensitive.
        desde:  Fecha inicio YYYYMMDD. Requerido.
        hasta:  Fecha fin   YYYYMMDD. Requerido.

    Returns:
        Envelope con data (list de indicadores por fecha) o {"error": "..."}.
    """
    return await _fetch_stock_data(SQL_TECHNICAL_INDICATORS, ticker, desde, hasta)


async def get_price_action(ticker: str, desde: str, hasta: str) -> dict:
    """
    Devuelve las features de price action y patrones de vela para un ticker.

    Incluye: cuerpo/sombras de vela, patrones (doji, hammer, engulfing, etc.),
    inside/outside bar, tendencia de velas, volumen relativo y flujo A/D.
    Periodo maximo: 360 dias.

    Args:
        ticker: Simbolo del activo. Case-sensitive.
        desde:  Fecha inicio YYYYMMDD. Requerido.
        hasta:  Fecha fin   YYYYMMDD. Requerido.

    Returns:
        Envelope con data (list de features PA por fecha) o {"error": "..."}.
    """
    return await _fetch_stock_data(SQL_PRICE_ACTION, ticker, desde, hasta)


async def get_market_structure(ticker: str, desde: str, hasta: str) -> dict:
    """
    Devuelve las features de market structure (SMC) para un ticker.

    Incluye para ventanas de 5 y 10 velas: swing highs/lows, estructura
    (HH/HL/LH/LL), distancias a SH/SL, impulsos, BOS y CHoCH bull/bear.
    Periodo maximo: 360 dias.

    Args:
        ticker: Simbolo del activo. Case-sensitive.
        desde:  Fecha inicio YYYYMMDD. Requerido.
        hasta:  Fecha fin   YYYYMMDD. Requerido.

    Returns:
        Envelope con data (list de features MS por fecha) o {"error": "..."}.
    """
    return await _fetch_stock_data(SQL_MARKET_STRUCTURE, ticker, desde, hasta)
