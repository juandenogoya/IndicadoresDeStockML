"""
yahooquery_options.py
Wrapper de yahooquery para descarga de option chains.

Ventaja sobre yfinance: yahooquery devuelve el chain ENTERO (todas las
expirations, calls + puts) en UNA sola llamada HTTP. yfinance requiere
1 call para listar `Ticker.options` + N calls (una por expiration) para
`option_chain(exp)`. En el snapshot de 199 tickers eso pasa de ~6.000
requests a ~199 (30x menos carga sobre la IP).

Shape devuelto por Ticker(t).option_chain:
    DataFrame con MultiIndex (symbol, expiration, optionType)
      - symbol      : str        (siempre el ticker)
      - expiration  : Timestamp  (datetime, no string como yfinance)
      - optionType  : str        'calls' / 'puts' (plural; yfinance usa singular)
    Columnas relevantes (mismo nombre que yfinance):
      strike, volume, openInterest, impliedVolatility, bid, ask
    Columnas extra (no usamos): contractSymbol, lastPrice, change,
      percentChange, currency, contractSize, lastTradeDate, inTheMoney

Este modulo SOLO descarga y normaliza shape. El filtrado por DTE / OI /
volumen es responsabilidad del caller (33_opciones_snapshot.py), para
que la logica de negocio quede en un solo lugar.
"""

from typing import Optional

import pandas as pd


class OptionsRateLimit(Exception):
    """Yahoo respondio con throttle al chain endpoint."""
    pass


_RATELIMIT_HINTS = (
    "too many requests",
    "rate limited",
    "ratelimit",
    "429",
)


def _is_ratelimit_msg(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(h in t for h in _RATELIMIT_HINTS)


def descargar_chain(ticker: str) -> Optional[pd.DataFrame]:
    """
    Descarga el option chain completo del ticker.

    Returns:
        DataFrame multi-indexed (symbol, expiration, optionType) o None si
        no hay datos (ticker sin opciones, error transitorio sin throttle).

    Raises:
        OptionsRateLimit: si Yahoo respondio con rate limit. Quien llama
                          decide si dormir + reintentar o abortar.
    """
    from yahooquery import Ticker

    try:
        yqt = Ticker(ticker)
        df = yqt.option_chain
    except Exception as e:
        if _is_ratelimit_msg(str(e)):
            raise OptionsRateLimit(str(e))
        return None

    # yahooquery puede devolver dict {symbol: "error message"} cuando falla
    # toda la consulta (rate limit, ticker invalido, sin opciones, etc.).
    if isinstance(df, dict):
        joined = " | ".join(str(v) for v in df.values())
        if _is_ratelimit_msg(joined):
            raise OptionsRateLimit(joined[:200])
        return None

    if df is None or df.empty:
        return None

    return df
