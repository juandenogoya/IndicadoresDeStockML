"""
yahooquery_loader.py
Wrapper sobre yahooquery que expone la MISMA interfaz que el download_batch()
basado en yfinance que vive dentro de scripts/recovery_incremental.py.

Motivo (20/5/2026): yfinance dejo de servir confiablemente el endpoint
/v7/finance/options desde ~18/5/2026 (bloqueo de Yahoo contra ese cliente,
verificado en 5 IPs distintas: Oracle, Azure GH Actions, residencial,
4G movil). yahooquery usa otros endpoints (/v7/finance/download y
/v7/finance/quoteSummary) sobre el mismo proveedor y sigue respondiendo.

Diseño:
- Misma firma que download_batch() del recovery_incremental.py:
      download_batch(tickers, start, end) -> {ticker: DataFrame}
- DataFrame por ticker con index = datetime e columnas:
      Open, High, Low, Close, Adj Close, Volume
  (idéntico shape al group_by='ticker' de yf.download(), para que
   df_yf_a_df_db() del recovery acepte indistintamente cualquiera).
- RateLimitDetected es la MISMA clase que define recovery_incremental.py;
  para evitar import circular reusamos por duck-typing: si detectamos
  rate limit, lanzamos la excepcion del modulo llamador via attribute.

Uso:
    from src.utils import yahooquery_loader as yq
    res = yq.download_batch(['AAPL','SPY'], start=date(2026,5,18), end=date(2026,5,19))
    # res == {'AAPL': DataFrame(2 filas), 'SPY': DataFrame(2 filas)}
"""

from datetime import date, timedelta

import pandas as pd


class RateLimitDetected(Exception):
    """Levantada cuando yahooquery devuelve indicios de rate limit."""
    pass


# Frases en errores de Yahoo que indican throttling. Misma logica que el
# _is_ratelimit_msg() del recovery_incremental.py, replicada aca para que
# este modulo no dependa del script.
_RATELIMIT_HINTS = (
    "too many requests",
    "rate limited",
    "ratelimit",
    "yfratelimit",
    "429",
)


def _is_ratelimit_msg(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(h in t for h in _RATELIMIT_HINTS)


def download_batch(tickers: list, start: date, end: date) -> dict:
    """
    Descarga OHLCV diario via yahooquery.

    Args:
        tickers: lista de symbols. Puede ser uno o varios.
        start:   fecha inicio (inclusive).
        end:     fecha fin (inclusive, igual semantica que el caller).

    Returns:
        dict {ticker: pd.DataFrame} con shape estilo yf.download(group_by='ticker'):
            index : DatetimeIndex
            cols  : Open, High, Low, Close, Adj Close, Volume

    Raises:
        RateLimitDetected: si yahooquery devuelve mensajes de throttle.
    """
    from yahooquery import Ticker

    # yahooquery: end es EXCLUSIVO en algunas versiones e inclusivo en otras.
    # Para uniformar comportamiento con yf.download (end exclusivo), sumamos
    # un dia al end recibido, igual que hace download_batch() de yfinance.
    end_exclusive = end + timedelta(days=1)

    try:
        t = Ticker(tickers, asynchronous=True)
        df = t.history(
            start=start.strftime("%Y-%m-%d"),
            end=end_exclusive.strftime("%Y-%m-%d"),
            interval="1d",
            adj_ohlc=False,
        )
    except Exception as e:
        msg = str(e)
        if _is_ratelimit_msg(msg):
            raise RateLimitDetected(msg)
        # Otros errores: devolvemos vacio (mismo contrato que el yfinance loader).
        return {}

    # yahooquery puede devolver:
    #  - DataFrame con MultiIndex (symbol, date) cuando hubo datos para >=1 ticker
    #  - DataFrame con index plano = date cuando se paso un solo ticker
    #  - dict {symbol: 'error message'} cuando TODOS los symbols fallaron
    #  - DataFrame vacio
    if isinstance(df, dict):
        joined_msgs = " | ".join(str(v) for v in df.values())
        if _is_ratelimit_msg(joined_msgs):
            raise RateLimitDetected(joined_msgs[:200])
        return {}

    if df is None or df.empty:
        return {}

    # Tope inclusivo: yahooquery puede devolver, ADEMAS de las barras hasta `end`,
    # la barra EN CURSO del dia de hoy (sobre todo en futuros 24h). El contrato de
    # download_batch es `end` INCLUSIVO (igual que yf.download con end exclusivo +1):
    # recortamos cualquier fila posterior a `end`. Sin esto, la barra parcial de hoy
    # se persistiria con la fecha de hoy y bloquearia la carga de la barra real
    # cuando ese dia cierre (regla #10: data de sesion abierta etiquetada como cierre).
    end_ts = pd.Timestamp(end)

    def _clip(shaped):
        return shaped[shaped.index <= end_ts]

    result = {}

    if isinstance(df.index, pd.MultiIndex):
        # Caso normal: (symbol, date). Iteramos por symbol.
        symbols_presentes = df.index.get_level_values(0).unique()
        for sym in symbols_presentes:
            sub = df.loc[sym]
            if sub.empty:
                continue
            shaped = _clip(_to_yf_shape(sub))
            if not shaped.empty:
                result[sym] = shaped
    else:
        # Single ticker: index = date.
        if len(tickers) == 1:
            shaped = _clip(_to_yf_shape(df))
            if not shaped.empty:
                result[tickers[0]] = shaped

    return result


def _to_yf_shape(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta el DataFrame de yahooquery al shape que produce yf.download().

    yahooquery: index=date (date object), cols open/high/low/close/volume/adjclose.
    yfinance  : index=DatetimeIndex,      cols Open/High/Low/Close/Volume/Adj Close.

    df_yf_a_df_db() del recovery_incremental.py lee 'Open', 'High', 'Low',
    'Close', 'Adj Close', 'Volume', asi que mapeamos a esos nombres exactos.
    """
    sub = sub.copy()
    # Index: yahooquery puede MEZCLAR en el mismo index (sobre todo en futuros,
    # que cotizan ~24h):
    #   - datetime.date            -> barras diarias COMPLETAS (tz-naive)
    #   - datetime.datetime tz-aware (America/New_York) -> la barra EN CURSO del
    #     dia de hoy (sesion abierta).
    # pandas 3.x ya NO coacciona ese mix: pd.to_datetime(mix) lanza
    # "ValueError: Mixed timezones detected". Normalizamos cada entrada a su
    # fecha de calendario LOCAL: las tz-aware se vuelven naive conservando su
    # wall-clock (NO convertimos a UTC, para no correr la fecha de la barra).
    # La barra en curso queda con la fecha de hoy y la descarta luego el filtro
    # target_date del recovery; la barra completa conserva su fecha real.
    idx = []
    for x in sub.index:
        ts = pd.Timestamp(x)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        idx.append(ts.normalize())
    sub.index = pd.DatetimeIndex(idx)

    # Algunos tickers pueden no traer adjclose (cuando es identico a close, yq
    # a veces lo omite). Fallback a close.
    adj_series = sub["adjclose"] if "adjclose" in sub.columns else sub["close"]

    out = pd.DataFrame({
        "Open":      sub["open"].astype("float64"),
        "High":      sub["high"].astype("float64"),
        "Low":       sub["low"].astype("float64"),
        "Close":     sub["close"].astype("float64"),
        "Adj Close": adj_series.astype("float64"),
        "Volume":    sub["volume"].fillna(0).astype("int64"),
    }, index=sub.index)
    return out.dropna(subset=["Close"])
