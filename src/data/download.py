"""
download.py
Descarga datos OHLCV históricos desde Stooq (gratuito, sin API key).
yfinance se mantiene como fuente de respaldo para actualizaciones.

Fuentes disponibles:
  - Stooq     : fuente primaria, datos históricos desde 2000, sin límite
  - yfinance  : respaldo para actualizaciones puntuales
  - FMP/AV    : reservado para datos fundamentales y earnings
"""

import time
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from datetime import date, datetime
from src.utils.config import ALL_TICKERS, START_DATE
from src.data.database import upsert_precios


# ─────────────────────────────────────────────────────────────
# STOOQ — Fuente primaria (gratuita, sin API key)
# ─────────────────────────────────────────────────────────────

def descargar_ticker_stooq(ticker: str, start: str = START_DATE,
                            end: str = None) -> pd.DataFrame:
    """
    Descarga OHLCV histórico de un ticker desde Stooq.

    Args:
        ticker: símbolo del activo (ej: "JPM")
        start:  fecha inicio "YYYY-MM-DD"
        end:    fecha fin (None = hoy)

    Returns:
        DataFrame con columnas: ticker, fecha, open, high, low, close, volume, adj_close
    """
    end = end or str(date.today())
    stooq_ticker = f"{ticker}.US"

    try:
        raw = web.DataReader(
            stooq_ticker, "stooq",
            start=datetime.strptime(start, "%Y-%m-%d"),
            end=datetime.strptime(end, "%Y-%m-%d"),
        )

        if raw.empty:
            print(f"  [WARN] {ticker}: sin datos en Stooq.")
            return pd.DataFrame()

        # Stooq devuelve orden descendente — invertir
        raw = raw.sort_index(ascending=True).reset_index()
        raw.columns = [c.lower() for c in raw.columns]

        # Construir el DataFrame con dict para que pandas haga broadcast correcto
        df = pd.DataFrame({
            "ticker":    ticker,
            "fecha":     pd.to_datetime(raw["date"]).dt.date,
            "open":      raw["open"].round(4).values,
            "high":      raw["high"].round(4).values,
            "low":       raw["low"].round(4).values,
            "close":     raw["close"].round(4).values,
            "volume":    raw["volume"].fillna(0).astype("int64").values,
            "adj_close": raw["close"].round(4).values,
        })

        df = df.dropna(subset=["close"]).copy()

        print(
            f"  {ticker}: {len(df)} registros "
            f"({df['fecha'].min()} -> {df['fecha'].max()})"
        )
        return df

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return pd.DataFrame()


def descargar_todos(tickers: list = None, start: str = START_DATE,
                    end: str = None, guardar_db: bool = True,
                    delay: float = 0.5) -> dict:
    """
    Descarga OHLCV para todos los tickers desde Stooq.

    Args:
        tickers:    lista de tickers (None = ALL_TICKERS)
        start:      fecha inicio "YYYY-MM-DD"
        end:        fecha fin (None = hoy)
        guardar_db: si True, persiste en PostgreSQL
        delay:      segundos entre requests

    Returns:
        dict  {ticker: DataFrame}
    """
    tickers = tickers or ALL_TICKERS
    resultados = {}

    print(f"\nDescargando {len(tickers)} activos desde Stooq (desde {start})...\n")
    print("-" * 55)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:02d}/{len(tickers)}] ", end="")
        df = descargar_ticker_stooq(ticker, start=start, end=end)

        if df.empty:
            time.sleep(delay)
            continue

        resultados[ticker] = df

        if guardar_db:
            upsert_precios(df)

        time.sleep(delay)

    print("-" * 55)
    print(f"Descarga completada: {len(resultados)}/{len(tickers)} activos OK.")
    return resultados


# ─────────────────────────────────────────────────────────────
# yfinance — Respaldo para actualización de datos recientes
# ─────────────────────────────────────────────────────────────

def actualizar_ticker_yf(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Actualización puntual con yfinance (últimos días).
    Usar solo cuando Stooq no tenga el dato más reciente.
    """
    end = end or str(date.today())
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)

        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]

        df = pd.DataFrame()
        df["ticker"]    = ticker
        df["fecha"]     = pd.to_datetime(raw["date"]).dt.date
        df["open"]      = raw["open"].round(4)
        df["high"]      = raw["high"].round(4)
        df["low"]       = raw["low"].round(4)
        df["close"]     = raw["close"].round(4)
        df["volume"]    = raw["volume"].astype("int64")
        df["adj_close"] = raw["close"].round(4)

        return df.dropna(subset=["close"]).copy()

    except Exception as e:
        print(f"  [ERROR yf] {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Lectura desde PostgreSQL
# ─────────────────────────────────────────────────────────────

def obtener_precios_db(ticker: str, start: str = None,
                       end: str = None) -> pd.DataFrame:
    """Lee precios de PostgreSQL para un ticker."""
    from src.data.database import query_df

    where_clauses = ["ticker = :ticker"]
    params = {"ticker": ticker}

    if start:
        where_clauses.append("fecha >= :start")
        params["start"] = start
    if end:
        where_clauses.append("fecha <= :end")
        params["end"] = end

    where = " AND ".join(where_clauses)
    sql = f"""
        SELECT ticker, fecha, open, high, low, close, volume, adj_close
        FROM precios_diarios
        WHERE {where}
        ORDER BY fecha ASC
    """
    df = query_df(sql, params=params)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha")
