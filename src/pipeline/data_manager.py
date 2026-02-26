"""
data_manager.py
Gestion de datos para el scanner de alertas.

Funciones:
    obtener_info_ticker(ticker)    -- busca sector y fechas disponibles en DB
    descargar_yfinance(ticker)     -- descarga historico desde yfinance
    preparar_ticker(ticker, ...)   -- pipeline completo: DB o yfinance + opcional persistencia
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import date, timedelta

from src.data.database import query_df, get_connection
from src.utils.config import TICKER_SECTOR


# ─────────────────────────────────────────────────────────────
# Minimo de barras requeridas para calcular features confiables
# ─────────────────────────────────────────────────────────────

MIN_BARRAS = 250   # ~1 año; necesario para SMA200, ADX, vol_relativo, etc.


# ─────────────────────────────────────────────────────────────
# Consultar si el ticker ya existe en la DB
# ─────────────────────────────────────────────────────────────

def obtener_info_ticker(ticker: str) -> Optional[Dict]:
    """
    Busca informacion del ticker en las tablas activos y precios_diarios.

    Returns:
        dict con keys: sector, n_barras, fecha_min, fecha_max
        None si el ticker no esta en la DB
    """
    sql = """
        SELECT a.sector, COUNT(p.fecha) AS n_barras,
               MIN(p.fecha) AS fecha_min, MAX(p.fecha) AS fecha_max
        FROM activos a
        LEFT JOIN precios_diarios p ON p.ticker = a.ticker
        WHERE a.ticker = :ticker
        GROUP BY a.sector
    """
    try:
        df = query_df(sql, params={"ticker": ticker})
    except Exception:
        return None

    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "sector":    row["sector"],
        "n_barras":  int(row["n_barras"]) if not pd.isna(row["n_barras"]) else 0,
        "fecha_min": row["fecha_min"],
        "fecha_max": row["fecha_max"],
    }


# ─────────────────────────────────────────────────────────────
# Cargar precios desde la DB
# ─────────────────────────────────────────────────────────────

def cargar_precios_db(ticker: str, ultimas_n: int = 500) -> pd.DataFrame:
    """
    Carga las ultimas N barras OHLCV de precios_diarios para un ticker.
    """
    sql = """
        SELECT fecha, open, high, low, close, volume
        FROM precios_diarios
        WHERE ticker = :ticker
          AND close > 0
        ORDER BY fecha DESC
        LIMIT :n
    """
    df = query_df(sql, params={"ticker": ticker, "n": ultimas_n})
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["ticker"] = ticker
    return df


# ─────────────────────────────────────────────────────────────
# Descargar desde yfinance
# ─────────────────────────────────────────────────────────────

def descargar_yfinance(ticker: str, periodo: str = "3y") -> pd.DataFrame:
    """
    Descarga historico OHLCV desde yfinance.

    Args:
        ticker:  simbolo del activo (p.ej. 'AAPL', 'SPY')
        periodo: periodo de descarga ('1y', '2y', '3y', '5y')

    Returns:
        DataFrame con columnas: fecha, open, high, low, close, volume, ticker
        DataFrame vacio si falla la descarga.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [ERROR] yfinance no instalado. Ejecutar: pip install yfinance")
        return pd.DataFrame()

    try:
        raw = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  [WARN] yfinance no retorno datos para {ticker}")
            return pd.DataFrame()

        # Normalizar columnas (yfinance puede retornar MultiIndex)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.reset_index()
        raw = raw.rename(columns={"date": "fecha", "Date": "fecha"})
        raw["fecha"] = pd.to_datetime(raw["fecha"])
        raw = raw[["fecha", "open", "high", "low", "close", "volume"]].copy()
        raw = raw.dropna(subset=["close"])
        raw = raw[raw["close"] > 0]
        raw["ticker"] = ticker
        raw = raw.sort_values("fecha").reset_index(drop=True)

        print(f"  [yfinance] {ticker}: {len(raw)} barras descargadas "
              f"({raw['fecha'].min().date()} a {raw['fecha'].max().date()})")
        return raw

    except Exception as e:
        print(f"  [ERROR] yfinance fallo para {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Persistir ticker nuevo en DB
# ─────────────────────────────────────────────────────────────

def persistir_ticker_nuevo(df_ohlcv: pd.DataFrame, ticker: str,
                            sector: Optional[str], nombre: Optional[str] = None):
    """
    Inserta un ticker nuevo en activos y sus precios en precios_diarios.
    Solo se llama si persistir=True.
    """
    from src.data.database import upsert_precios

    nombre = nombre or ticker

    # Insertar en activos (ON CONFLICT DO NOTHING para idempotencia)
    sql_activo = """
        INSERT INTO activos (ticker, nombre, sector, activo)
        VALUES (%s, %s, %s, TRUE)
        ON CONFLICT (ticker) DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_activo, (ticker, nombre, sector))

    # Upsert precios (upsert_precios espera df con columna adj_close)
    df_save = df_ohlcv.copy()
    if "adj_close" not in df_save.columns:
        df_save["adj_close"] = df_save["close"]  # auto_adjust=True => close ya es ajustado
    upsert_precios(df_save)
    print(f"  [DB] Ticker {ticker} persistido: {len(df_ohlcv)} barras.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal: preparar ticker para el scanner
# ─────────────────────────────────────────────────────────────

def preparar_ticker(ticker: str,
                    persistir: bool = False) -> Tuple[pd.DataFrame, Optional[str], bool]:
    """
    Obtiene datos OHLCV y sector para un ticker, desde DB o yfinance.

    Args:
        ticker:    simbolo del activo
        persistir: si True y el ticker no esta en DB, lo guarda

    Returns:
        (df_ohlcv, sector, es_nuevo)
        df_ohlcv: DataFrame con columnas fecha/open/high/low/close/volume/ticker
        sector:   nombre del sector (None si desconocido)
        es_nuevo: True si el ticker no estaba en la DB

    Raises:
        ValueError si no se pueden obtener datos suficientes.
    """
    ticker = ticker.upper().strip()

    # ── 1. Intentar desde DB ──────────────────────────────────
    info = obtener_info_ticker(ticker)

    if info is not None and info["n_barras"] >= MIN_BARRAS:
        sector = info["sector"]
        df = cargar_precios_db(ticker, ultimas_n=500)
        if len(df) >= MIN_BARRAS:
            return df, sector, False

    # ── 2. Ticker no esta en DB o tiene pocos datos: descargar ─
    es_nuevo = (info is None)
    df = descargar_yfinance(ticker, periodo="3y")

    if df.empty or len(df) < MIN_BARRAS:
        raise ValueError(
            f"Datos insuficientes para {ticker} "
            f"(se requieren >= {MIN_BARRAS} barras, obtenidas: {len(df)})"
        )

    # ── 3. Determinar sector ──────────────────────────────────
    # Primero de TICKER_SECTOR (config), luego de la info de DB
    sector = TICKER_SECTOR.get(ticker)
    if sector is None and info is not None:
        sector = info.get("sector")
    # Si es un ticker nuevo y no lo encontramos, sector = None (usara global)

    # ── 4. Persistir si se indica ─────────────────────────────
    if persistir and es_nuevo:
        persistir_ticker_nuevo(df, ticker, sector)

    return df, sector, es_nuevo
