"""
technical.py
Cálculo de todos los indicadores técnicos del Modelo AT.
Usa la librería `ta` sobre DataFrames OHLCV.
"""

import pandas as pd
import numpy as np
import ta
from src.utils.config import (
    SMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, BB_PERIOD, BB_STD, ADX_PERIOD, VOL_MA_PERIOD,
)
from src.data.database import upsert_indicadores


# ─────────────────────────────────────────────────────────────
# Cálculo de indicadores para un DataFrame OHLCV
# ─────────────────────────────────────────────────────────────

def calcular_indicadores(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Recibe un DataFrame con columnas: fecha, open, high, low, close, volume
    Retorna un DataFrame con todos los indicadores calculados.

    Args:
        df:     DataFrame OHLCV (puede tener fecha como columna o index)
        ticker: código del activo

    Returns:
        DataFrame con columnas de indicadores, listo para persistir en DB
    """
    df = df.copy()

    # Normalizar: fecha como columna
    if df.index.name == "fecha" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "fecha" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "fecha"})

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    result = pd.DataFrame()
    result["fecha"]  = df["fecha"].dt.date
    result["ticker"] = ticker  # asignar DESPUÉS para que pandas haga broadcast correcto

    # ── Medias Móviles Simples ────────────────────────────────
    result["sma21"]  = close.rolling(SMA_PERIODS[0]).mean().round(4)
    result["sma50"]  = close.rolling(SMA_PERIODS[1]).mean().round(4)
    result["sma200"] = close.rolling(SMA_PERIODS[2]).mean().round(4)

    # ── Distancia % del precio de cierre a cada SMA ───────────
    # Positivo = precio sobre la SMA, Negativo = precio bajo la SMA
    result["dist_sma21"]  = ((close - result["sma21"])  / result["sma21"]  * 100).round(4)
    result["dist_sma50"]  = ((close - result["sma50"])  / result["sma50"]  * 100).round(4)
    result["dist_sma200"] = ((close - result["sma200"]) / result["sma200"] * 100).round(4)

    # ── RSI ───────────────────────────────────────────────────
    rsi = ta.momentum.RSIIndicator(close=close, window=RSI_PERIOD)
    result["rsi14"] = rsi.rsi().round(4)

    # ── MACD ──────────────────────────────────────────────────
    macd_ind = ta.trend.MACD(
        close=close,
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    result["macd"]        = macd_ind.macd().round(6)
    result["macd_signal"] = macd_ind.macd_signal().round(6)
    result["macd_hist"]   = macd_ind.macd_diff().round(6)   # histograma = MACD - Signal

    # ── ATR (volatilidad) ─────────────────────────────────────
    atr_ind = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=ATR_PERIOD
    )
    result["atr14"] = atr_ind.average_true_range().round(4)

    # ── Bandas de Bollinger ───────────────────────────────────
    bb = ta.volatility.BollingerBands(
        close=close, window=BB_PERIOD, window_dev=BB_STD
    )
    result["bb_upper"]  = bb.bollinger_hband().round(4)
    result["bb_middle"] = bb.bollinger_mavg().round(4)
    result["bb_lower"]  = bb.bollinger_lband().round(4)

    # ── OBV (On-Balance Volume) ───────────────────────────────
    obv_ind = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    result["obv"] = obv_ind.on_balance_volume().round(2)

    # ── Volumen Relativo (vs promedio móvil de N días) ────────
    vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
    result["vol_relativo"] = (volume / vol_ma).round(4)

    # ── ADX (fuerza de tendencia) ─────────────────────────────
    adx_ind = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=ADX_PERIOD
    )
    result["adx"] = adx_ind.adx().round(4)

    # ── Momentum (diferencia de precio a N períodos) ──────────
    # Usamos el mismo período que RSI para consistencia
    result["momentum"] = close.diff(RSI_PERIOD).round(4)

    # ── Limpiar filas sin suficientes datos (inicio de serie) ──
    result = result.dropna(subset=["sma200"]).copy()   # SMA200 es la más restrictiva

    return result


# ─────────────────────────────────────────────────────────────
# Proceso completo: calcular + guardar en DB
# ─────────────────────────────────────────────────────────────

def procesar_indicadores_ticker(ticker: str, df_ohlcv: pd.DataFrame,
                                 guardar_db: bool = True) -> pd.DataFrame:
    """
    Calcula indicadores para un ticker y opcionalmente los persiste en DB.

    Args:
        ticker:     código del activo
        df_ohlcv:   DataFrame OHLCV
        guardar_db: si True, persiste en indicadores_tecnicos

    Returns:
        DataFrame con indicadores calculados
    """
    print(f"  Calculando indicadores: {ticker}...", end=" ")

    if df_ohlcv.empty:
        print("sin datos.")
        return pd.DataFrame()

    df_ind = calcular_indicadores(df_ohlcv, ticker)

    if df_ind.empty:
        print("resultado vacio.")
        return pd.DataFrame()

    print(f"{len(df_ind)} registros.")

    if guardar_db:
        upsert_indicadores(df_ind)

    return df_ind


def procesar_indicadores_todos(datos: dict, guardar_db: bool = True) -> dict:
    """
    Procesa indicadores para todos los activos.

    Args:
        datos:      dict {ticker: DataFrame OHLCV}
        guardar_db: persistir en DB

    Returns:
        dict {ticker: DataFrame indicadores}
    """
    resultados = {}
    print(f"\nCalculando indicadores para {len(datos)} activos...\n")
    print("-" * 50)

    for ticker, df in datos.items():
        df_ind = procesar_indicadores_ticker(ticker, df, guardar_db=guardar_db)
        if not df_ind.empty:
            resultados[ticker] = df_ind

    print("-" * 50)
    print(f"Indicadores procesados: {len(resultados)}/{len(datos)} activos OK.")
    return resultados


# ─────────────────────────────────────────────────────────────
# Utilidad: leer indicadores desde DB
# ─────────────────────────────────────────────────────────────

def obtener_indicadores_db(ticker: str, start: str = None,
                            end: str = None) -> pd.DataFrame:
    """
    Lee indicadores técnicos de la base de datos para un ticker.
    """
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
        SELECT *
        FROM indicadores_tecnicos
        WHERE {where}
        ORDER BY fecha ASC
    """
    df = query_df(sql, params=params)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df
