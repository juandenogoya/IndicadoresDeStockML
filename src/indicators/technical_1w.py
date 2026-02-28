"""
technical_1w.py
Calculo de indicadores tecnicos sobre barras semanales (1W).

Espejo exacto de technical.py, adaptado para:
- Input:  precios_semanales (OHLCV semanal, fecha_semana como fecha)
- Output: indicadores_tecnicos_1w

Mismos periodos que 1D — en contexto semanal:
    SMA21   = 21 semanas (~5 meses)
    SMA50   = 50 semanas (~1 ano)
    SMA200  = 200 semanas (~4 anos)
    RSI14   = 14 semanas
    MACD    = 12/26/9 semanas
    ATR14   = 14 semanas
    BB20    = 20 semanas
    ADX14   = 14 semanas
"""

import pandas as pd
import numpy as np
import psycopg2.extras
import ta

from src.utils.config import (
    SMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, BB_PERIOD, BB_STD, ADX_PERIOD, VOL_MA_PERIOD,
)
from src.data.database import get_connection, query_df


# ─────────────────────────────────────────────────────────────
# Upsert indicadores 1W
# ─────────────────────────────────────────────────────────────

def upsert_indicadores_1w(df: pd.DataFrame):
    """
    Inserta o actualiza indicadores tecnicos semanales en indicadores_tecnicos_1w.
    Patron identico a upsert_indicadores() de database.py.
    """
    records = df.to_dict(orient="records")

    # Sanitizar NaN -> None
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None

    sql = """
        INSERT INTO indicadores_tecnicos_1w
            (ticker, fecha, sma21, sma50, sma200,
             dist_sma21, dist_sma50, dist_sma200,
             rsi14, macd, macd_signal, macd_hist,
             atr14, bb_upper, bb_middle, bb_lower,
             obv, vol_relativo, adx, momentum)
        VALUES
            (%(ticker)s, %(fecha)s, %(sma21)s, %(sma50)s, %(sma200)s,
             %(dist_sma21)s, %(dist_sma50)s, %(dist_sma200)s,
             %(rsi14)s, %(macd)s, %(macd_signal)s, %(macd_hist)s,
             %(atr14)s, %(bb_upper)s, %(bb_middle)s, %(bb_lower)s,
             %(obv)s, %(vol_relativo)s, %(adx)s, %(momentum)s)
        ON CONFLICT (ticker, fecha)
        DO UPDATE SET
            sma21        = EXCLUDED.sma21,
            sma50        = EXCLUDED.sma50,
            sma200       = EXCLUDED.sma200,
            dist_sma21   = EXCLUDED.dist_sma21,
            dist_sma50   = EXCLUDED.dist_sma50,
            dist_sma200  = EXCLUDED.dist_sma200,
            rsi14        = EXCLUDED.rsi14,
            macd         = EXCLUDED.macd,
            macd_signal  = EXCLUDED.macd_signal,
            macd_hist    = EXCLUDED.macd_hist,
            atr14        = EXCLUDED.atr14,
            bb_upper     = EXCLUDED.bb_upper,
            bb_middle    = EXCLUDED.bb_middle,
            bb_lower     = EXCLUDED.bb_lower,
            obv          = EXCLUDED.obv,
            vol_relativo = EXCLUDED.vol_relativo,
            adx          = EXCLUDED.adx,
            momentum     = EXCLUDED.momentum,
            updated_at   = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)


# ─────────────────────────────────────────────────────────────
# Calculo de indicadores sobre OHLCV semanal
# ─────────────────────────────────────────────────────────────

def calcular_indicadores_1w(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calcula todos los indicadores tecnicos sobre barras semanales.
    Logica identica a calcular_indicadores() de technical.py.

    Args:
        df:     DataFrame OHLCV semanal con columnas:
                [fecha_semana (o fecha), open, high, low, close, volume, adj_close]
        ticker: codigo del activo

    Returns:
        DataFrame con indicadores calculados, listo para persistir en indicadores_tecnicos_1w.
        Columna 'fecha' = fecha_semana (ultimo dia habil de la semana).
    """
    df = df.copy()

    # Normalizar nombre de columna de fecha
    if "fecha_semana" in df.columns:
        df = df.rename(columns={"fecha_semana": "fecha"})

    if df.index.name in ("fecha", "fecha_semana") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    result = pd.DataFrame()
    result["fecha"]  = df["fecha"].dt.date
    result["ticker"] = ticker

    # ── Medias Moviles Simples ─────────────────────────────────
    result["sma21"]  = close.rolling(SMA_PERIODS[0]).mean().round(4)
    result["sma50"]  = close.rolling(SMA_PERIODS[1]).mean().round(4)
    result["sma200"] = close.rolling(SMA_PERIODS[2]).mean().round(4)

    # ── Distancia % del cierre a cada SMA ─────────────────────
    result["dist_sma21"]  = ((close - result["sma21"])  / result["sma21"]  * 100).round(4)
    result["dist_sma50"]  = ((close - result["sma50"])  / result["sma50"]  * 100).round(4)
    result["dist_sma200"] = ((close - result["sma200"]) / result["sma200"] * 100).round(4)

    # ── RSI ────────────────────────────────────────────────────
    rsi_ind = ta.momentum.RSIIndicator(close=close, window=RSI_PERIOD)
    result["rsi14"] = rsi_ind.rsi().round(4)

    # ── MACD ───────────────────────────────────────────────────
    macd_ind = ta.trend.MACD(
        close=close,
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    result["macd"]        = macd_ind.macd().round(6)
    result["macd_signal"] = macd_ind.macd_signal().round(6)
    result["macd_hist"]   = macd_ind.macd_diff().round(6)

    # ── ATR ────────────────────────────────────────────────────
    atr_ind = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=ATR_PERIOD
    )
    result["atr14"] = atr_ind.average_true_range().round(4)

    # ── Bandas de Bollinger ────────────────────────────────────
    bb = ta.volatility.BollingerBands(
        close=close, window=BB_PERIOD, window_dev=BB_STD
    )
    result["bb_upper"]  = bb.bollinger_hband().round(4)
    result["bb_middle"] = bb.bollinger_mavg().round(4)
    result["bb_lower"]  = bb.bollinger_lband().round(4)

    # ── OBV ────────────────────────────────────────────────────
    obv_ind = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    result["obv"] = obv_ind.on_balance_volume().round(2)

    # ── Volumen Relativo ───────────────────────────────────────
    vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
    result["vol_relativo"] = (volume / vol_ma).round(4)

    # ── ADX ────────────────────────────────────────────────────
    adx_ind = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=ADX_PERIOD
    )
    result["adx"] = adx_ind.adx().round(4)

    # ── Momentum ───────────────────────────────────────────────
    result["momentum"] = close.diff(RSI_PERIOD).round(4)

    # ── Eliminar filas sin suficientes datos (warmup SMA200) ───
    # Con 260 semanas tenemos suficiente para ~60 semanas post-warmup
    result = result.dropna(subset=["sma200"]).copy()

    return result


# ─────────────────────────────────────────────────────────────
# Cargar precios semanales de un ticker desde Railway
# ─────────────────────────────────────────────────────────────

def cargar_semanales_ticker(ticker: str) -> pd.DataFrame:
    """
    Carga todos los precios semanales de un ticker desde precios_semanales.

    Returns:
        DataFrame con [fecha_semana, open, high, low, close, volume, adj_close]
        Ordenado por fecha_semana ASC.
    """
    return query_df(
        """
        SELECT fecha_semana, open, high, low, close, volume, adj_close
        FROM   precios_semanales
        WHERE  ticker = :ticker
        ORDER  BY fecha_semana ASC
        """,
        params={"ticker": ticker},
    )


# ─────────────────────────────────────────────────────────────
# Pipeline completo para un ticker
# ─────────────────────────────────────────────────────────────

def procesar_indicadores_ticker_1w(ticker: str,
                                    guardar_db: bool = True) -> pd.DataFrame:
    """
    Carga precios semanales, calcula indicadores y opcionalmente persiste en DB.

    Args:
        ticker:     codigo del activo
        guardar_db: si True, persiste en indicadores_tecnicos_1w

    Returns:
        DataFrame con indicadores calculados (vacio si hay error).
    """
    df_sem = cargar_semanales_ticker(ticker)

    if df_sem.empty:
        return pd.DataFrame()

    df_ind = calcular_indicadores_1w(df_sem, ticker)

    if df_ind.empty:
        return pd.DataFrame()

    if guardar_db:
        upsert_indicadores_1w(df_ind)

    return df_ind
