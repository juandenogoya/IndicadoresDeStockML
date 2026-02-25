"""
rule_based.py
Motor de scoring técnico basado en reglas.

Calcula 6 condiciones binarias, un score ponderado y genera la señal
LONG / NEUTRAL para cada activo en cada sesión.

Estructura del scoring:
    Condición           Peso    Señal alcista si...
    ─────────────────────────────────────────────────
    RSI14               20%     RSI < umbral oversold (35)
    MACD Histograma     20%     macd_hist > 0 (momentum positivo)
    Precio vs SMA21     10%     close > SMA21
    Precio vs SMA50     15%     close > SMA50
    Precio vs SMA200    20%     close > SMA200
    Momentum            15%     momentum > 0 (precio sube vs N días atrás)

    score_ponderado = suma de pesos de condiciones cumplidas
    senal = LONG si score_ponderado >= SCORE_ENTRADA_UMBRAL (0.60)
            NEUTRAL en caso contrario
"""

import pandas as pd
import numpy as np
from src.utils.config import (
    ALL_TICKERS, START_DATE,
    SCORING_WEIGHTS, SCORE_ENTRADA_UMBRAL,
    RSI_OVERSOLD,
)
from src.data.database import query_df


# ─────────────────────────────────────────────────────────────
# Cálculo de scoring para un DataFrame de indicadores
# ─────────────────────────────────────────────────────────────

def calcular_scoring(df_ind: pd.DataFrame, df_precios: pd.DataFrame,
                     ticker: str) -> pd.DataFrame:
    """
    Calcula el scoring rule-based combinando indicadores y precios.

    Args:
        df_ind:    DataFrame con indicadores técnicos del ticker
        df_precios: DataFrame con precios OHLCV del ticker
        ticker:    código del activo

    Returns:
        DataFrame listo para upsert en scoring_tecnico
    """
    # ── Merge indicadores + precios (por fecha) ───────────────
    df = pd.merge(
        df_ind[["fecha", "rsi14", "macd_hist", "sma21", "sma50",
                "sma200", "momentum"]],
        df_precios[["fecha", "close"]],
        on="fecha",
        how="inner",
    ).sort_values("fecha").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # ── 6 Condiciones Binarias ────────────────────────────────

    # 1. RSI en zona oversold (señal de posible rebote alcista)
    df["cond_rsi"] = df["rsi14"] < RSI_OVERSOLD

    # 2. MACD histograma positivo (momentum alcista)
    df["cond_macd"] = df["macd_hist"] > 0

    # 3. Precio por encima de SMA21 (tendencia corta alcista)
    df["cond_sma21"] = df["close"] > df["sma21"]

    # 4. Precio por encima de SMA50 (tendencia media alcista)
    df["cond_sma50"] = df["close"] > df["sma50"]

    # 5. Precio por encima de SMA200 (tendencia larga alcista)
    df["cond_sma200"] = df["close"] > df["sma200"]

    # 6. Momentum positivo (precio mayor que N días atrás)
    df["cond_momentum"] = df["momentum"] > 0

    # ── Score Ponderado ───────────────────────────────────────
    w = SCORING_WEIGHTS
    df["score_ponderado"] = (
        df["cond_rsi"].astype(float)      * w["rsi"]      +
        df["cond_macd"].astype(float)     * w["macd"]     +
        df["cond_sma21"].astype(float)    * w["sma21"]    +
        df["cond_sma50"].astype(float)    * w["sma50"]    +
        df["cond_sma200"].astype(float)   * w["sma200"]   +
        df["cond_momentum"].astype(float) * w["momentum"]
    ).round(4)

    # ── Condiciones OK (cantidad de condiciones cumplidas) ────
    cond_cols = ["cond_rsi", "cond_macd", "cond_sma21",
                 "cond_sma50", "cond_sma200", "cond_momentum"]
    df["condiciones_ok"] = df[cond_cols].sum(axis=1).astype(int)

    # ── Señal ─────────────────────────────────────────────────
    df["senal"] = df["score_ponderado"].apply(
        lambda s: "LONG" if s >= SCORE_ENTRADA_UMBRAL else "NEUTRAL"
    )

    # ── Seleccionar columnas para la DB ───────────────────────
    result = df[[
        "fecha", "cond_rsi", "cond_macd", "cond_sma21",
        "cond_sma50", "cond_sma200", "cond_momentum",
        "score_ponderado", "condiciones_ok", "senal",
    ]].copy()

    result["ticker"] = ticker

    # Reordenar columnas
    result = result[[
        "ticker", "fecha", "cond_rsi", "cond_macd", "cond_sma21",
        "cond_sma50", "cond_sma200", "cond_momentum",
        "score_ponderado", "condiciones_ok", "senal",
    ]]

    return result


# ─────────────────────────────────────────────────────────────
# Proceso completo: leer DB, calcular, persistir
# ─────────────────────────────────────────────────────────────

def procesar_scoring_ticker(ticker: str, guardar_db: bool = True) -> pd.DataFrame:
    """
    Lee indicadores y precios de la DB, calcula scoring y persiste.

    Args:
        ticker:     código del activo
        guardar_db: si True, upsert en scoring_tecnico

    Returns:
        DataFrame con scoring calculado
    """
    from src.data.database import upsert_scoring

    # Leer indicadores
    df_ind = query_df("""
        SELECT fecha, rsi14, macd_hist, sma21, sma50, sma200, momentum
        FROM indicadores_tecnicos
        WHERE ticker = :ticker
        ORDER BY fecha ASC
    """, params={"ticker": ticker})

    if df_ind.empty:
        print(f"  [WARN] {ticker}: sin indicadores en DB.")
        return pd.DataFrame()

    df_ind["fecha"] = pd.to_datetime(df_ind["fecha"])

    # Leer precios
    df_precios = query_df("""
        SELECT fecha, close
        FROM precios_diarios
        WHERE ticker = :ticker
        ORDER BY fecha ASC
    """, params={"ticker": ticker})

    if df_precios.empty:
        print(f"  [WARN] {ticker}: sin precios en DB.")
        return pd.DataFrame()

    df_precios["fecha"] = pd.to_datetime(df_precios["fecha"])

    # Calcular scoring
    result = calcular_scoring(df_ind, df_precios, ticker)

    if result.empty:
        print(f"  [WARN] {ticker}: scoring vacío.")
        return pd.DataFrame()

    longs   = (result["senal"] == "LONG").sum()
    neutros = (result["senal"] == "NEUTRAL").sum()
    print(
        f"  {ticker}: {len(result)} sesiones | "
        f"LONG: {longs} ({longs/len(result)*100:.1f}%) | "
        f"NEUTRAL: {neutros}"
    )

    if guardar_db:
        upsert_scoring(result)

    return result


def procesar_scoring_todos(tickers: list = None,
                           guardar_db: bool = True) -> dict:
    """
    Calcula el scoring para todos los activos del universo.

    Returns:
        dict {ticker: DataFrame scoring}
    """
    tickers = tickers or ALL_TICKERS
    resultados = {}

    print(f"\nCalculando scoring rule-based para {len(tickers)} activos...\n")
    print("-" * 65)

    for ticker in tickers:
        df = procesar_scoring_ticker(ticker, guardar_db=guardar_db)
        if not df.empty:
            resultados[ticker] = df

    print("-" * 65)
    print(f"Scoring completado: {len(resultados)}/{len(tickers)} activos OK.")
    return resultados


# ─────────────────────────────────────────────────────────────
# Lectura y análisis del scoring desde DB
# ─────────────────────────────────────────────────────────────

def obtener_scoring_db(ticker: str, start: str = None,
                       end: str = None) -> pd.DataFrame:
    """Lee el scoring histórico de la DB para un ticker."""
    where = ["ticker = :ticker"]
    params = {"ticker": ticker}
    if start:
        where.append("fecha >= :start")
        params["start"] = start
    if end:
        where.append("fecha <= :end")
        params["end"] = end

    sql = f"""
        SELECT ticker, fecha, cond_rsi, cond_macd, cond_sma21,
               cond_sma50, cond_sma200, cond_momentum,
               score_ponderado, condiciones_ok, senal
        FROM scoring_tecnico
        WHERE {" AND ".join(where)}
        ORDER BY fecha ASC
    """
    df = query_df(sql, params=params)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def resumen_scoring(ticker: str = None) -> pd.DataFrame:
    """
    Genera un resumen del scoring por ticker:
    total sesiones, señales LONG y %, score promedio.
    """
    where = f"WHERE ticker = '{ticker}'" if ticker else ""
    sql = f"""
        SELECT
            ticker,
            COUNT(*) as total_sesiones,
            SUM(CASE WHEN senal = 'LONG' THEN 1 ELSE 0 END) as total_long,
            ROUND(
                SUM(CASE WHEN senal = 'LONG' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
            ) as pct_long,
            ROUND(AVG(score_ponderado)::NUMERIC, 4) as score_promedio,
            ROUND(AVG(condiciones_ok)::NUMERIC, 2) as cond_promedio
        FROM scoring_tecnico
        {where}
        GROUP BY ticker
        ORDER BY pct_long DESC
    """
    return query_df(sql)


def señal_actual(tickers: list = None) -> pd.DataFrame:
    """
    Retorna la señal más reciente de cada ticker.
    Útil para el dashboard de monitoreo diario.
    """
    tickers = tickers or ALL_TICKERS
    tickers_sql = "', '".join(tickers)

    sql = f"""
        SELECT DISTINCT ON (ticker)
            ticker, fecha, score_ponderado, condiciones_ok, senal,
            cond_rsi, cond_macd, cond_sma21, cond_sma50, cond_sma200, cond_momentum
        FROM scoring_tecnico
        WHERE ticker IN ('{tickers_sql}')
        ORDER BY ticker, fecha DESC
    """
    df = query_df(sql)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("score_ponderado", ascending=False).reset_index(drop=True)
