"""
sector_features.py
Calcula features sectoriales usando Z-Score y métricas de breadth.

Para cada ticker y fecha calcula su posición relativa dentro del sector:
    z_rsi_sector         : RSI normalizado vs peers del sector
    z_retorno_1d_sector  : outperformance diaria vs sector
    z_retorno_5d_sector  : outperformance semanal vs sector
    z_vol_sector         : volumen relativo vs sector
    z_dist_sma50_sector  : distancia a SMA50 relativa al sector
    z_adx_sector         : fuerza de tendencia vs sector
    pct_long_sector      : % de tickers del sector con señal LONG
    rank_retorno_sector  : ranking del ticker por retorno (1 = mejor)
    rsi_sector_avg       : RSI promedio del sector
    adx_sector_avg       : ADX promedio del sector
    retorno_1d_sector_avg: retorno medio del sector
"""

import pandas as pd
import numpy as np
from src.data.database import query_df, get_connection
import psycopg2.extras


# ─────────────────────────────────────────────────────────────
# Carga de datos completos (todos los tickers a la vez)
# ─────────────────────────────────────────────────────────────

def cargar_datos_completos() -> pd.DataFrame:
    """
    Carga precios + indicadores + scoring + sector para todos los tickers.
    Retorna un DataFrame con todas las columnas necesarias para el cálculo.
    """
    sql = """
        SELECT
            p.ticker,
            a.nombre,
            a.sector,
            p.fecha,
            p.close,
            i.rsi14,
            i.macd_hist,
            i.dist_sma50,
            i.adx,
            i.vol_relativo,
            s.score_ponderado,
            s.senal
        FROM precios_diarios      p
        JOIN activos              a ON p.ticker = a.ticker
        JOIN indicadores_tecnicos i ON p.ticker = i.ticker AND p.fecha = i.fecha
        JOIN scoring_tecnico      s ON p.ticker = s.ticker AND p.fecha = s.fecha
        ORDER BY p.ticker, p.fecha
    """
    df = query_df(sql)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


# ─────────────────────────────────────────────────────────────
# Cálculo de Z-Scores y métricas sectoriales
# ─────────────────────────────────────────────────────────────

def calcular_features_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los features sectoriales sobre el DataFrame completo.

    Args:
        df: DataFrame con columnas de todos los tickers (resultado de cargar_datos_completos)

    Returns:
        DataFrame con features sectoriales, una fila por (ticker, fecha)
    """
    df = df.copy().sort_values(["ticker", "fecha"]).reset_index(drop=True)

    # ── Retornos (price action) ───────────────────────────────
    df["retorno_1d"] = df.groupby("ticker")["close"].pct_change(1) * 100
    df["retorno_5d"] = df.groupby("ticker")["close"].pct_change(5) * 100

    # ── Señal LONG binaria (para breadth) ─────────────────────
    df["es_long"] = (df["senal"] == "LONG").astype(float)

    # ── Función auxiliar para Z-Score por sector/fecha ────────
    def z_score_sector(df: pd.DataFrame, col: str) -> pd.Series:
        """Z-score de una columna agrupado por sector y fecha."""
        grp   = df.groupby(["sector", "fecha"])[col]
        media = grp.transform("mean")
        std   = grp.transform("std")
        return ((df[col] - media) / std.replace(0, np.nan)).round(4)

    # ── Z-Scores ──────────────────────────────────────────────
    df["z_rsi_sector"]         = z_score_sector(df, "rsi14")
    df["z_retorno_1d_sector"]  = z_score_sector(df, "retorno_1d")
    df["z_retorno_5d_sector"]  = z_score_sector(df, "retorno_5d")
    df["z_vol_sector"]         = z_score_sector(df, "vol_relativo")
    df["z_dist_sma50_sector"]  = z_score_sector(df, "dist_sma50")
    df["z_adx_sector"]         = z_score_sector(df, "adx")

    # ── Breadth: % tickers LONG en el sector ─────────────────
    df["pct_long_sector"] = (
        df.groupby(["sector", "fecha"])["es_long"]
        .transform("mean")
        .round(4)
    )

    # ── Ranking por retorno diario dentro del sector ──────────
    df["rank_retorno_sector"] = (
        df.groupby(["sector", "fecha"])["retorno_1d"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    # ── Promedios sectoriales (para contexto) ─────────────────
    df["rsi_sector_avg"] = (
        df.groupby(["sector", "fecha"])["rsi14"]
        .transform("mean").round(4)
    )
    df["adx_sector_avg"] = (
        df.groupby(["sector", "fecha"])["adx"]
        .transform("mean").round(4)
    )
    df["retorno_1d_sector_avg"] = (
        df.groupby(["sector", "fecha"])["retorno_1d"]
        .transform("mean").round(4)
    )

    # ── Seleccionar columnas finales ──────────────────────────
    cols = [
        "ticker", "fecha", "sector",
        "z_rsi_sector", "z_retorno_1d_sector", "z_retorno_5d_sector",
        "z_vol_sector", "z_dist_sma50_sector", "z_adx_sector",
        "pct_long_sector", "rank_retorno_sector",
        "rsi_sector_avg", "adx_sector_avg", "retorno_1d_sector_avg",
    ]
    result = df[cols].copy()
    result["fecha"] = result["fecha"].dt.date

    # Eliminar filas donde no hay suficientes datos para Z-score
    # (primeras filas de retornos, sectores con 1 solo ticker en alguna fecha)
    result = result.dropna(subset=["z_rsi_sector"]).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────────────────────
# Persistencia en PostgreSQL
# ─────────────────────────────────────────────────────────────

def upsert_features_sector(df: pd.DataFrame):
    """Inserta o actualiza features_sector en PostgreSQL."""
    # Convertir rank a int nativo para psycopg2
    df = df.copy()
    df["rank_retorno_sector"] = df["rank_retorno_sector"].astype(object).where(
        df["rank_retorno_sector"].notna(), None
    )

    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        # Convertir NaN a None para PostgreSQL
        rec = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
               for k, v in rec.items()}
        # Convertir numpy int a Python int
        if rec.get("rank_retorno_sector") is not None:
            rec["rank_retorno_sector"] = int(rec["rank_retorno_sector"])
        records.append(rec)

    sql = """
        INSERT INTO features_sector
            (ticker, fecha, sector,
             z_rsi_sector, z_retorno_1d_sector, z_retorno_5d_sector,
             z_vol_sector, z_dist_sma50_sector, z_adx_sector,
             pct_long_sector, rank_retorno_sector,
             rsi_sector_avg, adx_sector_avg, retorno_1d_sector_avg)
        VALUES
            (%(ticker)s, %(fecha)s, %(sector)s,
             %(z_rsi_sector)s, %(z_retorno_1d_sector)s, %(z_retorno_5d_sector)s,
             %(z_vol_sector)s, %(z_dist_sma50_sector)s, %(z_adx_sector)s,
             %(pct_long_sector)s, %(rank_retorno_sector)s,
             %(rsi_sector_avg)s, %(adx_sector_avg)s, %(retorno_1d_sector_avg)s)
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            z_rsi_sector          = EXCLUDED.z_rsi_sector,
            z_retorno_1d_sector   = EXCLUDED.z_retorno_1d_sector,
            z_retorno_5d_sector   = EXCLUDED.z_retorno_5d_sector,
            z_vol_sector          = EXCLUDED.z_vol_sector,
            z_dist_sma50_sector   = EXCLUDED.z_dist_sma50_sector,
            z_adx_sector          = EXCLUDED.z_adx_sector,
            pct_long_sector       = EXCLUDED.pct_long_sector,
            rank_retorno_sector   = EXCLUDED.rank_retorno_sector,
            rsi_sector_avg        = EXCLUDED.rsi_sector_avg,
            adx_sector_avg        = EXCLUDED.adx_sector_avg,
            retorno_1d_sector_avg = EXCLUDED.retorno_1d_sector_avg
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"  features_sector upsert: {len(records):,} registros.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def procesar_features_sector(guardar_db: bool = True) -> pd.DataFrame:
    """
    Pipeline completo: carga datos, calcula features, persiste.

    Returns:
        DataFrame con features sectoriales calculados
    """
    print("  Cargando datos de todos los tickers...", end=" ")
    df_raw = cargar_datos_completos()
    print(f"{len(df_raw):,} registros ({df_raw['ticker'].nunique()} tickers)")

    print("  Calculando Z-scores y metricas sectoriales...", end=" ")
    df_features = calcular_features_sector(df_raw)
    print(f"{len(df_features):,} registros listos.")

    # Resumen por sector
    print("\n  Muestra de Z-scores (ultima fecha disponible):")
    ultima = df_features[df_features["fecha"] == df_features["fecha"].max()]
    cols_show = ["ticker", "sector", "z_rsi_sector", "z_retorno_1d_sector",
                 "pct_long_sector", "rank_retorno_sector"]
    print(ultima[cols_show].sort_values(
        ["sector", "rank_retorno_sector"]).to_string(index=False))

    if guardar_db:
        upsert_features_sector(df_features)

    return df_features
