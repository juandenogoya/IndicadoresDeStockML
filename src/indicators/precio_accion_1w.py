"""
precio_accion_1w.py
Calcula 32 features de estructura de precio y volumen sobre barras semanales (1W).

Espejo exacto de precio_accion.py, adaptado para:
    Input:  precios_semanales JOIN indicadores_tecnicos_1w
    Output: tabla features_precio_accion_1w

Los nombres de columna son identicos al 1D.
Las ventanas rolling (5, 10, 20) son en semanas:
    velas_alcistas_5d  -> 5 semanas (~1 mes)
    velas_alcistas_10d -> 10 semanas (~2.5 meses)
    dist_max_20d       -> 20 semanas (~5 meses)
    chaikin_mf_20      -> 20 semanas

Grupos:
    1. Anatomia de vela  (9 features) : estructura OHLC de la vela semanal
    2. Patrones clasicos (8 features) : senales de inversion/continuacion
    3. Estructura rolling(8 features) : contexto N-semanas de tendencia y rango
    4. Volumen direccional(7 features): flujo comprador/vendedor semanal
"""

import numpy as np
import pandas as pd
import psycopg2.extras

from src.data.database import query_df, get_connection


# ─────────────────────────────────────────────────────────────
# Carga de datos base (precios_semanales + indicadores_tecnicos_1w)
# ─────────────────────────────────────────────────────────────

def cargar_datos_base_1w() -> pd.DataFrame:
    """
    Carga OHLCV semanal + atr14 + vol_relativo desde Railway.
    Solo semanas con indicadores 1W calculados.
    """
    df = query_df("""
        SELECT
            p.ticker,
            p.fecha_semana  AS fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14,
            i.vol_relativo
        FROM precios_semanales p
        JOIN indicadores_tecnicos_1w i
            ON p.ticker = i.ticker AND p.fecha_semana = i.fecha
        WHERE p.close  > 0
          AND p.high   > 0
          AND p.low    > 0
          AND p.open   > 0
          AND p.volume > 0
        ORDER BY p.ticker, p.fecha_semana
    """)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"    Datos base 1W cargados: {len(df):,} filas ({df['ticker'].nunique()} tickers)")
    return df


# ─────────────────────────────────────────────────────────────
# Calculo de features por ticker (logica identica al 1D)
# ─────────────────────────────────────────────────────────────

def _calcular_grupo_1w(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las 32 features para un unico ticker sobre barras semanales."""
    df = df.sort_values("fecha").reset_index(drop=True).copy()

    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    close  = df["close"].astype(float)
    open_  = df["open"].astype(float)
    vol    = df["volume"].astype(float)
    atr14  = df["atr14"].astype(float)

    rango = (high - low).replace(0.0, np.nan)

    # ── Grupo 1: Anatomia de vela (semanal) ──────────────────
    df["body_pct"]         = (close - open_) / open_.replace(0, np.nan) * 100
    df["body_ratio"]       = (close - open_).abs() / rango
    df["body_ratio"]       = df["body_ratio"].clip(0.0, 1.0)
    df["upper_shadow_pct"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / rango
    df["lower_shadow_pct"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / rango
    df["upper_shadow_pct"] = df["upper_shadow_pct"].clip(0.0, 1.0)
    df["lower_shadow_pct"] = df["lower_shadow_pct"].clip(0.0, 1.0)
    df["es_alcista"]       = (close > open_).astype(int)
    df["gap_apertura_pct"] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100
    df["rango_diario_pct"] = (high - low) / close.replace(0, np.nan) * 100  # rango semanal en 1W
    df["rango_rel_atr"]    = (high - low) / atr14.replace(0, np.nan)
    df["clv"]              = ((close - low) - (high - close)) / rango
    df["clv"]              = df["clv"].clip(-1.0, 1.0)

    # ── Grupo 2: Patrones clasicos ────────────────────────────
    body_ratio = df["body_ratio"]
    upper_s    = df["upper_shadow_pct"]
    lower_s    = df["lower_shadow_pct"]
    es_alc     = df["es_alcista"]
    body_abs   = (close - open_).abs()
    prev_body  = body_abs.shift(1)
    prev_alc   = es_alc.shift(1)

    df["patron_doji"]           = (body_ratio < 0.05).astype(int)
    df["patron_hammer"]         = ((lower_s > 0.60) & (body_ratio < 0.30) & (es_alc == 1)).astype(int)
    df["patron_shooting_star"]  = ((upper_s > 0.60) & (body_ratio < 0.30) & (es_alc == 0)).astype(int)
    df["patron_marubozu"]       = (body_ratio > 0.85).astype(int)
    df["patron_engulfing_bull"] = (
        (body_abs > prev_body) & (es_alc == 1) & (prev_alc == 0)
    ).fillna(False).astype(int)
    df["patron_engulfing_bear"] = (
        (body_abs > prev_body) & (es_alc == 0) & (prev_alc == 1)
    ).fillna(False).astype(int)
    df["inside_bar"]  = ((high < high.shift(1)) & (low > low.shift(1))).fillna(False).astype(int)
    df["outside_bar"] = ((high > high.shift(1)) & (low < low.shift(1))).fillna(False).astype(int)

    # ── Grupo 3: Estructura rolling N-semanas ────────────────
    body_pct  = df["body_pct"]
    rango_pct = df["rango_diario_pct"]

    df["body_pct_ma5"]       = body_pct.rolling(5,  min_periods=3).mean()
    df["velas_alcistas_5d"]  = es_alc.rolling(5,  min_periods=5).sum()   # 5 semanas
    df["velas_alcistas_10d"] = es_alc.rolling(10, min_periods=10).sum()  # 10 semanas

    rango_ma10 = rango_pct.rolling(10, min_periods=5).mean()
    df["rango_expansion"] = (rango_pct > 1.5 * rango_ma10).astype(int)

    max_high_20  = high.rolling(20, min_periods=10).max()
    min_low_20   = low.rolling(20,  min_periods=10).min()
    max_close_20 = close.rolling(20, min_periods=10).max()
    min_close_20 = close.rolling(20, min_periods=10).min()
    rng_close_20 = (max_close_20 - min_close_20).replace(0, np.nan)

    df["dist_max_20d"]  = (close - max_high_20) / max_high_20.replace(0, np.nan) * 100
    df["dist_min_20d"]  = (close - min_low_20)  / min_low_20.replace(0, np.nan)  * 100
    df["pos_rango_20d"] = ((close - min_close_20) / rng_close_20).clip(0.0, 1.0)
    df["tendencia_velas"] = (df["velas_alcistas_5d"].fillna(0) * 2 - 5).astype(int)

    # ── Grupo 4: Volumen direccional ──────────────────────────
    vol_ma5  = vol.rolling(5,  min_periods=3).mean().replace(0, np.nan)
    vol_ma20 = vol.rolling(20, min_periods=10).mean().replace(0, np.nan)

    df["vol_ratio_5d"] = vol / vol_ma5

    vol_rel = df["vol_relativo"].where(df["vol_relativo"].notna(), vol / vol_ma20)
    df["vol_spike"] = (vol_rel > 2.0).fillna(False).astype(int)

    up_vol       = (vol * es_alc).rolling(5, min_periods=3).sum()
    total_vol_5d = vol.rolling(5, min_periods=3).sum().replace(0, np.nan)
    df["up_vol_5d"] = (up_vol / total_vol_5d).clip(0.0, 1.0)

    df["ad_flow"]     = df["clv"] * vol
    ad_flow_20        = df["ad_flow"].rolling(20, min_periods=10).sum()
    vol_sum_20        = vol.rolling(20, min_periods=10).sum().replace(0, np.nan)
    df["chaikin_mf_20"] = (ad_flow_20 / vol_sum_20).clip(-1.0, 1.0)

    body_abs_pct = body_pct.abs()
    mean_body_20 = body_abs_pct.rolling(20, min_periods=10).mean()
    df["vol_price_confirm"] = (
        (df["vol_spike"] == 1) & (body_abs_pct > mean_body_20)
    ).fillna(False).astype(int)
    df["vol_price_diverge"] = (
        (df["vol_spike"] == 1) & (body_ratio < 0.10)
    ).fillna(False).astype(int)

    return df


def calcular_features_precio_accion_1w(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el calculo de features por ticker via groupby."""
    resultado = df.groupby("ticker", group_keys=False).apply(_calcular_grupo_1w)
    print(f"    Features 1W calculadas: {len(resultado):,} filas")
    return resultado


# ─────────────────────────────────────────────────────────────
# Columnas (identicas al 1D)
# ─────────────────────────────────────────────────────────────

_ALL_FEAT_COLS = [
    # Grupo 1 (9)
    "body_pct", "body_ratio",
    "upper_shadow_pct", "lower_shadow_pct",
    "es_alcista", "gap_apertura_pct",
    "rango_diario_pct", "rango_rel_atr", "clv",
    # Grupo 2 (8)
    "patron_doji", "patron_hammer", "patron_shooting_star", "patron_marubozu",
    "patron_engulfing_bull", "patron_engulfing_bear",
    "inside_bar", "outside_bar",
    # Grupo 3 (8)
    "body_pct_ma5",
    "velas_alcistas_5d", "velas_alcistas_10d",
    "rango_expansion",
    "dist_max_20d", "dist_min_20d", "pos_rango_20d",
    "tendencia_velas",
    # Grupo 4 (7)
    "vol_ratio_5d", "vol_spike", "up_vol_5d",
    "ad_flow", "chaikin_mf_20",
    "vol_price_confirm", "vol_price_diverge",
]  # 32 features

_INT_COLS = {
    "es_alcista",
    "patron_doji", "patron_hammer", "patron_shooting_star", "patron_marubozu",
    "patron_engulfing_bull", "patron_engulfing_bear", "inside_bar", "outside_bar",
    "velas_alcistas_5d", "velas_alcistas_10d",
    "rango_expansion", "tendencia_velas",
    "vol_spike", "vol_price_confirm", "vol_price_diverge",
}


# ─────────────────────────────────────────────────────────────
# Persistencia -> features_precio_accion_1w
# ─────────────────────────────────────────────────────────────

def upsert_features_pa_1w(df: pd.DataFrame):
    """Persiste todas las features en features_precio_accion_1w (upsert)."""
    save_cols = ["ticker", "fecha"] + _ALL_FEAT_COLS
    df_save = df[save_cols].copy()
    df_save["fecha"] = pd.to_datetime(df_save["fecha"]).dt.date

    records = df_save.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if v is None:
                continue
            try:
                if pd.isna(v):
                    rec[k] = None
                    continue
            except (TypeError, ValueError):
                pass
            if k in _INT_COLS:
                rec[k] = int(v) if v is not None else None
            elif isinstance(v, (bool, np.bool_)):
                rec[k] = int(v)
            elif isinstance(v, np.integer):
                rec[k] = int(v)
            elif isinstance(v, np.floating):
                rec[k] = float(v)

    all_cols     = ", ".join(save_cols)
    placeholders = ", ".join(f"%({c})s" for c in save_cols)
    updates      = ", ".join(f"{c} = EXCLUDED.{c}" for c in _ALL_FEAT_COLS)

    sql = f"""
        INSERT INTO features_precio_accion_1w ({all_cols})
        VALUES ({placeholders})
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            {updates}
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"    Upsert completado: {len(records):,} filas en features_precio_accion_1w.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def procesar_features_precio_accion_1w():
    """Pipeline completo: carga datos semanales, calcula 32 features y persiste."""
    df_base = cargar_datos_base_1w()
    print(f"    Calculando 32 features de estructura precio/volumen (1W)...")
    df_feat = calcular_features_precio_accion_1w(df_base)
    upsert_features_pa_1w(df_feat)
    return df_feat
