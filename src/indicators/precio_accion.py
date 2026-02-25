"""
precio_accion.py
Calcula 32 features de estructura de precio y volumen para el Feature Store V2.

Fuente: precios_diarios JOIN indicadores_tecnicos (para atr14, vol_relativo)
Output: tabla features_precio_accion

Grupos:
    1. Anatomia de vela  (9 features) : estructura OHLC intradía
    2. Patrones clasicos (8 features) : señales de reversión/continuación
    3. Estructura rolling(8 features) : contexto N-dias de tendencia y rango
    4. Volumen direccional(7 features): flujo comprador/vendedor
"""

import numpy as np
import pandas as pd
import psycopg2.extras

from src.data.database import query_df, get_connection, ejecutar_sql


# ─────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS features_precio_accion (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,

    -- Grupo 1: Anatomia de vela (9)
    body_pct              NUMERIC(10,4),   -- retorno intradía signed %
    body_ratio            NUMERIC(6,4),    -- cuerpo / rango total (0-1)
    upper_shadow_pct      NUMERIC(6,4),    -- sombra superior / rango (0-1)
    lower_shadow_pct      NUMERIC(6,4),    -- sombra inferior / rango (0-1)
    es_alcista            SMALLINT,        -- 1 si close > open
    gap_apertura_pct      NUMERIC(10,4),   -- (open - prev_close)/prev_close %
    rango_diario_pct      NUMERIC(10,4),   -- (high-low)/close %
    rango_rel_atr         NUMERIC(10,4),   -- (high-low)/atr14
    clv                   NUMERIC(7,4),    -- Close Location Value (-1 a +1)

    -- Grupo 2: Patrones clasicos (8)
    patron_doji              SMALLINT,     -- body_ratio < 0.05
    patron_hammer            SMALLINT,     -- lower_shadow > 60%, body < 30%, alcista
    patron_shooting_star     SMALLINT,     -- upper_shadow > 60%, body < 30%, bajista
    patron_marubozu          SMALLINT,     -- body_ratio > 0.85 (vela de fuerza)
    patron_engulfing_bull    SMALLINT,     -- envolvente alcista (2 velas)
    patron_engulfing_bear    SMALLINT,     -- envolvente bajista (2 velas)
    inside_bar               SMALLINT,     -- rango dentro del bar previo
    outside_bar              SMALLINT,     -- rango fuera del bar previo

    -- Grupo 3: Estructura rolling (8)
    body_pct_ma5          NUMERIC(10,4),  -- media de body_pct 5d
    velas_alcistas_5d     SMALLINT,       -- count alcistas en 5d (0-5)
    velas_alcistas_10d    SMALLINT,       -- count alcistas en 10d (0-10)
    rango_expansion       SMALLINT,       -- rango > 1.5x media 10d
    dist_max_20d          NUMERIC(10,4),  -- (close - max_high_20d)/max_high_20d %
    dist_min_20d          NUMERIC(10,4),  -- (close - min_low_20d)/min_low_20d %
    pos_rango_20d         NUMERIC(6,4),   -- posicion en rango 20d (0=min, 1=max)
    tendencia_velas       SMALLINT,       -- balance alcistas 5d (-5 a +5)

    -- Grupo 4: Volumen direccional (7)
    vol_ratio_5d          NUMERIC(10,4),  -- volume / media_5d
    vol_spike             SMALLINT,       -- vol_relativo > 2.0
    up_vol_5d             NUMERIC(6,4),   -- % vol alcista en 5d (0-1)
    ad_flow               NUMERIC(20,4),  -- CLV * volume (flujo Chaikin)
    chaikin_mf_20         NUMERIC(7,4),   -- Chaikin Money Flow 20d (-1 a +1)
    vol_price_confirm     SMALLINT,       -- vol_spike AND mov > media_body
    vol_price_diverge     SMALLINT,       -- vol_spike AND body pequeño

    PRIMARY KEY (ticker, fecha)
);
CREATE INDEX IF NOT EXISTS idx_fpa_ticker_fecha
    ON features_precio_accion (ticker, fecha);
"""


def crear_tabla():
    ejecutar_sql(_DDL)


# ─────────────────────────────────────────────────────────────
# Carga de datos base
# ─────────────────────────────────────────────────────────────

def cargar_datos_base() -> pd.DataFrame:
    """
    Carga OHLCV + atr14 + vol_relativo desde la DB.
    Solo fechas con indicadores técnicos calculados (warmup >= 14d).
    """
    df = query_df("""
        SELECT
            p.ticker, p.fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14,
            i.vol_relativo
        FROM precios_diarios p
        JOIN indicadores_tecnicos i
            ON p.ticker = i.ticker AND p.fecha = i.fecha
        WHERE p.close  > 0
          AND p.high   > 0
          AND p.low    > 0
          AND p.open   > 0
          AND p.volume > 0
        ORDER BY p.ticker, p.fecha
    """)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"    Datos base cargados: {len(df):,} filas ({df['ticker'].nunique()} tickers)")
    return df


# ─────────────────────────────────────────────────────────────
# Cálculo de features por ticker
# ─────────────────────────────────────────────────────────────

def _calcular_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las 32 features para un único ticker (vía groupby.apply)."""
    df = df.sort_values("fecha").reset_index(drop=True).copy()

    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    close  = df["close"].astype(float)
    open_  = df["open"].astype(float)
    vol    = df["volume"].astype(float)
    atr14  = df["atr14"].astype(float)

    # Rango con guard para evitar division por cero
    rango = (high - low).replace(0.0, np.nan)

    # ── Grupo 1: Anatomia de vela ─────────────────────────────
    df["body_pct"]         = (close - open_) / open_.replace(0, np.nan) * 100
    df["body_ratio"]       = (close - open_).abs() / rango
    df["body_ratio"]       = df["body_ratio"].clip(0.0, 1.0)
    df["upper_shadow_pct"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / rango
    df["lower_shadow_pct"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / rango
    df["upper_shadow_pct"] = df["upper_shadow_pct"].clip(0.0, 1.0)
    df["lower_shadow_pct"] = df["lower_shadow_pct"].clip(0.0, 1.0)
    df["es_alcista"]       = (close > open_).astype(int)
    df["gap_apertura_pct"] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100
    df["rango_diario_pct"] = (high - low) / close.replace(0, np.nan) * 100
    df["rango_rel_atr"]    = (high - low) / atr14.replace(0, np.nan)
    df["clv"]              = ((close - low) - (high - close)) / rango
    df["clv"]              = df["clv"].clip(-1.0, 1.0)

    # ── Grupo 2: Patrones clásicos ────────────────────────────
    body_ratio = df["body_ratio"]
    upper_s    = df["upper_shadow_pct"]
    lower_s    = df["lower_shadow_pct"]
    es_alc     = df["es_alcista"]
    body_abs   = (close - open_).abs()
    prev_body  = body_abs.shift(1)
    prev_alc   = es_alc.shift(1)

    df["patron_doji"]          = (body_ratio < 0.05).astype(int)
    df["patron_hammer"]        = ((lower_s > 0.60) & (body_ratio < 0.30) & (es_alc == 1)).astype(int)
    df["patron_shooting_star"] = ((upper_s > 0.60) & (body_ratio < 0.30) & (es_alc == 0)).astype(int)
    df["patron_marubozu"]      = (body_ratio > 0.85).astype(int)
    df["patron_engulfing_bull"] = (
        (body_abs > prev_body) & (es_alc == 1) & (prev_alc == 0)
    ).fillna(False).astype(int)
    df["patron_engulfing_bear"] = (
        (body_abs > prev_body) & (es_alc == 0) & (prev_alc == 1)
    ).fillna(False).astype(int)
    df["inside_bar"]  = ((high < high.shift(1)) & (low > low.shift(1))).fillna(False).astype(int)
    df["outside_bar"] = ((high > high.shift(1)) & (low < low.shift(1))).fillna(False).astype(int)

    # ── Grupo 3: Estructura rolling N-dias ───────────────────
    body_pct   = df["body_pct"]
    rango_pct  = df["rango_diario_pct"]

    df["body_pct_ma5"]       = body_pct.rolling(5,  min_periods=3).mean()
    df["velas_alcistas_5d"]  = es_alc.rolling(5,  min_periods=5).sum()
    df["velas_alcistas_10d"] = es_alc.rolling(10, min_periods=10).sum()

    rango_ma10 = rango_pct.rolling(10, min_periods=5).mean()
    df["rango_expansion"]    = (rango_pct > 1.5 * rango_ma10).astype(int)

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

    # vol_spike: usa vol_relativo ya calculado si está disponible
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


def calcular_features_precio_accion(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el cálculo de features por ticker via groupby."""
    resultado = df.groupby("ticker", group_keys=False).apply(_calcular_grupo)
    print(f"    Features calculadas: {len(resultado):,} filas")
    return resultado


# ─────────────────────────────────────────────────────────────
# Lista de features de precio/acción que van al modelo V2
# ─────────────────────────────────────────────────────────────

FEATURE_COLS_PA = [
    # Grupo 1: Anatomia de vela (9)
    "body_pct", "body_ratio",
    "upper_shadow_pct", "lower_shadow_pct",
    "es_alcista", "gap_apertura_pct",
    "rango_diario_pct", "rango_rel_atr", "clv",
    # Grupo 2: Patrones (8)
    "patron_doji", "patron_hammer", "patron_shooting_star", "patron_marubozu",
    "patron_engulfing_bull", "patron_engulfing_bear",
    "inside_bar", "outside_bar",
    # Grupo 3: Rolling (7 — sin tendencia_velas que es redundante con velas_alcistas_5d)
    "body_pct_ma5",
    "velas_alcistas_5d", "velas_alcistas_10d",
    "rango_expansion",
    "dist_max_20d", "dist_min_20d", "pos_rango_20d",
    # Grupo 4: Volumen (6 — sin ad_flow que es intermedio; se usa chaikin_mf_20)
    "vol_ratio_5d", "vol_spike", "up_vol_5d",
    "chaikin_mf_20", "vol_price_confirm", "vol_price_diverge",
]  # 30 features


# ─────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────

_ALL_FEAT_COLS = FEATURE_COLS_PA + ["ad_flow", "tendencia_velas"]  # tabla tiene 32

_INT_COLS = {
    "es_alcista",
    "patron_doji", "patron_hammer", "patron_shooting_star", "patron_marubozu",
    "patron_engulfing_bull", "patron_engulfing_bear", "inside_bar", "outside_bar",
    "velas_alcistas_5d", "velas_alcistas_10d",
    "rango_expansion", "tendencia_velas",
    "vol_spike", "vol_price_confirm", "vol_price_diverge",
}


def upsert_features_pa(df: pd.DataFrame):
    """Persiste todas las features en features_precio_accion (upsert)."""
    save_cols = ["ticker", "fecha"] + _ALL_FEAT_COLS
    df_save = df[save_cols].copy()
    df_save["fecha"] = pd.to_datetime(df_save["fecha"]).dt.date

    # Convertir a tipos Python nativos
    records = df_save.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if v is None:
                continue
            # Pandas NA / numpy nan -> None
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
        INSERT INTO features_precio_accion ({all_cols})
        VALUES ({placeholders})
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            {updates}
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"    Upsert completado: {len(records):,} filas en features_precio_accion.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def procesar_features_precio_accion():
    """Pipeline completo: crea tabla, carga datos, calcula y persiste."""
    crear_tabla()
    df_base = cargar_datos_base()
    print(f"    Calculando 32 features de estructura precio/volumen...")
    df_feat = calcular_features_precio_accion(df_base)
    upsert_features_pa(df_feat)
    return df_feat
