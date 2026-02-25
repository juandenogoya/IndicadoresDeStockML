"""
market_structure.py
Calcula 24 features de estructura de mercado (swings, HH/HL/LH/LL, BOS/CHoCH)
para el Feature Store V3.

Fuente : precios_diarios (OHLCV)
Output : tabla features_market_structure

Para cada ventana N en [5, 10] (12 features x 2 = 24 total):
    is_sh_N, is_sl_N       -- pivot detectado en la barra (0/1)
    estructura_N           -- tendencia -1 (LH+LL) / 0 / +1 (HH+HL)
    dist_sh_N_pct          -- distancia del close al ultimo swing high (%)
    dist_sl_N_pct          -- distancia del close al ultimo swing low (%)
    dias_sh_N, dias_sl_N   -- barras transcurridas desde el ultimo swing
    impulso_N_pct          -- amplitud del ultimo swing |SH - SL| / SL * 100
    bos_bull_N, bos_bear_N -- break of structure (confirma tendencia)
    choch_bull_N, choch_bear_N -- change of character (posible reversion)

Nota sobre deteccion de pivots:
    Se usa rolling(2*N+1, center=True), es decir ventana simetrica que
    incluye N barras futuras.  Correcto para training historico; las
    N ultimas barras de cada ticker quedaran sin confirmacion (NaN).
"""

import numpy as np
import pandas as pd
import psycopg2.extras

from src.data.database import query_df, get_connection, ejecutar_sql


# ─────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS features_market_structure (
    ticker VARCHAR(10) NOT NULL,
    fecha  DATE        NOT NULL,

    -- Ventana N=5  (tactico, ~11 barras)
    is_sh_5        SMALLINT,       -- 1 si la barra es swing high con N=5
    is_sl_5        SMALLINT,       -- 1 si la barra es swing low con N=5
    estructura_5   SMALLINT,       -- -1 bajista / 0 neutral / +1 alcista
    dist_sh_5_pct  NUMERIC(10,4),  -- (close - last_sh) / last_sh * 100
    dist_sl_5_pct  NUMERIC(10,4),  -- (close - last_sl) / last_sl * 100
    dias_sh_5      SMALLINT,       -- barras desde ultimo swing high
    dias_sl_5      SMALLINT,       -- barras desde ultimo swing low
    impulso_5_pct  NUMERIC(10,4),  -- |last_sh - last_sl| / last_sl * 100
    bos_bull_5     SMALLINT,       -- 1 si break of structure alcista
    bos_bear_5     SMALLINT,       -- 1 si break of structure bajista
    choch_bull_5   SMALLINT,       -- 1 si cambio de caracter alcista
    choch_bear_5   SMALLINT,       -- 1 si cambio de caracter bajista

    -- Ventana N=10 (estrategico, ~21 barras, alineado con retorno_20d)
    is_sh_10       SMALLINT,
    is_sl_10       SMALLINT,
    estructura_10  SMALLINT,
    dist_sh_10_pct NUMERIC(10,4),
    dist_sl_10_pct NUMERIC(10,4),
    dias_sh_10     SMALLINT,
    dias_sl_10     SMALLINT,
    impulso_10_pct NUMERIC(10,4),
    bos_bull_10    SMALLINT,
    bos_bear_10    SMALLINT,
    choch_bull_10  SMALLINT,
    choch_bear_10  SMALLINT,

    PRIMARY KEY (ticker, fecha)
);
CREATE INDEX IF NOT EXISTS idx_fms_ticker_fecha
    ON features_market_structure (ticker, fecha);
"""


def crear_tabla():
    ejecutar_sql(_DDL)


# ─────────────────────────────────────────────────────────────
# Carga de datos base
# ─────────────────────────────────────────────────────────────

def cargar_datos_base() -> pd.DataFrame:
    """Carga OHLCV desde precios_diarios (solo barras con precios validos)."""
    df = query_df("""
        SELECT ticker, fecha, open, high, low, close, volume
        FROM precios_diarios
        WHERE close > 0
          AND high  > 0
          AND low   > 0
          AND open  > 0
        ORDER BY ticker, fecha
    """)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"    Datos base cargados: {len(df):,} filas ({df['ticker'].nunique()} tickers)")
    return df


# ─────────────────────────────────────────────────────────────
# Helpers de deteccion de pivots
# ─────────────────────────────────────────────────────────────

def _secuencia_pivots_lookup(
    prices: np.ndarray,
    is_pivot: np.ndarray,
):
    """
    Para cada barra t retorna los datos del ultimo y penultimo pivot
    vistos hasta t (inclusive).

    Parametros
    ----------
    prices   : array de precios (high para SH, low para SL)
    is_pivot : array booleano True donde se confirma el pivot

    Retorna
    -------
    last_price  : precio del ultimo pivot hasta t
    prev_price  : precio del penultimo pivot hasta t
    last_bar    : indice (posicion) del ultimo pivot
    """
    n = len(prices)
    pivot_idx = np.where(is_pivot)[0]

    nan_arr = np.full(n, np.nan)
    if len(pivot_idx) == 0:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()

    pivot_prices = prices[pivot_idx]
    # cuantos pivots confirmados hasta t (inclusive)
    cumcount = np.cumsum(is_pivot.astype(int))

    last_price = np.full(n, np.nan)
    prev_price = np.full(n, np.nan)
    last_bar   = np.full(n, np.nan)

    mask1 = cumcount >= 1
    idx1  = cumcount[mask1] - 1
    last_price[mask1] = pivot_prices[idx1]
    last_bar[mask1]   = pivot_idx[idx1].astype(float)

    mask2 = cumcount >= 2
    idx2  = cumcount[mask2] - 2
    prev_price[mask2] = pivot_prices[idx2]

    return last_price, prev_price, last_bar


# ─────────────────────────────────────────────────────────────
# Calculo de features para una ventana N
# ─────────────────────────────────────────────────────────────

def _calcular_estructura_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Agrega las 12 features de market structure para la ventana N al DataFrame.
    Trabaja sobre un DataFrame de un solo ticker, ordenado por fecha.
    """
    h   = df["high"].astype(float).values
    lo  = df["low"].astype(float).values
    c   = df["close"].astype(float).values
    N   = len(h)
    pos = np.arange(N, dtype=float)

    # ── Deteccion de pivots via ventana simetrica ─────────────────────
    h_s = pd.Series(h)
    l_s = pd.Series(lo)

    roll_max = h_s.rolling(2 * n + 1, center=True, min_periods=n + 1).max()
    roll_min = l_s.rolling(2 * n + 1, center=True, min_periods=n + 1).min()

    is_sh = (h_s == roll_max).values   # array bool
    is_sl = (l_s == roll_min).values   # array bool

    df[f"is_sh_{n}"] = is_sh.astype(int)
    df[f"is_sl_{n}"] = is_sl.astype(int)

    # ── Secuencias de pivots ──────────────────────────────────────────
    last_sh, prev_sh, last_sh_bar = _secuencia_pivots_lookup(h, is_sh)
    last_sl, prev_sl, last_sl_bar = _secuencia_pivots_lookup(lo, is_sl)

    sh_cumcount = np.cumsum(is_sh.astype(int))
    sl_cumcount = np.cumsum(is_sl.astype(int))
    has_2sh     = sh_cumcount >= 2
    has_2sl     = sl_cumcount >= 2

    # ── Estructura: HH+HL = +1, LH+LL = -1, else = 0 ─────────────────
    # np.where evalua x e y en todo el array -> comparar con NaN da False, OK
    hh = np.where(has_2sh & ~np.isnan(prev_sh), last_sh > prev_sh, False)
    hl = np.where(has_2sl & ~np.isnan(prev_sl), last_sl > prev_sl, False)
    lh = np.where(has_2sh & ~np.isnan(prev_sh), last_sh < prev_sh, False)
    ll = np.where(has_2sl & ~np.isnan(prev_sl), last_sl < prev_sl, False)

    estructura = np.where(
        hh & hl, 1,
        np.where(lh & ll, -1, 0)
    ).astype(int)
    # Solo valido cuando tenemos al menos 2 pivots de cada tipo
    estructura = np.where(has_2sh & has_2sl, estructura, 0)
    df[f"estructura_{n}"] = estructura

    # ── Distancias al ultimo swing ────────────────────────────────────
    sh_safe = np.where(np.isnan(last_sh) | (last_sh == 0), np.nan, last_sh)
    sl_safe = np.where(np.isnan(last_sl) | (last_sl == 0), np.nan, last_sl)
    c_safe  = np.where(c == 0, np.nan, c)

    df[f"dist_sh_{n}_pct"] = (c_safe - sh_safe) / sh_safe * 100
    df[f"dist_sl_{n}_pct"] = (c_safe - sl_safe) / sl_safe * 100

    # ── Barras desde ultimo swing ─────────────────────────────────────
    # Resultado float con NaN para posiciones sin pivot aun
    dias_sh = np.where(~np.isnan(last_sh_bar), np.clip(pos - last_sh_bar, 0, 252), np.nan)
    dias_sl = np.where(~np.isnan(last_sl_bar), np.clip(pos - last_sl_bar, 0, 252), np.nan)
    df[f"dias_sh_{n}"] = dias_sh
    df[f"dias_sl_{n}"] = dias_sl

    # ── Impulso: amplitud del swing actual ────────────────────────────
    impulso = np.where(
        ~np.isnan(last_sh) & ~np.isnan(last_sl) & (sl_safe > 0),
        np.abs(last_sh - last_sl) / sl_safe * 100,
        np.nan
    )
    df[f"impulso_{n}_pct"] = impulso

    # ── BOS y CHoCH ──────────────────────────────────────────────────
    # Usando el ultimo swing del bar anterior como nivel de referencia
    c_s          = pd.Series(c)
    last_sh_prev = pd.Series(last_sh).shift(1)
    last_sl_prev = pd.Series(last_sl).shift(1)
    est_prev     = pd.Series(estructura).shift(1).fillna(0)

    # Cruce por encima del ultimo swing high
    cross_above = (c_s > last_sh_prev) & (c_s.shift(1) <= last_sh_prev)
    # Cruce por debajo del ultimo swing low
    cross_below = (c_s < last_sl_prev) & (c_s.shift(1) >= last_sl_prev)

    # BOS: cruce que CONFIRMA la estructura existente
    bos_bull = (cross_above & (est_prev >= 0)).fillna(False).astype(int)
    bos_bear = (cross_below & (est_prev <= 0)).fillna(False).astype(int)

    # CHoCH: cruce que VA CONTRA la estructura (posible reversion)
    choch_bull = (cross_above & (est_prev < 0)).fillna(False).astype(int)
    choch_bear = (cross_below & (est_prev > 0)).fillna(False).astype(int)

    df[f"bos_bull_{n}"]   = bos_bull.values
    df[f"bos_bear_{n}"]   = bos_bear.values
    df[f"choch_bull_{n}"] = choch_bull.values
    df[f"choch_bear_{n}"] = choch_bear.values

    return df


# ─────────────────────────────────────────────────────────────
# Pipeline por ticker
# ─────────────────────────────────────────────────────────────

def _calcular_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las 24 features de market structure para un unico ticker."""
    df = df.sort_values("fecha").reset_index(drop=True).copy()
    for n in [5, 10]:
        df = _calcular_estructura_n(df, n)
    return df


def calcular_features_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el calculo de features por ticker via groupby."""
    resultado = df.groupby("ticker", group_keys=False).apply(_calcular_ticker)
    print(f"    Features calculadas: {len(resultado):,} filas")
    return resultado


# ─────────────────────────────────────────────────────────────
# Lista de features para el modelo V3
# ─────────────────────────────────────────────────────────────

FEATURE_COLS_MS: list = []
for _n in [5, 10]:
    FEATURE_COLS_MS.extend([
        f"is_sh_{_n}", f"is_sl_{_n}", f"estructura_{_n}",
        f"dist_sh_{_n}_pct", f"dist_sl_{_n}_pct",
        f"dias_sh_{_n}", f"dias_sl_{_n}", f"impulso_{_n}_pct",
        f"bos_bull_{_n}", f"bos_bear_{_n}",
        f"choch_bull_{_n}", f"choch_bear_{_n}",
    ])
# 24 features total


# ─────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────

_INT_COLS_MS = {
    "is_sh_5", "is_sl_5", "estructura_5",
    "dias_sh_5", "dias_sl_5",
    "bos_bull_5", "bos_bear_5", "choch_bull_5", "choch_bear_5",
    "is_sh_10", "is_sl_10", "estructura_10",
    "dias_sh_10", "dias_sl_10",
    "bos_bull_10", "bos_bear_10", "choch_bull_10", "choch_bear_10",
}


def upsert_features_ms(df: pd.DataFrame):
    """Persiste las features de market structure (upsert)."""
    save_cols = ["ticker", "fecha"] + FEATURE_COLS_MS
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
            if k in _INT_COLS_MS:
                rec[k] = int(v) if v is not None else None
            elif isinstance(v, (bool, np.bool_)):
                rec[k] = int(v)
            elif isinstance(v, np.integer):
                rec[k] = int(v)
            elif isinstance(v, np.floating):
                rec[k] = float(v)

    all_cols     = ", ".join(save_cols)
    placeholders = ", ".join(f"%({c})s" for c in save_cols)
    updates      = ", ".join(f"{c} = EXCLUDED.{c}" for c in FEATURE_COLS_MS)

    sql = f"""
        INSERT INTO features_market_structure ({all_cols})
        VALUES ({placeholders})
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            {updates}
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"    Upsert completado: {len(records):,} filas en features_market_structure.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def procesar_features_market_structure():
    """Pipeline completo: crea tabla, carga, calcula y persiste."""
    crear_tabla()
    df_base = cargar_datos_base()
    print(f"    Calculando 24 features de market structure (N=5, N=10)...")
    df_feat = calcular_features_market_structure(df_base)
    upsert_features_ms(df_feat)
    return df_feat
