"""
feature_store.py
Construye la tabla features_ml: join completo de todas las features
con retornos futuros y etiquetas para entrenamiento de modelos ML.

Columnas finales de features_ml:
    Identificación  : ticker, nombre, sector, fecha, segmento
    Precio/Volumen  : close, vol_relativo
    Indicadores     : rsi14, macd_hist, dist_sma21/50/200, adx, atr14,
                      momentum, bb_upper, bb_middle, bb_lower
    Scoring         : score_ponderado, condiciones_ok, cond_rsi/macd/sma21/
                      sma50/sma200/momentum
    Sector (Z-Score): z_rsi_sector, z_retorno_1d_sector, z_retorno_5d_sector,
                      z_vol_sector, z_dist_sma50_sector, z_adx_sector,
                      pct_long_sector, rank_retorno_sector,
                      rsi_sector_avg, adx_sector_avg, retorno_1d_sector_avg
    Targets         : retorno_1d, retorno_5d, retorno_10d, retorno_20d
    Labels          : label (GANANCIA/PERDIDA/NEUTRO), label_binario (1/0)

Estrategia de etiquetado:
    - retorno_20d > +UMBRAL_NEUTRO  → GANANCIA → label_binario = 1
    - retorno_20d < -UMBRAL_NEUTRO  → PERDIDA  → label_binario = 0
    - else                          → NEUTRO   → label_binario = 0
    UMBRAL_NEUTRO = 1.0% (consistente con backtesting)

Split temporal (igual al backtesting, por ticker):
    TRAIN    : primeros 70% de fechas
    TEST     : siguientes 15%
    BACKTEST : últimos 15%
"""

import pandas as pd
import numpy as np
import psycopg2.extras

from src.data.database import query_df, get_connection
from src.utils.config import TRAIN_RATIO, TEST_RATIO, UMBRAL_NEUTRO


# ─────────────────────────────────────────────────────────────
# Carga de datos base
# ─────────────────────────────────────────────────────────────

def cargar_datos_base() -> pd.DataFrame:
    """
    Une precios + indicadores + scoring + features_sector para todos los tickers.
    Retorna un DataFrame ordenado por (ticker, fecha).
    """
    sql = """
        SELECT
            p.ticker,
            a.nombre,
            a.sector,
            p.fecha,

            -- Precio base
            p.close,

            -- Indicadores técnicos
            i.rsi14,
            i.macd_hist,
            i.dist_sma21,
            i.dist_sma50,
            i.dist_sma200,
            i.adx,
            i.atr14,
            i.vol_relativo,
            i.momentum,
            i.bb_upper,
            i.bb_middle,
            i.bb_lower,

            -- Scoring rule-based
            s.score_ponderado,
            s.condiciones_ok,
            s.cond_rsi,
            s.cond_macd,
            s.cond_sma21,
            s.cond_sma50,
            s.cond_sma200,
            s.cond_momentum,

            -- Features sectoriales
            f.z_rsi_sector,
            f.z_retorno_1d_sector,
            f.z_retorno_5d_sector,
            f.z_vol_sector,
            f.z_dist_sma50_sector,
            f.z_adx_sector,
            f.pct_long_sector,
            f.rank_retorno_sector,
            f.rsi_sector_avg,
            f.adx_sector_avg,
            f.retorno_1d_sector_avg

        FROM precios_diarios      p
        JOIN activos              a ON p.ticker = a.ticker
        JOIN indicadores_tecnicos i ON p.ticker = i.ticker AND p.fecha = i.fecha
        JOIN scoring_tecnico      s ON p.ticker = s.ticker AND p.fecha = s.fecha
        JOIN features_sector      f ON p.ticker = f.ticker AND p.fecha = f.fecha
        ORDER BY p.ticker, p.fecha
    """
    df = query_df(sql)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"    Base cargada: {len(df):,} registros, {df['ticker'].nunique()} tickers")
    return df


# ─────────────────────────────────────────────────────────────
# Retornos futuros (targets)
# ─────────────────────────────────────────────────────────────

def agregar_retornos_futuros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los retornos futuros a 1, 5, 10 y 20 días para cada (ticker, fecha).

    Retorno N días = (close[t+N] / close[t] - 1) * 100  [en %]

    Los últimos N registros de cada ticker tendrán NaN para retorno_Nd.
    Esas filas NO recibirán label (se conservan para predicción en producción).
    """
    df = df.copy().sort_values(["ticker", "fecha"]).reset_index(drop=True)

    for n in [1, 5, 10, 20]:
        future_close = df.groupby("ticker")["close"].shift(-n)
        df[f"retorno_{n}d"] = ((future_close / df["close"]) - 1) * 100
        df[f"retorno_{n}d"] = df[f"retorno_{n}d"].round(4)

    return df


# ─────────────────────────────────────────────────────────────
# Segmento temporal (TRAIN / TEST / BACKTEST)
# ─────────────────────────────────────────────────────────────

def agregar_segmento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna TRAIN / TEST / BACKTEST a cada (ticker, fecha) usando
    la misma partición temporal que el backtesting (70/15/15).

    La división es por posición (índice ordenado por fecha), NO aleatoria.
    """
    df = df.copy()
    segmentos = []

    for ticker, grupo in df.groupby("ticker"):
        n = len(grupo)
        i_train = int(n * TRAIN_RATIO)
        i_test  = int(n * (TRAIN_RATIO + TEST_RATIO))

        segs = ["TRAIN"] * i_train + ["TEST"] * (i_test - i_train) + \
               ["BACKTEST"] * (n - i_test)
        segmentos.extend(segs)

    df["segmento"] = segmentos
    return df


# ─────────────────────────────────────────────────────────────
# Labels (variable objetivo)
# ─────────────────────────────────────────────────────────────

def agregar_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las variables objetivo basadas en retorno_20d:

        label          : GANANCIA / PERDIDA / NEUTRO
        label_binario  : 1 (GANANCIA) / 0 (NEUTRO o PERDIDA)

    Umbral: UMBRAL_NEUTRO * 100 en % (ej: 0.01 → 1.0%)
    """
    df = df.copy()

    umbral_pct = UMBRAL_NEUTRO * 100  # convertir a %

    conditions = [
        df["retorno_20d"] >  umbral_pct,
        df["retorno_20d"] < -umbral_pct,
    ]
    choices = ["GANANCIA", "PERDIDA"]

    df["label"] = np.select(conditions, choices, default="NEUTRO")

    # NaN en retorno_20d → sin etiqueta (últimas filas sin futuro conocido)
    df.loc[df["retorno_20d"].isna(), "label"] = None

    df["label_binario"] = (df["label"] == "GANANCIA").astype("Int64")
    df.loc[df["label"].isna(), "label_binario"] = None

    return df


# ─────────────────────────────────────────────────────────────
# Persistencia en PostgreSQL
# ─────────────────────────────────────────────────────────────

def upsert_features_ml(df: pd.DataFrame):
    """Inserta o actualiza todas las features en features_ml."""

    # Convertir booleanos numpy a Python bool
    bool_cols = ["cond_rsi", "cond_macd", "cond_sma21",
                 "cond_sma50", "cond_sma200", "cond_momentum"]
    df = df.copy()
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) else None)

    # Convertir fecha a date
    df["fecha"] = df["fecha"].dt.date

    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        # NaN / pd.NA / numpy NaN → None para PostgreSQL
        rec = {
            k: (None if (
                v is pd.NA or
                (isinstance(v, float) and np.isnan(v)) or
                v is None
            ) else v)
            for k, v in rec.items()
        }
        # Convertir numpy int / Int64 a Python int
        for col in ["condiciones_ok", "rank_retorno_sector", "label_binario"]:
            if rec.get(col) is not None:
                try:
                    rec[col] = int(rec[col])
                except (ValueError, TypeError):
                    rec[col] = None
        records.append(rec)

    sql = """
        INSERT INTO features_ml (
            ticker, nombre, sector, fecha, segmento,
            close, vol_relativo,
            rsi14, macd_hist, dist_sma21, dist_sma50, dist_sma200,
            adx, atr14, momentum, bb_upper, bb_middle, bb_lower,
            score_ponderado, condiciones_ok,
            cond_rsi, cond_macd, cond_sma21, cond_sma50, cond_sma200, cond_momentum,
            z_rsi_sector, z_retorno_1d_sector, z_retorno_5d_sector,
            z_vol_sector, z_dist_sma50_sector, z_adx_sector,
            pct_long_sector, rank_retorno_sector,
            rsi_sector_avg, adx_sector_avg, retorno_1d_sector_avg,
            retorno_1d, retorno_5d, retorno_10d, retorno_20d,
            label, label_binario
        ) VALUES (
            %(ticker)s, %(nombre)s, %(sector)s, %(fecha)s, %(segmento)s,
            %(close)s, %(vol_relativo)s,
            %(rsi14)s, %(macd_hist)s, %(dist_sma21)s, %(dist_sma50)s, %(dist_sma200)s,
            %(adx)s, %(atr14)s, %(momentum)s, %(bb_upper)s, %(bb_middle)s, %(bb_lower)s,
            %(score_ponderado)s, %(condiciones_ok)s,
            %(cond_rsi)s, %(cond_macd)s, %(cond_sma21)s, %(cond_sma50)s,
            %(cond_sma200)s, %(cond_momentum)s,
            %(z_rsi_sector)s, %(z_retorno_1d_sector)s, %(z_retorno_5d_sector)s,
            %(z_vol_sector)s, %(z_dist_sma50_sector)s, %(z_adx_sector)s,
            %(pct_long_sector)s, %(rank_retorno_sector)s,
            %(rsi_sector_avg)s, %(adx_sector_avg)s, %(retorno_1d_sector_avg)s,
            %(retorno_1d)s, %(retorno_5d)s, %(retorno_10d)s, %(retorno_20d)s,
            %(label)s, %(label_binario)s
        )
        ON CONFLICT (ticker, fecha) DO UPDATE SET
            segmento               = EXCLUDED.segmento,
            close                  = EXCLUDED.close,
            vol_relativo           = EXCLUDED.vol_relativo,
            rsi14                  = EXCLUDED.rsi14,
            macd_hist              = EXCLUDED.macd_hist,
            dist_sma21             = EXCLUDED.dist_sma21,
            dist_sma50             = EXCLUDED.dist_sma50,
            dist_sma200            = EXCLUDED.dist_sma200,
            adx                    = EXCLUDED.adx,
            atr14                  = EXCLUDED.atr14,
            momentum               = EXCLUDED.momentum,
            bb_upper               = EXCLUDED.bb_upper,
            bb_middle              = EXCLUDED.bb_middle,
            bb_lower               = EXCLUDED.bb_lower,
            score_ponderado        = EXCLUDED.score_ponderado,
            condiciones_ok         = EXCLUDED.condiciones_ok,
            cond_rsi               = EXCLUDED.cond_rsi,
            cond_macd              = EXCLUDED.cond_macd,
            cond_sma21             = EXCLUDED.cond_sma21,
            cond_sma50             = EXCLUDED.cond_sma50,
            cond_sma200            = EXCLUDED.cond_sma200,
            cond_momentum          = EXCLUDED.cond_momentum,
            z_rsi_sector           = EXCLUDED.z_rsi_sector,
            z_retorno_1d_sector    = EXCLUDED.z_retorno_1d_sector,
            z_retorno_5d_sector    = EXCLUDED.z_retorno_5d_sector,
            z_vol_sector           = EXCLUDED.z_vol_sector,
            z_dist_sma50_sector    = EXCLUDED.z_dist_sma50_sector,
            z_adx_sector           = EXCLUDED.z_adx_sector,
            pct_long_sector        = EXCLUDED.pct_long_sector,
            rank_retorno_sector    = EXCLUDED.rank_retorno_sector,
            rsi_sector_avg         = EXCLUDED.rsi_sector_avg,
            adx_sector_avg         = EXCLUDED.adx_sector_avg,
            retorno_1d_sector_avg  = EXCLUDED.retorno_1d_sector_avg,
            retorno_1d             = EXCLUDED.retorno_1d,
            retorno_5d             = EXCLUDED.retorno_5d,
            retorno_10d            = EXCLUDED.retorno_10d,
            retorno_20d            = EXCLUDED.retorno_20d,
            label                  = EXCLUDED.label,
            label_binario          = EXCLUDED.label_binario
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"  features_ml upsert: {len(records):,} registros.")


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def construir_feature_store(guardar_db: bool = True) -> pd.DataFrame:
    """
    Pipeline completo:
        1. Carga todos los datos (precios + indicadores + scoring + sector)
        2. Añade retornos futuros (1d, 5d, 10d, 20d)
        3. Asigna segmento temporal (TRAIN / TEST / BACKTEST)
        4. Crea labels desde retorno_20d
        5. Persiste en features_ml

    Returns:
        DataFrame completo con todas las features y labels
    """
    print("  [1/4] Cargando datos base (join 4 tablas)...")
    df = cargar_datos_base()

    print("  [2/4] Calculando retornos futuros (1d, 5d, 10d, 20d)...")
    df = agregar_retornos_futuros(df)
    sin_futuro = df["retorno_20d"].isna().sum()
    print(f"        Filas sin retorno_20d (últimas N por ticker): {sin_futuro:,}")

    print("  [3/4] Asignando segmentos temporales (70/15/15)...")
    df = agregar_segmento(df)
    for seg in ["TRAIN", "TEST", "BACKTEST"]:
        n = (df["segmento"] == seg).sum()
        print(f"        {seg:<9}: {n:>7,} filas")

    print("  [4/4] Creando labels desde retorno_20d...")
    df = agregar_labels(df)

    # Resumen de distribución de labels (solo donde hay label)
    df_con_label = df[df["label"].notna()]
    dist = df_con_label["label"].value_counts()
    total_labeled = len(df_con_label)
    print(f"        Total filas con label : {total_labeled:,}")
    for lbl, cnt in dist.items():
        print(f"          {lbl:<10}: {cnt:>6,}  ({cnt/total_labeled*100:.1f}%)")

    if guardar_db:
        print("\n  Persistiendo en features_ml...")
        upsert_features_ml(df)

    return df


# ─────────────────────────────────────────────────────────────
# Acceso rápido para entrenamiento ML
# ─────────────────────────────────────────────────────────────

def cargar_features_entrenamiento(segmento: str = "TRAIN",
                                  solo_con_label: bool = True) -> pd.DataFrame:
    """
    Carga las features del segmento indicado desde features_ml.
    Útil para alimentar directamente los modelos ML.

    Args:
        segmento        : 'TRAIN' | 'TEST' | 'BACKTEST'
        solo_con_label  : si True, retorna solo filas donde label IS NOT NULL

    Returns:
        DataFrame listo para sklearn / XGBoost / LightGBM
    """
    where_label = "AND label IS NOT NULL" if solo_con_label else ""
    sql = f"""
        SELECT *
        FROM features_ml
        WHERE segmento = :segmento
        {where_label}
        ORDER BY ticker, fecha
    """
    df = query_df(sql, params={"segmento": segmento})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df
