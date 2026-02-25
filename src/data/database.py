"""
database.py
Gestión de conexión a PostgreSQL y operaciones base.
"""

import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from src.utils.config import DB_CONFIG


# ── Engine SQLAlchemy (para pandas read/write) ────────────────
def get_engine():
    """Retorna un engine de SQLAlchemy para uso con pandas."""
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
    return create_engine(url)


# ── Conexión psycopg2 (para operaciones directas) ─────────────
@contextmanager
def get_connection():
    """Context manager que abre y cierra la conexión automáticamente."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ejecutar_sql(sql: str, params=None):
    """Ejecuta una sentencia SQL sin retorno de datos."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Ejecuta un SELECT y retorna un DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params)


def insertar_df(df: pd.DataFrame, tabla: str, if_exists: str = "append"):
    """
    Inserta un DataFrame en una tabla PostgreSQL.
    if_exists: 'append' (default) | 'replace' | 'fail'
    """
    engine = get_engine()
    df.to_sql(tabla, engine, if_exists=if_exists, index=False)


def upsert_precios(df: pd.DataFrame):
    """
    Inserta o actualiza precios diarios usando ON CONFLICT DO NOTHING.
    Evita duplicados por (ticker, fecha).
    """
    records = df.to_dict(orient="records")
    sql = """
        INSERT INTO precios_diarios
            (ticker, fecha, open, high, low, close, volume, adj_close)
        VALUES
            (%(ticker)s, %(fecha)s, %(open)s, %(high)s, %(low)s,
             %(close)s, %(volume)s, %(adj_close)s)
        ON CONFLICT (ticker, fecha) DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
    print(f"  Upsert completado: {len(records)} registros procesados.")


def upsert_indicadores(df: pd.DataFrame):
    """
    Inserta o actualiza indicadores técnicos.
    Evita duplicados por (ticker, fecha).
    """
    records = df.to_dict(orient="records")
    sql = """
        INSERT INTO indicadores_tecnicos
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
            sma21       = EXCLUDED.sma21,
            sma50       = EXCLUDED.sma50,
            sma200      = EXCLUDED.sma200,
            dist_sma21  = EXCLUDED.dist_sma21,
            dist_sma50  = EXCLUDED.dist_sma50,
            dist_sma200 = EXCLUDED.dist_sma200,
            rsi14       = EXCLUDED.rsi14,
            macd        = EXCLUDED.macd,
            macd_signal = EXCLUDED.macd_signal,
            macd_hist   = EXCLUDED.macd_hist,
            atr14       = EXCLUDED.atr14,
            bb_upper    = EXCLUDED.bb_upper,
            bb_middle   = EXCLUDED.bb_middle,
            bb_lower    = EXCLUDED.bb_lower,
            obv         = EXCLUDED.obv,
            vol_relativo= EXCLUDED.vol_relativo,
            adx         = EXCLUDED.adx,
            momentum    = EXCLUDED.momentum,
            updated_at  = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
    print(f"  Indicadores upsert: {len(records)} registros.")


def upsert_scoring(df: pd.DataFrame):
    """
    Inserta o actualiza el scoring técnico rule-based.
    Evita duplicados por (ticker, fecha).
    """
    records = df.to_dict(orient="records")

    # Convertir numpy bool a Python bool para psycopg2
    for r in records:
        for col in ["cond_rsi", "cond_macd", "cond_sma21",
                    "cond_sma50", "cond_sma200", "cond_momentum"]:
            r[col] = bool(r[col])

    sql = """
        INSERT INTO scoring_tecnico
            (ticker, fecha, cond_rsi, cond_macd, cond_sma21,
             cond_sma50, cond_sma200, cond_momentum,
             score_ponderado, condiciones_ok, senal)
        VALUES
            (%(ticker)s, %(fecha)s, %(cond_rsi)s, %(cond_macd)s, %(cond_sma21)s,
             %(cond_sma50)s, %(cond_sma200)s, %(cond_momentum)s,
             %(score_ponderado)s, %(condiciones_ok)s, %(senal)s)
        ON CONFLICT (ticker, fecha)
        DO UPDATE SET
            cond_rsi       = EXCLUDED.cond_rsi,
            cond_macd      = EXCLUDED.cond_macd,
            cond_sma21     = EXCLUDED.cond_sma21,
            cond_sma50     = EXCLUDED.cond_sma50,
            cond_sma200    = EXCLUDED.cond_sma200,
            cond_momentum  = EXCLUDED.cond_momentum,
            score_ponderado= EXCLUDED.score_ponderado,
            condiciones_ok = EXCLUDED.condiciones_ok,
            senal          = EXCLUDED.senal
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)
    print(f"  Scoring upsert: {len(records)} registros.")


def test_conexion():
    """Verifica que la conexión a la base de datos funciona."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
    print(f"Conexion OK: {version}")
    return True
