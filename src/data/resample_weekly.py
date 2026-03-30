"""
resample_weekly.py
Convierte precios OHLCV diarios a semanales (1W).

Logica de resample:
    - Anchor: viernes (W-FRI) — cada semana va de sabado a viernes
    - Open  = primer precio del primer dia habil de la semana
    - High  = maximo de todos los dias de la semana
    - Low   = minimo de todos los dias de la semana
    - Close = ultimo precio del ultimo dia habil de la semana
    - Volume = suma de volumen de todos los dias
    - fecha_semana = ULTIMO DIA HABIL real (no siempre viernes)
    - n_dias = cantidad de dias habiles en la semana (1-5)

Semana incompleta (semana en curso):
    - Excluida automaticamente para no tener datos parciales.
    - Se detecta comparando fecha_semana con el lunes de la semana actual.
"""

import pandas as pd
import psycopg2.extras
from datetime import date
from src.data.database import get_connection, query_df


# ─────────────────────────────────────────────────────────────
# Resample
# ─────────────────────────────────────────────────────────────

def resample_a_semanal(df_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un DataFrame OHLCV diario a semanal.

    Args:
        df_diario: DataFrame con columnas [fecha, open, high, low, close, volume, adj_close]
                   Puede tener columna 'ticker' (se ignora en el calculo).

    Returns:
        DataFrame semanal con columnas:
            [fecha_semana, open, high, low, close, volume, adj_close, n_dias]
        Ordenado por fecha_semana ASC.
        Semana en curso excluida.
    """
    df = df_diario.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Asegurar columna adj_close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # ── Agrupar por semana (anchor viernes) ─────────────────────
    # to_period('W-FRI') asigna cada fecha a su semana Mon-Fri
    df["_week"] = df["fecha"].dt.to_period("W-FRI")

    weekly = (
        df.groupby("_week", sort=True)
        .agg(
            fecha_semana=("fecha",    "last"),   # ultimo dia habil real
            open        =("open",     "first"),  # primer open de la semana
            high        =("high",     "max"),
            low         =("low",      "min"),
            close       =("close",    "last"),
            volume      =("volume",   "sum"),
            adj_close   =("adj_close","last"),
            n_dias      =("fecha",    "count"),
        )
        .reset_index(drop=True)
    )

    # ── Excluir semana en curso (incompleta) ─────────────────────
    hoy = pd.Timestamp.today().normalize()
    lunes_actual = hoy - pd.Timedelta(days=hoy.weekday())  # lunes de esta semana
    weekly = weekly[
        pd.to_datetime(weekly["fecha_semana"]) < lunes_actual
    ].copy()

    # ── Tipos ────────────────────────────────────────────────────
    weekly["fecha_semana"] = pd.to_datetime(weekly["fecha_semana"]).dt.date
    weekly["n_dias"]       = weekly["n_dias"].astype(int)
    weekly = weekly.sort_values("fecha_semana").reset_index(drop=True)

    return weekly


# ─────────────────────────────────────────────────────────────
# Cargar precios diarios de un ticker desde Railway
# ─────────────────────────────────────────────────────────────

def cargar_diarios_ticker(ticker: str) -> pd.DataFrame:
    """
    Carga todos los precios diarios de un ticker desde precios_diarios.

    Returns:
        DataFrame con columnas [fecha, open, high, low, close, volume, adj_close]
        Ordenado por fecha ASC.
    """
    df = query_df(
        """
        SELECT fecha, open, high, low, close, volume, adj_close
        FROM   precios_diarios
        WHERE  ticker = :ticker
        ORDER  BY fecha ASC
        """,
        params={"ticker": ticker},
    )
    return df


# ─────────────────────────────────────────────────────────────
# Upsert precios semanales
# ─────────────────────────────────────────────────────────────

def upsert_precios_semanales(df: pd.DataFrame, ticker: str):
    """
    Inserta o actualiza registros en precios_semanales para un ticker.

    Args:
        df:     DataFrame semanal (salida de resample_a_semanal).
        ticker: codigo del ticker.
    """
    if df.empty:
        return

    df = df.copy()
    df["ticker"] = ticker

    # Sanitizar NaN -> None para psycopg2
    df = df.where(pd.notnull(df), None)

    records = df[
        ["ticker", "fecha_semana", "open", "high", "low",
         "close", "volume", "adj_close", "n_dias"]
    ].to_dict(orient="records")

    sql = """
        INSERT INTO precios_semanales
            (ticker, fecha_semana, open, high, low, close, volume, adj_close, n_dias)
        VALUES
            (%(ticker)s, %(fecha_semana)s, %(open)s, %(high)s, %(low)s,
             %(close)s, %(volume)s, %(adj_close)s, %(n_dias)s)
        ON CONFLICT (ticker, fecha_semana) DO UPDATE SET
            open       = EXCLUDED.open,
            high       = EXCLUDED.high,
            low        = EXCLUDED.low,
            close      = EXCLUDED.close,
            volume     = EXCLUDED.volume,
            adj_close  = EXCLUDED.adj_close,
            n_dias     = EXCLUDED.n_dias,
            updated_at = NOW()
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)


# ─────────────────────────────────────────────────────────────
# Pipeline completo para un ticker
# ─────────────────────────────────────────────────────────────

def procesar_ticker_semanal(ticker: str) -> dict:
    """
    Carga diarios, resamplea a semanal y persiste en precios_semanales.

    Returns:
        dict con keys: ticker, n_semanas, fecha_primera, fecha_ultima, ok (bool), error (str)
    """
    resultado = {"ticker": ticker, "n_semanas": 0,
                 "fecha_primera": None, "fecha_ultima": None,
                 "ok": False, "error": None}
    try:
        df_diario = cargar_diarios_ticker(ticker)

        if df_diario.empty:
            raise ValueError("sin datos en precios_diarios")

        df_semanal = resample_a_semanal(df_diario)

        if df_semanal.empty:
            raise ValueError("resample no produjo datos")

        upsert_precios_semanales(df_semanal, ticker)

        resultado.update({
            "n_semanas":    len(df_semanal),
            "fecha_primera": df_semanal["fecha_semana"].iloc[0],
            "fecha_ultima":  df_semanal["fecha_semana"].iloc[-1],
            "ok":            True,
        })

    except Exception as e:
        resultado["error"] = str(e)

    return resultado
