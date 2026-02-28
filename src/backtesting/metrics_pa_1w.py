"""
metrics_pa_1w.py
Calculo de metricas de performance para el backtesting PA 1W challenger.

Lee de operaciones_bt_pa_1w, escribe en resultados_bt_pa_1w.
Logica identica a metrics_pa.py, adaptada a tablas 1W.

Nota: dias_promedio_posicion en este contexto = semanas_promedio_posicion.
"""

import pandas as pd
import numpy as np
from src.data.database import get_connection, query_df
from src.backtesting.metrics import calcular_metricas
import psycopg2.extras


def _native(v):
    """Convierte numpy scalars a tipos Python nativos para psycopg2."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _sanitizar(record: dict) -> dict:
    """Aplica _native a todos los valores de un dict."""
    return {k: _native(v) for k, v in record.items()}


# ─────────────────────────────────────────────────────────────
# Calculo y persistencia del resumen PA 1W
# ─────────────────────────────────────────────────────────────

def calcular_y_guardar_resultados_pa_1w(df_ops: pd.DataFrame):
    """
    Agrega operaciones PA 1W por combinacion EV x SV x Ticker x Segmento,
    calcula metricas y persiste en resultados_bt_pa_1w.

    Genera filas por ticker Y filas globales (ticker = NULL) para
    comparacion directa con el backtesting 1D.
    """
    if df_ops.empty:
        print("  [WARN] Sin operaciones PA 1W para calcular metricas.")
        return

    registros = []

    # ── Por ticker y estrategia ───────────────────────────────
    grupos = df_ops.groupby(
        ["estrategia_entrada", "estrategia_salida", "ticker", "segmento"]
    )
    for (ee, es, ticker, seg), grupo in grupos:
        m = calcular_metricas(grupo)
        registros.append(_sanitizar({
            "estrategia_entrada": ee,
            "estrategia_salida":  es,
            "ticker":             ticker,
            "segmento":           seg,
            **m,
        }))

    # ── Global por estrategia (ticker = NULL) ─────────────────
    grupos_global = df_ops.groupby(
        ["estrategia_entrada", "estrategia_salida", "segmento"]
    )
    for (ee, es, seg), grupo in grupos_global:
        m = calcular_metricas(grupo)
        registros.append(_sanitizar({
            "estrategia_entrada": ee,
            "estrategia_salida":  es,
            "ticker":             None,
            "segmento":           seg,
            **m,
        }))

    # ── Persistir ─────────────────────────────────────────────
    sql = """
        INSERT INTO resultados_bt_pa_1w
            (estrategia_entrada, estrategia_salida, ticker, segmento,
             total_operaciones, ganancias, perdidas, neutros,
             win_rate, retorno_promedio_pct, retorno_total_pct,
             max_drawdown_pct, profit_factor, dias_promedio_posicion)
        VALUES
            (%(estrategia_entrada)s, %(estrategia_salida)s, %(ticker)s, %(segmento)s,
             %(total_operaciones)s, %(ganancias)s, %(perdidas)s, %(neutros)s,
             %(win_rate)s, %(retorno_promedio_pct)s, %(retorno_total_pct)s,
             %(max_drawdown_pct)s, %(profit_factor)s, %(dias_promedio_posicion)s)
        ON CONFLICT (estrategia_entrada, estrategia_salida, ticker, segmento)
        DO UPDATE SET
            total_operaciones      = EXCLUDED.total_operaciones,
            ganancias              = EXCLUDED.ganancias,
            perdidas               = EXCLUDED.perdidas,
            neutros                = EXCLUDED.neutros,
            win_rate               = EXCLUDED.win_rate,
            retorno_promedio_pct   = EXCLUDED.retorno_promedio_pct,
            retorno_total_pct      = EXCLUDED.retorno_total_pct,
            max_drawdown_pct       = EXCLUDED.max_drawdown_pct,
            profit_factor          = EXCLUDED.profit_factor,
            dias_promedio_posicion = EXCLUDED.dias_promedio_posicion
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, registros, page_size=200)

    print(f"  Resultados PA 1W persistidos: {len(registros):,} filas en resultados_bt_pa_1w.")


# ─────────────────────────────────────────────────────────────
# Ranking de estrategias PA 1W (lectura desde DB)
# ─────────────────────────────────────────────────────────────

def ranking_estrategias_pa_1w(segmento: str = "FULL",
                               min_ops: int = 3) -> pd.DataFrame:
    """
    Retorna el ranking de las 16 combinaciones EV x SV 1W ordenado por
    profit_factor y win_rate para un segmento dado.

    Nota: dias_promedio_posicion = semanas promedio en posicion en contexto 1W.
    """
    sql = """
        SELECT
            estrategia_entrada, estrategia_salida,
            total_operaciones, ganancias, perdidas,
            ROUND(win_rate * 100, 1)          AS win_rate_pct,
            ROUND(retorno_promedio_pct, 2)     AS ret_promedio,
            ROUND(retorno_total_pct, 2)        AS ret_total,
            ROUND(max_drawdown_pct, 2)         AS max_dd,
            ROUND(profit_factor::NUMERIC, 2)   AS profit_factor,
            ROUND(dias_promedio_posicion, 1)   AS semanas_prom
        FROM resultados_bt_pa_1w
        WHERE segmento          = :segmento
          AND ticker            IS NULL
          AND total_operaciones >= :min_ops
        ORDER BY profit_factor DESC, win_rate DESC
    """
    return query_df(sql, params={"segmento": segmento, "min_ops": min_ops})


def ranking_por_ticker_pa_1w(estrategia_entrada: str, estrategia_salida: str,
                              segmento: str = "FULL") -> pd.DataFrame:
    """Retorna el ranking de tickers para una combinacion EV x SV 1W dada."""
    sql = """
        SELECT
            ticker,
            total_operaciones,
            ROUND(win_rate * 100, 1)          AS win_rate_pct,
            ROUND(retorno_total_pct, 2)        AS ret_total,
            ROUND(profit_factor::NUMERIC, 2)   AS profit_factor,
            ROUND(max_drawdown_pct, 2)         AS max_dd
        FROM resultados_bt_pa_1w
        WHERE estrategia_entrada = :ee
          AND estrategia_salida  = :es
          AND segmento           = :seg
          AND ticker IS NOT NULL
        ORDER BY profit_factor DESC
    """
    return query_df(sql, params={"ee": estrategia_entrada,
                                  "es": estrategia_salida,
                                  "seg": segmento})
