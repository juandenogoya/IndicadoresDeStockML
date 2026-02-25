"""
metrics.py
Cálculo de métricas de performance para cada combinación E×S×Ticker×Segmento.

Métricas calculadas:
    win_rate               : % de operaciones ganadoras
    retorno_promedio_pct   : retorno medio por operación
    retorno_total_pct      : suma total de retornos
    max_drawdown_pct       : máxima caída acumulada desde el pico
    profit_factor          : suma ganancias / suma pérdidas absolutas
    dias_promedio_posicion : días promedio en posición
    ganancias / perdidas / neutros: conteos
"""

import pandas as pd
import numpy as np
from src.data.database import ejecutar_sql, get_connection, query_df
import psycopg2.extras


# ─────────────────────────────────────────────────────────────
# Cálculo de métricas para un grupo de operaciones
# ─────────────────────────────────────────────────────────────

def calcular_metricas(df_ops: pd.DataFrame) -> dict:
    """
    Calcula métricas de performance para un conjunto de operaciones.

    Args:
        df_ops: DataFrame con operaciones (retorno_pct, resultado, dias_posicion)

    Returns:
        dict con todas las métricas
    """
    if df_ops.empty:
        return _metricas_vacias()

    total   = len(df_ops)
    gan     = (df_ops["resultado"] == "GANANCIA").sum()
    per     = (df_ops["resultado"] == "PERDIDA").sum()
    neu     = (df_ops["resultado"] == "NEUTRO").sum()

    win_rate        = round(gan / total, 4) if total > 0 else 0.0
    ret_promedio    = round(df_ops["retorno_pct"].mean(), 4)
    ret_total       = round(df_ops["retorno_pct"].sum(), 4)
    dias_promedio   = round(df_ops["dias_posicion"].mean(), 2)

    # Profit Factor: suma de ganancias / |suma de pérdidas|
    # Se capa en 99.9999 para evitar infinito en PostgreSQL NUMERIC
    ganancias_sum = df_ops[df_ops["retorno_pct"] > 0]["retorno_pct"].sum()
    perdidas_sum  = abs(df_ops[df_ops["retorno_pct"] < 0]["retorno_pct"].sum())
    if perdidas_sum > 0:
        profit_factor = round(min(ganancias_sum / perdidas_sum, 99.9999), 4)
    elif ganancias_sum > 0:
        profit_factor = 99.9999   # sin pérdidas = máximo representable
    else:
        profit_factor = 0.0

    # Max Drawdown sobre la curva de retornos acumulados
    retornos_acum  = df_ops["retorno_pct"].cumsum()
    pico           = retornos_acum.cummax()
    drawdown       = retornos_acum - pico
    max_drawdown   = round(drawdown.min(), 4)

    return {
        "total_operaciones":      int(total),
        "ganancias":              int(gan),
        "perdidas":               int(per),
        "neutros":                int(neu),
        "win_rate":               win_rate,
        "retorno_promedio_pct":   ret_promedio,
        "retorno_total_pct":      ret_total,
        "max_drawdown_pct":       max_drawdown,
        "profit_factor":          profit_factor,
        "dias_promedio_posicion": dias_promedio,
    }


def _metricas_vacias() -> dict:
    return {
        "total_operaciones": 0, "ganancias": 0, "perdidas": 0, "neutros": 0,
        "win_rate": 0.0, "retorno_promedio_pct": 0.0, "retorno_total_pct": 0.0,
        "max_drawdown_pct": 0.0, "profit_factor": 0.0, "dias_promedio_posicion": 0.0,
    }


# ─────────────────────────────────────────────────────────────
# Cálculo y persistencia del resumen global
# ─────────────────────────────────────────────────────────────

def calcular_y_guardar_resultados(df_ops: pd.DataFrame):
    """
    Agrega las operaciones por combinación E×S×Ticker×Segmento,
    calcula métricas y persiste en resultados_backtest.
    """
    if df_ops.empty:
        print("  [WARN] Sin operaciones para calcular métricas.")
        return

    registros = []

    # ── Por ticker y estrategia ───────────────────────────────
    grupos = df_ops.groupby(
        ["estrategia_entrada", "estrategia_salida", "ticker", "segmento"]
    )

    for (ee, es, ticker, seg), grupo in grupos:
        m = calcular_metricas(grupo)
        registros.append({
            "estrategia_entrada":     ee,
            "estrategia_salida":      es,
            "ticker":                 ticker,
            "segmento":               seg,
            **m,
        })

    # ── Global por estrategia (ticker = NULL) ─────────────────
    grupos_global = df_ops.groupby(
        ["estrategia_entrada", "estrategia_salida", "segmento"]
    )

    for (ee, es, seg), grupo in grupos_global:
        m = calcular_metricas(grupo)
        registros.append({
            "estrategia_entrada":     ee,
            "estrategia_salida":      es,
            "ticker":                 None,
            "segmento":               seg,
            **m,
        })

    # ── Persistir ─────────────────────────────────────────────
    sql = """
        INSERT INTO resultados_backtest
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

    print(f"  Resultados persistidos: {len(registros):,} filas en resultados_backtest.")


# ─────────────────────────────────────────────────────────────
# Ranking de estrategias (lectura desde DB)
# ─────────────────────────────────────────────────────────────

def ranking_estrategias(segmento: str = "TRAIN",
                        min_ops: int = 10) -> pd.DataFrame:
    """
    Retorna el ranking de las 16 combinaciones E×S ordenado por
    profit_factor y win_rate para un segmento dado.
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
            ROUND(dias_promedio_posicion, 1)   AS dias_prom
        FROM resultados_backtest
        WHERE segmento   = :segmento
          AND ticker      IS NULL
          AND total_operaciones >= :min_ops
        ORDER BY profit_factor DESC, win_rate DESC
    """
    return query_df(sql, params={"segmento": segmento, "min_ops": min_ops})


def ranking_por_ticker(estrategia_entrada: str, estrategia_salida: str,
                       segmento: str = "TRAIN") -> pd.DataFrame:
    """
    Retorna el ranking de tickers para una combinación E×S dada.
    """
    sql = """
        SELECT
            ticker,
            total_operaciones,
            ROUND(win_rate * 100, 1)          AS win_rate_pct,
            ROUND(retorno_total_pct, 2)        AS ret_total,
            ROUND(profit_factor::NUMERIC, 2)   AS profit_factor,
            ROUND(max_drawdown_pct, 2)         AS max_dd
        FROM resultados_backtest
        WHERE estrategia_entrada = :ee
          AND estrategia_salida  = :es
          AND segmento           = :seg
          AND ticker IS NOT NULL
        ORDER BY profit_factor DESC
    """
    return query_df(sql, params={"ee": estrategia_entrada,
                                  "es": estrategia_salida,
                                  "seg": segmento})
