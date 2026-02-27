"""
simulator_pa.py
Motor de simulacion de operaciones Long para el backtesting PA challenger.

Usa features de precio/accion y estructura de mercado como senales de
entrada y salida (EV1-EV4 x SV1-SV4).

Carga datos de 5 tablas:
    precios_diarios + indicadores_tecnicos + features_precio_accion
    + features_market_structure + scoring_tecnico

Escribe en operaciones_bt_pa (sin tocar operaciones_backtest).

Reglas de ejecucion:
    - Una posicion abierta a la vez por ticker
    - Senal en dia T -> entrada al OPEN del dia T+1
    - SL/TP evaluados contra HIGH/LOW intraday
    - Cierre por condicion estructural o score: al CLOSE del dia
    - Posicion abierta al final del historico: cierre forzado al CLOSE

Periodo fijo: desde FECHA_INICIO_BT (2023-01-01) hasta hoy.
Las estrategias son reglas fijas (no ML), no hay data leakage.
Un solo segmento "FULL" — sin split TRAIN/TEST/BACKTEST.
"""

import pandas as pd
import numpy as np
from typing import List
from src.utils.config import ALL_TICKERS
from src.backtesting.strategies_pa import (
    check_entrada_pa,
    calcular_stops_iniciales_pa,
    check_salida_pa,
    clasificar_resultado_pa,
)
from src.data.database import query_df, get_connection


# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────

FECHA_INICIO_BT        = "2023-01-01"   # Fecha fija de inicio del backtesting
ESTRATEGIAS_ENTRADA_PA = ["EV1", "EV2", "EV3", "EV4"]
ESTRATEGIAS_SALIDA_PA  = ["SV1", "SV2", "SV3", "SV4"]


# ─────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────

def cargar_datos_ticker_pa(ticker: str) -> pd.DataFrame:
    """
    Carga datos combinados de 5 tablas para un ticker desde FECHA_INICIO_BT.

    LEFT JOIN para features_precio_accion, features_market_structure y scoring_tecnico:
    las primeras N barras no tienen estas features (periodo de calentamiento).
    """
    sql = """
        SELECT
            p.fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14,

            -- Features precio/accion (pueden ser NULL)
            pa.es_alcista,
            pa.patron_hammer,
            pa.patron_engulfing_bull,
            pa.vol_spike,
            pa.up_vol_5d,
            pa.vol_price_confirm,

            -- Features market structure (pueden ser NULL)
            ms.estructura_5,    ms.estructura_10,
            ms.bos_bull_5,      ms.bos_bear_5,
            ms.choch_bull_5,    ms.choch_bear_5,
            ms.bos_bull_10,     ms.bos_bear_10,
            ms.choch_bull_10,   ms.choch_bear_10,
            ms.dist_sh_5_pct,   ms.dist_sl_5_pct,
            ms.dias_sh_5,       ms.dias_sl_5,
            ms.dist_sh_10_pct,  ms.dist_sl_10_pct,
            ms.dias_sh_10,      ms.dias_sl_10,
            ms.impulso_5_pct,   ms.impulso_10_pct,

            -- Scoring rule-based (puede ser NULL, usado en SV3)
            s.score_ponderado

        FROM precios_diarios p
        JOIN  indicadores_tecnicos          i  ON p.ticker = i.ticker  AND p.fecha = i.fecha
        LEFT JOIN features_precio_accion   pa  ON p.ticker = pa.ticker AND p.fecha = pa.fecha
        LEFT JOIN features_market_structure ms  ON p.ticker = ms.ticker AND p.fecha = ms.fecha
        LEFT JOIN scoring_tecnico           s  ON p.ticker = s.ticker  AND p.fecha = s.fecha
        WHERE p.ticker = :ticker
          AND p.fecha  >= :fecha_inicio
        ORDER BY p.fecha ASC
    """
    df = query_df(sql, params={"ticker": ticker, "fecha_inicio": FECHA_INICIO_BT})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Simulador de un segmento
# ─────────────────────────────────────────────────────────────

def simular_segmento_pa(df: pd.DataFrame, ticker: str,
                         est_entrada: str, est_salida: str,
                         segmento: str) -> List[dict]:
    """
    Simula operaciones Long PA para un ticker, estrategia y segmento.

    Reglas:
        - Una posicion abierta a la vez
        - Senal en dia T -> entrada al OPEN de T+1
        - SL evaluado intraday (low <= stop_loss)
        - TP fijo evaluado intraday (SV4): high >= take_profit
        - Salida por condicion estructural/score: al CLOSE del dia
        - Timeout: al CLOSE del ultimo dia permitido
        - Fin de segmento con posicion abierta: cierre forzado al ultimo CLOSE

    Returns:
        Lista de dicts con el detalle de cada operacion cerrada
    """
    operaciones    = []
    n              = len(df)

    en_posicion    = False
    precio_entrada = None
    fecha_entrada  = None
    score_entrada  = None
    stop_loss      = None
    take_profit    = None
    dias_posicion  = 0

    for i in range(n):
        row = df.iloc[i]

        # ── FUERA DE POSICION: evaluar entrada ────────────────
        if not en_posicion:
            if check_entrada_pa(row, est_entrada):
                if i + 1 >= n:
                    break   # no hay dia siguiente

                next_row = df.iloc[i + 1]
                if pd.isna(next_row["open"]) or next_row["open"] <= 0:
                    continue

                atr_hoy = float(row["atr14"]) if not pd.isna(row["atr14"]) \
                          else float(row["close"]) * 0.02

                precio_entrada = float(next_row["open"])
                fecha_entrada  = next_row["fecha"]

                # score_entrada: score_ponderado disponible o None
                sp = row.get("score_ponderado")
                score_entrada = round(float(sp), 4) if (sp is not None and not pd.isna(sp)) else None

                stop_loss, take_profit = calcular_stops_iniciales_pa(
                    precio_entrada, atr_hoy, est_salida
                )
                dias_posicion = 0
                en_posicion   = True

            continue   # avanzar al siguiente dia

        # ── EN POSICION: gestion diaria ───────────────────────
        dias_posicion += 1

        cerrar, motivo, precio_salida = check_salida_pa(
            row, est_salida, stop_loss, take_profit, dias_posicion
        )

        if cerrar:
            retorno_pct = (precio_salida / precio_entrada - 1) * 100
            resultado   = clasificar_resultado_pa(retorno_pct)

            operaciones.append({
                "estrategia_entrada": est_entrada,
                "estrategia_salida":  est_salida,
                "ticker":             ticker,
                "segmento":           segmento,
                "fecha_entrada":      fecha_entrada.date() if hasattr(fecha_entrada, "date") else fecha_entrada,
                "precio_entrada":     round(precio_entrada, 4),
                "score_entrada":      score_entrada,
                "fecha_salida":       row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"],
                "precio_salida":      precio_salida,
                "motivo_salida":      motivo,
                "dias_posicion":      dias_posicion,
                "retorno_pct":        round(retorno_pct, 4),
                "resultado":          resultado,
            })

            # Resetear estado
            en_posicion    = False
            precio_entrada = None
            stop_loss      = None
            take_profit    = None

    # ── Posicion abierta al final del segmento: cierre forzado
    if en_posicion and precio_entrada is not None:
        precio_salida = float(df.iloc[-1]["close"])
        retorno_pct   = (precio_salida / precio_entrada - 1) * 100
        operaciones.append({
            "estrategia_entrada": est_entrada,
            "estrategia_salida":  est_salida,
            "ticker":             ticker,
            "segmento":           segmento,
            "fecha_entrada":      fecha_entrada.date() if hasattr(fecha_entrada, "date") else fecha_entrada,
            "precio_entrada":     round(precio_entrada, 4),
            "score_entrada":      score_entrada,
            "fecha_salida":       df.iloc[-1]["fecha"].date(),
            "precio_salida":      round(precio_salida, 4),
            "motivo_salida":      "FIN_SEGMENTO",
            "dias_posicion":      dias_posicion,
            "retorno_pct":        round(retorno_pct, 4),
            "resultado":          clasificar_resultado_pa(retorno_pct),
        })

    return operaciones


# ─────────────────────────────────────────────────────────────
# Backtesting completo: todos los tickers x todas las combis
# ─────────────────────────────────────────────────────────────

def ejecutar_backtesting_pa(tickers: list = None,
                             guardar_db: bool = True) -> pd.DataFrame:
    """
    Ejecuta la matriz 4x4 PA para todos los tickers.

    Usa el periodo completo desde FECHA_INICIO_BT hasta hoy,
    sin split TRAIN/TEST/BACKTEST (las estrategias son reglas fijas, sin entrenamiento).
    Todas las operaciones se etiquetan con segmento = "FULL".

    Args:
        tickers:    lista de tickers (None = ALL_TICKERS)
        guardar_db: persistir operaciones en PostgreSQL

    Returns:
        DataFrame con todas las operaciones simuladas
    """
    tickers = tickers or ALL_TICKERS

    todas_ops    = []
    total_combis = len(tickers) * len(ESTRATEGIAS_ENTRADA_PA) * len(ESTRATEGIAS_SALIDA_PA)

    print(f"\n  Periodo: {FECHA_INICIO_BT} -> hoy  |  Segmento: FULL")
    print(f"  {total_combis} combinaciones ({len(tickers)} tickers x 4 entradas x 4 salidas)\n")
    print("-" * 65)

    for ticker in tickers:
        print(f"  [{ticker}] Cargando datos...", end=" ")
        df_full = cargar_datos_ticker_pa(ticker)

        if len(df_full) < 50:
            print("insuficientes datos, omitido.")
            continue

        print(f"OK | {len(df_full)} barras ({df_full['fecha'].min().date()} a {df_full['fecha'].max().date()})")

        for ee in ESTRATEGIAS_ENTRADA_PA:
            for es in ESTRATEGIAS_SALIDA_PA:
                ops = simular_segmento_pa(df_full, ticker, ee, es, "FULL")
                todas_ops.extend(ops)

    print("-" * 65)
    print(f"  Simulacion PA completada: {len(todas_ops):,} operaciones totales.")

    df_ops = pd.DataFrame(todas_ops) if todas_ops else pd.DataFrame()

    if guardar_db and not df_ops.empty:
        _persistir_operaciones_pa(df_ops)

    return df_ops


def _persistir_operaciones_pa(df_ops: pd.DataFrame):
    """
    Persiste operaciones en operaciones_bt_pa.
    TRUNCATE previo para garantizar reproducibilidad (re-run limpio).
    """
    import psycopg2.extras
    from src.data.database import ejecutar_sql

    ejecutar_sql("TRUNCATE TABLE operaciones_bt_pa RESTART IDENTITY")
    ejecutar_sql("TRUNCATE TABLE resultados_bt_pa  RESTART IDENTITY")

    records = df_ops.to_dict(orient="records")

    sql = """
        INSERT INTO operaciones_bt_pa
            (estrategia_entrada, estrategia_salida, ticker, segmento,
             fecha_entrada, precio_entrada, score_entrada,
             fecha_salida, precio_salida, motivo_salida,
             dias_posicion, retorno_pct, resultado)
        VALUES
            (%(estrategia_entrada)s, %(estrategia_salida)s, %(ticker)s, %(segmento)s,
             %(fecha_entrada)s, %(precio_entrada)s, %(score_entrada)s,
             %(fecha_salida)s, %(precio_salida)s, %(motivo_salida)s,
             %(dias_posicion)s, %(retorno_pct)s, %(resultado)s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"  Operaciones PA persistidas: {len(records):,} registros.")
