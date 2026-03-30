"""
simulator_pa_1w.py
Motor de simulacion de operaciones Long para el backtesting PA 1W challenger.

Espejo de simulator_pa.py adaptado para barras semanales (1W).

Carga datos de 4 tablas semanales:
    precios_semanales + indicadores_tecnicos_1w
    + features_precio_accion_1w + features_market_structure_1w

No incluye scoring_tecnico (no existe version 1W):
    SV3 solo evalua estructura_10==-1 (score_ponderado=None -> senal_score=False)

Escribe en operaciones_bt_pa_1w (sin tocar operaciones_bt_pa).

Reglas de ejecucion (identicas al 1D, en semanas):
    - Una posicion abierta a la vez por ticker
    - Senal en semana T -> entrada al OPEN de la semana T+1
    - SL/TP evaluados contra HIGH/LOW intraweek
    - Cierre por condicion estructural: al CLOSE de la semana
    - Posicion abierta al final del historico: cierre forzado al CLOSE
    - Timeout = 20 semanas (~5 meses)

Periodo fijo: desde FECHA_INICIO_BT_1W (2023-01-01) hasta hoy.
Un solo segmento "FULL" -- sin split TRAIN/TEST/BACKTEST.
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

FECHA_INICIO_BT_1W     = "2023-01-01"   # suficiente historico semanal desde 2020
ESTRATEGIAS_ENTRADA_PA = ["EV1", "EV2", "EV3", "EV4"]
ESTRATEGIAS_SALIDA_PA  = ["SV1", "SV2", "SV3", "SV4"]


# ─────────────────────────────────────────────────────────────
# Carga de datos semanales
# ─────────────────────────────────────────────────────────────

def cargar_datos_ticker_pa_1w(ticker: str) -> pd.DataFrame:
    """
    Carga datos semanales combinados de 4 tablas para un ticker desde FECHA_INICIO_BT_1W.

    LEFT JOIN para features_precio_accion_1w y features_market_structure_1w:
    las primeras N semanas no tienen estas features (periodo de calentamiento).

    Nota: NO incluye scoring_tecnico. SV3 funcionara solo con estructura_10
    (score_ponderado ausente -> senal_score=False -> solo dispara si estructura_10==-1).
    """
    sql = """
        SELECT
            p.fecha_semana  AS fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14,

            -- Features precio/accion 1W (pueden ser NULL)
            pa.es_alcista,
            pa.patron_hammer,
            pa.patron_engulfing_bull,
            pa.vol_spike,
            pa.up_vol_5d,
            pa.vol_price_confirm,

            -- Features market structure 1W (pueden ser NULL)
            ms.estructura_5,    ms.estructura_10,
            ms.bos_bull_5,      ms.bos_bear_5,
            ms.choch_bull_5,    ms.choch_bear_5,
            ms.bos_bull_10,     ms.bos_bear_10,
            ms.choch_bull_10,   ms.choch_bear_10,
            ms.dist_sh_5_pct,   ms.dist_sl_5_pct,
            ms.dias_sh_5,       ms.dias_sl_5,
            ms.dist_sh_10_pct,  ms.dist_sl_10_pct,
            ms.dias_sh_10,      ms.dias_sl_10,
            ms.impulso_5_pct,   ms.impulso_10_pct

        FROM precios_semanales p
        JOIN  indicadores_tecnicos_1w          i  ON p.ticker = i.ticker  AND p.fecha_semana = i.fecha
        LEFT JOIN features_precio_accion_1w   pa  ON p.ticker = pa.ticker AND p.fecha_semana = pa.fecha
        LEFT JOIN features_market_structure_1w ms  ON p.ticker = ms.ticker AND p.fecha_semana = ms.fecha
        WHERE p.ticker       = :ticker
          AND p.fecha_semana >= :fecha_inicio
        ORDER BY p.fecha_semana ASC
    """
    df = query_df(sql, params={"ticker": ticker, "fecha_inicio": FECHA_INICIO_BT_1W})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Simulador de un segmento (logica identica al 1D, en semanas)
# ─────────────────────────────────────────────────────────────

def simular_segmento_pa_1w(df: pd.DataFrame, ticker: str,
                             est_entrada: str, est_salida: str,
                             segmento: str) -> List[dict]:
    """
    Simula operaciones Long PA 1W para un ticker, estrategia y segmento.

    Reglas (identicas a simular_segmento_pa, en semanas):
        - Una posicion abierta a la vez
        - Senal en semana T -> entrada al OPEN de T+1
        - SL evaluado intraweek (low <= stop_loss)
        - TP fijo intraweek (SV4): high >= take_profit
        - Salida por condicion estructural: al CLOSE de la semana
        - Timeout: 20 semanas
        - Fin de segmento con posicion abierta: cierre forzado al ultimo CLOSE

    Returns:
        Lista de dicts con el detalle de cada operacion cerrada
    """
    operaciones    = []
    n              = len(df)

    en_posicion    = False
    precio_entrada = None
    fecha_entrada  = None
    stop_loss      = None
    take_profit    = None
    dias_posicion  = 0   # en realidad: semanas_posicion

    for i in range(n):
        row = df.iloc[i]

        # ── FUERA DE POSICION: evaluar entrada ────────────────
        if not en_posicion:
            if check_entrada_pa(row, est_entrada):
                if i + 1 >= n:
                    break   # no hay semana siguiente

                next_row = df.iloc[i + 1]
                if pd.isna(next_row["open"]) or next_row["open"] <= 0:
                    continue

                atr_hoy = float(row["atr14"]) if not pd.isna(row["atr14"]) \
                          else float(row["close"]) * 0.02

                precio_entrada = float(next_row["open"])
                fecha_entrada  = next_row["fecha"]

                # score_entrada: no existe en 1W (sin scoring_tecnico semanal)
                stop_loss, take_profit = calcular_stops_iniciales_pa(
                    precio_entrada, atr_hoy, est_salida
                )
                dias_posicion = 0
                en_posicion   = True

            continue   # avanzar a la siguiente semana

        # ── EN POSICION: gestion semanal ──────────────────────
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
                "score_entrada":      None,   # sin scoring semanal
                "fecha_salida":       row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"],
                "precio_salida":      precio_salida,
                "motivo_salida":      motivo,
                "dias_posicion":      dias_posicion,
                "retorno_pct":        round(retorno_pct, 4),
                "resultado":          resultado,
                "stop_loss":          round(stop_loss, 4) if stop_loss is not None else None,
                "take_profit":        round(take_profit, 4) if take_profit is not None else None,
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
            "score_entrada":      None,
            "fecha_salida":       df.iloc[-1]["fecha"].date(),
            "precio_salida":      round(precio_salida, 4),
            "motivo_salida":      "FIN_SEGMENTO",
            "dias_posicion":      dias_posicion,
            "retorno_pct":        round(retorno_pct, 4),
            "resultado":          clasificar_resultado_pa(retorno_pct),
            "stop_loss":          round(stop_loss, 4) if stop_loss is not None else None,
            "take_profit":        round(take_profit, 4) if take_profit is not None else None,
        })

    return operaciones


# ─────────────────────────────────────────────────────────────
# Backtesting completo: todos los tickers x todas las combis
# ─────────────────────────────────────────────────────────────

def ejecutar_backtesting_pa_1w(tickers: list = None,
                                guardar_db: bool = True,
                                truncate: bool = True) -> pd.DataFrame:
    """
    Ejecuta la matriz 4x4 PA 1W para todos los tickers.

    Usa el periodo completo desde FECHA_INICIO_BT_1W hasta hoy,
    sin split TRAIN/TEST/BACKTEST (estrategias son reglas fijas).
    Todas las operaciones se etiquetan con segmento = "FULL".

    Args:
        tickers:    lista de tickers (None = ALL_TICKERS)
        guardar_db: persistir operaciones en PostgreSQL
        truncate:   si True (default), TRUNCATE antes de insertar

    Returns:
        DataFrame con todas las operaciones simuladas
    """
    tickers = tickers or ALL_TICKERS

    todas_ops    = []
    total_combis = len(tickers) * len(ESTRATEGIAS_ENTRADA_PA) * len(ESTRATEGIAS_SALIDA_PA)

    print(f"\n  Periodo: {FECHA_INICIO_BT_1W} -> hoy  |  Segmento: FULL  |  Timeframe: 1W")
    print(f"  {total_combis} combinaciones ({len(tickers)} tickers x 4 entradas x 4 salidas)\n")
    print("-" * 70)

    for ticker in tickers:
        print(f"  [{ticker}] Cargando datos...", end=" ")
        df_full = cargar_datos_ticker_pa_1w(ticker)

        if len(df_full) < 20:
            print("insuficientes semanas, omitido.")
            continue

        print(
            f"OK | {len(df_full)} semanas "
            f"({df_full['fecha'].min().date()} a {df_full['fecha'].max().date()})"
        )

        for ee in ESTRATEGIAS_ENTRADA_PA:
            for es in ESTRATEGIAS_SALIDA_PA:
                ops = simular_segmento_pa_1w(df_full, ticker, ee, es, "FULL")
                todas_ops.extend(ops)

    print("-" * 70)
    print(f"  Simulacion PA 1W completada: {len(todas_ops):,} operaciones totales.")

    df_ops = pd.DataFrame(todas_ops) if todas_ops else pd.DataFrame()

    if guardar_db and not df_ops.empty:
        _persistir_operaciones_pa_1w(
            df_ops,
            truncate=truncate,
            tickers_filtro=tickers if not truncate else None,
        )

    return df_ops


def _persistir_operaciones_pa_1w(df_ops: pd.DataFrame,
                                   truncate: bool = True,
                                   tickers_filtro: list = None):
    """
    Persiste operaciones en operaciones_bt_pa_1w.

    Si truncate=True (default): TRUNCATE ambas tablas antes de insertar.
    Si truncate=False: DELETE solo los registros de tickers_filtro.
    """
    import psycopg2.extras
    from src.data.database import ejecutar_sql

    if truncate:
        ejecutar_sql("TRUNCATE TABLE operaciones_bt_pa_1w RESTART IDENTITY")
        ejecutar_sql("TRUNCATE TABLE resultados_bt_pa_1w  RESTART IDENTITY")
    elif tickers_filtro:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM operaciones_bt_pa_1w WHERE ticker = ANY(%s)",
                    (tickers_filtro,)
                )

    records = df_ops.to_dict(orient="records")

    sql = """
        INSERT INTO operaciones_bt_pa_1w
            (estrategia_entrada, estrategia_salida, ticker, segmento,
             fecha_entrada, precio_entrada, score_entrada,
             fecha_salida, precio_salida, motivo_salida,
             dias_posicion, retorno_pct, resultado,
             stop_loss, take_profit)
        VALUES
            (%(estrategia_entrada)s, %(estrategia_salida)s, %(ticker)s, %(segmento)s,
             %(fecha_entrada)s, %(precio_entrada)s, %(score_entrada)s,
             %(fecha_salida)s, %(precio_salida)s, %(motivo_salida)s,
             %(dias_posicion)s, %(retorno_pct)s, %(resultado)s,
             %(stop_loss)s, %(take_profit)s)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=500)

    print(f"  Operaciones PA 1W persistidas: {len(records):,} registros.")
