"""
simulator.py
Motor de simulación de operaciones Long para el backtesting challenger.

Lógica general:
  - Lee datos combinados (precios + indicadores + scoring) desde PostgreSQL
  - Divide temporalmente en TRAIN / TEST / BACKTEST (70/15/15)
  - Para cada combinación entrada×salida simula operaciones sesión a sesión
  - Entrada: al OPEN del día siguiente a la señal
  - SL/TP:   evaluados contra HIGH/LOW intraday (sin look-ahead en la señal)
  - Salida por señal: al CLOSE del día de reversión
  - Salida por timeout: al CLOSE del día N
  - No hay posiciones simultáneas por ticker (una a la vez)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils.config import ALL_TICKERS, TRAIN_RATIO, TEST_RATIO
from src.backtesting.strategies import (
    check_entrada, calcular_stops_iniciales,
    check_salida, actualizar_trailing_stop,
    clasificar_resultado,
)
from src.data.database import query_df


# ─────────────────────────────────────────────────────────────
# Carga y preparación de datos
# ─────────────────────────────────────────────────────────────

def cargar_datos_ticker(ticker: str) -> pd.DataFrame:
    """
    Carga y une precios + indicadores + scoring para un ticker.
    Retorna DataFrame ordenado por fecha con todas las columnas necesarias.
    """
    sql = """
        SELECT
            p.fecha,
            p.open, p.high, p.low, p.close, p.volume,
            i.atr14, i.sma21, i.sma50, i.sma200,
            s.score_ponderado, s.condiciones_ok, s.senal,
            s.cond_rsi, s.cond_macd, s.cond_sma21,
            s.cond_sma50, s.cond_sma200, s.cond_momentum
        FROM precios_diarios p
        JOIN indicadores_tecnicos i ON p.ticker = i.ticker AND p.fecha = i.fecha
        JOIN scoring_tecnico      s ON p.ticker = s.ticker AND p.fecha = s.fecha
        WHERE p.ticker = :ticker
        ORDER BY p.fecha ASC
    """
    df = query_df(sql, params={"ticker": ticker})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.reset_index(drop=True)


def dividir_segmentos(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Divide el DataFrame en TRAIN / TEST / BACKTEST según proporciones configuradas.
    La división es TEMPORAL (no aleatoria) para evitar data leakage.

    Returns:
        dict con keys 'TRAIN', 'TEST', 'BACKTEST'
    """
    n = len(df)
    i_train = int(n * TRAIN_RATIO)
    i_test  = int(n * (TRAIN_RATIO + TEST_RATIO))

    return {
        "TRAIN":     df.iloc[:i_train].reset_index(drop=True),
        "TEST":      df.iloc[i_train:i_test].reset_index(drop=True),
        "BACKTEST":  df.iloc[i_test:].reset_index(drop=True),
    }


# ─────────────────────────────────────────────────────────────
# Simulador de un segmento para una combinación E×S
# ─────────────────────────────────────────────────────────────

def simular_segmento(df: pd.DataFrame, ticker: str,
                     est_entrada: str, est_salida: str,
                     segmento: str) -> List[dict]:
    """
    Simula operaciones Long para un ticker, estrategia y segmento.

    Reglas de simulación:
      - Solo una posición abierta a la vez por ticker
      - Señal en día T → entrada al OPEN del día T+1
      - SL/TP se evalúan contra el HIGH/LOW del día de posición
      - Si LOW <= SL y HIGH >= TP en el mismo día: SL gana (conservador)
      - Trailing stop (S3): se ajusta al cierre de cada día

    Returns:
        Lista de dicts con el detalle de cada operación cerrada
    """
    operaciones = []
    n = len(df)

    en_posicion    = False
    precio_entrada = None
    fecha_entrada  = None
    score_entrada  = None
    stop_loss      = None
    take_profit    = None
    high_water_mark = None
    dias_posicion  = 0

    for i in range(n):
        row = df.iloc[i]

        # ── FUERA DE POSICIÓN: evaluar entrada ────────────────
        if not en_posicion:
            if check_entrada(row, est_entrada):
                # Entrada al OPEN del día siguiente (evita look-ahead)
                if i + 1 >= n:
                    break   # no hay día siguiente, fin del segmento

                next_row = df.iloc[i + 1]

                # Si el open del día siguiente es inválido, salteamos
                if pd.isna(next_row["open"]) or next_row["open"] <= 0:
                    continue

                atr_hoy = row["atr14"] if not pd.isna(row["atr14"]) else row["close"] * 0.02

                precio_entrada  = float(next_row["open"])
                fecha_entrada   = next_row["fecha"]
                score_entrada   = float(row["score_ponderado"])
                stop_loss, take_profit = calcular_stops_iniciales(
                    precio_entrada, atr_hoy, est_salida
                )
                high_water_mark = precio_entrada
                dias_posicion   = 0
                en_posicion     = True

            continue   # ya procesamos este día como señal, avanzar

        # ── EN POSICIÓN: gestión diaria ───────────────────────
        dias_posicion += 1

        # Actualizar trailing stop si aplica (S3)
        atr_hoy = row["atr14"] if not pd.isna(row["atr14"]) else row["close"] * 0.02
        stop_loss, high_water_mark = actualizar_trailing_stop(
            stop_loss, high_water_mark, float(row["close"]),
            float(atr_hoy), est_salida
        )

        # Evaluar condición de salida
        cerrar, motivo, precio_salida = check_salida(
            row, est_salida, stop_loss, take_profit, dias_posicion
        )

        if cerrar:
            retorno_pct = (precio_salida / precio_entrada - 1) * 100
            resultado   = clasificar_resultado(retorno_pct)

            operaciones.append({
                "estrategia_entrada": est_entrada,
                "estrategia_salida":  est_salida,
                "ticker":             ticker,
                "segmento":           segmento,
                "fecha_entrada":      fecha_entrada.date() if hasattr(fecha_entrada, "date") else fecha_entrada,
                "precio_entrada":     round(precio_entrada, 4),
                "score_entrada":      round(score_entrada, 4),
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

    # ── Posición abierta al final del segmento → cierre forzado
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
            "score_entrada":      round(score_entrada, 4),
            "fecha_salida":       df.iloc[-1]["fecha"].date(),
            "precio_salida":      round(precio_salida, 4),
            "motivo_salida":      "FIN_SEGMENTO",
            "dias_posicion":      dias_posicion,
            "retorno_pct":        round(retorno_pct, 4),
            "resultado":          clasificar_resultado(retorno_pct),
        })

    return operaciones


# ─────────────────────────────────────────────────────────────
# Backtesting completo: todos los tickers × todas las combis
# ─────────────────────────────────────────────────────────────

ESTRATEGIAS_ENTRADA = ["E1", "E2", "E3", "E4"]
ESTRATEGIAS_SALIDA  = ["S1", "S2", "S3", "S4"]
SEGMENTOS           = ["TRAIN", "TEST", "BACKTEST"]


def ejecutar_backtesting(tickers: list = None,
                         segmentos: list = None,
                         guardar_db: bool = True) -> pd.DataFrame:
    """
    Ejecuta la matriz completa 4×4 para todos los tickers y segmentos.

    Args:
        tickers:    lista de tickers (None = ALL_TICKERS)
        segmentos:  lista de segmentos (None = todos)
        guardar_db: persistir operaciones en PostgreSQL

    Returns:
        DataFrame con todas las operaciones simuladas
    """
    from src.data.database import ejecutar_sql
    import psycopg2.extras

    tickers   = tickers   or ALL_TICKERS
    segmentos = segmentos or SEGMENTOS

    todas_ops = []
    total_combis = len(tickers) * len(ESTRATEGIAS_ENTRADA) * len(ESTRATEGIAS_SALIDA)

    print(f"\nCargando datos y ejecutando {total_combis} combinaciones...\n")
    print("-" * 65)

    for ticker in tickers:
        print(f"  [{ticker}] Cargando datos...", end=" ")
        df_completo = cargar_datos_ticker(ticker)

        if len(df_completo) < 50:
            print("insuficientes datos, omitido.")
            continue

        segms = dividir_segmentos(df_completo)
        print(
            f"OK | TRAIN:{len(segms['TRAIN'])} "
            f"TEST:{len(segms['TEST'])} "
            f"BACKTEST:{len(segms['BACKTEST'])}"
        )

        for seg_nombre in segmentos:
            df_seg = segms[seg_nombre]
            if df_seg.empty:
                continue

            for ee in ESTRATEGIAS_ENTRADA:
                for es in ESTRATEGIAS_SALIDA:
                    ops = simular_segmento(df_seg, ticker, ee, es, seg_nombre)
                    todas_ops.extend(ops)

    print("-" * 65)
    print(f"Simulacion completada: {len(todas_ops):,} operaciones totales.")

    df_ops = pd.DataFrame(todas_ops) if todas_ops else pd.DataFrame()

    if guardar_db and not df_ops.empty:
        _persistir_operaciones(df_ops)

    return df_ops


def _persistir_operaciones(df_ops: pd.DataFrame):
    """Persiste las operaciones en la tabla operaciones_backtest."""
    from src.data.database import get_connection, ejecutar_sql
    import psycopg2.extras

    # Limpiar operaciones previas para re-run limpio
    ejecutar_sql("TRUNCATE TABLE operaciones_backtest RESTART IDENTITY")
    ejecutar_sql("TRUNCATE TABLE resultados_backtest  RESTART IDENTITY")

    records = df_ops.to_dict(orient="records")

    sql = """
        INSERT INTO operaciones_backtest
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

    print(f"  Operaciones persistidas: {len(records):,} registros.")
